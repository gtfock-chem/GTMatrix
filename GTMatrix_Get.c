#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#include "GTMatrix_Retval.h"
#include "GTMatrix_Typedef.h"
#include "GTMatrix_Get.h"
#include "GTMatrix_Other.h"
#include "utils.h"

// Post the operation of getting a blocking from a process using MPI_Get
// The get operation is not complete when this function returns
// Input parameters:
//   gtm        : GTMatrix handle
//   dst_rank   : Target process
//   row_start  : 1st row of the required block
//   row_num    : Number of rows the required block has
//   col_start  : 1st column of the required block
//   col_num    : Number of columns the required block has
//   src_buf_ld : Leading dimension of the received buffer
// Output parameter:
//   *src_buf : Receive buffer
int GTM_getBlockFromProcess(
    GTMatrix_t gtm, int dst_rank, 
    int row_start, int row_num,
    int col_start, int col_num,
    void *src_buf, int src_buf_ld
)
{
    if (gtm == NULL) return GTM_NULL_PTR;
    
    int row_end       = row_start + row_num;
    int col_end       = col_start + col_num;
    int dst_rowblk    = dst_rank / gtm->c_blocks;
    int dst_colblk    = dst_rank % gtm->c_blocks;
    int dst_blk_ld    = gtm->ld_local; // gtm->ld_blks[dst_rank];
    int dst_row_start = gtm->r_displs[dst_rowblk];
    int dst_col_start = gtm->c_displs[dst_colblk];
    int dst_row_end   = gtm->r_displs[dst_rowblk + 1];
    int dst_col_end   = gtm->c_displs[dst_colblk + 1];
    
    // Sanity check
    if ((row_start < dst_row_start) ||
        (col_start < dst_col_start) ||
        (row_end   > dst_row_end)   ||
        (col_end   > dst_col_end)   ||
        (row_num   * col_num == 0)) return GTM_INVALID_BLOCK;

    // Check if the target process is in the shared memory communicator
    int shm_rank  = getElementIndexInArray(dst_rank, gtm->shm_global_ranks, gtm->shm_size);
    void *shm_ptr = (shm_rank == -1) ? NULL : gtm->shm_mat_blocks[shm_rank];
    
    char *src_ptr = (char*) src_buf;
    int row_bytes = col_num * gtm->unit_size;
    int dst_pos = (row_start - dst_row_start) * dst_blk_ld;
    dst_pos += col_start - dst_col_start;

    if (shm_rank != -1)
    {
        // Target process and current process is in same node, use memcpy
        char *dst_ptr  = shm_ptr + dst_pos * gtm->unit_size;
        int src_ptr_ld = src_buf_ld * gtm->unit_size;
        int dst_ptr_ld = dst_blk_ld * gtm->unit_size;
        for (int irow = 0; irow < row_num; irow++)
        {
            memcpy(src_ptr, dst_ptr, row_bytes);
            src_ptr += src_ptr_ld;
            dst_ptr += dst_ptr_ld;
        }
    } else {
        // Target process and current process isn't in same node, use MPI_Get
        int src_ptr_ld = src_buf_ld * gtm->unit_size;
        if (row_num <= MPI_DT_SB_DIM_MAX && col_num <= MPI_DT_SB_DIM_MAX)  
        {
            // Block is small, use predefined data type or define a new 
            // data type to reduce MPI_Get overhead
            int block_dt_id = (row_num - 1) * MPI_DT_SB_DIM_MAX + (col_num - 1);
            MPI_Datatype *dst_dt = &gtm->sb_stride[block_dt_id];
            if (col_num == src_buf_ld)
            {
                MPI_Datatype *rcv_dt_ns = &gtm->sb_nostride[block_dt_id];
                MPI_Get(src_ptr, 1, *rcv_dt_ns, dst_rank, dst_pos, 1, *dst_dt, gtm->mpi_win);
            } else {
                if (gtm->ld_local == src_buf_ld)
                {
                    MPI_Get(src_ptr, 1, *dst_dt, dst_rank, dst_pos, 1, *dst_dt, gtm->mpi_win);
                } else {
                    MPI_Datatype rcv_dt;
                    MPI_Type_vector(row_num, col_num, src_buf_ld, gtm->datatype, &rcv_dt);
                    MPI_Type_commit(&rcv_dt);
                    MPI_Get(src_ptr, 1, rcv_dt, dst_rank, dst_pos, 1, *dst_dt, gtm->mpi_win);
                    MPI_Type_free(&rcv_dt);
                }
            }
        } else {
            // Define a MPI data type to reduce number of request
            MPI_Datatype dst_dt, rcv_dt;
            MPI_Type_vector(row_num, col_num, dst_blk_ld, gtm->datatype, &dst_dt);
            MPI_Type_vector(row_num, col_num, src_buf_ld, gtm->datatype, &rcv_dt);
            MPI_Type_commit(&dst_dt);
            MPI_Type_commit(&rcv_dt);
            MPI_Get(src_ptr, 1, rcv_dt, dst_rank, dst_pos, 1, dst_dt, gtm->mpi_win);
            MPI_Type_free(&dst_dt);
            MPI_Type_free(&rcv_dt);
        }
    }
    return GTM_SUCCESS;
}

// Get a block from all related processes using MPI_Get
// Non-blocking, data may not be ready before synchronization
// This call is not collective, thread-safe
// Input paramaters:
//   gtm      : GTMatrix handle
//   row_start   : 1st row of the required block
//   row_num     : Number of rows the required block has
//   col_start   : 1st column of the required block
//   col_num     : Number of columns the required block has
//   src_buf_ld  : Leading dimension of the received buffer
//   access_mode : Access mode, see GTMatrix_Typedef.h
// Output parameter:
//   *src_buf : Receive buffer
int GTM_getBlock_(GTM_PARAM, int access_mode)
{
    if (gtm == NULL) return GTM_NULL_PTR;
    
    // Sanity check
    if ((row_start < 0) || (col_start < 0) ||
        (row_start + row_num > gtm->nrows)  ||
        (col_start + col_num > gtm->ncols)  ||
        (row_num * col_num == 0)) return GTM_INVALID_BLOCK;
    
    // Find the processes that contain the requested block
    int s_blk_r, e_blk_r, s_blk_c, e_blk_c;
    int row_end = row_start + row_num - 1;
    int col_end = col_start + col_num - 1;
    for (int i = 0; i < gtm->r_blocks; i++)
    {
        if ((gtm->r_displs[i] <= row_start) && 
            (row_start < gtm->r_displs[i+1])) s_blk_r = i;
        if ((gtm->r_displs[i] <= row_end)   && 
            (row_end   < gtm->r_displs[i+1])) e_blk_r = i;
    }
    for (int i = 0; i < gtm->c_blocks; i++)
    {
        if ((gtm->c_displs[i] <= col_start) && 
            (col_start < gtm->c_displs[i+1])) s_blk_c = i;
        if ((gtm->c_displs[i] <= col_end)   && 
            (col_end   < gtm->c_displs[i+1])) e_blk_c = i;
    }
    
    // Fetch data from each process
    int blk_r_s, blk_r_e, blk_c_s, blk_c_e, need_to_fetch;
    for (int blk_r = s_blk_r; blk_r <= e_blk_r; blk_r++)      // Notice: <=
    {
        int dst_r_s = gtm->r_displs[blk_r];
        int dst_r_e = gtm->r_displs[blk_r + 1] - 1;
        for (int blk_c = s_blk_c; blk_c <= e_blk_c; blk_c++)  // Notice: <=
        {
            int dst_c_s  = gtm->c_displs[blk_c];
            int dst_c_e  = gtm->c_displs[blk_c + 1] - 1;
            int dst_rank = blk_r * gtm->c_blocks + blk_c;
            getRectIntersection(
                dst_r_s,   dst_r_e, dst_c_s,   dst_c_e,
                row_start, row_end, col_start, col_end,
                &need_to_fetch, &blk_r_s, &blk_r_e, &blk_c_s, &blk_c_e
            );
            assert(need_to_fetch == 1);
            int blk_r_num = blk_r_e - blk_r_s + 1;
            int blk_c_num = blk_c_e - blk_c_s + 1;
            int row_dist  = blk_r_s - row_start;
            int col_dist  = blk_c_s - col_start;
            char *blk_ptr = (char*) src_buf;
            blk_ptr += (row_dist * src_buf_ld + col_dist) * gtm->unit_size;
            
            int ret = GTM_SUCCESS;
            
            if (access_mode == BLOCKING_ACCESS)
            {
                MPI_Win_lock(MPI_LOCK_SHARED, dst_rank, 0, gtm->mpi_win);
                ret = GTM_getBlockFromProcess(
                    gtm, dst_rank, blk_r_s, blk_r_num, 
                    blk_c_s, blk_c_num, blk_ptr, src_buf_ld
                );
                MPI_Win_unlock(dst_rank, gtm->mpi_win);
            }
            
            if (access_mode == NONBLOCKING_ACCESS)
            {
                if (gtm->nb_op_proc_cnt[dst_rank] == 0)
                    MPI_Win_lock(MPI_LOCK_SHARED, dst_rank, 0, gtm->mpi_win);
                
                ret = GTM_getBlockFromProcess(
                    gtm, dst_rank, blk_r_s, blk_r_num, 
                    blk_c_s, blk_c_num, blk_ptr, src_buf_ld
                );
                
                gtm->nb_op_proc_cnt[dst_rank]++;
                gtm->nb_op_cnt++;
                if (gtm->nb_op_cnt >= gtm->max_nb_get)
                    GTM_waitNB(gtm);
            }
            
            if (access_mode == BATCH_ACCESS)
            {
                GTM_Req_Vector_t req_vec = gtm->req_vec[dst_rank];
                ret = GTM_pushToReqVector(
                    req_vec, MPI_NO_OP, blk_r_s, blk_r_num, 
                    blk_c_s, blk_c_num, blk_ptr, src_buf_ld
                );
            }
            
            if (ret != GTM_SUCCESS) return ret;
        }
    }
    return GTM_SUCCESS;
}

// Get a block from the global matrix
int GTM_getBlock(GTM_PARAM)
{
    if (gtm == NULL) return GTM_NULL_PTR;
    return GTM_getBlock_(
        gtm,
        row_start, row_num,
        col_start, col_num,
        src_buf, src_buf_ld, BLOCKING_ACCESS
    );
}

// Nonblocking get a block from the global matrix
int GTM_getBlockNB(GTM_PARAM)
{
    if (gtm == NULL) return GTM_NULL_PTR;
    return GTM_getBlock_(
        gtm, row_start, row_num, col_start, col_num,
        src_buf, src_buf_ld, NONBLOCKING_ACCESS
    );
}

// Add a request to get a block from the global matrix
int GTM_addGetBlockRequest(GTM_PARAM)
{
    if (gtm == NULL) return GTM_NULL_PTR;
    return GTM_getBlock_(
        gtm, row_start, row_num, col_start, col_num,
        src_buf, src_buf_ld, BATCH_ACCESS
    );
}

// Start a batch get epoch and allow to submit get requests
int GTM_startBatchGet(GTMatrix_t gtm)
{
    if (gtm == NULL) return GTM_NULL_PTR;
    if (gtm->in_batch_put) return GTM_IN_BATCHED_PUT;
    if (gtm->in_batch_get) return GTM_IN_BATCHED_GET;
    if (gtm->in_batch_acc) return GTM_IN_BATCHED_ACC;
    
    for (int i = 0; i < gtm->comm_size; i++)
        GTM_resetReqVector(gtm->req_vec[i]);
    gtm->in_batch_get = 1;
    return GTM_SUCCESS;
}

// Execute all get requests in the queues
int GTM_execBatchGet(GTMatrix_t gtm)
{
    if (gtm == NULL) return GTM_NULL_PTR;
    if (gtm->in_batch_get == 0) return GTM_NO_BATCHED_GET;
    
    for (int _dst_rank = gtm->my_rank; _dst_rank < gtm->comm_size + gtm->my_rank; _dst_rank++)
    {    
        int dst_rank = _dst_rank % gtm->comm_size;
        GTM_Req_Vector_t req_vec = gtm->req_vec[dst_rank];
        
        if (req_vec->curr_size > 0)
        {
            MPI_Win_lock(MPI_LOCK_SHARED, dst_rank, 0, gtm->mpi_win);
            for (int i = 0; i < req_vec->curr_size; i++)
            {
                int blk_r_s    = req_vec->row_starts[i];
                int blk_r_num  = req_vec->row_nums[i];
                int blk_c_s    = req_vec->col_starts[i];
                int blk_c_num  = req_vec->col_nums[i];
                void *blk_ptr  = req_vec->src_bufs[i];
                int src_buf_ld = req_vec->src_buf_lds[i];
                int ret = GTM_getBlockFromProcess(
                    gtm, dst_rank, blk_r_s, blk_r_num, 
                    blk_c_s, blk_c_num, blk_ptr, src_buf_ld
                );
                if (ret != GTM_SUCCESS) return ret;
            }
            MPI_Win_unlock(dst_rank, gtm->mpi_win);
        }
        
        GTM_resetReqVector(req_vec);
    }
    return GTM_SUCCESS;
}

// Stop a batch get epoch and disallow to submit get requests
int GTM_stopBatchGet(GTMatrix_t gtm)
{
    if (gtm == NULL) return GTM_NULL_PTR;
    if (gtm->in_batch_get == 0) return GTM_NO_BATCHED_GET;
    gtm->in_batch_get = 0;
    return GTM_SUCCESS;
}
