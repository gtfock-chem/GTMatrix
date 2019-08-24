#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#include "GTMatrix_Retval.h"
#include "GTMatrix_Typedef.h"
#include "GTMatrix_Get.h"
#include "GTMatrix_Other.h"

int GTM_sync(GTMatrix_t gt_mat)
{
    if (gt_mat == NULL) return GTM_NULL_PTR;
    if (SYNC_IBARRIER == 1)
    {
        // Observed that on some Skylake & KNL machine with IMPI 17, when not all
        // MPI processes have RMA calls, a MPI_Barrier will lead to deadlock when 
        // running more than 16 MPI processes on the same node. Don't know why using  
        // a MPI_Ibarrier + MPI_Ibarrier can solve this problem...
        MPI_Status  status;
        MPI_Request req;
        MPI_Ibarrier(gt_mat->mpi_comm, &req);
        MPI_Wait(&req, &status);
    } else {
        MPI_Barrier(gt_mat->mpi_comm);
    }
    return GTM_SUCCESS;
}

int GTM_waitNB(GTMatrix_t gt_mat)
{
    if (gt_mat == NULL) return GTM_NULL_PTR;
    for (int dst_rank = 0; dst_rank < gt_mat->comm_size; dst_rank++)
    {
        if (gt_mat->nb_op_proc_cnt[dst_rank] != 0)
        {
            MPI_Win_unlock(dst_rank, gt_mat->mpi_win);
            gt_mat->nb_op_proc_cnt[dst_rank] = 0;
        }
    }
    gt_mat->nb_op_cnt = 0;
    return GTM_SUCCESS;
}

int GTM_fill(GTMatrix_t gt_mat, void *value)
{
    if (gt_mat == NULL) return GTM_NULL_PTR;
    if (gt_mat->unit_size == 4)
    {
        int _value, *ptr;
        memcpy(&_value, value, 4);
        ptr = (int*) gt_mat->mat_block;
        for (int i = 0; i < gt_mat->my_nrows; i++)
            #pragma vector
            for (int j = 0; j < gt_mat->my_ncols; j++) 
                ptr[i * gt_mat->ld_local + j] = _value;
    }
    if (gt_mat->unit_size == 8)
    {
        double _value, *ptr;
        memcpy(&_value, value, 8);
        ptr = (double*) gt_mat->mat_block;
        for (int i = 0; i < gt_mat->my_nrows; i++)
            #pragma vector
            for (int j = 0; j < gt_mat->my_ncols; j++) 
                ptr[i * gt_mat->ld_local + j] = _value;
    }
    return GTM_SUCCESS;
}

int GTM_symmetrize(GTMatrix_t gt_mat)
{
    if (gt_mat == NULL) return GTM_NULL_PTR;
    if (gt_mat->nrows != gt_mat->ncols) return GTM_NOT_SQUARE_MAT;
    
    // This process holds [rs:re, cs:ce], need to fetch [cs:ce, rs:re]
    void *rcv_buf = gt_mat->symm_buf;

    int my_row_start = gt_mat->r_displs[gt_mat->my_rowblk];
    int my_col_start = gt_mat->c_displs[gt_mat->my_colblk];
    
    GTM_sync(gt_mat);
    
    GTM_getBlock(
        gt_mat, 
        my_col_start, gt_mat->my_ncols, 
        my_row_start, gt_mat->my_nrows, 
        rcv_buf, gt_mat->my_nrows
    );
    
    // Wait all processes to get the symmetric block before modifying
    // local block, or some processes will get the modified block
    GTM_sync(gt_mat);
    
    if (MPI_INT == gt_mat->datatype)
    {
        int *src_buf = (int*) gt_mat->mat_block;
        int *dst_buf = (int*) rcv_buf;
        for (int irow = 0; irow < gt_mat->my_nrows; irow++)
        {
            for (int icol = 0; icol < gt_mat->my_ncols; icol++)
            {
                int idx_s = irow * gt_mat->ld_local + icol;
                int idx_d = icol * gt_mat->my_nrows + irow;
                src_buf[idx_s] += dst_buf[idx_d];
                src_buf[idx_s] /= 2;
            }
        }
    }
    if (MPI_DOUBLE == gt_mat->datatype)
    {
        double *src_buf = (double*) gt_mat->mat_block;
        double *dst_buf = (double*) rcv_buf;
        for (int irow = 0; irow < gt_mat->my_nrows; irow++)
        {
            for (int icol = 0; icol < gt_mat->my_ncols; icol++)
            {
                int idx_s = irow * gt_mat->ld_local + icol;
                int idx_d = icol * gt_mat->my_nrows + irow;
                src_buf[idx_s] += dst_buf[idx_d];
                src_buf[idx_s] *= 0.5;
            }
        }
    }
    
    return GTM_sync(gt_mat);
}