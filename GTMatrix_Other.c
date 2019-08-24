#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#include "GTMatrix_Retval.h"
#include "GTMatrix_Typedef.h"
#include "GTMatrix_Get.h"
#include "GTMatrix_Other.h"

int GTM_sync(GTMatrix_t gtm)
{
    if (gtm == NULL) return GTM_NULL_PTR;
    if (SYNC_IBARRIER == 1)
    {
        // Observed that on some Skylake & KNL machine with IMPI 17, when not all
        // MPI processes have RMA calls, a MPI_Barrier will lead to deadlock when 
        // running more than 16 MPI processes on the same node. Don't know why using  
        // a MPI_Ibarrier + MPI_Ibarrier can solve this problem...
        MPI_Status  status;
        MPI_Request req;
        MPI_Ibarrier(gtm->mpi_comm, &req);
        MPI_Wait(&req, &status);
    } else {
        MPI_Barrier(gtm->mpi_comm);
    }
    return GTM_SUCCESS;
}

int GTM_waitNB(GTMatrix_t gtm)
{
    if (gtm == NULL) return GTM_NULL_PTR;
    for (int dst_rank = 0; dst_rank < gtm->comm_size; dst_rank++)
    {
        if (gtm->nb_op_proc_cnt[dst_rank] != 0)
        {
            MPI_Win_unlock(dst_rank, gtm->mpi_win);
            gtm->nb_op_proc_cnt[dst_rank] = 0;
        }
    }
    gtm->nb_op_cnt = 0;
    return GTM_SUCCESS;
}

int GTM_fill(GTMatrix_t gtm, void *value)
{
    if (gtm == NULL) return GTM_NULL_PTR;
    if (gtm->unit_size == 4)
    {
        int _value, *ptr;
        memcpy(&_value, value, 4);
        ptr = (int*) gtm->mat_block;
        for (int i = 0; i < gtm->my_nrows; i++)
        {
            int offset_i = i * gtm->ld_local;
            for (int j = 0; j < gtm->my_ncols; j++) 
                ptr[offset_i + j] = _value;
        }
    }
    if (gtm->unit_size == 8)
    {
        double _value, *ptr;
        memcpy(&_value, value, 8);
        ptr = (double*) gtm->mat_block;
        for (int i = 0; i < gtm->my_nrows; i++)
        {
            int offset_i = i * gtm->ld_local;
            for (int j = 0; j < gtm->my_ncols; j++) 
                ptr[offset_i + j] = _value;
        }
    }
    return GTM_SUCCESS;
}

int GTM_symmetrize(GTMatrix_t gtm)
{
    if (gtm == NULL) return GTM_NULL_PTR;
    if (gtm->nrows != gtm->ncols) return GTM_NOT_SQUARE_MAT;
    
    // This process holds [rs:re, cs:ce], need to fetch [cs:ce, rs:re]
    void *rcv_buf = gtm->symm_buf;

    int my_row_start = gtm->r_displs[gtm->my_rowblk];
    int my_col_start = gtm->c_displs[gtm->my_colblk];
    
    GTM_sync(gtm);
    
    GTM_getBlock(
        gtm, my_col_start, gtm->my_ncols, 
        my_row_start, gtm->my_nrows, 
        rcv_buf, gtm->my_nrows
    );
    
    // Wait all processes to get the symmetric block before modifying
    // local block, or some processes will get the modified block
    GTM_sync(gtm);
    
    if (MPI_INT == gtm->datatype)
    {
        int *src_buf = (int*) gtm->mat_block;
        int *dst_buf = (int*) rcv_buf;
        for (int irow = 0; irow < gtm->my_nrows; irow++)
        {
            for (int icol = 0; icol < gtm->my_ncols; icol++)
            {
                int idx_s = irow * gtm->ld_local + icol;
                int idx_d = icol * gtm->my_nrows + irow;
                src_buf[idx_s] += dst_buf[idx_d];
                src_buf[idx_s] /= 2;
            }
        }
    }
    if (MPI_DOUBLE == gtm->datatype)
    {
        double *src_buf = (double*) gtm->mat_block;
        double *dst_buf = (double*) rcv_buf;
        for (int irow = 0; irow < gtm->my_nrows; irow++)
        {
            for (int icol = 0; icol < gtm->my_ncols; icol++)
            {
                int idx_s = irow * gtm->ld_local + icol;
                int idx_d = icol * gtm->my_nrows + irow;
                src_buf[idx_s] += dst_buf[idx_d];
                src_buf[idx_s] *= 0.5;
            }
        }
    }
    
    return GTM_sync(gtm);
}
