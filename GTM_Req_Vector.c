#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#include "GTMatrix_Retval.h"
#include "GTM_Req_Vector.h"

int GTM_createReqVector(GTM_Req_Vector_t *gtm_rv_)
{
    GTM_Req_Vector_t gtm_rv = (GTM_Req_Vector_t) malloc(sizeof(struct GTM_Req_Vector));
    if (gtm_rv == NULL) return GTM_RV_ALLOC_FAILED;
    
    gtm_rv->curr_size   = 0;
    gtm_rv->max_size    = DEFAULT_REQ_VEC_LEN;
    gtm_rv->row_starts  = (int*)    malloc(gtm_rv->max_size * sizeof(int));
    gtm_rv->row_nums    = (int*)    malloc(gtm_rv->max_size * sizeof(int));
    gtm_rv->col_starts  = (int*)    malloc(gtm_rv->max_size * sizeof(int));
    gtm_rv->col_nums    = (int*)    malloc(gtm_rv->max_size * sizeof(int));
    gtm_rv->src_bufs    = (void**)  malloc(gtm_rv->max_size * sizeof(void*));
    gtm_rv->src_buf_lds = (int*)    malloc(gtm_rv->max_size * sizeof(int));
    gtm_rv->ops         = (MPI_Op*) malloc(gtm_rv->max_size * sizeof(MPI_Op));
    if ((gtm_rv->row_starts == NULL) || (gtm_rv->row_nums == NULL) || 
        (gtm_rv->col_starts == NULL) || (gtm_rv->col_nums == NULL) || 
        (gtm_rv->src_bufs   == NULL) || (gtm_rv->src_buf_lds == NULL) || (gtm_rv->ops == NULL))
    {
        return GTM_RV_ALLOC_FAILED;
    }
    
    *gtm_rv_ = gtm_rv;
    return GTM_RV_SUCCESS;
}

int GTM_pushToReqVector(
    GTM_Req_Vector_t gtm_rv, MPI_Op op, 
    int row_start, int row_num,
    int col_start, int col_num,
    void *src_buf, int src_buf_ld
)
{
    if (gtm_rv == NULL) return GTM_RV_NULL_PTR;
    
    // Check if we need to reallocate 
    if (gtm_rv->curr_size == gtm_rv->max_size)  
    {
        int *row_starts  = (int*)    malloc(gtm_rv->max_size * 2 * sizeof(int));
        int *row_nums    = (int*)    malloc(gtm_rv->max_size * 2 * sizeof(int));
        int *col_starts  = (int*)    malloc(gtm_rv->max_size * 2 * sizeof(int));
        int *col_nums    = (int*)    malloc(gtm_rv->max_size * 2 * sizeof(int));
        void **src_bufs  = (void**)  malloc(gtm_rv->max_size * 2 * sizeof(void*));
        int *src_buf_lds = (int*)    malloc(gtm_rv->max_size * 2 * sizeof(int));
        MPI_Op *ops      = (MPI_Op*) malloc(gtm_rv->max_size * 2 * sizeof(MPI_Op));
        if ((row_starts == NULL) || (row_nums == NULL) || (col_starts == NULL) ||
            (col_nums == NULL) || (src_bufs == NULL) || (src_buf_lds == NULL) || (ops == NULL))
        {
            return GTM_RV_RESIZE_FAILED;
        }
        
        memcpy(row_starts,  gtm_rv->row_starts,  gtm_rv->max_size * sizeof(int));
        memcpy(row_nums,    gtm_rv->row_nums,    gtm_rv->max_size * sizeof(int));
        memcpy(col_starts,  gtm_rv->col_starts,  gtm_rv->max_size * sizeof(int));
        memcpy(col_nums,    gtm_rv->col_nums,    gtm_rv->max_size * sizeof(int));
        memcpy(src_bufs,    gtm_rv->src_bufs,    gtm_rv->max_size * sizeof(void*));
        memcpy(src_buf_lds, gtm_rv->src_buf_lds, gtm_rv->max_size * sizeof(int));
        memcpy(ops,         gtm_rv->ops,         gtm_rv->max_size * sizeof(MPI_Op));
        
        gtm_rv->max_size *= 2;
        
        free(gtm_rv->row_starts);
        free(gtm_rv->row_nums);
        free(gtm_rv->col_starts);
        free(gtm_rv->col_nums);
        free(gtm_rv->src_bufs);
        free(gtm_rv->src_buf_lds);
        free(gtm_rv->ops);
        
        gtm_rv->row_starts  = row_starts;
        gtm_rv->row_nums    = row_nums;
        gtm_rv->col_starts  = col_starts;
        gtm_rv->col_nums    = col_nums;
        gtm_rv->src_bufs    = src_bufs;
        gtm_rv->src_buf_lds = src_buf_lds;
        gtm_rv->ops         = ops;
    }
    
    int idx = gtm_rv->curr_size;
    gtm_rv->row_starts[idx]  = row_start;
    gtm_rv->row_nums[idx]    = row_num;
    gtm_rv->col_starts[idx]  = col_start;
    gtm_rv->col_nums[idx]    = col_num;
    gtm_rv->src_bufs[idx]    = src_buf;
    gtm_rv->src_buf_lds[idx] = src_buf_ld;
    gtm_rv->ops[idx]         = op;
    gtm_rv->curr_size++;
    return GTM_RV_SUCCESS;
}

int GTM_resetReqVector(GTM_Req_Vector_t gtm_rv)
{
    if (gtm_rv == NULL) return GTM_RV_NULL_PTR;
    gtm_rv->curr_size = 0;
    return GTM_RV_SUCCESS;
}

int GTM_destroyReqVector(GTM_Req_Vector_t gtm_rv)
{
    if (gtm_rv == NULL) return GTM_RV_NULL_PTR;
    free(gtm_rv->row_starts);
    free(gtm_rv->row_nums);
    free(gtm_rv->col_starts);
    free(gtm_rv->col_nums);
    free(gtm_rv->src_bufs);
    free(gtm_rv->src_buf_lds);
    free(gtm_rv->ops);
    free(gtm_rv);
    return GTM_RV_SUCCESS;
}