#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#include "Buzz_Req_Vector.h"

void Buzz_createReqVector(Buzz_Req_Vector_t *brv_)
{
    Buzz_Req_Vector_t brv = (Buzz_Req_Vector_t) malloc(sizeof(struct Buzz_Req_Vector));
    assert(brv != NULL);
    
    brv->curr_size   = 0;
    brv->max_size    = DEFAULT_REQ_VEC_LEN;
    brv->row_starts  = (int*)    malloc(brv->max_size * sizeof(int));
    brv->row_nums    = (int*)    malloc(brv->max_size * sizeof(int));
    brv->col_starts  = (int*)    malloc(brv->max_size * sizeof(int));
    brv->col_nums    = (int*)    malloc(brv->max_size * sizeof(int));
    brv->src_bufs    = (void**)  malloc(brv->max_size * sizeof(void*));
    brv->src_buf_lds = (int*)    malloc(brv->max_size * sizeof(int));
    brv->ops         = (MPI_Op*) malloc(brv->max_size * sizeof(MPI_Op));
    assert(brv->row_starts  != NULL);
    assert(brv->row_nums    != NULL);
    assert(brv->col_starts  != NULL);
    assert(brv->col_nums    != NULL);
    assert(brv->src_bufs    != NULL);
    assert(brv->src_buf_lds != NULL);
    assert(brv->ops         != NULL);
    
    *brv_ = brv;
}

void Buzz_pushToReqVector(
    Buzz_Req_Vector_t brv, MPI_Op op, 
    int row_start, int row_num,
    int col_start, int col_num,
    void *src_buf, int src_buf_ld
)
{
    // Check if we need to reallocate 
    if (brv->curr_size == brv->max_size)  
    {
        int *row_starts  = (int*)    malloc(brv->max_size * 2 * sizeof(int));
        int *row_nums    = (int*)    malloc(brv->max_size * 2 * sizeof(int));
        int *col_starts  = (int*)    malloc(brv->max_size * 2 * sizeof(int));
        int *col_nums    = (int*)    malloc(brv->max_size * 2 * sizeof(int));
        void **src_bufs  = (void**)  malloc(brv->max_size * 2 * sizeof(void*));
        int *src_buf_lds = (int*)    malloc(brv->max_size * 2 * sizeof(int));
        MPI_Op *ops      = (MPI_Op*) malloc(brv->max_size * 2 * sizeof(MPI_Op));
        assert(row_starts  != NULL);
        assert(row_nums    != NULL);
        assert(col_starts  != NULL);
        assert(col_nums    != NULL);
        assert(src_bufs    != NULL);
        assert(src_buf_lds != NULL);
        assert(ops         != NULL);
        
        memcpy(row_starts,  brv->row_starts,  brv->max_size * sizeof(int));
        memcpy(row_nums,    brv->row_nums,    brv->max_size * sizeof(int));
        memcpy(col_starts,  brv->col_starts,  brv->max_size * sizeof(int));
        memcpy(col_nums,    brv->col_nums,    brv->max_size * sizeof(int));
        memcpy(src_bufs,    brv->src_bufs,    brv->max_size * sizeof(void*));
        memcpy(src_buf_lds, brv->src_buf_lds, brv->max_size * sizeof(int));
        memcpy(ops,         brv->ops,         brv->max_size * sizeof(MPI_Op));
        
        brv->max_size *= 2;
        
        free(brv->row_starts);
        free(brv->row_nums);
        free(brv->col_starts);
        free(brv->col_nums);
        free(brv->src_bufs);
        free(brv->src_buf_lds);
        free(brv->ops);
        
        brv->row_starts  = row_starts;
        brv->row_nums    = row_nums;
        brv->col_starts  = col_starts;
        brv->col_nums    = col_nums;
        brv->src_bufs    = src_bufs;
        brv->src_buf_lds = src_buf_lds;
        brv->ops         = ops;
    }
    
    int idx = brv->curr_size;
    brv->row_starts[idx]  = row_start;
    brv->row_nums[idx]    = row_num;
    brv->col_starts[idx]  = col_start;
    brv->col_nums[idx]    = col_num;
    brv->src_bufs[idx]    = src_buf;
    brv->src_buf_lds[idx] = src_buf_ld;
    brv->ops[idx]         = op;
    brv->curr_size++;
}

void Buzz_resetReqVector(Buzz_Req_Vector_t brv)
{
    brv->curr_size = 0;
}

void Buzz_destroyReqVector(Buzz_Req_Vector_t brv)
{
    free(brv->row_starts);
    free(brv->row_nums);
    free(brv->col_starts);
    free(brv->col_nums);
    free(brv->src_bufs);
    free(brv->src_buf_lds);
    free(brv->ops);
    
    free(brv);
}