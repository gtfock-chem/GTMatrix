#ifndef __GTM_REQ_VECTOR__
#define __GTM_REQ_VECTOR__

#include <mpi.h>

struct GTM_Req_Vector
{
    int *row_starts, *row_nums;
    int *col_starts, *col_nums;
    void **src_bufs;
    int *src_buf_lds;
    MPI_Op *ops;
    int curr_size, max_size;
};

typedef struct GTM_Req_Vector* GTM_Req_Vector_t;

#define DEFAULT_REQ_VEC_LEN 128

int GTM_createReqVector(GTM_Req_Vector_t *gtm_rv_);

int GTM_pushToReqVector(
    GTM_Req_Vector_t gtm_rv, MPI_Op op, 
    int row_start, int row_num,
    int col_start, int col_num,
    void *src_buf, int src_buf_ld
);

int GTM_resetReqVector(GTM_Req_Vector_t gtm_rv);

int GTM_destroyReqVector(GTM_Req_Vector_t gtm_rv);

#endif
