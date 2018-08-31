#ifndef __BUZZ_REQ_VECTOR__
#define __BUZZ_REQ_VECTOR__

#include <mpi.h>

struct Buzz_Req_Vector
{
	int *row_starts, *row_nums;
	int *col_starts, *col_nums;
	void **src_bufs;
	int *src_buf_lds;
	MPI_Op *ops;
	int curr_size, max_size;
};

typedef struct Buzz_Req_Vector* Buzz_Req_Vector_t;

#define DEFAULT_REQ_VEC_LEN 128

void Buzz_createReqVector(Buzz_Req_Vector_t *brv_);

void Buzz_pushToReqVector(
	Buzz_Req_Vector_t brv, MPI_Op op, 
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld
);

void Buzz_resetReqVector(Buzz_Req_Vector_t brv);

void Buzz_destroyReqVector(Buzz_Req_Vector_t brv);

#endif
