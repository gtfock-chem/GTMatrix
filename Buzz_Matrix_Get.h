#ifndef __BUZZ_MATRIX_GET_H__
#define __BUZZ_MATRIX_GET_H__

#include <mpi.h>
#include "Buzz_Matrix_Typedef.h"

// Get a block from a process using MPI_Get
// This function should not be directly called, use Buzz_getBlock() instead
// [in]  dst_proc   : Target process
// [in]  row_start  : 1st row of the required block
// [in]  row_num    : Number of rows the required block has
// [in]  col_start  : 1st column of the required block
// [in]  col_num    : Number of columns the required block has
// [out] *src_buf   : Receive buffer
// [in]  src_buf_ld : Leading dimension of the received buffer
// [out] @return    : Number of RMA requests issued
// [in] dst_locked : If the target rank has been locked with MPI_Win_lock, = 0 will 
//                   use MPI_Win_lock & MPI_Win_unlock and the function is blocking
int Buzz_getBlockFromProcess(
	Buzz_Matrix_t Buzz_mat, int dst_rank, 
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld,
	int dst_locked
);

// Get a block from all related processes using MPI_Get
// Non-blocking, data may not be ready before synchronization
// This call is not collective, thread-safe
// [out] *proc_cnt  : Array for counting how many MPI_Get requests a process has
// [in]  row_start  : 1st row of the required block
// [in]  row_num    : Number of rows the required block has
// [in]  col_start  : 1st column of the required block
// [in]  col_num    : Number of columns the required block has
// [out] *src_buf   : Receive buffer
// [in]  src_buf_ld : Leading dimension of the received buffer
// [in]  blocking   : If blocking = 0, the update request will be put in queues and 
//                    finished later with Buzz_execBatchUpdate(); otherwise the update 
//                    is finished when this function returns
// [in] is_mt       : = 1 means this function is called from a multi-thread region and
//                    blocking should be 0
void Buzz_getBlock(
	Buzz_Matrix_t Buzz_mat, int *proc_cnt, 
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld, 
	int blocking,  int is_mt
);

// Add a get to put a block to all related processes using 
// Buzz_getBlock(), non-blocking operation
// This call is not collective, not thread-safe
void Buzz_addGetBlockRequest(
	Buzz_Matrix_t Buzz_mat,
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld
);

// Start a read-only epoch to a Buzz_Matrix, user should guarantee no 
// modification to matrix data in the this epoch 
// A MPI_Barrier is called at the beginning of this function 
// This call is collective, not thread-safe
void Buzz_startBuzzMatrixReadOnlyEpoch(Buzz_Matrix_t Buzz_mat);

// Finish all RMA operations and stop the read-only epoch
// A MPI_Barrier is called at the end of this function 
// This call is collective, not thread-safe
void Buzz_stopBuzzMatrixReadOnlyEpoch(Buzz_Matrix_t Buzz_mat);

// Synchronize and complete all outstanding MPI_Get requests according to the 
// counter array and reset the counter array
// This call is not collective, thread-safe
// [inout] *proc_cnt  : Array for counting how many MPI_Get requests a process has
void Buzz_flushProcListGetRequests(Buzz_Matrix_t Buzz_mat, int *proc_cnt);

// Start a batch get epoch and allow to submit update requests
// This call is not collective, not thread-safe
void Buzz_startBatchGet(Buzz_Matrix_t Buzz_mat);

// Execute all get requests in the queues
// This call is not collective, not thread-safe
void Buzz_execBatchGet(Buzz_Matrix_t Buzz_mat);

// Stop a batch get epoch and disallow to submit update requests
// This call is not collective, not thread-safe
void Buzz_stopBatchGet(Buzz_Matrix_t Buzz_mat);

// Get a list of blocks from all related processes using MPI_Get
// This function is designed to be called in multi-thread parallel region
// This call is not collective, thread-safe
// [in]  nblocks          : Number of blocks to get
// [in]  tid              : Thread ID
// [in]  *row_start       : Array that stores 1st row of the required blocks
// [in]  *row_num         : Array that stores Number of rows the required blocks has
// [in]  *col_start       : Array that stores 1st column of the required blocks
// [in]  *col_num         : Array that stores Number of columns the required blocks has
// [out] **thread_src_buf : Pointer to this thread's receive buffer. Blocks are stored 
//                          one by one without padding, col_num[i] is leading dimension
// [out] @return          : Number of requested blocks, < nblocks means the receive buffer 
//                          is not large enough
int Buzz_getBlockList_mt(
	Buzz_Matrix_t Buzz_mat, int nblocks, int tid, 
	int *row_start, int *row_num,
	int *col_start, int *col_num,
	void **thread_src_buf
);

// Complete all outstanding MPI_Get requests from the same thread 
// This function is designed to be called in multi-thread parallel region
// This call is not collective, thread-safe
// tid : Thread ID
void Buzz_completeGetBlocks_mt(Buzz_Matrix_t Buzz_mat, int tid);

#endif
