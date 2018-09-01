#ifndef __BUZZ_MATRIX_UPDATE_H__
#define __BUZZ_MATRIX_UPDATE_H__

#include <mpi.h>
#include "Buzz_Matrix_Typedef.h"

// Update (put or accumulate) a block to a process using MPI_Accumulate
// Blocking, target process will be locked with MPI_Win_lock
// This function should not be directly called, use Buzz_updateBlock() instead
// [in] dst_rank   : Target process
// [in] op         : MPI operation, only support MPI_SUM (accumulate) and MPI_REPLACE (MPI_Put)
// [in] row_start  : 1st row of the required block
// [in] row_num    : Number of rows the required block has
// [in] col_start  : 1st column of the required block
// [in] col_num    : Number of columns the required block has
// [in] *src_buf   : Source buffer
// [in] src_buf_ld : Leading dimension of the source buffer
void Buzz_updateBlockToProcess(
	Buzz_Matrix_t Buzz_mat, int dst_rank, MPI_Op op, 
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld, 
	int update_target_locked
);

// Put a block to a process using Buzz_updateBlockToProcess()
// This function should not be directly called, use Buzz_putBlock() instead
void Buzz_putBlockToProcess(
	Buzz_Matrix_t Buzz_mat, int dst_rank,
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld
);

// Accumulate a block to a process using Buzz_updateBlockToProcess()
// This function should not be directly called, use Buzz_accumulateBlock() instead
void Buzz_accumulateBlockToProcess(
	Buzz_Matrix_t Buzz_mat, int dst_rank,
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld
);

// Update (put or accumulate) a block to all related processes using MPI_Accumulate
// This call is not collective, not thread-safe
// [in] op         : MPI operation, only support MPI_SUM (accumulate) and MPI_REPLACE (MPI_Put)
// [in] row_start  : 1st row of the required block
// [in] row_num    : Number of rows the required block has
// [in] col_start  : 1st column of the required block
// [in] col_num    : Number of columns the required block has
// [in] *src_buf   : Receive buffer
// [in] src_buf_ld : Leading dimension of the received buffer
// [in] blocking   : If blocking = 0, the update request will be put in queues and finished
//                   later with Buzz_execBatchUpdate(); otherwise the update is finished when
//                   this function returns
void Buzz_updateBlock(
	Buzz_Matrix_t Buzz_mat, MPI_Op op, 
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld,
	int blocking
);

// Put a block to all related processes using Buzz_updateBlock(), blocking operation
// This call is not collective, not thread-safe
void Buzz_putBlock(
	Buzz_Matrix_t Buzz_mat,
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld
);

// Accumulate a block to all related processes using Buzz_updateBlock(), blocking operation
// This call is not collective, not thread-safe
void Buzz_accumulateBlock(
	Buzz_Matrix_t Buzz_mat,
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld
);

// Add a request to put a block to all related processes using 
// Buzz_updateBlock(), non-blocking operation
// This call is not collective, not thread-safe
void Buzz_addPutBlockRequest(
	Buzz_Matrix_t Buzz_mat,
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld
);

// Add a request to accumulate a block to all related processes using 
// Buzz_updateBlock(), non-blocking operation
// This call is not collective, not thread-safe
void Buzz_addAccumulateBlockRequest(
	Buzz_Matrix_t Buzz_mat,
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld
);

// Start a batch update epoch and allow to submit update requests
// This call is not collective, not thread-safe
void Buzz_startBatchUpdate(Buzz_Matrix_t Buzz_mat);

// Execute all update requests in the queues
// This call is not collective, not thread-safe
void Buzz_execBatchUpdate(Buzz_Matrix_t Buzz_mat);

// Stop a batch update epoch and disallow to submit update requests
// This call is not collective, not thread-safe
void Buzz_stopBatchUpdate(Buzz_Matrix_t Buzz_mat);


#endif
