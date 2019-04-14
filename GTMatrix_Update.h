#ifndef __GTMATRIX_UPDATE_H__
#define __GTMATRIX_UPDATE_H__

#include <mpi.h>
#include "GTMatrix_Typedef.h"

// Notice: user should guarantee the write sequence is correct or write targets
// do not overlap with each other when putting blocks of data. 

// Update (put or accumulate) a block to a process using MPI_Accumulate
// This function should not be directly called, use GTM_updateBlock() instead
// [in] dst_rank   : Target process
// [in] op         : MPI operation, only support MPI_SUM (accumulate) and MPI_REPLACE (MPI_Put)
// [in] row_start  : 1st row of the required block
// [in] row_num    : Number of rows the required block has
// [in] col_start  : 1st column of the required block
// [in] col_num    : Number of columns the required block has
// [in] *src_buf   : Source buffer
// [in] src_buf_ld : Leading dimension of the source buffer
// [in] dst_locked : If the target rank has been locked with MPI_Win_lock, = 0 will 
//                   use MPI_Win_lock & MPI_Win_unlock and the function is blocking
void GTM_updateBlockToProcess(
    GTMatrix_t gt_mat, int dst_rank, MPI_Op op, 
    int row_start, int row_num,
    int col_start, int col_num,
    void *src_buf, int src_buf_ld, 
    int dst_locked
);

// Put a block to a process using GTM_updateBlockToProcess()
// This function should not be directly called, use GTM_putBlock() instead
void GTM_putBlockToProcess(
    GTMatrix_t gt_mat, int dst_rank,
    int row_start, int row_num,
    int col_start, int col_num,
    void *src_buf, int src_buf_ld, 
    int dst_locked
);

// Accumulate a block to a process using GTM_updateBlockToProcess()
// This function should not be directly called, use GTM_accumulateBlock() instead
void GTM_accumulateBlockToProcess(
    GTMatrix_t gt_mat, int dst_rank,
    int row_start, int row_num,
    int col_start, int col_num,
    void *src_buf, int src_buf_ld, 
    int dst_locked
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
// [in] blocking   : If blocking = 0, the update request will be put in queues and 
//                   finished later with GTM_execBatchUpdate(); otherwise the update 
//                   is finished when this function returns
void GTM_updateBlock(
    GTMatrix_t gt_mat, MPI_Op op, 
    int row_start, int row_num,
    int col_start, int col_num,
    void *src_buf, int src_buf_ld,
    int blocking
);

// Put a block to all related processes using GTM_updateBlock(), blocking operation
// This call is not collective, not thread-safe
void GTM_putBlock(
    GTMatrix_t gt_mat,
    int row_start, int row_num,
    int col_start, int col_num,
    void *src_buf, int src_buf_ld
);

// Accumulate a block to all related processes using GTM_updateBlock(), blocking operation
// This call is not collective, not thread-safe
void GTM_accumulateBlock(
    GTMatrix_t gt_mat,
    int row_start, int row_num,
    int col_start, int col_num,
    void *src_buf, int src_buf_ld
);

// Add a request to put a block to all related processes using 
// GTM_updateBlock(), non-blocking operation
// This call is not collective, not thread-safe
void GTM_addPutBlockRequest(
    GTMatrix_t gt_mat,
    int row_start, int row_num,
    int col_start, int col_num,
    void *src_buf, int src_buf_ld
);

// Add a request to accumulate a block to all related processes using 
// GTM_updateBlock(), non-blocking operation
// This call is not collective, not thread-safe
void GTM_addAccumulateBlockRequest(
    GTMatrix_t gt_mat,
    int row_start, int row_num,
    int col_start, int col_num,
    void *src_buf, int src_buf_ld
);

// Start a batch update epoch and allow to submit update requests
// This call is not collective, not thread-safe
void GTM_startBatchUpdate(GTMatrix_t gt_mat);

// Execute all update requests in the queues
// This call is not collective, not thread-safe
void GTM_execBatchUpdate(GTMatrix_t gt_mat);

// Stop a batch update epoch and disallow to submit update requests
// This call is not collective, not thread-safe
void GTM_stopBatchUpdate(GTMatrix_t gt_mat);


#endif
