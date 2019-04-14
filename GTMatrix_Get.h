#ifndef __GTMATRIX_GET_H__
#define __GTMATRIX_GET_H__

#include <mpi.h>
#include "GTMatrix_Typedef.h"

// Get a block from a process using MPI_Get
// This function should not be directly called, use GTM_getBlock() instead
// [in]  dst_rank   : Target process
// [in]  row_start  : 1st row of the required block
// [in]  row_num    : Number of rows the required block has
// [in]  col_start  : 1st column of the required block
// [in]  col_num    : Number of columns the required block has
// [out] *src_buf   : Receive buffer
// [in]  src_buf_ld : Leading dimension of the received buffer
// [in]  dst_locked : If the target rank has been locked with MPI_Win_lock, = 0 will 
//                    use MPI_Win_lock & MPI_Win_unlock and the function is blocking
void GTM_getBlockFromProcess(
    GTMatrix_t gt_mat, int dst_rank, 
    int row_start, int row_num,
    int col_start, int col_num,
    void *src_buf, int src_buf_ld,
    int dst_locked
);

// Get a block from all related processes using MPI_Get
// Non-blocking, data may not be ready before synchronization
// This call is not collective, thread-safe
// [in]  row_start  : 1st row of the required block
// [in]  row_num    : Number of rows the required block has
// [in]  col_start  : 1st column of the required block
// [in]  col_num    : Number of columns the required block has
// [out] *src_buf   : Receive buffer
// [in]  src_buf_ld : Leading dimension of the received buffer
// [in]  blocking   : If blocking = 0, the update request will be put in queues and 
//                    finished later with GTM_execBatchUpdate(); otherwise the update 
//                    is finished when this function returns
void GTM_getBlock(
    GTMatrix_t gt_mat,
    int row_start, int row_num,
    int col_start, int col_num,
    void *src_buf, int src_buf_ld, 
    int blocking
);

// Add a request to get a block to all related processes using 
// GTM_getBlock(), non-blocking operation
// This call is not collective, not thread-safe
void GTM_addGetBlockRequest(
    GTMatrix_t gt_mat,
    int row_start, int row_num,
    int col_start, int col_num,
    void *src_buf, int src_buf_ld
);

// Start a batch get epoch and allow to submit update requests
// This call is not collective, not thread-safe
void GTM_startBatchGet(GTMatrix_t gt_mat);

// Execute all get requests in the queues
// This call is not collective, not thread-safe
void GTM_execBatchGet(GTMatrix_t gt_mat);

// Stop a batch get epoch and allow to submit update requests
// This call is not collective, not thread-safe
void GTM_stopBatchGet(GTMatrix_t gt_mat);


#endif
