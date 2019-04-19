#ifndef __GTMATRIX_UPDATE_H__
#define __GTMATRIX_UPDATE_H__

#include <mpi.h>
#include "GTMatrix_Typedef.h"

// Notice: user should guarantee the write sequence is correct or write targets
// do not overlap with each other when putting blocks of data. 

// Put a block to the global matrix
// Blocking call, the access operation is finished when function returns
// This call is not collective, not thread-safe
void GTM_putBlock(GTM_PARAM);

// Accumulate a block to the global matrix
// Blocking call, the access operation is finished when function returns
// This call is not collective, not thread-safe
void GTM_accumulateBlock(GTM_PARAM);

// Add a request to put a block to the global matrix
// Nonblocking call, the access operation is pushed to the request queue but not posted
// This call is not collective, not thread-safe
void GTM_addPutBlockRequest(GTM_PARAM);

// Add a request to accumulate a block to the global matrix
// Nonblocking call, the access operation is pushed to the request queue but not posted
// This call is not collective, not thread-safe
void GTM_addAccumulateBlockRequest(GTM_PARAM);

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
