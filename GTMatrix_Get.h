#ifndef __GTMATRIX_GET_H__
#define __GTMATRIX_GET_H__

#include <mpi.h>
#include "GTMatrix_Typedef.h"

// Get a block from the global matrix
// Blocking call, the access operation is finished when function returns
// This call is not collective, not thread-safe
void GTM_getBlock(GTM_PARAM);

// Nonblocking get a block from the global matrix
// Nonblocking call, the access operation is posted but not finished
// This call is not collective, not thread-safe
void GTM_getBlockNB(GTM_PARAM);

// Add a request to get a block from the global matrix
// Nonblocking call, the access operation is pushed to the request queue but not posted
// This call is not collective, not thread-safe
void GTM_addGetBlockRequest(GTM_PARAM);

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
