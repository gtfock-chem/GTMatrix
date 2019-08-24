#ifndef __GTMATRIX_GET_H__
#define __GTMATRIX_GET_H__

#include <mpi.h>
#include "GTMatrix_Typedef.h"

// All functions in this header file are not collective, not thread-safe.

// Get a block from the global matrix
// Blocking call, the access operation is finished when function returns
int GTM_getBlock(GTM_PARAM);

// Nonblocking get a block from the global matrix
// Nonblocking call, the access operation is posted but not finished
int GTM_getBlockNB(GTM_PARAM);

// Add a request to get a block from the global matrix
// Nonblocking call, the access operation is pushed to the request queue but not posted
int GTM_addGetBlockRequest(GTM_PARAM);

// Start a batch get epoch and allow to submit get requests
int GTM_startBatchGet(GTMatrix_t gt_mat);

// Execute all get requests in the queues
int GTM_execBatchGet(GTMatrix_t gt_mat);

// Stop a batch get epoch and disallow to submit get requests
int GTM_stopBatchGet(GTMatrix_t gt_mat);


#endif
