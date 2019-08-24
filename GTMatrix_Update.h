#ifndef __GTMATRIX_UPDATE_H__
#define __GTMATRIX_UPDATE_H__

#include <mpi.h>
#include "GTMatrix_Typedef.h"

// Notice: user should guarantee the write sequence is correct or write targets
// do not overlap with each other when putting blocks of data. 

// All functions in this header file are not collective, not thread-safe.

// Put / accumulate a block to the global matrix
// Blocking call, the access operation is finished when function returns
int GTM_putBlock(GTM_PARAM);
int GTM_accBlock(GTM_PARAM);

// Nonblocking put / accumulate a block to the global matrix
// Nonblocking call, the access operation is posted but not finished
int GTM_putBlockNB(GTM_PARAM);
int GTM_accBlockNB(GTM_PARAM);

// Add a request to put / accumulate a block to the global matrix
// Nonblocking call, the access operation is pushed to the request queue but not posted
int GTM_addPutBlockRequest(GTM_PARAM);
int GTM_addAccBlockRequest(GTM_PARAM);

// Start a batch update epoch and allow to submit update requests
int GTM_startBatchPut(GTMatrix_t gt_mat);
int GTM_startBatchAcc(GTMatrix_t gt_mat);

// Execute all update requests in the queues
int GTM_execBatchPut(GTMatrix_t gt_mat);
int GTM_execBatchAcc(GTMatrix_t gt_mat);

// Stop a batch update epoch and disallow to submit update requests
int GTM_stopBatchPut(GTMatrix_t gt_mat);
int GTM_stopBatchAcc(GTMatrix_t gt_mat);


#endif
