#ifndef __GTMATRIX_OTHER_H__
#define __GTMATRIX_OTHER_H__

#include "GTMatrix_Typedef.h"

#define SYNC_IBARRIER 1

// Synchronize all processes
void GTM_Sync(GTMatrix_t gt_mat);

// Fill the GTMatrix with a single value
// This call is collective, not thread-safe
// [in] *value : Pointer to the value of appropriate type that matches
//               GTMatrix's unit_size, now support int and double
void GTM_fillGTMatrix(GTMatrix_t gt_mat, void *value);

// Symmetrize a matrix, i.e. (A + A^T) / 2, now support int and double data type
// This call is collective, not thread-safe
void GTM_symmetrizeGTMatrix(GTMatrix_t gt_mat);

#endif
