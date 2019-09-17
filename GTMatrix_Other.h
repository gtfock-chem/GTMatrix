#ifndef __GTMATRIX_OTHER_H__
#define __GTMATRIX_OTHER_H__

#include "GTMatrix_Typedef.h"

#define SYNC_IBARRIER 1

// Synchronize all processes
int GTM_sync(GTMatrix_t gtm);

// Complete all nonblocking accesses
int GTM_waitNB(GTMatrix_t gtm);

// Fill the GTMatrix with a single value
// This call is collective, not thread-safe
// Input parameter:
// *value : Pointer to the value of appropriate type that matches
//          GTMatrix's unit_size, now support int, double, and double _Complex
int GTM_fill(GTMatrix_t gtm, void *value);

// Symmetrize a matrix, i.e. (A + A^T) / 2, now support int, double, and double _Complex
// This call is collective, not thread-safe
int GTM_symmetrize(GTMatrix_t gtm);

#endif
