#ifndef __BUZZ_MATRIX_OTHER_H__
#define __BUZZ_MATRIX_OTHER_H__

#include "Buzz_Matrix_Typedef.h"

// Fill the Buzz_Matrix with a single value
// This call is collective, not thread-safe
// [in] *value : Pointer to the value of appropriate type that matches
//               Buzz_Matrix's unit_size, now support int and double
void Buzz_fillBuzzMatrix(Buzz_Matrix_t Buzz_mat, void *value);

// Symmetrize a matrix, i.e. (A + A^T) / 2, now support int and double data type
// This call is collective, not thread-safe
void Buzz_symmetrizeBuzzMatrix(Buzz_Matrix_t Buzz_mat);

#endif
