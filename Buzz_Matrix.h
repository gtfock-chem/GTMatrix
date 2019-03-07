#ifndef __BUZZ_MATRIX_H__
#define __BUZZ_MATRIX_H__

#ifdef __cplusplus
extern "C" {
#endif

// Buzz_Matrix structure definition, constructor and destructor 
#include "Buzz_Matrix_Typedef.h"

// Buzz_Matrix one-sided get operations
#include "Buzz_Matrix_Get.h"

// Buzz_Matrix one-sided put and accumulation operations
#include "Buzz_Matrix_Update.h"

// Buzz_Matrix other operations: symmetrize, fill with value
#include "Buzz_Matrix_Other.h"

#ifdef __cplusplus
}
#endif

#endif
