#ifndef __GTMATRIX_H__
#define __GTMATRIX_H__

#ifdef __cplusplus
extern "C" {
#endif

// GTMatrix structure definition, constructor and destructor 
#include "GTMatrix_Typedef.h"

// GTMatrix one-sided get operations
#include "GTMatrix_Get.h"

// GTMatrix one-sided put and accumulation operations
#include "GTMatrix_Update.h"

// GTMatrix other operations: symmetrize, fill with value
#include "GTMatrix_Other.h"

#ifdef __cplusplus
}
#endif

#endif
