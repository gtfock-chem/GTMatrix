#ifndef __GTMATRIX_RETVAL_H__
#define __GTMATRIX_RETVAL_H__

#define GTM_SUCCESS          0x0000  // GTMatrix operation is performed successfully
#define GTM_NULL_PTR         0x0001  // GTMatrix pointer is NULL
#define GTM_ALLOC_FAILED     0x0002  // GTMatrix failed to allocate memory
#define GTM_INVALID_RANK     0x0003  // GTMatrix failed to create with invalid my_rank
#define GTM_INVALID_RCBLOCK  0x0004  // GTMatrix failed to create since r_blocks*c_blocks != comm_size
#define GTM_INVALID_R_DISPLS 0x0005  // GTMatrix failed to create with invalid r_displs 
#define GTM_INVALID_C_DISPLS 0x0006  // GTMatrix failed to create with invalid c_displs 
#define GTM_INVALID_BLOCK    0x0007  // GTMatrix failed to access a block with invalid range
#define GTM_INVALID_SRC_LD   0x0008  // GTMatrix failed to access a block with invalid src_ld
#define GTM_NO_BATCHED_GET   0x0009  // GTMatrix is not in batched get mode
#define GTM_NO_BATCHED_PUT   0x000A  // GTMatrix is not in batched put mode
#define GTM_NO_BATCHED_ACC   0x000B  // GTMatrix is not in batched acc mode
#define GTM_IN_BATCHED_GET   0x000C  // GTMatrix is in batched get mode
#define GTM_IN_BATCHED_PUT   0x000D  // GTMatrix is in batched put mode
#define GTM_IN_BATCHED_ACC   0x000E  // GTMatrix is in batched acc mode
#define GTM_NOT_SQUARE_MAT   0x000F  // GTMatrix failed to symmetrize a non-square matrix

#define GTM_RV_SUCCESS       0x0000  // GTMatrix request vector operation is performed successfully
#define GTM_RV_NULL_PTR      0x0101  // GTMatrix request vector pointer is NULL
#define GTM_RV_ALLOC_FAILED  0x0102  // GTMatrix request vector failed to allocate memory
#define GTM_RV_RESIZE_FAILED 0x0103  // GTMatrix request vector failed to allocate memory when resizing

#define GTM_TQ_SUCCESS       0x0000  // GTMatrix task queue operation is performed successfully
#define GTM_TQ_NULL_PTR     -0x0201  // GTMatrix task queue pointer is NULL
#define GTM_TQ_ALLOC_FAILED  0x0202  // GTMatrix task queue failed to allocate memory
#define GTM_TQ_INVALID_RANK -0x0203  // GTMatrix task queue target rank is invalid

#endif
