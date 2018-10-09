#ifndef __BUZZ_MATRIX_TYPE_H__
#define __BUZZ_MATRIX_TYPE_H__

#include <mpi.h>
#include "Buzz_Req_Vector.h"

// Distributed matrix, 2D checkerboard partition, no cyclic 
struct Buzz_Matrix
{
	// MPI components
	MPI_Comm mpi_comm;           // Target communicator
	MPI_Win  mpi_win;            // MPI window for distribute matrix
	MPI_Datatype datatype;       // Matrix data type
	
	// Matrix size and partition
	int nrows, ncols;            // Matrix size
	int r_blocks,  c_blocks;     // Number of blocks on row and column directions, r_blocks * c_blocks == comm_size
	int my_rowblk, my_colblk;    // Which row & column block this process is
	int my_nrows,  my_ncols;     // How many row & column local block has
	int *r_displs, *r_blklens;   // Displacements and length of each block on row direction
	int *c_displs, *c_blklens;   // Displacements and length of each block on column direction
	//int *ld_blks;                // Leading dimensions of each matrix block
	int ld_local;                // Local matrix block's leading dimension
	
	// MPI Global window
	int unit_size;               // Size of matrix data type, unit is byte
	int my_rank, comm_size;      // Rank of this process and number of process in the global communicator
	void *mat_block;             // Local matrix block
	void *recv_buff;             // Receive buffer for Buzz_getBlockList()
	int rcvbuf_size;             // Size of recv_buff, unit is byte
	int *proc_cnt;               // Array for counting how many MPI_Get requests 
	int nthreads;                // Maximum number of thread that calls getBlock
	void *symm_buf;              // Buffer for symmetrization
	Buzz_Req_Vector_t *req_vec;  // Update requests for each process
	int is_batch_updating;       // If we can submit update request
	int is_batch_getting;        // If we can submit get request
	
	// Predefined small block data types
	MPI_Datatype *sb_stride;     // Data type for stride != columns 
	MPI_Datatype *sb_nostride;   // Data type for stride == columns 
};

typedef struct Buzz_Matrix* Buzz_Matrix_t;

#define DEFAULE_RCV_BUF_SIZE 262144  // 32^2 * 16 * 8 bytes
#define MPI_DT_SB_DIM_MAX    16

// Create and initialize a Buzz_Matrix structure
// Each process has one matrix block, process ranks are arranged in row-major style
// This call is collective, thread-safe
// [out] Buzz_mat   : Return pointer to the created Buzz_Matrix structure
// [in]  comm       : MPI communicator used in this distributed matrix
// [in]  datatype   : Matrix data type
// [in]  unit_size  : Size of matrix data type, unit is byte
// [in]  my_rank    : MPI Rank of this process
// [in]  nrows      : Number of rows in matrix 
// [in]  ncols      : Number of columns in matrix
// [in]  r_blocks   : Number of blocks on row direction
// [in]  c_blocks   : Number of blocks on column direction
// [in]  *r_displs  : Row direction displacement array, nrows+1 elements
// [in]  *c_displs  : Column direction displacement array, ncols+1 elements
// [in]  nthreads   : Maximum number of thread that calls getBlock
// [in]  buf_size   : Receive buffer size (bytes) of each thread, <= 0 will use default value (256KB)
void Buzz_createBuzzMatrix(
	Buzz_Matrix_t *Buzz_mat, MPI_Comm comm, MPI_Datatype datatype,
	int unit_size, int my_rank, int nrows, int ncols,
	int r_blocks, int c_blocks, int *r_displs, int *c_displs,
	int nthreads, int buf_size
);

// Free a Buzz_Matrix structure
// This call is collective, thread-safe
void Buzz_destroyBuzzMatrix(Buzz_Matrix_t Buzz_mat);

#endif
