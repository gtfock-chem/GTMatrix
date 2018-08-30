#ifndef __BUZZ_MATRIX_H__
#define __BUZZ_MATRIX_H__

#include <mpi.h>

// Distributed matrix, 2D checkerboard partition, no cyclic 
struct Buzz_Matrix
{
	// MPI components
	MPI_Comm mpi_comm, shm_comm; // Target communicator
	MPI_Win  mpi_win,  shm_win;  // MPI window for distribute matrix
	MPI_Datatype datatype;       // Matrix data type
	
	// Matrix size and partition
	int nrows, ncols;            // Matrix size
	int r_blocks,  c_blocks;     // Number of blocks on row and column directions, r_blocks * c_blocks == comm_size
	int my_rowblk, my_colblk;    // Which row & column block this process is
	int my_nrows,  my_ncols;     // How many row & column local block has
	int *r_displs, *r_blklens;   // Displacements and length of each block on row direction
	int *c_displs, *c_blklens;   // Displacements and length of each block on column direction
	int *ld_blks;                // Leading dimensions of each matrix block
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
	
	// MPI Shared memory window
	int shm_rank, shm_size;      // Rank of this process and number of process in the shared memory communicator
	int *shm_global_ranks;       // Global ranks (in mpi_comm) of the processes in shm_comm
	void **shm_mat_blocks;       // Arrays of all shared memory ranks' pointers
	
	// Predefined small block data types
	MPI_Datatype *sb_stride;     // Data type for stride != columns 
	MPI_Datatype *sb_nostride;   // Data type for stride == columns 
};

typedef struct Buzz_Matrix* Buzz_Matrix_t;

#define DEFAULE_RCV_BUF_SIZE 262144  // 32^2 * 16 * 8 bytes
#define MPI_DT_SB_DIM_MAX    16

// Create and initialize a Buzz_Matrix structure
// Each process has one matrix block, process ranks are arranged in row-major style
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
void Buzz_destroyBuzzMatrix(Buzz_Matrix_t Buzz_mat);

// Start a read-only epoch to a Buzz_Matrix, user should guarantee no 
// modification to matrix data in the this epoch 
// A MPI_Barrier is called at the beginning of this function 
void Buzz_startBuzzMatrixReadOnlyEpoch(Buzz_Matrix_t Buzz_mat);

// Finish all RMA operations and stop the read-only epoch
// A MPI_Barrier is called at the end of this function 
void Buzz_stopBuzzMatrixReadOnlyEpoch(Buzz_Matrix_t Buzz_mat);

// Get a block from all related processes using MPI_Get
// Non-blocking, data may not be ready before synchronization
// [out] *proc_cnt  : Array for counting how many MPI_Get requests a process has
// [in]  row_start  : 1st row of the required block
// [in]  row_num    : Number of rows the required block has
// [in]  col_start  : 1st column of the required block
// [in]  col_num    : Number of columns the required block has
// [out] *src_buf   : Receive buffer
// [in]  src_buf_ld : Leading dimension of the received buffer
void Buzz_getBlock(
	Buzz_Matrix_t Buzz_mat, int *proc_cnt, 
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld
);

// Synchronize and complete all outstanding MPI_Get requests according to the 
// counter array and reset the counter array
// [inout] *proc_cnt  : Array for counting how many MPI_Get requests a process has
void Buzz_flushProcListGetRequests(Buzz_Matrix_t Buzz_mat, int *proc_cnt);

// Get a list of blocks from all related processes using MPI_Get
// [in]  nblocks          : Number of blocks to get
// [in]  tid              : Thread ID
// [in]  *row_start       : Array that stores 1st row of the required blocks
// [in]  *row_num         : Array that stores Number of rows the required blocks has
// [in]  *col_start       : Array that stores 1st column of the required blocks
// [in]  *col_num         : Array that stores Number of columns the required blocks has
// [out] **thread_src_buf : Pointer to this thread's receive buffer. Blocks are stored 
//                          one by one without padding, col_num[i] is leading dimension
// [out] @return          : Number of requested blocks, < nblocks means the receive buffer 
//                          is not large enough
int Buzz_getBlockList(
	Buzz_Matrix_t Buzz_mat, int nblocks, int tid, 
	int *row_start, int *row_num,
	int *col_start, int *col_num,
	void **thread_src_buf
);

// Complete all outstanding MPI_Get requests from the same thread 
// tid : Thread ID
void Buzz_completeGetBlocks(Buzz_Matrix_t Buzz_mat, int tid);

// Update (put or accumulate) a block to all related processes using MPI_Accumulate
// Blocking, target processes will be locked with MPI_Win_lock
// [in] op         : MPI operation, only support MPI_SUM (accumulate) and MPI_REPLACE (MPI_Put)
// [in] row_start  : 1st row of the required block
// [in] row_num    : Number of rows the required block has
// [in] col_start  : 1st column of the required block
// [in] col_num    : Number of columns the required block has
// [in] *src_buf   : Receive buffer
// [in] src_buf_ld : Leading dimension of the received buffer
void Buzz_updateBlock(
	Buzz_Matrix_t Buzz_mat, MPI_Op op, 
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld
);

// Put a block to all related processes using Buzz_updateBlock()
void Buzz_putBlock(
	Buzz_Matrix_t Buzz_mat,
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld
);

// Accumulate a block to all related processes using Buzz_updateBlock()
void Buzz_accumulateBlock(
	Buzz_Matrix_t Buzz_mat,
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld
);

// Fill the Buzz_Matrix with a single value
// [in] *value : Pointer to the value of appropriate type that matches
//               Buzz_Matrix's unit_size, now support int and double
void Buzz_fillBuzzMatrix(Buzz_Matrix_t Buzz_mat, void *value);

// Symmetrize a matrix, i.e. (A + A^T) / 2, now support int and double data type
void Buzz_symmetrizeBuzzMatrix(Buzz_Matrix_t Buzz_mat);

// *=================== Internal Functions ===================*
// *  Functions below are used by functions above, they are   *
// *  listed here just as references. USE AT YOUR OWN RISK!   *
// *==========================================================*

// Get a block from a process using MPI_Get
// Non-blocking, data may not be ready before synchronization
// This function should not be directly called, use Buzz_getBlock() instead
// [in]  dst_proc   : Target process
// [in]  row_start  : 1st row of the required block
// [in]  row_num    : Number of rows the required block has
// [in]  col_start  : 1st column of the required block
// [in]  col_num    : Number of columns the required block has
// [out] *src_buf   : Receive buffer
// [in]  src_buf_ld : Leading dimension of the received buffer
void Buzz_getBlockFromProcess(
	Buzz_Matrix_t Buzz_mat, int dst_rank, 
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld
);

// Update (put or accumulate) a block to a process using MPI_Accumulate
// Blocking, target process will be locked with MPI_Win_lock
// This function should not be directly called, use Buzz_updateBlock() instead
// [in] dst_rank   : Target process
// [in] op         : MPI operation, only support MPI_SUM (accumulate) and MPI_REPLACE (MPI_Put)
// [in] row_start  : 1st row of the required block
// [in] row_num    : Number of rows the required block has
// [in] col_start  : 1st column of the required block
// [in] col_num    : Number of columns the required block has
// [in] *src_buf   : Source buffer
// [in] src_buf_ld : Leading dimension of the source buffer
void Buzz_updateBlockToProcess(
	Buzz_Matrix_t Buzz_mat, int dst_rank, MPI_Op op, 
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld
);

// Put a block to a process using Buzz_updateBlockToProcess()
// This function should not be directly called, use Buzz_putBlock() instead
void Buzz_putBlockToProcess(
	Buzz_Matrix_t Buzz_mat, int dst_rank,
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld
);

// Accumulate a block to a process using Buzz_updateBlockToProcess()
// This function should not be directly called, use Buzz_accumulateBlock() instead
void Buzz_accumulateBlockToProcess(
	Buzz_Matrix_t Buzz_mat, int dst_rank,
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld
);

#endif
