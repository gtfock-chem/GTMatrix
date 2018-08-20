#ifndef __BUZZ_MATRIX_H__
#define __BUZZ_MATRIX_H__

#include <mpi.h>

// Distributed matrix, 2D checkerboard partition, no cyclic 
struct Buzz_Matrix
{
	MPI_Comm mpi_comm;           // Target communicator
	MPI_Win mpi_win;             // MPI window for distribute matrix
	MPI_Info mpi_info;           // MPI info
	MPI_Datatype datatype;       // Matrix data type
	size_t unit_size;            // Size of matrix data type, unit is byte
	int my_rank, comm_size;      // Rank of this node and number of process in the communicator
	int nrows, ncols;            // Matrix size
	int r_blocks, c_blocks;      // Number of blocks on row and column directions, r_blocks * c_blocks == comm_size
	int my_rowblk, my_colblk;    // Which row & column block this process is
	int my_nrows, my_ncols;      // How many row & column local block has
	int *r_displs, *r_blklens;   // Displacements and length of each block on row direction
	int *c_displs, *c_blklens;   // Displacements and length of each block on column direction
	int *ld_blks;                // Leading dimensions of each matrix block
	int ld_local, mb_alloc;      // Local matrix block's leading dimension and if mat_block is allocated by Buzz_Matrix
	void *mat_block;             // Pointer to local matrix block
	void *recv_buff;             // Pointer to receive buffer
	size_t rcvbuf_size;          // Size of recv_buff, unit is byte
};

typedef struct Buzz_Matrix* Buzz_Matrix_t;

#define DEFAULE_RCV_BUF_SIZE 1048576  // 1M

// Create and initialize a Buzz_Matrix structure
// Each process has one matrix block, process ranks are arranged in row-major style
// Buzz_mat   : Return pointer to the created Buzz_Matrix structure
// comm       : MPI communicator used in this distributed matrix
// datatype   : Matrix data type
// unit_size  : Size of matrix data type, unit is byte
// my_rank    : MPI Rank of this process
// nrows      : Number of rows in matrix 
// ncols      : Number of columns in matrix
// r_blocks   : Number of blocks on row direction
// c_blocks   : Number of blocks on column direction
// *r_displs  : Pointer to row direction displacement array, should have nrows+1 elements
// *c_displs  : Pointer to column direction displacement array, should have ncols+1 elements
// *mat_block : Pointer to exist local matrix block, optional
// ld_local   : Leading dimension of exist local matrix block, <= 0 will create 
//              a new buffer for local matrix block and ld_local = c_blklens[my_colblk]
void Buzz_createBuzzMatrix(
	Buzz_Matrix_t *Buzz_mat, MPI_Comm comm, MPI_Datatype datatype,
	size_t unit_size, int my_rank, int nrows, int ncols,
	int r_blocks, int c_blocks, int *r_displs, int *c_displs,
	void *mat_block, int ld_local
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

// Get a block from a process using MPI_Get
// Non-blocking, data may not be ready before synchronization
// target_proc    : Target process
// req_row_start  : 1st row of the required block
// req_row_num    : Number of rows the required block has
// req_col_start  : 1st column of the required block
// req_col_num    : Number of columns the required block has
// *req_rcv_buf   : Receive buffer
// req_rcv_buf_ld : Leading dimension of the received buffer
void Buzz_getBlockFromProcess(
	Buzz_Matrix_t Buzz_mat, int target_proc, 
	int req_row_start, int req_row_num,
	int req_col_start, int req_col_num,
	void *req_rcv_buf, int req_rcv_buf_ld
);

// Get a block from all related processes using MPI_Get
// Non-blocking, data may not be ready before synchronization
// *proc_req_cnt  : Array for counting how many MPI_Get requests a process has
// req_row_start  : 1st row of the required block
// req_row_num    : Number of rows the required block has
// req_col_start  : 1st column of the required block
// req_col_num    : Number of columns the required block has
// *req_rcv_buf   : Receive buffer
// req_rcv_buf_ld : Leading dimension of the received buffer
void Buzz_getBlock(
	Buzz_Matrix_t Buzz_mat, int *proc_req_cnt, 
	int req_row_start, int req_row_num,
	int req_col_start, int req_col_num,
	void *req_rcv_buf, int req_rcv_buf_ld
);

// Synchronize and complete all outstanding MPI_Get requests 
// and reset the counter array
// *proc_req_cnt  : Array for counting how many MPI_Get requests a process has
void Buzz_completeAllGetRequests(Buzz_Matrix_t Buzz_mat, int *proc_req_cnt);

#endif
