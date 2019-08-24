#ifndef __GTMATRIX_TYPE_H__
#define __GTMATRIX_TYPE_H__

#include <mpi.h>
#include "GTM_Req_Vector.h"

// Distributed matrix, 2D checkerboard partition, no cyclic 
struct GTMatrix
{
    // MPI components
    MPI_Comm mpi_comm, shm_comm; // Target communicator
    MPI_Win  mpi_win,  shm_win;  // MPI window for distribute matrix
    MPI_Datatype datatype;       // Matrix data type
    int acc_lock_type;           // MPI window lock type for update (accumulate & put)
    
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
    void *symm_buf;              // Buffer for symmetrization
    GTM_Req_Vector_t *req_vec;   // Update requests for each process
    int in_batch_get;            // If GTMatrix is in batched get access
    int in_batch_put;            // If GTMatrix is in batched put access
    int in_batch_acc;            // If GTMatrix is in batched acc access
    int *nb_op_proc_cnt;         // Number of outstanding RMA operations on each process from nonblocking calls
    int nb_op_cnt;               // Total number of outstanding RMA operations from nonblocking calls
    int max_nb_acc, max_nb_get;  // Maximum number of outstanding update / get operations from nonblocking calls
    
    // MPI Shared memory window
    int shm_rank, shm_size;      // Rank of this process and number of process in the shared memory communicator
    int *shm_global_ranks;       // Global ranks (in mpi_comm) of the processes in shm_comm
    void **shm_mat_blocks;       // Arrays of all shared memory ranks' pointers
    
    // Predefined small block data types
    MPI_Datatype *sb_stride;     // Data type for stride != columns 
    MPI_Datatype *sb_nostride;   // Data type for stride == columns 
};

typedef struct GTMatrix* GTMatrix_t;

#define MPI_DT_SB_DIM_MAX    16

#define BLOCKING_ACCESS      0  // The access operation is finished when function returns
#define NONBLOCKING_ACCESS   1  // The access operation is posted but not finished when function returns
#define BATCH_ACCESS         2  // The access operation is pushed to the request queue but not posted

#define GTM_PARAM \
    GTMatrix_t gtm, int row_start, int row_num, \
    int col_start, int col_num, void *src_buf, int src_buf_ld
// gtm        : GTMatrix handle
// row_start  : 1st row of the block
// row_num    : Number of rows the block has
// col_start  : 1st column of the block
// col_num    : Number of columns the block has
// *src_buf   : Local buffer for storing the fetched block or 
//              containing the block to be sent
// src_buf_ld : Leading dimension of src_buf

// Create and initialize a GTMatrix structure
// Each process has one matrix block, process ranks are arranged in row-major style
// This call is collective, thread-safe
// Input parameters:
//   comm      : MPI communicator used in this distributed matrix
//   datatype  : Matrix data type
//   unit_size : Size of matrix data type, unit is byte
//   my_rank   : MPI Rank of this process
//   nrows     : Number of rows in matrix 
//   ncols     : Number of columns in matrix
//   r_blocks  : Number of blocks on row direction
//   c_blocks  : Number of blocks on column direction
//   *r_displs : Row direction displacement array, nrows+1 elements
//   *c_displs : Column direction displacement array, ncols+1 elements
// Output parameter:
//   *_gtm : Pointer to the created GTMatrix structure
int GTM_create(
    GTMatrix_t *_gtm, MPI_Comm comm, MPI_Datatype datatype,
    int unit_size, int my_rank, int nrows, int ncols,
    int r_blocks, int c_blocks, int *r_displs, int *c_displs
);

// Free a GTMatrix structure
// This call is collective, thread-safe
int GTM_destroy(GTMatrix_t gtm);

#endif
