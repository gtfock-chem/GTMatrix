#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

#include "Buzz_Matrix.h"
#include "utils.h"

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	
	int r_displs[3] = {0, 2, 5};
	int c_displs[3] = {0, 3, 6};
	int bs[4] = {6, 6, 9, 9};
	int offset[4] = {0, 3, 12, 15};
	int recv_buf[9], mat[30];
	
	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	
	MPI_Comm comm_world;
	MPI_Comm_dup(MPI_COMM_WORLD, &comm_world);
	
	Buzz_Matrix_t bm;
	
	memset(&mat[0], 0, 4 * 30);
	
	// 2 * 2 proc grid, matrix size 5 * 6
	Buzz_createBuzzMatrix(
		&bm, comm_world, MPI_INT, 4, my_rank, 5, 6, 
		2, 2, &r_displs[0], &c_displs[0], NULL, 0
	);
	
	// Set local data
	int *mat_block = (int*) bm->mat_block;
	for (int i = 0; i < bm->my_nrows * bm->my_ncols; i++)
		mat_block[i] = 100 * my_rank + i;
	
	// Start to fetch blocks from other processes
	Buzz_startBuzzMatrixReadOnlyEpoch(bm);
	if (my_rank == 0)
	{
		Buzz_requestBlock(bm, 1, 1, 1, 3, &mat[7], 6);
		MPI_Win_flush_all(bm->mpi_win);
		
		print_int_mat(&mat[0], 6, 5, 6, "Full matrix");
	}
	Buzz_stopBuzzMatrixReadOnlyEpoch(bm);
	
	// Update matrix data
	for (int i = 0; i < bm->my_nrows * bm->my_ncols; i++)
		mat_block[i] += 114;
	// Start to fetch blocks from other processes again
	Buzz_startBuzzMatrixReadOnlyEpoch(bm);
	if (my_rank == 0)
	{
		Buzz_requestBlock(bm, 1, 3, 2, 3, &mat[8], 6);
		Buzz_requestBlock(bm, 3, 2, 4, 2, &mat[22], 6);
		MPI_Win_flush_all(bm->mpi_win);
		
		print_int_mat(&mat[0], 6, 5, 6, "Full matrix");
	}
	Buzz_stopBuzzMatrixReadOnlyEpoch(bm);
	
	Buzz_destroyBuzzMatrix(bm);
	
	MPI_Finalize();
}