#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

#include "Buzz_Matrix.h"
#include "utils.h"

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	
	int r_displs[5] = {0, 3, 6, 8, 12};
	int c_displs[5] = {0, 2, 4, 6, 8};
	int mat[96];
	
	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	
	MPI_Comm comm_world;
	MPI_Comm_dup(MPI_COMM_WORLD, &comm_world);
	
	Buzz_Matrix_t bm;
	
	// 4 * 4 proc grid, matrix size 10 * 10
	Buzz_createBuzzMatrix(
		&bm, comm_world, MPI_INT, 4, my_rank, 12, 8, 
		4, 4, &r_displs[0], &c_displs[0], NULL, 0
	);
	
	// Set local data
	int *mat_block = (int*) bm->mat_block;
	for (int i = 0; i < bm->my_nrows * bm->my_ncols; i++)
		mat_block[i] = 100 * my_rank + i;
	
	// Start to fetch blocks from other processes
	memset(&mat[0], 0, 4 * 96);
	Buzz_startBuzzMatrixReadOnlyEpoch(bm);
	if (my_rank == 0)
	{
		Buzz_requestBlock(bm, 0, 12, 0, 8, &mat[0], 8);
		printf("Proc 0 ready to flush all\n");
		MPI_Win_flush_all(bm->mpi_win);
		
		print_int_mat(&mat[0], 8, 12, 8, "Full matrix");
	}
	Buzz_stopBuzzMatrixReadOnlyEpoch(bm);
	
	// Update matrix data
	for (int i = 0; i < bm->my_nrows * bm->my_ncols; i++)
		mat_block[i] += 114;
	// Start to fetch blocks from other processes again
	memset(&mat[0], 0, 4 * 96);
	Buzz_startBuzzMatrixReadOnlyEpoch(bm);
	if (my_rank == 0)
	{
		Buzz_requestBlock(bm, 1, 5, 1, 5, &mat[9], 8);
		Buzz_requestBlock(bm, 10, 2, 1, 7, &mat[81], 8);
		MPI_Win_flush_all(bm->mpi_win);
		
		print_int_mat(&mat[0], 8, 12, 8, "Full matrix");
	}
	Buzz_stopBuzzMatrixReadOnlyEpoch(bm);
	
	Buzz_destroyBuzzMatrix(bm);
	
	MPI_Finalize();
}