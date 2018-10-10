#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include "Buzz_Matrix.h"
#include "utils.h"

#define ACTOR_RANK 5

/*
Run with: mpirun -np 16 ./test_Symmetrize.x
*/

int main(int argc, char **argv)
{
	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	
	int r_displs[5] = {0, 1, 4, 6, 10};
	int c_displs[5] = {0, 2, 5, 7, 10};
	double mat[100];
	
	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	
	MPI_Comm comm_world;
	MPI_Comm_dup(MPI_COMM_WORLD, &comm_world);
	
	Buzz_Matrix_t bm;
	
	// 4 * 4 proc grid, matrix size 10 * 10
	Buzz_createBuzzMatrix(
		&bm, comm_world, MPI_DOUBLE, 8, my_rank, 10, 10, 
		4, 4, &r_displs[0], &c_displs[0]
	);
	
	// Set local data
	double d_fill = (double) (my_rank + 10);
	Buzz_fillBuzzMatrix(bm, &d_fill);
	
	if (my_rank == ACTOR_RANK)
	{
		Buzz_getBlock(bm, 0, 10, 0, 10, &mat[0], 10, 1);
		print_double_mat(&mat[0], 10, 10, 10, "Initial matrix");
	}
	
	// Bug: without the following barrier, the initial matrix is incorrect;
	// but using the following barrier will lead to dead lock
	//MPI_Barrier(MPI_COMM_WORLD);
	
	// Symmetrizing
	Buzz_symmetrizeBuzzMatrix(bm);
	if (my_rank == ACTOR_RANK)
	{
		Buzz_getBlock(bm, 0, 10, 0, 10, &mat[0], 10, 1);
		print_double_mat(&mat[0], 10, 10, 10, "Symmetrized matrix");
	}
	
	// Bug: we should use a MPI_Barrier below to prevent some processes
	// destroy the Buzz Matrix earlier than the Buzz_getBlock. But adding 
	// them will also lead to dead lock
	//MPI_Barrier(MPI_COMM_WORLD);
	
	Buzz_destroyBuzzMatrix(bm);
	
	MPI_Finalize();
}