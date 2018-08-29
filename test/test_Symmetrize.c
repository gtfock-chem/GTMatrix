#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include "Buzz_Matrix.h"
#include "utils.h"

#define ACTOR_RANK 5

// mpirun -np 12 ./test_Get_Put.x

int main(int argc, char **argv)
{
	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	
	int r_displs[5] = {0, 1, 4, 6, 10};
	int c_displs[5] = {0, 2, 7, 10};
	double mat[100];
	
	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	
	MPI_Comm comm_world;
	MPI_Comm_dup(MPI_COMM_WORLD, &comm_world);
	
	Buzz_Matrix_t bm;
	
	// 4 * 3 proc grid, matrix size 10 * 10
	Buzz_createBuzzMatrix(
		&bm, comm_world, MPI_DOUBLE, 8, my_rank, 10, 10, 
		4, 3, &r_displs[0], &c_displs[0], 2, 0
	);
	
	// Set local data
	double d_fill = (double) my_rank;
	Buzz_fillBuzzMatrix(bm, &d_fill);
	
	Buzz_startBuzzMatrixReadOnlyEpoch(bm);
	if (my_rank == ACTOR_RANK)
	{
		Buzz_getBlock(bm, bm->proc_cnt, 0, 10, 0, 10, &mat[0], 10);
		print_double_mat(&mat[0], 10, 10, 10, "Initial matrix");
	}
	Buzz_stopBuzzMatrixReadOnlyEpoch(bm);
	
	// Symmetrizing
	Buzz_symmetrizeBuzzMatrix(bm);
	Buzz_startBuzzMatrixReadOnlyEpoch(bm);
	if (my_rank == ACTOR_RANK)
	{
		Buzz_getBlock(bm, bm->proc_cnt, 0, 10, 0, 10, &mat[0], 10);
		print_double_mat(&mat[0], 10, 10, 10, "Symmetrized matrix");
	}
	Buzz_stopBuzzMatrixReadOnlyEpoch(bm);
	
	Buzz_destroyBuzzMatrix(bm);
	
	MPI_Finalize();
}