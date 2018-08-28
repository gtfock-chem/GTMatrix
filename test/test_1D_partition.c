#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include "Buzz_Matrix.h"
#include "utils.h"

#define ACTOR_RANK 3

// mpirun -np 4 ./test_Get_Put.x

int main(int argc, char **argv)
{
	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

	int r_displs[5] = {0, 1, 3, 5, 9};
	int c_displs[3] = {0, 8};
	double mat[8 * 9];

	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	
	MPI_Comm comm_world;
	MPI_Comm_dup(MPI_COMM_WORLD, &comm_world);
	
	Buzz_Matrix_t bm;

	// 4 * 1 proc grid, matrix size 9 * 8
	Buzz_createBuzzMatrix(
		&bm, comm_world, MPI_DOUBLE, 8, my_rank, 9, 8,
		4, 1, &r_displs[0], &c_displs[0], 2, 0
	);

	double d = 10.0 + (double) my_rank;
	Buzz_fillBuzzMatrix(bm, &d);

	Buzz_startBuzzMatrixReadOnlyEpoch(bm);
	if (my_rank == ACTOR_RANK)
		Buzz_getBlock(bm, bm->proc_cnt, 0, 9, 0, 8, &mat[0], 8);
	Buzz_stopBuzzMatrixReadOnlyEpoch(bm);

	if (my_rank == ACTOR_RANK)
		print_double_mat(&mat[0], 8, 9, 8, "Recv matrix");

	Buzz_destroyBuzzMatrix(bm);

	MPI_Finalize();
}