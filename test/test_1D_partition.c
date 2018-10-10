#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include "Buzz_Matrix.h"
#include "utils.h"

#define ACTOR_RANK 3

/*
Run with: mpirun -np 16 ./test_1D_partition.x
Correct output:
Recv matrix:
 10.0000	 10.0000	 10.0000	 10.0000	
 11.0000	 11.0000	 11.0000	 11.0000	
 12.0000	 12.0000	 12.0000	 12.0000	
 13.0000	 13.0000	 13.0000	 13.0000	
 13.0000	 13.0000	 13.0000	 13.0000	
 14.0000	 14.0000	 14.0000	 14.0000	
 15.0000	 15.0000	 15.0000	 15.0000	
 15.0000	 15.0000	 15.0000	 15.0000	
 16.0000	 16.0000	 16.0000	 16.0000	
 17.0000	 17.0000	 17.0000	 17.0000	
 18.0000	 18.0000	 18.0000	 18.0000	
 18.0000	 18.0000	 18.0000	 18.0000	
 19.0000	 19.0000	 19.0000	 19.0000	
 20.0000	 20.0000	 20.0000	 20.0000	
 20.0000	 20.0000	 20.0000	 20.0000	
 21.0000	 21.0000	 21.0000	 21.0000	
 22.0000	 22.0000	 22.0000	 22.0000	
 22.0000	 22.0000	 22.0000	 22.0000	
 22.0000	 22.0000	 22.0000	 22.0000	
 23.0000	 23.0000	 23.0000	 23.0000	
 24.0000	 24.0000	 24.0000	 24.0000	
 24.0000	 24.0000	 24.0000	 24.0000	
 25.0000	 25.0000	 25.0000	 25.0000	
 25.0000	 25.0000	 25.0000	 25.0000
*/

int main(int argc, char **argv)
{
	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

	int r_displs[17] = {0, 1, 2, 3, 5, 6, 8, 9, 10, 12, 13, 15, 16, 19, 20, 22, 24};
	int c_displs[2] = {0, 4};
	double mat[24 * 4];

	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	
	MPI_Comm comm_world;
	MPI_Comm_dup(MPI_COMM_WORLD, &comm_world);
	
	Buzz_Matrix_t bm;

	// 16 * 1 proc grid, matrix size 24 * 4
	Buzz_createBuzzMatrix(
		&bm, comm_world, MPI_DOUBLE, 8, my_rank, 24, 4,
		16, 1, &r_displs[0], &c_displs[0]
	);

	double d = 10.0 + (double) my_rank;
	Buzz_fillBuzzMatrix(bm, &d);

	if (my_rank == ACTOR_RANK)
	{
		Buzz_getBlock(bm, 0, 24, 0, 4, &mat[0], 4, 1);
		print_double_mat(&mat[0], 4, 24, 4, "Recv matrix");
	}

	Buzz_Sync(bm);
	
	Buzz_destroyBuzzMatrix(bm);

	MPI_Finalize();
}