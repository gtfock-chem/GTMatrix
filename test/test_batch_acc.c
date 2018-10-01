#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include "Buzz_Matrix.h"
#include "utils.h"

#define ACTOR_RANK 4

/*
Run with: mpirun -np 9 ./test_batch_acc.x
Correct output:
 36	 36	 36	 36	 36	 36	 36	 36	
 36	 36	 36	 36	 36	 36	 36	 36	
 36	 36	 36	 36	 36	 36	 36	 36	
 36	 36	 36	 36	 36	 36	 36	 36	
 36	 36	 36	 36	 36	 36	 36	 36	
 36	 36	 36	 36	 36	 36	 36	 36	
 36	 36	 36	 36	 36	 36	 36	 36	
 36	 36	 36	 36	 36	 36	 36	 36 
*/

int main(int argc, char **argv)
{
	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	
	int r_displs[5] = {0, 3, 7, 8};
	int c_displs[5] = {0, 4, 6, 8};
	int mat[64];
	
	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	
	MPI_Comm comm_world;
	MPI_Comm_dup(MPI_COMM_WORLD, &comm_world);
	
	Buzz_Matrix_t bm;
	
	// 3 * 3 proc grid, matrix size 8 * 8
	Buzz_createBuzzMatrix(
		&bm, comm_world, MPI_INT, 4, my_rank, 8, 8, 
		3, 3, &r_displs[0], &c_displs[0], 2, 0
	);
	
	int ifill = 0;
	Buzz_fillBuzzMatrix(bm, &ifill);

	for (int i = 0; i < 64; i++) mat[i] = my_rank;

	Buzz_startBatchUpdate(bm);
	
	for (int irow = 0; irow < 8; irow++)
		Buzz_addAccumulateBlockRequest(bm, irow, 1, 0, 8, &mat[irow * 8], 8);
	printf("Rank %d add requests done\n", my_rank);
	
	Buzz_execBatchUpdate(bm);
	Buzz_stopBatchUpdate(bm);
	
	// Wait all process to finish their update
	MPI_Barrier(MPI_COMM_WORLD);

	if (my_rank == ACTOR_RANK)
	{
		Buzz_getBlock(bm, bm->proc_cnt, 0, 8, 0, 8, &mat[0], 8, 1, 0);
		print_int_mat(&mat[0], 8, 8, 8, "Updated matrix");
	}
	
	Buzz_destroyBuzzMatrix(bm);
	
	MPI_Finalize();
}