#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include "GTMatrix.h"
#include "utils.h"

#define ACTOR_RANK 4
#define ACCU_RANK  10

/*
Run with: mpirun -np 16 ./test_batch_acc.x
Correct output:
Updated matrix:
 45     45     45     45     45     45     45     45    
 45     45     45     45     45     45     45     45    
 45     45     45     45     45     45     45     45    
 45     45     45     45     45     45     45     45    
 45     45     45     45     45     45     45     45    
 45     45     45     45     45     45     45     45    
 45     45     45     45     45     45     45     45    
 45     45     45     45     45     45     45     45
*/

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    
    int r_displs[5] = {0, 2, 3, 7, 8};
    int c_displs[5] = {0, 2, 4, 6, 8};
    int mat[64];
    
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    MPI_Comm comm_world;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm_world);
    
    GTMatrix_t gtm;
    
    // 4 * 4 proc grid, matrix size 8 * 8
    GTM_create(
        &gtm, comm_world, MPI_INT, 4, my_rank, 8, 8, 
        4, 4, &r_displs[0], &c_displs[0]
    );
    
    int ifill = 0;
    GTM_fill(gtm, &ifill);

    for (int i = 0; i < 64; i++) mat[i] = my_rank;

    GTM_sync(gtm);

    if (my_rank < ACCU_RANK)
    {
        GTM_startBatchAcc(gtm);
        for (int irow = 0; irow < 8; irow++)
            GTM_addAccBlockRequest(gtm, irow, 1, 0, 8, &mat[irow * 8], 8);
        printf("Rank %d add requests done\n", my_rank);
        GTM_execBatchAcc(gtm);
        GTM_stopBatchAcc(gtm);
        
        //for (int irow = 0; irow < 8; irow++)
        //    GTM_accBlock(gtm, irow, 1, 0, 8, &mat[irow * 8], 8);
    }
    
    // Wait all process to finish their update
    GTM_sync(gtm);

    if (my_rank == ACTOR_RANK)
    {
        for (int i = 0; i < 64; i++) mat[i] = -1;
        GTM_getBlock(gtm, 0, 8, 0, 8, &mat[0], 8);
        print_int_mat(&mat[0], 8, 8, 8, "Updated matrix");
    }
    
    GTM_sync(gtm);
    
    GTM_destroy(gtm);
    
    MPI_Finalize();
}
