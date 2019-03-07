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
Correct output:
Initial matrix:
 10.0000     10.0000     11.0000     11.0000     11.0000     12.0000     12.0000     13.0000     13.0000     13.0000    
 14.0000     14.0000     15.0000     15.0000     15.0000     16.0000     16.0000     17.0000     17.0000     17.0000    
 14.0000     14.0000     15.0000     15.0000     15.0000     16.0000     16.0000     17.0000     17.0000     17.0000    
 14.0000     14.0000     15.0000     15.0000     15.0000     16.0000     16.0000     17.0000     17.0000     17.0000    
 18.0000     18.0000     19.0000     19.0000     19.0000     20.0000     20.0000     21.0000     21.0000     21.0000    
 18.0000     18.0000     19.0000     19.0000     19.0000     20.0000     20.0000     21.0000     21.0000     21.0000    
 22.0000     22.0000     23.0000     23.0000     23.0000     24.0000     24.0000     25.0000     25.0000     25.0000    
 22.0000     22.0000     23.0000     23.0000     23.0000     24.0000     24.0000     25.0000     25.0000     25.0000    
 22.0000     22.0000     23.0000     23.0000     23.0000     24.0000     24.0000     25.0000     25.0000     25.0000    
 22.0000     22.0000     23.0000     23.0000     23.0000     24.0000     24.0000     25.0000     25.0000     25.0000    

Symmetrized matrix:
 10.0000     12.0000     12.5000     12.5000     14.5000     15.0000     17.0000     17.5000     17.5000     17.5000    
 12.0000     14.0000     14.5000     14.5000     16.5000     17.0000     19.0000     19.5000     19.5000     19.5000    
 12.5000     14.5000     15.0000     15.0000     17.0000     17.5000     19.5000     20.0000     20.0000     20.0000    
 12.5000     14.5000     15.0000     15.0000     17.0000     17.5000     19.5000     20.0000     20.0000     20.0000    
 14.5000     16.5000     17.0000     17.0000     19.0000     19.5000     21.5000     22.0000     22.0000     22.0000    
 15.0000     17.0000     17.5000     17.5000     19.5000     20.0000     22.0000     22.5000     22.5000     22.5000    
 17.0000     19.0000     19.5000     19.5000     21.5000     22.0000     24.0000     24.5000     24.5000     24.5000    
 17.5000     19.5000     20.0000     20.0000     22.0000     22.5000     24.5000     25.0000     25.0000     25.0000    
 17.5000     19.5000     20.0000     20.0000     22.0000     22.5000     24.5000     25.0000     25.0000     25.0000    
 17.5000     19.5000     20.0000     20.0000     22.0000     22.5000     24.5000     25.0000     25.0000     25.0000
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
    
    Buzz_Sync(bm);
    
    // Symmetrizing
    Buzz_symmetrizeBuzzMatrix(bm);
    if (my_rank == ACTOR_RANK)
    {
        Buzz_getBlock(bm, 0, 10, 0, 10, &mat[0], 10, 1);
        print_double_mat(&mat[0], 10, 10, 10, "Symmetrized matrix");
    }
    
    Buzz_Sync(bm);
    
    Buzz_destroyBuzzMatrix(bm);
    
    MPI_Finalize();
}