#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>

#include "GTM_Task_Queue.h"

int main(int argc, char **argv)
{
    int nprocs, my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    GTM_Task_Queue_t btq;
    GTM_createGTMTaskQueue(&btq, MPI_COMM_WORLD);

    int accu = 0, task, total;
    int dst_rank = 1;

    MPI_Barrier(MPI_COMM_WORLD);
    
    for (int i = 0; i < 100; i++)
    {
        task = GTM_getNextTasks(btq, dst_rank, 1);
        accu += task;
    }
    
    MPI_Request req;
    MPI_Status  status;
    MPI_Ibarrier(MPI_COMM_WORLD, &req);
    MPI_Wait(&req, &status);

    MPI_Allreduce(&accu, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (my_rank == 0)
    {
        int expected = nprocs * 100;
        expected = expected * (expected - 1) / 2;
        printf("All proc tasks sum = %d, expected value = %d\n", total, expected);
    }

    GTM_destroyGTMTaskQueue(btq);
    MPI_Finalize();
}