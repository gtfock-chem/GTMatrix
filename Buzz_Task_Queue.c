#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <x86intrin.h>

#include "Buzz_Task_Queue.h"
#include "utils.h"

void Buzz_createBuzzTaskQueue(Buzz_Task_Queue_t *_btq, MPI_Comm comm)
{
    Buzz_Task_Queue_t btq = (Buzz_Task_Queue_t) malloc(sizeof(struct Buzz_Task_Queue));
    assert(btq != NULL);

    MPI_Comm_size(comm, &btq->comm_size);
    MPI_Comm_rank(comm, &btq->my_rank);
    MPI_Comm_dup (comm, &btq->mpi_comm);

    // Align the counter to 64-byte address
    btq->task_counter = (int*) _mm_malloc(INT_SIZE * 32, 64);
    assert(btq->task_counter != NULL);
    
    MPI_Info mpi_info;
    MPI_Info_create(&mpi_info);
    MPI_Win_create(btq->task_counter, INT_SIZE, INT_SIZE, mpi_info, btq->mpi_comm, &btq->mpi_win);
    MPI_Info_free(&mpi_info);

    btq->task_counter[0] = 0;

    *_btq = btq;
}

void Buzz_destroyBuzzTaskQueue(Buzz_Task_Queue_t btq)
{
    _mm_free(btq->task_counter);

    MPI_Win_free(&btq->mpi_win);
    MPI_Comm_free(&btq->mpi_comm);

    free(btq);
}

void Buzz_resetBuzzTaskQueue(Buzz_Task_Queue_t btq)
{
    btq->task_counter[0] = 0;
}

int Buzz_getNextTasks(Buzz_Task_Queue_t btq, int dst_rank, int ntasks)
{
    int ret;

    if (dst_rank == btq->my_rank)
    {
        // Self counter, use built-in atomic operation
        ret = __sync_fetch_and_add(btq->task_counter, ntasks);
    } else {
        // Counter is on remote process, use MPI_Fetch_and_op
        MPI_Win_lock(MPI_LOCK_SHARED, dst_rank, 0, btq->mpi_win);
        MPI_Fetch_and_op(&ntasks, &ret, MPI_INT, dst_rank, 0, MPI_SUM, btq->mpi_win);
        MPI_Win_unlock(dst_rank, btq->mpi_win);
    }

    return ret;
}
