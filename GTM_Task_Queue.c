#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <x86intrin.h>

#include "GTM_Task_Queue.h"
#include "utils.h"

void GTM_createGTMTaskQueue(GTM_Task_Queue_t *_gtm_tq, MPI_Comm comm)
{
    GTM_Task_Queue_t gtm_tq = (GTM_Task_Queue_t) malloc(sizeof(struct GTM_Task_Queue));
    assert(gtm_tq != NULL);

    MPI_Comm_size(comm, &gtm_tq->comm_size);
    MPI_Comm_rank(comm, &gtm_tq->my_rank);
    MPI_Comm_dup (comm, &gtm_tq->mpi_comm);

    // Align the counter to 64-byte address
    gtm_tq->task_counter = (int*) _mm_malloc(INT_SIZE * 32, 64);
    assert(gtm_tq->task_counter != NULL);
    
    MPI_Info mpi_info;
    MPI_Info_create(&mpi_info);
    MPI_Win_create(gtm_tq->task_counter, INT_SIZE, INT_SIZE, mpi_info, gtm_tq->mpi_comm, &gtm_tq->mpi_win);
    MPI_Info_free(&mpi_info);

    gtm_tq->task_counter[0] = 0;

    *_gtm_tq = gtm_tq;
}

void GTM_destroyGTMTaskQueue(GTM_Task_Queue_t gtm_tq)
{
    _mm_free(gtm_tq->task_counter);

    MPI_Win_free(&gtm_tq->mpi_win);
    MPI_Comm_free(&gtm_tq->mpi_comm);

    free(gtm_tq);
}

void GTM_resetGTMTaskQueue(GTM_Task_Queue_t gtm_tq)
{
    gtm_tq->task_counter[0] = 0;
}

int GTM_getNextTasks(GTM_Task_Queue_t gtm_tq, int dst_rank, int ntasks)
{
    int ret;

    if (dst_rank == gtm_tq->my_rank)
    {
        // Self counter, use built-in atomic operation
        ret = __sync_fetch_and_add(gtm_tq->task_counter, ntasks);
    } else {
        // Counter is on remote process, use MPI_Fetch_and_op
        MPI_Win_lock(MPI_LOCK_SHARED, dst_rank, 0, gtm_tq->mpi_win);
        MPI_Fetch_and_op(&ntasks, &ret, MPI_INT, dst_rank, 0, MPI_SUM, gtm_tq->mpi_win);
        MPI_Win_unlock(dst_rank, gtm_tq->mpi_win);
    }

    return ret;
}
