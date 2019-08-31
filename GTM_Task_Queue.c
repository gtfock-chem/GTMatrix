#define _POSIX_C_SOURCE 200112L  // For posix_memalign

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#include "GTMatrix_Retval.h"
#include "GTM_Task_Queue.h"
#include "utils.h"

int GTM_createTaskQueue(GTM_Task_Queue_t *_gtm_tq, MPI_Comm comm)
{
    GTM_Task_Queue_t gtm_tq = (GTM_Task_Queue_t) malloc(sizeof(struct GTM_Task_Queue));
    if (gtm_tq == NULL) return GTM_TQ_ALLOC_FAILED;

    MPI_Comm_size(comm, &gtm_tq->comm_size);
    MPI_Comm_rank(comm, &gtm_tq->my_rank);
    MPI_Comm_dup (comm, &gtm_tq->mpi_comm);

    // Align the counter to 64-byte address
    gtm_tq->task_counter = NULL;
    posix_memalign((void**)&gtm_tq->task_counter, 64, INT_SIZE * 32);
    if (gtm_tq->task_counter == NULL) return GTM_TQ_ALLOC_FAILED;
    
    MPI_Info mpi_info;
    MPI_Info_create(&mpi_info);
    MPI_Win_create(gtm_tq->task_counter, INT_SIZE, INT_SIZE, mpi_info, gtm_tq->mpi_comm, &gtm_tq->mpi_win);
    MPI_Info_free(&mpi_info);

    gtm_tq->task_counter[0] = 0;

    *_gtm_tq = gtm_tq;
    return GTM_TQ_SUCCESS;
}

int GTM_destroyTaskQueue(GTM_Task_Queue_t gtm_tq)
{
    if (gtm_tq == NULL) return GTM_TQ_NULL_PTR;
    
    free(gtm_tq->task_counter);
    MPI_Win_free(&gtm_tq->mpi_win);
    MPI_Comm_free(&gtm_tq->mpi_comm);
    free(gtm_tq);
    return GTM_TQ_SUCCESS;
}

int GTM_resetTaskQueue(GTM_Task_Queue_t gtm_tq)
{
    if (gtm_tq == NULL) return GTM_TQ_NULL_PTR;
    gtm_tq->task_counter[0] = 0;
    return GTM_TQ_SUCCESS;
}

int GTM_getNextTasks(GTM_Task_Queue_t gtm_tq, int dst_rank, int ntasks)
{
    if (gtm_tq == NULL) return GTM_TQ_NULL_PTR;
    if ((dst_rank < 0) || (dst_rank >= gtm_tq->comm_size)) return GTM_TQ_INVALID_RANK;

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
