#ifndef __GTM_TASK_QUEUE_H__
#define __GTM_TASK_QUEUE_H__

struct GTM_Task_Queue
{
    MPI_Comm mpi_comm;      // Target communicator
    MPI_Win  mpi_win;       // MPI window for counter
    int *task_counter;      // Task counter
    int my_rank, comm_size; // Rank of this process and number of process in the global communicator
};

typedef struct GTM_Task_Queue* GTM_Task_Queue_t;

// Create and initialize a GTM_Task_Queue structure
// This call is collective, thread-safe
// [in] comm : MPI communicator used in this distributed task queue
void GTM_createGTMTaskQueue(GTM_Task_Queue_t *_gtm_tq, MPI_Comm comm);

// Free a GTM_Task_Queue structure
// This call is collective, thread-safe
void GTM_destroyGTMTaskQueue(GTM_Task_Queue_t gtm_tq);

// Free counter in a GTM_Task_Queue structure
// This call is collective, thread-safe
void GTM_resetGTMTaskQueue(GTM_Task_Queue_t gtm_tq);

// Read and then increment the task_counter value on the given process
// This call is not collective, thread-safe
// [in] dst_rank : Target process rank
// [in] ntasks   : Number of tasks to get
// [out ]@return : The task_counter value before incremention
int GTM_getNextTasks(GTM_Task_Queue_t gtm_tq, int dst_rank, int ntasks);

#endif
