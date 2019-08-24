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
// Input parameter:
//   comm : MPI communicator used in this distributed task queue
// Output paramater:
//   _gtm_tq : Pointer to a initialized GTM_Task_Queue structure
int GTM_createTaskQueue(GTM_Task_Queue_t *_gtm_tq, MPI_Comm comm);

// Free a GTM_Task_Queue structure
// This call is collective, thread-safe
int GTM_destroyTaskQueue(GTM_Task_Queue_t gtm_tq);

// Free counter in a GTM_Task_Queue structure
// This call is collective, thread-safe
int GTM_resetTaskQueue(GTM_Task_Queue_t gtm_tq);

// Read and then increment the task_counter value on the given process
// This call is not collective, thread-safe
// Input parameter:
//   dst_rank : Target process rank
//   ntasks   : Number of tasks to get
// Output paramater:
//   @return : The task_counter value before incremention.
//             If the operation is invalid, the returning value will be negative.
int GTM_getNextTasks(GTM_Task_Queue_t gtm_tq, int dst_rank, int ntasks);

#endif
