#ifndef __BUZZ_TASK_QUEUE_H__
#define __BUZZ_TASK_QUEUE_H__

struct Buzz_Task_Queue
{
    MPI_Comm mpi_comm;      // Target communicator
    MPI_Win  mpi_win;       // MPI window for counter
    int *task_counter;      // Task counter
    int my_rank, comm_size; // Rank of this process and number of process in the global communicator
};

typedef struct Buzz_Task_Queue* Buzz_Task_Queue_t;

// Create and initialize a Buzz_Task_Queue structure
// This call is collective, thread-safe
// [in] comm : MPI communicator used in this distributed task queue
void Buzz_createBuzzTaskQueue(Buzz_Task_Queue_t *_btq, MPI_Comm comm);

// Free a Buzz_Task_Queue structure
// This call is collective, thread-safe
void Buzz_destroyBuzzTaskQueue(Buzz_Task_Queue_t btq);

// Free counter in a Buzz_Task_Queue structure
// This call is collective, thread-safe
void Buzz_resetBuzzTaskQueue(Buzz_Task_Queue_t btq);

// Read and then increment the task_counter value on the given process
// This call is not collective, thread-safe
// [in] dst_rank : Target process rank
// [in] ntasks   : Number of tasks to get
// [out ]@return : The task_counter value before incremention
int Buzz_getNextTasks(Buzz_Task_Queue_t btq, int dst_rank, int ntasks);

#endif
