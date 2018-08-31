#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#include "Buzz_Task_Queue.h"
#include "utils.h"

void Buzz_createBuzzTaskQueue(Buzz_Task_Queue_t *_btq, MPI_Comm comm)
{
	Buzz_Task_Queue_t btq = (Buzz_Task_Queue_t) malloc(sizeof(struct Buzz_Task_Queue));
	assert(btq != NULL);

	MPI_Comm_size(comm, &btq->comm_size);
	MPI_Comm_rank(comm, &btq->my_rank);
	MPI_Comm_dup (comm, &btq->mpi_comm);

	// Allocate shared memory and its MPI window
	// (1) Split communicator to get shared memory communicator
	MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, btq->my_rank, MPI_INFO_NULL, &btq->shm_comm);
	MPI_Comm_rank(btq->shm_comm, &btq->shm_rank);
	MPI_Comm_size(btq->shm_comm, &btq->shm_size);
	btq->shm_global_ranks = (int*) malloc(INT_SIZE * btq->shm_size);
	assert(btq->shm_global_ranks != NULL);
	MPI_Allgather(&btq->my_rank, 1, MPI_INT, btq->shm_global_ranks, 1, MPI_INT, btq->shm_comm);
	// (2) Allocate shared memory 
	MPI_Info shm_info;
	MPI_Info_create(&shm_info);
	MPI_Info_set(shm_info, "alloc_shared_noncontig", "true");
	int shm_mem_size = INT_SIZE * btq->shm_size;
	MPI_Win_allocate_shared(
		shm_mem_size, INT_SIZE, shm_info, btq->shm_comm, 
		&btq->task_counter, &btq->shm_win
	);
	MPI_Info_free(&shm_info);
	// (3) Get pointers of all processes in the shared memory communicator
	MPI_Aint _size;
	int _disp;
	btq->shm_ptrs = (void**) malloc(sizeof(void*) * btq->shm_size);
	assert(btq->shm_global_ranks != NULL);
	for (int i = 0; i < btq->shm_size; i++)
		MPI_Win_shared_query(btq->shm_win, i, &_size, &_disp, &btq->shm_ptrs[i]);

	// Bind local task counter to global MPI window
	MPI_Info mpi_info;
	MPI_Info_create(&mpi_info);
	MPI_Win_create(btq->task_counter, INT_SIZE, INT_SIZE, mpi_info, btq->mpi_comm, &btq->mpi_win);
	MPI_Info_free(&mpi_info);

	btq->task_counter[0] = 0;

	*_btq = btq;
}

void Buzz_destroyBuzzTaskQueue(Buzz_Task_Queue_t btq)
{
	free(btq->shm_global_ranks);
	free(btq->shm_ptrs);

	MPI_Win_free(&btq->mpi_win);
	MPI_Win_free(&btq->shm_win);
	MPI_Comm_free(&btq->mpi_comm);
	MPI_Comm_free(&btq->shm_comm);

	free(btq);
}

void Buzz_resetBuzzTaskQueue(Buzz_Task_Queue_t btq)
{
	btq->task_counter[0] = 0;
}

int Buzz_getNextTasks(Buzz_Task_Queue_t btq, int dst_rank, int ntasks)
{
	int shm_rank = getElementIndexInArray(dst_rank, btq->shm_global_ranks, btq->shm_size);
	int *counter = (shm_rank == -1) ? NULL : btq->shm_ptrs[shm_rank];
	int ret;

	if (shm_rank != -1)  
	{
		// Counter is on shared memory, use built-in atomic operation
		ret = __sync_fetch_and_add(counter, ntasks);
	} else {
		// Counter is on remote node, use MPI_Fetch_and_op
		MPI_Win_lock(MPI_LOCK_SHARED, dst_rank, 0, btq->mpi_win);
		MPI_Fetch_and_op(&ntasks, &ret, MPI_INT, dst_rank, 0, MPI_SUM, btq->mpi_win);
		MPI_Win_unlock(dst_rank, btq->mpi_win);
	}

	return ret;
}
