#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#include "Buzz_Matrix.h"

void Buzz_createBuzzMatrix(
	Buzz_Matrix_t *Buzz_mat, MPI_Comm comm, MPI_Datatype datatype,
	int unit_size, int my_rank, int nrows, int ncols,
	int r_blocks, int c_blocks, int *r_displs, int *c_displs,
	void *mat_block, int ld_local, int nthreads, int buf_size
)
{
	Buzz_Matrix_t bm = (Buzz_Matrix_t) malloc(sizeof(struct Buzz_Matrix));
	assert(bm != NULL);
	
	// Copy and validate matrix and process info
	int comm_size;
	MPI_Comm_size(comm, &comm_size);
	assert(my_rank < comm_size);
	assert(r_blocks * c_blocks == comm_size);
	assert(nthreads >= 1);
	MPI_Comm_dup(comm, &bm->mpi_comm);
	bm->datatype  = datatype;
	bm->unit_size = unit_size;
	bm->my_rank   = my_rank;
	bm->comm_size = comm_size;
	bm->nrows     = nrows;
	bm->ncols     = ncols;
	bm->r_blocks  = r_blocks;
	bm->c_blocks  = c_blocks;
	bm->my_rowblk = my_rank / c_blocks;
	bm->my_colblk = my_rank % c_blocks;
	
	// Allocate space for displacement arrays
	int r_displs_mem_size = sizeof(int) * (r_blocks + 1);
	int c_displs_mem_size = sizeof(int) * (c_blocks + 1);
	bm->r_displs  = (int*) malloc(r_displs_mem_size);
	bm->c_displs  = (int*) malloc(c_displs_mem_size);
	bm->r_blklens = (int*) malloc(sizeof(int) * r_blocks);
	bm->c_blklens = (int*) malloc(sizeof(int) * c_blocks);
	assert(bm->r_displs  != NULL && bm->c_displs  != NULL);
	assert(bm->r_blklens != NULL && bm->c_blklens != NULL);
	memcpy(bm->r_displs, r_displs, r_displs_mem_size);
	memcpy(bm->c_displs, c_displs, c_displs_mem_size);
	
	// Validate r_displs and c_displs, then generate r_blklens and c_blklens
	int r_displs_valid = 1, c_displs_valid = 1;
	if (r_displs[0] != 0) r_displs_valid = 0;
	if (c_displs[0] != 0) c_displs_valid = 0;
	if (r_displs[r_blocks] != nrows) r_displs_valid = 0;
	if (c_displs[c_blocks] != ncols) c_displs_valid = 0;
	for (int i = 0; i < r_blocks; i++)
	{
		bm->r_blklens[i] = r_displs[i + 1] - r_displs[i];
		if (bm->r_blklens[i] <= 0) r_displs_valid = 0;
	}
	for (int i = 0; i < c_blocks; i++)
	{
		bm->c_blklens[i] = c_displs[i + 1] - c_displs[i];
		if (bm->c_blklens[i] <= 0) c_displs_valid = 0;
	}
	if (r_displs_valid == 0 && my_rank == 0)
	{
		printf("[FATAL] Buzz_Matrix: Invalid r_displs!\n");
		assert(r_displs_valid == 1);
	}
	if (c_displs_valid == 0 && my_rank == 0)
	{
		printf("[FATAL] Buzz_Matrix: Invalid c_displs!\n");
		assert(c_displs_valid == 1);
	}
	
	// Bind or allocate local matrix block to MPI window
	int mb_size;
	bm->my_nrows = bm->r_blklens[bm->my_rowblk];
	bm->my_ncols = bm->c_blklens[bm->my_colblk];
	bm->ld_blks  = (int*) malloc(sizeof(int) * bm->comm_size);
	assert(bm->ld_blks != NULL);
	if (ld_local > 0)
	{
		mb_size = unit_size * ((bm->my_nrows - 1) * ld_local + bm->my_ncols);
		bm->mat_block = mat_block;
		bm->ld_local  = ld_local;
		bm->mb_alloc  = 0;
	} else {
		mb_size = unit_size * bm->my_nrows * bm->my_ncols;
		
		bm->mat_block = (void*) malloc(mb_size);
		assert(bm->mat_block != NULL);
		
		bm->ld_local = bm->c_blklens[bm->my_colblk];
		bm->mb_alloc = 1;
	}
	MPI_Info_create(&bm->mpi_info);
	MPI_Win_create(bm->mat_block, mb_size, bm->unit_size, bm->mpi_info, bm->mpi_comm, &bm->mpi_win);
	MPI_Allgather(&bm->ld_local, 1, MPI_INT, bm->ld_blks, 1, MPI_INT, bm->mpi_comm);
	
	// Allocate space for receive buffer
	if (buf_size <= 0) buf_size = DEFAULE_RCV_BUF_SIZE;
	bm->rcvbuf_size  = buf_size;
	bm->recv_buff    = (void*) malloc(nthreads * bm->rcvbuf_size);
	bm->proc_req_cnt = (int*)  malloc(sizeof(int) * nthreads * bm->comm_size);
	assert(bm->recv_buff != NULL && bm->proc_req_cnt != NULL);
	memset(bm->proc_req_cnt, 0, sizeof(int) * comm_size * nthreads);
	
	*Buzz_mat = bm;
}

void Buzz_destroyBuzzMatrix(Buzz_Matrix_t Buzz_mat)
{
	Buzz_Matrix_t bm = Buzz_mat;
	assert(bm != NULL);
	
	MPI_Win_free(&bm->mpi_win);
	
	free(bm->r_displs);
	free(bm->r_blklens);
	free(bm->c_displs);
	free(bm->c_blklens);
	free(bm->ld_blks);
	free(bm->recv_buff);
	free(bm->proc_req_cnt);
	if (bm->mb_alloc) free(bm->mat_block);
	
	MPI_Comm_free(&bm->mpi_comm);
	bm->unit_size   = 0;
	bm->my_rank     = 0;
	bm->comm_size   = 0;
	bm->nrows       = 0;
	bm->ncols       = 0;
	bm->r_blocks    = 0;
	bm->c_blocks    = 0;
	bm->my_rowblk   = 0;
	bm->my_colblk   = 0;
	bm->rcvbuf_size = 0;
	bm->ld_local    = 0;
	bm->mb_alloc    = 0;
	
	free(bm);
}

void Buzz_startBuzzMatrixReadOnlyEpoch(Buzz_Matrix_t Buzz_mat)
{
	MPI_Barrier(Buzz_mat->mpi_comm);
	MPI_Win_lock_all(0, Buzz_mat->mpi_win);
}

void Buzz_stopBuzzMatrixReadOnlyEpoch(Buzz_Matrix_t Buzz_mat)
{
	MPI_Win_unlock_all(Buzz_mat->mpi_win);
	MPI_Barrier(Buzz_mat->mpi_comm);
}

void Buzz_getBlockFromProcess(
	Buzz_Matrix_t Buzz_mat, int target_proc, 
	int req_row_start, int req_row_num,
	int req_col_start, int req_col_num,
	void *req_rcv_buf, int req_rcv_buf_ld
)
{
	Buzz_Matrix_t bm  = Buzz_mat;
	int target_rowblk = target_proc / bm->c_blocks;
	int target_colblk = target_proc % bm->c_blocks;
	int target_blk_ld = bm->ld_blks[target_proc];
	int target_row_start = bm->r_displs[target_rowblk];
	int target_col_start = bm->c_displs[target_colblk];
	int req_row_end    = req_row_start + req_row_num;
	int req_col_end    = req_col_start + req_col_num;
	int target_row_end = bm->r_displs[target_rowblk + 1];
	int target_col_end = bm->c_displs[target_colblk + 1];
	
	// Sanity check
	if ((req_row_start < target_row_start) ||
	    (req_col_start < target_col_start) ||
	    (req_row_end   > target_row_end)   ||
	    (req_col_end   > target_col_end)) return;
	
	// Start RMA requests
	char *recv_ptr = (char*) req_rcv_buf;
	int recv_bytes = req_col_num * bm->unit_size;
	int target_pos = (req_row_start - target_row_start) * target_blk_ld;
	target_pos += req_col_start - target_col_start;
	for (int irow = 0; irow < req_row_num; irow++)
	{
		MPI_Get(recv_ptr,   recv_bytes, MPI_BYTE, target_proc, 
		        target_pos, recv_bytes, MPI_BYTE, bm->mpi_win);
		recv_ptr   += req_rcv_buf_ld * bm->unit_size;
		target_pos += target_blk_ld;
	}
}

// Get the intersection of segment [s0, e0] and [s1, e1]
void getSegmentIntersection(int s0, int e0, int s1, int e1, int *intersection, int *is, int *ie)
{
	if (s0 > s1)
	{
		int swap;
		swap = s0; s0 = s1; s1 = swap;
		swap = e0; e0 = e1; e1 = swap;
	}
	
	if (s1 > e0)  // No intersection
	{
		*is = s1;
		*ie = s1 - 1;
		*intersection = 0;
		return;
	}
	
	*intersection = 1;
	if (s1 <= e0)
	{
		*is = s1;
		if (e0 < e1) *ie = e0; else *ie = e1;
	}
}

// Get the intersection of rectangle [xs0:xe0, ys0:ye0] and [xs1:xe1, ys1:ye1]
void getRectIntersection(
	int xs0, int xe0, int ys0, int ye0,
	int xs1, int xe1, int ys1, int ye1,
	int *intersection,
	int *ixs, int *ixe, int *iys, int *iye
)
{
	getSegmentIntersection(xs0, xe0, xs1, xe1, intersection, ixs, ixe);
	if (*intersection == 0) return;
	getSegmentIntersection(ys0, ye0, ys1, ye1, intersection, iys, iye);
}

void Buzz_getBlock(
	Buzz_Matrix_t Buzz_mat, int *proc_req_cnt, 
	int req_row_start, int req_row_num,
	int req_col_start, int req_col_num,
	void *req_rcv_buf, int req_rcv_buf_ld
)
{
	Buzz_Matrix_t bm = Buzz_mat;
	
	// Sanity check
	if ((req_row_start < 0) || (req_col_start < 0) ||
	    (req_row_start + req_row_num > bm->nrows)  ||
	    (req_col_start + req_col_num > bm->ncols)) return;
	
	// Find the processes that contain the requested block
	int s_blk_r, e_blk_r, s_blk_c, e_blk_c;
	int req_row_end = req_row_start + req_row_num - 1;
	int req_col_end = req_col_start + req_col_num - 1;
	for (int i = 0; i < bm->r_blocks; i++)
	{
		if ((bm->r_displs[i] <= req_row_start) && 
		    (req_row_start < bm->r_displs[i+1])) s_blk_r = i;
		if ((bm->r_displs[i] <= req_row_end)   && 
		    (req_row_end   < bm->r_displs[i+1])) e_blk_r = i;
	}
	for (int i = 0; i < bm->c_blocks; i++)
	{
		if ((bm->c_displs[i] <= req_col_start) && 
		    (req_col_start < bm->c_displs[i+1])) s_blk_c = i;
		if ((bm->c_displs[i] <= req_col_end)   && 
		    (req_col_end   < bm->c_displs[i+1])) e_blk_c = i;
	}
	
	// Fetch data from each process
	int blk_r_s, blk_r_e, blk_c_s, blk_c_e, need_to_fetch;
	for (int blk_r = s_blk_r; blk_r <= e_blk_r; blk_r++)      // Notice: <=
	{
		int target_r_s = bm->r_displs[blk_r];
		int target_r_e = bm->r_displs[blk_r + 1] - 1;
		for (int blk_c = s_blk_c; blk_c <= e_blk_c; blk_c++)  // Notice: <=
		{
			int target_c_s  = bm->c_displs[blk_c];
			int target_c_e  = bm->c_displs[blk_c + 1] - 1;
			int target_proc = blk_r * bm->c_blocks + blk_c;
			getRectIntersection(
				target_r_s,    target_r_e,  target_c_s,    target_c_e,
				req_row_start, req_row_end, req_col_start, req_col_end,
				&need_to_fetch, &blk_r_s, &blk_r_e, &blk_c_s, &blk_c_e
			);
			assert(need_to_fetch == 1);
			int blk_r_num = blk_r_e - blk_r_s + 1;
			int blk_c_num = blk_c_e - blk_c_s + 1;
			int row_dist  = blk_r_s - req_row_start;
			int col_dist  = blk_c_s - req_col_start;
			char *blk_ptr = (char*) req_rcv_buf;
			blk_ptr += (row_dist * req_rcv_buf_ld + col_dist) * bm->unit_size;
			Buzz_getBlockFromProcess(
				bm, target_proc, blk_r_s, blk_r_num, 
				blk_c_s, blk_c_num, blk_ptr, req_rcv_buf_ld
			);
			proc_req_cnt[target_proc] += blk_r_num;
		}
	}
}

void Buzz_flushProcListGetRequests(Buzz_Matrix_t Buzz_mat, int *proc_req_cnt)
{
	Buzz_Matrix_t bm = Buzz_mat;
	for (int i = 0; i < bm->comm_size; i++)
		if (proc_req_cnt[i] > 0)
		{
			MPI_Win_flush(i, bm->mpi_win);
			proc_req_cnt[i] = 0;
		}
}

int Buzz_getBlockList(
	Buzz_Matrix_t Buzz_mat, int nblocks, int tid, 
	int *req_row_start, int *req_row_num,
	int *req_col_start, int *req_col_num,
	void **thread_rcv_buf
)
{
	Buzz_Matrix_t bm = Buzz_mat;
	
	if (nblocks <= 0) return 0;
	
	// Set the pointer to this thread's receive buffer
	char *thread_rcv_ptr = (char*) bm->recv_buff;
	thread_rcv_ptr += bm->rcvbuf_size * tid;
	*thread_rcv_buf = thread_rcv_ptr;
	
	// Get each block
	int ret = 0, recv_bytes = 0;
	int *proc_req_cnt = bm->proc_req_cnt + bm->comm_size * tid;
	for (int i = 0; i < nblocks; i++)
	{
		int block_bytes = bm->unit_size * req_row_num[i] * req_col_num[i];
		if (recv_bytes + block_bytes > bm->rcvbuf_size)
		{
			ret = i;
			break;
		}
		
		Buzz_getBlock(
			bm, proc_req_cnt, 
			req_row_start[i], req_row_num[i],
			req_col_start[i], req_col_num[i],
			(void*) thread_rcv_ptr, req_col_num[i]
		);
		thread_rcv_ptr += block_bytes;
	}
	
	return ret;
}

void Buzz_completeGetBlocks(Buzz_Matrix_t Buzz_mat, int tid)
{
	Buzz_Matrix_t bm  = Buzz_mat;
	int *proc_req_cnt = bm->proc_req_cnt + bm->comm_size * tid;
	Buzz_flushProcListGetRequests(bm, proc_req_cnt);
}
