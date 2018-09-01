#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#include "Buzz_Matrix.h"
#include "utils.h"

int Buzz_getBlockFromProcess(
	Buzz_Matrix_t Buzz_mat, int dst_rank, 
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld,
	int dst_locked
)
{
	Buzz_Matrix_t bm  = Buzz_mat;
	int row_end       = row_start + row_num;
	int col_end       = col_start + col_num;
	int dst_rowblk    = dst_rank / bm->c_blocks;
	int dst_colblk    = dst_rank % bm->c_blocks;
	int dst_blk_ld    = bm->ld_local; // bm->ld_blks[dst_rank];
	int dst_row_start = bm->r_displs[dst_rowblk];
	int dst_col_start = bm->c_displs[dst_colblk];
	int dst_row_end   = bm->r_displs[dst_rowblk + 1];
	int dst_col_end   = bm->c_displs[dst_colblk + 1];
	int ret = 0;
	
	// Sanity check
	if ((row_start < dst_row_start) ||
	    (col_start < dst_col_start) ||
	    (row_end   > dst_row_end)   ||
	    (col_end   > dst_col_end)   ||
		(row_num   * col_num == 0)) return ret;
	
	// Check if the target process is in the shared memory communicator
	int shm_rank  = getElementIndexInArray(dst_rank, bm->shm_global_ranks, bm->shm_size);
	void *shm_ptr = (shm_rank == -1) ? NULL : bm->shm_mat_blocks[shm_rank];
		
	// Use memcpy instead of MPI_Get for shared memory window
	char *src_ptr = (char*) src_buf;
	int row_bytes = col_num * bm->unit_size;
	int dst_pos = (row_start - dst_row_start) * dst_blk_ld;
	dst_pos += col_start - dst_col_start;
	if (dst_locked == 0)
		MPI_Win_lock(MPI_LOCK_SHARED, dst_rank, 0, bm->mpi_win);
	if (shm_rank != -1)
	{
		if (dst_locked == 0)
			MPI_Win_lock(MPI_LOCK_SHARED, shm_rank, 0, bm->shm_win);
		// Target process and current process is in same node, use memcpy
		char *dst_ptr  = shm_ptr + dst_pos * bm->unit_size;
		int src_ptr_ld = src_buf_ld * bm->unit_size;
		int dst_ptr_ld = dst_blk_ld * bm->unit_size;
		for (int irow = 0; irow < row_num; irow++)
		{
			memcpy(src_ptr, dst_ptr, row_bytes);
			src_ptr += src_ptr_ld;
			dst_ptr += dst_ptr_ld;
		}
		if (dst_locked == 0)
			MPI_Win_unlock(shm_rank, bm->shm_win);
	} else {
		// Target process and current process isn't in same node, use MPI_Get
		int src_ptr_ld = src_buf_ld * bm->unit_size;
		if (row_num <= MPI_DT_SB_DIM_MAX && col_num <= MPI_DT_SB_DIM_MAX)  
		{
			// Block is small, use predefined data type or define a new 
			// data type to reduce MPI_Get overhead
			int block_dt_id = (row_num - 1) * MPI_DT_SB_DIM_MAX + (col_num - 1);
			MPI_Datatype *dst_dt = &bm->sb_stride[block_dt_id];
			if (col_num == src_buf_ld)
			{
				MPI_Datatype *rcv_dt_ns = &bm->sb_nostride[block_dt_id];
				MPI_Get(src_ptr, 1, *rcv_dt_ns, dst_rank, dst_pos, 1, *dst_dt, bm->mpi_win);
			} else {
				if (col_num == bm->ld_local)
				{
					MPI_Get(src_ptr, 1, *dst_dt, dst_rank, dst_pos, 1, *dst_dt, bm->mpi_win);
				} else {
					MPI_Datatype rcv_dt;
					MPI_Type_vector(row_num, col_num, src_buf_ld, bm->datatype, &rcv_dt);
					MPI_Type_commit(&rcv_dt);
					MPI_Get(src_ptr, 1, rcv_dt, dst_rank, dst_pos, 1, *dst_dt, bm->mpi_win);
					MPI_Type_free(&rcv_dt);
				}
			}
			ret = 1;
		} else {   
			// Doesn't has predefined MPI data type
			if (row_num > MPI_DT_SB_DIM_MAX)
			{
				// Many rows, define a MPI data type to reduce number of request
				MPI_Datatype dst_dt, rcv_dt;
				MPI_Type_vector(row_num, col_num, dst_blk_ld, bm->datatype, &dst_dt);
				MPI_Type_vector(row_num, col_num, src_buf_ld, bm->datatype, &rcv_dt);
				MPI_Type_commit(&dst_dt);
				MPI_Type_commit(&rcv_dt);
				MPI_Get(src_ptr, 1, rcv_dt, dst_rank, dst_pos, 1, dst_dt, bm->mpi_win);
				MPI_Type_free(&dst_dt);
				MPI_Type_free(&rcv_dt);
				ret = 1;
			} else {
				// A few long rows, use direct get
				for (int irow = 0; irow < row_num; irow++)
				{
					MPI_Get(src_ptr, row_bytes, MPI_BYTE, dst_rank, 
							dst_pos, row_bytes, MPI_BYTE, bm->mpi_win);
					src_ptr += src_ptr_ld;
					dst_pos += dst_blk_ld;
				}
				ret = row_num;
			}
		}
	}
	if (dst_locked == 0)
	{
		MPI_Win_unlock(dst_rank, bm->mpi_win);
		ret = 0;
	}
	return ret;
}

void Buzz_getBlock(
	Buzz_Matrix_t Buzz_mat, int *proc_cnt, 
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld, 
	int blocking,  int is_mt
)
{
	Buzz_Matrix_t bm = Buzz_mat;
	
	if (blocking && is_mt) return;
	
	// Sanity check
	if ((row_start < 0) || (col_start < 0) ||
	    (row_start + row_num > bm->nrows)  ||
	    (col_start + col_num > bm->ncols)  ||
		(row_num * col_num == 0)) return;
	
	// Find the processes that contain the requested block
	int s_blk_r, e_blk_r, s_blk_c, e_blk_c;
	int row_end = row_start + row_num - 1;
	int col_end = col_start + col_num - 1;
	for (int i = 0; i < bm->r_blocks; i++)
	{
		if ((bm->r_displs[i] <= row_start) && 
		    (row_start < bm->r_displs[i+1])) s_blk_r = i;
		if ((bm->r_displs[i] <= row_end)   && 
		    (row_end   < bm->r_displs[i+1])) e_blk_r = i;
	}
	for (int i = 0; i < bm->c_blocks; i++)
	{
		if ((bm->c_displs[i] <= col_start) && 
		    (col_start < bm->c_displs[i+1])) s_blk_c = i;
		if ((bm->c_displs[i] <= col_end)   && 
		    (col_end   < bm->c_displs[i+1])) e_blk_c = i;
	}
	
	// Fetch data from each process
	int blk_r_s, blk_r_e, blk_c_s, blk_c_e, need_to_fetch;
	for (int blk_r = s_blk_r; blk_r <= e_blk_r; blk_r++)      // Notice: <=
	{
		int dst_r_s = bm->r_displs[blk_r];
		int dst_r_e = bm->r_displs[blk_r + 1] - 1;
		for (int blk_c = s_blk_c; blk_c <= e_blk_c; blk_c++)  // Notice: <=
		{
			int dst_c_s  = bm->c_displs[blk_c];
			int dst_c_e  = bm->c_displs[blk_c + 1] - 1;
			int dst_rank = blk_r * bm->c_blocks + blk_c;
			getRectIntersection(
				dst_r_s,   dst_r_e, dst_c_s,   dst_c_e,
				row_start, row_end, col_start, col_end,
				&need_to_fetch, &blk_r_s, &blk_r_e, &blk_c_s, &blk_c_e
			);
			assert(need_to_fetch == 1);
			int blk_r_num = blk_r_e - blk_r_s + 1;
			int blk_c_num = blk_c_e - blk_c_s + 1;
			int row_dist  = blk_r_s - row_start;
			int col_dist  = blk_c_s - col_start;
			char *blk_ptr = (char*) src_buf;
			blk_ptr += (row_dist * src_buf_ld + col_dist) * bm->unit_size;
			Buzz_Req_Vector_t req_vec = bm->req_vec[dst_rank];
			
			if (is_mt)
			{
				int nGet = Buzz_getBlockFromProcess(
					bm, dst_rank, blk_r_s, blk_r_num, 
					blk_c_s, blk_c_num, blk_ptr, src_buf_ld, 1
				);
				proc_cnt[dst_rank] += nGet;
			} 
			
			if (blocking)
			{
				Buzz_getBlockFromProcess(
					bm, dst_rank, blk_r_s, blk_r_num, 
					blk_c_s, blk_c_num, blk_ptr, src_buf_ld, 0
				);
			} else {
				Buzz_pushToReqVector(
					req_vec, MPI_REPLACE, blk_r_s, blk_r_num, 
					blk_c_s, blk_c_num, blk_ptr, src_buf_ld
				);
			}
		}
	}
}

void Buzz_addGetBlockRequest(
	Buzz_Matrix_t Buzz_mat,
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld
)
{
	Buzz_getBlock(
		Buzz_mat, Buzz_mat->proc_cnt,
		row_start, row_num,
		col_start, col_num,
		src_buf, src_buf_ld, 
		0, 0
	);
}


void Buzz_startBuzzMatrixReadOnlyEpoch(Buzz_Matrix_t Buzz_mat)
{
	MPI_Barrier(Buzz_mat->mpi_comm);
	MPI_Win_lock_all(0, Buzz_mat->mpi_win);
	MPI_Win_lock_all(0, Buzz_mat->shm_win);
}

void Buzz_stopBuzzMatrixReadOnlyEpoch(Buzz_Matrix_t Buzz_mat)
{
	MPI_Win_unlock_all(Buzz_mat->shm_win);
	MPI_Win_unlock_all(Buzz_mat->mpi_win);
	MPI_Barrier(Buzz_mat->mpi_comm);
}


void Buzz_flushProcListGetRequests(Buzz_Matrix_t Buzz_mat, int *proc_cnt)
{
	Buzz_Matrix_t bm = Buzz_mat;
	for (int i = 0; i < bm->comm_size; i++)
	{
		int dst_rank = (i + bm->my_rank) % bm->comm_size;
		if (proc_cnt[dst_rank] > 0)
		{
			MPI_Win_flush(dst_rank, bm->mpi_win);
			proc_cnt[dst_rank] = 0;
		}
	}
}

int Buzz_getBlockList_mt(
	Buzz_Matrix_t Buzz_mat, int nblocks, int tid, 
	int *row_start, int *row_num,
	int *col_start, int *col_num,
	void **thread_src_buf
)
{
	Buzz_Matrix_t bm = Buzz_mat;
	
	if (nblocks <= 0) return 0;
	
	// Set the pointer to this thread's receive buffer
	char *thread_rcv_ptr = (char*) bm->recv_buff;
	thread_rcv_ptr += bm->rcvbuf_size * tid;
	*thread_src_buf = thread_rcv_ptr;
	
	// Get each block
	int ret = -1, recv_bytes = 0;
	int *proc_cnt = bm->proc_cnt + bm->comm_size * tid;
	for (int i = 0; i < nblocks; i++)
	{
		int block_bytes = bm->unit_size * row_num[i] * col_num[i];
		if (recv_bytes + block_bytes > bm->rcvbuf_size)
		{
			ret = i;
			break;
		}
		
		Buzz_getBlock(
			bm, proc_cnt, 
			row_start[i], row_num[i],
			col_start[i], col_num[i], 
			(void*) thread_rcv_ptr, col_num[i], 
			0, 1
		);
		thread_rcv_ptr += block_bytes;
	}
	if (ret == -1) ret = nblocks; 
	
	return ret;
}

void Buzz_completeGetBlocks_mt(Buzz_Matrix_t Buzz_mat, int tid)
{
	Buzz_Matrix_t bm = Buzz_mat;
	int *proc_cnt = bm->proc_cnt + bm->comm_size * tid;
	Buzz_flushProcListGetRequests(bm, proc_cnt);
}

void Buzz_startBatchGet(Buzz_Matrix_t Buzz_mat)
{
	Buzz_Matrix_t bm = Buzz_mat;
	if (bm->is_batch_updating) return;
	bm->is_batch_getting = 1;
	for (int i = 0; i < bm->comm_size; i++)
		Buzz_resetReqVector(bm->req_vec[i]);
}

void Buzz_execBatchGet(Buzz_Matrix_t Buzz_mat)
{
	Buzz_Matrix_t bm = Buzz_mat;
	if (bm->is_batch_getting == 0) return;
	
	for (int _dst_rank = bm->my_rank; _dst_rank < bm->comm_size + bm->my_rank; _dst_rank++)
	{	
		int dst_rank = _dst_rank % bm->comm_size;
		Buzz_Req_Vector_t req_vec = bm->req_vec[dst_rank];
		if (req_vec->curr_size == 0) continue;
		
		int shm_rank = getElementIndexInArray(dst_rank, bm->shm_global_ranks, bm->shm_size);

		for (int i = 0; i < req_vec->curr_size; i++)
		{
			int blk_r_s    = req_vec->row_starts[i];
			int blk_r_num  = req_vec->row_nums[i];
			int blk_c_s    = req_vec->col_starts[i];
			int blk_c_num  = req_vec->col_nums[i];
			void *blk_ptr  = req_vec->src_bufs[i];
			int src_buf_ld = req_vec->src_buf_lds[i];
			int nGet = Buzz_getBlockFromProcess(
				bm, dst_rank, blk_r_s, blk_r_num, 
				blk_c_s, blk_c_num, blk_ptr, src_buf_ld, 1
			);
			bm->proc_cnt[dst_rank] += nGet;
		}
		
		Buzz_resetReqVector(req_vec);
	}
	Buzz_flushProcListGetRequests(bm, bm->proc_cnt);
}

void Buzz_stopBatchGet(Buzz_Matrix_t Buzz_mat)
{
	Buzz_Matrix_t bm = Buzz_mat;
	bm->is_batch_getting = 0;
}
