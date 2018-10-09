#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#include "Buzz_Matrix.h"
#include "utils.h"

void Buzz_updateBlockToProcess(
	Buzz_Matrix_t Buzz_mat, int dst_rank, MPI_Op op, 
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
	
	// Sanity check
	if ((row_start < dst_row_start) ||
	    (col_start < dst_col_start) ||
	    (row_end   > dst_row_end)   ||
	    (col_end   > dst_col_end)   ||
		(row_num   * col_num == 0)) return;
	
	char *src_ptr = (char*) src_buf;
	int row_bytes = col_num * bm->unit_size;
	int dst_pos = (row_start - dst_row_start) * dst_blk_ld;
	dst_pos += col_start - dst_col_start;

	// For accumulation, only element-wise atomicity is needed, use MPI_LOCK_SHARED
	// For replacement, user should guarantee the write sequence and handle conflict,
	// still use MPI_LOCK_SHARED
	if (dst_locked == 0)
		MPI_Win_lock(MPI_LOCK_SHARED, dst_rank, 0, bm->mpi_win);
	
	int src_ptr_ld = src_buf_ld * bm->unit_size;
	if (row_num <= MPI_DT_SB_DIM_MAX && col_num <= MPI_DT_SB_DIM_MAX)  
	{
		// Block is small, use predefined data type or define a new 
		// data type to reduce MPI_Accumulate overhead
		int block_dt_id = (row_num - 1) * MPI_DT_SB_DIM_MAX + (col_num - 1);
		MPI_Datatype *dst_dt = &bm->sb_stride[block_dt_id];
		if (col_num == src_buf_ld)
		{
			MPI_Datatype *rcv_dt_ns = &bm->sb_nostride[block_dt_id];
			MPI_Accumulate(src_ptr, 1, *rcv_dt_ns, dst_rank, dst_pos, 1, *dst_dt, op, bm->mpi_win);
		} else {
			if (bm->ld_local == src_buf_ld)
			{
				MPI_Accumulate(src_ptr, 1, *dst_dt, dst_rank, dst_pos, 1, *dst_dt, op, bm->mpi_win);
			} else {
				MPI_Datatype rcv_dt;
				MPI_Type_vector(row_num, col_num, src_buf_ld, bm->datatype, &rcv_dt);
				MPI_Type_commit(&rcv_dt);
				MPI_Accumulate(src_ptr, 1, rcv_dt, dst_rank, dst_pos, 1, *dst_dt, op, bm->mpi_win);
				MPI_Type_free(&rcv_dt);
			}
		}
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
			MPI_Accumulate(src_ptr, 1, rcv_dt, dst_rank, dst_pos, 1, dst_dt, op, bm->mpi_win);
			MPI_Type_free(&dst_dt);
			MPI_Type_free(&rcv_dt);
		} else {
			// A few long rows, use direct put
			for (int irow = 0; irow < row_num; irow++)
			{
				MPI_Accumulate(
					src_ptr, col_num, bm->datatype, dst_rank, 
					dst_pos, col_num, bm->datatype, op, bm->mpi_win
				);
				src_ptr += src_ptr_ld;
				dst_pos += dst_blk_ld;
			}
		}
	}
	
	if (dst_locked == 0)
		MPI_Win_unlock(dst_rank, bm->mpi_win);
}

void Buzz_putBlockToProcess(
	Buzz_Matrix_t Buzz_mat, int dst_rank,
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld,
	int dst_locked
)
{
	Buzz_updateBlockToProcess(
		Buzz_mat, dst_rank, MPI_REPLACE,
		row_start, row_num,
		col_start, col_num,
		src_buf, src_buf_ld, dst_locked
	);
}

void Buzz_accumulateBlockToProcess(
	Buzz_Matrix_t Buzz_mat, int dst_rank,
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld,
	int dst_locked
)
{
	Buzz_updateBlockToProcess(
		Buzz_mat, dst_rank, MPI_SUM,
		row_start, row_num,
		col_start, col_num,
		src_buf, src_buf_ld, dst_locked
	);
}

void Buzz_updateBlock(
	Buzz_Matrix_t Buzz_mat, MPI_Op op, 
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld,
	int blocking
)
{
	Buzz_Matrix_t bm = Buzz_mat;
	
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
	
	// Update data to each process
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
			
			// If it is not a blocking call, then it is from batch updating
			// epoch, just put the request in request queues, otherwise
			// execute the update
			if (blocking == 0)
			{
				Buzz_pushToReqVector(
					req_vec, op, blk_r_s, blk_r_num, 
					blk_c_s, blk_c_num, blk_ptr, src_buf_ld
				);
			} else {
				Buzz_updateBlockToProcess(
					bm, dst_rank, op, blk_r_s, blk_r_num, 
					blk_c_s, blk_c_num, blk_ptr, src_buf_ld, 0
				);
			}
		}
	}
}

void Buzz_putBlock(
	Buzz_Matrix_t Buzz_mat,
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld
)
{
	Buzz_updateBlock(
		Buzz_mat, MPI_REPLACE, 
		row_start, row_num,
		col_start, col_num,
		src_buf, src_buf_ld, 1
	);
}

void Buzz_accumulateBlock(
	Buzz_Matrix_t Buzz_mat,
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld
)
{
	Buzz_updateBlock(
		Buzz_mat, MPI_SUM, 
		row_start, row_num,
		col_start, col_num,
		src_buf, src_buf_ld, 1
	);
}

void Buzz_addPutBlockRequest(
	Buzz_Matrix_t Buzz_mat,
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld
)
{
	Buzz_updateBlock(
		Buzz_mat, MPI_REPLACE, 
		row_start, row_num,
		col_start, col_num,
		src_buf, src_buf_ld, 0
	);
}

void Buzz_addAccumulateBlockRequest(
	Buzz_Matrix_t Buzz_mat,
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld
)
{
	Buzz_updateBlock(
		Buzz_mat, MPI_SUM, 
		row_start, row_num,
		col_start, col_num,
		src_buf, src_buf_ld, 0
	);
}

void Buzz_startBatchUpdate(Buzz_Matrix_t Buzz_mat)
{
	Buzz_Matrix_t bm = Buzz_mat;
	if (bm->is_batch_getting) return;
	
	MPI_Win_lock_all(0, bm->mpi_win);
	for (int i = 0; i < bm->comm_size; i++)
		Buzz_resetReqVector(bm->req_vec[i]);
	bm->is_batch_updating = 1;
}

void Buzz_execBatchUpdate(Buzz_Matrix_t Buzz_mat)
{
	Buzz_Matrix_t bm = Buzz_mat;
	if (bm->is_batch_updating == 0) return;
	
	for (int _dst_rank = bm->my_rank; _dst_rank < bm->comm_size + bm->my_rank; _dst_rank++)
	{	
		int dst_rank = _dst_rank % bm->comm_size;
		Buzz_Req_Vector_t req_vec = bm->req_vec[dst_rank];
		
		if (req_vec->curr_size > 0) 
		{
			for (int i = 0; i < req_vec->curr_size; i++)
			{
				MPI_Op op      = req_vec->ops[i];
				int blk_r_s    = req_vec->row_starts[i];
				int blk_r_num  = req_vec->row_nums[i];
				int blk_c_s    = req_vec->col_starts[i];
				int blk_c_num  = req_vec->col_nums[i];
				void *blk_ptr  = req_vec->src_bufs[i];
				int src_buf_ld = req_vec->src_buf_lds[i];
				Buzz_updateBlockToProcess(
					bm, dst_rank, op, blk_r_s, blk_r_num, 
					blk_c_s, blk_c_num, blk_ptr, src_buf_ld, 1
				);
			}
		}
		
		Buzz_resetReqVector(req_vec);
	}
}

void Buzz_stopBatchUpdate(Buzz_Matrix_t Buzz_mat)
{
	Buzz_Matrix_t bm = Buzz_mat;
	if (bm->is_batch_updating == 0) return;
	
	MPI_Win_unlock_all(bm->mpi_win);
	bm->is_batch_updating = 0;
}
