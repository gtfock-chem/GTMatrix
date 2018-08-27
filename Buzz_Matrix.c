#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#include "Buzz_Matrix.h"
#include "utils.h"

void Buzz_createBuzzMatrix(
	Buzz_Matrix_t *Buzz_mat, MPI_Comm comm, MPI_Datatype datatype,
	int unit_size, int my_rank, int nrows, int ncols,
	int r_blocks, int c_blocks, int *r_displs, int *c_displs,
	int nthreads, int buf_size
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
	bm->my_nrows = bm->r_blklens[bm->my_rowblk];
	bm->my_ncols = bm->c_blklens[bm->my_colblk];
	// bm->ld_local = bm->my_ncols;
	// Use the same local leading dimension for all processes
	MPI_Allreduce(&bm->my_ncols, &bm->ld_local, 1, MPI_INT, MPI_MAX, bm->mpi_comm);
	
	// Allocate shared memory and its MPI window
	// (1) Split communicator to get shared memory communicator
	MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, my_rank, MPI_INFO_NULL, &bm->shm_comm);
	MPI_Comm_rank(bm->shm_comm, &bm->shm_rank);
	MPI_Comm_size(bm->shm_comm, &bm->shm_size);
	bm->shm_global_ranks = (int*) malloc(sizeof(int) * bm->shm_size);
	assert(bm->shm_global_ranks != NULL);
	MPI_Allgather(&bm->my_rank, 1, MPI_INT, bm->shm_global_ranks, 1, MPI_INT, bm->shm_comm);
	// (2) Allocate shared memory 
	int shm_max_nrow, shm_max_ncol, shm_mb_bytes;
	MPI_Allreduce(&bm->my_nrows, &shm_max_nrow, 1, MPI_INT, MPI_MAX, bm->shm_comm);
	MPI_Allreduce(&bm->ld_local, &shm_max_ncol, 1, MPI_INT, MPI_MAX, bm->shm_comm);
	shm_mb_bytes  = shm_max_ncol * shm_max_nrow * bm->shm_size * unit_size;
	MPI_Info shm_info;
	MPI_Info_create(&shm_info);
	MPI_Info_set(shm_info, "alloc_shared_noncontig", "true");
	MPI_Win_allocate_shared(
		shm_mb_bytes, unit_size, shm_info, bm->shm_comm, 
		&bm->mat_block, &bm->shm_win
	);
	MPI_Info_free(&shm_info);
	// (3) Get pointers of all processes in the shared memory communicator
	MPI_Aint _size;
	int _disp;
	bm->shm_mat_blocks = (void**) malloc(sizeof(void*) * bm->shm_size);
	assert(bm->shm_global_ranks != NULL);
	for (int i = 0; i < bm->shm_size; i++)
		MPI_Win_shared_query(bm->shm_win, i, &_size, &_disp, &bm->shm_mat_blocks[i]);

	// Bind local matrix block to global MPI window
	MPI_Info mpi_info;
	MPI_Info_create(&mpi_info);
	int my_block_size = bm->my_nrows * shm_max_ncol;
	MPI_Win_create(bm->mat_block, my_block_size * unit_size, unit_size, mpi_info, bm->mpi_comm, &bm->mpi_win);
	bm->ld_blks = (int*) malloc(sizeof(int) * bm->comm_size);
	assert(bm->ld_blks != NULL);
	MPI_Allgather(&bm->ld_local, 1, MPI_INT, bm->ld_blks, 1, MPI_INT, bm->mpi_comm);
	MPI_Info_free(&mpi_info);
	
	// Allocate space for receive buffer
	if (buf_size <= 0) buf_size = DEFAULE_RCV_BUF_SIZE;
	bm->rcvbuf_size  = buf_size;
	bm->recv_buff = (void*) malloc(nthreads * bm->rcvbuf_size);
	bm->proc_cnt  = (int*)  malloc(sizeof(int) * nthreads * bm->comm_size);
	assert(bm->recv_buff != NULL && bm->proc_cnt != NULL);
	memset(bm->proc_cnt, 0, sizeof(int) * comm_size * nthreads);
	
	// Define small block data types
	bm->sb_stride   = (MPI_Datatype*) malloc(sizeof(MPI_Datatype) * MPI_DT_SB_DIM_MAX * MPI_DT_SB_DIM_MAX);
	bm->sb_nostride = (MPI_Datatype*) malloc(sizeof(MPI_Datatype) * MPI_DT_SB_DIM_MAX * MPI_DT_SB_DIM_MAX);
	assert(bm->sb_stride != NULL && bm->sb_nostride != NULL);
	for (int irow = 0; irow < MPI_DT_SB_DIM_MAX; irow++)
	{
		for (int icol = 0; icol < MPI_DT_SB_DIM_MAX; icol++)
		{
			int id = irow * MPI_DT_SB_DIM_MAX + icol;
			if (irow == 0 && icol == 0) 
			{
				// Single element, use the original data type
				MPI_Type_dup(datatype, &bm->sb_stride[id]);
				MPI_Type_dup(datatype, &bm->sb_nostride[id]);
			} else {
				if (irow == 0)
				{
					// Only one row, use contiguous type
					MPI_Type_contiguous(icol + 1, datatype, &bm->sb_stride[id]);
					MPI_Type_contiguous(icol + 1, datatype, &bm->sb_nostride[id]);
				} else {
					// More than 1 row, use vector type
					MPI_Type_vector(irow + 1, icol + 1, bm->ld_local, datatype, &bm->sb_stride[id]);
					MPI_Type_vector(irow + 1, icol + 1,     icol + 1, datatype, &bm->sb_nostride[id]);
				}
			}
			MPI_Type_commit(&bm->sb_stride[id]);
			MPI_Type_commit(&bm->sb_nostride[id]);
		}
	}
	
	*Buzz_mat = bm;
}

void Buzz_destroyBuzzMatrix(Buzz_Matrix_t Buzz_mat)
{
	Buzz_Matrix_t bm = Buzz_mat;
	assert(bm != NULL);
	
	MPI_Win_free(&bm->mpi_win);
	MPI_Win_free(&bm->shm_win);        // This will also free *mat_block
	MPI_Comm_free(&bm->mpi_comm);
	MPI_Comm_free(&bm->shm_comm);
	
	free(bm->r_displs);
	free(bm->r_blklens);
	free(bm->c_displs);
	free(bm->c_blklens);
	free(bm->ld_blks);
	free(bm->recv_buff);
	free(bm->proc_cnt);
	free(bm->shm_global_ranks);
	free(bm->shm_mat_blocks);
	
	for (int i = 0; i < MPI_DT_SB_DIM_MAX * MPI_DT_SB_DIM_MAX; i++)
	{
		MPI_Type_free(&bm->sb_stride[i]);
		MPI_Type_free(&bm->sb_nostride[i]);
	}
	free(bm->sb_stride);
	free(bm->sb_nostride);
	
	free(bm);
}

void Buzz_fillBuzzMatrix(Buzz_Matrix_t Buzz_mat, void *value)
{
	Buzz_Matrix_t bm = Buzz_mat;
	if (bm->unit_size == 4)
	{
		int _value, *ptr;
		memcpy(&_value, value, 4);
		ptr = (int*) bm->mat_block;
		for (int i = 0; i < bm->my_nrows; i++)
			#pragma vector
			for (int j = 0; j < bm->my_ncols; j++) 
				ptr[i * bm->ld_local + j] = _value;
	}
	if (bm->unit_size == 8)
	{
		double _value, *ptr;
		memcpy(&_value, value, 8);
		ptr = (double*) bm->mat_block;
		for (int i = 0; i < bm->my_nrows; i++)
			#pragma vector
			for (int j = 0; j < bm->my_ncols; j++) 
				ptr[i * bm->ld_local + j] = _value;
	}
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

void Buzz_getBlockFromProcess(
	Buzz_Matrix_t Buzz_mat, int dst_rank, 
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld
)
{
	Buzz_Matrix_t bm  = Buzz_mat;
	int row_end       = row_start + row_num;
	int col_end       = col_start + col_num;
	int dst_rowblk    = dst_rank / bm->c_blocks;
	int dst_colblk    = dst_rank % bm->c_blocks;
	int dst_blk_ld    = bm->ld_blks[dst_rank];
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
	
	// Check if the target process is in the shared memory communicator
	int  shm_target = -1;
	void *shm_ptr   = NULL;
	for (int i = 0; i < bm->shm_size; i++)
		if (bm->shm_global_ranks[i] == dst_rank)
		{
			shm_target = i;
			shm_ptr = bm->shm_mat_blocks[i];
			break;
		}
		
	// Use memcpy instead of MPI_Get for shared memory window
	char *src_ptr = (char*) src_buf;
	int row_bytes = col_num * bm->unit_size;
	int dst_pos = (row_start - dst_row_start) * dst_blk_ld;
	dst_pos += col_start - dst_col_start;
	if (shm_target != -1)
	{
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
			} else {
				// A few long rows, use direct get
				for (int irow = 0; irow < row_num; irow++)
				{
					MPI_Get(src_ptr, row_bytes, MPI_BYTE, dst_rank, 
							dst_pos, row_bytes, MPI_BYTE, bm->mpi_win);
					src_ptr += src_ptr_ld;
					dst_pos += dst_blk_ld;
				}
			}
		}
	}
}

void Buzz_getBlock(
	Buzz_Matrix_t Buzz_mat, int *proc_cnt, 
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld
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
			Buzz_getBlockFromProcess(
				bm, dst_rank, blk_r_s, blk_r_num, 
				blk_c_s, blk_c_num, blk_ptr, src_buf_ld
			);
			if (dst_rank != bm->my_rank) proc_cnt[dst_rank] += blk_r_num;
		}
	}
}

void Buzz_flushProcListGetRequests(Buzz_Matrix_t Buzz_mat, int *proc_cnt)
{
	Buzz_Matrix_t bm = Buzz_mat;
	for (int i = 0; i < bm->comm_size; i++)
		if (proc_cnt[i] > 0)
		{
			MPI_Win_flush(i, bm->mpi_win);
			proc_cnt[i] = 0;
		}
}

int Buzz_getBlockList(
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
			bm, proc_cnt, row_start[i], row_num[i],
			col_start[i], col_num[i], (void*) thread_rcv_ptr, col_num[i]
		);
		thread_rcv_ptr += block_bytes;
	}
	if (ret == -1) ret = nblocks; 
	
	return ret;
}

void Buzz_completeGetBlocks(Buzz_Matrix_t Buzz_mat, int tid)
{
	Buzz_Matrix_t bm = Buzz_mat;
	int *proc_cnt = bm->proc_cnt + bm->comm_size * tid;
	Buzz_flushProcListGetRequests(bm, proc_cnt);
}

void Buzz_putBlockToProcess(
	Buzz_Matrix_t Buzz_mat, int dst_rank, 
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld
)
{
	Buzz_Matrix_t bm  = Buzz_mat;
	int row_end       = row_start + row_num;
	int col_end       = col_start + col_num;
	int dst_rowblk    = dst_rank / bm->c_blocks;
	int dst_colblk    = dst_rank % bm->c_blocks;
	int dst_blk_ld    = bm->ld_blks[dst_rank];
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
	
	// Check if the target process is in the shared memory communicator
	int  shm_target = -1;
	void *shm_ptr   = NULL;
	for (int i = 0; i < bm->shm_size; i++)
		if (bm->shm_global_ranks[i] == dst_rank)
		{
			shm_target = i;
			shm_ptr    = bm->shm_mat_blocks[i];
			break;
		}
		
	// Use memcpy instead of MPI_Put for shared memory window
	char *src_ptr = (char*) src_buf;
	int row_bytes = col_num * bm->unit_size;
	int dst_pos = (row_start - dst_row_start) * dst_blk_ld;
	dst_pos += col_start - dst_col_start;
	MPI_Win_lock(MPI_LOCK_EXCLUSIVE, dst_rank, 0, bm->mpi_win);
	if (shm_target != -1)
	{
		// Target process and current process is in same node, use memcpy
		char *dst_ptr  = shm_ptr + dst_pos * bm->unit_size;
		int src_ptr_ld = src_buf_ld * bm->unit_size;
		int dst_ptr_ld = dst_blk_ld * bm->unit_size;
		for (int irow = 0; irow < row_num; irow++)
		{
			memcpy(dst_ptr, src_ptr, row_bytes);
			src_ptr += src_ptr_ld;
			dst_ptr += dst_ptr_ld;
		}
	} else {
		// Target process and current process isn't in same node, use MPI_Put
		int src_ptr_ld = src_buf_ld * bm->unit_size;
		if (row_num <= MPI_DT_SB_DIM_MAX && col_num <= MPI_DT_SB_DIM_MAX)  
		{
			// Block is small, use predefined data type or define a new 
			// data type to reduce MPI_Put overhead
			int block_dt_id = (row_num - 1) * MPI_DT_SB_DIM_MAX + (col_num - 1);
			MPI_Datatype *dst_dt = &bm->sb_stride[block_dt_id];
			if (col_num == src_buf_ld)
			{
				MPI_Datatype *rcv_dt_ns = &bm->sb_nostride[block_dt_id];
				MPI_Put(src_ptr, 1, *rcv_dt_ns, dst_rank, dst_pos, 1, *dst_dt, bm->mpi_win);
			} else {
				if (col_num == bm->ld_local)
				{
					MPI_Put(src_ptr, 1, *dst_dt, dst_rank, dst_pos, 1, *dst_dt, bm->mpi_win);
				} else {
					MPI_Datatype rcv_dt;
					MPI_Type_vector(row_num, col_num, src_buf_ld, bm->datatype, &rcv_dt);
					MPI_Type_commit(&rcv_dt);
					MPI_Put(src_ptr, 1, rcv_dt, dst_rank, dst_pos, 1, *dst_dt, bm->mpi_win);
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
				MPI_Put(src_ptr, 1, rcv_dt, dst_rank, dst_pos, 1, dst_dt, bm->mpi_win);
				MPI_Type_free(&dst_dt);
				MPI_Type_free(&rcv_dt);
			} else {
				// A few long rows, use direct put
				for (int irow = 0; irow < row_num; irow++)
				{
					MPI_Put(src_ptr, row_bytes, MPI_BYTE, dst_rank, 
							dst_pos, row_bytes, MPI_BYTE, bm->mpi_win);
					src_ptr += src_ptr_ld;
					dst_pos += dst_blk_ld;
				}
			}
		}
	}
	MPI_Win_unlock(dst_rank, bm->mpi_win);
}

void Buzz_putBlock(
	Buzz_Matrix_t Buzz_mat,
	int row_start, int row_num,
	int col_start, int col_num,
	void *src_buf, int src_buf_ld
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
			Buzz_putBlockToProcess(
				bm, dst_rank, blk_r_s, blk_r_num, 
				blk_c_s, blk_c_num, blk_ptr, src_buf_ld
			);
		}
	}
}
