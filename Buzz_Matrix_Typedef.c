#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#include "Buzz_Matrix_Typedef.h"
#include "Buzz_Req_Vector.h"
#include "utils.h"

void Buzz_createBuzzMatrix(
	Buzz_Matrix_t *Buzz_mat, MPI_Comm comm, MPI_Datatype datatype,
	int unit_size, int my_rank, int nrows, int ncols,
	int r_blocks, int c_blocks, int *r_displs, int *c_displs
)
{
	Buzz_Matrix_t bm = (Buzz_Matrix_t) malloc(sizeof(struct Buzz_Matrix));
	assert(bm != NULL);
	
	// Copy and validate matrix and process info
	int comm_size;
	MPI_Comm_size(comm, &comm_size);
	assert(my_rank < comm_size);
	assert(r_blocks * c_blocks == comm_size);
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
	
	bm->is_batch_updating = 0;
	bm->is_batch_getting  = 0;
	
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
	
	// Allocate local matrix block and symmetrization buffer
	int local_mb_size = bm->my_nrows * bm->ld_local;
	bm->mat_block = malloc(unit_size * local_mb_size);
	bm->symm_buf  = malloc(unit_size * bm->my_nrows * bm->my_ncols);
	assert(bm->mat_block != NULL && bm->symm_buf != NULL);

	// Bind local matrix block to global MPI window
	MPI_Info mpi_info;
	MPI_Info_create(&mpi_info);
	
	MPI_Win_create(bm->mat_block, unit_size * local_mb_size, unit_size, mpi_info, bm->mpi_comm, &bm->mpi_win);
	//bm->ld_blks = (int*) malloc(sizeof(int) * bm->comm_size);
	//assert(bm->ld_blks != NULL);
	//MPI_Allgather(&bm->ld_local, 1, MPI_INT, bm->ld_blks, 1, MPI_INT, bm->mpi_comm);
	MPI_Info_free(&mpi_info);
	
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
	
	// Allocate update request vector
	bm->req_vec = (Buzz_Req_Vector_t*) malloc(bm->comm_size * sizeof(Buzz_Req_Vector_t));
	for (int i = 0; i < bm->comm_size; i++)
		Buzz_createReqVector(&bm->req_vec[i]);

	*Buzz_mat = bm;
}

void Buzz_destroyBuzzMatrix(Buzz_Matrix_t Buzz_mat)
{
	Buzz_Matrix_t bm = Buzz_mat;
	assert(bm != NULL);
	
	MPI_Win_free(&bm->mpi_win);
	MPI_Comm_free(&bm->mpi_comm);
	
	free(bm->r_displs);
	free(bm->r_blklens);
	free(bm->c_displs);
	free(bm->c_blklens);
	free(bm->mat_block);
	//free(bm->ld_blks);
	free(bm->symm_buf);
	
	for (int i = 0; i < MPI_DT_SB_DIM_MAX * MPI_DT_SB_DIM_MAX; i++)
	{
		MPI_Type_free(&bm->sb_stride[i]);
		MPI_Type_free(&bm->sb_nostride[i]);
	}
	free(bm->sb_stride);
	free(bm->sb_nostride);
	
	for (int i = 0; i < bm->comm_size; i++)
		Buzz_destroyReqVector(bm->req_vec[i]);

	free(bm);
}
