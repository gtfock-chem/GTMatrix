#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#include "Buzz_Matrix_Typedef.h"
#include "Buzz_Matrix_Get.h"

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

void Buzz_symmetrizeBuzzMatrix(Buzz_Matrix_t Buzz_mat)
{
	Buzz_Matrix_t bm = Buzz_mat;
	
	// Sanity check
	if (bm->nrows != bm->ncols) return;
	
	// This process holds [rs:re, cs:ce], need to fetch [cs:ce, rs:re]
	void *rcv_buf = bm->symm_buf;

	int my_row_start = bm->r_displs[bm->my_rowblk];
	int my_col_start = bm->c_displs[bm->my_colblk];
	Buzz_startBatchGet(bm);
    Buzz_addGetBlockRequest(
		bm, 
		my_col_start, bm->my_ncols, 
		my_row_start, bm->my_nrows, 
		rcv_buf, bm->my_nrows
	);
	Buzz_execBatchGet(bm);
	Buzz_stopBatchGet(bm);
	
	if (MPI_INT == bm->datatype)
	{
		int *src_buf = (int*) bm->mat_block;
		int *dst_buf = (int*) rcv_buf;
		for (int irow = 0; irow < bm->my_nrows; irow++)
		{
			for (int icol = 0; icol < bm->my_ncols; icol++)
			{
				int idx_s = irow * bm->ld_local + icol;
				int idx_d = icol * bm->my_nrows + irow;
				src_buf[idx_s] += dst_buf[idx_d];
				src_buf[idx_s] /= 2;
			}
		}
	}
	if (MPI_DOUBLE == bm->datatype)
	{
		double *src_buf = (double*) bm->mat_block;
		double *dst_buf = (double*) rcv_buf;
		for (int irow = 0; irow < bm->my_nrows; irow++)
		{
			for (int icol = 0; icol < bm->my_ncols; icol++)
			{
				int idx_s = irow * bm->ld_local + icol;
				int idx_d = icol * bm->my_nrows + irow;
				src_buf[idx_s] += dst_buf[idx_d];
				src_buf[idx_s] *= 0.5;
			}
		}
	}
}
