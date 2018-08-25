#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include "Buzz_Matrix.h"
#include "utils.h"

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	
	int r_displs[5] = {0, 3, 6, 8, 12};
	int c_displs[5] = {0, 4, 6, 8};
	int mat[96], req_cnt[100];
	
	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	
	MPI_Comm comm_world;
	MPI_Comm_dup(MPI_COMM_WORLD, &comm_world);
	
	Buzz_Matrix_t bm;
	
	// 4 * 3 proc grid, matrix size 10 * 10
	Buzz_createBuzzMatrix(
		&bm, comm_world, MPI_INT, 4, my_rank, 12, 8, 
		4, 3, &r_displs[0], &c_displs[0], 2, 0
	);
	
	// Set local data
	int *mat_block = (int*) bm->mat_block;
	for (int i = 0; i < bm->my_nrows * bm->my_ncols; i++)
		mat_block[i] = 100 * my_rank + i;
	
	// Start to fetch blocks from other processes
	memset(&mat[0], 0, 4 * 96);
	memset(&req_cnt[0], 0, 4 * 100);
	Buzz_startBuzzMatrixReadOnlyEpoch(bm);
	if (my_rank == 0)
	{
		Buzz_getBlock(bm, &req_cnt[0], 0, 12, 0, 8, &mat[0], 8);
		Buzz_flushProcListGetRequests(bm, &req_cnt[0]);
		
		print_int_mat(&mat[0], 8, 12, 8, "Full matrix");
	}
	Buzz_stopBuzzMatrixReadOnlyEpoch(bm);
	
	// Update matrix data
	for (int i = 0; i < bm->my_nrows * bm->my_ncols; i++)
		mat_block[i] += 114;
	// Start to fetch blocks from other processes again
	memset(&mat[0], 0, 4 * 96);
	int rs_[4] = {1, 2, 3, 10};
	int rn_[4] = {1, 1, 3, 2};
	int cs_[4] = {1, 1, 1, 1};
	int cn_[4] = {5, 5, 5, 7};
	Buzz_startBuzzMatrixReadOnlyEpoch(bm);
	if (my_rank == 0)
	{
		//Buzz_getBlock(bm, &req_cnt[0], 1, 5, 1, 5, &mat[9], 8);
		//Buzz_getBlock(bm, &req_cnt[0], 10, 2, 1, 7, &mat[81], 8);
		//Buzz_flushProcListGetRequests(bm, &req_cnt[0]);
		
		#pragma omp parallel num_threads(2)
		{
			int tid = omp_get_thread_num();
			int *thread_rcv_buf;
			int *rs = &rs_[tid * 2];
			int *rn = &rn_[tid * 2];
			int *cs = &cs_[tid * 2];
			int *cn = &cn_[tid * 2];
			Buzz_getBlockList(bm, 2, tid, rs, rn, cs, cn, (void**) &thread_rcv_buf);
			Buzz_completeGetBlocks(bm, tid);
			
			for (int i = 0; i < 2; i++)
			{
				int *dst = &mat[rs[i] * 8 + cs[i]];
				copy_int_matrix_block(dst, 8, thread_rcv_buf, cn[i], rn[i], cn[i]);
				thread_rcv_buf += rn[i] * cn[i];
			}
		}
		
		print_int_mat(&mat[0], 8, 12, 8, "Recv matrix");
	}
	Buzz_stopBuzzMatrixReadOnlyEpoch(bm);
	
	Buzz_destroyBuzzMatrix(bm);
	
	MPI_Finalize();
}