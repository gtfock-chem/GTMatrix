#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#include "GTMatrix_Typedef.h"
#include "GTM_Req_Vector.h"
#include "utils.h"

void GTM_createGTMatrix(
    GTMatrix_t *_gt_mat, MPI_Comm comm, MPI_Datatype datatype,
    int unit_size, int my_rank, int nrows, int ncols,
    int r_blocks, int c_blocks, int *r_displs, int *c_displs
)
{
    GTMatrix_t gt_mat = (GTMatrix_t) malloc(sizeof(struct GTMatrix));
    assert(gt_mat != NULL);
    
    // Copy and validate matrix and process info
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    assert(my_rank < comm_size);
    assert(r_blocks * c_blocks == comm_size);
    MPI_Comm_dup(comm, &gt_mat->mpi_comm);
    gt_mat->datatype  = datatype;
    gt_mat->unit_size = unit_size;
    gt_mat->my_rank   = my_rank;
    gt_mat->comm_size = comm_size;
    gt_mat->nrows     = nrows;
    gt_mat->ncols     = ncols;
    gt_mat->r_blocks  = r_blocks;
    gt_mat->c_blocks  = c_blocks;
    gt_mat->my_rowblk = my_rank / c_blocks;
    gt_mat->my_colblk = my_rank % c_blocks;
    
    gt_mat->is_batch_updating = 0;
    gt_mat->is_batch_getting  = 0;
    
    // Allocate space for displacement arrays
    int r_displs_mem_size = sizeof(int) * (r_blocks + 1);
    int c_displs_mem_size = sizeof(int) * (c_blocks + 1);
    gt_mat->r_displs  = (int*) malloc(r_displs_mem_size);
    gt_mat->c_displs  = (int*) malloc(c_displs_mem_size);
    gt_mat->r_blklens = (int*) malloc(sizeof(int) * r_blocks);
    gt_mat->c_blklens = (int*) malloc(sizeof(int) * c_blocks);
    assert(gt_mat->r_displs  != NULL && gt_mat->c_displs  != NULL);
    assert(gt_mat->r_blklens != NULL && gt_mat->c_blklens != NULL);
    memcpy(gt_mat->r_displs, r_displs, r_displs_mem_size);
    memcpy(gt_mat->c_displs, c_displs, c_displs_mem_size);
    
    // Validate r_displs and c_displs, then generate r_blklens and c_blklens
    int r_displs_valid = 1, c_displs_valid = 1;
    if (r_displs[0] != 0) r_displs_valid = 0;
    if (c_displs[0] != 0) c_displs_valid = 0;
    if (r_displs[r_blocks] != nrows) r_displs_valid = 0;
    if (c_displs[c_blocks] != ncols) c_displs_valid = 0;
    for (int i = 0; i < r_blocks; i++)
    {
        gt_mat->r_blklens[i] = r_displs[i + 1] - r_displs[i];
        if (gt_mat->r_blklens[i] <= 0) r_displs_valid = 0;
    }
    for (int i = 0; i < c_blocks; i++)
    {
        gt_mat->c_blklens[i] = c_displs[i + 1] - c_displs[i];
        if (gt_mat->c_blklens[i] <= 0) c_displs_valid = 0;
    }
    if (r_displs_valid == 0 && my_rank == 0)
    {
        printf("[FATAL] GTMatrix: Invalid r_displs!\n");
        assert(r_displs_valid == 1);
    }
    if (c_displs_valid == 0 && my_rank == 0)
    {
        printf("[FATAL] GTMatrix: Invalid c_displs!\n");
        assert(c_displs_valid == 1);
    }
    gt_mat->my_nrows = gt_mat->r_blklens[gt_mat->my_rowblk];
    gt_mat->my_ncols = gt_mat->c_blklens[gt_mat->my_colblk];
    // gt_mat->ld_local = gt_mat->my_ncols;
    // Use the same local leading dimension for all processes
    MPI_Allreduce(&gt_mat->my_ncols, &gt_mat->ld_local, 1, MPI_INT, MPI_MAX, gt_mat->mpi_comm);
    
    gt_mat->symm_buf = malloc(unit_size * gt_mat->my_nrows * gt_mat->my_ncols);
    assert(gt_mat->symm_buf != NULL);
    
    // Allocate shared memory and its MPI window
    // (1) Split communicator to get shared memory communicator
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, my_rank, MPI_INFO_NULL, &gt_mat->shm_comm);
    MPI_Comm_rank(gt_mat->shm_comm, &gt_mat->shm_rank);
    MPI_Comm_size(gt_mat->shm_comm, &gt_mat->shm_size);
    gt_mat->shm_global_ranks = (int*) malloc(sizeof(int) * gt_mat->shm_size);
    assert(gt_mat->shm_global_ranks != NULL);
    MPI_Allgather(&gt_mat->my_rank, 1, MPI_INT, gt_mat->shm_global_ranks, 1, MPI_INT, gt_mat->shm_comm);
    // (2) Allocate shared memory 
    int shm_max_nrow, shm_max_ncol, shm_mb_bytes;
    MPI_Allreduce(&gt_mat->my_nrows, &shm_max_nrow, 1, MPI_INT, MPI_MAX, gt_mat->shm_comm);
    MPI_Allreduce(&gt_mat->ld_local, &shm_max_ncol, 1, MPI_INT, MPI_MAX, gt_mat->shm_comm);
    shm_mb_bytes  = shm_max_ncol * shm_max_nrow * gt_mat->shm_size * unit_size;
    MPI_Info shm_info;
    MPI_Info_create(&shm_info);
    MPI_Info_set(shm_info, "alloc_shared_noncontig", "true");
    MPI_Win_allocate_shared(
        shm_mb_bytes, unit_size, shm_info, gt_mat->shm_comm, 
        &gt_mat->mat_block, &gt_mat->shm_win
    );
    MPI_Info_free(&shm_info);
    // (3) Get pointers of all processes in the shared memory communicator
    MPI_Aint _size;
    int _disp;
    gt_mat->shm_mat_blocks = (void**) malloc(sizeof(void*) * gt_mat->shm_size);
    assert(gt_mat->shm_mat_blocks != NULL);
    for (int i = 0; i < gt_mat->shm_size; i++)
        MPI_Win_shared_query(gt_mat->shm_win, i, &_size, &_disp, &gt_mat->shm_mat_blocks[i]);

    // Bind local matrix block to global MPI window
    MPI_Info mpi_info;
    MPI_Info_create(&mpi_info);
    int my_block_size = gt_mat->my_nrows * shm_max_ncol;
    MPI_Win_create(gt_mat->mat_block, my_block_size * unit_size, unit_size, mpi_info, gt_mat->mpi_comm, &gt_mat->mpi_win);
    //gt_mat->ld_blks = (int*) malloc(sizeof(int) * gt_mat->comm_size);
    //assert(gt_mat->ld_blks != NULL);
    //MPI_Allgather(&gt_mat->ld_local, 1, MPI_INT, gt_mat->ld_blks, 1, MPI_INT, gt_mat->mpi_comm);
    MPI_Info_free(&mpi_info);
    
    // Define small block data types
    gt_mat->sb_stride   = (MPI_Datatype*) malloc(sizeof(MPI_Datatype) * MPI_DT_SB_DIM_MAX * MPI_DT_SB_DIM_MAX);
    gt_mat->sb_nostride = (MPI_Datatype*) malloc(sizeof(MPI_Datatype) * MPI_DT_SB_DIM_MAX * MPI_DT_SB_DIM_MAX);
    assert(gt_mat->sb_stride != NULL && gt_mat->sb_nostride != NULL);
    for (int irow = 0; irow < MPI_DT_SB_DIM_MAX; irow++)
    {
        for (int icol = 0; icol < MPI_DT_SB_DIM_MAX; icol++)
        {
            int id = irow * MPI_DT_SB_DIM_MAX + icol;
            if (irow == 0 && icol == 0) 
            {
                // Single element, use the original data type
                MPI_Type_dup(datatype, &gt_mat->sb_stride[id]);
                MPI_Type_dup(datatype, &gt_mat->sb_nostride[id]);
            } else {
                if (irow == 0)
                {
                    // Only one row, use contiguous type
                    MPI_Type_contiguous(icol + 1, datatype, &gt_mat->sb_stride[id]);
                    MPI_Type_contiguous(icol + 1, datatype, &gt_mat->sb_nostride[id]);
                } else {
                    // More than 1 row, use vector type
                    MPI_Type_vector(irow + 1, icol + 1, gt_mat->ld_local, datatype, &gt_mat->sb_stride[id]);
                    MPI_Type_vector(irow + 1, icol + 1,     icol + 1, datatype, &gt_mat->sb_nostride[id]);
                }
            }
            MPI_Type_commit(&gt_mat->sb_stride[id]);
            MPI_Type_commit(&gt_mat->sb_nostride[id]);
        }
    }
    
    // Allocate update request vector
    gt_mat->req_vec = (GTM_Req_Vector_t*) malloc(gt_mat->comm_size * sizeof(GTM_Req_Vector_t));
    for (int i = 0; i < gt_mat->comm_size; i++)
        GTM_createReqVector(&gt_mat->req_vec[i]);

    *_gt_mat = gt_mat;
}

void GTM_destroyGTMatrix(GTMatrix_t gt_mat)
{
    assert(gt_mat != NULL);
    
    MPI_Win_free(&gt_mat->mpi_win);
    MPI_Win_free(&gt_mat->shm_win);        // This will also free *mat_block
    MPI_Comm_free(&gt_mat->mpi_comm);
    MPI_Comm_free(&gt_mat->shm_comm);
    
    free(gt_mat->r_displs);
    free(gt_mat->r_blklens);
    free(gt_mat->c_displs);
    free(gt_mat->c_blklens);
    //free(gt_mat->mat_block);
    //free(gt_mat->ld_blks);
    free(gt_mat->symm_buf);
    
    for (int i = 0; i < MPI_DT_SB_DIM_MAX * MPI_DT_SB_DIM_MAX; i++)
    {
        MPI_Type_free(&gt_mat->sb_stride[i]);
        MPI_Type_free(&gt_mat->sb_nostride[i]);
    }
    free(gt_mat->sb_stride);
    free(gt_mat->sb_nostride);
    
    for (int i = 0; i < gt_mat->comm_size; i++)
        GTM_destroyReqVector(gt_mat->req_vec[i]);

    free(gt_mat);
}
