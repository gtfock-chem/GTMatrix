#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#include "GTMatrix_Retval.h"
#include "GTMatrix_Typedef.h"
#include "GTM_Req_Vector.h"
#include "utils.h"

int GTM_create(
    GTMatrix_t *_gtm, MPI_Comm comm, MPI_Datatype datatype,
    int unit_size, int my_rank, int nrows, int ncols,
    int r_blocks, int c_blocks, int *r_displs, int *c_displs
)
{
    GTMatrix_t gtm = (GTMatrix_t) malloc(sizeof(struct GTMatrix));
    if (gtm == NULL) return GTM_ALLOC_FAILED;
    
    // Copy and validate matrix and process info
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_dup (comm, &gtm->mpi_comm);
    if ((my_rank < 0) || (my_rank >= comm_size)) return GTM_INVALID_RANK;
    if (r_blocks * c_blocks != comm_size) return GTM_INVALID_RCBLOCK;
    gtm->datatype  = datatype;
    gtm->unit_size = unit_size;
    gtm->my_rank   = my_rank;
    gtm->comm_size = comm_size;
    gtm->nrows     = nrows;
    gtm->ncols     = ncols;
    gtm->r_blocks  = r_blocks;
    gtm->c_blocks  = c_blocks;
    gtm->my_rowblk = my_rank / c_blocks;
    gtm->my_colblk = my_rank % c_blocks;
    
    gtm->in_batch_get = 0;
    gtm->in_batch_put = 0;
    gtm->in_batch_acc = 0;
    
    // Allocate space for displacement arrays
    size_t r_displs_msize = sizeof(int) * (r_blocks + 1);
    size_t c_displs_msize = sizeof(int) * (c_blocks + 1);
    gtm->r_displs  = (int*) malloc(r_displs_msize);
    gtm->c_displs  = (int*) malloc(c_displs_msize);
    gtm->r_blklens = (int*) malloc(r_displs_msize);
    gtm->c_blklens = (int*) malloc(c_displs_msize);
    if ((gtm->r_displs  == NULL) || (gtm->c_displs  == NULL) ||
        (gtm->r_blklens == NULL) || (gtm->c_blklens == NULL))
    {
        return GTM_ALLOC_FAILED;
    }
    memcpy(gtm->r_displs, r_displs, r_displs_msize);
    memcpy(gtm->c_displs, c_displs, c_displs_msize);
    
    // Validate r_displs and c_displs, then generate r_blklens and c_blklens
    int r_displs_valid = 1, c_displs_valid = 1;
    if (r_displs[0] != 0) r_displs_valid = 0;
    if (c_displs[0] != 0) c_displs_valid = 0;
    if (r_displs[r_blocks] != nrows) r_displs_valid = 0;
    if (c_displs[c_blocks] != ncols) c_displs_valid = 0;
    for (int i = 0; i < r_blocks; i++)
    {
        gtm->r_blklens[i] = r_displs[i + 1] - r_displs[i];
        if (gtm->r_blklens[i] <= 0) r_displs_valid = 0;
    }
    for (int i = 0; i < c_blocks; i++)
    {
        gtm->c_blklens[i] = c_displs[i + 1] - c_displs[i];
        if (gtm->c_blklens[i] <= 0) c_displs_valid = 0;
    }
    if (r_displs_valid == 0) return GTM_INVALID_R_DISPLS;
    if (c_displs_valid == 0) return GTM_INVALID_C_DISPLS;
    gtm->my_nrows = gtm->r_blklens[gtm->my_rowblk];
    gtm->my_ncols = gtm->c_blklens[gtm->my_colblk];
    // gtm->ld_local = gtm->my_ncols;
    // Use the same local leading dimension for all processes
    MPI_Allreduce(&gtm->my_ncols, &gtm->ld_local, 1, MPI_INT, MPI_MAX, gtm->mpi_comm);
    size_t symm_buf_msize = (size_t)unit_size * (size_t)gtm->my_nrows * (size_t)gtm->my_ncols;
    gtm->symm_buf = malloc(symm_buf_msize);
    if (gtm->symm_buf == NULL) return GTM_ALLOC_FAILED;
    
    // Allocate shared memory and its MPI window
    // (1) Split communicator to get shared memory communicator
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, my_rank, MPI_INFO_NULL, &gtm->shm_comm);
    MPI_Comm_rank(gtm->shm_comm, &gtm->shm_rank);
    MPI_Comm_size(gtm->shm_comm, &gtm->shm_size);
    gtm->shm_global_ranks = (int*) malloc(sizeof(int) * gtm->shm_size);
    if (gtm->shm_global_ranks == NULL) return GTM_ALLOC_FAILED;
    MPI_Allgather(&gtm->my_rank, 1, MPI_INT, gtm->shm_global_ranks, 1, MPI_INT, gtm->shm_comm);
    // (2) Allocate shared memory 
    int shm_max_nrow, shm_max_ncol;
    MPI_Allreduce(&gtm->my_nrows, &shm_max_nrow, 1, MPI_INT, MPI_MAX, gtm->shm_comm);
    MPI_Allreduce(&gtm->ld_local, &shm_max_ncol, 1, MPI_INT, MPI_MAX, gtm->shm_comm);
    MPI_Aint shm_msize = (MPI_Aint)shm_max_ncol * (MPI_Aint)shm_max_nrow * (MPI_Aint)gtm->shm_size * (MPI_Aint)unit_size;
    MPI_Info shm_info;
    MPI_Info_create(&shm_info);
    MPI_Info_set(shm_info, "alloc_shared_noncontig", "true");
    MPI_Win_allocate_shared(
        shm_msize, unit_size, shm_info, gtm->shm_comm, 
        &gtm->mat_block, &gtm->shm_win
    );
    MPI_Info_free(&shm_info);
    // (3) Get pointers of all processes in the shared memory communicator
    MPI_Aint _size;
    int _disp;
    gtm->shm_mat_blocks = (void**) malloc(sizeof(void*) * gtm->shm_size);
    if (gtm->shm_mat_blocks == NULL) return GTM_ALLOC_FAILED;
    for (int i = 0; i < gtm->shm_size; i++)
        MPI_Win_shared_query(gtm->shm_win, i, &_size, &_disp, &gtm->shm_mat_blocks[i]);

    // Bind local matrix block to global MPI window
    MPI_Info mpi_info;
    MPI_Info_create(&mpi_info);
    MPI_Aint my_block_msize = (MPI_Aint)gtm->my_nrows * (MPI_Aint)shm_max_ncol * (MPI_Aint)unit_size;
    MPI_Win_create(gtm->mat_block, my_block_msize, unit_size, mpi_info, gtm->mpi_comm, &gtm->mpi_win);
    //gtm->ld_blks = (int*) malloc(sizeof(int) * gtm->comm_size);
    //assert(gtm->ld_blks != NULL);
    //MPI_Allgather(&gtm->ld_local, 1, MPI_INT, gtm->ld_blks, 1, MPI_INT, gtm->mpi_comm);
    MPI_Info_free(&mpi_info);
    
    // Define small block data types
    size_t DDTs_msize = sizeof(MPI_Datatype) * MPI_DT_SB_DIM_MAX * MPI_DT_SB_DIM_MAX;
    gtm->sb_stride   = (MPI_Datatype*) malloc(DDTs_msize);
    gtm->sb_nostride = (MPI_Datatype*) malloc(DDTs_msize);
    if (gtm->sb_stride   == NULL) return GTM_ALLOC_FAILED;
    if (gtm->sb_nostride == NULL) return GTM_ALLOC_FAILED;
    for (int irow = 0; irow < MPI_DT_SB_DIM_MAX; irow++)
    {
        for (int icol = 0; icol < MPI_DT_SB_DIM_MAX; icol++)
        {
            int id = irow * MPI_DT_SB_DIM_MAX + icol;
            if (irow == 0 && icol == 0) 
            {
                // Single element, use the original data type
                MPI_Type_dup(datatype, &gtm->sb_stride[id]);
                MPI_Type_dup(datatype, &gtm->sb_nostride[id]);
            } else {
                if (irow == 0)
                {
                    // Only one row, use contiguous type
                    MPI_Type_contiguous(icol + 1, datatype, &gtm->sb_stride[id]);
                    MPI_Type_contiguous(icol + 1, datatype, &gtm->sb_nostride[id]);
                } else {
                    // More than 1 row, use vector type
                    MPI_Type_vector(irow + 1, icol + 1, gtm->ld_local, datatype, &gtm->sb_stride[id]);
                    MPI_Type_vector(irow + 1, icol + 1,     icol + 1, datatype, &gtm->sb_nostride[id]);
                }
            }
            MPI_Type_commit(&gtm->sb_stride[id]);
            MPI_Type_commit(&gtm->sb_nostride[id]);
        }
    }
    
    // Allocate update request vector
    gtm->req_vec = (GTM_Req_Vector_t*) malloc(gtm->comm_size * sizeof(GTM_Req_Vector_t));
    if (gtm->req_vec == NULL) return GTM_ALLOC_FAILED;
    for (int i = 0; i < gtm->comm_size; i++)
        GTM_createReqVector(&gtm->req_vec[i]);
    
    // Set up nonblocking access threshold
    gtm->nb_op_proc_cnt = (int*) malloc(gtm->comm_size * sizeof(int));
    if (gtm->nb_op_proc_cnt == NULL) return GTM_ALLOC_FAILED;;
    memset(gtm->nb_op_proc_cnt, 0, gtm->comm_size * sizeof(int));
    gtm->nb_op_cnt  = 0;
    gtm->max_nb_acc = 8;
    gtm->max_nb_get = 128;
    char *max_nb_acc_p = getenv("GTM_MAX_NB_READ");
    char *max_nb_get_p = getenv("GTM_MAX_NB_UPDATE");
    if (max_nb_acc_p != NULL) gtm->max_nb_acc = atoi(max_nb_acc_p);
    if (max_nb_get_p != NULL) gtm->max_nb_get = atoi(max_nb_get_p);
    if (gtm->max_nb_acc <    4) gtm->max_nb_acc =    4;
    if (gtm->max_nb_acc > 1024) gtm->max_nb_acc = 1024;
    if (gtm->max_nb_get <    4) gtm->max_nb_get =    4;
    if (gtm->max_nb_get > 1024) gtm->max_nb_get = 1024;
    
    // Set up MPI window lock type for update 
    // By default: (1) for accumulation, only element-wise atomicity is needed, use 
    // MPI_LOCK_SHARED; (2) for replacement, user should guarantee the write sequence 
    // and handle conflict, still use MPI_LOCK_SHARED. 
    gtm->acc_lock_type = MPI_LOCK_SHARED;
    char *acc_lock_type_p = getenv("GTM_UPDATE_ATOMICITY");
    if (acc_lock_type_p != NULL) 
    {
        gtm->acc_lock_type = atoi(acc_lock_type_p);
        switch (gtm->acc_lock_type)
        {
            case 1:  gtm->acc_lock_type = MPI_LOCK_SHARED;    break; 
            case 2:  gtm->acc_lock_type = MPI_LOCK_EXCLUSIVE; break; 
            default: gtm->acc_lock_type = MPI_LOCK_SHARED;    break; 
        }
    }
    
    *_gtm = gtm;
    return GTM_SUCCESS;
}

int GTM_destroy(GTMatrix_t gtm)
{
    if (gtm == NULL) return GTM_NULL_PTR;
    
    MPI_Win_free(&gtm->mpi_win);
    MPI_Win_free(&gtm->shm_win);        // This will also free *mat_block
    MPI_Comm_free(&gtm->mpi_comm);
    MPI_Comm_free(&gtm->shm_comm);
    
    free(gtm->r_displs);
    free(gtm->r_blklens);
    free(gtm->c_displs);
    free(gtm->c_blklens);
    //free(gtm->mat_block);
    //free(gtm->ld_blks);
    free(gtm->symm_buf);
    
    for (int dst_rank = 0; dst_rank < gtm->comm_size; dst_rank++)
    {
        if (gtm->nb_op_proc_cnt[dst_rank] != 0)
        {
            MPI_Win_unlock(dst_rank, gtm->mpi_win);
            gtm->nb_op_proc_cnt[dst_rank] = 0;
        }
    }
    free(gtm->nb_op_proc_cnt);
    
    for (int i = 0; i < MPI_DT_SB_DIM_MAX * MPI_DT_SB_DIM_MAX; i++)
    {
        MPI_Type_free(&gtm->sb_stride[i]);
        MPI_Type_free(&gtm->sb_nostride[i]);
    }
    free(gtm->sb_stride);
    free(gtm->sb_nostride);
    
    for (int i = 0; i < gtm->comm_size; i++)
        GTM_destroyReqVector(gtm->req_vec[i]);

    free(gtm);
    
    return GTM_SUCCESS;
}
