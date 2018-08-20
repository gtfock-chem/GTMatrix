#ifndef __HUANGH223_UTILS_H__
#define __HUANGH223_UTILS_H__

// Helper functions

#define ALIGN64B_MALLOC(x) _mm_malloc((x), 64)
#define ALIGN64B_FREE(x)   _mm_free(x)
#define DBL_SIZE           sizeof(double)
#define INT_SIZE           sizeof(int)

// Get current wall-clock time, similar to omp_get_wtime()
double get_wtime_sec();

// Copy a block of source matrix to the destination matrix
void copy_int_matrix_block(
	int *dst, const int ldd, int *src, const int lds, 
	const int nrows, const int ncols
);

void copy_double_matrix_block(
	double *dst, const int ldd, double *src, const int lds, 
	const int nrows, const int ncols
);

// For debug, print a dense matrix
void print_int_mat(int *mat, const int ldm, const int nrows, const int ncols, const char *mat_name);

void print_double_mat(double *mat, const int ldm, const int nrows, const int ncols, const char *mat_name);

#endif
