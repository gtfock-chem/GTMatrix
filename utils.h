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

// Get the intersection of segment [s0, e0] and [s1, e1]
void getSegmentIntersection(int s0, int e0, int s1, int e1, int *intersection, int *is, int *ie);

// Get the intersection of rectangle [xs0:xe0, ys0:ye0] and [xs1:xe1, ys1:ye1]
void getRectIntersection(
	int xs0, int xe0, int ys0, int ye0,
	int xs1, int xe1, int ys1, int ye1,
	int *intersection,
	int *ixs, int *ixe, int *iys, int *iye
);

// Get the (first) index of an integer element in an array, returning -1 means no such element
int getElementIndexInArray(const int elem, const int *array, const int array_size);

#endif
