#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>

#include "utils.h"

double get_wtime_sec()
{
    double sec;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    sec = tv.tv_sec + (double) tv.tv_usec / 1000000.0;
    return sec;
}

void copy_int_matrix_block(
	int *dst, const int ldd, int *src, const int lds, 
	const int nrows, const int ncols
)
{
	for (int irow = 0; irow < nrows; irow++)
		memcpy(dst + irow * ldd, src + irow * lds, INT_SIZE * ncols);
} 

void copy_double_matrix_block(
	double *dst, const int ldd, double *src, const int lds, 
	const int nrows, const int ncols
)
{
	for (int irow = 0; irow < nrows; irow++)
		memcpy(dst + irow * ldd, src + irow * lds, DBL_SIZE * ncols);
} 

void print_int_mat(int *mat, const int ldm, const int nrows, const int ncols, const char *mat_name)
{
	printf("%s:\n", mat_name);
	for (int i = 0; i < nrows; i++)
	{
		for (int j = 0; j < ncols; j++) 
		{
			int idx = i * ldm + j;
			int x = mat[idx];
			if (x >= 0.0) printf(" ");
			printf("%d\t", x);
		}
		printf("\n");
	}
	printf("\n");
}

void print_double_mat(double *mat, const int ldm, const int nrows, const int ncols, const char *mat_name)
{
	printf("%s:\n", mat_name);
	for (int i = 0; i < nrows; i++)
	{
		for (int j = 0; j < ncols; j++) 
		{
			int idx = i * ldm + j;
			double x = mat[idx];
			if (x >= 0.0) printf(" ");
			printf("%.4lf\t", x);
		}
		printf("\n");
	}
	printf("\n");
}
