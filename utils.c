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

void getSegmentIntersection(int s0, int e0, int s1, int e1, int *intersection, int *is, int *ie)
{
    if (s0 > s1)
    {
        int swap;
        swap = s0; s0 = s1; s1 = swap;
        swap = e0; e0 = e1; e1 = swap;
    }
    
    if (s1 > e0)  // No intersection
    {
        *is = -1;
        *ie = -1;
        *intersection = 0;
        return;
    }
    
    *intersection = 1;
    if (s1 <= e0)
    {
        *is = s1;
        if (e0 < e1) *ie = e0; else *ie = e1;
    }
}

void getRectIntersection(
    int xs0, int xe0, int ys0, int ye0,
    int xs1, int xe1, int ys1, int ye1,
    int *intersection,
    int *ixs, int *ixe, int *iys, int *iye
)
{
    getSegmentIntersection(xs0, xe0, xs1, xe1, intersection, ixs, ixe);
    if (*intersection == 0) return;
    getSegmentIntersection(ys0, ye0, ys1, ye1, intersection, iys, iye);
}

int getElementIndexInArray(const int elem, const int *array, const int array_size)
{
    int ret = -1;
    for (int i = 0; i < array_size; i++)
        if (elem == array[i]) 
        {
            ret = i;
            break;
        }
    return ret;
}
