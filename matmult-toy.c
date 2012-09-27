/**
 * Copyright (c) 2011-2012 Los Alamos National Security, LLC.
 *                         All rights reserved.
 *
 * This program was prepared by Los Alamos National Security, LLC at Los Alamos
 * National Laboratory (LANL) under contract No. DE-AC52-06NA25396 with the U.S.
 * Department of Energy (DOE). All rights in the program are reserved by the DOE
 * and Los Alamos National Security, LLC. Permission is granted to the public to
 * copy and use this software without charge, provided that this Notice and any
 * statement of authorship are reproduced on all copies. Neither the U.S.
 * Government nor LANS makes any warranty, express or implied, or assumes any
 * liability or responsibility for the use of this software.
 */

/**
 * @author Samuel K. Gutierrez
 *
 * simple OpenMP [+ MPI] code that does nothing useful.
 * primarily used to test affinity
 */

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#ifdef _HAVE_MPI
#include "mpi.h"
#endif

/* number of matrix rows and columns -- change if need be */
#define MAT_RC_SIZE 1024

#ifdef _HAVE_MPI
#define MPICHK(mpi_ret, gtl) \
do { \
    if ( MPI_SUCCESS != (mpi_ret) ) { \
        fprintf(stderr, "MPI_SUCCESS not returned @ %s : line %d\n", __func__, \
                __LINE__); \
        goto gtl; \
    } \
} while(0)
#endif

#ifdef _HAVE_MPI
#define HN_BUF_LEN MPI_MAX_PROCESSOR_NAME
#else
#define HN_BUF_LEN 128
#endif

static double
get_time(void)
{
#ifdef _HAVE_MPI
    return MPI_Wtime();
#else
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1.0e-6;
#endif
}

/**
 * based on omp_mm.c OpenMP example from computing.llnl.gov
 */
static void
mat_mult(void)
{
    int tid = 0, nt = 0, i = 0, j = 0, k = 0, chunk = 0;
    double  **a = NULL, **b = NULL, **c = NULL;

    if (NULL == (a = (double **)calloc(MAT_RC_SIZE, sizeof(double *)))) {
        printf("out of resources\n");
        goto err;
    }
    if (NULL == (b = (double **)calloc(MAT_RC_SIZE, sizeof(double *)))) {
        printf("out of resources\n");
        goto err;
    }
    if (NULL == (c = (double **)calloc(MAT_RC_SIZE, sizeof(double *)))) {
        printf("out of resources\n");
        goto err;
    }
    for (i = 0; i < MAT_RC_SIZE; ++i) {
        if (NULL == (a[i] = calloc(MAT_RC_SIZE, sizeof(double)))) {
            printf("out of resources\n");
            goto err;
        }
        if (NULL == (b[i] = calloc(MAT_RC_SIZE, sizeof(double)))) {
            printf("out of resources\n");
            goto err;
        }
        if (NULL == (c[i] = calloc(MAT_RC_SIZE, sizeof(double)))) {
            printf("out of resources\n");
            goto err;
        }
    }

    /* set chunk size */
    chunk = 32;

    #pragma omp parallel default(shared) private(tid, i, j, k)
    {
        nt = omp_get_num_threads();
        tid = omp_get_thread_num();
        if (tid == 0) {
            printf("--- initializing matrices... ");
    }

    /* initialize matrices */
    #pragma omp for schedule(static, chunk)
    for (i = 0; i < MAT_RC_SIZE; i++)
        for (j = 0; j < MAT_RC_SIZE; j++)
            a[i][j]= i + j;
    #pragma omp for schedule(static, chunk)
    for (i = 0; i < MAT_RC_SIZE; i++)
        for (j = 0; j < MAT_RC_SIZE; j++)
            b[i][j]= i * j;
    #pragma omp for schedule(static, chunk)
    for (i = 0; i < MAT_RC_SIZE; i++)
        for (j = 0; j < MAT_RC_SIZE; j++)
            c[i][j] = 0;

    if (tid == 0) {
        printf("done!\n");
    }

    /* do matrix multiply sharing iterations on outer loop */
    if (tid == 0) {
        printf("--- starting matrix multiply (a x b) with %d threads... ", nt);
    }

    #pragma omp for schedule(static, chunk)
    for (i=0; i<MAT_RC_SIZE; i++) {
        for(j = 0; j < MAT_RC_SIZE; j++)
            for (k = 0; k < MAT_RC_SIZE; k++)
                c[i][j] += a[i][k] * b[k][j];
    }

    if (tid == 0) {
        printf("done\n");
    }

    }   /* end of parallel region */
    
    /* free up used resources */
    for (i = 0; i < MAT_RC_SIZE; ++i) {
        free(a[i]);
        free(b[i]);
        free(c[i]);
    }
    free(a);
    free(b);
    free(c);

    return;
err:
    exit(EXIT_FAILURE);
}

int
main(int argc, char **argv)
{
#ifdef _HAVE_MPI
    int num_mpi_ranks = 0;
    int mpi_hn_len = 0;
#endif
    int my_mpi_rank = 0;
    char my_hostname[HN_BUF_LEN + 1];
    int openmp_iam = 0, openmp_np = 1;
    double mat_start = 0.0, mat_end = 0.0;

    setbuf(stdout, NULL);
    memset(my_hostname, '\0', sizeof(my_hostname));

    /* init */
#ifdef _HAVE_MPI
    MPICHK(MPI_Init(&argc, &argv), err);
    MPICHK(MPI_Comm_size(MPI_COMM_WORLD, &num_mpi_ranks), err);
    MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &my_mpi_rank), err);
    MPICHK(MPI_Get_processor_name(my_hostname, &mpi_hn_len), err);

    if (0 == my_mpi_rank) {
        printf("=== num mpi ranks: %d\n", num_mpi_ranks);
    }
#else
    gethostname(my_hostname, sizeof(my_hostname) - 1);
#endif

    #pragma omp parallel default(shared) private(openmp_iam, openmp_np)
    {
        openmp_np = omp_get_num_threads();
        openmp_iam = omp_get_thread_num();
        if (0 == openmp_iam) {
#ifdef _HAVE_MPI
            printf("--- num openmp threads: %d on mpi rank %d on %s\n",
                    openmp_np, my_mpi_rank, my_hostname);
#else
            printf("--- num openmp threads: %d on %s\n", openmp_np,
                   my_hostname);
#endif
        }
    }

    if (0 == my_mpi_rank) {
        printf("*** matrix info:\n"
               "    a (%d x %d)\n    b (%d x %d)\n    c (%d x %d)\n",
               MAT_RC_SIZE, MAT_RC_SIZE,
               MAT_RC_SIZE, MAT_RC_SIZE,
               MAT_RC_SIZE, MAT_RC_SIZE);
    }

    mat_start = get_time();
    mat_mult();
    mat_end = get_time();

    printf("--- total execution time for matrix operations: %lf\n",
           mat_end - mat_start);

#ifdef _HAVE_MPI
    MPICHK(MPI_Finalize(), err);
#endif
    return EXIT_SUCCESS;

#ifdef _HAVE_MPI
err:
    return EXIT_FAILURE;
#endif
}
