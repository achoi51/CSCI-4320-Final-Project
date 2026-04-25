#include "utils.h"
#include "mpi_comm.h"
#include "io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const char *solver_name(solver_kind_t s)
{
    switch (s) {
        case SOLVER_LANCZOS: return "lanczos";
        case SOLVER_ARNOLDI: return "arnoldi";
        case SOLVER_POWER:   return "power";
    }
    return "?";
}

static const char *matrix_name(matrix_kind_t m)
{
    switch (m) {
        case MATRIX_DENSE_SYM:    return "dense-sym";
        case MATRIX_LAPLACIAN_2D: return "laplacian2d";
        case MATRIX_FROM_FILE:    return "file";
    }
    return "?";
}

void parse_options(int argc, char **argv, solver_options_t *opt)
{
    opt->solver = SOLVER_LANCZOS;
    opt->matrix = MATRIX_DENSE_SYM;
    opt->N = 1024;
    opt->num_eigs = 8;
    opt->max_iters = 200;
    opt->tolerance = 1e-9;
    opt->seed = 42UL;
    opt->use_gpu = 1;
    opt->checkpoint_every = 0;
    opt->input_path[0] = '\0';
    snprintf(opt->output_path, sizeof(opt->output_path), "results/eigvals.dat");

    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (!strcmp(a, "--matrix-size") && v)         { opt->N = atoi(v); ++i; }
        else if (!strcmp(a, "--num-eigenvalues") && v){ opt->num_eigs = atoi(v); ++i; }
        else if (!strcmp(a, "--max-iterations") && v) { opt->max_iters = atoi(v); ++i; }
        else if (!strcmp(a, "--tolerance") && v)      { opt->tolerance = atof(v); ++i; }
        else if (!strcmp(a, "--seed") && v)           { opt->seed = strtoul(v, NULL, 10); ++i; }
        else if (!strcmp(a, "--checkpoint-every") && v){ opt->checkpoint_every = atoi(v); ++i; }
        else if (!strcmp(a, "--cpu-only"))            { opt->use_gpu = 0; }
        else if (!strcmp(a, "--solver") && v) {
            if      (!strcmp(v, "lanczos")) opt->solver = SOLVER_LANCZOS;
            else if (!strcmp(v, "arnoldi")) opt->solver = SOLVER_ARNOLDI;
            else if (!strcmp(v, "power"))   opt->solver = SOLVER_POWER;
            ++i;
        }
        else if (!strcmp(a, "--matrix") && v) {
            if      (!strcmp(v, "dense"))       opt->matrix = MATRIX_DENSE_SYM;
            else if (!strcmp(v, "laplacian2d")) opt->matrix = MATRIX_LAPLACIAN_2D;
            else if (!strcmp(v, "file"))        opt->matrix = MATRIX_FROM_FILE;
            ++i;
        }
        else if (!strcmp(a, "--input") && v) {
            strncpy(opt->input_path, v, sizeof(opt->input_path) - 1);
            opt->matrix = MATRIX_FROM_FILE;
            ++i;
        }
        else if (!strcmp(a, "--output") && v) {
            strncpy(opt->output_path, v, sizeof(opt->output_path) - 1);
            ++i;
        }
    }
}

void print_options(const solver_options_t *opt)
{
    printf("=== Eigensolver configuration ===\n");
    printf("  solver       : %s\n", solver_name(opt->solver));
    printf("  matrix       : %s\n", matrix_name(opt->matrix));
    printf("  N            : %d\n", opt->N);
    printf("  num_eigs     : %d\n", opt->num_eigs);
    printf("  max_iters    : %d\n", opt->max_iters);
    printf("  tolerance    : %.2e\n", opt->tolerance);
    printf("  use_gpu      : %s\n", opt->use_gpu ? "yes" : "no");
    if (opt->matrix == MATRIX_FROM_FILE)
        printf("  input        : %s\n", opt->input_path);
    printf("  output       : %s\n", opt->output_path);
    fflush(stdout);
}

void print_stats(const solver_stats_t *s, int rank)
{
    if (rank != 0) return;
    printf("=== Solver stats ===\n");
    printf("  iterations   : %d\n", s->iters);
    printf("  converged    : %s\n", s->converged ? "yes" : "no");
    printf("  t_compute    : %.6f s\n", s->t_compute);
    printf("  t_comm       : %.6f s\n", s->t_comm);
    printf("  t_io         : %.6f s\n", s->t_io);
    printf("  t_total      : %.6f s\n", s->t_total);
    fflush(stdout);
}

double drand_seeded(unsigned long *state)
{
    *state = (*state) * 6364136223846793005UL + 1442695040888963407UL;
    return (double)((*state) >> 11) / (double)(1UL << 53);
}

int matrix_dense_random_symmetric(dist_matrix_t *A, unsigned long seed,
                                  const parallel_ctx_t *ctx)
{
    int N = A->N;
    int rows_per_rank = N / ctx->size;
    int local_offset = ctx->rank * rows_per_rank;
    int local_n = (ctx->rank == ctx->size - 1)
                      ? (N - local_offset)
                      : rows_per_rank;

    A->local_n = local_n;
    A->local_offset = local_offset;
    A->kind = MATRIX_DENSE_SYM;

    double *full = NULL;
    if (ctx->rank == 0) {
        full = (double *)malloc((size_t)N * N * sizeof(double));
        if (!full) return -1;
        unsigned long s = seed;
        for (int i = 0; i < N; ++i) {
            for (int j = i; j < N; ++j) {
                double v = (drand_seeded(&s) - 0.5) * 2.0;
                full[(size_t)i * N + j] = v;
                full[(size_t)j * N + i] = v;
            }
            full[(size_t)i * N + i] += (double)N;
        }
    }

    A->h_dense = (double *)malloc((size_t)local_n * N * sizeof(double));
    if (!A->h_dense) { free(full); return -1; }

    int *counts = (int *)malloc(ctx->size * sizeof(int));
    int *displs = (int *)malloc(ctx->size * sizeof(int));
    for (int r = 0; r < ctx->size; ++r) {
        int off = r * rows_per_rank;
        int rn = (r == ctx->size - 1) ? (N - off) : rows_per_rank;
        counts[r] = rn * N;
        displs[r] = off * N;
    }
    MPI_Scatterv(full, counts, displs, MPI_DOUBLE,
                 A->h_dense, local_n * N, MPI_DOUBLE,
                 0, ctx->comm);
    free(counts); free(displs);
    if (ctx->rank == 0) free(full);

    A->h_row_ptr = NULL;
    A->h_col_idx = NULL;
    A->h_values = NULL;
    A->local_nnz = 0;
    return 0;
}

int matrix_laplacian_2d(dist_matrix_t *A, const parallel_ctx_t *ctx)
{
    int N = A->N;
    int side = (int)floor(sqrt((double)N) + 0.5);
    if (side * side != N) {
        if (ctx->rank == 0)
            fprintf(stderr, "laplacian2d: N must be a perfect square\n");
        return -1;
    }

    int rows_per_rank = N / ctx->size;
    int local_offset = ctx->rank * rows_per_rank;
    int local_n = (ctx->rank == ctx->size - 1)
                      ? (N - local_offset)
                      : rows_per_rank;

    A->local_n = local_n;
    A->local_offset = local_offset;
    A->kind = MATRIX_LAPLACIAN_2D;

    int max_nnz = local_n * 5;
    int *row_ptr = (int *)malloc((local_n + 1) * sizeof(int));
    int *col_idx = (int *)malloc(max_nnz * sizeof(int));
    double *vals = (double *)malloc(max_nnz * sizeof(double));
    if (!row_ptr || !col_idx || !vals) return -1;

    int nz = 0;
    row_ptr[0] = 0;
    for (int li = 0; li < local_n; ++li) {
        int gi = local_offset + li;
        int r = gi / side;
        int c = gi % side;

        col_idx[nz] = gi;     vals[nz] = 4.0; ++nz;
        if (r > 0)        { col_idx[nz] = gi - side; vals[nz] = -1.0; ++nz; }
        if (r < side - 1) { col_idx[nz] = gi + side; vals[nz] = -1.0; ++nz; }
        if (c > 0)        { col_idx[nz] = gi - 1;    vals[nz] = -1.0; ++nz; }
        if (c < side - 1) { col_idx[nz] = gi + 1;    vals[nz] = -1.0; ++nz; }

        row_ptr[li + 1] = nz;
    }

    A->h_row_ptr = row_ptr;
    A->h_col_idx = col_idx;
    A->h_values = vals;
    A->local_nnz = nz;
    A->h_dense = NULL;
    return 0;
}

int matrix_build(const solver_options_t *opt, const parallel_ctx_t *ctx,
                 dist_matrix_t *A)
{
    memset(A, 0, sizeof(*A));
    A->N = opt->N;
    switch (opt->matrix) {
        case MATRIX_DENSE_SYM:
            return matrix_dense_random_symmetric(A, opt->seed, ctx);
        case MATRIX_LAPLACIAN_2D:
            return matrix_laplacian_2d(A, ctx);
        case MATRIX_FROM_FILE:
            return io_read_dense_matrix(opt->input_path, A, ctx, NULL);
    }
    return -1;
}

void matrix_free(dist_matrix_t *A)
{
    free(A->h_dense);
    free(A->h_row_ptr);
    free(A->h_col_idx);
    free(A->h_values);
    memset(A, 0, sizeof(*A));
}

double residual_norm_dense(const dist_matrix_t *A, const double *v_full,
                           double lambda, const parallel_ctx_t *ctx)
{
    int N = A->N;
    int local_n = A->local_n;
    double local_sum = 0.0;
    for (int li = 0; li < local_n; ++li) {
        double y = 0.0;
        const double *row = &A->h_dense[(size_t)li * N];
        for (int j = 0; j < N; ++j) y += row[j] * v_full[j];
        double r = y - lambda * v_full[A->local_offset + li];
        local_sum += r * r;
    }
    double t = 0.0;
    double global_sum = timed_allreduce_sum(local_sum, ctx, &t);
    return sqrt(global_sum);
}
