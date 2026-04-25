#ifndef EIGENSOLVER_H
#define EIGENSOLVER_H

#include <stdint.h>
#include <stddef.h>
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SOLVER_LANCZOS = 0,
    SOLVER_ARNOLDI = 1,
    SOLVER_POWER   = 2
} solver_kind_t;

typedef enum {
    MATRIX_DENSE_SYM = 0,
    MATRIX_LAPLACIAN_2D = 1,
    MATRIX_FROM_FILE = 2
} matrix_kind_t;

typedef struct {
    int N;
    int local_n;
    int local_offset;
    matrix_kind_t kind;

    double *h_dense;

    int    *h_row_ptr;
    int    *h_col_idx;
    double *h_values;
    int     local_nnz;
} dist_matrix_t;

typedef struct {
    int rank;
    int size;
    MPI_Comm comm;
    int gpu_id;
    int have_gpu;
} parallel_ctx_t;

typedef struct {
    solver_kind_t solver;
    matrix_kind_t matrix;
    int    N;
    int    num_eigs;
    int    max_iters;
    double tolerance;
    unsigned long seed;
    int    use_gpu;
    int    checkpoint_every;
    char   input_path[512];
    char   output_path[512];
} solver_options_t;

typedef struct {
    double t_compute;
    double t_comm;
    double t_io;
    double t_total;
    int    iters;
    int    converged;
} solver_stats_t;

int  parallel_init(int *argc, char ***argv, parallel_ctx_t *ctx, int want_gpu);
void parallel_finalize(parallel_ctx_t *ctx);

int  matrix_build(const solver_options_t *opt, const parallel_ctx_t *ctx,
                  dist_matrix_t *A);
void matrix_free(dist_matrix_t *A);

int  run_lanczos(const dist_matrix_t *A, const parallel_ctx_t *ctx,
                 const solver_options_t *opt,
                 double *eigvals, solver_stats_t *stats);
int  run_arnoldi(const dist_matrix_t *A, const parallel_ctx_t *ctx,
                 const solver_options_t *opt,
                 double *eigvals_re, double *eigvals_im,
                 solver_stats_t *stats);
int  run_power(const dist_matrix_t *A, const parallel_ctx_t *ctx,
               const solver_options_t *opt,
               double *eigval, solver_stats_t *stats);

int tridiag_eigenvalues(const double *alpha, const double *beta,
                        int m, double *eigvals);
int hessenberg_eigenvalues(const double *H, int m,
                           double *eigvals_re, double *eigvals_im);

int io_write_eigenvalues(const char *path, const double *vals, int k,
                         const parallel_ctx_t *ctx);
int io_checkpoint_vector(const char *path, const double *local_v,
                         int local_n, int local_offset,
                         const parallel_ctx_t *ctx);

void allreduce_sum_double(const double *in, double *out, int n,
                          const parallel_ctx_t *ctx, double *t_comm);
void allgather_full_vector(const double *local, double *full,
                           int local_n, int N,
                           const parallel_ctx_t *ctx, double *t_comm);

#ifdef __cplusplus
}
#endif

#endif
