#include "solver_internal.h"
#include "mpi_comm.h"
#include "clockcycle.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

int run_arnoldi(const dist_matrix_t *A, const parallel_ctx_t *ctx,
                const solver_options_t *opt,
                double *eigvals_re, double *eigvals_im,
                solver_stats_t *stats)
{
    int N = A->N;
    int local_n = A->local_n;
    int k = opt->num_eigs;
    int m = opt->max_iters;
    if (m > N) m = N;

    memset(stats, 0, sizeof(*stats));
    uint64_t t_start = clock_now();

    gpu_workspace_t *ws = gpu_alloc(local_n, N, A->local_nnz, opt->use_gpu);
    if (gpu_is_active(ws)) {
        if (A->kind == MATRIX_DENSE_SYM || A->kind == MATRIX_FROM_FILE)
            gpu_upload_dense_block(ws, A->h_dense, local_n, N);
        else
            gpu_upload_csr_block(ws, A->h_row_ptr, A->h_col_idx,
                                 A->h_values, local_n, A->local_nnz);
    }

    double **V = (double **)malloc((m + 1) * sizeof(double *));
    for (int i = 0; i <= m; ++i)
        V[i] = (double *)calloc(local_n, sizeof(double));

    double *H = (double *)calloc((size_t)(m + 1) * m, sizeof(double));
    double *w_local = (double *)calloc(local_n, sizeof(double));
    double *x_full  = (double *)calloc(N, sizeof(double));

    unsigned long s = opt->seed + (unsigned long)ctx->rank * 7919UL + 3;
    for (int i = 0; i < local_n; ++i)
        V[0][i] = drand_seeded(&s) - 0.5;
    double n0 = parallel_norm(ws, V[0], local_n, ctx, &stats->t_comm);
    if (n0 == 0.0) n0 = 1.0;
    for (int i = 0; i < local_n; ++i) V[0][i] /= n0;

    int j = 0;
    for (j = 0; j < m; ++j) {
        distributed_matvec(A, ws, V[j], x_full, w_local, ctx,
                           &stats->t_compute, &stats->t_comm);

        for (int i = 0; i <= j; ++i) {
            double local = gpu_dot_local(ws, V[i], w_local, local_n);
            double global = timed_allreduce_sum(local, ctx, &stats->t_comm);
            H[i * m + j] = global;
            gpu_axpy(ws, -global, V[i], w_local, local_n);
        }

        double hjj = parallel_norm(ws, w_local, local_n, ctx, &stats->t_comm);
        H[(j + 1) * m + j] = hjj;
        if (hjj < 1e-14) { ++j; break; }
        for (int i = 0; i < local_n; ++i)
            V[j + 1][i] = w_local[i] / hjj;
    }

    int mm = j;
    if (mm > 0) {
        double *Hsq = (double *)calloc((size_t)mm * mm, sizeof(double));
        for (int r = 0; r < mm; ++r)
            for (int c = 0; c < mm; ++c)
                Hsq[r * mm + c] = H[r * m + c];
        double *wr = (double *)calloc(mm, sizeof(double));
        double *wi = (double *)calloc(mm, sizeof(double));
        hessenberg_eigenvalues(Hsq, mm, wr, wi);

        int *idx = (int *)malloc(mm * sizeof(int));
        for (int i = 0; i < mm; ++i) idx[i] = i;
        for (int i = 0; i < mm - 1; ++i) {
            int best = i;
            double bm = wr[idx[i]] * wr[idx[i]] + wi[idx[i]] * wi[idx[i]];
            for (int p = i + 1; p < mm; ++p) {
                double cm = wr[idx[p]] * wr[idx[p]] + wi[idx[p]] * wi[idx[p]];
                if (cm > bm) { bm = cm; best = p; }
            }
            int t = idx[i]; idx[i] = idx[best]; idx[best] = t;
        }
        int kk = (k < mm) ? k : mm;
        for (int i = 0; i < kk; ++i) {
            eigvals_re[i] = wr[idx[i]];
            eigvals_im[i] = wi[idx[i]];
        }
        for (int i = kk; i < k; ++i) {
            eigvals_re[i] = 0.0; eigvals_im[i] = 0.0;
        }
        free(Hsq); free(wr); free(wi); free(idx);
    }

    stats->iters = mm;
    stats->converged = (mm == m) ? 0 : 1;
    uint64_t t_end = clock_now();
    stats->t_total = clock_seconds(t_end - t_start);

    for (int i = 0; i <= m; ++i) free(V[i]);
    free(V); free(H); free(w_local); free(x_full);
    gpu_free(ws);
    return 0;
}
