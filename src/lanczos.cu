#include "solver_internal.h"
#include "mpi_comm.h"
#include "clockcycle.h"
#include "utils.h"
#include "io.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

int run_lanczos(const dist_matrix_t *A, const parallel_ctx_t *ctx,
                const solver_options_t *opt,
                double *eigvals, solver_stats_t *stats)
{
    int N = A->N;
    int local_n = A->local_n;
    int max_iters = opt->max_iters;
    int k = opt->num_eigs;
    if (max_iters > N) max_iters = N;

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

    double **V = (double **)malloc((max_iters + 1) * sizeof(double *));
    for (int i = 0; i <= max_iters; ++i)
        V[i] = (double *)calloc(local_n, sizeof(double));

    double *alpha = (double *)calloc(max_iters, sizeof(double));
    double *beta  = (double *)calloc(max_iters, sizeof(double));
    double *w_local = (double *)calloc(local_n, sizeof(double));
    double *x_full  = (double *)calloc(N, sizeof(double));

    unsigned long s = opt->seed + (unsigned long)ctx->rank * 7919UL + 1;
    for (int i = 0; i < local_n; ++i)
        V[0][i] = drand_seeded(&s) - 0.5;

    double n0 = parallel_norm(ws, V[0], local_n, ctx, &stats->t_comm);
    if (n0 == 0.0) n0 = 1.0;
    for (int i = 0; i < local_n; ++i) V[0][i] /= n0;

    double *prev_eigs = (double *)calloc(k, sizeof(double));
    double *cur_eigs  = (double *)calloc(max_iters, sizeof(double));
    int converged = 0;
    int j = 0;

    for (j = 0; j < max_iters; ++j) {
        distributed_matvec(A, ws, V[j], x_full, w_local, ctx,
                           &stats->t_compute, &stats->t_comm);

        uint64_t tc0 = clock_now();
        double local_a = gpu_dot_local(ws, V[j], w_local, local_n);
        uint64_t tc1 = clock_now();
        stats->t_compute += clock_seconds(tc1 - tc0);

        alpha[j] = timed_allreduce_sum(local_a, ctx, &stats->t_comm);

        gpu_axpy(ws, -alpha[j], V[j], w_local, local_n);
        if (j > 0) gpu_axpy(ws, -beta[j - 1], V[j - 1], w_local, local_n);

        modified_gram_schmidt(ws, V, j, local_n, w_local, ctx, &stats->t_comm);

        beta[j] = parallel_norm(ws, w_local, local_n, ctx, &stats->t_comm);

        if (beta[j] < 1e-14) {
            if (ctx->rank == 0)
                fprintf(stderr, "lanczos: invariant subspace at j=%d\n", j);
            ++j;
            break;
        }

        for (int i = 0; i < local_n; ++i)
            V[j + 1][i] = w_local[i] / beta[j];

        if (j + 1 >= k) {
            int rc = tridiag_eigenvalues(alpha, beta, j + 1, cur_eigs);
            if (rc == 0 && j + 1 > k) {
                double max_diff = 0.0;
                for (int i = 0; i < k; ++i) {
                    double d = fabs(cur_eigs[j + 1 - k + i] - prev_eigs[i]);
                    if (d > max_diff) max_diff = d;
                }
                if (max_diff < opt->tolerance) {
                    converged = 1;
                    ++j;
                    break;
                }
                memcpy(prev_eigs, cur_eigs + (j + 1 - k),
                       k * sizeof(double));
            }
        }

        if (opt->checkpoint_every > 0 &&
            (j + 1) % opt->checkpoint_every == 0) {
            char path[600];
            snprintf(path, sizeof(path), "checkpoint_lanczos_%05d.dat", j + 1);
            uint64_t ti0 = clock_now();
            io_checkpoint_vector(path, V[j + 1], local_n, A->local_offset, ctx);
            uint64_t ti1 = clock_now();
            stats->t_io += clock_seconds(ti1 - ti0);
        }
    }

    int m = j;
    if (m > 0) {
        tridiag_eigenvalues(alpha, beta, m, cur_eigs);
        int kk = (k < m) ? k : m;
        for (int i = 0; i < kk; ++i)
            eigvals[i] = cur_eigs[m - kk + i];
        for (int i = kk; i < k; ++i) eigvals[i] = 0.0;
    }

    stats->iters = m;
    stats->converged = converged;
    uint64_t t_end = clock_now();
    stats->t_total = clock_seconds(t_end - t_start);

    for (int i = 0; i <= max_iters; ++i) free(V[i]);
    free(V); free(alpha); free(beta); free(w_local); free(x_full);
    free(prev_eigs); free(cur_eigs);
    gpu_free(ws);
    return 0;
}
