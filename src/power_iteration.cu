#include "solver_internal.h"
#include "mpi_comm.h"
#include "clockcycle.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

int run_power(const dist_matrix_t *A, const parallel_ctx_t *ctx,
              const solver_options_t *opt,
              double *eigval, solver_stats_t *stats)
{
    int N = A->N;
    int local_n = A->local_n;

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

    double *v_local = (double *)calloc(local_n, sizeof(double));
    double *w_local = (double *)calloc(local_n, sizeof(double));
    double *x_full  = (double *)calloc(N, sizeof(double));

    unsigned long s = opt->seed + (unsigned long)ctx->rank * 6151UL + 5;
    for (int i = 0; i < local_n; ++i) v_local[i] = drand_seeded(&s) - 0.5;
    double n0 = parallel_norm(ws, v_local, local_n, ctx, &stats->t_comm);
    if (n0 == 0.0) n0 = 1.0;
    for (int i = 0; i < local_n; ++i) v_local[i] /= n0;

    double lambda = 0.0, lambda_prev = 0.0;
    int converged = 0, it;
    for (it = 0; it < opt->max_iters; ++it) {
        distributed_matvec(A, ws, v_local, x_full, w_local, ctx,
                           &stats->t_compute, &stats->t_comm);

        double local_d = gpu_dot_local(ws, v_local, w_local, local_n);
        lambda = timed_allreduce_sum(local_d, ctx, &stats->t_comm);

        double nrm = parallel_norm(ws, w_local, local_n, ctx, &stats->t_comm);
        if (nrm == 0.0) nrm = 1.0;
        for (int i = 0; i < local_n; ++i) v_local[i] = w_local[i] / nrm;

        if (it > 0 && fabs(lambda - lambda_prev) < opt->tolerance) {
            converged = 1; ++it; break;
        }
        lambda_prev = lambda;
    }

    *eigval = lambda;
    stats->iters = it;
    stats->converged = converged;
    uint64_t t_end = clock_now();
    stats->t_total = clock_seconds(t_end - t_start);

    free(v_local); free(w_local); free(x_full);
    gpu_free(ws);
    return 0;
}
