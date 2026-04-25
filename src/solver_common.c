#include "solver_internal.h"
#include "clockcycle.h"
#include <stdlib.h>
#include <string.h>

void cpu_dense_matvec(const dist_matrix_t *A,
                      const double *x_full, double *y_local)
{
    int N = A->N;
    int local_n = A->local_n;
    for (int i = 0; i < local_n; ++i) {
        double s = 0.0;
        const double *row = &A->h_dense[(size_t)i * N];
        for (int j = 0; j < N; ++j) s += row[j] * x_full[j];
        y_local[i] = s;
    }
}

void cpu_csr_matvec(const dist_matrix_t *A,
                    const double *x_full, double *y_local)
{
    int local_n = A->local_n;
    for (int i = 0; i < local_n; ++i) {
        double s = 0.0;
        int rs = A->h_row_ptr[i];
        int re = A->h_row_ptr[i + 1];
        for (int p = rs; p < re; ++p)
            s += A->h_values[p] * x_full[A->h_col_idx[p]];
        y_local[i] = s;
    }
}

void distributed_matvec(const dist_matrix_t *A, gpu_workspace_t *ws,
                        const double *x_local, double *x_full,
                        double *y_local,
                        const parallel_ctx_t *ctx,
                        double *t_compute, double *t_comm)
{
    allgather_full_vector(x_local, x_full, A->local_n, A->N, ctx, t_comm);

    uint64_t t0 = clock_now();
    if (A->kind == MATRIX_DENSE_SYM || A->kind == MATRIX_FROM_FILE) {
        if (gpu_is_active(ws))
            gpu_dense_matvec_local(ws, x_full, y_local, A->local_n, A->N);
        else
            cpu_dense_matvec(A, x_full, y_local);
    } else {
        if (gpu_is_active(ws))
            gpu_csr_matvec_local(ws, x_full, y_local, A->local_n);
        else
            cpu_csr_matvec(A, x_full, y_local);
    }
    uint64_t t1 = clock_now();
    if (t_compute) *t_compute += clock_seconds(t1 - t0);
}
