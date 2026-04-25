#ifndef SOLVER_INTERNAL_H
#define SOLVER_INTERNAL_H

#include "eigensolver.h"
#include "kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

void modified_gram_schmidt(gpu_workspace_t *ws,
                           double **V_local, int j, int local_n,
                           double *w_local,
                           const parallel_ctx_t *ctx,
                           double *t_comm);

double parallel_norm(gpu_workspace_t *ws, const double *v_local,
                     int local_n,
                     const parallel_ctx_t *ctx, double *t_comm);

void cpu_dense_matvec(const dist_matrix_t *A,
                      const double *x_full, double *y_local);
void cpu_csr_matvec(const dist_matrix_t *A,
                    const double *x_full, double *y_local);

void distributed_matvec(const dist_matrix_t *A, gpu_workspace_t *ws,
                        const double *x_local, double *x_full,
                        double *y_local,
                        const parallel_ctx_t *ctx,
                        double *t_compute, double *t_comm);

#ifdef __cplusplus
}
#endif

#endif
