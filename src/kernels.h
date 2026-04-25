#ifndef KERNELS_H
#define KERNELS_H

#include <stddef.h>
#include "eigensolver.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gpu_workspace_s gpu_workspace_t;

gpu_workspace_t *gpu_alloc(int local_n, int N, int local_nnz, int use_gpu);
void             gpu_free(gpu_workspace_t *ws);
int              gpu_is_active(const gpu_workspace_t *ws);

void gpu_upload_dense_block(gpu_workspace_t *ws, const double *A,
                            int local_n, int N);
void gpu_upload_csr_block(gpu_workspace_t *ws,
                          const int *row_ptr, const int *col_idx,
                          const double *vals, int local_n, int local_nnz);

void gpu_dense_matvec_local(gpu_workspace_t *ws,
                            const double *x_full, double *y_local,
                            int local_n, int N);

void gpu_csr_matvec_local(gpu_workspace_t *ws,
                          const double *x_full, double *y_local,
                          int local_n);

double gpu_dot_local(gpu_workspace_t *ws,
                     const double *x, const double *y, int n);

void   gpu_axpy(gpu_workspace_t *ws, double alpha,
                const double *x, double *y, int n);

void   gpu_scale(gpu_workspace_t *ws, double s, double *x, int n);

double gpu_norm_local(gpu_workspace_t *ws, const double *x, int n);

#ifdef __cplusplus
}
#endif

#endif
