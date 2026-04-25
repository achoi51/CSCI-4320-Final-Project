#ifndef IO_H
#define IO_H

#include "eigensolver.h"

#ifdef __cplusplus
extern "C" {
#endif

int io_read_dense_matrix(const char *path, dist_matrix_t *A,
                         const parallel_ctx_t *ctx, double *t_io);

int io_write_dense_vector(const char *path, const double *local_v,
                          int local_n, int local_offset,
                          const parallel_ctx_t *ctx, double *t_io);

#ifdef __cplusplus
}
#endif

#endif
