#ifndef UTILS_H
#define UTILS_H

#include "eigensolver.h"

#ifdef __cplusplus
extern "C" {
#endif

void parse_options(int argc, char **argv, solver_options_t *opt);
void print_options(const solver_options_t *opt);
void print_stats(const solver_stats_t *s, int rank);

double drand_seeded(unsigned long *state);

int   matrix_dense_random_symmetric(dist_matrix_t *A, unsigned long seed,
                                    const parallel_ctx_t *ctx);
int   matrix_laplacian_2d(dist_matrix_t *A, const parallel_ctx_t *ctx);

double residual_norm_dense(const dist_matrix_t *A, const double *v_full,
                           double lambda, const parallel_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif
