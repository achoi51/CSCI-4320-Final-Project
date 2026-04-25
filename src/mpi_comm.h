#ifndef MPI_COMM_H
#define MPI_COMM_H

#include "eigensolver.h"

#ifdef __cplusplus
extern "C" {
#endif

double timed_allreduce_sum(double local, const parallel_ctx_t *ctx,
                           double *t_comm);

void timed_allreduce_sum_array(const double *in, double *out, int n,
                               const parallel_ctx_t *ctx, double *t_comm);

void timed_allgather(const double *send, int send_n,
                     double *recv, int recv_per_rank,
                     const parallel_ctx_t *ctx, double *t_comm);

#ifdef __cplusplus
}
#endif

#endif
