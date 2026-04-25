#include "mpi_comm.h"
#include "clockcycle.h"

double timed_allreduce_sum(double local, const parallel_ctx_t *ctx,
                           double *t_comm)
{
    double global = 0.0;
    uint64_t t0 = clock_now();
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, ctx->comm);
    uint64_t t1 = clock_now();
    if (t_comm) *t_comm += clock_seconds(t1 - t0);
    return global;
}

void timed_allreduce_sum_array(const double *in, double *out, int n,
                               const parallel_ctx_t *ctx, double *t_comm)
{
    uint64_t t0 = clock_now();
    MPI_Allreduce(in, out, n, MPI_DOUBLE, MPI_SUM, ctx->comm);
    uint64_t t1 = clock_now();
    if (t_comm) *t_comm += clock_seconds(t1 - t0);
}

void timed_allgather(const double *send, int send_n,
                     double *recv, int recv_per_rank,
                     const parallel_ctx_t *ctx, double *t_comm)
{
    uint64_t t0 = clock_now();
    MPI_Allgather(send, send_n, MPI_DOUBLE,
                  recv, recv_per_rank, MPI_DOUBLE, ctx->comm);
    uint64_t t1 = clock_now();
    if (t_comm) *t_comm += clock_seconds(t1 - t0);
}

void allreduce_sum_double(const double *in, double *out, int n,
                          const parallel_ctx_t *ctx, double *t_comm)
{
    timed_allreduce_sum_array(in, out, n, ctx, t_comm);
}

void allgather_full_vector(const double *local, double *full,
                           int local_n, int N,
                           const parallel_ctx_t *ctx, double *t_comm)
{
    (void)N;
    timed_allgather(local, local_n, full, local_n, ctx, t_comm);
}
