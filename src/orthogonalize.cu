#include "solver_internal.h"
#include "mpi_comm.h"
#include <math.h>

void modified_gram_schmidt(gpu_workspace_t *ws,
                           double **V_local, int j, int local_n,
                           double *w_local,
                           const parallel_ctx_t *ctx,
                           double *t_comm)
{
    for (int i = 0; i <= j; ++i) {
        double local = gpu_dot_local(ws, V_local[i], w_local, local_n);
        double global = timed_allreduce_sum(local, ctx, t_comm);
        gpu_axpy(ws, -global, V_local[i], w_local, local_n);
    }
}

double parallel_norm(gpu_workspace_t *ws, const double *v_local,
                     int local_n,
                     const parallel_ctx_t *ctx, double *t_comm)
{
    double local = gpu_dot_local(ws, v_local, v_local, local_n);
    double global = timed_allreduce_sum(local, ctx, t_comm);
    return sqrt(global);
}
