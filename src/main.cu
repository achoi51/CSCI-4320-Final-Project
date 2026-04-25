#include "eigensolver.h"
#include "utils.h"
#include "io.h"
#include "clockcycle.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

int parallel_init(int *argc, char ***argv, parallel_ctx_t *ctx, int want_gpu)
{
    MPI_Init(argc, argv);
    ctx->comm = MPI_COMM_WORLD;
    MPI_Comm_rank(ctx->comm, &ctx->rank);
    MPI_Comm_size(ctx->comm, &ctx->size);
    ctx->gpu_id = -1;
    ctx->have_gpu = 0;

#ifdef USE_CUDA
    if (want_gpu) {
        int n_devices = 0;
        if (cudaGetDeviceCount(&n_devices) == cudaSuccess && n_devices > 0) {
            int gpu = ctx->rank % n_devices;
            cudaSetDevice(gpu);
            ctx->gpu_id = gpu;
            ctx->have_gpu = 1;
        }
    }
#else
    (void)want_gpu;
#endif
    return 0;
}

void parallel_finalize(parallel_ctx_t *ctx)
{
    (void)ctx;
    MPI_Finalize();
}

static void print_eigvals(const char *label, const double *vals, int k)
{
    printf("=== %s (top %d) ===\n", label, k);
    for (int i = 0; i < k; ++i) printf("  lambda[%d] = %.10e\n", i, vals[i]);
    fflush(stdout);
}

int main(int argc, char **argv)
{
    solver_options_t opt;
    parse_options(argc, argv, &opt);

    parallel_ctx_t ctx;
    parallel_init(&argc, &argv, &ctx, opt.use_gpu);

    if (ctx.rank == 0) {
        print_options(&opt);
        printf("MPI ranks: %d\n", ctx.size);
#ifdef USE_CUDA
        printf("CUDA: enabled (this rank gpu=%d)\n", ctx.gpu_id);
#else
        printf("CUDA: disabled at compile time\n");
#endif
        fflush(stdout);
    }

    if (opt.N % ctx.size != 0) {
        if (ctx.rank == 0)
            fprintf(stderr, "warning: N=%d is not divisible by ranks=%d; "
                            "last rank gets remainder\n", opt.N, ctx.size);
    }

    dist_matrix_t A;
    if (matrix_build(&opt, &ctx, &A) != 0) {
        if (ctx.rank == 0) fprintf(stderr, "matrix_build failed\n");
        parallel_finalize(&ctx);
        return 1;
    }

    solver_stats_t stats;
    int rc = 0;

    switch (opt.solver) {
        case SOLVER_LANCZOS: {
            double *eigs = (double *)calloc(opt.num_eigs, sizeof(double));
            rc = run_lanczos(&A, &ctx, &opt, eigs, &stats);
            if (ctx.rank == 0) {
                print_eigvals("Lanczos eigenvalues", eigs, opt.num_eigs);
                io_write_eigenvalues(opt.output_path, eigs, opt.num_eigs, &ctx);
            }
            free(eigs);
            break;
        }
        case SOLVER_ARNOLDI: {
            double *re = (double *)calloc(opt.num_eigs, sizeof(double));
            double *im = (double *)calloc(opt.num_eigs, sizeof(double));
            rc = run_arnoldi(&A, &ctx, &opt, re, im, &stats);
            if (ctx.rank == 0) {
                printf("=== Arnoldi eigenvalues (top %d) ===\n", opt.num_eigs);
                for (int i = 0; i < opt.num_eigs; ++i)
                    printf("  lambda[%d] = %.10e + %.10ei\n", i, re[i], im[i]);
                io_write_eigenvalues(opt.output_path, re, opt.num_eigs, &ctx);
            }
            free(re); free(im);
            break;
        }
        case SOLVER_POWER: {
            double lam = 0.0;
            rc = run_power(&A, &ctx, &opt, &lam, &stats);
            if (ctx.rank == 0) {
                printf("=== Power-iteration dominant eigenvalue ===\n");
                printf("  lambda = %.10e\n", lam);
                io_write_eigenvalues(opt.output_path, &lam, 1, &ctx);
            }
            break;
        }
    }

    print_stats(&stats, ctx.rank);

    matrix_free(&A);
    parallel_finalize(&ctx);
    return rc;
}
