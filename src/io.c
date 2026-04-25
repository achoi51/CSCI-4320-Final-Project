#include "io.h"
#include "clockcycle.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int io_read_dense_matrix(const char *path, dist_matrix_t *A,
                         const parallel_ctx_t *ctx, double *t_io)
{
    MPI_File fh;
    MPI_Status status;

    uint64_t t0 = clock_now();
    int rc = MPI_File_open(ctx->comm, (char *)path,
                           MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (rc != MPI_SUCCESS) {
        if (ctx->rank == 0)
            fprintf(stderr, "io: cannot open '%s'\n", path);
        return -1;
    }

    int header[2];
    if (ctx->rank == 0) {
        MPI_File_read_at(fh, 0, header, 2, MPI_INT, &status);
    }
    MPI_Bcast(header, 2, MPI_INT, 0, ctx->comm);

    int N = header[0];
    if (header[1] != N) {
        if (ctx->rank == 0)
            fprintf(stderr, "io: only square matrices supported\n");
        MPI_File_close(&fh);
        return -1;
    }

    int rows_per_rank = N / ctx->size;
    int local_offset = ctx->rank * rows_per_rank;
    int local_n = (ctx->rank == ctx->size - 1)
                      ? (N - local_offset)
                      : rows_per_rank;

    double *buf = (double *)malloc((size_t)local_n * N * sizeof(double));
    if (!buf) {
        MPI_File_close(&fh);
        return -1;
    }

    MPI_Offset disp = 2 * sizeof(int) +
                      (MPI_Offset)local_offset * N * sizeof(double);
    MPI_File_read_at_all(fh, disp, buf,
                         local_n * N, MPI_DOUBLE, &status);
    MPI_File_close(&fh);
    uint64_t t1 = clock_now();
    if (t_io) *t_io += clock_seconds(t1 - t0);

    A->N = N;
    A->local_n = local_n;
    A->local_offset = local_offset;
    A->kind = MATRIX_FROM_FILE;
    A->h_dense = buf;
    A->h_row_ptr = NULL;
    A->h_col_idx = NULL;
    A->h_values = NULL;
    A->local_nnz = 0;
    return 0;
}

int io_write_dense_vector(const char *path, const double *local_v,
                          int local_n, int local_offset,
                          const parallel_ctx_t *ctx, double *t_io)
{
    MPI_File fh;
    uint64_t t0 = clock_now();
    int rc = MPI_File_open(ctx->comm, (char *)path,
                           MPI_MODE_CREATE | MPI_MODE_WRONLY,
                           MPI_INFO_NULL, &fh);
    if (rc != MPI_SUCCESS) return -1;

    MPI_Offset disp = (MPI_Offset)local_offset * sizeof(double);
    MPI_File_write_at_all(fh, disp, local_v, local_n, MPI_DOUBLE,
                          MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
    uint64_t t1 = clock_now();
    if (t_io) *t_io += clock_seconds(t1 - t0);
    return 0;
}

int io_write_eigenvalues(const char *path, const double *vals, int k,
                         const parallel_ctx_t *ctx)
{
    if (ctx->rank != 0) return 0;
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "# %d eigenvalues\n", k);
    for (int i = 0; i < k; ++i)
        fprintf(f, "%d %.16e\n", i, vals[i]);
    fclose(f);
    return 0;
}

int io_checkpoint_vector(const char *path, const double *local_v,
                         int local_n, int local_offset,
                         const parallel_ctx_t *ctx)
{
    return io_write_dense_vector(path, local_v, local_n, local_offset,
                                 ctx, NULL);
}
