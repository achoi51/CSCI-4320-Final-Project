#include "kernels.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#define CUDA_CHECK(call) do {                                  \
    cudaError_t _e = (call);                                   \
    if (_e != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(_e));   \
        abort();                                               \
    }                                                          \
} while (0)
#endif

struct gpu_workspace_s {
    int active;
    int local_n;
    int N;
    int local_nnz;

#ifdef USE_CUDA
    double *d_dense;
    int    *d_row_ptr;
    int    *d_col_idx;
    double *d_values;
    double *d_x;
    double *d_y;
    double *d_reduce;
    double *h_pinned;
#endif
};

gpu_workspace_t *gpu_alloc(int local_n, int N, int local_nnz, int use_gpu)
{
    gpu_workspace_t *ws = (gpu_workspace_t *)calloc(1, sizeof(*ws));
    ws->local_n = local_n;
    ws->N = N;
    ws->local_nnz = local_nnz;
    ws->active = 0;

#ifdef USE_CUDA
    if (use_gpu) {
        int n_devices = 0;
        if (cudaGetDeviceCount(&n_devices) == cudaSuccess && n_devices > 0) {
            ws->active = 1;
            CUDA_CHECK(cudaMalloc((void **)&ws->d_x, (size_t)N * sizeof(double)));
            CUDA_CHECK(cudaMalloc((void **)&ws->d_y, (size_t)local_n * sizeof(double)));
            CUDA_CHECK(cudaMalloc((void **)&ws->d_reduce, 1024 * sizeof(double)));
            ws->d_dense = NULL;
            ws->d_row_ptr = NULL;
            ws->d_col_idx = NULL;
            ws->d_values = NULL;
            ws->h_pinned = NULL;
        }
    }
#else
    (void)use_gpu;
#endif
    return ws;
}

void gpu_free(gpu_workspace_t *ws)
{
    if (!ws) return;
#ifdef USE_CUDA
    if (ws->active) {
        if (ws->d_dense)   cudaFree(ws->d_dense);
        if (ws->d_row_ptr) cudaFree(ws->d_row_ptr);
        if (ws->d_col_idx) cudaFree(ws->d_col_idx);
        if (ws->d_values)  cudaFree(ws->d_values);
        if (ws->d_x)       cudaFree(ws->d_x);
        if (ws->d_y)       cudaFree(ws->d_y);
        if (ws->d_reduce)  cudaFree(ws->d_reduce);
        if (ws->h_pinned)  cudaFreeHost(ws->h_pinned);
    }
#endif
    free(ws);
}

int gpu_is_active(const gpu_workspace_t *ws) { return ws && ws->active; }

void gpu_upload_dense_block(gpu_workspace_t *ws, const double *A,
                            int local_n, int N)
{
#ifdef USE_CUDA
    if (!ws->active) return;
    size_t bytes = (size_t)local_n * N * sizeof(double);
    if (!ws->d_dense) CUDA_CHECK(cudaMalloc((void **)&ws->d_dense, bytes));
    CUDA_CHECK(cudaMemcpy(ws->d_dense, A, bytes, cudaMemcpyHostToDevice));
#else
    (void)ws; (void)A; (void)local_n; (void)N;
#endif
}

void gpu_upload_csr_block(gpu_workspace_t *ws,
                          const int *row_ptr, const int *col_idx,
                          const double *vals, int local_n, int local_nnz)
{
#ifdef USE_CUDA
    if (!ws->active) return;
    if (!ws->d_row_ptr)
        CUDA_CHECK(cudaMalloc((void **)&ws->d_row_ptr, (local_n + 1) * sizeof(int)));
    if (!ws->d_col_idx)
        CUDA_CHECK(cudaMalloc((void **)&ws->d_col_idx, local_nnz * sizeof(int)));
    if (!ws->d_values)
        CUDA_CHECK(cudaMalloc((void **)&ws->d_values, local_nnz * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(ws->d_row_ptr, row_ptr,
                          (local_n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ws->d_col_idx, col_idx,
                          local_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ws->d_values, vals,
                          local_nnz * sizeof(double), cudaMemcpyHostToDevice));
#else
    (void)ws; (void)row_ptr; (void)col_idx; (void)vals;
    (void)local_n; (void)local_nnz;
#endif
}

#ifdef USE_CUDA

__global__ void k_dense_matvec(const double *A, const double *x,
                               double *y, int local_n, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= local_n) return;
    double sum = 0.0;
    const double *Arow = A + (size_t)row * N;
    for (int j = 0; j < N; ++j) sum += Arow[j] * x[j];
    y[row] = sum;
}

__global__ void k_csr_matvec(const int *row_ptr, const int *col_idx,
                             const double *vals, const double *x,
                             double *y, int local_n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= local_n) return;
    int rs = row_ptr[row];
    int re = row_ptr[row + 1];
    double sum = 0.0;
    for (int p = rs; p < re; ++p) sum += vals[p] * x[col_idx[p]];
    y[row] = sum;
}

__global__ void k_axpy(double alpha, const double *x, double *y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += alpha * x[i];
}

__global__ void k_scale(double s, double *x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

__global__ void k_dot_partial(const double *x, const double *y,
                              double *out, int n)
{
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    double v = (i < n) ? x[i] * y[i] : 0.0;
    sdata[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

#endif

void gpu_dense_matvec_local(gpu_workspace_t *ws,
                            const double *x_full, double *y_local,
                            int local_n, int N)
{
#ifdef USE_CUDA
    if (ws->active) {
        CUDA_CHECK(cudaMemcpy(ws->d_x, x_full, (size_t)N * sizeof(double),
                              cudaMemcpyHostToDevice));
        int threads = 128;
        int blocks = (local_n + threads - 1) / threads;
        k_dense_matvec<<<blocks, threads>>>(ws->d_dense, ws->d_x, ws->d_y,
                                            local_n, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(y_local, ws->d_y,
                              (size_t)local_n * sizeof(double),
                              cudaMemcpyDeviceToHost));
        return;
    }
#endif
    /* CPU fallback (host A pointer carried out-of-band via ws->d_dense unused) */
    /* Caller must use cpu_dense_matvec when GPU is inactive. */
    (void)ws; (void)x_full; (void)y_local; (void)local_n; (void)N;
}

void gpu_csr_matvec_local(gpu_workspace_t *ws,
                          const double *x_full, double *y_local, int local_n)
{
#ifdef USE_CUDA
    if (ws->active) {
        CUDA_CHECK(cudaMemcpy(ws->d_x, x_full, (size_t)ws->N * sizeof(double),
                              cudaMemcpyHostToDevice));
        int threads = 128;
        int blocks = (local_n + threads - 1) / threads;
        k_csr_matvec<<<blocks, threads>>>(ws->d_row_ptr, ws->d_col_idx,
                                          ws->d_values, ws->d_x, ws->d_y,
                                          local_n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(y_local, ws->d_y,
                              (size_t)local_n * sizeof(double),
                              cudaMemcpyDeviceToHost));
        return;
    }
#endif
    (void)ws; (void)x_full; (void)y_local; (void)local_n;
}

double gpu_dot_local(gpu_workspace_t *ws,
                     const double *x, const double *y, int n)
{
#ifdef USE_CUDA
    if (ws->active) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        if (blocks > 1024) blocks = 1024;
        double *dx, *dy;
        CUDA_CHECK(cudaMalloc(&dx, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dy, n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(dx, x, n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dy, y, n * sizeof(double), cudaMemcpyHostToDevice));
        k_dot_partial<<<blocks, threads, threads * sizeof(double)>>>(dx, dy, ws->d_reduce, n);
        CUDA_CHECK(cudaGetLastError());
        double *partials = (double *)malloc(blocks * sizeof(double));
        CUDA_CHECK(cudaMemcpy(partials, ws->d_reduce,
                              blocks * sizeof(double), cudaMemcpyDeviceToHost));
        double s = 0.0;
        for (int i = 0; i < blocks; ++i) s += partials[i];
        free(partials);
        cudaFree(dx); cudaFree(dy);
        return s;
    }
#else
    (void)ws;
#endif
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += x[i] * y[i];
    return s;
}

void gpu_axpy(gpu_workspace_t *ws, double alpha,
              const double *x, double *y, int n)
{
    (void)ws;
    for (int i = 0; i < n; ++i) y[i] += alpha * x[i];
}

void gpu_scale(gpu_workspace_t *ws, double s, double *x, int n)
{
    (void)ws;
    for (int i = 0; i < n; ++i) x[i] *= s;
}

double gpu_norm_local(gpu_workspace_t *ws, const double *x, int n)
{
    return gpu_dot_local(ws, x, x, n);
}
