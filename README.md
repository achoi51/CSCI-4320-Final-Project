# Parallelized Computational Eigensolver

**Authors:** Adam Choi (`choia9`), Jordan Krishnayah (`krishj4`)
**Course:** Parallel Computing / Programming (PCPG) - Spring 2026, Rensselaer Polytechnic Institute
**Instructor:** Prof. Christopher D. Carothers
**Deadline:** April 29, 2026, 11:59 PM

---

## Table of Contents

1. [Overview](#overview)
2. [Motivation](#motivation)
3. [Architecture](#architecture)
4. [Algorithms](#algorithms)
5. [Implementation Guide](#implementation-guide)
6. [Building and Running](#building-and-running)
7. [Performance Evaluation](#performance-evaluation)
8. [Project Requirements Checklist](#project-requirements-checklist)
9. [References](#references)

---

## Overview

This project implements a parallelized computational eigensolver targeting the AiMOS supercomputer's POWER9 architecture with NVIDIA V100 GPUs. The solver uses a hybrid **MPI + CUDA** approach to compute eigenvalues of large dense and sparse matrices via iterative methods, avoiding full matrix diagonalization to reduce computational cost and improve scalability.

Eigenvalue computations are fundamental to quantum mechanics, graph theory, machine learning, structural engineering, and climate modeling, but they represent a major computational bottleneck as matrix sizes grow. This project aims to push those boundaries by leveraging massively parallel hardware.

---

## Motivation

Traditional direct eigensolvers (e.g., solving the characteristic polynomial or performing full diagonalization) scale poorly with matrix size, often reaching O(n^3) complexity. Iterative methods such as the Lanczos algorithm (for sparse/symmetric matrices) and the Arnoldi iteration (for general matrices) allow computation of selected eigenvalues without full decomposition, making them ideal candidates for parallelization.

By distributing work across MPI ranks and offloading dense linear algebra kernels to GPUs via CUDA, this solver targets problems that are otherwise intractable on single-node systems, including:

- Climate and fluid dynamics simulations
- Structural engineering finite element models
- Quantum chemistry (density functional theory)
- Graph Laplacian spectral analysis
- PDE-derived sparse systems

---

## Architecture

### Hardware Target

| Component | Specification |
|-----------|--------------|
| **System** | AiMOS Supercomputer (CCI, RPI) |
| **CPU** | IBM POWER9 |
| **GPU** | NVIDIA Tesla V100 |
| **Interconnect** | NVLink (CPU-GPU), InfiniBand (inter-node) |
| **Storage** | GPFS + NVMe local drives |

### Software Stack

| Layer | Technology |
|-------|-----------|
| **Inter-node communication** | MPI (Message Passing Interface) |
| **GPU computation** | CUDA |
| **Parallel I/O** | MPI I/O |
| **Timing** | POWER9 cycle counters (`clock_now`, 512 MHz resolution) |
| **Build system** | Makefile / CMake |

### Parallelization Strategy

```
+-----------------------------------------------------------+
|                    MPI Rank Distribution                   |
|                                                           |
|  Rank 0          Rank 1          ...        Rank N-1      |
|  +----------+   +----------+              +----------+   |
|  | CPU Work  |   | CPU Work  |              | CPU Work  |  |
|  |  - Local  |   |  - Local  |              |  - Local  |  |
|  |    matrix |   |    matrix |              |    matrix |  |
|  |    block  |   |    block  |              |    block  |  |
|  +-----+----+   +-----+----+              +-----+----+   |
|        |              |                          |        |
|        v              v                          v        |
|  +-----------+  +-----------+             +-----------+  |
|  | GPU Kernel|  | GPU Kernel|             | GPU Kernel|  |
|  |  - MatVec |  |  - MatVec |             |  - MatVec |  |
|  |  - Norms  |  |  - Norms  |             |  - Norms  |  |
|  |  - Ortho  |  |  - Ortho  |             |  - Ortho  |  |
|  +-----------+  +-----------+             +-----------+  |
|                                                           |
|  <-- MPI Communication (Allreduce, Allgather, etc.) -->   |
+-----------------------------------------------------------+
```

- **MPI** distributes matrix rows/blocks across ranks (1D block-row or 2D block-cyclic distribution)
- **CUDA** accelerates per-rank compute-intensive operations (matrix-vector products, orthogonalization, norm computations)
- **MPI I/O** handles parallel reading/writing of matrix data and checkpointing
- Communication patterns include `MPI_Allreduce` for global dot products/norms and `MPI_Allgather` for basis vector assembly

---

## Algorithms

### Iterative Eigensolver Methods

#### Lanczos Algorithm (Symmetric/Hermitian Matrices)

Best suited for large sparse symmetric matrices. Builds an orthonormal basis for the Krylov subspace and reduces the eigenvalue problem to a tridiagonal form.

```
Pseudocode:
  1. Choose random starting vector v_1, normalize
  2. For j = 1, 2, ..., m:
       a. w = A * v_j                          [GPU: parallel SpMV]
       b. alpha_j = v_j^T * w                  [GPU: dot product, MPI: Allreduce]
       c. w = w - alpha_j * v_j - beta_{j-1} * v_{j-1}   [GPU: axpy]
       d. Re-orthogonalize w against all v_i    [GPU: Gram-Schmidt]
       e. beta_j = ||w||                        [GPU: norm, MPI: Allreduce]
       f. v_{j+1} = w / beta_j
  3. Compute eigenvalues of tridiagonal matrix T (alpha, beta)
  4. Recover eigenvectors of A from Ritz vectors
```

#### Arnoldi Iteration (General/Non-symmetric Matrices)

Generalizes Lanczos to non-symmetric matrices, producing an upper Hessenberg matrix.

#### Power Iteration (Dominant Eigenvalue)

Simple baseline method for finding the largest eigenvalue. Used for validation and comparison.

### Parallelization of Key Kernels

| Kernel | Parallelization Approach |
|--------|------------------------|
| **Sparse Matrix-Vector Multiply (SpMV)** | Each MPI rank owns a block of rows; CUDA kernel performs local SpMV; halo exchange via MPI |
| **Dense Matrix-Vector Multiply** | 1D block-row distribution; local GEMV on GPU; `MPI_Allgather` to assemble result |
| **Dot Product / Norms** | Local partial sum on GPU; `MPI_Allreduce(MPI_SUM)` for global result |
| **Orthogonalization** | Modified Gram-Schmidt with GPU-accelerated projections and MPI reductions |
| **Tridiagonal Eigensolver** | Sequential on rank 0 (small m x m matrix) or parallel divide-and-conquer |

---

## Implementation Guide

### Project Structure

```
parallel/
|-- src/
|   |-- main.c / main.cu        # Entry point, MPI init, solver dispatch
|   |-- eigensolver.cu           # CUDA kernels for eigensolver iterations
|   |-- lanczos.cu               # Lanczos-specific implementation
|   |-- matvec.cu                # Matrix-vector product kernels
|   |-- orthogonalize.cu         # Gram-Schmidt orthogonalization
|   |-- mpi_comm.c               # MPI communication wrappers
|   |-- io.c                     # MPI I/O for matrix read/write
|   |-- clockcycle.h             # POWER9 cycle counter
|   |-- utils.c                  # Matrix generation, validation
|-- Makefile / CMakeLists.txt
|-- scripts/
|   |-- submit.sh                # SLURM job submission script
|   |-- run_strong_scaling.sh    # Strong scaling experiment
|   |-- run_weak_scaling.sh      # Weak scaling experiment
|-- data/                        # Input matrices, output results
|-- README.md
|-- proposal.pdf
|-- projectinfo.pdf
```

### POWER9 Cycle Counter

Use the provided `clock_now()` function for high-resolution timing on AiMOS (512 MHz resolution):

```c
// clockcycle.h - ONLY WORKS ON POWER9/AiMOS
#ifndef CLOCKCYCLE_H
#define CLOCKCYCLE_H

#include <stdint.h>

uint64_t clock_now(void)
{
    unsigned int tbl, tbu0, tbu1;
    do {
        __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
        __asm__ __volatile__ ("mftb %0" : "=r"(tbl));
        __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
    } while (tbu0 != tbu1);
    return (((uint64_t)tbu0) << 32) | tbl;
}

#endif // CLOCKCYCLE_H
```

Convert cycles to seconds: `time_seconds = (end - start) / 512000000.0`

### Key Implementation Steps

1. **MPI Initialization** - Set up communicator, determine rank and size, compute local matrix dimensions
2. **Matrix Distribution** - Distribute matrix rows across MPI ranks using block-row partitioning
3. **GPU Memory Setup** - Allocate device memory for local matrix block, vectors, and workspace
4. **Iterative Solver Loop** - Execute Lanczos/Arnoldi iterations with GPU-accelerated kernels and MPI communication
5. **Convergence Check** - Monitor residual norms; terminate when eigenvalues converge to tolerance
6. **Result Collection** - Gather eigenvectors to rank 0; write results via MPI I/O
7. **Timing & Profiling** - Instrument computation, communication, and I/O phases separately using `clock_now()`

### MPI Communication Patterns

```c
// Global dot product: local GPU reduction + MPI Allreduce
double local_dot = gpu_dot_product(d_x, d_y, local_n);
double global_dot;
MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

// Distributed SpMV: local computation + halo exchange
gpu_local_spmv(d_local_A, d_x_local, d_y_local, local_nnz, local_n);
MPI_Allgather(d_x_local, local_n, MPI_DOUBLE, d_x_global, local_n, MPI_DOUBLE, MPI_COMM_WORLD);
```

### MPI I/O for Checkpointing

```c
MPI_File fh;
MPI_File_open(MPI_COMM_WORLD, "checkpoint.dat",
              MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
MPI_File_set_view(fh, rank * local_n * sizeof(double),
                  MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
MPI_File_write(fh, local_data, local_n, MPI_DOUBLE, MPI_STATUS_IGNORE);
MPI_File_close(&fh);
```

---

## Building and Running

### Prerequisites

- AiMOS supercomputer access (CCI, RPI)
- GCC with POWER9 support or IBM XL compilers
- CUDA Toolkit (for NVIDIA V100)
- OpenMPI or Spectrum MPI
- SLURM workload manager

### Compilation

```bash
# Using Makefile
make clean && make

# Or using CMake
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=70   # V100 = SM 7.0
make -j
```

### Running on AiMOS

```bash
# Interactive debug (for testing)
salloc -N 1 -n 4 --gres=gpu:4 -t 00:30:00 -p debug
mpirun -np 4 ./eigensolver --matrix-size 4096 --num-eigenvalues 10

# Batch submission
sbatch scripts/submit.sh
```

### Example SLURM Script

```bash
#!/bin/bash
#SBATCH --job-name=eigensolver
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=results_%j.out

module load gcc cuda openmpi

mpirun -np 16 ./eigensolver \
    --matrix-size 16384 \
    --num-eigenvalues 20 \
    --tolerance 1e-10 \
    --max-iterations 500 \
    --output results/output.dat
```

### NVMe Local Storage

For I/O-intensive runs, use the node-local NVMe drives for temporary data:

```bash
# NVMe is typically mounted at /mnt/nvme on AiMOS nodes
export TMPDIR=/mnt/nvme/$SLURM_JOB_ID
mkdir -p $TMPDIR
```

---

## Performance Evaluation

### Benchmark Matrices

| Matrix Type | Description | Use Case |
|-------------|-------------|----------|
| Dense random symmetric | Random entries, symmetric | Baseline correctness/performance |
| Sparse PDE (Laplacian) | 5-point stencil from 2D Poisson | Scientific computing representative |
| Graph Laplacian | From random/real-world graphs | Network analysis |
| Ill-conditioned | High condition number | Numerical robustness testing |

### Scaling Studies

#### Strong Scaling

Fixed problem size (e.g., N = 16384), varying the number of MPI ranks / GPUs:

| Ranks | GPUs | Expected Behavior |
|-------|------|-------------------|
| 1     | 1    | Baseline serial-parallel |
| 4     | 4    | Near-linear speedup |
| 16    | 16   | Communication overhead begins to show |
| 64    | 64   | Saturation point analysis |

Metrics to collect:
- Total wall-clock time
- Computation time (GPU kernels)
- Communication time (MPI calls)
- I/O time (MPI I/O reads/writes)
- Speedup = T(1) / T(p)
- Efficiency = Speedup / p

#### Weak Scaling

Problem size grows proportionally with rank count (e.g., 4096 rows per rank):

| Ranks | Matrix Size | Rows/Rank |
|-------|------------|-----------|
| 1     | 4096       | 4096      |
| 4     | 8192       | ~4096     |
| 16    | 16384      | ~4096     |
| 64    | 32768      | ~4096     |

Ideal: constant runtime as ranks and problem size grow together.

#### CPU-only vs. Hybrid CPU/GPU

Compare MPI-only (CPU) runs against MPI + CUDA (hybrid) runs to quantify GPU acceleration benefit.

### Timing Instrumentation

```c
uint64_t t_start, t_end;

// Time computation
t_start = clock_now();
gpu_lanczos_iteration(...);
cudaDeviceSynchronize();
t_end = clock_now();
double compute_time = (t_end - t_start) / 512000000.0;

// Time communication
t_start = clock_now();
MPI_Allreduce(...);
t_end = clock_now();
double comm_time = (t_end - t_start) / 512000000.0;

// Time I/O
t_start = clock_now();
MPI_File_write(...);
t_end = clock_now();
double io_time = (t_end - t_start) / 512000000.0;
```

### Validation

- Compare computed eigenvalues against known analytical solutions (e.g., Laplacian eigenvalues)
- Check residual norm: ||Av - lambda*v|| / ||v|| < tolerance
- Cross-validate with serial LAPACK `dsyev` for small matrices

---

## Project Requirements Checklist

Based on the course project specification (Prof. Carothers):

| Requirement | Status |
|-------------|--------|
| Non-embarrassingly parallel with high interaction | Iterative solver with frequent MPI reductions |
| Uses MPI | Block-row distribution, Allreduce, Allgather |
| Uses CUDA | GPU-accelerated SpMV, dot products, orthogonalization |
| MPI I/O | Parallel matrix read/write and checkpointing |
| Abstract (150+ words) | To be included in final report |
| Overview / Introduction | Motivation and background in report |
| Parallel algorithm description with pseudocode | Lanczos/Arnoldi parallel algorithm |
| Literature review (10+ references) | Google Scholar survey of parallel eigensolvers |
| Strong scaling study | Fixed N, increasing ranks/GPUs |
| Weak scaling study | Proportional N growth with ranks |
| CPU-only vs. CPU/GPU comparison | MPI-only vs. MPI+CUDA |
| Communication overhead analysis | `clock_now()` instrumented comm vs. compute |
| MPI I/O performance analysis | Timed parallel read/write operations |
| Summary and future work | In final report |
| 6+ pages, double-column, 10pt (undergrad) | ACM Master Conference Format via Overleaf |
| 10+ references | IEEE, ACM, SIAM sources |

---

## References

Key literature and resources for this project:

1. Hernandez, V., Roman, J. E., & Vidal, V. (2005). SLEPc: A scalable and flexible toolkit for the solution of eigenvalue problems. *ACM Transactions on Mathematical Software*, 31(3), 351-362.
2. Yamazaki, I., et al. (2019). Performance of LOBPCG method for computing eigenvalues on GPU. *IEEE International Parallel and Distributed Processing Symposium*.
3. Stathopoulos, A., & McCombs, J. R. (2010). PRIMME: Preconditioned iterative multimethod eigensolver. *ACM Transactions on Mathematical Software*, 37(2).
4. Dongarra, J., et al. (2000). A set of level 3 basic linear algebra subprograms. *ACM Transactions on Mathematical Software*.
5. Balay, S., et al. PETSc: Portable, Extensible Toolkit for Scientific Computation.
6. Lee, V., et al. (2010). Debunking the 100X GPU vs. CPU myth. *ACM SIGARCH Computer Architecture News*, 38(3), 451-460. DOI: 10.1145/1815961.1816021.
7. AiMOS / DCS Supercomputer Documentation. CCI, RPI. https://docs.cci.rpi.edu/clusters/DCS_Supercomputer/
8. NVIDIA CUDA Programming Guide. NVIDIA Corporation.
9. Gropp, W., Lusk, E., & Thakur, R. (1999). *Using MPI-2: Advanced Features of the Message Passing Interface*. MIT Press.
10. ACM Supercomputing 2025 Proceedings. DOI: 10.1145/3712285.

---

## Team Contributions

| Member | Responsibilities |
|--------|-----------------|
| **Adam Choi** (`choia9`) | TBD - describe specific contributions |
| **Jordan Krishnayah** (`krishj4`) | TBD - describe specific contributions |

---

## License

This project is submitted as coursework for PCPG Spring 2026 at Rensselaer Polytechnic Institute.
