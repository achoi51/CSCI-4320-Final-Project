# How to Build and Run the Eigensolver

This document covers compiling and running the Parallelized Computational
Eigensolver in three environments:

1. **AiMOS / DCS** — full hybrid MPI + CUDA build (the production target)
2. **Any Linux/macOS workstation with MPI** — CPU-only build for development
3. **Single-rank smoke test** — quick validation without `mpirun`

---

## 1. Prerequisites

| Environment | Required modules / packages |
|-------------|------------------------------|
| AiMOS (DCS) | `gcc`, `cuda`, `spectrum-mpi` (or `openmpi`), `slurm` |
| Local Linux | `gcc`/`clang`, `openmpi` or `mpich` (provides `mpicc`/`mpicxx`) |
| Local macOS | `brew install open-mpi` |

The Makefile detects the toolchain from the standard MPI compiler wrappers
(`mpicc`, `mpicxx`) and `nvcc` for CUDA builds.

---

## 2. Building

### 2.1 Hybrid MPI + CUDA (AiMOS)

On AiMOS, `cuda` is gated behind the IBM XL compiler module — load `xl_r`
first or the bare `cuda` request fails with `cannot be loaded as
requested`:

```bash
module load xl_r spectrum-mpi cuda   # default is cuda 11.2
make
```

This produces `build/eigensolver`. The CUDA architecture defaults to
`sm_70` (V100), which is what the dcs nodes have. Override for other GPUs
with:

```bash
make CUDA_ARCH=sm_80
```

### 2.2 CPU-only (no CUDA)

For local development, scaling tests on CPU clusters, or to produce the
"MPI-only" baseline required by the project rubric:

```bash
make cpu
```

This produces `build/eigensolver_cpu`. All `.cu` files are compiled as
plain C++ via `mpicxx`; the GPU kernels are stubbed out and the solver
falls back to portable CPU code paths.

### 2.3 Cleaning

```bash
make clean
```

---

## 3. Running

### 3.1 Command-line options

```
--solver {lanczos|arnoldi|power}     iterative method (default: lanczos)
--matrix {dense|laplacian2d|file}    input matrix kind (default: dense)
--matrix-size N                      global matrix dimension N x N
--num-eigenvalues K                  number of eigenvalues to extract
--max-iterations M                   Krylov subspace dimension / iterations
--tolerance EPS                      convergence tolerance (default 1e-9)
--seed S                             RNG seed for matrix generation
--cpu-only                           skip GPU even if CUDA is compiled in
--checkpoint-every N                 write Lanczos vector every N iters
--input PATH                         binary matrix file (sets matrix=file)
--output PATH                        eigenvalue output (default results/eigvals.dat)
```

### 3.2 Single-node interactive (AiMOS class workflow)

The CSCI-4320 class workflow on AiMOS is `salloc` + `ssh`, not direct
`mpirun` from the front end. Max 30 min per job, max 16 nodes per job.

```bash
salloc -N 1 --partition=el8-rpi --gres=gpu:4 -t 30
squeue -u $USER                 # find the dcsXYZ name (e.g. dcs24)
ssh dcs24                       # ssh to the allocated node
cd <repo>
module load xl_r spectrum-mpi cuda
mpirun -np 4 ./build/eigensolver \
    --solver lanczos --matrix dense \
    --matrix-size 4096 --num-eigenvalues 10
exit                            # back to front end
scancel <JOBID>                 # if you're done before the walltime
```

### 3.3 Batch submission (AiMOS)

```bash
sbatch scripts/submit.sh
```

#### 3.3.1 Bare-minimum smoke test (AiMOS)

Use this when you just need *something* to land on a node and produce a
result file. It requests **1 node, 2 ranks, 1 GPU, 10 minutes** on the
`el8-rpi` partition — well under the 30-minute class limit, so it slots
in quickly:

```bash
module load xl_r spectrum-mpi cuda     # xl_r must come first or `cuda` fails to resolve
make                                   # or `make cpu` if CUDA isn't available yet
sbatch scripts/submit_smoke.sh
```

The script falls back to `build/eigensolver_cpu` automatically if the
CUDA build isn't there.

#### 3.3.2 Interactive smoke test (no sbatch)

If you'd rather use the class's `salloc` + `ssh` flow:

```bash
salloc -N 1 --partition=el8-rpi --gres=gpu:4 -t 30
squeue -u $USER                        # find dcsXYZ
ssh dcsXYZ
cd <repo>
./scripts/run_smoke_interactive.sh
```

What you'll get back (in `results/smoke_<jobid>.out`):

- a header line with hostname / job id / ntasks
- the solver's `=== Eigensolver configuration ===` block
- top-4 eigenvalues of a 512x512 dense symmetric matrix
- the `=== Solver stats ===` timing summary (`t_compute`, `t_comm`, `t_total`)

And `results/smoke_eigvals_<jobid>.dat` with the four eigenvalues, one per
line. Total runtime is a few seconds; the 10-minute wall clock is just
slack so the scheduler doesn't reject it.

Check status / fetch results:

```bash
squeue -u $USER
ls -lt results/smoke_*.out | head
cat results/smoke_*.out
```

Override defaults via environment variables:

```bash
N=32768 NEV=20 SOLVER=lanczos MATRIX=laplacian2d sbatch scripts/submit.sh
```

### 3.4 Local smoke test (CPU build)

```bash
make cpu
mpirun -np 2 ./build/eigensolver_cpu \
    --solver power --matrix dense \
    --matrix-size 256 --num-eigenvalues 1 \
    --max-iterations 200 --tolerance 1e-8
```

Expected output: a printed dominant eigenvalue, residual implicitly under
the tolerance, and a timing summary (`t_compute`, `t_comm`, `t_io`).

### 3.5 Single-rank, no MPI launcher

The binary still requires MPI initialization, but you can run it directly
under `mpirun -np 1`:

```bash
mpirun -np 1 ./build/eigensolver_cpu --matrix-size 128 --num-eigenvalues 4
```

---

## 4. Scaling Studies

### 4.1 Strong scaling

Fixed problem size, growing rank count:

```bash
N=16384 SOLVER=lanczos ./scripts/run_strong_scaling.sh
```

Submits jobs for `ranks ∈ {1, 2, 4, 8, 16, 32, 64}`. Output files land in
`results/strong_n{N}_p{ranks}.out`.

### 4.2 Weak scaling

Fixed rows per rank, growing problem size:

```bash
ROWS_PER_RANK=4096 ./scripts/run_weak_scaling.sh
```

Submits jobs for `ranks ∈ {1, 4, 16, 64}` with `N = ROWS_PER_RANK * ranks`.

### 4.3 CPU-only vs hybrid comparison

Build both binaries, then submit pairs of jobs with the same parameters:

```bash
make cpu && make
mpirun -np 16 ./build/eigensolver_cpu --matrix-size 4096 --num-eigenvalues 10
mpirun -np 16 ./build/eigensolver     --matrix-size 4096 --num-eigenvalues 10
```

Compare the `t_compute` lines from each `=== Solver stats ===` block.

---

## 5. Output Format

Each run prints (rank 0):

```
=== Eigensolver configuration ===
  solver       : lanczos
  matrix       : dense-sym
  ...
=== Lanczos eigenvalues (top K) ===
  lambda[0] = ...
  ...
=== Solver stats ===
  iterations   : 153
  converged    : yes
  t_compute    : 1.234567 s    # GPU/CPU compute kernels
  t_comm       : 0.123456 s    # MPI Allreduce/Allgather
  t_io         : 0.012345 s    # MPI I/O reads/writes
  t_total      : 1.370368 s
```

Eigenvalues are also written to `--output PATH` (one per line, ASCII).

---

## 6. Input Matrix Format (for `--matrix file`)

Binary layout, native endianness:

```
[int32 rows] [int32 cols]
[double A[0,0]] [double A[0,1]] ... [double A[rows-1,cols-1]]
```

Rows must equal columns. The MPI I/O reader (`io_read_dense_matrix`) splits
the file into block-row chunks per rank.

A small generator can be added under `scripts/` if needed; the built-in
generators (`dense`, `laplacian2d`) are sufficient for the rubric's
benchmark matrices.

---

## 7. Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `nvcc: command not found` during `make` | Use `make cpu`, or `module load xl_r spectrum-mpi cuda` (on AiMOS, `xl_r` must come first) |
| `cuda ... cannot be loaded as requested` (Lmod) | On AiMOS, load `xl_r` first: `module load xl_r spectrum-mpi cuda` |
| `mpicc: command not found` | Install OpenMPI / load the MPI module (on AiMOS: `spectrum-mpi`) |
| sbatch job never runs / pending forever | Check `--partition=el8-rpi` is set and `--time` is `<= 00:30:00` (class limit) |
| `warning: N is not divisible by ranks` | Tail rank takes remainder; for clean scaling pick `N % ranks == 0` |
| Lanczos returns garbage eigenvalues | Increase `--max-iterations`, lower `--tolerance`, or check matrix is symmetric (use `--solver arnoldi` for non-symmetric) |
| `clock_now()` returns wall-clock on local builds | Expected — POWER9 timebase only available on AiMOS; the macOS/Linux fallback uses `CLOCK_MONOTONIC` scaled to the same units |

---

## 8. Repository Layout

```
.
├── INSTRUCTIONS.md       # this file
├── README.md             # project overview / design doc
├── Makefile              # build system (gpu/cpu targets)
├── src/
│   ├── main.cu           # CLI entry point, MPI/CUDA init
│   ├── eigensolver.h     # public API
│   ├── solver_internal.h # internal helpers shared by .cu files
│   ├── lanczos.cu        # Lanczos algorithm
│   ├── arnoldi.cu        # Arnoldi iteration
│   ├── power_iteration.cu# power method (baseline / validation)
│   ├── matvec.cu         # GPU dense + sparse SpMV kernels
│   ├── orthogonalize.cu  # modified Gram-Schmidt + parallel norm
│   ├── solver_common.c   # CPU fallback matvecs, allgather wrapper
│   ├── tridiag.c         # tridiagonal/Hessenberg eigenvalue solvers
│   ├── mpi_comm.{c,h}    # timed MPI wrappers
│   ├── io.{c,h}          # MPI I/O for matrices + checkpoints
│   ├── utils.{c,h}       # arg parsing, matrix generation, reporting
│   ├── kernels.h         # GPU workspace API
│   └── clockcycle.h      # POWER9 timebase + portable fallback
├── scripts/
│   ├── submit.sh             # single-job SLURM submission
│   ├── run_strong_scaling.sh # strong-scaling sweep
│   └── run_weak_scaling.sh   # weak-scaling sweep
├── data/                 # input matrices (gitignored at runtime)
└── results/              # solver output / SLURM logs (created on demand)
```
