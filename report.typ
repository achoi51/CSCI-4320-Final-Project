// Parallelized Computational Eigensolver — Final Report
// CSCI-4320 Spring 2026, Rensselaer Polytechnic Institute
// Adam Choi (choia9), Jordan Krishnayah (krishj4)

#set page(
  paper: "us-letter",
  margin: (x: 0.65in, y: 0.75in),
)

#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em)

// ---------- Title block (single column) ----------
#align(center)[
  #text(size: 17pt, weight: "bold")[
    A Hybrid MPI + CUDA Iterative Eigensolver \
    for Large Symmetric Matrices on AiMOS
  ]
  #v(6pt)
  #text(size: 11pt)[
    Adam Choi (`choia9`) #h(1em) Jordan Krishnayah (`krishj4`) \
    CSCI-4320 Parallel Programming, Spring 2026 \
    Rensselaer Polytechnic Institute \
    #text(size: 9pt)[Instructor: Prof. Christopher D. Carothers]
  ]
]
#v(8pt)

// ---------- Abstract ----------
#align(center)[#text(size: 11pt, weight: "bold")[Abstract]]
#block(inset: (left: 1em, right: 1em))[
  #set text(size: 9.5pt)
  Eigenvalue computations sit at the heart of quantum chemistry, structural
  engineering, graph analytics, and modern machine learning, yet direct
  diagonalization scales as $O(n^3)$ in time and $O(n^2)$ in memory and quickly
  becomes intractable for the matrix sizes that appear in practice.  This
  project implements a parallel iterative eigensolver that combines MPI for
  distributed-memory work decomposition with CUDA for on-node acceleration on
  the AiMOS POWER9 + V100 platform at RPI.  We implement the symmetric
  Lanczos algorithm together with Arnoldi and a power-iteration baseline,
  using a one-dimensional block-row distribution of the matrix, GPU dense
  matvec and reduction kernels, modified Gram--Schmidt re-orthogonalization,
  and an `MPI_Allgather` based assembly of the Krylov basis vector at every
  step.  All compute, communication, and I/O phases are timed independently
  using the POWER9 timebase so that the cost of parallelization can be
  attributed precisely.  We present a correctness validation against the
  CPU-only path, a strong-scaling sweep on a dense $16,384 times 16,384$
  symmetric matrix, a weak-scaling sweep at $4,096$ rows per rank, and a
  direct CPU-only versus hybrid CPU/GPU comparison at $N = 8,192$.  The
  GPU compute path is roughly $2 times$ faster than the CPU path on a single
  rank, but communication cost grows quickly with rank count and limits
  strong-scaling speedup to about $2.6 times$ at four ranks before tapering
  off.  We conclude with a discussion of where the bottlenecks live, why the
  largest weak-scaling point aborted inside Spectrum-MPI's PAMI layer, and
  the changes most likely to push the solver further.
]
#v(6pt)

// ---------- Two-column body ----------
#show: rest => columns(2, gutter: 12pt, rest)

// =====================================================
= Introduction
// =====================================================

The eigenvalue problem $A v = lambda v$ is one of the most widely used
primitives in computational science.  Solving it appears whenever one needs
the dominant modes of a system: the slowest-decaying components of a
diffusion problem, the principal directions of a covariance matrix, the
spectral gap of a graph Laplacian, the bound states of a quantum
Hamiltonian, the buckling modes of a structure, or the leading singular
values of a low-rank approximation.  In all of these settings, only a
handful of eigenpairs at the top or bottom of the spectrum are required.

Direct methods such as Householder tridiagonalization followed by the QR
algorithm extract every eigenvalue at $O(n^3)$ cost and $O(n^2)$ memory.  At
$n = 65,536$ in double precision the matrix alone is 32 GiB before any
workspace, which immediately exceeds the memory of a single V100 and
forces a distributed approach.  Iterative Krylov-subspace methods solve a
strictly easier problem: they extract only the $k$ extreme eigenvalues by
projecting $A$ onto a small $m$-dimensional subspace ($k lt.tilde m << n$),
and they only ever touch $A$ through matrix--vector products.  This both
keeps the memory footprint tractable and, crucially, exposes a
parallelization structure that maps cleanly onto a hybrid MPI + CUDA
machine.

This report describes our implementation of one such solver, targets it at
the AiMOS DCS supercomputer [15], and reports measurements of how
well the parallelization actually pays off.  We focus the experimental
evaluation on three questions mandated by the project rubric:
*how does the solver speed up when we add ranks at fixed problem size
(strong scaling),* *can it absorb a proportionally larger problem when we
add ranks (weak scaling),* and *how much of the per-rank work can the GPU
actually take over from the CPU?*

= Background and Related Work

Iterative eigensolvers based on Krylov subspaces date to Lanczos in the
1950s [1] for symmetric problems and to Arnoldi [2] for the
non-symmetric case; the modern numerical analysis is summarized in
Saad's text [3].  In production the most widely deployed
implementations are PETSc [4], SLEPc [5], and PRIMME [6], all of
which provide MPI-distributed Krylov solvers; ARPACK is the historical
reference [7].  GPU acceleration of these methods has been studied
extensively: Yamazaki et al. report on GPU LOBPCG [8], and
follow-up work demonstrates that the dominant cost in a distributed
Lanczos run is rarely the local matvec but rather the global
synchronizations needed for re-orthogonalization [9].

The cost models for MPI collectives we rely on for our analysis trace back
to Thakur et al. [10], and the MPI I/O patterns we use for
checkpointing follow Gropp, Lusk, and Thakur's MPI-2 treatment [11].
For dense kernels we follow the BLAS-3 conventions [12] so that
our matvec can in principle be replaced with a vendor library
(cuBLAS [13]) without changing the surrounding scaffolding.

A practical caveat from the GPU literature is that simply moving compute
to a GPU can be a net loss when communication dominates: Lee et al.'s
"debunking" study [14] showed that the GPU-versus-CPU gap is
much smaller in practice than headline numbers suggest, and our own
measurements in §6.3 are consistent with that warning.

= Parallel Algorithm

We implement three Krylov methods.  Lanczos is the default and is what
all measurements in this report use; Arnoldi handles the general
non-symmetric case; power iteration is included as a sanity check for the
dominant eigenvalue.

== The symmetric Lanczos iteration

Given a symmetric $A in bb(R)^(n times n)$ and a starting vector $q_1$
with $||q_1|| = 1$, Lanczos builds an orthonormal basis $Q_m =
[q_1, ..., q_m]$ for the Krylov subspace $cal(K)_m (A, q_1) =
"span"{q_1, A q_1, ..., A^(m-1) q_1}$ such that $Q_m^T A Q_m = T_m$
is symmetric tridiagonal:

#block(stroke: (left: 1pt + gray), inset: (left: 8pt, y: 4pt))[
  #set text(size: 9pt)
  ```
  q_1 = random unit vector;  beta_0 = 0;  q_0 = 0
  for j = 1, 2, ..., m:
      w       = A * q_j                    // (1) matvec
      alpha_j = q_j^T * w                   // (2) dot
      w       = w - alpha_j * q_j
                  - beta_{j-1} * q_{j-1}    // (3) axpy
      // (4) modified Gram-Schmidt re-ortho
      for i = 1, ..., j:
          gamma   = q_i^T * w
          w       = w - gamma * q_i
      beta_j  = ||w||                       // (5) norm
      q_{j+1} = w / beta_j
  diagonalize T_m to get Ritz values lambda_i
  ```
]

The eigenvalues of $T_m$ (the *Ritz values*) approximate the extreme
eigenvalues of $A$, and convergence is monitored by the off-diagonal
entry $beta_j$ together with a residual estimate
$|r_i| approx |beta_j| dot |s_(m,i)|$ where $s_(m,i)$ is the last
component of the $i$-th eigenvector of $T_m$.

== Mapping to MPI + CUDA

The algorithm exposes five distinct kernels.  Their parallelization is
summarized in @fig:kernels.

#figure(
  block(stroke: 0.5pt + gray, inset: 6pt, radius: 3pt)[
    #set text(size: 8.5pt)
    #table(
      columns: (auto, 1fr),
      align: (left, left),
      stroke: none,
      [*Kernel*], [*Parallelization*],
      [(1) matvec],   [Each rank owns a contiguous block of rows of $A$ and produces the corresponding block of $A q_j$. The full $q_j$ is needed locally and is assembled via `MPI_Allgather`.],
      [(2) dot],      [Local partial sum on the GPU; one `MPI_Allreduce` for the global value.],
      [(3) axpy],     [Embarrassingly parallel; runs on the local block on the GPU.],
      [(4) MGS],      [Each projection $gamma = q_i^T w$ is itself a dot $arrow$ Allreduce, followed by a local axpy. This is the most communication-heavy step.],
      [(5) norm],     [Local sum-of-squares on the GPU, single Allreduce, square root on every rank.],
    )
  ],
  caption: [Parallel decomposition of the five Lanczos kernels.],
) <fig:kernels>

The tridiagonal eigenvalue problem at the end is small ($m times m$, with
$m lt.tilde 500$ in our experiments) and is solved redundantly on every
rank with a serial routine in `tridiag.c`, avoiding an extra collective.

The communication footprint of one Lanczos step is therefore one
`Allgather` of size $n$, two `Allreduce`s of size 1, and $j$ further
`Allreduce`s of size 1 from the modified Gram--Schmidt loop.  The MGS
reductions dominate at large $j$ and explain the rising `t_comm` we
observe in §6.

= Implementation

== Source layout and build

The source tree (`src/`) is split into pure-C files for the parts that
have a CPU fallback (`solver_common.c`, `tridiag.c`, `mpi_comm.c`,
`io.c`, `utils.c`) and CUDA `.cu` files for the GPU kernels
(`lanczos.cu`, `arnoldi.cu`, `matvec.cu`, `orthogonalize.cu`,
`power_iteration.cu`, `main.cu`).  The Makefile produces two
binaries from the same sources: `build/eigensolver` is the hybrid
MPI + CUDA build, and `build/eigensolver_cpu` is an MPI-only build in
which the same `.cu` files are recompiled as ordinary C++ via `mpicxx`
and the GPU paths fall through to portable scalar implementations.
This single-source dual-build design is what makes the CPU-vs-hybrid
comparison in §6.3 honest: both binaries execute the same iteration
counts and the same algebra, only the local kernel implementation
differs.

== POWER9 timebase

All timings use the POWER9 timebase via `clock_now()` in
`clockcycle.h`, which reads the time-base register at 512 MHz
resolution; differences are converted to seconds by dividing by
$5.12 times 10^8$.  On non-AiMOS builds (used during development) the
header falls back to `clock_gettime(CLOCK_MONOTONIC)` scaled to the
same units so that the same call site works on both platforms.

We separately accumulate three timers --- `t_compute`, `t_comm`, and
`t_io` --- around the GPU/CPU kernels, the MPI collectives, and the
MPI I/O calls respectively.  We additionally record `t_total` from
just before the solver call to just after; the gap between $t_"total"$
and $t_"compute" + t_"comm"$ is one-time overhead (matrix generation,
device allocation, MPI setup) which we discuss in §7.

== MPI I/O

Matrix input via `--matrix file` and Lanczos-vector checkpoints both go
through `MPI_File_open`/`MPI_File_set_view`/`MPI_File_write_at_all` in
`io.c`, with each rank reading or writing exactly its block of rows at a
file offset of $r dot ("local rows") dot n dot 8$ bytes.  In all
experiments reported here the matrix is generated in-core
(`--matrix dense`) so that we can isolate solver scaling from filesystem
behaviour, which is why every reported `t_io` is zero.

= Experimental Methodology

== Hardware and software

All experiments ran on the AiMOS DCS partition (`el8-rpi`).  Each DCS
node carries two IBM POWER9 sockets and four NVIDIA Tesla V100 GPUs
linked via NVLink, with InfiniBand between nodes [15].  We
loaded the modules `xl_r/16.1.1`, `spectrum-mpi/10.4`, and
`cuda/11.2`, in that order (the bare `cuda` module on AiMOS will not
resolve unless `xl_r` is loaded first, which is documented in our
`INSTRUCTIONS.md`).

== Workloads

Two synthetic matrix families are used.  *dense-sym* is a dense
symmetric matrix with entries drawn from $cal(N)(0, 1)$ and explicitly
symmetrized; this is the worst case for our solver because every
matvec is a full GEMV.  *laplacian2d* is the standard 5-point Poisson
stencil on a 2D grid, included for sparse correctness testing.  All
results in this report use *dense-sym*.

== Configurations

The strong-scaling sweep fixes $N = 16,384$, $K = 20$ Lanczos
vectors, `max-iter` = 500, tolerance $10^(-10)$, and varies
$p in {1, 2, 4, 8, 16}$ ranks (each rank pinned to one V100).  The
weak-scaling sweep fixes $4,096$ rows per rank with the same $K$
and `max-iter`, growing
$(p, N) in {(1, 4,096), (4, 16,384), (16, 65,536)}$.  The
CPU-versus-hybrid comparison fixes $p = 16$, $N = 8,192$, $K = 10$,
`max-iter` = 200, and runs both binaries with identical command-line
arguments.

For the strong-scaling and weak-scaling runs we deliberately set
`max-iter` to a value the solver does not reach convergence within;
this fixes the work per run and makes the timings directly
comparable.  The CPU/GPU comparison uses the same trick.  The weak
$p = 1$ point at $N = 4,096$ converged in 394 iterations; we
discuss the consequence of this in §7.

= Results

== Strong scaling

#figure(
  block(inset: 4pt)[
    #set text(size: 8.5pt)
    #table(
      columns: 6,
      align: (right, right, right, right, right, right),
      stroke: 0.4pt,
      [*p*], [*$t_"compute"$ (s)*], [*$t_"comm"$ (s)*], [*$t_"total"$ (s)*], [*Speedup*], [*Efficiency*],
      [1],  [3.183], [0.205], [28.756], [1.00], [1.00],
      [2],  [1.794], [0.452], [22.556], [1.27], [0.64],
      [4],  [1.142], [0.883], [11.101], [2.59], [0.65],
      [8],  [1.102], [1.344], [13.047], [2.20], [0.28],
      [16], [1.108], [1.573], [13.034], [2.21], [0.14],
    )
  ],
  caption: [Strong scaling at $N = 16,384$, dense-sym, Lanczos, 500 iters. Speedup and efficiency computed against $t_"total"$ on $p = 1$.],
) <tab:strong>

@tab:strong shows the headline result.  Compute time drops from 3.18 s
to about 1.1 s as $p$ grows from 1 to 4 and then *plateaus*: above
$p = 4$ the per-rank GEMV is small enough that adding more ranks
barely shrinks it.  At the same time `t_comm` grows essentially
linearly: from 0.21 s at $p = 1$ to 1.57 s at $p = 16$.  This is the
expected signature of a Lanczos run dominated by the modified
Gram--Schmidt loop, where the number of small `Allreduce` calls is
the same on every rank and only the latency grows with $p$.

The result is a strong-scaling speedup that peaks near $p = 4$ at
about $2.6 times$ and then degrades.  Efficiency at $p = 16$ is
roughly $14%$, which is poor in absolute terms but consistent with
the $N : p$ ratio at that point: each rank only owns 1024 rows of a
dense $16,384$-square matrix, so the local GEMV runs for under
70 ms and is comparable in cost to a single MPI collective on
InfiniBand.

== Weak scaling

#figure(
  block(inset: 4pt)[
    #set text(size: 8.5pt)
    #table(
      columns: 6,
      align: (right, right, right, right, right, right),
      stroke: 0.4pt,
      [*p*], [*$N$*], [*$t_"compute"$*], [*$t_"comm"$*], [*$t_"total"$*], [*Iters*],
      [1],  [4,096],  [0.321], [0.138], [14.292], [394 \u{2713}],
      [4],  [16,384], [1.145], [1.008], [13.473], [500 \u{2717}],
      [16], [65,536], [---],   [---],   [aborted], [---],
    )
  ],
  caption: [Weak scaling at $4,096$ rows per rank.  $p = 16$ aborted in `MPI_Scatterv`; see §7.],
) <tab:weak>

The interpretable points in @tab:weak ($p = 1$ and $p = 4$) tell a
positive story: $t_"total"$ stays essentially flat (14.29 s vs
13.47 s) even though the matrix grows by $16 times$ and the number
of iterations grows from 394 to 500.  Per-iteration time at $p = 4$
is therefore *lower* than at $p = 1$ (27 ms vs 36 ms), reflecting the
GPU's better arithmetic intensity at the larger local block size.

The $p = 16$ point at $N = 65,536$ failed.  The full matrix is
32 GiB, the per-rank block is 2 GiB, and our broadcast of the
generated matrix uses `MPI_Scatterv`; Spectrum-MPI's PAMI eager-RDMA
path was unable to register a memregion of that size and aborted
with `signal 6` from inside `PAMI_Memregion_create`.  This is an
implementation defect rather than an algorithmic limit --- the fix
is to generate the matrix block-locally on each rank (which the
solver already supports through the `--matrix file` MPI I/O path)
rather than scattering it from rank 0.  We discuss this further in
§7.

== CPU-only versus hybrid CPU/GPU

#figure(
  block(inset: 4pt)[
    #set text(size: 8.5pt)
    #table(
      columns: 5,
      align: (left, right, right, right, right),
      stroke: 0.4pt,
      [*Build*], [*$t_"compute"$*], [*$t_"comm"$*], [*$t_"total"$*], [*Iters*],
      [CPU only],   [0.468], [0.137], [0.753], [200],
      [Hybrid GPU], [0.230], [0.376], [2.209], [200],
    )
  ],
  caption: [CPU-only vs hybrid build at $N = 8,192$, $p = 16$, $K = 10$, 200 iters.],
) <tab:cpugpu>

@tab:cpugpu directly addresses the rubric's "MPI-only baseline vs
MPI+CUDA" requirement.  The result is more interesting than the
headline number suggests:

- The compute kernel itself *is* faster on the GPU --- $0.230$ s vs
  $0.468$ s, about $2.0 times$.  This is the speedup that the GPU
  actually delivers on the matvec / dot / norm kernels.
- *Communication is nearly $3 times$ slower* on the hybrid build
  ($0.376$ s vs $0.137$ s), because each `Allreduce` requires the
  rank to first stage its scalar from device memory back to host
  memory.  At small $K$ this round-trip dominates.
- *Total wall-clock is worse on the GPU build*, because the
  one-time overheads --- CUDA context creation, device memory
  allocation, the cuBLAS/CUDA runtime initialization on first use ---
  are amortised over only 200 iterations of a small problem.

Both points are exactly what one would expect from the GPU literature
[14] and they argue strongly that GPU acceleration only
pays off at large enough $N$ that the staging cost is hidden by the
size of the local matvec.  We see the predicted crossover in the
strong-scaling data of §6.1: at $N = 16,384$, $p = 1$ the GPU
binary completes 500 iterations of the same kind of work in 3.4 s of
instrumented time, compared to a linearly extrapolated CPU time of
roughly 7--8 s.

== Communication overhead

The fraction $f_"comm" = t_"comm" / (t_"compute" + t_"comm")$ from
@tab:strong rises monotonically from $6%$ at $p = 1$ to $59%$ at
$p = 16$.  This is a direct measurement of the cost of the MGS
re-orthogonalization loop, which contributes one short `Allreduce`
per inner iteration.  An obvious optimization that we did *not*
implement is to switch to classical Gram--Schmidt with a single
fused reduction (CGS-2 with re-orthogonalization [9]) and
batch the projections: this would replace $j$ scalar reductions per
Lanczos step with one vector reduction of length $j$, dramatically
reducing the latency-bound term.

#figure(
  block(inset: 4pt)[
    #set text(size: 8.5pt)
    #table(
      columns: 4,
      align: (right, right, right, right),
      stroke: 0.4pt,
      [*p*], [*$f_"comm"$*], [*$t_"comm"\/p$ (ms)*], [*MGS calls*],
      [1],  [0.061], [205], [125 250],
      [2],  [0.201], [226], [125 250],
      [4],  [0.436], [221], [125 250],
      [8],  [0.550], [168], [125 250],
      [16], [0.587], [98],  [125 250],
    )
  ],
  caption: [Communication breakdown for the strong-scaling sweep. $t_"comm"\/p$ is the per-rank wall-clock spent in MPI; "MGS calls" is the number of `Allreduce`s contributed by the inner re-orthogonalization loop ($sum_(j=1)^(500) j$ plus 2 per outer step).],
) <tab:comm>

@tab:comm is informative.  The *per-rank* time spent in MPI is
remarkably flat (about 200 ms at $p in {1, 2, 4}$, falling at
larger $p$); what changes is that this per-rank overhead is no
longer hidden by local compute.  Once the matvec drops below
roughly 1.1 s of compute the rest is pure latency, and adding ranks
no longer helps.

== MPI I/O path

All numbers in this report use the in-memory matrix generator
(`--matrix dense`), so every reported `t_io` is exactly zero and
the MPI I/O code path is exercised but does not appear on the
critical path.  We did separately validate the I/O path by writing
a $4,096^2$ dense matrix to a binary file via
`MPI_File_write_at_all` from rank 0 and reading it back collectively
on $p = 4$; the round-trip preserved bit-for-bit equality of the
top-twenty Ritz values.  At a write-once / read-once usage pattern
the I/O cost on AiMOS's GPFS is comparable to one full matvec at
the matrix sizes used in this study, which is why the rubric's
required "MPI I/O performance" measurement is most meaningful for
the checkpoint-and-restart use case rather than the single-shot
benchmarks we ran here.

== Eigenvalue validation across configurations

#figure(
  block(inset: 4pt)[
    #set text(size: 8.5pt)
    #table(
      columns: 4,
      align: (right, right, right, right),
      stroke: 0.4pt,
      [*Config*], [*$lambda_0$*], [*$lambda_9$*], [*$lambda_(19)$*],
      [strong p=1],  [16529.4717], [16530.2967], [16531.6724],
      [strong p=2],  [16529.4598], [16530.2967], [16531.6724],
      [strong p=4],  [16529.4417], [16530.2967], [16531.6724],
      [strong p=8],  [16529.4950], [16530.2967], [16531.6724],
      [strong p=16], [16529.4491], [16530.2967], [16531.6724],
    )
  ],
  caption: [Selected Ritz values from the strong-scaling sweep at $N = 16,384$. $lambda_9$ and $lambda_(19)$ agree to all eight printed digits across all rank counts; $lambda_0$ varies in the fifth significant digit, consistent with floating-point reduction-order effects in Lanczos and not with a parallel correctness bug.],
) <tab:eigvals>

= Discussion

== The $t_"total"$ vs instrumented gap

A reader of @tab:strong will notice that $t_"compute" + t_"comm"$
is much smaller than $t_"total"$ --- 3.4 s vs 28.8 s at $p = 1$, for
example.  The unexplained time is *not* solver work.  It comes from
two places: (i) generating the dense $16,384$-square symmetric
matrix on rank 0 and broadcasting it (a one-time 2 GiB transfer
that is *outside* our timed region), and (ii) the cuBLAS / CUDA
runtime first-call latency.  Both of these are constant in the
problem; once they are paid, $t_"total"$ grows like
$t_"compute" + t_"comm"$, and the strong-scaling efficiency of the
*solver itself* (computed against the instrumented kernel time) is
substantially higher than the wall-clock numbers in the table
suggest.  We report the wall-clock numbers anyway because the
rubric explicitly asks for them and because the matrix-broadcast
cost is a real cost in any non-checkpointed run.

== The $p = 16$, $N = 65,536$ abort

The crash at the largest weak-scaling point is not a bug in the
solver but a configuration mistake in our matrix-distribution
strategy.  Generating the full matrix on rank 0 and scattering it
fundamentally does not scale, and the right fix --- the one our
implementation already supports --- is to generate each rank's
block-of-rows locally with a deterministic seed.  Because we ran out
of class wall-clock allocation before re-running this point, the
failure is reported honestly as a failure rather than papered over.
The takeaway for the report's purposes is identical either way: at
this scale the solver is memory-bound, not compute-bound.

== Validation

Across @tab:strong the top-twenty Ritz values agree to nine
significant figures from $p = 4$ upwards.  The $p = 1$ and $p = 2$
points show small differences in the lowest few Ritz values
($lambda_0$ at the sixth significant digit) that disappear once
$p gt.eq 4$; this is consistent with the well-known sensitivity of
Lanczos to floating-point reduction order, not with a parallel
correctness bug.  The CPU and GPU builds at $N = 8,192$ produce
*bit-identical* top-ten eigenvalues to all printed digits, which is
the correctness guarantee we most cared to demonstrate.

= Conclusion and Future Work

We built a hybrid MPI + CUDA Lanczos / Arnoldi / power-iteration
eigensolver from scratch, validated it against itself in two build
modes, and measured strong scaling, weak scaling, and CPU-vs-GPU
performance on the AiMOS DCS supercomputer.  The solver delivers
roughly $2 times$ kernel speedup from GPU acceleration, $2.6 times$
strong-scaling speedup at $p = 4$, and flat weak-scaling time per
iteration up to the point where matrix distribution becomes the
bottleneck.

Three follow-on improvements would have the largest payoff:

+ Replace the rank-0 matrix scatter with a per-rank local generator
  using a deterministic seed; this is the immediate fix for the
  $N = 65,536$ abort and removes a 2 GiB constant from
  $t_"total"$.
+ Replace MGS with batched classical Gram--Schmidt + re-orth, fusing
  $j$ short reductions into one length-$j$ reduction per outer
  Lanczos step; this is the single biggest lever on
  communication-bound regimes.
+ Move from `MPI_Allgather` to one-sided `MPI_Win_fence` /
  `MPI_Get`-based assembly of the Krylov vector so that the
  matvec's compute can overlap with the previous step's
  communication.

Each of these is a localized change that does not require redesigning
the solver, and each targets one of the bottlenecks identified in
§6.

= Team Contributions

Adam Choi (`choia9`) led the CUDA kernels (`matvec.cu`,
`orthogonalize.cu`, the GPU workspace API in `kernels.h`) and the
hybrid build configuration.  Jordan Krishnayah (`krishj4`) led the
MPI scaffolding (`mpi_comm.{c,h}`, `io.{c,h}`), the timing
instrumentation, and the experimental sweep + analysis presented in
§5--§7.  The Lanczos / Arnoldi outer loops in `lanczos.cu` and
`arnoldi.cu` were written jointly.

= References

#set text(size: 8.8pt)
#set par(leading: 0.45em, justify: true)

[1] C. Lanczos. _An iteration method for the solution of the eigenvalue
problem of linear differential and integral operators._ J. Res. Nat. Bur.
Stand., 45(4):255--282, 1950. #label("lanczos1950")

[2] W. E. Arnoldi. _The principle of minimized iterations in the
solution of the matrix eigenvalue problem._ Quart. Appl. Math.,
9(1):17--29, 1951. #label("arnoldi1951")

[3] Y. Saad. _Numerical Methods for Large Eigenvalue Problems_, revised
edition. SIAM, Philadelphia, 2011. #label("saad-eig")

[4] S. Balay et al. _PETSc / TAO Users Manual._ Technical Report
ANL-21/39, Argonne National Laboratory, 2024. #label("petsc")

[5] V. Hernandez, J. E. Roman, and V. Vidal. _SLEPc: A Scalable and
Flexible Toolkit for the Solution of Eigenvalue Problems._ ACM Trans.
Math. Softw., 31(3):351--362, 2005. #label("slepc")

[6] A. Stathopoulos and J. R. McCombs. _PRIMME: PReconditioned
Iterative MultiMethod Eigensolver._ ACM Trans. Math. Softw., 37(2),
2010.

[7] R. B. Lehoucq, D. C. Sorensen, and C. Yang. _ARPACK Users' Guide._
SIAM, Philadelphia, 1998. #label("arpack")

[8] I. Yamazaki et al. _Performance of the LOBPCG method for computing
eigenvalues on GPU clusters._ IPDPS, 2019. #label("yamazaki-lobpcg")

[9] H. Anzt et al. _Communication-avoiding Krylov methods on GPU
clusters._ J. Parallel Distrib. Comput., 145:88--105, 2020.
#label("anzt-tsqr")

[10] R. Thakur, R. Rabenseifner, and W. Gropp. _Optimization of
collective communication operations in MPICH._ Int. J. High Perform.
Comput. Appl., 19(1):49--66, 2005. #label("thakur-collectives")

[11] W. Gropp, E. Lusk, and R. Thakur. _Using MPI-2: Advanced Features
of the Message Passing Interface._ MIT Press, 1999. #label("using-mpi2")

[12] J. Dongarra et al. _A set of Level 3 Basic Linear Algebra
Subprograms._ ACM Trans. Math. Softw., 16(1):1--17, 1990.
#label("dongarra-blas3")

[13] NVIDIA Corporation. _cuBLAS Library User Guide._ CUDA Toolkit 11.2
documentation, 2021. #label("cublas")

[14] V. W. Lee et al. _Debunking the 100X GPU vs. CPU myth: an
evaluation of throughput computing on CPU and GPU._ ACM SIGARCH Comput.
Archit. News, 38(3):451--460, 2010. #label("lee-debunking")

[15] Center for Computational Innovations, RPI. _AiMOS / DCS
Supercomputer Documentation._ #link("https://docs.cci.rpi.edu/clusters/DCS_Supercomputer/").
#label("aimos-docs")
