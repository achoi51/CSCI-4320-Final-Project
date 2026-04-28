#!/bin/bash
#SBATCH --partition=el8-rpi
#SBATCH --time=00:30:00
#
# Parameterized SBATCH script used by the strong/weak scaling sweeps.
# Resource counts (--nodes, --ntasks-per-node, --gres, --job-name,
# --output, --error) are passed on the sbatch command line; problem
# parameters come in as env vars via --export.
#
# Required env vars:
#   RANKS      total MPI ranks
#   N          matrix size
#   NEV        number of eigenvalues
#   TOL        tolerance
#   MAXIT      max iterations
#   SOLVER     lanczos | arnoldi | power
#   MATRIX     dense | laplacian2d | file
#   RUN_TAG    short label used in the output .dat filename

set -euo pipefail

module load xl_r spectrum-mpi cuda

mkdir -p results

EXE=./build/eigensolver
[ -x "$EXE" ] || EXE=./build/eigensolver_cpu

echo "host       : $(hostname)"
echo "job id     : ${SLURM_JOB_ID:-n/a}"
echo "ntasks     : ${SLURM_NTASKS:-$RANKS}"
echo "executable : $EXE"
echo "params     : RANKS=$RANKS N=$N NEV=$NEV SOLVER=$SOLVER MATRIX=$MATRIX"
module list 2>&1 || true
echo

mpirun -np "$RANKS" "$EXE" \
    --solver         "$SOLVER" \
    --matrix         "$MATRIX" \
    --matrix-size    "$N" \
    --num-eigenvalues "$NEV" \
    --tolerance      "$TOL" \
    --max-iterations "$MAXIT" \
    --output         "results/${RUN_TAG}.dat"
