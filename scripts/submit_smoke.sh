#!/bin/bash
#SBATCH --job-name=eig_smoke
#SBATCH --partition=el8-rpi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=results/smoke_%j.out
#SBATCH --error=results/smoke_%j.err

set -euo pipefail

# AiMOS / CSCI-4320 class environment.
# `cuda` is gated behind the xl_r compiler module — load xl_r first.
module load xl_r spectrum-mpi cuda

mkdir -p results

EXE=./build/eigensolver
[ -x "$EXE" ] || EXE=./build/eigensolver_cpu

echo "=== smoke test ==="
echo "host       : $(hostname)"
echo "date       : $(date)"
echo "job id     : ${SLURM_JOB_ID:-n/a}"
echo "ntasks     : ${SLURM_NTASKS:-2}"
echo "executable : $EXE"
module list 2>&1 || true
echo

mpirun -np ${SLURM_NTASKS:-2} $EXE \
    --solver        lanczos \
    --matrix        dense \
    --matrix-size   512 \
    --num-eigenvalues 4 \
    --max-iterations 100 \
    --tolerance     1e-6 \
    --output        results/smoke_eigvals_${SLURM_JOB_ID:-local}.dat

echo
echo "=== smoke test done ==="
