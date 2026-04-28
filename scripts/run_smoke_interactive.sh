#!/bin/bash
# Run this AFTER you've sshed onto the compute node (e.g. dcs24) that
# `salloc` granted you. It loads modules and runs a tiny Lanczos job.
#
#   salloc -N 1 --partition=el8-rpi --gres=gpu:4 -t 30
#   squeue -u $USER             # find dcsXYZ
#   ssh dcsXYZ
#   cd <repo>
#   ./scripts/run_smoke_interactive.sh

set -euo pipefail

module load xl_r spectrum-mpi cuda

mkdir -p results

EXE=./build/eigensolver
[ -x "$EXE" ] || EXE=./build/eigensolver_cpu

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=results/smoke_${STAMP}.out
EIG=results/smoke_eigvals_${STAMP}.dat

{
    echo "=== smoke test ==="
    echo "host       : $(hostname)"
    echo "date       : $(date)"
    echo "executable : $EXE"
    module list 2>&1 || true
    echo

    mpirun -np 2 $EXE \
        --solver        lanczos \
        --matrix        dense \
        --matrix-size   512 \
        --num-eigenvalues 4 \
        --max-iterations 100 \
        --tolerance     1e-6 \
        --output        "$EIG"

    echo
    echo "=== smoke test done ==="
} | tee "$OUT"

echo
echo "wrote: $OUT"
echo "wrote: $EIG"
