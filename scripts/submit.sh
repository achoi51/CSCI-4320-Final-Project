#!/bin/bash
#SBATCH --job-name=eigensolver
#SBATCH --partition=el8-rpi
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=results/eigensolver_%j.out
#SBATCH --error=results/eigensolver_%j.err

set -euo pipefail

module load xl_r spectrum-mpi cuda

mkdir -p results

EXE=./build/eigensolver
N=${N:-16384}
NEV=${NEV:-20}
TOL=${TOL:-1e-10}
MAXIT=${MAXIT:-500}
SOLVER=${SOLVER:-lanczos}
MATRIX=${MATRIX:-dense}

mpirun -np $SLURM_NTASKS $EXE \
    --solver        $SOLVER \
    --matrix        $MATRIX \
    --matrix-size   $N \
    --num-eigenvalues $NEV \
    --tolerance     $TOL \
    --max-iterations $MAXIT \
    --output        results/eigvals_${SLURM_JOB_ID}.dat
