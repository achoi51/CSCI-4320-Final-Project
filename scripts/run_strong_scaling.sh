#!/bin/bash
#
# Strong-scaling sweep: fixed N, vary number of MPI ranks / GPUs.
# Each line of the SBATCH submission keeps the same problem size and
# changes the rank count.

set -euo pipefail

mkdir -p results

N=${N:-16384}
NEV=${NEV:-20}
TOL=${TOL:-1e-10}
MAXIT=${MAXIT:-500}
SOLVER=${SOLVER:-lanczos}
MATRIX=${MATRIX:-dense}

for ranks in 1 2 4 8 16 32 64; do
    nodes=$(( (ranks + 3) / 4 ))
    sbatch \
        --job-name=eig_strong_${ranks} \
        --nodes=$nodes \
        --ntasks-per-node=$(( ranks < 4 ? ranks : 4 )) \
        --gres=gpu:$(( ranks < 4 ? ranks : 4 )) \
        --time=01:00:00 \
        --output=results/strong_n${N}_p${ranks}.out \
        --wrap="module load gcc cuda openmpi || true; \
                mpirun -np ${ranks} ./build/eigensolver \
                    --solver $SOLVER --matrix $MATRIX \
                    --matrix-size $N --num-eigenvalues $NEV \
                    --tolerance $TOL --max-iterations $MAXIT \
                    --output results/strong_n${N}_p${ranks}.dat"
done
