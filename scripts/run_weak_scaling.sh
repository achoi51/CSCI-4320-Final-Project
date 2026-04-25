#!/bin/bash
#
# Weak-scaling sweep: keep rows-per-rank constant (~ROWS_PER_RANK),
# grow the global problem size with the rank count.

set -euo pipefail

mkdir -p results

ROWS_PER_RANK=${ROWS_PER_RANK:-4096}
NEV=${NEV:-20}
TOL=${TOL:-1e-10}
MAXIT=${MAXIT:-500}
SOLVER=${SOLVER:-lanczos}
MATRIX=${MATRIX:-dense}

for ranks in 1 4 16 64; do
    N=$(( ROWS_PER_RANK * ranks ))
    nodes=$(( (ranks + 3) / 4 ))
    sbatch \
        --job-name=eig_weak_${ranks} \
        --nodes=$nodes \
        --ntasks-per-node=$(( ranks < 4 ? ranks : 4 )) \
        --gres=gpu:$(( ranks < 4 ? ranks : 4 )) \
        --time=01:00:00 \
        --output=results/weak_p${ranks}_n${N}.out \
        --wrap="module load gcc cuda openmpi || true; \
                mpirun -np ${ranks} ./build/eigensolver \
                    --solver $SOLVER --matrix $MATRIX \
                    --matrix-size $N --num-eigenvalues $NEV \
                    --tolerance $TOL --max-iterations $MAXIT \
                    --output results/weak_p${ranks}_n${N}.dat"
done
