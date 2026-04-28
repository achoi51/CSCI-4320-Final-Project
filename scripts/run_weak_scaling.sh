#!/bin/bash
#
# Weak-scaling sweep: keep rows-per-rank constant (~ROWS_PER_RANK),
# grow the global problem size with the rank count.
# Tuned for the CSCI-4320 AiMOS environment:
#   - partition el8-rpi
#   - module set: xl_r spectrum-mpi cuda
#   - max 30 min per job, max 16 nodes per job

set -euo pipefail

mkdir -p results

ROWS_PER_RANK=${ROWS_PER_RANK:-4096}
NEV=${NEV:-20}
TOL=${TOL:-1e-10}
MAXIT=${MAXIT:-500}
SOLVER=${SOLVER:-lanczos}
MATRIX=${MATRIX:-dense}

RANKS_LIST=${RANKS_LIST:-"1 4 16 64"}

for ranks in $RANKS_LIST; do
    N=$(( ROWS_PER_RANK * ranks ))
    nodes=$(( (ranks + 3) / 4 ))
    tpn=$(( ranks < 4 ? ranks : 4 ))
    sbatch \
        --job-name=eig_weak_${ranks} \
        --partition=el8-rpi \
        --nodes=$nodes \
        --ntasks-per-node=$tpn \
        --gres=gpu:$tpn \
        --time=00:30:00 \
        --output=results/weak_p${ranks}_n${N}.out \
        --error=results/weak_p${ranks}_n${N}.err \
        --wrap="module load xl_r spectrum-mpi cuda; \
                mpirun -np ${ranks} ./build/eigensolver \
                    --solver $SOLVER --matrix $MATRIX \
                    --matrix-size $N --num-eigenvalues $NEV \
                    --tolerance $TOL --max-iterations $MAXIT \
                    --output results/weak_p${ranks}_n${N}.dat"
done
