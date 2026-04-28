#!/bin/bash
#
# Strong-scaling sweep: fixed N, vary number of MPI ranks / GPUs.
# Tuned for the CSCI-4320 AiMOS environment:
#   - partition el8-rpi
#   - module set: xl_r spectrum-mpi cuda
#   - max 30 min per job, max 16 nodes per job

set -euo pipefail

mkdir -p results

N=${N:-16384}
NEV=${NEV:-20}
TOL=${TOL:-1e-10}
MAXIT=${MAXIT:-500}
SOLVER=${SOLVER:-lanczos}
MATRIX=${MATRIX:-dense}

# 64 ranks @ 4 ranks/node = 16 nodes, which is the class limit.
# Override with: RANKS_LIST="1 2 4 8" ./scripts/run_strong_scaling.sh
RANKS_LIST=${RANKS_LIST:-"1 2 4 8 16 32 64"}

for ranks in $RANKS_LIST; do
    nodes=$(( (ranks + 3) / 4 ))
    tpn=$(( ranks < 4 ? ranks : 4 ))
    sbatch \
        --job-name=eig_strong_${ranks} \
        --partition=el8-rpi \
        --nodes=$nodes \
        --ntasks-per-node=$tpn \
        --gres=gpu:$tpn \
        --time=00:30:00 \
        --output=results/strong_n${N}_p${ranks}.out \
        --error=results/strong_n${N}_p${ranks}.err \
        --wrap="module load xl_r spectrum-mpi cuda; \
                mpirun -np ${ranks} ./build/eigensolver \
                    --solver $SOLVER --matrix $MATRIX \
                    --matrix-size $N --num-eigenvalues $NEV \
                    --tolerance $TOL --max-iterations $MAXIT \
                    --output results/strong_n${N}_p${ranks}.dat"
done
