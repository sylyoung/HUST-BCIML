#!/bin/bash
# Delta sweep: the test-time-adaptation family on BNCI2014001 (cross-subject
# LOSO), sequential on one GPU. Adds Tent / PL / BN-adapt to a results dir that
# already holds the EA-EEGNet / EA-DANN / T-TIME runs, so the leaderboard can
# rank the whole TTA family together without recomputing the earlier algorithms.
# Usage: sweep_tta.sh GPU_ID ITR   (ITR = number of seeds, seeds start at 1)
set -u
GPU=${1:-5}
ITR=${2:-1}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
cd /home/sylyoung

COMMON="--dataset BNCI2014001 --device cuda --epochs 100 --itr $ITR --seed 1 --results_dir /home/sylyoung/hustbciml_results"

echo "===== Tent (BN-affine entropy min TTA) ====="
python -m hustbciml.run --algorithm Tent $COMMON
echo "===== PL (pseudo-label TTA) ====="
python -m hustbciml.run --algorithm PL $COMMON
echo "===== BN-adapt (BatchNorm stat TTA) ====="
python -m hustbciml.run --algorithm BN-adapt $COMMON
echo "SWEEP_TTA_DONE"
