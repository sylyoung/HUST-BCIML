#!/bin/bash
# Delta sweep: the gradient domain-adaptation family on BNCI2014001
# (cross-subject LOSO), sequential on one GPU. Adds MCC / CDAN / DAN / JAN to a
# results dir that already holds the earlier runs (DANN is done), so the
# leaderboard ranks the whole DA family together without recomputation.
# Usage: sweep_da.sh GPU_ID ITR   (ITR = number of seeds, seeds start at 1)
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

echo "===== MCC (minimum class confusion) ====="
python -m hustbciml.run --algorithm MCC $COMMON
echo "===== CDAN (conditional domain-adversarial) ====="
python -m hustbciml.run --algorithm CDAN $COMMON
echo "===== DAN (multi-kernel MMD) ====="
python -m hustbciml.run --algorithm DAN $COMMON
echo "===== JAN (joint MMD) ====="
python -m hustbciml.run --algorithm JAN $COMMON
echo "===== MDD (margin disparity discrepancy) ====="
python -m hustbciml.run --algorithm MDD $COMMON
echo "SWEEP_DA_DONE"
