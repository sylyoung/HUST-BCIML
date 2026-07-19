#!/bin/bash
# Delta run: SHOT (source-free DA / SHOT-IM) + SAR (sharpness-aware TTA) on
# BNCI2014001 cross-subject LOSO. Adds them to the shared coverage results dir
# (seed 1) so the transfer/adaptation-strategy comparison ranks them with the
# rest of the strategy family, without recomputing the earlier runs.
# Usage: sweep_shot_sar.sh GPU_ID ITR   (ITR = number of seeds, seeds start at 1)
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

echo "===== SHOT (source-free DA / SHOT-IM) ====="
python -m hustbciml.run --algorithm SHOT $COMMON
echo "===== SAR (sharpness-aware TTA) ====="
python -m hustbciml.run --algorithm SAR $COMMON
echo "SWEEP_SHOT_SAR_DONE"
