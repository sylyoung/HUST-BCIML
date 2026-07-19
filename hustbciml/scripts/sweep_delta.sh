#!/bin/bash
# Delta run: DELTA (degradation-free fully test-time adaptation) on BNCI2014001
# cross-subject LOSO. Adds it to the shared coverage results dir (seed 1) so the
# transfer/adaptation-strategy comparison ranks it with the rest of the strategy
# family, without recomputing the earlier runs.
# Usage: sweep_delta.sh GPU_ID ITR   (ITR = number of seeds, seeds start at 1)
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

echo "===== DELTA (degradation-free fully test-time adaptation) ====="
python -m hustbciml.run --algorithm DELTA $COMMON
echo "SWEEP_DELTA_DONE"
