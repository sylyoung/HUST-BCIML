#!/bin/bash
# Algorithm-coverage sweep on BNCI2014001 (cross-subject LOSO), one GPU, sequential.
# Usage: sweep_bnci.sh GPU_ID ITR   (ITR = number of seeds, seeds start at 1)
# Writes results to /home/sylyoung/hustbciml_results/*/metrics.json and prints a
# SWEEP_DONE marker at the end (for the completion monitor).
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

echo "===== EA-EEGNet (ERM baseline) ====="
python -m hustbciml.run --algorithm EA-EEGNet $COMMON
echo "===== NoAlign-EEGNet (EA ablation) ====="
python -m hustbciml.run --aligner Identity --backbone EEGNet --head Linear --strategy ERM $COMMON
echo "===== EA-DANN (domain-adversarial) ====="
python -m hustbciml.run --algorithm EA-DANN $COMMON
echo "===== T-TIME (test-time adaptation) ====="
python -m hustbciml.run --algorithm T-TIME $COMMON
echo "===== EA-ShallowConvNet (ERM) ====="
python -m hustbciml.run --aligner EA --backbone ShallowConvNet --head Linear --strategy ERM $COMMON
echo "===== EA-DeepConvNet (ERM) ====="
python -m hustbciml.run --aligner EA --backbone DeepConvNet --head Linear --strategy ERM $COMMON
echo "SWEEP_DONE"
