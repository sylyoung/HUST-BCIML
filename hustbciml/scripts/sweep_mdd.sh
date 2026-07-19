#!/bin/bash
# Delta run: MDD (margin disparity discrepancy) on BNCI2014001 cross-subject LOSO.
# MDD was added to the gradient-DA family after the 4-method sweep_da.sh had
# already run on the server, so this runs just MDD into the shared results dir
# without recomputing MCC / CDAN / DAN / JAN. Runs on one GPU.
# Usage: sweep_mdd.sh GPU_ID ITR   (ITR = number of seeds, seeds start at 1)
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

echo "===== MDD (margin disparity discrepancy) ====="
python -m hustbciml.run --algorithm MDD $COMMON
echo "SWEEP_MDD_DONE"
