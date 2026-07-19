#!/bin/bash
# Delta run: complement the controlled-comparison tables with the three
# stage-axis methods that had no lab source and were implemented from scratch
# (faithful to their papers), on BNCI2014001 cross-subject LOSO:
#   RA-EEGNet      -> alignment axis  (Riemannian Alignment vs EA vs none)
#   CR-EEGNet      -> augmentation axis (Channel Reflection vs none; no-align)
#   EA-EEGConformer-> network axis    (Conformer vs EEGNet/Shallow/Deep)
# Added to the shared coverage results dir (seed 1) so compare.py picks them up.
# Usage: sweep_ra_cr_conformer.sh GPU_ID ITR   (ITR = seeds, starting at 1)
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

echo "===== RA-EEGNet (Riemannian Alignment; alignment axis) ====="
python -m hustbciml.run --algorithm RA-EEGNet $COMMON
echo "===== CR-EEGNet (Channel Reflection; augmentation axis, no-align) ====="
python -m hustbciml.run --algorithm CR-EEGNet $COMMON
echo "===== EA-EEGConformer (EEG Conformer backbone; network axis) ====="
python -m hustbciml.run --algorithm EA-EEGConformer $COMMON
echo "SWEEP_COMPLEMENT_DONE"
