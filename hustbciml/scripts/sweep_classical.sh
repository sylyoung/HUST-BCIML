#!/bin/bash
# Classical-track sweep: CSP+LDA on BNCI2014001 (cross-subject LOSO). CPU-only —
# CSP/LDA are scikit-learn/MNE, no GPU needed — thread-limited for the shared box.
# Adds to the shared results dir so the leaderboard ranks it with the rest.
# Usage: sweep_classical.sh ITR   (ITR = number of seeds, seeds start at 1)
set -u
ITR=${1:-1}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
cd /home/sylyoung

COMMON="--dataset BNCI2014001 --device cpu --itr $ITR --seed 1 --results_dir /home/sylyoung/hustbciml_results"

echo "===== CSP-LDA (classical baseline) ====="
python -m hustbciml.run --algorithm CSP-LDA $COMMON
echo "===== Riemann-MDM (minimum distance to Riemannian mean) ====="
python -m hustbciml.run --algorithm Riemann-MDM $COMMON
echo "SWEEP_CLASSICAL_DONE"
