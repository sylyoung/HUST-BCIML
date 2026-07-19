#!/bin/bash
# Quick re-validation of the two fixed methods (MDMAML Adam meta-opt, MEKT scale
# normalization) on BNCI2014001, seed 1. MEKT_DIAG prints the init-vs-projected
# accuracy trajectory so we can see whether the projection helps.
set -u
PY=/home/sylyoung/micromamba/envs/syl-work/bin/python
DATA=/home/sylyoung/data
R2=/home/sylyoung/hustbciml_results_newsmoke2
cd /home/sylyoung
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$R2"
: > "$R2/_progress.log"

CUDA_VISIBLE_DEVICES=0 $PY -m hustbciml.run --algorithm MDMAML --dataset BNCI2014001 \
  --device cuda --seed 1 --itr 1 --results_dir "$R2/MDMAML" --data_dir "$DATA" \
  > "$R2/MDMAML.log" 2>&1 && echo "DONE MDMAML" >> "$R2/_progress.log" &

CUDA_VISIBLE_DEVICES=5 MEKT_DIAG=1 $PY -m hustbciml.run --algorithm MEKT --dataset BNCI2014001 \
  --device cuda --seed 1 --itr 1 --results_dir "$R2/MEKT" --data_dir "$DATA" \
  > "$R2/MEKT.log" 2>&1 && echo "DONE MEKT" >> "$R2/_progress.log" &

wait
echo "RERUN_DONE" >> "$R2/_progress.log"
