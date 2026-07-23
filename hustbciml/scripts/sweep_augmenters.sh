#!/bin/bash
# Augmentation-family sweep: the comparison-baseline augmenters from the CSDA and
# Channel Reflection papers (Noise, Flip, Scale, FShift, FSurr, FComb, HS, Symm)
# x 3 MOABB MI datasets x 3 seeds, cross-subject LOSO on the EA-aligned EEGNet+ERM
# setup. Results share hustbciml_results_aug so extract_augmenters_3ds.py picks
# them up uniformly. Resume-safe (per-run done-markers). One GPU per invocation;
# launch several in parallel across free GPUs.
#   Usage: sweep_augmenters.sh <GPU_ID> <ALGO-preset> [ALGO-preset ...]
set -u
GPU=${1:?gpu id}; shift
METHODS=("$@")
# The env python needs its own lib dir on LD_LIBRARY_PATH or torch fails to find
# the matching libcudart (calling bin/python directly does not activate the env).
export LD_LIBRARY_PATH=/home/sylyoung/.conda/envs/syl-work/lib:${LD_LIBRARY_PATH:-}
PY=/home/sylyoung/.conda/envs/syl-work/bin/python
DATA=/home/sylyoung/data
RES=/home/sylyoung/hustbciml_results_aug
cd /home/sylyoung
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$RES"
DATASETS=(BNCI2014001 BNCI2014002 BNCI2015001)
for algo in "${METHODS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for s in 1 2 3; do
      marker="$RES/.done_${ds}_${algo}_seed${s}"
      if [ -f "$marker" ]; then echo "SKIP $ds $algo seed$s (done)"; continue; fi
      echo "===== $ds $algo seed$s [$(date '+%H:%M:%S')] ====="
      if $PY -m hustbciml.run --algorithm "$algo" --dataset "$ds" --device cuda \
           --seed "$s" --itr 1 --results_dir "$RES" --data_dir "$DATA"; then
        touch "$marker"
      else
        echo "FAILED $ds $algo seed$s (continuing)"
      fi
    done
  done
done
echo "AUGSWEEP_DONE gpu$GPU (${METHODS[*]})"
