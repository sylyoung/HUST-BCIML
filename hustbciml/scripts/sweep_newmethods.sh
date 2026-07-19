#!/bin/bash
# Full 3-dataset x 3-seed runs for the 4 new network-based transfer/privacy methods
# (MEKT, MDMAML, ASFA, SAFE). The 2 new backbones (TIE-EEGNet, KDFNet) are tuned
# separately via tune_networks (LR grid). Results share hustbciml_results_3ds so
# extract_3ds_v2.py picks them up uniformly. Resume-safe (done-markers). One GPU per
# invocation; launch several in parallel across free GPUs.
#   Usage: sweep_newmethods.sh <GPU_ID> <ALGO> [ALGO ...]
set -u
GPU=${1:?gpu id}; shift
METHODS=("$@")
PY=/home/sylyoung/micromamba/envs/syl-work/bin/python
DATA=/home/sylyoung/data
RES=/home/sylyoung/hustbciml_results_3ds
cd /home/sylyoung
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$RES"
DATASETS=(BNCI2014001 BNCI2014002 BNCI2015001)
for algo in "${METHODS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    # SAFE is a privacy/federated method: BNCI2014001 is native 4-class in the
    # privacy table (as with Centralized/FedBS/FedAvg/MSDT), keyed BNCI2014001-4.
    dsrun="$ds"
    if [ "$algo" = "SAFE" ] && [ "$ds" = "BNCI2014001" ]; then dsrun="BNCI2014001-4"; fi
    for s in 1 2 3; do
      marker="$RES/.done_new_${dsrun}_${algo}_seed${s}"
      if [ -f "$marker" ]; then echo "SKIP $dsrun $algo seed$s (done)"; continue; fi
      echo "===== $dsrun $algo seed$s [$(date '+%H:%M:%S')] ====="
      if $PY -m hustbciml.run --algorithm "$algo" --dataset "$dsrun" --device cuda \
           --seed "$s" --itr 1 --results_dir "$RES" --data_dir "$DATA"; then
        touch "$marker"
      else
        echo "FAILED $dsrun $algo seed$s (continuing)"
      fi
    done
  done
done
echo "NEWSWEEP_DONE gpu$GPU (${METHODS[*]})"
