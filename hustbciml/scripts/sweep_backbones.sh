#!/bin/bash
# Network-axis sweep: the 10 newly ported backbones (ADFCNN, CTNet, MSCFormer,
# MSVTNet, TMSANet, EEGWaveNet, SlimSeiz, FBMSNet, EEGNeX, EEGDeformer) each on the
# EA-aligned ERM + Linear-head setup, x 3 MOABB MI datasets x 3 seeds, cross-subject
# LOSO. Results share hustbciml_results_backbones so extract_backbones_3ds.py picks
# them up uniformly. Resume-safe (per-run done-markers). One GPU per invocation;
# launch several in parallel across free GPUs.
#   Usage: sweep_backbones.sh <GPU_ID> <ALGO-preset> [ALGO-preset ...]
set -u
GPU=${1:?gpu id}; shift
METHODS=("$@")
export LD_LIBRARY_PATH=/home/sylyoung/.conda/envs/syl-work/lib:${LD_LIBRARY_PATH:-}
PY=/home/sylyoung/.conda/envs/syl-work/bin/python
DATA=/home/sylyoung/data
RES=/home/sylyoung/hustbciml_results_backbones
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
echo "BACKBONESWEEP_DONE gpu$GPU (${METHODS[*]})"
