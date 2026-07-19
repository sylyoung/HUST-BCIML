#!/bin/bash
# Full production sweep: the complete algorithm set x ITR seeds on BNCI2014001
# (cross-subject LOSO), for the mean+/-std leaderboard. Writes to a FRESH results
# dir (hustbciml_results_3seed), separate from the single-seed coverage sweep.
#
# Parallel across GPUs by sharding the method list: each instance runs the methods
# whose position %% NSHARD == SHARD. Disjoint shards never touch the same method,
# so the shared done-markers are race-free. Resume-safe: each (method, seed) is a
# separate --itr 1 call guarded by a marker, so a crash/outage resumes where it
# stopped. A failed run is logged and skipped, not fatal.
#
# Usage:
#   prod_sweep_full.sh <GPU_ID> <SHARD> <NSHARD> [ITR]   # neural methods on a GPU
#   prod_sweep_full.sh cpu       0       1       [ITR]   # the 2 classical methods
set -u
GPU=${1:?gpu id or 'cpu'}; SHARD=${2:?shard idx}; NSHARD=${3:?num shards}; ITR=${4:-3}
RESULTS=/home/sylyoung/hustbciml_results_3seed
DATA=/home/sylyoung/data
PY=/home/sylyoung/micromamba/envs/syl-work/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
[ "$GPU" != "cpu" ] && export CUDA_VISIBLE_DEVICES=$GPU
cd /home/sylyoung
mkdir -p "$RESULTS"

# Neural methods (GPU). Heavy methods (MMD: DAN/JAN/MDD; transformers: EEGConformer/
# DBConformer; MVCNet) are front-loaded at indices 0-5 so index%%NSHARD
# spreads them across distinct shards instead of piling onto one GPU.
NEURAL=(
  "DAN|--algorithm DAN"
  "JAN|--algorithm JAN"
  "MDD|--algorithm MDD"
  "EA-EEGConformer|--algorithm EA-EEGConformer"
  "EA-DBConformer|--algorithm EA-DBConformer"
  "MVCNet|--algorithm MVCNet"
  "EA-EEGNet|--algorithm EA-EEGNet"
  "NoAlign-EEGNet|--aligner Identity --backbone EEGNet --head Linear --strategy ERM"
  "EA-ShallowConvNet|--aligner EA --backbone ShallowConvNet --head Linear --strategy ERM"
  "EA-DeepConvNet|--aligner EA --backbone DeepConvNet --head Linear --strategy ERM"
  "EA-DANN|--algorithm EA-DANN"
  "MCC|--algorithm MCC"
  "CDAN|--algorithm CDAN"
  "T-TIME|--algorithm T-TIME"
  "Tent|--algorithm Tent"
  "PL|--algorithm PL"
  "BN-adapt|--algorithm BN-adapt"
  "SHOT|--algorithm SHOT"
  "SAR|--algorithm SAR"
  "ISFDA|--algorithm ISFDA"
  "DELTA|--algorithm DELTA"
  "RA-EEGNet|--algorithm RA-EEGNet"
  "CR-EEGNet|--algorithm CR-EEGNet"
  "CSDA-EEGNet|--algorithm CSDA-EEGNet"
  "ABAT|--algorithm ABAT"
  "BFT|--algorithm BFT"
)

# Classical (CPU, fit-mode; epochs ignored). Run once, on the cpu invocation.
CLASSICAL=(
  "CSP-LDA|--algorithm CSP-LDA"
  "Riemann-MDM|--algorithm Riemann-MDM"
)

run_one () {
  local label="$1"; local dev="$2"; shift 2
  local extra=""; [ "$dev" = "cuda" ] && extra="--epochs 100"
  local s marker
  for s in $(seq 1 "$ITR"); do
    marker="$RESULTS/.done_${label}_seed${s}"
    if [ -f "$marker" ]; then echo "SKIP $label seed $s (done)"; continue; fi
    echo "===== $label (seed $s/$ITR) [$(date '+%H:%M:%S')] ====="
    if $PY -m hustbciml.run "$@" $extra --dataset BNCI2014001 --device "$dev" \
         --itr 1 --seed "$s" --results_dir "$RESULTS" --data_dir "$DATA"; then
      touch "$marker"
    else
      echo "FAILED $label seed $s (continuing)"
    fi
  done
}

if [ "$GPU" = "cpu" ]; then
  for entry in "${CLASSICAL[@]}"; do
    IFS='|' read -r label args <<< "$entry"
    run_one "$label" cpu $args
  done
  echo "CPU SHARD DONE"
else
  i=0
  for entry in "${NEURAL[@]}"; do
    if [ $(( i % NSHARD )) -eq "$SHARD" ]; then
      IFS='|' read -r label args <<< "$entry"
      run_one "$label" cuda $args
    fi
    i=$((i + 1))
  done
  echo "SHARD ${SHARD}/${NSHARD} (gpu ${GPU}) DONE"
fi
