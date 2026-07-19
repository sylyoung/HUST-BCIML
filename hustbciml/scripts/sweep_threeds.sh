#!/bin/bash
# Three-dataset sweep: run the DISPLAYED stage-table method set on ONE dataset
# (cross-subject LOSO), ITR seeds, for the mean+/-std leaderboard. Used to extend
# the benchmark from BNCI2014001 to the two additional MOABB datasets BNCI2014002
# and BNCI2015001, so every comparison table spans all three datasets.
#
# Same shard/marker/resume design as prod_sweep_full.sh: methods are sharded by
# position %% NSHARD across GPUs (disjoint shards, race-free shared markers); each
# (dataset, method, seed) is a separate --itr 1 call guarded by a done-marker, so a
# crash/outage resumes where it stopped. A failed run is logged and skipped.
#
# Results dir is shared across datasets: Config.setting() embeds the dataset name
# ({dataset}_{protocol}_{algo}_seed{seed}), so runs never collide; compare.py
# --dataset filters. Marker names include the dataset too.
#
# CR-EEGNet is intentionally ABSENT: Channel Reflection's hemisphere-reflect +
# label-swap is defined only for 2-class LEFT/RIGHT hand; these two datasets are
# right-hand-vs-feet (and BNCI2014002 has no 10-20 montage), so it is inapplicable.
# T3A/CoTTA/CSP-LDA/Riemann-MDM (not shown on the site) and MSDT (privacy-only,
# O(N^2) refit too slow at 512 Hz) are also omitted here.
#
# Usage:
#   sweep_threeds.sh <DATASET> <GPU_ID> <SHARD> <NSHARD> [ITR]   # neural on a GPU
#   sweep_threeds.sh <DATASET> cpu       0       1       [ITR]   # LSFT (fit, CPU)
set -u
DATASET=${1:?dataset}; GPU=${2:?gpu id or cpu}; SHARD=${3:?shard idx}; NSHARD=${4:?num shards}; ITR=${5:-3}
RESULTS=/home/sylyoung/hustbciml_results_3ds
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
# DBConformer; MVCNet) are front-loaded at indices 0-5 so index%%NSHARD spreads them
# across distinct shards instead of piling onto one GPU.
NEURAL=(
  "DAN|--algorithm DAN"
  "JAN|--algorithm JAN"
  "MDD|--algorithm MDD"
  "EA-EEGConformer|--algorithm EA-EEGConformer"
  "EA-DBConformer|--algorithm EA-DBConformer"
  "MVCNet|--algorithm MVCNet"
  "CSP-Net|--algorithm CSP-Net"
  "EA-EEGNet|--algorithm EA-EEGNet"
  "NoAlign-EEGNet|--aligner Identity --backbone EEGNet --head Linear --strategy ERM"
  "EA-ShallowConvNet|--aligner EA --backbone ShallowConvNet --head Linear --strategy ERM"
  "EA-DeepConvNet|--aligner EA --backbone DeepConvNet --head Linear --strategy ERM"
  "RA-EEGNet|--algorithm RA-EEGNet"
  "EA-DANN|--algorithm EA-DANN"
  "MCC|--algorithm MCC"
  "CDAN|--algorithm CDAN"
  "DJP-MMD|--algorithm DJP-MMD"
  "ABAT|--algorithm ABAT"
  "CSDA-EEGNet|--algorithm CSDA-EEGNet"
  "T-TIME|--algorithm T-TIME"
  "Tent|--algorithm Tent"
  "PL|--algorithm PL"
  "BN-adapt|--algorithm BN-adapt"
  "SHOT|--algorithm SHOT"
  "SAR|--algorithm SAR"
  "ISFDA|--algorithm ISFDA"
  "DELTA|--algorithm DELTA"
  "BFT|--algorithm BFT"
)
# Classical fit-mode (CPU): LSFT (tangent-space source-free).
CLASSICAL=(
  "LSFT|--algorithm LSFT"
)

run_one () {
  local label="$1"; local dev="$2"; shift 2
  local extra=""; [ "$dev" = "cuda" ] && extra="--epochs 100"
  local s marker
  for s in $(seq 1 "$ITR"); do
    marker="$RESULTS/.done_${DATASET}_${label}_seed${s}"
    if [ -f "$marker" ]; then echo "SKIP $DATASET $label seed $s (done)"; continue; fi
    echo "===== $DATASET $label (seed $s/$ITR) [$(date '+%H:%M:%S')] ====="
    if $PY -m hustbciml.run "$@" $extra --dataset "$DATASET" --device "$dev" \
         --itr 1 --seed "$s" --results_dir "$RESULTS" --data_dir "$DATA"; then
      touch "$marker"
    else
      echo "FAILED $DATASET $label seed $s (continuing)"
    fi
  done
}

if [ "$GPU" = "cpu" ]; then
  for entry in "${CLASSICAL[@]}"; do
    IFS='|' read -r label args <<< "$entry"
    run_one "$label" cpu $args
  done
  echo "CPU SHARD DONE ($DATASET)"
else
  i=0
  for entry in "${NEURAL[@]}"; do
    if [ $(( i % NSHARD )) -eq "$SHARD" ]; then
      IFS='|' read -r label args <<< "$entry"
      run_one "$label" cuda $args
    fi
    i=$((i + 1))
  done
  echo "SHARD ${SHARD}/${NSHARD} (gpu ${GPU}, $DATASET) DONE"
fi
