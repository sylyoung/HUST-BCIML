#!/bin/bash
# rerun_v12_ensemble.sh  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
#
# The ensemble-table half of the v1.2.0 re-measurement.
#
# RERUN.md originally scoped this to the three combiners whose own maths changed
# (SML's global sign repair, LAA's logits, PM's per-round normalisation). That was
# too narrow. The decentralized ensemble's per-source learners are Tangent+LDA,
# Tangent+SVM, EEGNet, ShallowConvNet and CSP-Net — and ShallowConvNet is one of
# the backbones whose construction-time shape probe was fixed. Its initial weights
# therefore differ, so the hard votes every combiner consumes differ, so EVERY row
# in the table moves — including majority voting, which is the table's baseline.
# There is no way to re-measure three rows against an unchanged baseline here.
#
# Cost is modest despite the row count: each learner is trained on ONE subject's
# data, not the pooled sources, and two of the five are classical.
#
#   Usage: rerun_v12_ensemble.sh <GPU_ID> <dataset> [dataset ...]
set -u
GPU=${1:?gpu id}; shift
DATASETS=("$@")

PY=${PY:-/home/sylyoung/micromamba/envs/syl-work/bin/python}
# Fail before the dataset loop, not once per dataset. The default above is one
# box's path; on a box where the env lives elsewhere, forgetting PY otherwise turns
# every dataset into a FAILED line in a few seconds, which reads like a sweep that
# ran and lost rather than one that never started. (The same omission cost a full
# 119-job pass on 20022 before rerun_v12.sh grew this guard.)
if [ ! -x "$PY" ]; then
  echo "[gpu$GPU] no interpreter at $PY — set PY to this box's syl-work python" >&2
  exit 1
fi
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib:${LD_LIBRARY_PATH:-}"
DATA=${DATA:-/home/sylyoung/data}
RES=${RES:-/home/sylyoung/hustbciml_v12_ensemble}
LOGS=${LOGS:-/home/sylyoung/v12_logs}
cd "${WORKDIR:-/home/sylyoung}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$RES" "$LOGS"

for ds in "${DATASETS[@]}"; do
  out="$RES/decentralized_${ds}_hetero_EA-EEGNet.json"
  if [ -f "$out" ]; then echo "SKIP $ds (already aggregated)"; continue; fi
  echo "===== [gpu$GPU] ensemble $ds  [$(date '+%F %H:%M:%S')] ====="
  if $PY -m hustbciml.scripts.decentralized --dataset "$ds" --base hetero \
       --seeds 1,2,3 --device cuda --results_dir "$RES" --data_dir "$DATA" \
       > "$LOGS/ensemble_${ds}.log" 2>&1; then
    tail -20 "$LOGS/ensemble_${ds}.log"
  else
    echo "FAILED ensemble $ds — see $LOGS/ensemble_${ds}.log"
    tail -8 "$LOGS/ensemble_${ds}.log"
  fi
done
echo "[gpu$GPU] ENSEMBLE DONE (${DATASETS[*]})  [$(date '+%F %H:%M:%S')]"
