#!/usr/bin/env bash
# rerun_v12_classical.sh  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
#
# Verify the network-free classical rows — CSP-LDA and Riemann-MDM — are unchanged at
# v1.2.0. RERUN.md argues they cannot have moved: they run EA (untouched) into a
# scikit-learn/MNE/pyriemann pipeline, with no backbone, so none of the release's
# changes can reach them. Every other "unaffected" claim in this release was checked by
# re-running the row and requiring it to reproduce per subject rather than by reading
# the diff, and these two rows are about to be published on the leaderboard for the
# first time, so they get the same treatment.
#
# CPU only (no GPU argument): these pipelines never touch CUDA. That also means this
# can run alongside the GPU sweeps without competing for a device.
#
# Provenance, from cell_origin.tsv: BNCI2014001 belongs to the MKL family (20022), the
# other two datasets to 7002. Re-measuring a cell on the wrong box would compare a BLAS
# change against a code change, so each box takes only its own datasets.
#
#   Usage: rerun_v12_classical.sh              # this box's cells, per JOB_ORIGIN
#          JOB_ORIGIN=7002 rerun_v12_classical.sh
set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ORIGIN_MAP=${ORIGIN_MAP:-$SCRIPT_DIR/cell_origin.tsv}
if [ ! -r "$ORIGIN_MAP" ]; then
  echo "no origin map at $ORIGIN_MAP — refusing to guess which cells belong here" >&2
  exit 1
fi

PY=${PY:-/home/sylyoung/micromamba/envs/syl-work/bin/python}
if [ ! -x "$PY" ]; then
  echo "no interpreter at $PY — set PY to this box's syl-work python" >&2
  exit 1
fi
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib:${LD_LIBRARY_PATH:-}"
DATA=${DATA:-/home/sylyoung/data}
RES=${RES:-/home/sylyoung/hustbciml_v12_classical}
LOGS=${LOGS:-/home/sylyoung/v12_logs}
SEEDS=${SEEDS:-1 2 3}
cd "${WORKDIR:-/home/sylyoung}"
# Four threads: CSP eigendecompositions and Riemannian means are BLAS-heavy and would
# otherwise spread over every core on a box shared with other users' jobs.
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$RES" "$LOGS"

# 10022 and 20022 are bit-identical (both MKL), so either may stand in for the other.
family() { case "$1" in 10022|20022) echo 10022 ;; *) echo "$1" ;; esac; }

in_scope() {                                     # in_scope <algorithm> <dataset>
  [ -z "${JOB_ORIGIN:-}" ] && return 0
  own=$(awk -F'\t' -v k="$1" -v d="$2" '$1==k && $2==d {print $3}' "$ORIGIN_MAP")
  [ -z "$own" ] && return 1
  [ "$(family "$own")" = "$(family "$JOB_ORIGIN")" ]
}

ran=0; skipped=0; failed=0
for algo in CSP-LDA Riemann-MDM; do
  for ds in BNCI2014001 BNCI2014002 BNCI2015001; do
    in_scope "$algo" "$ds" || { skipped=$((skipped+1)); continue; }
    for seed in $SEEDS; do
      key="${ds}_${algo}_seed${seed}"
      [ -d "$RES/.done_$key" ] && { skipped=$((skipped+1)); continue; }
      echo "[classical] $key"
      if "$PY" -m hustbciml.run --algorithm "$algo" --dataset "$ds" \
           --seed "$seed" --itr 1 --device cpu \
           --results_dir "$RES" --data_dir "$DATA" \
           >"$LOGS/classical_$key.log" 2>&1; then
        mkdir -p "$RES/.done_$key"; ran=$((ran+1))
      else
        echo "[classical] FAILED $key — see $LOGS/classical_$key.log" >&2
        mkdir -p "$RES/.failed_$key"; failed=$((failed+1))
      fi
    done
  done
done
echo "[classical] ran=$ran skipped=$skipped failed=$failed"
