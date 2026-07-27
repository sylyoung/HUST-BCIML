#!/bin/bash
# rerun_v12_nettune.sh  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
#
# The learning-rate-tuned half of the v1.2.0 Networks-table re-measurement.
#
# Four of the backbones whose construction-time shape probe was fixed have their
# learning rate SELECTED rather than fixed (ShallowConvNet, DeepConvNet,
# EEGConformer, DBConformer). Re-running only the reported seeds would be wrong:
# the probe fix changes the initial RNG state, so the validation curve that picked
# the learning rate is itself re-drawn, and the selection has to be redone from the
# grid. Each pair therefore costs 4 grid runs + 3 seed runs.
#
# The other four tuned backbones (EEGNet, CSP-Net, TIE-EEGNet, KDFNet) never probed
# and are deliberately not re-run — see RERUN.md.
#
# Every (dataset, backbone) pair gets its OWN results dir. tune_networks.py writes
# its verdict to <results_dir>/tuned_<dataset>.json, read-modify-write, so two
# workers sharing a dir would silently drop each other's entries. Separate dirs
# make the parallelism safe; extract_v12.py --nettune_dir stitches the pieces back.
#
#   Usage: rerun_v12_nettune.sh <GPU_ID> <shard_index> <shard_count>
set -u
GPU=${1:?gpu id}
SHARD=${2:-0}
NSHARD=${3:-1}

# Resolve the script's own directory BEFORE the cd below. ``dirname "$0"`` is
# relative when the script is launched by a relative path, and the cd then makes it
# point somewhere else entirely — cell_origin.tsv silently fails to open and the
# origin filter rejects every pair, which looks exactly like "nothing to do here".
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

PY=${PY:-/home/sylyoung/micromamba/envs/syl-work/bin/python}
# Fail before the grid, not once per pair — see the same guard in rerun_v12.sh.
if [ ! -x "$PY" ]; then
  echo "[gpu$GPU] no interpreter at $PY — set PY to this box's syl-work python" >&2
  exit 1
fi
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib:${LD_LIBRARY_PATH:-}"
DATA=${DATA:-/home/sylyoung/data}
ROOT=${ROOT:-/home/sylyoung/hustbciml_v12_nettune}
LOGS=${LOGS:-/home/sylyoung/v12_logs}
cd "${WORKDIR:-/home/sylyoung}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$ROOT" "$LOGS"

# Overridable so a subset can be re-run without going through the shard arithmetic.
# The shard index counts in-scope pairs *before* the already-tuned check, so asking for
# "the one pair that is missing" by shard number is fragile: a pair still being tuned by
# a live worker has no verdict file yet, so it does not skip, and a second worker would
# land in the same results directory. Naming the subset directly avoids that.
#   DS_LIST="BNCI2014001" BB_LIST="DBConformer" rerun_v12_nettune.sh 2
read -ra DATASETS <<< "${DS_LIST:-BNCI2014001 BNCI2014002 BNCI2015001}"
read -ra BACKBONES <<< "${BB_LIST:-ShallowConvNet DeepConvNet EEGConformer DBConformer}"

# Same rule as rerun_v12.sh: a cell is re-measured on the box that produced it,
# because a cross-machine delta mixes the code change with a BLAS change. The
# published values for these twelve came from tune_networks.py, whose verdict file
# records the selected learning rate and the 3-seed mean, so provenance is resolved by
# reading ``tuned_<dataset>.json`` rather than by matching run directories.
#
# All twelve resolve on 7002. An earlier revision of this comment said only nine did,
# and that DeepConvNet, EEGConformer and DBConformer on BNCI2014001 "reproduce on no
# surviving tree on any reachable box" — that was wrong. Their means and stds sit in
# ``hustbciml_results_nettune/tuned_BNCI2014001.json`` on 7002, in the same file that
# resolved ShallowConvNet on the same dataset. Three affected cells were therefore
# skipped as provenance gaps and would have shipped with pre-fix numbers.
#
# Set JOB_ORIGIN to this box's family to enforce the rule; leave it empty to run the
# whole grid (useful on a fresh machine with nothing to be consistent with).
ORIGIN_MAP=${ORIGIN_MAP:-$SCRIPT_DIR/cell_origin.tsv}
JOB_ORIGIN=${JOB_ORIGIN:-}
if [ -n "$JOB_ORIGIN" ] && [ ! -r "$ORIGIN_MAP" ]; then
  # Without the map every pair would be filtered out and the sweep would report a
  # clean "nothing in scope" finish having measured nothing at all.
  echo "[gpu$GPU] JOB_ORIGIN=$JOB_ORIGIN but no origin map at $ORIGIN_MAP" >&2
  exit 1
fi
# 10022 and 20022 are bit-identical (both MKL), so either may stand in for the other.
# The fold has to happen on BOTH sides: this script previously compared the map's value
# to JOB_ORIGIN literally, so a worker on 20022 told `JOB_ORIGIN=20022` would reject
# every cell recorded as 10022 and report a clean "nothing in scope" finish. The same
# trap cost a wasted launch in rerun_v12.sh, which now folds; this one did not, and the
# only reason it never fired is that all twelve tuned cells happen to live on 7002.
family() { case "$1" in 10022|20022) echo 10022 ;; *) echo "$1" ;; esac; }

in_scope() {                                       # in_scope <dataset> <backbone>
  [ -z "$JOB_ORIGIN" ] && return 0
  local want cell
  want=$(family "$JOB_ORIGIN")
  cell=$(awk -v map="$ORIGIN_MAP" -v key="EA-$2" -v ds="$1" '
    BEGIN {
      while ((getline line < map) > 0) {
        if (line ~ /^#/ || line == "") continue
        if (split(line, f, "\t") >= 3 && f[1] == key && f[2] == ds) { print f[3]; exit }
      }
    }')
  [ -n "$cell" ] && [ "$(family "$cell")" = "$want" ]
}

i=0
for ds in "${DATASETS[@]}"; do
  for bb in "${BACKBONES[@]}"; do
    if ! in_scope "$ds" "$bb"; then
      echo "SKIP $ds $bb (origin is not $JOB_ORIGIN)"; continue
    fi
    if [ $((i % NSHARD)) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
    i=$((i+1))
    res="$ROOT/${ds}_${bb}"
    if [ -f "$res/tuned_${ds}.json" ]; then
      echo "SKIP $ds $bb (already tuned)"; continue
    fi
    echo "===== [gpu$GPU] tune $ds $bb  [$(date '+%F %H:%M:%S')] ====="
    mkdir -p "$res"
    if $PY -m hustbciml.scripts.tune_networks --dataset "$ds" --backbones "$bb" \
         --device cuda --results_dir "$res" --data_dir "$DATA" \
         > "$LOGS/nettune_${ds}_${bb}.log" 2>&1; then
      grep '^\[final\]' "$LOGS/nettune_${ds}_${bb}.log" | tail -1
    else
      echo "FAILED tune $ds $bb — see $LOGS/nettune_${ds}_${bb}.log"
      tail -5 "$LOGS/nettune_${ds}_${bb}.log"
    fi
  done
done
echo "[gpu$GPU] NETTUNE SHARD $SHARD/$NSHARD DONE  [$(date '+%F %H:%M:%S')]"
