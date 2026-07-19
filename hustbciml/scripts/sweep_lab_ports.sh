#!/bin/bash
# Incremental sweep for the four lab-method ports — CSP-Net (backbone), DJP-MMD
# (strategy), LSFT and MSDT (classical source-free / multi-source strategies) —
# on BNCI2014001 (cross-subject LOSO), ITR seeds. Writes into the SAME 3-seed
# results dir as prod_sweep.sh so scripts/compare.py ranks them with the other
# methods. Resume-safe: each (method, seed) is a separate --itr 1 call guarded by
# a done-marker, so a crash/outage resumes where it stopped. Neural methods run
# on the pinned GPU; the two classical (tangent-feature) methods run on CPU.
#
# Usage: sweep_lab_ports.sh GPU_ID ITR   (default GPU 5, ITR 3; seeds start at 1)
set -u
GPU=${1:-5}
ITR=${2:-3}
RESULTS=/home/sylyoung/hustbciml_results_3seed
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
cd /home/sylyoung
mkdir -p "$RESULTS"

# run_one LABEL DEVICE <hustbciml algorithm/stage args...>
run_one () {
  local label="$1"; local dev="$2"; shift 2
  local extra=""
  [ "$dev" = "cuda" ] && extra="--epochs 100"   # classical (fit mode) ignores epochs
  local s marker
  for s in $(seq 1 "$ITR"); do
    marker="$RESULTS/.done_${label}_seed${s}"
    if [ -f "$marker" ]; then
      echo "SKIP $label seed $s (already done)"
      continue
    fi
    echo "===== $label (seed $s / $ITR) ====="
    if python -m hustbciml.run "$@" $extra --dataset BNCI2014001 --device "$dev" \
         --itr 1 --seed "$s" --results_dir "$RESULTS"; then
      touch "$marker"
    else
      echo "FAILED $label seed $s (continuing)"
    fi
  done
}

# --- lab ports ---
run_one CSP-Net  cuda --algorithm CSP-Net    # EEGNet w/ CSP-initialized spatial conv
run_one DJP-MMD  cuda --algorithm DJP-MMD    # discriminative joint-probability MMD
run_one LSFT     cpu  --algorithm LSFT       # lightweight source-free transfer (tangent)
run_one MSDT     cpu  --algorithm MSDT       # multi-source decentralized transfer (tangent)
echo "SWEEP_LAB_PORTS_DONE"
