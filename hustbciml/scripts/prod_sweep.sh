#!/bin/bash
# Production run: all 16 algorithms x ITR seeds on BNCI2014001 (cross-subject
# LOSO), for the mean+/-std leaderboard. Writes to a FRESH results dir, kept
# separate from the single-seed coverage sweep in /home/sylyoung/hustbciml_results.
# Neural methods run on one GPU; the two classical methods run on CPU.
#
# Resume-safe: each (algorithm, seed) is a separate --itr 1 call guarded by a
# done-marker, so a mid-run crash/outage resumes where it stopped instead of
# restarting the ~multi-hour job. A failed run is logged and skipped, not fatal.
#
# Usage: prod_sweep.sh GPU_ID ITR   (default GPU 5, ITR 3; seeds start at 1)
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
# LABEL is used only for the resume marker + log header; the metrics.json the
# run saves carries the real algorithm identity the leaderboard groups on.
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

# --- neural (GPU) ---
run_one EA-EEGNet         cuda --algorithm EA-EEGNet
run_one NoAlign-EEGNet    cuda --aligner Identity --backbone EEGNet --head Linear --strategy ERM
run_one EA-ShallowConvNet cuda --aligner EA --backbone ShallowConvNet --head Linear --strategy ERM
run_one EA-DeepConvNet    cuda --aligner EA --backbone DeepConvNet --head Linear --strategy ERM
run_one EA-DANN           cuda --algorithm EA-DANN
run_one T-TIME            cuda --algorithm T-TIME
run_one Tent              cuda --algorithm Tent
run_one PL                cuda --algorithm PL
run_one BN-adapt          cuda --algorithm BN-adapt
run_one MCC               cuda --algorithm MCC
run_one CDAN              cuda --algorithm CDAN
run_one DAN               cuda --algorithm DAN
run_one JAN               cuda --algorithm JAN
run_one MDD               cuda --algorithm MDD

# --- classical (CPU) ---
run_one CSP-LDA           cpu  --algorithm CSP-LDA
run_one Riemann-MDM       cpu  --algorithm Riemann-MDM

# --- lab ports (new) ---
run_one CSP-Net           cuda --algorithm CSP-Net      # backbone: CSP-initialized EEGNet
run_one DJP-MMD           cuda --algorithm DJP-MMD      # strategy: discriminative joint-probability MMD
run_one LSFT              cpu  --algorithm LSFT         # classical: lightweight source-free transfer
run_one MSDT              cpu  --algorithm MSDT         # classical: multi-source decentralized transfer

echo "PROD_SWEEP_DONE"
