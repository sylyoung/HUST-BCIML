#!/bin/bash
# GAP runs that complete RESULTS.md's three-dataset extension: the classical
# network-free baselines that sweep_threeds.sh omitted but RESULTS.md documents.
#   cls : CSP-LDA, Riemann-MDM  (classical network-free baselines, fit-mode, CPU)
# Same done-marker / resume design as sweep_threeds.sh; results share the same
# hustbciml_results_3ds dir so extract_3ds_v2.py picks them up uniformly. The
# multi-seed self-ensemble (T-TIME x5 + combiners) is launched separately via
# hustbciml.scripts.ensemble, which resumes from the seeds already in that dir.
#
# Usage:
#   sweep_threeds_extra.sh <DATASET> cpu cls [ITR]
set -u
DATASET=${1:?dataset}; DEV=${2:?gpu id or cpu}; KIND=${3:?gpu|cls}; ITR=${4:-3}
RESULTS=/home/sylyoung/hustbciml_results_3ds
DATA=/home/sylyoung/data
PY=/home/sylyoung/micromamba/envs/syl-work/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
[ "$DEV" != "cpu" ] && export CUDA_VISIBLE_DEVICES=$DEV
cd /home/sylyoung
mkdir -p "$RESULTS"

# classical network-free baselines only (the GPU test-time methods CoTTA/T3A were removed)
METHODS=("CSP-LDA|--algorithm CSP-LDA" "Riemann-MDM|--algorithm Riemann-MDM"); DEVICE=cpu; EXTRA=""

for entry in "${METHODS[@]}"; do
  IFS='|' read -r label args <<< "$entry"
  for s in $(seq 1 "$ITR"); do
    marker="$RESULTS/.done_${DATASET}_${label}_seed${s}"
    if [ -f "$marker" ]; then echo "SKIP $DATASET $label seed $s (done)"; continue; fi
    echo "===== $DATASET $label (seed $s/$ITR) [$(date '+%H:%M:%S')] ====="
    if $PY -m hustbciml.run $args $EXTRA --dataset "$DATASET" --device "$DEVICE" \
         --itr 1 --seed "$s" --results_dir "$RESULTS" --data_dir "$DATA"; then
      touch "$marker"
    else
      echo "FAILED $DATASET $label seed $s (continuing)"
    fi
  done
done
echo "EXTRA $KIND DONE ($DATASET)"
