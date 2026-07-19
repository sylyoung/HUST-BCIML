#!/bin/bash
# Full 3-dataset runs for the 6 new methods, parallel across 4 free GPUs (0,3,5,7).
#   GPU0: MDMAML sweep (slowest, dedicated)         3 ds x 3 seeds
#   GPU3: ASFA + SAFE sweeps                         3 ds x 3 seeds each (SAFE 2014001 -> 4-class)
#   GPU5: MEKT sweep (fast, mostly CPU)             3 ds x 3 seeds
#   GPU7: tune_networks TIE-EEGNet + KDFNet          LR grid + 3 seeds, all 3 datasets
# Method sweeps -> hustbciml_results_3ds; backbone tuning -> hustbciml_results_nettune
# (appends TIE-EEGNet/KDFNet to the existing tuned_<DS>.json). Fire-and-poll.
set -u
SC=/home/sylyoung/hustbciml/scripts
NT=/home/sylyoung/hustbciml_results_nettune
PY=/home/sylyoung/micromamba/envs/syl-work/bin/python
DATA=/home/sylyoung/data
LOG=/home/sylyoung/fullrun_logs
cd /home/sylyoung
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$LOG"
rm -f "$LOG/_done.marker"

nohup bash "$SC/sweep_newmethods.sh" 0 MDMAML    > "$LOG/sweep_mdmaml.log" 2>&1 &
nohup bash "$SC/sweep_newmethods.sh" 3 ASFA SAFE > "$LOG/sweep_asfa_safe.log" 2>&1 &
nohup bash "$SC/sweep_newmethods.sh" 5 MEKT      > "$LOG/sweep_mekt.log" 2>&1 &

(
  for ds in BNCI2014001 BNCI2014002 BNCI2015001; do
    CUDA_VISIBLE_DEVICES=7 $PY -m hustbciml.scripts.tune_networks --dataset "$ds" \
      --backbones TIE-EEGNet,KDFNet --device cuda \
      --results_dir "$NT" --data_dir "$DATA"
  done
) > "$LOG/tune_newbackbones.log" 2>&1 &

wait
echo "FULLRUN_ALL_DONE $(date '+%F %T')" > "$LOG/_done.marker"
