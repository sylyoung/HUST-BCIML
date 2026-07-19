#!/bin/bash
# Real-data validation smoke for the 7 newly ported methods on BNCI2014001, 1 seed.
# Confirms each runs end-to-end on real MI data and gives a sane (above-chance,
# roughly competitive) number before committing to the full 3-dataset x 3-seed runs.
# Especially validates MEKT, whose transductive pseudo-label loop is inconclusive on
# synthetic toy data. Fire-and-poll: launch with nohup, poll _progress.log.
set -u
PY=/home/sylyoung/micromamba/envs/syl-work/bin/python
DATA=/home/sylyoung/data
RES=/home/sylyoung/hustbciml_results_newsmoke
DS=BNCI2014001
cd /home/sylyoung
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$RES"
: > "$RES/_progress.log"

run() {  # gpu, algo
  local gpu=$1 algo=$2
  CUDA_VISIBLE_DEVICES=$gpu $PY -m hustbciml.run --algorithm "$algo" --dataset $DS \
    --device cuda --seed 1 --itr 1 --results_dir "$RES/$algo" --data_dir "$DATA" \
    > "$RES/${algo}.log" 2>&1
  echo "DONE $algo rc=$?" >> "$RES/_progress.log"
}

# Wave 1: four fast/medium methods, one per free GPU (0,3,5,7).
run 0 EA-TIEEEGNet &
run 3 EA-KDFNet &
run 5 ASFA &
run 7 MDMAML &
wait
# Wave 2: the remaining two.
run 0 MEKT &
run 3 SAFE &
wait
echo "ALL_SMOKE_DONE" >> "$RES/_progress.log"
