#!/bin/bash
# Fan the 8 augmentation-baseline sweeps across three idle GPUs (4,5,6), balanced
# by rough per-method cost (FFT/DCT-based FSurr/FComb are grouped together). Each
# sweep is resume-safe, so re-running this after an interruption only fills gaps.
SC=/home/sylyoung/hustbciml/scripts
LOG=/home/sylyoung/aug_logs
mkdir -p "$LOG"; rm -f "$LOG/_done.marker"
nohup bash "$SC/sweep_augmenters.sh" 4 Noise-EEGNet Flip-EEGNet Scale-EEGNet   > "$LOG/g4.log" 2>&1 &
nohup bash "$SC/sweep_augmenters.sh" 5 FShift-EEGNet FSurr-EEGNet FComb-EEGNet > "$LOG/g5.log" 2>&1 &
nohup bash "$SC/sweep_augmenters.sh" 6 HS-EEGNet Symm-EEGNet                   > "$LOG/g6.log" 2>&1 &
wait
echo "AUG_ALL_DONE $(date '+%F %T')" > "$LOG/_done.marker"
