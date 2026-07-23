#!/bin/bash
# Fan the 10 new-backbone sweeps across three GPUs (4,5,6), grouping the heavier
# transformer backbones (EEGDeformer, the multi-scale transformers) so no single
# GPU carries all of them. Each sweep is resume-safe, so re-running only fills gaps.
SC=/home/sylyoung/hustbciml/scripts
LOG=/home/sylyoung/backbone_logs
mkdir -p "$LOG"; rm -f "$LOG/_done.marker"
nohup bash "$SC/sweep_backbones.sh" 4 EA-EEGDeformer EA-FBMSNet EA-ADFCNN          > "$LOG/g4.log" 2>&1 &
nohup bash "$SC/sweep_backbones.sh" 5 EA-CTNet EA-MSCFormer EA-MSVTNet EA-TMSANet  > "$LOG/g5.log" 2>&1 &
nohup bash "$SC/sweep_backbones.sh" 6 EA-EEGWaveNet EA-SlimSeiz EA-EEGNeX          > "$LOG/g6.log" 2>&1 &
wait
echo "BACKBONE_ALL_DONE $(date '+%F %T')" > "$LOG/_done.marker"
