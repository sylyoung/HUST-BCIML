#!/bin/bash
# hustbciml GPU-pinned launcher for the HUST servers.
# Usage: server_launch.sh GPU_ID --algorithm EA-EEGNet --dataset BNCI2014001 ...
# Pins the GPU (env set inside the script survives nohup+ssh, unlike a shell
# prefix), limits CPU threads on the shared box, and runs from /home/sylyoung
# where the hustbciml package lives.
GPU=$1; shift
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
cd /home/sylyoung
exec python -m hustbciml.run "$@"
