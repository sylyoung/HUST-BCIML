#!/bin/bash
# One-shot launcher for the three-dataset sweep: start all 8 jobs (6 GPU shards +
# 2 CPU LSFT) as detached nohup processes, then exit. Invoked via a single short
# SSH so a VPN blip during launch cannot leave a half-started fleet; the jobs are
# nohup-detached before this returns, so they survive even if the SSH drops.
# Resume-safe via per-(dataset,method,seed) markers, so re-running is idempotent.
#
# GPU assignment (7002, 2026-07-16): 0,1,2,3,5,6 are free; GPU4 (~16GB) and GPU7
# (~400MB) are the user's OTHER experiments — avoided.
set -u
cd /home/sylyoung
S=hustbciml/scripts/sweep_threeds.sh
ITR=3
nohup bash $S BNCI2014002 0 0 3 $ITR </dev/null > /tmp/sweep3ds_2014002_s0.log 2>&1 &
nohup bash $S BNCI2014002 1 1 3 $ITR </dev/null > /tmp/sweep3ds_2014002_s1.log 2>&1 &
nohup bash $S BNCI2014002 2 2 3 $ITR </dev/null > /tmp/sweep3ds_2014002_s2.log 2>&1 &
nohup bash $S BNCI2015001 3 0 3 $ITR </dev/null > /tmp/sweep3ds_2015001_s0.log 2>&1 &
nohup bash $S BNCI2015001 5 1 3 $ITR </dev/null > /tmp/sweep3ds_2015001_s1.log 2>&1 &
nohup bash $S BNCI2015001 6 2 3 $ITR </dev/null > /tmp/sweep3ds_2015001_s2.log 2>&1 &
nohup bash $S BNCI2014002 cpu 0 1 $ITR </dev/null > /tmp/sweep3ds_2014002_cpu.log 2>&1 &
nohup bash $S BNCI2015001 cpu 0 1 $ITR </dev/null > /tmp/sweep3ds_2015001_cpu.log 2>&1 &
sleep 4
echo "LAUNCHED $(pgrep -u sylyoung -fc 'sweep_threeds.sh') sweep shells"
