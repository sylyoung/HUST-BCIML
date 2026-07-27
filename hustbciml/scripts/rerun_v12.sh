#!/bin/bash
# rerun_v12.sh  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
#
# Re-measures exactly the leaderboard cells that RERUN.md lists as invalidated by
# the v1.2.0 behaviour fixes, plus a set of control runs that must reproduce their
# v1.1.x values bit-for-bit. Everything else in the leaderboard is left alone.
#
# Workers pull from a shared job list and claim each job with an atomic ``mkdir``,
# so several GPUs can run the same command with no coordination and no duplicated
# work; the slowest jobs do not stall a GPU that has finished its share. Re-running
# after an interruption only fills the gaps (a job with a ``.done_`` marker is
# skipped), which matters because a full pass is several hours over a VPN that
# drops.
#
#   Usage: rerun_v12.sh <GPU_ID> [group]
#          group = affected (default) | control | all
#
# Launch one per free GPU:
#   for g in 0 2 7; do nohup bash rerun_v12.sh $g all > ~/v12_logs/g$g.log 2>&1 & done
#
# DO NOT copy a new version of this file over a copy that workers are executing.
# Bash reads a script lazily and remembers a byte offset, so rewriting the file
# under a running worker makes it resume at a stale position: mid-run the loop dies
# with something like "syntax error near unexpected token `fi'" at a line number
# that looks nothing like the bug. Results already written are safe (Python writes
# them, not the shell) but the tail of that worker's job list is silently never
# attempted, which looks identical to a finished sweep. This happened on 7002.
# Stop the workers first, or deploy under a new filename and launch that.
set -u
GPU=${1:?gpu id}
GROUP=${2:-affected}

# The syl-work env sits under a different manager on each box (micromamba on 7002,
# conda on 20022/10022), and calling the interpreter directly does not activate the
# env — so its lib dir has to go on LD_LIBRARY_PATH or torch cannot find the
# matching libcudart. Override PY to move this sweep to another server.
# Resolve the script's own directory BEFORE the cd below, so ORIGIN_MAP does not
# depend on how the script was invoked. ``dirname "$0"`` is relative for a relative
# launch, and the cd then repoints it: awk opens nothing, every job is judged out of
# scope, and the sweep reports a clean finish having measured nothing. It works today
# only because the launch cwd happens to equal WORKDIR.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

PY=${PY:-/home/sylyoung/micromamba/envs/syl-work/bin/python}
# Fail before the queue, not once per job. The default above is 7002's path; on a
# box where the env lives elsewhere, forgetting PY otherwise marches through the
# whole job list turning every entry into a .failed_ marker in a few seconds,
# which reads like a sweep that ran and lost rather than one that never started.
if [ ! -x "$PY" ]; then
  echo "[gpu$GPU] no interpreter at $PY — set PY to this box's syl-work python" >&2
  exit 1
fi
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib:${LD_LIBRARY_PATH:-}"
DATA=${DATA:-/home/sylyoung/data}
RES=${RES:-/home/sylyoung/hustbciml_v12_results}
LOGS=${LOGS:-/home/sylyoung/v12_logs}
cd "${WORKDIR:-/home/sylyoung}"
# Four threads is plenty for the covariance/EA maths and the DataLoader; the
# default would spawn on all 80 cores and starve the other users on this box.
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$RES" "$LOGS"

DS3=(BNCI2014001 BNCI2014002 BNCI2015001)
SEEDS=(1 2 3)

# ---------------------------------------------------------------- job list ----
# Cells RERUN.md marks as invalidated. Presets whose computation changed:
#   FSurr   phase-rotated surrogate, DC/Nyquist copied verbatim
#   CSDA    matched DWT boundary mode, partner drawn from a different subject
#   RA      per-covariance ridge
#   MDMAML  BatchNorm buffers restored between inner-loop task pairs
#   MVCNet  hp-driven loss weights; the reflected view is dropped where the
#           montage/class pair makes it invalid (BNCI2014002, BNCI2015001)
#   CR      fails closed on a non-anatomical montage or non-left/right classes,
#           so only BNCI2014001 is defined — the other two stay n/a by design
# ...plus every Networks-table backbone that probed its own output shape at
# construction and so perturbed BatchNorm and the RNG (fixed-hyperparameter
# presets here; the four learning-rate-tuned ones go through tune_networks.py).
#
# EA-SlimSeiz belongs here even though RERUN.md first listed SlimSeiz among the
# backbones that do not probe. It does: SlimSeiz.py runs a dummy forward to infer
# its feature width, exactly like the other fourteen. The probe set is fifteen
# backbones, not fourteen.
AFFECTED_PRESETS=(FSurr-EEGNet CSDA-EEGNet RA-EEGNet MDMAML MVCNet
                  EA-CTNet EA-ADFCNN EA-MSCFormer EA-MSVTNet EA-TMSANet
                  EA-EEGWaveNet EA-FBMSNet EA-EEGNeX EA-EEGDeformer EA-SlimSeiz)

# Control runs: RERUN.md asserts these are untouched. Re-running them on the new
# code turns that assertion into evidence — each must reproduce its v1.1.x number
# exactly, and a mismatch means the blast radius was mis-scoped. Each control uses
# a backbone that does no shape probe (EEGNet), so the claim is that nothing in
# the aligner, augmenter or training path moved either.
CONTROL_PRESETS=(EA-EEGNet NoAlign-EEGNet Noise-EEGNet FShift-EEGNet)

emit_jobs() {
  local group=$1
  if [ "$group" = "affected" ] || [ "$group" = "all" ]; then
    for algo in "${AFFECTED_PRESETS[@]}"; do
      for ds in "${DS3[@]}"; do
        for s in "${SEEDS[@]}"; do echo "$ds $algo $s"; done
      done
    done
    # Channel Reflection is defined only on the 10-20 / left-vs-right dataset.
    for s in "${SEEDS[@]}"; do echo "BNCI2014001 CR-EEGNet $s"; done
  fi
  if [ "$group" = "control" ] || [ "$group" = "all" ]; then
    # One seed is enough here. A control passes by reproducing its v1.1.x number
    # *exactly* — same per-subject accuracies, not the same mean within noise — so
    # a single seed either matches bit for bit or it does not. Extra seeds would
    # add hours of GPU time and no additional evidence.
    for algo in "${CONTROL_PRESETS[@]}"; do
      for ds in "${DS3[@]}"; do echo "$ds $algo 1"; done
    done
  fi
}

# Controls first: they are the cheap runs that would expose a mis-scoped fix, and
# finding that out early is worth more than finishing one affected row sooner.
JOBS=$( { emit_jobs control; emit_jobs affected; } | awk '!seen[$0]++' )
[ "$GROUP" = "affected" ] && JOBS=$(emit_jobs affected)
[ "$GROUP" = "control" ]  && JOBS=$(emit_jobs control)

# Which box a job belongs on is NOT a load-balancing question. The benchmark is
# bit-reproducible on a given machine but not across machines: the EA whitening
# step goes through a LAPACK eigendecomposition whose last bits differ between
# BLAS builds, and training amplifies that into up to ~2 accuracy points on a
# single seed. (NoAlign-EEGNet, which skips EA, is identical on every box — that
# is what locates the divergence.) So a cell re-measured on the wrong box yields
# an old-vs-new delta that mixes the v1.2.0 code change with a machine change,
# which is precisely the question this sweep exists to answer.
#
# cell_origin.tsv records which box produced each published cell. JOB_ORIGIN
# restricts this worker to the cells belonging to its own numerical family:
#   20022 (bit-identical to 10022):  JOB_ORIGIN=10022
#   7002:                            JOB_ORIGIN=7002
# ORPHAN_ORIGINS additionally claims cells whose origin box is unavailable; those
# deltas do carry a machine component, so the report flags them separately.
ORIGIN_MAP=${ORIGIN_MAP:-$SCRIPT_DIR/cell_origin.tsv}
JOB_ORIGIN=${JOB_ORIGIN:-}
ORPHAN_ORIGINS=${ORPHAN_ORIGINS:-}
if [ -n "$JOB_ORIGIN" ] && [ ! -r "$ORIGIN_MAP" ]; then
  # Otherwise the filter below quietly keeps nothing and the sweep "finishes".
  echo "[gpu$GPU] JOB_ORIGIN=$JOB_ORIGIN but no origin map at $ORIGIN_MAP" >&2
  exit 1
fi
if [ -n "$JOB_ORIGIN" ]; then
  JOBS=$(echo "$JOBS" | awk -v map="$ORIGIN_MAP" -v want="$JOB_ORIGIN,$ORPHAN_ORIGINS" '
    # 10022 and 20022 are bit-identical (both MKL), so a cell recorded against either
    # may be re-measured on either. Fold both onto one family name on *both* sides of
    # the comparison, so this filter no longer depends on the caller and the map having
    # independently agreed to write "10022" for a run that happened on 20022. Without
    # it, JOB_ORIGIN=20022 selects only the cells literally labelled 20022 and the
    # worker reports "0 jobs in scope", which reads exactly like "nothing left to do".
    function fam(m) { return (m == "20022") ? "10022" : m }
    BEGIN {
      while ((getline line < map) > 0) {
        if (line ~ /^#/ || line == "") continue
        if (split(line, f, "\t") >= 3) origin[f[1] "|" f[2]] = fam(f[3])
      }
      n = split(want, w, ","); for (i = 1; i <= n; i++) if (w[i] != "") keep[fam(w[i])] = 1
    }
    # A cell with no recorded origin cannot be machine-matched (no surviving run
    # reproduces its published value). Skipping it here is deliberate: it is a
    # provenance gap, reported as such rather than silently measured somewhere.
    { k = $2 "|" $1; if ((k in origin) && (origin[k] in keep)) print }')
fi

# Two boxes can belong to the SAME origin family (10022 and 20022 are bit-identical),
# in which case both are entitled to the same cells and the origin filter alone would
# have each of them run the whole list — the mkdir claim below only coordinates GPUs
# within one box, since the servers share no filesystem. JOB_SHARDS/JOB_SHARD_SET cut
# the already-origin-filtered list into disjoint slices, sized to the GPUs each box
# contributes. Both boxes derive the slice from the same job list and the same
# cell_origin.tsv, so the partition agrees without them talking to each other.
#   10022 (4 GPUs): JOB_SHARDS=6 JOB_SHARD_SET=0,1,2,3
#   20022 (2 GPUs): JOB_SHARDS=6 JOB_SHARD_SET=4,5
JOB_SHARDS=${JOB_SHARDS:-1}
JOB_SHARD_SET=${JOB_SHARD_SET:-0}
if [ "$JOB_SHARDS" -gt 1 ]; then
  JOBS=$(echo "$JOBS" | awk -v n="$JOB_SHARDS" -v set=",$JOB_SHARD_SET," \
           '{ if (index(set, "," (NR-1) % n ",")) print }')
fi

total=$(echo "$JOBS" | grep -c .)
echo "[gpu$GPU] $total jobs in scope (group=$GROUP, origin=${JOB_ORIGIN:-any}${ORPHAN_ORIGINS:+ +orphans:$ORPHAN_ORIGINS}, shard $JOB_SHARD_SET of $JOB_SHARDS), results -> $RES"

# Print the resolved job list and stop. Worth having: sending a cell to the wrong
# box silently corrupts its delta, and the filter is the only thing preventing it,
# so it should be checkable without launching anything.
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "$JOBS" | awk 'NF { printf "  %-12s %-16s seed%s\n", $1, $2, $3 }'
  exit 0
fi

# Reap abandoned claims before starting. A worker killed mid-job (a dropped VPN, a
# server going down) leaves a claim with no result behind it, and every later pass
# would skip that job forever — the sweep would report itself finished with a hole
# in it. Each claim records its owner's PID, so a claim whose owner is gone and
# which produced no result is safe to release. This only ever removes claims of
# dead processes, so it cannot race a live worker; taking the job back is still the
# atomic mkdir below.
for c in "$RES"/.claim_*; do
  [ -e "$c" ] || continue
  k=$(basename "$c"); k=${k#.claim_}
  [ -d "$RES/.done_$k" ] && continue
  owner=$(cat "$c/pid" 2>/dev/null || echo "")
  if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; then
    continue                      # owner loop alive — the job is genuinely in flight
  fi
  # A dead owner loop does NOT mean a dead job. Killing a worker leaves its
  # ``python -m hustbciml.run`` child running, reparented to init, still computing
  # and still writing its own metrics.json at the end. Releasing the claim on the
  # strength of the dead loop alone hands that same cell to a fresh worker, and two
  # GPUs then spend ~15 minutes each producing one number. Observed on 20022:
  # Noise-EEGNet/BNCI2014002 seed 1 ran on GPU6 and GPU7 simultaneously. So before
  # taking a claim back, confirm no process is still working on it.
  ds_k=${k%%_*}; rest=${k#*_}; algo_k=${rest%_seed*}; seed_k=${k##*_seed}
  if pgrep -u "$(id -un)" -f \
       "hustbciml\.run .*--algorithm $algo_k --dataset $ds_k .*--seed $seed_k " \
       >/dev/null 2>&1; then
    echo "[gpu$GPU] claim $k has no live owner but the job is still running — left alone"
    continue
  fi
  rm -rf "$c"; echo "[gpu$GPU] released abandoned claim: $k"
done

ran=0; skipped=0; failed=0
while read -r ds algo seed; do
  [ -z "${ds:-}" ] && continue
  key="${ds}_${algo}_seed${seed}"
  [ -d "$RES/.done_$key" ] && { skipped=$((skipped+1)); continue; }
  # Skip a cell that some process is already computing without holding a claim.
  # The atomic mkdir below is the normal defence, but it cannot see a job whose
  # claim was wrongly released while it kept running — exactly the state the reaper
  # above used to create. One pgrep is cheap next to fifteen minutes of GPU time.
  if pgrep -u "$(id -un)" -f \
       "hustbciml\.run .*--algorithm $algo --dataset $ds .*--seed $seed " \
       >/dev/null 2>&1; then
    echo "[gpu$GPU] $key already running in another process — skipping"
    skipped=$((skipped+1)); continue
  fi
  # Atomic claim: mkdir fails if another GPU's worker already took this job.
  mkdir "$RES/.claim_$key" 2>/dev/null || { skipped=$((skipped+1)); continue; }
  echo $$ > "$RES/.claim_$key/pid"
  echo "===== [gpu$GPU] $key  [$(date '+%F %H:%M:%S')] ====="
  # A retry of a job that died mid-write leaves a partial results folder behind,
  # and the overwrite guard would (correctly) refuse to write over it. We only get
  # here when the job has no .done_ marker, so clearing it is the intended retry.
  rm -rf "$RES/${ds}_cross_subject_${algo}_seed${seed}"
  if $PY -m hustbciml.run --algorithm "$algo" --dataset "$ds" --device cuda \
       --seed "$seed" --itr 1 --results_dir "$RES" --data_dir "$DATA" \
       > "$LOGS/$key.log" 2>&1; then
    mkdir -p "$RES/.done_$key"; ran=$((ran+1))
    tail -2 "$LOGS/$key.log"
  else
    # Leave no .done_ marker and drop the claim, so a later pass retries this job
    # instead of silently reporting a hole as a finished sweep. `rm -rf`, not `rmdir`:
    # the claim directory holds the owner's `pid` file, so rmdir always failed here —
    # silently, since its error went to /dev/null — and the claim outlived the failure.
    # The job then stayed claimed for the rest of the run and every worker skipped it;
    # only the reaper at the *next* launch could free it. Observed after 37 jobs failed
    # on GPU memory: 120 claims against 77 results, none of them retryable in place.
    rm -rf "$RES/.claim_$key"
    mkdir -p "$RES/.failed_$key"; failed=$((failed+1))
    echo "FAILED $key — see $LOGS/$key.log"
    tail -5 "$LOGS/$key.log"
  fi
done <<< "$JOBS"

echo "[gpu$GPU] DONE ran=$ran skipped=$skipped failed=$failed  [$(date '+%F %H:%M:%S')]"
