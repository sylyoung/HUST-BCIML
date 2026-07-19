# tune_algorithm.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""General hyperparameter tuning for strategy/algorithm presets.

For each flagged method we sweep a small per-method grid of knobs — learning
rates, epochs, batch size, and the method-specific loss trade-offs / internal
LRs / capacities now exposed through ``--hp`` — pick ONE config by an honest
source-only signal, then evaluate that winner on seeds {1,2,3} and report the
TEST accuracy mean +/- std and mean kappa.

Selection signals (never the reported target/test set):

  select="val"  — held-out-source validation accuracy (``summary.val_primary``),
                  available for methods whose ``fit`` trains a source model with
                  the shared early-stopping trainer (ERM/nets, ABAT, CSDA). The
                  adversarial/augmentation knobs change that source model, so its
                  held-out accuracy is a valid, cheap selector.

  select="dev"  — mean accuracy over a few held-out SOURCE subjects used as
                  pseudo-targets (via ``--hp dev_targets=...``). Adaptation-phase
                  knobs (ASFA beta, Tent test_batch/steps, DJP-MMD mu, MSDT/LSFT/
                  MDMAML internals) do not move the source-val signal, so they are
                  selected on this leave-source-subjects-out proxy instead. One
                  global HP is chosen and then applied to the full cohort — the
                  standard practice, disclosed in the card. The real target
                  subjects' labels are never used for selection.

Each grid point runs as its own subprocess into its own results dir, so global
state (torch/CUDA, method caches) cannot leak across configs and one bad config
cannot kill the sweep. Resume-safe: a config whose ``metrics.json`` already
exists is reused.

    python -m hustbciml.scripts.tune_algorithm --dataset BNCI2015001 --device cuda \
        --methods ABAT,ASFA,Tent,DJP-MMD,MDMAML,MSDT,LSFT,CSDA-EEGNet \
        --results_dir /home/sylyoung/hustbciml_results_algotune \
        --data_dir /home/sylyoung/data
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import json
import os
import subprocess
import sys

# knobs that are first-class run.py CLI flags; everything else is passed as an
# ``--hp key=value`` method-specific override.
CLI_FLAGS = {
    "lr", "batch_size", "epochs", "test_batch", "steps", "stride", "temperature",
    "weight_decay", "val_ratio", "early_stop_patience", "dropout", "F1", "D", "F2",
}

# per-dataset subject count (0-indexed subjects), used to pick a spread of dev
# pseudo-targets for select="dev".
N_SUBJECTS = {"BNCI2014001": 9, "BNCI2014001-4": 9, "BNCI2014002": 14,
              "BNCI2015001": 12, "Toy": 4}


def _dev_spread(dataset: str, k: int = 3):
    """A spread of ``k`` subject ids across the cohort (not just the easy first
    few) to use as held-out dev pseudo-targets."""
    n = N_SUBJECTS.get(dataset, 9)
    if k >= n:
        return list(range(n))
    return sorted({int(round(i * (n - 1) / (k - 1))) for i in range(k)})


# method -> {preset, select, grid}. The grid is a dict knob->[values]; the tuner
# runs the Cartesian product. Defaults (omitted knobs) keep each method faithful.
METHODS = {
    # ---- source-model knobs: selectable by held-out-source validation ----
    "ABAT": dict(preset="ABAT", select="val",
                 grid={"lr": [3e-4, 1e-3, 3e-3], "abat_eps": [0.005, 0.01, 0.02]}),
    "CSDA-EEGNet": dict(preset="CSDA-EEGNet", select="val",
                        grid={"lr": [3e-4, 1e-3, 3e-3], "batch_size": [8, 32]}),
    # MVCNet is a composite (IFNet + multi-view contrastive). Its strategy overrides
    # the training loop and does NOT emit a held-out-source val_primary (empirically
    # None for every grid point), so select="val" has no signal. Select on the dev
    # pseudo-target proxy instead — the same honest source-only signal the adaptation
    # methods use. LR-only sweep to match the four-point lr grid of tune_networks;
    # lamda1/lamda2 kept at the preset 1.0.
    "MVCNet": dict(preset="MVCNet", select="dev",
                   grid={"lr": [1e-4, 3e-4, 1e-3, 3e-3]}),
    # ---- adaptation-phase knobs: selectable by the dev pseudo-target proxy ----
    "ASFA": dict(preset="ASFA", select="dev",
                 grid={"asfa_lr": [3e-3, 1e-2, 3e-2], "asfa_beta": [0.1, 0.3, 1.0]}),
    "Tent": dict(preset="Tent", select="dev",
                 grid={"lr": [1e-4, 1e-3], "test_batch": [8, 32], "steps": [1, 3]}),
    # BFT (backprop-free TTA). The prediction is a reliability-weighted average of
    # 12 label-preserving views; the aggregation temperature (bft_temp) and the
    # BN-refresh window (test_batch) only rescale/refresh and empirically never move
    # the argmax, so the meaningful knob is bft_lp_epochs — how long the reliability
    # predictor trains, which sets the per-view weights. Selected on the dev
    # pseudo-target proxy (adaptation knobs do not move the source-val signal).
    "BFT": dict(preset="BFT", select="dev",
                grid={"bft_lp_epochs": [5, 10, 20, 40], "bft_temp": [0.1, 0.25, 0.5]}),
    "DJP-MMD": dict(preset="DJP-MMD", select="dev",
                    grid={"lr": [3e-4, 1e-3, 3e-3], "djpmmd_mu": [0.1, 0.3, 1.0]}),
    "MDMAML": dict(preset="MDMAML", select="dev",
                   grid={"mdmaml_meta_lr": [3e-4, 1e-3, 3e-3], "mdmaml_inner_lr": [1e-3, 1e-2]}),
    "MSDT": dict(preset="MSDT", select="dev",
                 grid={"msdt_batch": [4, 8, 16, 32], "msdt_src_lr": [1e-2, 3e-2]}),
    "LSFT": dict(preset="LSFT", select="dev",
                 grid={"lsft_dim": [10, 20, 40], "lsft_mu": [0.1, 0.5]}),
}


def _tag(overrides: dict) -> str:
    """Short, stable directory tag for a config (order-independent)."""
    s = ",".join(f"{k}={overrides[k]}" for k in sorted(overrides))
    return hashlib.sha1(s.encode()).hexdigest()[:10] if s else "base"


def _grid_points(grid: dict):
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def _build_argv(preset, dataset, seed, device, rdir, data_dir, overrides, dev_targets=None):
    argv = ["--algorithm", preset, "--dataset", dataset, "--seed", str(seed),
            "--device", device, "--itr", "1", "--results_dir", rdir, "--data_dir", data_dir]
    for k, v in overrides.items():
        if k in CLI_FLAGS:
            argv += [f"--{k}", str(v)]
        else:
            argv += ["--hp", f"{k}={v}"]
    if dev_targets:
        argv += ["--hp", "dev_targets=" + ",".join(str(s) for s in dev_targets)]
    return argv


def _run_cfg(preset, dataset, seed, device, rdir, data_dir, overrides, dev_targets=None):
    """Run one config as a subprocess into its own dir; reuse if already done."""
    os.makedirs(rdir, exist_ok=True)
    found = glob.glob(os.path.join(rdir, "*", "metrics.json"))
    if found:
        return json.load(open(found[0]))
    argv = _build_argv(preset, dataset, seed, device, rdir, data_dir, overrides, dev_targets)
    env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
           "OPENBLAS_NUM_THREADS": "4"}
    r = subprocess.run([sys.executable, "-m", "hustbciml.run"] + argv,
                       env=env, capture_output=True, text=True)
    found = glob.glob(os.path.join(rdir, "*", "metrics.json"))
    if found:
        return json.load(open(found[0]))
    print(f"[error] {preset} {overrides} rc={r.returncode}\n{r.stderr[-1000:]}", flush=True)
    return None


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _std_pop(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def _sel_score(m, select):
    """The scalar used to rank a config: held-out-source val (select=val) or the
    mean accuracy over the dev pseudo-targets (select=dev)."""
    sm = (m or {}).get("summary", {})
    if select == "val":
        return ((sm.get("val_primary") or {}).get("mean"))
    return ((sm.get("primary") or {}).get("mean"))


def tune_one(name, spec, a):
    preset, select, grid = spec["preset"], spec["select"], spec["grid"]
    # Phase-1 selection always runs on a small spread of held-out SOURCE subjects
    # (dev pseudo-targets): for select="dev" that spread IS the signal (mean
    # primary), for select="val" it just bounds selection cost while the honest
    # held-out-source val accuracy is read per fold. Either way selection never
    # touches the full reported cohort.
    sel_dev = _dev_spread(a.dataset, a.dev_k)

    # ---- phase 1: grid, select by the honest source-only signal ----
    scored = []
    for ov in _grid_points(grid):
        rdir = os.path.join(a.results_dir, "grid", a.dataset, name, _tag(ov))
        m = _run_cfg(preset, a.dataset, a.sel_seed, a.device, rdir, a.data_dir, ov, sel_dev)
        sc = _sel_score(m, select)
        scored.append((ov, sc))
        print(f"[grid ] {a.dataset} {name:12s} {select} {ov} -> {sc}", flush=True)
    cand = [(ov, sc) for ov, sc in scored if sc is not None]
    if not cand:
        print(f"[skip ] {a.dataset} {name}: no successful grid point", flush=True)
        return None
    best_ov = max(cand, key=lambda t: t[1])[0]

    # ---- phase 2: evaluate the winner on the reported seeds (full cohort, test) ----
    accs, kaps = [], []
    for s in [int(x) for x in a.seeds.split(",")]:
        rdir = os.path.join(a.results_dir, "final", a.dataset, name, f"seed{s}")
        m = _run_cfg(preset, a.dataset, s, a.device, rdir, a.data_dir, best_ov, dev_targets=None)
        sm = (m or {}).get("summary", {})
        if sm.get("primary"):
            accs.append(sm["primary"]["mean"])
        if sm.get("kappa"):
            kaps.append(sm["kappa"]["mean"])
    res = {
        "preset": preset, "select": select, "best_config": best_ov,
        "dev_targets": sel_dev, "grid": [(ov, sc) for ov, sc in scored],
        "acc_mean": round(_mean(accs), 2) if accs else None,
        "acc_std": round(_std_pop(accs), 2),
        "kappa_mean": round(_mean(kaps), 3) if kaps else None,
        "per_seed_acc": [round(x, 2) for x in accs],
    }
    print(f"[final] {a.dataset} {name:12s} best={best_ov} "
          f"acc={res['acc_mean']}+/-{res['acc_std']} kappa={res['kappa_mean']}", flush=True)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(prog="hustbciml.scripts.tune_algorithm")
    p.add_argument("--dataset", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--results_dir", required=True)
    p.add_argument("--data_dir", default="/home/sylyoung/data")
    p.add_argument("--methods", default=",".join(METHODS),
                   help="comma-separated method names (keys of METHODS)")
    p.add_argument("--sel_seed", type=int, default=1)     # seed used for HP selection
    p.add_argument("--seeds", default="1,2,3")            # seeds for the reported test numbers
    p.add_argument("--dev_k", type=int, default=3)        # number of dev pseudo-target subjects
    a = p.parse_args(argv)

    out_path = os.path.join(a.results_dir, f"tuned_{a.dataset}.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    os.makedirs(a.results_dir, exist_ok=True)

    for name in [m for m in a.methods.split(",") if m]:
        if name not in METHODS:
            print(f"[warn ] unknown method {name!r}; known: {sorted(METHODS)}", flush=True)
            continue
        res = tune_one(name, METHODS[name], a)
        if res is not None:
            out.setdefault(a.dataset, {})[name] = res
            json.dump(out, open(out_path, "w"), indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
