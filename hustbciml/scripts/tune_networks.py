# tune_networks.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Per-backbone hyperparameter tuning for the Network comparison.

For each backbone the learning rate is tuned over a grid and the training length is
tuned implicitly by validation-based early stopping over a long epoch ceiling. The
selection signal is the held-out-source VALIDATION accuracy
(``summary["val_primary"]`` recorded by Exp_CrossSubject) — never the target/test
set — so there is no test peeking. The learning rate with the best mean validation
accuracy is then re-run for ``--seeds`` seeds and the TEST accuracy mean +/- std and
mean kappa are reported. Each run uses its own results dir (the stage setting string
does not encode the learning rate, so distinct dirs prevent grid runs from
colliding); resume-safe (a run whose metrics.json exists is reused).

    python -m hustbciml.scripts.tune_networks --dataset BNCI2014001 --device cuda \
        --results_dir /home/sylyoung/hustbciml_results_nettune --data_dir /home/sylyoung/data
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from hustbciml import run as run_module

# backbone display name -> its --backbone value (EA + Linear + ERM held fixed)
BACKBONES = {
    "EEGNet": "EEGNet",
    "ShallowConvNet": "ShallowConvNet",
    "DeepConvNet": "DeepConvNet",
    "EEGConformer": "EEGConformer",
    "DBConformer": "DBConformer",
    "CSP-Net": "CSPNet",
    "TIE-EEGNet": "TIEEEGNet",
    "KDFNet": "KDFNet",
}


def _tag(lr: float) -> str:
    return ("%g" % lr).replace("-", "m").replace(".", "p")


def _run(dataset, backbone, lr, epochs, seed, device, rdir, data_dir):
    """Run one config into its own results dir; reuse if already done."""
    os.makedirs(rdir, exist_ok=True)
    found = glob.glob(os.path.join(rdir, "*", "metrics.json"))
    if found:
        return json.load(open(found[0]))
    try:
        run_module.main(["--aligner", "EA", "--backbone", backbone, "--head", "Linear",
                         "--strategy", "ERM", "--dataset", dataset, "--lr", str(lr),
                         "--epochs", str(epochs), "--seed", str(seed), "--device", device,
                         "--itr", "1", "--results_dir", rdir, "--data_dir", data_dir])
    except Exception as e:      # one bad config must not kill the whole sweep
        print(f"[error] {dataset} {backbone} lr={lr} seed={seed}: {type(e).__name__}: {e}", flush=True)
        return None
    found = glob.glob(os.path.join(rdir, "*", "metrics.json"))
    return json.load(open(found[0])) if found else None


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _std_pop(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def main(argv=None):
    p = argparse.ArgumentParser(prog="hustbciml.scripts.tune_networks")
    p.add_argument("--dataset", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--results_dir", required=True)
    p.add_argument("--data_dir", default="/home/sylyoung/data")
    p.add_argument("--backbones", default=",".join(BACKBONES))
    p.add_argument("--lrs", default="0.0001,0.0003,0.001,0.003")
    p.add_argument("--epochs", type=int, default=300)   # ceiling; early stopping tunes the length
    p.add_argument("--sel_seed", type=int, default=1)   # seed used for LR selection
    p.add_argument("--seeds", default="1,2,3")          # seeds used for the reported test numbers
    a = p.parse_args(argv)

    lrs = [float(x) for x in a.lrs.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    backbones = [b for b in a.backbones.split(",") if b]
    out_path = os.path.join(a.results_dir, f"tuned_{a.dataset}.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}

    for b in backbones:
        bb = BACKBONES[b]
        # ---- phase 1: LR grid, select by held-out validation accuracy ----
        val_by_lr = {}
        for lr in lrs:
            rdir = os.path.join(a.results_dir, "grid", a.dataset, b, f"lr{_tag(lr)}")
            m = _run(a.dataset, bb, lr, a.epochs, a.sel_seed, a.device, rdir, a.data_dir)
            vp = ((m or {}).get("summary", {}).get("val_primary") or {}).get("mean")
            val_by_lr["%g" % lr] = vp
            print(f"[grid ] {a.dataset} {b:15s} lr={lr:<7g} val={vp}", flush=True)
        cand = {lr: val_by_lr["%g" % lr] for lr in lrs if val_by_lr["%g" % lr] is not None}
        best_lr = max(cand, key=cand.get) if cand else lrs[len(lrs) // 2]

        # ---- phase 2: evaluate the selected LR on the reported seeds (test) ----
        accs, kaps = [], []
        for s in seeds:
            rdir = os.path.join(a.results_dir, "final", a.dataset, b, f"seed{s}")
            m = _run(a.dataset, bb, best_lr, a.epochs, s, a.device, rdir, a.data_dir)
            sm = (m or {}).get("summary", {})
            if sm.get("primary"):
                accs.append(sm["primary"]["mean"])
            if sm.get("kappa"):
                kaps.append(sm["kappa"]["mean"])
        out.setdefault(a.dataset, {})[b] = {
            "best_lr": best_lr, "val_by_lr": val_by_lr,
            "acc_mean": round(_mean(accs), 2) if accs else None,
            "acc_std": round(_std_pop(accs), 2),
            "kappa_mean": round(_mean(kaps), 3) if kaps else None,
            "per_seed_acc": [round(x, 2) for x in accs],
        }
        json.dump(out, open(out_path, "w"), indent=2)
        r = out[a.dataset][b]
        print(f"[final] {a.dataset} {b:15s} best_lr={best_lr:<7g} "
              f"acc={r['acc_mean']}+/-{r['acc_std']} kappa={r['kappa_mean']}", flush=True)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
