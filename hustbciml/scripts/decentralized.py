# decentralized.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Decentralized single-source black-box ensemble.

The privacy-preserving recast of the multi-seed ensemble: instead of K random
seeds of one model trained on the pooled sources, each SOURCE subject trains its
own model on ONLY its own EEG (never sharing raw data). Under leave-one-subject-out,
for target subject t the models of the other subjects each predict t, and their
per-trial HARD votes are fused by the same post-hoc black-box combiners as the
multi-seed ensemble — hard majority voting (the baseline), the crowdsourcing
aggregators Dawid-Skene / Wawa / M-MSR / MACE / GLAD / ZenCrowd / PM / LA / LAA /
EBCC, and the lab's SML / SML-OVR / StackingNet (see ``algorithms/ensembles/``). There is
no soft-score averaging combiner: every method sees only hard votes, so none has an
information advantage. The diversity that the combiners exploit now comes from the
subjects themselves, and no source data ever leaves its owner.

Because Euclidean Alignment is per-subject and label-free, every subject is aligned
by its own reference once, one EEGNet is trained per subject (N trainings, not
N*(N-1)), and for each target the other N-1 models are aggregated. Reports each
combiner's accuracy mean +/- std across seeds, plus the mean single-source model
accuracy (one local model alone, averaged over all source->target pairs) as context.

    python -m hustbciml.scripts.decentralized --dataset BNCI2014001-4 \
        --seeds 1,2,3 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.metrics import accuracy_score

from hustbciml.core import registry
from hustbciml.core.config import resolve_config
from hustbciml.core.context import RunContext
from hustbciml.core.pipeline import build_pipeline
from hustbciml.data_provider.data_factory import get_epochs
from hustbciml.algorithms.strategies._common import forward_logits, supervised_train
from hustbciml.algorithms.ensembles import build_combiners
from hustbciml.utils.seed import fix_random_seed, resolve_device

# name -> combiner instance, auto-discovered from algorithms/ensembles/ (one file per method).
COMBINERS = build_combiners()


def _load_aligned(cfg):
    """Load the dataset, inject data-derived dims, and EA-align every subject by
    its own reference (per-domain, label-free — no cross-subject leakage)."""
    epochs = get_epochs(cfg)
    cfg.n_chans = epochs.n_channels
    cfg.n_times = epochs.n_times
    cfg.n_classes = epochs.n_classes
    cfg.sfreq = epochs.sfreq
    cfg.ch_names = list(epochs.ch_names)
    aligner = registry.build("aligners", cfg.aligner)
    aligner.fit(epochs)
    return aligner.transform(epochs), epochs.n_classes


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _tangent_at_identity(X: np.ndarray) -> np.ndarray:
    """Wen Zhang's tangent-space features (MEKT/MSDT): per-trial OAS covariance
    mapped to the Riemannian tangent space at the IDENTITY reference. Because the
    trials are already Euclidean-aligned (each subject whitened to an ~identity
    covariance reference), a fixed identity reference keeps the tangent vectors
    directly comparable across subjects, so a source classifier transfers to the
    target — the alignment-then-tangent recipe of MEKT."""
    from pyriemann.estimation import Covariances
    from pyriemann.utils.tangentspace import tangent_space
    cov = Covariances(estimator="oas").transform(X.astype(np.float64))
    C = cov.shape[-1]
    return tangent_space(cov, np.eye(C))                    # (N, C(C+1)/2), fixed reference


def _onehot(pred: np.ndarray, C: int) -> np.ndarray:
    """Hard class labels -> one-hot rows (N, C). The ensemble uses HARD votes
    only (never class probabilities), so each source learner emits a one-hot of
    its predicted class, not a soft score."""
    return np.eye(C, dtype=np.float64)[np.asarray(pred, dtype=int)]


def _base_eegnet(cfg, dev, epochs_a, subjects):
    """Original base: one EEGNet per subject -> per-target softmax scores."""
    models = {}
    for s in subjects:
        pipe = build_pipeline(cfg)
        model = pipe.model.to(dev)
        ctx = RunContext(cfg=cfg, device=dev, augmenter=pipe.augmenter,
                         aligner=pipe.aligner, log=lambda m: None)
        supervised_train(model, epochs_a.select(epochs_a.domain == s), ctx)
        models[s] = model
    ytrue, scores = {}, {t: {} for t in subjects}
    for t in subjects:
        tgt = epochs_a.select(epochs_a.domain == t)
        ytrue[t] = tgt.y
        for s in subjects:
            if s != t:
                scores[t][s] = _softmax(forward_logits(models[s], tgt, dev))
    return ytrue, scores


def _base_tangent_lda(cfg, epochs_a, subjects, C):
    """Redesigned base (Wen Zhang tangent-space + shrinkage LDA): per source
    subject, fit an sLDA on its EA-aligned tangent features; each source learner
    predicts HARD class labels on the target's tangent features (one-hot, never
    probabilities). No neural network, no seed dependence in the base."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    feats, clfs = {}, {}
    for s in subjects:                                   # per-subject tangent map + sLDA
        e = epochs_a.select(epochs_a.domain == s)
        feats[s] = _tangent_at_identity(e.X)
        clfs[s] = (LDA(solver="lsqr", shrinkage="auto").fit(feats[s], e.y), e.y)
    ytrue, scores = {}, {t: {} for t in subjects}
    for t in subjects:
        ytrue[t] = epochs_a.select(epochs_a.domain == t).y
        for s in subjects:
            if s != t:
                pred = clfs[s][0].predict(feats[t])      # HARD labels on target tangent feats
                scores[t][s] = _onehot(pred, C)
    return ytrue, scores


def _base_hetero(cfg, dev, epochs_a, subjects, C):
    """Heterogeneous single-source ensemble: FIVE diverse learners per source
    subject — Tangent+LDA, Tangent+SVM, EEGNet, ShallowConvNet, CSPNet — so each
    source contributes five conditionally-diverse HARD votes and target t is
    decided by (N-1)*5 predictions.

    Motivation: the homogeneous bases (all EEGNet, or all tangent+LDA) share one
    inductive bias, so their errors are strongly correlated and the spectral/crowd
    combiners — which need base models above chance AND roughly conditionally
    independent — have little to exploit (they collapse onto majority voting). A
    Riemannian-classical pair plus three different CNN priors raises the base
    diversity, giving the aggregators more to work with. Every learner emits a HARD
    one-hot vote, so only predicted labels ever leave a source (privacy-preserving).
    """
    import copy
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.svm import SVC

    backbones = ["EEGNet", "ShallowConvNet", "CSPNet"]
    tang, lda, svm, neural = {}, {}, {}, {}
    for s in subjects:                                       # fit the 5 learners on each source
        e = epochs_a.select(epochs_a.domain == s)
        tang[s] = _tangent_at_identity(e.X)
        lda[s] = LDA(solver="lsqr", shrinkage="auto").fit(tang[s], e.y)
        svm[s] = SVC(kernel="rbf", C=1.0, gamma="scale").fit(tang[s], e.y)
        for bb in backbones:
            cfg_bb = copy.deepcopy(cfg)
            cfg_bb.backbone = bb                            # same EA-aligned data, different net
            pipe = build_pipeline(cfg_bb)
            model = pipe.model.to(dev)
            ctx = RunContext(cfg=cfg_bb, device=dev, augmenter=pipe.augmenter,
                             aligner=pipe.aligner, log=lambda m: None)
            supervised_train(model, e, ctx)
            neural[(s, bb)] = model

    ytrue, scores = {}, {t: {} for t in subjects}
    for t in subjects:                                      # (N-1)*5 HARD votes decide target t
        tgt = epochs_a.select(epochs_a.domain == t)
        ytrue[t] = tgt.y
        tf = _tangent_at_identity(tgt.X)
        for s in subjects:
            if s == t:
                continue
            scores[t][f"{s}::TangentLDA"] = _onehot(lda[s].predict(tf), C)
            scores[t][f"{s}::TangentSVM"] = _onehot(svm[s].predict(tf), C)
            for bb in backbones:
                pred = forward_logits(neural[(s, bb)], tgt, dev).argmax(1)
                scores[t][f"{s}::{bb}"] = _onehot(pred, C)
    return ytrue, scores


def _seed_run(dataset, seed, device, data_dir, results_dir, algorithm, combiners, base):
    """Train one single-source learner per subject, then fuse per target."""
    cfg, _ = resolve_config(["--algorithm", algorithm, "--dataset", dataset,
                             "--seed", str(seed), "--device", device,
                             "--data_dir", data_dir, "--results_dir", results_dir])
    fix_random_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    epochs_a, C = _load_aligned(cfg)
    subjects = [int(s) for s in np.unique(epochs_a.domain)]

    if base == "tangent_lda":
        ytrue, scores = _base_tangent_lda(cfg, epochs_a, subjects, C)
    elif base == "hetero":
        ytrue, scores = _base_hetero(cfg, dev, epochs_a, subjects, C)
    else:
        ytrue, scores = _base_eegnet(cfg, dev, epochs_a, subjects)

    # generic over the per-target learners: {s} for the homogeneous bases,
    # {s}::{learner} for hetero — so (N-1) or (N-1)*5 votes per target.
    single = [accuracy_score(ytrue[t], v.argmax(1))
              for t in subjects for v in scores[t].values()]

    out = {}
    for c in combiners:
        if c == "SML" and C != 2:                # binary SML undefined for >2 classes
            continue
        fn = COMBINERS[c]
        if c == "SML-OVR" and C == 2:            # SML-OVR is the multi-class (K>2) one-vs-rest
            fn = COMBINERS["SML"]                 # extension of SML; on 2 classes it reduces exactly
                                                  # to binary SML, so report that number (identical to
                                                  # the SML row) instead of skipping. The native multi-
                                                  # class path (C>2, e.g. BNCI2014001-4) still runs
                                                  # sml_ovr — the multi-class option is kept in code.
        try:
            accs = []
            for t in subjects:
                stack = np.stack(list(scores[t].values()))   # (n_learners, N, C)
                accs.append(accuracy_score(ytrue[t], fn(stack)))
            out[c] = float(np.mean(accs) * 100)
        except Exception as e:                   # a degenerate combiner must not kill the run;
            print(f"[warn] combiner {c!r} failed, skipping it — {type(e).__name__}: {e}")
    return float(np.mean(single) * 100), out


def main(argv=None):
    p = argparse.ArgumentParser(prog="hustbciml.scripts.decentralized",
                                description="decentralized single-source black-box ensemble")
    p.add_argument("--dataset", default="Toy")
    p.add_argument("--algorithm", default="EA-EEGNet", help="single-source base preset (EEGNet base)")
    p.add_argument("--base", default="eegnet", choices=["eegnet", "tangent_lda", "hetero"],
                   help="per-source learner: 'eegnet' (softmax scores), 'tangent_lda' "
                        "(Wen Zhang EA+tangent-space features + shrinkage LDA, HARD votes), or "
                        "'hetero' (5 diverse learners per source — Tangent+LDA, Tangent+SVM, "
                        "EEGNet, ShallowConvNet, CSPNet — giving (N-1)*5 HARD votes per target)")
    p.add_argument("--seeds", default="1,2,3", help="comma-separated seeds")
    p.add_argument("--device", default="auto")
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--combiners",
                   default="voting,Dawid-Skene,Wawa,M-MSR,MACE,GLAD,ZenCrowd,PM,"
                           "LA,LAA,EBCC,SML,SML-OVR,StackingNet")
    a = p.parse_args(argv)

    seeds = [int(s) for s in a.seeds.split(",")]
    combiners = [c for c in a.combiners.split(",") if c]
    single_all, comb_all = [], {c: [] for c in combiners}
    os.makedirs(a.results_dir, exist_ok=True)
    out_path = os.path.join(a.results_dir, f"decentralized_{a.dataset}_{a.base}_{a.algorithm}.json")

    for seed in seeds:
        single, out = _seed_run(a.dataset, seed, a.device, a.data_dir,
                                a.results_dir, a.algorithm, combiners, a.base)
        single_all.append(single)
        for c in out:
            comb_all[c].append(out[c])
        json.dump({"dataset": a.dataset, "algorithm": a.algorithm,
                   "seeds_done": seeds[:len(single_all)],
                   "single_source": single_all, "combiners": comb_all},
                  open(out_path, "w"), indent=2)     # incremental save (flaky-link safe)
        print(f"[seed {seed}] single-source {single:.2f} | "
              + " ".join(f"{c} {out[c]:.2f}" for c in out))

    def ms(v):
        arr = np.array(v)
        return arr.mean(), arr.std()

    base_desc = {"tangent_lda": "tangent-space + sLDA (Wen Zhang)",
                 "hetero": "heterogeneous 5-learner/source (Tangent+LDA, Tangent+SVM, "
                           "EEGNet, ShallowConvNet, CSPNet)"}.get(a.base, a.algorithm)
    print(f"\n=== decentralized single-source ensemble: {base_desc} on {a.dataset} "
          f"— {len(seeds)} seeds {seeds} ===")
    m, s = ms(single_all)
    print(f"single-source model (mean over source->target pairs): {m:.2f} +/- {s:.2f}")
    print(f"{'combiner':14s} {'acc':>8s} {'std':>7s}   delta-vs-single")
    for c in combiners:
        if not comb_all[c]:
            reason = "binary only" if c == "SML" else "failed — see [warn] above"
            print(f"{c:14s}   (skipped: {reason})")
            continue
        m2, s2 = ms(comb_all[c])
        print(f"{c:14s} {m2:8.2f} {s2:7.2f}   {m2 - m:+.2f}")


if __name__ == "__main__":
    main()
