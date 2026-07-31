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

    python -m hustbciml.scripts.decentralized --dataset BNCI2014001 \
        --base hetero3 --seeds 1,2,3 --device cuda
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import uuid
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
from sklearn.metrics import accuracy_score

from hustbciml.core import registry
from hustbciml.core.config import resolve_config
from hustbciml.core.context import RunContext
from hustbciml.core.pipeline import build_pipeline
from hustbciml.data_provider.data_factory import get_epochs
from hustbciml.algorithms.strategies._common import forward_logits, supervised_train
from hustbciml.algorithms.ensembles import build_combiners, combiner_manifest
from hustbciml.utils.io import atomic_json_dump, atomic_savez
from hustbciml.utils.provenance import runtime_provenance
from hustbciml.utils.seed import fix_random_seed, resolve_device


def _load_aligned(cfg):
    """Load the dataset, inject data-derived dims, and EA-align every subject by
    its own reference (per-domain, label-free — no cross-subject leakage)."""
    epochs = get_epochs(cfg)
    cfg.n_chans = epochs.n_channels
    cfg.n_times = epochs.n_times
    cfg.n_classes = epochs.n_classes
    cfg.sfreq = epochs.sfreq
    cfg.ch_names = list(epochs.ch_names)
    cfg.classes = list(epochs.classes)
    cfg.data_provenance = dict(epochs.provenance or {})
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


def _base_hetero3(cfg, dev, epochs_a, subjects, C):
    """Heterogeneous single-source ensemble with THREE learners per source subject —
    Tangent+LR, CSPNet, EEGConformer — so target t is decided by (N-1)*3 HARD votes.

    Motivation: the homogeneous bases (all EEGNet, or all tangent+LDA) share one
    inductive bias, so their errors are strongly correlated and the spectral/crowd
    combiners — which need base models above chance AND roughly conditionally
    independent — have little to exploit (they collapse onto majority voting). This
    pool draws ONE learner from each of three families instead: a Riemannian
    tangent-space linear model, a CSP-initialized convolutional net, and a
    self-attention net. One member per family is what keeps the voters mutually
    decorrelated; adding a second member of a family raises the pool size but
    correlates it, which is the trade-off a wider pool loses on. Every learner emits
    a HARD one-hot vote, so only predicted labels ever leave a source
    (privacy-preserving).

    Caveat, stated because it bounds what the published table means: the two neural
    members are built by deep-copying the ``--algorithm`` preset and swapping
    ``backbone`` alone, so they train at that preset's learning rate and epoch
    ceiling rather than at the per-backbone values ``scripts/tune_networks.py``
    selects. Under the shipped ``EA-EEGNet`` preset (lr 1e-3, 100 epochs) that
    matters most for EEGConformer, whose tuning record selects 3e-4 on BNCI2014001
    and 1e-4 on the other two. These voters are therefore not the configuration
    published for the same names in the Networks table.
    """
    import copy
    from sklearn.linear_model import LogisticRegression

    backbones = ["CSPNet", "EEGConformer"]
    tang, lr, neural = {}, {}, {}
    for s in subjects:                                       # fit the 3 learners on each source
        e = epochs_a.select(epochs_a.domain == s)
        tang[s] = _tangent_at_identity(e.X)
        lr[s] = LogisticRegression(penalty="l2", max_iter=500).fit(tang[s], e.y)
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
    for t in subjects:                                      # (N-1)*3 HARD votes decide target t
        tgt = epochs_a.select(epochs_a.domain == t)
        ytrue[t] = tgt.y
        tf = _tangent_at_identity(tgt.X)
        for s in subjects:
            if s == t:
                continue
            scores[t][f"{s}::TangentLR"] = _onehot(lr[s].predict(tf), C)
            for bb in backbones:
                pred = forward_logits(neural[(s, bb)], tgt, dev).argmax(1)
                scores[t][f"{s}::{bb}"] = _onehot(pred, C)
    return ytrue, scores


def _seed_run(cfg, device, epochs_a, C, combiner_map, base):
    """Train one single-source learner per subject and fail on any combiner error."""
    fix_random_seed(cfg.seed)
    subjects = [int(s) for s in np.unique(epochs_a.domain)]

    if base == "tangent_lda":
        ytrue, scores = _base_tangent_lda(cfg, epochs_a, subjects, C)
    elif base == "hetero3":
        ytrue, scores = _base_hetero3(cfg, device, epochs_a, subjects, C)
    else:
        ytrue, scores = _base_eegnet(cfg, device, epochs_a, subjects)

    single = [accuracy_score(ytrue[target], values.argmax(1))
              for target in subjects for values in scores[target].values()]
    per_target = {name: {} for name in combiner_map}
    predictions = {name: {} for name in combiner_map}
    hard_votes = {}
    worker_ids = {}

    for target in subjects:
        worker_ids[target] = [str(name) for name in scores[target]]
        stack = np.stack(list(scores[target].values()))      # (learners, trials, classes)
        hard_votes[target] = stack.argmax(axis=2)
        for name, combiner in combiner_map.items():
            if combiner.binary_only and C != 2:
                raise ValueError(
                    f"requested combiner {name!r} is binary-only but dataset has {C} classes; "
                    "remove it explicitly rather than silently skipping a reported method"
                )
            prediction = combiner(stack)
            predictions[name][target] = prediction
            per_target[name][target] = float(
                accuracy_score(ytrue[target], prediction) * 100
            )

    means = {
        name: float(np.mean([per_target[name][target] for target in subjects]))
        for name in combiner_map
    }
    return {
        "single_source": float(np.mean(single) * 100),
        "combiners": means,
        "per_target": per_target,
        "subjects": subjects,
        "worker_ids": worker_ids,
        "y_true": ytrue,
        "hard_votes": hard_votes,
        "predictions": predictions,
    }


def main(argv=None):
    p = argparse.ArgumentParser(prog="hustbciml.scripts.decentralized",
                                description="decentralized single-source black-box ensemble")
    p.add_argument("--dataset", default="Toy")
    p.add_argument("--algorithm", default="EA-EEGNet", help="single-source base preset")
    p.add_argument("--base", default="eegnet", choices=["eegnet", "tangent_lda", "hetero3"])
    p.add_argument("--seeds", default="1,2,3", help="comma-separated seeds")
    p.add_argument("--device", default="auto")
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--combiners",
                   default="voting,Dawid-Skene,Wawa,M-MSR,MACE,GLAD,ZenCrowd,PM,"
                           "LA,LAA,EBCC,SML,SML-OVR,StackingNet")
    p.add_argument("--zencrowd_iters", type=int, default=20)
    p.add_argument("--pm_iters", type=int, default=3)
    p.add_argument("--allow_legacy_cache", action="store_true",
                   help="exploratory only: permit an unprovenanced cache and mark the "
                        "entire artifact is_measurement=false")
    p.add_argument("--overwrite", action="store_true",
                   help="deliberately replace an existing artifact with a different identity")
    a = p.parse_args(argv)

    seeds = [int(value) for value in a.seeds.split(",")]
    names = [value for value in a.combiners.split(",") if value]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must be non-empty and contain no duplicates")
    if not names:
        raise ValueError("--combiners must request at least one combiner")
    settings = {
        "ZenCrowd": {"n_iter": a.zencrowd_iters},
        "PM": {"n_iter": a.pm_iters},
    }
    combiners = build_combiners(names, settings=settings)
    combiner_config = combiner_manifest(combiners)

    cli = [
        "--algorithm", a.algorithm, "--dataset", a.dataset,
        "--seed", str(seeds[0]), "--device", a.device,
        "--data_dir", a.data_dir, "--results_dir", a.results_dir,
    ]
    if a.allow_legacy_cache:
        cli.append("--allow_legacy_cache")
    base_cfg, _ = resolve_config(cli)
    epochs_a, class_count = _load_aligned(base_cfg)
    data_is_measurement = base_cfg.data_provenance.get("is_measurement") is True
    if not data_is_measurement and not a.allow_legacy_cache:
        raise RuntimeError(
            "reportable ensemble measurements require a cache with explicit preprocessing "
            "provenance; pass --allow_legacy_cache only for an exploratory artifact"
        )
    data_reason = base_cfg.data_provenance.get(
        "reason", "dataset provenance is explicitly non-measurement"
    )
    runtime = runtime_provenance()
    device = resolve_device(base_cfg.device)
    base_cfg.resolved_device = str(device)

    base_learners = {
        "eegnet": ["EEGNet"],
        "tangent_lda": ["TangentSpace+shrinkage-LDA"],
        "hetero3": ["TangentSpace+LogisticRegression", "CSPNet", "EEGConformer"],
    }[a.base]
    base_config = dataclasses.asdict(base_cfg)
    base_config["seed"] = None
    identity = {
        "schema_version": 3,
        "dataset": a.dataset,
        "algorithm": a.algorithm,
        "base": a.base,
        "base_learners": base_learners,
        "requested_seeds": seeds,
        "combiners": combiner_config,
        "base_config": base_config,
        "data_sha256": base_cfg.data_provenance.get("content_sha256"),
        "source_sha256": runtime.get("source_sha256"),
        "python": runtime.get("python"),
        "platform": runtime.get("platform"),
        "machine": runtime.get("machine"),
        "dependencies": runtime.get("dependencies"),
        "numpy_build": runtime.get("numpy_build"),
        "numerical_libraries": runtime.get("numerical_libraries"),
        "torch_runtime": runtime.get("torch_runtime"),
    }

    os.makedirs(a.results_dir, exist_ok=True)
    out_path = os.path.join(
        a.results_dir, f"decentralized_{a.dataset}_{a.base}_{a.algorithm}.json"
    )
    archive = None
    if os.path.exists(out_path) and not a.overwrite:
        try:
            with open(out_path, encoding="utf-8") as handle:
                archive = json.load(handle)
        except Exception as exc:
            raise FileExistsError(f"{out_path} exists but is not readable strict JSON") from exc
        if archive.get("identity") != identity:
            raise FileExistsError(
                f"{out_path} belongs to a different or legacy measurement identity; "
                "preserve it and use another results directory"
            )
        if not isinstance(archive.get("seed_results"), dict):
            raise RuntimeError(f"{out_path} has no valid seed_results mapping")
        unexpected = sorted(set(archive["seed_results"]) - {str(seed) for seed in seeds})
        if unexpected:
            raise RuntimeError(f"{out_path} contains unrequested seed records {unexpected}")
    if archive is None or a.overwrite:
        archive = {
            "identity": identity,
            "is_measurement": False,
            "non_measurement_reason": (
                data_reason if not data_is_measurement else "requested seed set is incomplete"
            ),
            "provenance": {"runtime": runtime, "data": base_cfg.data_provenance},
            "seed_results": {},
            "summary": None,
        }

    def safe_key(value):
        return "".join(character if character.isalnum() else "_" for character in str(value))

    for seed in seeds:
        key = str(seed)
        prediction_name = (
            f"decentralized_{a.dataset}_{a.base}_{a.algorithm}_seed{seed}_predictions.npz"
        )
        prediction_path = os.path.join(a.results_dir, prediction_name)
        prediction_identity = {"archive_identity": identity, "seed": seed}
        if key in archive["seed_results"]:
            recorded_file = archive["seed_results"][key].get("predictions_file")
            if recorded_file != prediction_name or not os.path.exists(prediction_path):
                raise FileNotFoundError(
                    f"seed {seed} manifest does not identify an existing expected archive"
                )
            try:
                with np.load(prediction_path, allow_pickle=False) as stored:
                    stored_identity = json.loads(str(stored["identity_json"].item()))
                    stored_artifact_id = str(stored["artifact_id"].item())
            except Exception as exc:
                raise RuntimeError(
                    f"seed {seed} prediction archive has no readable identity"
                ) from exc
            if stored_identity != prediction_identity:
                raise RuntimeError(f"seed {seed} prediction archive identity is mismatched")
            if (
                not archive["seed_results"][key].get("artifact_id")
                or stored_artifact_id != archive["seed_results"][key]["artifact_id"]
            ):
                raise RuntimeError(
                    f"seed {seed} manifest and prediction archive come from different writes"
                )
            continue
        if os.path.exists(prediction_path) and not a.overwrite:
            raise FileExistsError(
                f"{prediction_path} exists without a matching seed manifest; preserve the "
                "partial artifact or pass --overwrite for deliberate replacement"
            )

        cfg = dataclasses.replace(base_cfg, seed=seed)
        result = _seed_run(cfg, device, epochs_a, class_count, combiners, a.base)
        artifact_id = uuid.uuid4().hex
        arrays = {
            "artifact_id": np.asarray(artifact_id, dtype="U"),
            "identity_json": np.asarray(
                json.dumps(prediction_identity, sort_keys=True), dtype="U"
            )
        }
        for target in result["subjects"]:
            arrays[f"target_{target}_y_true"] = np.asarray(result["y_true"][target])
            arrays[f"target_{target}_hard_votes"] = np.asarray(result["hard_votes"][target])
            for name in names:
                arrays[f"target_{target}_{safe_key(name)}_prediction"] = np.asarray(
                    result["predictions"][name][target]
                )
        atomic_savez(prediction_path, **arrays)
        archive["seed_results"][key] = {
            "artifact_id": artifact_id,
            "seed": seed,
            "single_source": result["single_source"],
            "combiners": result["combiners"],
            "per_target": {
                name: {str(target): value for target, value in values.items()}
                for name, values in result["per_target"].items()
            },
            "worker_ids": {
                str(target): values for target, values in result["worker_ids"].items()
            },
            "predictions_file": prediction_name,
        }
        completed = [value for value in seeds if str(value) in archive["seed_results"]]
        archive["completed_seeds"] = completed
        archive["is_measurement"] = False
        archive["non_measurement_reason"] = (
            data_reason if not data_is_measurement else
            ("summary not finalized" if completed == seeds
             else "requested seed set is incomplete")
        )
        atomic_json_dump(archive, out_path)
        print(
            f"[seed {seed}] single-source {result['single_source']:.2f} | "
            + " ".join(f"{name} {result['combiners'][name]:.2f}" for name in names)
        )

    if [int(value) for value in archive.get("completed_seeds", [])] != seeds:
        raise RuntimeError("not every requested seed completed; refusing to aggregate")
    single_all = [archive["seed_results"][str(seed)]["single_source"] for seed in seeds]
    combiner_all = {
        name: [archive["seed_results"][str(seed)]["combiners"][name] for seed in seeds]
        for name in names
    }
    if any(len(values) != len(seeds) for values in combiner_all.values()):
        raise RuntimeError("combiner seed coverage differs from the requested seed set")

    archive["summary"] = {
        "single_source": {"mean": float(np.mean(single_all)), "std": float(np.std(single_all))},
        "combiners": {
            name: {"mean": float(np.mean(values)), "std": float(np.std(values))}
            for name, values in combiner_all.items()
        },
    }
    archive["is_measurement"] = data_is_measurement
    archive["non_measurement_reason"] = None if data_is_measurement else data_reason
    atomic_json_dump(archive, out_path)

    base_desc = {
        "tangent_lda": "tangent-space + sLDA",
        "hetero3": "heterogeneous 3-learner/source",
    }.get(a.base, a.algorithm)
    print(f"\n=== decentralized ensemble: {base_desc} on {a.dataset}; seeds {seeds} ===")
    baseline = archive["summary"]["single_source"]
    print(f"single-source {baseline['mean']:.2f} +/- {baseline['std']:.2f}")
    for name in names:
        result = archive["summary"]["combiners"][name]
        print(
            f"{name:14s} {result['mean']:8.2f} {result['std']:7.2f}   "
            f"{result['mean'] - baseline['mean']:+.2f}"
        )


if __name__ == "__main__":
    main()
