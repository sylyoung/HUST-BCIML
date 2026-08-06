# combined_ensemble.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Source-combined heterogeneous ensemble (the NON-decentralized paradigm).

The counterpart to ``decentralized.py --base hetero3``, which trains three models per
source subject and aggregates (N-1)*3 hard votes per target. This paradigm instead
trains FIVE architectures — EEGNet, ShallowFBCSPNet-AT, Deep4Net-AT, EEGConformer,
CSPNet — each ONCE on the POOLED source (all subjects except the target, Euclidean-
aligned), giving five per-trial predictions on the target. The diversity the
combiners exploit here comes from the heterogeneous *architectures* (different
inductive biases), not from the subjects.

Under leave-one-subject-out, for each target subject t the five nets are trained on
the union of the other subjects (5 * N trainings across a run) and each predicts t.
Aggregation is over HARD labels only — each net contributes a single predicted class
per trial, never a soft score — so hard majority voting is the baseline, alongside
the same crowd/lab combiners as the decentralized ensemble (Dawid-Skene / Wawa /
M-MSR / MACE / GLAD / ZenCrowd / PM / LA / LAA / EBCC and the lab's SML / SML-OVR /
StackingNet). Restricting to hard labels keeps the two paradigms comparable
column-for-column: the decentralized paradigm exposes only hard votes, so no
soft-averaging baseline is reported here either.

    python -m hustbciml.scripts.combined_ensemble --dataset BNCI2014001 \
        --seeds 1,2,3 --device cuda
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import uuid
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
from sklearn.metrics import accuracy_score

from hustbciml.core.config import resolve_config
from hustbciml.core.context import RunContext
from hustbciml.core.pipeline import build_pipeline
from hustbciml.algorithms.strategies._common import forward_logits, supervised_train
from hustbciml.algorithms.ensembles import build_combiners, combiner_manifest
from hustbciml.scripts.decentralized import _load_aligned, _onehot
from hustbciml.utils.io import atomic_json_dump, atomic_savez
from hustbciml.utils.provenance import runtime_provenance
from hustbciml.utils.seed import fix_random_seed, resolve_device


# Five heterogeneous backbones (distinct inductive biases): EEGNet (compact
# depthwise-separable conv), ShallowFBCSPNet-AT (shallow FBCSP-style), Deep4Net-AT
# (four reference convolution blocks), EEGConformer (conv stem + transformer),
# CSPNet (CSP-initialized spatial conv). Each trains on the same pooled EA-aligned source.
BACKBONES = ["EEGNet", "ShallowFBCSPNetAT", "Deep4NetAT", "EEGConformer", "CSPNet"]


def _combined_scores(cfg, dev, epochs_a, subjects, C, backbones):
    """LOSO source-combined training: for each target t, train each backbone on the
    POOLED other subjects (EA-aligned) and predict t. Returns per-target ground truth
    and hard one-hot votes — one per backbone (hard labels only, no soft scores)."""
    ytrue = {}
    hard = {t: {} for t in subjects}
    for t in subjects:
        src = epochs_a.select(epochs_a.domain != t)          # pooled source (all but t)
        tgt = epochs_a.select(epochs_a.domain == t)
        ytrue[t] = tgt.y
        for bb in backbones:
            cfg_bb = copy.deepcopy(cfg)
            cfg_bb.backbone = bb                              # same pooled data, different net
            pipe = build_pipeline(cfg_bb)
            model = pipe.model.to(dev)
            ctx = RunContext(cfg=cfg_bb, device=dev, augmenter=pipe.augmenter,
                             aligner=pipe.aligner, log=lambda m: None)
            supervised_train(model, src, ctx)
            logits = forward_logits(model, tgt, dev)
            hard[t][bb] = _onehot(logits.argmax(1), C)       # single predicted class per trial
    return ytrue, hard


def _seed_run(cfg, device, epochs_a, C, combiner_map, backbones):
    fix_random_seed(cfg.seed)
    subjects = [int(value) for value in np.unique(epochs_a.domain)]
    ytrue, hard = _combined_scores(cfg, device, epochs_a, subjects, C, backbones)
    single = [accuracy_score(ytrue[target], vote.argmax(1))
              for target in subjects for vote in hard[target].values()]

    per_target = {name: {} for name in combiner_map}
    predictions = {name: {} for name in combiner_map}
    hard_votes = {}
    for target in subjects:
        stack = np.stack(list(hard[target].values()))
        hard_votes[target] = stack.argmax(axis=2)
        for name, combiner in combiner_map.items():
            if combiner.binary_only and C != 2:
                raise ValueError(
                    f"requested combiner {name!r} is binary-only but dataset has {C} classes"
                )
            prediction = combiner(stack)
            predictions[name][target] = prediction
            per_target[name][target] = float(accuracy_score(ytrue[target], prediction) * 100)
    means = {
        name: float(np.mean(list(values.values()))) for name, values in per_target.items()
    }
    return {
        "single_model": float(np.mean(single) * 100),
        "combiners": means,
        "per_target": per_target,
        "subjects": subjects,
        "y_true": ytrue,
        "hard_votes": hard_votes,
        "predictions": predictions,
    }


def main(argv=None):
    p = argparse.ArgumentParser(prog="hustbciml.scripts.combined_ensemble",
                                description="source-combined heterogeneous ensemble")
    p.add_argument("--dataset", default="Toy")
    p.add_argument("--algorithm", default="EA-EEGNet")
    p.add_argument("--backbones", default=",".join(BACKBONES))
    p.add_argument("--seeds", default="1,2,3")
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
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)

    seeds = [int(value) for value in a.seeds.split(",")]
    backbones = [value for value in a.backbones.split(",") if value]
    names = [value for value in a.combiners.split(",") if value]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must be non-empty and contain no duplicates")
    if not backbones or len(set(backbones)) != len(backbones):
        raise ValueError("--backbones must be non-empty and contain no duplicates")
    if not names:
        raise ValueError("--combiners must request at least one combiner")
    combiners = build_combiners(names, settings={
        "ZenCrowd": {"n_iter": a.zencrowd_iters},
        "PM": {"n_iter": a.pm_iters},
    })
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
    base_config = dataclasses.asdict(base_cfg)
    base_config["seed"] = None
    identity = {
        "schema_version": 3,
        "dataset": a.dataset,
        "algorithm": a.algorithm,
        "base": "source_combined_heterogeneous",
        "backbones": backbones,
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
    out_path = os.path.join(a.results_dir, f"combined_{a.dataset}_hetero_{a.algorithm}.json")
    archive = None
    if os.path.exists(out_path) and not a.overwrite:
        try:
            with open(out_path, encoding="utf-8") as handle:
                archive = json.load(handle)
        except Exception as exc:
            raise FileExistsError(f"{out_path} exists but is not readable strict JSON") from exc
        if archive.get("identity") != identity:
            raise FileExistsError(
                f"{out_path} belongs to a different or legacy identity; preserve it and "
                "use another results directory"
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
            "seed_results": {}, "summary": None,
        }

    def safe_key(value):
        return "".join(character if character.isalnum() else "_" for character in str(value))

    for seed in seeds:
        key = str(seed)
        prediction_name = f"combined_{a.dataset}_hetero_{a.algorithm}_seed{seed}_predictions.npz"
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
        result = _seed_run(cfg, device, epochs_a, class_count, combiners, backbones)
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
            "single_model": result["single_model"],
            "combiners": result["combiners"],
            "per_target": {
                name: {str(target): value for target, value in values.items()}
                for name, values in result["per_target"].items()
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
            f"[seed {seed}] single-model {result['single_model']:.2f} | "
            + " ".join(f"{name} {result['combiners'][name]:.2f}" for name in names)
        )

    if [int(value) for value in archive.get("completed_seeds", [])] != seeds:
        raise RuntimeError("not every requested seed completed; refusing to aggregate")
    single_all = [archive["seed_results"][str(seed)]["single_model"] for seed in seeds]
    combiner_all = {
        name: [archive["seed_results"][str(seed)]["combiners"][name] for seed in seeds]
        for name in names
    }
    if any(len(values) != len(seeds) for values in combiner_all.values()):
        raise RuntimeError("combiner seed coverage differs from requested seeds")
    archive["summary"] = {
        "single_model": {"mean": float(np.mean(single_all)), "std": float(np.std(single_all))},
        "combiners": {
            name: {"mean": float(np.mean(values)), "std": float(np.std(values))}
            for name, values in combiner_all.items()
        },
    }
    archive["is_measurement"] = data_is_measurement
    archive["non_measurement_reason"] = None if data_is_measurement else data_reason
    atomic_json_dump(archive, out_path)

    print(f"\n=== source-combined heterogeneous ensemble ({'/'.join(backbones)}) "
          f"on {a.dataset}; seeds {seeds} ===")
    baseline = archive["summary"]["single_model"]
    print(f"single model {baseline['mean']:.2f} +/- {baseline['std']:.2f}")
    for name in names:
        result = archive["summary"]["combiners"][name]
        print(
            f"{name:14s} {result['mean']:8.2f} {result['std']:7.2f}   "
            f"{result['mean'] - baseline['mean']:+.2f}"
        )


if __name__ == "__main__":
    main()
