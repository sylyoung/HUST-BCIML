# ensemble.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Black-box test-time ensemble over K random seeds of one base algorithm.

Runs ``--algorithm`` for each ``--seed`` (reusing the normal Exp, so the ensemble
combines exactly the predictions the benchmark produces), then for every target
subject stacks the K seeds' per-trial predictions and fuses them with each post-hoc
black-box combiner — hard majority ``voting`` (the baseline), the crowd-label
aggregators (Dawid-Skene / Wawa / M-MSR / MACE / GLAD / ZenCrowd / PM / LA / LAA /
EBCC), and the lab's SML / SML-OVR / StackingNet (see ``algorithms/ensembles/``). Every
combiner sees only hard votes — there is deliberately no soft-score averaging
combiner, so none has an information advantage over the label-only aggregators.
Reports per-combiner accuracy mean ± std across subjects, against the single-seed
base for reference.

    python -m hustbciml.scripts.ensemble --algorithm T-TIME --dataset BNCI2014001 \
        --seeds 1,2,3,4,5 --device cuda

Needs >= a few seeds to be meaningful (the lab uses 5-11). Each seed is a full
run, so this is a server job on real data; on Toy it runs locally in seconds.
"""
from __future__ import annotations

import argparse
import json
import os
import uuid
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
from sklearn.metrics import accuracy_score

from hustbciml import run as run_module
from hustbciml.algorithms.ensembles import build_combiners, combiner_manifest
from hustbciml.utils.io import atomic_json_dump, atomic_savez
from hustbciml.utils.provenance import runtime_provenance


def _setting_dir(results_dir, dataset, algorithm, seed, protocol="cross_subject"):
    return os.path.join(results_dir, f"{dataset}_{protocol}_{algorithm}_seed{seed}")


def _validate_base_artifact(directory, algorithm, dataset, seed):
    prediction_path = os.path.join(directory, "predictions.npz")
    metadata_path = os.path.join(directory, "metrics.json")
    if not os.path.exists(prediction_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"{directory} must contain both predictions.npz and metrics.json"
        )
    try:
        with open(metadata_path, encoding="utf-8") as handle:
            measurement = json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"{metadata_path} is not readable strict JSON") from exc
    try:
        with np.load(prediction_path, allow_pickle=True) as predictions:
            prediction_artifact_id = str(predictions["artifact_id"].item())
    except Exception as exc:
        raise RuntimeError(
            f"{prediction_path} has no readable artifact identity"
        ) from exc
    if not measurement.get("artifact_id") or prediction_artifact_id != measurement["artifact_id"]:
        raise RuntimeError(
            f"{directory} contains metrics and predictions from different writes"
        )
    got = (
        measurement.get("dataset"), measurement.get("algorithm"),
        (measurement.get("config") or {}).get("seed"),
    )
    want = (dataset, algorithm, seed)
    if got != want:
        raise RuntimeError(
            f"cached predictions in {directory} describe {got}, not {want}; preserve the "
            "directory and write the requested run elsewhere"
        )
    provenance = measurement.get("provenance") or {}
    data = provenance.get("data") or {}
    runtime = provenance.get("runtime") or {}
    if (
        measurement.get("is_measurement") is not True
        or data.get("is_measurement") is not True
        or not data.get("content_sha256")
        or not runtime.get("source_sha256")
        or not isinstance(runtime.get("dependencies"), dict)
    ):
        raise RuntimeError(
            f"{directory} is a legacy/non-measurement base artifact; it cannot feed a "
            "reportable ensemble"
        )
    return measurement


def _ensure_run(algorithm, dataset, seed, device, results_dir, data_dir):
    d = _setting_dir(results_dir, dataset, algorithm, seed)
    if os.path.exists(os.path.join(d, "predictions.npz")):
        _validate_base_artifact(d, algorithm, dataset, seed)
        print(f"[skip] seed {seed} already has predictions ({d})")
        return d
    print(f"[run ] {algorithm} on {dataset}, seed {seed}")
    run_module.main(["--algorithm", algorithm, "--dataset", dataset, "--seed", str(seed),
                     "--itr", "1", "--device", device,
                     "--results_dir", results_dir, "--data_dir", data_dir])
    _validate_base_artifact(d, algorithm, dataset, seed)
    return d


def _load(d):
    with np.load(os.path.join(d, "predictions.npz"), allow_pickle=True) as archive:
        predictions = (archive["subjects"], archive["y_true"], archive["y_score"])
    with open(os.path.join(d, "metrics.json"), encoding="utf-8") as handle:
        metrics = json.load(handle)
    return (*predictions, metrics)


def _base_config_identity(measurement):
    config = dict(measurement.get("config") or {})
    for key in (
        "seed", "data_dir", "results_dir", "itr", "run_tag", "overwrite", "verbose",
    ):
        config.pop(key, None)
    return config


def _check_alignment(loaded, seeds):
    """Every seed must describe the same subjects, in the same order, with the
    same ground truth.

    The aggregation stacks ``loaded[si][2][j]`` positionally and scores the result
    against ``loaded[0]``'s labels. If one cached seed has a different subject
    order or a different trial order, that silently combines predictions for
    different held-out subjects and scores them against the wrong labels — and the
    output is a plausible accuracy, not an error.
    """
    ref_subjects, ref_true = loaded[0][0], loaded[0][1]
    ref_identity = loaded[0][3].get("provenance")
    ref_config = _base_config_identity(loaded[0][3])
    for si in range(1, len(loaded)):
        subs, yt, _, metrics = loaded[si]
        current_identity = metrics.get("provenance") or {}
        ref_data = (ref_identity or {}).get("data") or {}
        cur_data = current_identity.get("data") or {}
        ref_runtime = (ref_identity or {}).get("runtime") or {}
        cur_runtime = current_identity.get("runtime") or {}
        if _base_config_identity(metrics) != ref_config:
            raise RuntimeError(
                f"seed {seeds[si]} has a different resolved base configuration from "
                f"seed {seeds[0]}"
            )
        runtime_fields = (
            "source_sha256", "python", "platform", "machine", "dependencies",
            "numpy_build", "numerical_libraries", "torch_runtime",
        )
        if (
            cur_data.get("content_sha256") != ref_data.get("content_sha256")
            or any(cur_runtime.get(key) != ref_runtime.get(key) for key in runtime_fields)
        ):
            raise RuntimeError(
                f"seed {seeds[si]} has different data/source/runtime provenance "
                f"from seed {seeds[0]}"
            )
        if not np.array_equal(subs, ref_subjects):
            raise RuntimeError(
                f"seed {seeds[si]} covers subjects {list(subs)} but seed {seeds[0]} "
                f"covers {list(ref_subjects)}; the runs are not aligned.")
        for j in range(len(ref_subjects)):
            if not np.array_equal(yt[j], ref_true[j]):
                raise RuntimeError(
                    f"seed {seeds[si]} has different ground-truth labels for subject "
                    f"{ref_subjects[j]} than seed {seeds[0]}; the trial order differs.")


def main(argv=None):
    p = argparse.ArgumentParser(prog="hustbciml.scripts.ensemble",
                                description="black-box multi-seed ensemble")
    p.add_argument("--algorithm", required=True, help="base preset to ensemble (e.g. T-TIME)")
    p.add_argument("--dataset", default="Toy")
    p.add_argument("--seeds", default="1,2,3,4,5", help="comma-separated seeds")
    p.add_argument("--device", default="auto")
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--combiners",
                   default="voting,Dawid-Skene,Wawa,M-MSR,MACE,GLAD,ZenCrowd,PM,"
                           "LA,LAA,EBCC,SML,SML-OVR,StackingNet")
    p.add_argument("--zencrowd_iters", type=int, default=20)
    p.add_argument("--pm_iters", type=int, default=3)
    p.add_argument("--allow_failed_combiners", action="store_true",
                   help="continue when a combiner raises, recording it as failed "
                        "instead of aborting (off by default: a crashed aggregator "
                        "is not a measured low score)")
    p.add_argument("--overwrite", action="store_true",
                   help="deliberately replace an existing ensemble artifact")
    a = p.parse_args(argv)

    np.random.seed(0)                                   # voting tie-breaks
    seeds = [int(s) for s in a.seeds.split(",")]
    names = [name for name in a.combiners.split(",") if name]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must be non-empty and contain no duplicates")
    if not names:
        raise ValueError("--combiners must request at least one combiner")
    combiners = build_combiners(names, settings={
        "ZenCrowd": {"n_iter": a.zencrowd_iters},
        "PM": {"n_iter": a.pm_iters},
    })
    K = len(seeds)

    dirs = [_ensure_run(a.algorithm, a.dataset, s, a.device, a.results_dir, a.data_dir) for s in seeds]
    loaded = [_load(d) for d in dirs]
    _check_alignment(loaded, seeds)
    subjects = loaded[0][0]

    # single-seed base accuracy (mean over seeds), for reference
    base = [np.mean([accuracy_score(yt[j], ys[j].argmax(1)) for j in range(len(subs))])
            for subs, yt, ys, _ in loaded]
    print(f"\n=== {a.algorithm} on {a.dataset} — {K} seeds {seeds} ===")
    print(f"base single-seed acc: {np.mean(base) * 100:.2f} +/- {np.std(base) * 100:.2f}")

    results = {name: [] for name in names}
    predictions = {name: {} for name in names}
    failed = {}
    for j, subject in enumerate(subjects):
        yt = loaded[0][1][j]
        C = loaded[0][2][j].shape[1]
        scores = np.stack([loaded[seed_index][2][j] for seed_index in range(K)])
        for name, combiner in combiners.items():
            if name in failed:
                continue
            if combiner.binary_only and C != 2:
                raise ValueError(
                    f"requested combiner {name!r} is binary-only but subject {subject} "
                    f"has {C} classes"
                )
            try:
                prediction = combiner(scores)
                predictions[name][int(subject)] = prediction
                results[name].append(accuracy_score(yt, prediction))
            except Exception as exc:
                failed[name] = f"{type(exc).__name__}: {exc}"
                results[name] = []
                if not a.allow_failed_combiners:
                    raise RuntimeError(
                        f"combiner {name!r} failed on subject {subject}: {failed[name]}"
                    ) from exc
                print(f"[warn] combiner {name!r} failed — artifact is non-measurement")

    complete = not failed and all(len(results[name]) == len(subjects) for name in names)
    if not complete and not a.allow_failed_combiners:
        raise RuntimeError("not every requested combiner completed every subject")
    seed_tag = "-".join(str(seed) for seed in seeds)
    artifact_stem = f"multiseed_{a.dataset}_{a.algorithm}_seeds{seed_tag}"
    prediction_path = os.path.join(a.results_dir, f"{artifact_stem}_predictions.npz")
    manifest_path = os.path.join(a.results_dir, f"{artifact_stem}.json")
    runtime = runtime_provenance()
    base_provenance = loaded[0][3]["provenance"]
    base_runtime = base_provenance["runtime"]
    identity = {
        "schema_version": 3,
        "dataset": a.dataset,
        "algorithm": a.algorithm,
        "requested_seeds": seeds,
        "combiners": combiner_manifest(combiners),
        "allow_failed_combiners": bool(a.allow_failed_combiners),
        "base_measurements": [measurement[3]["setting"] for measurement in loaded],
        "base_config": _base_config_identity(loaded[0][3]),
        "subjects": [int(subject) for subject in subjects],
        "data_sha256": base_provenance["data"].get("content_sha256"),
        "base_runtime": {
            key: base_runtime.get(key) for key in (
                "source_sha256", "python", "platform", "machine", "dependencies",
                "numerical_libraries", "torch_runtime",
            )
        },
        "ensemble_runtime": {
            key: runtime.get(key) for key in (
                "source_sha256", "python", "platform", "machine", "dependencies",
                "numerical_libraries", "torch_runtime",
            )
        },
    }
    if not a.overwrite:
        if os.path.exists(prediction_path) and not os.path.exists(manifest_path):
            raise FileExistsError(
                f"{prediction_path} exists without its manifest; preserve the partial artifact "
                "or pass --overwrite for deliberate replacement"
            )
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, encoding="utf-8") as handle:
                    existing = json.load(handle)
            except Exception as exc:
                raise FileExistsError(
                    f"{manifest_path} exists but is not readable strict JSON"
                ) from exc
            if existing.get("identity") != identity:
                raise FileExistsError(
                    f"{manifest_path} belongs to a different or legacy measurement identity; "
                    "preserve it and use another results directory"
                )
            if not os.path.exists(prediction_path):
                raise FileExistsError(
                    f"{manifest_path} exists but {prediction_path} is missing"
                )
            try:
                with np.load(prediction_path, allow_pickle=False) as prediction_archive:
                    existing_prediction_id = str(prediction_archive["artifact_id"].item())
            except Exception as exc:
                raise RuntimeError(
                    f"{prediction_path} has no readable artifact identity"
                ) from exc
            if (
                not existing.get("artifact_id")
                or existing_prediction_id != existing["artifact_id"]
            ):
                raise RuntimeError(
                    f"{manifest_path} and {prediction_path} come from different writes"
                )

    artifact_id = uuid.uuid4().hex
    arrays = {
        "artifact_id": np.asarray(artifact_id, dtype="U"),
        "identity_json": np.asarray(json.dumps(identity, sort_keys=True), dtype="U"),
    }
    for j, subject in enumerate(subjects):
        arrays[f"subject_{subject}_y_true"] = np.asarray(loaded[0][1][j])
        arrays[f"subject_{subject}_hard_votes"] = np.stack(
            [loaded[seed_index][2][j].argmax(1) for seed_index in range(K)]
        )
        for name in names:
            if int(subject) in predictions[name]:
                safe = "".join(character if character.isalnum() else "_" for character in name)
                arrays[f"subject_{subject}_{safe}_prediction"] = predictions[name][int(subject)]
    atomic_savez(prediction_path, **arrays)

    manifest = {
        "artifact_id": artifact_id,
        "identity": identity,
        "is_measurement": complete,
        "non_measurement_reason": None if complete else f"failed combiners: {failed}",
        "provenance": {
            "runtime": runtime,
            "base_runs": [measurement[3]["provenance"] for measurement in loaded],
        },
        "requested_seeds": seeds,
        "completed_seeds": seeds,
        "requested_combiners": names,
        "failed_combiners": failed,
        "subjects": [int(subject) for subject in subjects],
        "predictions_file": os.path.basename(prediction_path),
        "base": {
            "per_seed": {str(seed): float(value * 100) for seed, value in zip(seeds, base)},
            "mean": float(np.mean(base) * 100), "std": float(np.std(base) * 100),
        },
        "combiners": {
            name: {
                "per_subject": {
                    str(subject): float(value * 100)
                    for subject, value in zip(subjects, results[name])
                },
                "mean": None if not results[name] else float(np.mean(results[name]) * 100),
                "std": None if not results[name] else float(np.std(results[name]) * 100),
            }
            for name in names
        },
    }
    atomic_json_dump(manifest, manifest_path)

    print(f"{'combiner':14s} {'acc':>8s} {'std':>7s}   delta-vs-base")
    for name in names:
        if name in failed:
            print(f"{name:14s}   (failed: {failed[name][:48]})")
            continue
        arr = np.array(results[name]) * 100
        print(
            f"{name:14s} {arr.mean():8.2f} {arr.std():7.2f}   "
            f"{arr.mean() - np.mean(base) * 100:+.2f}"
        )


if __name__ == "__main__":
    main()
