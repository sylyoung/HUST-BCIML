# tune_networks.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Nested per-target learning-rate selection for the Network comparison.

For each outer LOSO target, candidate learning rates are compared using only
held-out source subjects. The selection pass never aligns, predicts, scores, or
prints the outer target. A fresh model is then trained at that target's selected
rate for every requested final seed and evaluates the target exactly once.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import uuid
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np

from hustbciml.core.config import resolve_config
from hustbciml.exp.exp_basic import Exp_Basic, _json_safe
from hustbciml.exp.exp_cross_subject import Exp_CrossSubject
from hustbciml.data_provider.splitters import list_targets
from hustbciml.utils.io import atomic_json_dump, atomic_savez
from hustbciml.utils.provenance import runtime_provenance
from hustbciml.utils.seed import fix_random_seed

# Display name -> registry backbone name. MVCNet is intentionally absent because
# it changes the objective/batch setup rather than only the backbone stage.
BACKBONES = {
    "EEGNet": "EEGNet",
    "ShallowConvNet": "ShallowConvNet",
    "DeepConvNet": "DeepConvNet",
    "EEGConformer": "EEGConformer",
    "DBConformer": "DBConformer",
    "CSP-Net": "CSPNet",
    "TIE-EEGNet": "TIEEEGNet",
    "KDFNet": "KDFNet",
    "ADFCNN": "ADFCNN",
    "CTNet": "CTNet",
    "MSCFormer": "MSCFormer",
    "MSVTNet": "MSVTNet",
    "TMSA-Net": "TMSANet",
    "EEGWaveNet": "EEGWaveNet",
    "SlimSeiz": "SlimSeiz",
    "FBMSNet": "FBMSNet",
    "EEGNeX": "EEGNeX",
    "EEG-Deformer": "EEGDeformer",
}


def _tag(value: float) -> str:
    return ("%g" % value).replace("-", "m").replace(".", "p")


def _identity_base(cfg, runtime: dict) -> dict:
    return {
        "schema_version": 1,
        "dataset": cfg.dataset,
        "protocol": "cross_subject_nested",
        "stages": {
            "aligner": "EA", "augmenter": "Identity", "head": "Linear",
            "strategy": "ERM",
        },
        "data_sha256": cfg.data_provenance.get("content_sha256"),
        "source_sha256": runtime.get("source_sha256"),
        "hustbciml_version": runtime.get("hustbciml_version"),
        "python": runtime.get("python"),
        "platform": runtime.get("platform"),
        "machine": runtime.get("machine"),
        "dependencies": runtime.get("dependencies"),
        "numpy_build": runtime.get("numpy_build"),
        "numerical_libraries": runtime.get("numerical_libraries"),
        "torch_runtime": runtime.get("torch_runtime"),
        "device": cfg.resolved_device or cfg.device,
    }


def _load_exact(path: Path, identity: dict) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise FileExistsError(f"{path} exists but is not readable strict JSON") from exc
    if payload.get("identity") != identity:
        raise FileExistsError(
            f"{path} belongs to a different tuning request; preserve it and use a "
            "different results directory"
        )
    if payload.get("is_measurement") is not True:
        raise RuntimeError(f"{path} is explicitly non-measurement and cannot be reused")
    return payload


def _selection_record(base_cfg, epochs, runtime, backbone: str, lr: float,
                      epochs_ceiling: int, batch_size: int, selection_seed: int,
                      target: int, root: Path) -> dict:
    initialization_seed = selection_seed * 1000 + target
    cfg = dataclasses.replace(
        base_cfg, backbone=backbone, lr=lr, epochs=epochs_ceiling,
        batch_size=batch_size, seed=selection_seed, val_split="subject",
        fold_seed=True,
    )
    identity = {
        **_identity_base(cfg, runtime),
        "phase": "selection", "backbone": backbone, "lr": lr,
        "epochs": epochs_ceiling, "batch_size": batch_size,
        "selection_seed": selection_seed, "initialization_seed": initialization_seed,
        "target": target, "val_split": "subject",
    }
    path = root / "grid" / cfg.dataset / backbone / f"target{target}" / (
        f"lr{_tag(lr)}_ep{epochs_ceiling}_s{selection_seed}_valsubject.json"
    )
    cached = _load_exact(path, identity)
    if cached is not None:
        return cached

    fix_random_seed(initialization_seed)
    result = Exp_CrossSubject(cfg).run_fold(epochs, target, selection_only=True)
    payload = {
        "identity": identity,
        "is_measurement": bool(cfg.data_provenance.get("is_measurement", False)),
        "val_primary": result.val_primary,
        "config": dataclasses.asdict(cfg),
        "provenance": {"runtime": runtime, "data": cfg.data_provenance},
    }
    if payload["is_measurement"] is not True:
        raise RuntimeError("nested tuning refuses non-measurement dataset provenance")
    atomic_json_dump(_json_safe(payload), path)
    return payload


def _final_record(base_cfg, epochs, runtime, backbone: str, lr: float,
                  epochs_ceiling: int, batch_size: int, seed: int,
                  target: int, root: Path) -> dict:
    initialization_seed = seed * 1000 + target
    cfg = dataclasses.replace(
        base_cfg, backbone=backbone, lr=lr, epochs=epochs_ceiling,
        batch_size=batch_size, seed=seed, val_split="subject", fold_seed=True,
    )
    identity = {
        **_identity_base(cfg, runtime),
        "phase": "final", "backbone": backbone, "lr": lr,
        "epochs": epochs_ceiling, "batch_size": batch_size, "seed": seed,
        "initialization_seed": initialization_seed, "target": target,
        "val_split": "subject",
    }
    stem = f"lr{_tag(lr)}_ep{epochs_ceiling}_seed{seed}"
    directory = root / "final" / cfg.dataset / backbone / f"target{target}"
    path = directory / f"{stem}.json"
    prediction_path = directory / f"{stem}_predictions.npz"
    cached = _load_exact(path, identity)
    if cached is not None:
        if not prediction_path.exists():
            raise FileNotFoundError(f"{path} exists but {prediction_path} is missing")
        try:
            with np.load(prediction_path, allow_pickle=False) as prediction_archive:
                prediction_artifact_id = str(prediction_archive["artifact_id"].item())
        except Exception as exc:
            raise RuntimeError(
                f"{prediction_path} has no readable artifact identity"
            ) from exc
        if (
            not cached.get("artifact_id")
            or prediction_artifact_id != cached["artifact_id"]
        ):
            raise RuntimeError(f"{path} and {prediction_path} come from different writes")
        return cached

    fix_random_seed(initialization_seed)
    result = Exp_CrossSubject(cfg).run_fold(epochs, target, selection_only=False)
    payload = {
        "artifact_id": uuid.uuid4().hex,
        "identity": identity,
        "is_measurement": bool(cfg.data_provenance.get("is_measurement", False)),
        "metrics": result.metrics,
        "val_primary": result.val_primary,
        "config": dataclasses.asdict(cfg),
        "provenance": {"runtime": runtime, "data": cfg.data_provenance},
        "predictions_file": prediction_path.name,
    }
    if payload["is_measurement"] is not True:
        raise RuntimeError("nested tuning refuses non-measurement dataset provenance")
    prediction = result.prediction
    atomic_savez(
        prediction_path,
        artifact_id=np.asarray(payload["artifact_id"], dtype="U"),
        subject=np.asarray(prediction["subject"]),
        y_true=np.asarray(prediction["y_true"]),
        y_pred=np.asarray(prediction["y_pred"]),
        y_score=np.asarray(prediction["y_score"]),
    )
    atomic_json_dump(_json_safe(payload), path)
    return payload


def _aggregate(backbone: str, targets: list[int], seeds: list[int],
               selected_lr: dict[int, float], validation: dict,
               final_records: dict, runtime: dict, base_cfg) -> dict:
    per_seed = {}
    primary = []
    kappas = []
    for seed in seeds:
        metrics = [final_records[(target, seed)]["metrics"] for target in targets]
        summary = Exp_Basic.aggregate(metrics)
        per_seed[str(seed)] = {
            "summary": summary,
            "per_target": {
                str(target): final_records[(target, seed)]["metrics"] for target in targets
            },
        }
        primary.append(summary["primary"]["mean"])
        kappas.append(summary["kappa"]["mean"])

    return {
        "is_measurement": True,
        "identity": {
            **_identity_base(base_cfg, runtime),
            "phase": "nested_summary", "backbone": backbone,
            "targets": targets, "seeds": seeds, "val_split": "subject",
        },
        "selected_lr_by_target": {str(key): value for key, value in selected_lr.items()},
        "validation_by_target": validation,
        "requested_targets": targets,
        "completed_targets": targets,
        "requested_seeds": seeds,
        "completed_seeds": seeds,
        "per_seed": per_seed,
        "primary_mean": float(np.mean(primary)),
        "primary_std": float(np.std(primary)),
        "kappa_mean": float(np.mean(kappas)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hustbciml.scripts.tune_networks")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--backbones", default=",".join(BACKBONES))
    parser.add_argument("--lrs", default="0.0001,0.0003,0.001,0.003")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sel_seed", type=int, default=1)
    parser.add_argument("--seeds", default="1,2,3")
    args = parser.parse_args(argv)

    lrs = [float(value) for value in args.lrs.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    display_names = [value for value in args.backbones.split(",") if value]
    unknown = sorted(set(display_names) - set(BACKBONES))
    if unknown:
        raise KeyError(f"unknown backbone display names {unknown}; available: {sorted(BACKBONES)}")
    if not display_names or len(set(display_names)) != len(display_names):
        raise ValueError("--backbones must be non-empty and contain no duplicates")
    if not lrs or not seeds:
        raise ValueError("--lrs and --seeds must both be non-empty")
    if len(set(lrs)) != len(lrs):
        raise ValueError("--lrs contains duplicates")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds contains duplicates")
    if args.epochs <= 0 or args.batch_size <= 1:
        raise ValueError("--epochs must be positive and --batch_size must exceed one")

    cli = [
        "--dataset", args.dataset, "--aligner", "EA", "--augmenter", "Identity",
        "--backbone", "EEGNet", "--head", "Linear", "--strategy", "ERM",
        "--device", args.device, "--data_dir", args.data_dir,
        "--results_dir", args.results_dir, "--val_split", "subject",
    ]
    base_cfg, _ = resolve_config(cli)
    loader = Exp_CrossSubject(base_cfg)
    epochs = loader._get_data()
    if base_cfg.data_provenance.get("is_measurement") is not True:
        raise RuntimeError(
            "nested tuning cannot publish from a legacy/non-measurement cache; regenerate it "
            "with explicit preprocessing provenance"
        )
    runtime = runtime_provenance()
    targets = [int(value) for value in list_targets(epochs)]
    root = Path(args.results_dir).resolve()
    output_path = root / f"nested_tuned_{args.dataset}.json"
    request_identity = {
        **_identity_base(base_cfg, runtime),
        "phase": "nested_request",
        "backbones": {name: BACKBONES[name] for name in display_names},
        "learning_rates": lrs,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "selection_seed": args.sel_seed,
        "final_seeds": seeds,
        "targets": targets,
        "val_split": "subject",
    }
    if output_path.exists():
        try:
            with output_path.open(encoding="utf-8") as handle:
                output = json.load(handle)
        except Exception as exc:
            raise FileExistsError(
                f"{output_path} exists but is not readable strict JSON"
            ) from exc
        if output.get("identity") != request_identity:
            raise FileExistsError(
                f"{output_path} belongs to a different or legacy tuning request; "
                "preserve it and use another results directory"
            )
        if not isinstance(output.get("backbones"), dict):
            raise RuntimeError(f"{output_path} has no valid backbone result mapping")
    else:
        output = {
            "schema_version": 2,
            "identity": request_identity,
            "dataset": args.dataset,
            "is_measurement": False,
            "non_measurement_reason": "requested backbone set is incomplete",
            "provenance": {"runtime": runtime, "data": base_cfg.data_provenance},
            "requested_backbones": display_names,
            "completed_backbones": [],
            "backbones": {},
        }

    for display_name in display_names:
        if display_name in output.get("completed_backbones", []):
            if display_name not in output["backbones"]:
                raise RuntimeError(
                    f"{output_path} marks {display_name!r} complete but stores no summary"
                )
            continue
        backbone = BACKBONES[display_name]
        selected_lr = {}
        validation = {}
        for target in targets:
            scores = {}
            for lr in lrs:
                record = _selection_record(
                    base_cfg, epochs, runtime, backbone, lr, args.epochs,
                    args.batch_size, args.sel_seed, target, root,
                )
                scores["%g" % lr] = record["val_primary"]
                print(
                    f"[select] {args.dataset} {display_name} target={target} "
                    f"lr={lr:g} val={record['val_primary']:.4f}", flush=True,
                )
            best_lr = max(lrs, key=lambda value: scores["%g" % value])
            selected_lr[target] = best_lr
            validation[str(target)] = scores

        final_records = {}
        for target in targets:
            for seed in seeds:
                record = _final_record(
                    base_cfg, epochs, runtime, backbone, selected_lr[target],
                    args.epochs, args.batch_size, seed, target, root,
                )
                final_records[(target, seed)] = record
                print(
                    f"[final ] {args.dataset} {display_name} target={target} seed={seed} "
                    f"lr={selected_lr[target]:g} primary={record['metrics']['primary']:.2f}",
                    flush=True,
                )

        output["backbones"][display_name] = _aggregate(
            backbone, targets, seeds, selected_lr, validation,
            final_records, runtime, base_cfg,
        )
        completed = [
            name for name in display_names if name in output["backbones"]
        ]
        output["completed_backbones"] = completed
        output["is_measurement"] = completed == display_names
        output["non_measurement_reason"] = (
            None if output["is_measurement"] else "requested backbone set is incomplete"
        )
        atomic_json_dump(_json_safe(output), output_path)
        result = output["backbones"][display_name]
        print(
            f"[summary] {args.dataset} {display_name} "
            f"acc={result['primary_mean']:.2f}+/-{result['primary_std']:.2f} "
            f"kappa={result['kappa_mean']:.3f}", flush=True,
        )

    if output.get("completed_backbones") != display_names or output.get("is_measurement") is not True:
        raise RuntimeError("not every requested backbone completed; refusing reportable summary")
    print(json.dumps(_json_safe(output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
