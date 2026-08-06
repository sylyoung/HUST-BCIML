# tune_networks.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Literal target-isolated nested LOSO selection for the Network comparison."""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import os
import statistics
import uuid
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import torch

from hustbciml.algorithms.network_methods import (
    NETWORK_METHODS,
    NetworkMethod,
    network_method_manifest,
    select_network_methods,
)
from hustbciml.algorithms.strategies._common import forward_logits
from hustbciml.core.batch import UNLABELED, EEGEpochs
from hustbciml.core.config import resolve_config
from hustbciml.core.context import RunContext
from hustbciml.core.pipeline import build_pipeline
from hustbciml.data_provider.splitters import list_targets
from hustbciml.exp.exp_basic import Exp_Basic, _json_safe
from hustbciml.exp.exp_cross_subject import Exp_CrossSubject
from hustbciml.exp.network_training import cpu_state_dict, train_network
from hustbciml.utils.io import (
    atomic_json_dump,
    atomic_savez,
    atomic_torch_save,
    file_sha256,
)
from hustbciml.utils.metrics import score
from hustbciml.utils.provenance import runtime_provenance
from hustbciml.utils.seed import fix_random_seed

PRODUCTION_DATASETS = ("BNCI2014001", "BNCI2014002", "BNCI2015001")
PRODUCTION_LRS = (0.0001, 0.0003, 0.001, 0.003)
PRODUCTION_SEEDS = (1, 2, 3, 4, 5)
PRODUCTION_EPOCHS = 300
PRODUCTION_BATCH_SIZE = 32
PRODUCTION_SELECTION_SEED = 1


def _tag(value: float) -> str:
    return ("%g" % value).replace("-", "m").replace(".", "p")


def _load_torch(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _masked(epochs: EEGEpochs) -> EEGEpochs:
    return dataclasses.replace(
        epochs,
        y=np.full(len(epochs), UNLABELED, dtype=np.int64),
    )


def _identity_base(cfg, runtime: dict) -> dict:
    return {
        "schema_version": 3,
        "dataset": cfg.dataset,
        "protocol": "literal_nested_cross_subject_loso",
        "stages": {
            "aligner": "EA",
            "augmenter": "Identity",
            "head": "Linear",
            "strategy": "ERM",
        },
        "data_sha256": cfg.data_provenance.get("content_sha256"),
        "source_sha256": runtime.get("source_sha256"),
        "hustbciml_version": runtime.get("hustbciml_version"),
        "git": runtime.get("git"),
        "environment_lock": runtime.get("environment_lock"),
        "python": runtime.get("python"),
        "platform": runtime.get("platform"),
        "machine": runtime.get("machine"),
        "dependencies": runtime.get("dependencies"),
        "numpy_build": runtime.get("numpy_build"),
        "numerical_libraries": runtime.get("numerical_libraries"),
        "nvidia_driver": runtime.get("nvidia_driver"),
        "torch_runtime": runtime.get("torch_runtime"),
        "device": cfg.resolved_device or cfg.device,
    }


def _load_exact_json(path: Path, identity: dict) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise FileExistsError(f"{path} exists but is not readable strict JSON") from exc
    if payload.get("identity") != identity:
        raise FileExistsError(
            f"{path} belongs to a different request; preserve it and use another "
            "campaign root"
        )
    return payload


@contextmanager
def _method_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "campaign.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another writer holds {path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _model_configuration(model) -> dict:
    backbone = model.backbone
    convolutions = []
    dropouts = []
    for name, module in backbone.named_modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
            convolutions.append({
                "name": name,
                "type": type(module).__name__,
                "kernel_size": list(module.kernel_size),
                "stride": list(module.stride),
                "groups": int(module.groups),
                "bias": module.bias is not None,
            })
        elif isinstance(module, torch.nn.Dropout):
            dropouts.append({"name": name, "p": float(module.p)})
    configuration = {
        "backbone_class": type(backbone).__name__,
        "head_class": type(model.head).__name__,
        "out_features": int(backbone.out_features),
        "trainable_parameters": int(sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )),
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "convolutions": convolutions,
        "dropouts": dropouts,
        "objective": "cross_entropy",
    }
    for attribute in ("bands", "filter_orders", "drop_prob"):
        if hasattr(backbone, attribute):
            configuration[attribute] = _json_safe(getattr(backbone, attribute))
    return configuration


def _selection_initialization_seed(selection_seed: int, outer: int, inner: int) -> int:
    return int(selection_seed) * 1_000_000 + int(outer) * 10_000 + int(inner) * 100


def _selection_record(
    base_cfg,
    epochs: EEGEpochs,
    runtime: dict,
    method: NetworkMethod,
    lr: float,
    epochs_ceiling: int,
    batch_size: int,
    selection_seed: int,
    outer_target: int,
    inner_validation: int,
    method_root: Path,
    mode: str,
) -> dict:
    initialization_seed = _selection_initialization_seed(
        selection_seed,
        outer_target,
        inner_validation,
    )
    cfg = dataclasses.replace(
        base_cfg,
        algorithm=method.public_key,
        aligner=method.aligner,
        augmenter=method.augmenter,
        backbone=method.backbone,
        head=method.head,
        strategy=method.strategy,
        lr=lr,
        epochs=epochs_ceiling,
        batch_size=batch_size,
        seed=initialization_seed,
        val_split="subject",
        fold_seed=False,
    )
    identity = {
        **_identity_base(cfg, runtime),
        "phase": "inner_selection",
        "mode": mode,
        "method": dataclasses.asdict(method),
        "learning_rate": lr,
        "epochs_ceiling": epochs_ceiling,
        "batch_size": batch_size,
        "selection_seed": selection_seed,
        "initialization_seed": initialization_seed,
        "outer_target": outer_target,
        "inner_validation": inner_validation,
    }
    directory = (
        method_root
        / "selection"
        / f"outer{outer_target}"
        / f"inner{inner_validation}"
    )
    path = directory / f"lr{_tag(lr)}.json"
    cached = _load_exact_json(path, identity)
    if cached is not None:
        return cached

    inner_pool = epochs.select(epochs.domain != outer_target)
    if outer_target in set(np.unique(inner_pool.domain).tolist()):
        raise AssertionError("outer target remained in inner-selection data")
    inner_domains = set(int(value) for value in np.unique(inner_pool.domain))
    if inner_validation not in inner_domains:
        raise ValueError(
            f"inner validation subject {inner_validation} is absent after excluding "
            f"outer target {outer_target}"
        )

    fix_random_seed(initialization_seed)
    pipe = build_pipeline(cfg)
    pipe.aligner.fit(inner_pool)
    aligned = pipe.aligner.transform(inner_pool)
    train_epochs = aligned.select(aligned.domain != inner_validation)
    validation_epochs = aligned.select(aligned.domain == inner_validation)
    if outer_target in set(np.unique(train_epochs.domain).tolist()):
        raise AssertionError("outer target entered inner training")

    ctx = RunContext(
        cfg=cfg,
        device=torch.device(cfg.resolved_device or cfg.device),
        augmenter=pipe.augmenter,
        aligner=pipe.aligner,
        log=(lambda message: print(message, flush=True)) if cfg.verbose else (lambda _: None),
        target_unlabeled=None,
    )
    training = train_network(
        pipe.model,
        train_epochs,
        ctx,
        epochs=epochs_ceiling,
        validation_epochs=validation_epochs,
        patience=cfg.early_stop_patience,
        resume_path=directory / f"lr{_tag(lr)}.resume.pt",
        resume_identity=identity,
    )
    payload = {
        "identity": identity,
        "is_measurement": mode == "production",
        "non_measurement_reason": None if mode == "production" else f"{mode} run",
        "val_primary": training.best_validation,
        "best_epoch": training.best_epoch,
        "stop_epoch": training.stop_epoch,
        "optimizer_steps": training.optimizer_steps,
        "resumed_from_epoch": training.resumed_from_epoch,
        "model_configuration": _model_configuration(training.model),
        "config": dataclasses.asdict(cfg),
        "provenance": {"runtime": runtime, "data": cfg.data_provenance},
    }
    atomic_json_dump(_json_safe(payload), path)
    return payload


def _selection_summary(
    method: NetworkMethod,
    outer_target: int,
    inner_subjects: list[int],
    lrs: list[float],
    records: dict[tuple[int, float], dict],
    identity_base: dict,
    method_root: Path,
    mode: str,
) -> dict:
    candidates = {}
    for lr in lrs:
        values = [records[(inner, lr)] for inner in inner_subjects]
        scores = [float(value["val_primary"]) for value in values]
        best_epochs = [int(value["best_epoch"]) for value in values]
        candidates["%g" % lr] = {
            "mean_validation": float(np.mean(scores)),
            "per_inner_subject": {
                str(inner): float(records[(inner, lr)]["val_primary"])
                for inner in inner_subjects
            },
            "best_epoch_by_inner_subject": {
                str(inner): int(records[(inner, lr)]["best_epoch"])
                for inner in inner_subjects
            },
            "median_low_best_epoch": int(statistics.median_low(best_epochs)),
        }

    selected_lr = min(
        lrs,
        key=lambda value: (
            -candidates["%g" % value]["mean_validation"],
            candidates["%g" % value]["median_low_best_epoch"],
            value,
        ),
    )
    selected = candidates["%g" % selected_lr]
    identity = {
        **identity_base,
        "phase": "outer_selection_summary",
        "mode": mode,
        "method": dataclasses.asdict(method),
        "outer_target": outer_target,
        "inner_subjects": inner_subjects,
        "learning_rates": lrs,
    }
    payload = {
        "identity": identity,
        "is_measurement": mode == "production",
        "non_measurement_reason": None if mode == "production" else f"{mode} run",
        "selected_lr": selected_lr,
        "selected_epochs": selected["median_low_best_epoch"],
        "candidates": candidates,
    }
    path = method_root / "selection" / f"outer{outer_target}" / "summary.json"
    existing = _load_exact_json(path, identity)
    if existing is not None and existing != payload:
        raise RuntimeError(f"recomputed selection summary differs from {path}")
    atomic_json_dump(_json_safe(payload), path)
    return payload


def _validate_final_triplet(path: Path, identity: dict) -> dict | None:
    payload = _load_exact_json(path, identity)
    if payload is None:
        return None
    prediction_path = path.with_name(payload.get("predictions_file", ""))
    checkpoint_path = path.with_name(payload.get("checkpoint_file", ""))
    if not prediction_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"{path} does not have both declared sibling artifacts")
    if file_sha256(prediction_path) != payload.get("predictions_sha256"):
        raise RuntimeError(f"{prediction_path} digest does not match {path}")
    if file_sha256(checkpoint_path) != payload.get("checkpoint_sha256"):
        raise RuntimeError(f"{checkpoint_path} digest does not match {path}")
    try:
        with np.load(prediction_path, allow_pickle=False) as archive:
            prediction_artifact_id = str(archive["artifact_id"].item())
            prediction_identity = json.loads(str(archive["identity_json"].item()))
    except Exception as exc:
        raise RuntimeError(f"{prediction_path} has no readable strict identity") from exc
    checkpoint = _load_torch(checkpoint_path)
    if (
        prediction_artifact_id != payload.get("artifact_id")
        or checkpoint.get("artifact_id") != payload.get("artifact_id")
    ):
        raise RuntimeError(f"{path} triplet comes from different writes")
    if prediction_identity != identity or checkpoint.get("identity") != identity:
        raise RuntimeError(f"{path} triplet has mismatched measurement identity")
    return payload


def _final_record(
    base_cfg,
    epochs: EEGEpochs,
    runtime: dict,
    method: NetworkMethod,
    lr: float,
    selected_epochs: int,
    batch_size: int,
    final_seed: int,
    outer_target: int,
    method_root: Path,
    mode: str,
) -> dict:
    initialization_seed = final_seed * 1000 + outer_target
    cfg = dataclasses.replace(
        base_cfg,
        algorithm=method.public_key,
        aligner=method.aligner,
        augmenter=method.augmenter,
        backbone=method.backbone,
        head=method.head,
        strategy=method.strategy,
        lr=lr,
        epochs=selected_epochs,
        batch_size=batch_size,
        seed=initialization_seed,
        val_split="subject",
        fold_seed=False,
    )
    identity = {
        **_identity_base(cfg, runtime),
        "phase": "outer_final",
        "mode": mode,
        "method": dataclasses.asdict(method),
        "selected_lr": lr,
        "selected_epochs": selected_epochs,
        "batch_size": batch_size,
        "final_seed": final_seed,
        "initialization_seed": initialization_seed,
        "outer_target": outer_target,
    }
    directory = method_root / "final" / f"outer{outer_target}" / f"seed{final_seed}"
    path = directory / "record.json"
    cached = _validate_final_triplet(path, identity)
    if cached is not None:
        return cached

    source = epochs.select(epochs.domain != outer_target)
    target = epochs.select(epochs.domain == outer_target)
    if not len(source) or not len(target):
        raise ValueError(f"outer target {outer_target} produced an empty source or target")

    fix_random_seed(initialization_seed)
    pipe = build_pipeline(cfg)
    pipe.aligner.fit(source)
    source_aligned = pipe.aligner.transform(source)
    target_aligned = dataclasses.replace(
        pipe.aligner.transform(_masked(target)),
        y=target.y,
    )
    ctx = RunContext(
        cfg=cfg,
        device=torch.device(cfg.resolved_device or cfg.device),
        augmenter=pipe.augmenter,
        aligner=pipe.aligner,
        log=(lambda message: print(message, flush=True)) if cfg.verbose else (lambda _: None),
        target_unlabeled=None,
    )
    training = train_network(
        pipe.model,
        source_aligned,
        ctx,
        epochs=selected_epochs,
        validation_epochs=None,
        patience=None,
        resume_path=directory / "training.resume.pt",
        resume_identity=identity,
    )

    artifact_id = uuid.uuid4().hex
    checkpoint_name = "checkpoint.pt"
    prediction_name = "predictions.npz"
    checkpoint_path = directory / checkpoint_name
    prediction_path = directory / prediction_name
    checkpoint = {
        "schema_version": 1,
        "artifact_id": artifact_id,
        "identity": identity,
        "model_state": cpu_state_dict(training.model),
        "model_configuration": _model_configuration(training.model),
        "selected_lr": lr,
        "selected_epochs": selected_epochs,
        "optimizer_steps": training.optimizer_steps,
    }
    atomic_torch_save(checkpoint, checkpoint_path)

    logits = forward_logits(training.model, target_aligned, ctx.device)
    y_score = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    y_pred = logits.argmax(1)
    metrics = score(
        target.y,
        y_pred,
        y_score,
        paradigm=epochs.paradigm,
        n_classes=epochs.n_classes,
    )

    fix_random_seed(initialization_seed)
    reload_pipe = build_pipeline(cfg)
    reload_backbone = getattr(reload_pipe.model, "backbone", None)
    if reload_backbone is not None and hasattr(reload_backbone, "init_from_source"):
        reload_backbone.init_from_source(source_aligned)
    reload_pipe.model.load_state_dict(checkpoint["model_state"])
    reload_pipe.model.to(ctx.device)
    reloaded_logits = forward_logits(reload_pipe.model, target_aligned, ctx.device)
    if not np.allclose(reloaded_logits, logits, rtol=1e-5, atol=1e-6):
        max_error = float(np.max(np.abs(reloaded_logits - logits)))
        raise RuntimeError(
            f"checkpoint reload changed outer-target logits (max_abs={max_error})"
        )
    reload_max_abs = float(np.max(np.abs(reloaded_logits - logits)))

    atomic_savez(
        prediction_path,
        artifact_id=np.asarray(artifact_id, dtype="U"),
        identity_json=np.asarray(json.dumps(identity, sort_keys=True), dtype="U"),
        subject=np.asarray(outer_target),
        trial_index=np.arange(len(target), dtype=np.int64),
        y_true=np.asarray(target.y),
        y_pred=np.asarray(y_pred),
        y_score=np.asarray(y_score),
        logits=np.asarray(logits),
    )
    payload = {
        "artifact_id": artifact_id,
        "identity": identity,
        "is_measurement": mode == "production",
        "non_measurement_reason": None if mode == "production" else f"{mode} run",
        "metrics": metrics,
        "config": dataclasses.asdict(cfg),
        "training": {
            "optimizer_steps": training.optimizer_steps,
            "stop_epoch": training.stop_epoch,
            "resumed_from_epoch": training.resumed_from_epoch,
        },
        "model_configuration": checkpoint["model_configuration"],
        "checkpoint_reload": {"passed": True, "max_abs_logit_error": reload_max_abs},
        "checkpoint_file": checkpoint_name,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "predictions_file": prediction_name,
        "predictions_sha256": file_sha256(prediction_path),
        "provenance": {"runtime": runtime, "data": cfg.data_provenance},
    }
    atomic_json_dump(_json_safe(payload), path)
    return payload


def _method_summary(
    base_cfg,
    runtime: dict,
    method: NetworkMethod,
    targets: list[int],
    seeds: list[int],
    lrs: list[float],
    selection: dict[int, dict],
    finals: dict[tuple[int, int], dict],
    method_root: Path,
    mode: str,
) -> dict:
    per_seed = {}
    seed_primary = []
    for seed in seeds:
        metrics = [finals[(target, seed)]["metrics"] for target in targets]
        aggregate = Exp_Basic.aggregate(metrics)
        per_seed[str(seed)] = {
            "summary": aggregate,
            "per_target": {
                str(target): finals[(target, seed)]["metrics"]
                for target in targets
            },
        }
        seed_primary.append(float(aggregate["primary"]["mean"]))

    identity = {
        **_identity_base(base_cfg, runtime),
        "phase": "method_summary",
        "mode": mode,
        "method": dataclasses.asdict(method),
        "targets": targets,
        "learning_rates": lrs,
        "final_seeds": seeds,
    }
    payload = {
        "identity": identity,
        "is_measurement": mode == "production",
        "non_measurement_reason": None if mode == "production" else f"{mode} run",
        "method": dataclasses.asdict(method),
        "requested_targets": targets,
        "completed_targets": targets,
        "requested_seeds": seeds,
        "completed_seeds": seeds,
        "selected_by_target": {
            str(target): {
                "lr": selection[target]["selected_lr"],
                "epochs": selection[target]["selected_epochs"],
            }
            for target in targets
        },
        "per_seed": per_seed,
        "primary_mean": float(np.mean(seed_primary)),
        "primary_std": float(np.std(seed_primary, ddof=1)) if len(seed_primary) > 1 else 0.0,
        "std_definition": "sample standard deviation across seed-level subject-macro means",
        "n_classes": 2,
        "chance_percent": 50.0,
        "provenance": {"runtime": runtime, "data": base_cfg.data_provenance},
    }
    path = method_root / "summary.json"
    existing = _load_exact_json(path, identity)
    if existing is not None and existing != payload:
        raise RuntimeError(f"recomputed method summary differs from {path}")
    atomic_json_dump(_json_safe(payload), path)
    return payload


def _parse_csv(raw: str, cast):
    return [cast(value) for value in raw.split(",") if value != ""]


def _validate_production_request(
    args,
    methods: tuple[NetworkMethod, ...],
    lrs: list[float],
    seeds: list[int],
    targets: list[int],
    all_targets: list[int],
    runtime: dict,
) -> None:
    if args.dataset not in PRODUCTION_DATASETS:
        raise ValueError(f"production dataset must be one of {PRODUCTION_DATASETS}")
    if not methods:
        raise ValueError("production requires at least one method shard")
    if tuple(lrs) != PRODUCTION_LRS:
        raise ValueError(f"production learning rates must be exactly {PRODUCTION_LRS}")
    if tuple(seeds) != PRODUCTION_SEEDS:
        raise ValueError(f"production seeds must be exactly {PRODUCTION_SEEDS}")
    if args.epochs != PRODUCTION_EPOCHS or args.batch_size != PRODUCTION_BATCH_SIZE:
        raise ValueError(
            f"production requires epochs={PRODUCTION_EPOCHS} and "
            f"batch_size={PRODUCTION_BATCH_SIZE}"
        )
    if args.selection_seed != PRODUCTION_SELECTION_SEED:
        raise ValueError(
            f"production selection_seed must be {PRODUCTION_SELECTION_SEED}"
        )
    if targets != all_targets:
        raise ValueError("production cannot restrict outer targets")
    git = runtime.get("git") or {}
    if git.get("commit") is None or git.get("dirty") is not False:
        raise RuntimeError("production requires a clean, identifiable Git commit")
    environment_lock = runtime.get("environment_lock") or {}
    if environment_lock.get("sha256") is None:
        raise RuntimeError(
            "production requires an environment lock at "
            "requirements-network-production.txt"
        )
    if environment_lock.get("matches_installed") is not True:
        raise RuntimeError(
            "production environment differs from requirements-network-production.txt: "
            f"{environment_lock.get('mismatches')}"
        )
    expected_runtime = environment_lock.get("expected_runtime") or {}
    torch_runtime = runtime.get("torch_runtime") or {}
    device_names = sorted({
        device.get("name")
        for device in torch_runtime.get("devices") or []
        if device.get("name")
    })
    actual_runtime = {
        "python": runtime.get("python"),
        "nvidia_driver": runtime.get("nvidia_driver"),
        "cuda_runtime": torch_runtime.get("cuda_runtime"),
        "cudnn": torch_runtime.get("cudnn"),
        "gpu": device_names[0] if len(device_names) == 1 else device_names,
    }
    if expected_runtime != actual_runtime:
        raise RuntimeError(
            "production numerical runtime differs from the environment lock: "
            f"expected {expected_runtime}, got {actual_runtime}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hustbciml.scripts.tune_networks")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument(
        "--methods",
        default="EEGNet",
        help="comma-separated Network display names, or 'all'",
    )
    parser.add_argument("--lrs", default=",".join("%g" % value for value in PRODUCTION_LRS))
    parser.add_argument("--epochs", type=int, default=PRODUCTION_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=PRODUCTION_BATCH_SIZE)
    parser.add_argument("--selection_seed", type=int, default=PRODUCTION_SELECTION_SEED)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--targets", default="")
    parser.add_argument("--mode", choices=("smoke", "pilot", "production"), default="smoke")
    args = parser.parse_args(argv)

    method_names = (
        [method.display_name for method in NETWORK_METHODS]
        if args.methods == "all"
        else _parse_csv(args.methods, str)
    )
    methods = select_network_methods(method_names)
    lrs = _parse_csv(args.lrs, float)
    seeds = _parse_csv(args.seeds, int)
    if not lrs or len(set(lrs)) != len(lrs):
        raise ValueError("--lrs must be non-empty and contain no duplicates")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must be non-empty and contain no duplicates")
    if args.epochs <= 0 or args.batch_size <= 1:
        raise ValueError("--epochs must be positive and --batch_size must exceed one")

    cli = [
        "--dataset", args.dataset,
        "--aligner", "EA",
        "--augmenter", "Identity",
        "--backbone", "EEGNet",
        "--head", "Linear",
        "--strategy", "ERM",
        "--device", args.device,
        "--data_dir", args.data_dir,
        "--results_dir", args.results_dir,
        "--val_split", "subject",
    ]
    base_cfg, _ = resolve_config(cli)
    loader = Exp_CrossSubject(base_cfg)
    epochs = loader._get_data()
    if base_cfg.data_provenance.get("is_measurement") is not True:
        raise RuntimeError(
            "Network tuning requires a provenance-complete measurement cache"
        )
    if epochs.n_classes != 2 or args.dataset.endswith("-4"):
        raise ValueError("the reportable Network benchmark is strictly two-class")

    runtime = runtime_provenance()
    all_targets = [int(value) for value in list_targets(epochs)]
    targets = _parse_csv(args.targets, int) if args.targets else list(all_targets)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("--targets must be non-empty and contain no duplicates")
    unknown_targets = sorted(set(targets) - set(all_targets))
    if unknown_targets:
        raise KeyError(f"unknown target subjects {unknown_targets}; available: {all_targets}")

    if args.mode == "production":
        _validate_production_request(
            args,
            methods,
            lrs,
            seeds,
            targets,
            all_targets,
            runtime,
        )

    root = Path(args.results_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    campaign_manifest_path = root / "network_method_manifest.json"
    manifest_payload = {
        "schema_version": 1,
        "methods": network_method_manifest(),
    }
    if campaign_manifest_path.exists():
        try:
            existing_manifest = json.loads(campaign_manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"{campaign_manifest_path} is unreadable") from exc
        if existing_manifest != manifest_payload:
            raise RuntimeError(
                f"{campaign_manifest_path} differs from the executable Network manifest"
            )
    else:
        atomic_json_dump(manifest_payload, campaign_manifest_path)

    for method in methods:
        method_root = root / args.dataset / method.public_key
        with _method_lock(method_root):
            method_cfg = dataclasses.replace(
                base_cfg,
                algorithm=method.public_key,
                aligner=method.aligner,
                augmenter=method.augmenter,
                backbone=method.backbone,
                head=method.head,
                strategy=method.strategy,
            )
            identity_base = _identity_base(method_cfg, runtime)
            selections = {}
            for outer_target in targets:
                inner_subjects = [
                    subject for subject in all_targets if subject != outer_target
                ]
                records = {}
                for inner_validation in inner_subjects:
                    for lr in lrs:
                        record = _selection_record(
                            method_cfg,
                            epochs,
                            runtime,
                            method,
                            lr,
                            args.epochs,
                            args.batch_size,
                            args.selection_seed,
                            outer_target,
                            inner_validation,
                            method_root,
                            args.mode,
                        )
                        records[(inner_validation, lr)] = record
                        print(
                            f"[inner] {args.dataset} {method.display_name} "
                            f"outer={outer_target} inner={inner_validation} lr={lr:g} "
                            f"val={record['val_primary']:.2f} epoch={record['best_epoch']}",
                            flush=True,
                        )
                selections[outer_target] = _selection_summary(
                    method,
                    outer_target,
                    inner_subjects,
                    lrs,
                    records,
                    identity_base,
                    method_root,
                    args.mode,
                )

            finals = {}
            for outer_target in targets:
                selected = selections[outer_target]
                for seed in seeds:
                    record = _final_record(
                        method_cfg,
                        epochs,
                        runtime,
                        method,
                        float(selected["selected_lr"]),
                        int(selected["selected_epochs"]),
                        args.batch_size,
                        seed,
                        outer_target,
                        method_root,
                        args.mode,
                    )
                    finals[(outer_target, seed)] = record
                    print(
                        f"[final] {args.dataset} {method.display_name} "
                        f"outer={outer_target} seed={seed} "
                        f"primary={record['metrics']['primary']:.2f}",
                        flush=True,
                    )

            summary = _method_summary(
                method_cfg,
                runtime,
                method,
                targets,
                seeds,
                lrs,
                selections,
                finals,
                method_root,
                args.mode,
            )
            print(
                f"[summary] {args.dataset} {method.display_name} "
                f"acc={summary['primary_mean']:.2f}+/-{summary['primary_std']:.2f} "
                f"measurement={summary['is_measurement']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
