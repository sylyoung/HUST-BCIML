"""Explicit validation/fixed-epoch training used by the Network benchmark."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from hustbciml.algorithms.strategies._common import forward_logits
from hustbciml.core.batch import EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.data_provider.collate import iterate_batches
from hustbciml.utils.io import atomic_torch_save
from hustbciml.utils.metrics import accuracy


@dataclass
class NetworkTrainingResult:
    model: nn.Module
    best_validation: float | None
    best_epoch: int | None
    stop_epoch: int
    optimizer_steps: int
    resumed_from_epoch: int


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _load_training_state(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument
        return torch.load(path, map_location="cpu")


def train_network(
    model: nn.Module,
    train_epochs: EEGEpochs,
    ctx: RunContext,
    *,
    epochs: int,
    validation_epochs: EEGEpochs | None,
    patience: int | None,
    resume_path: str | Path | None,
    resume_identity: dict,
    resume_interval: int = 10,
) -> NetworkTrainingResult:
    """Train with an explicit validation subject or a fixed epoch count.

    Unlike ``supervised_train``, this function never creates a hidden split.
    Nested selection passes one whole inner validation subject; final refitting
    passes ``validation_epochs=None`` and runs exactly ``epochs`` epochs.
    """
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if resume_interval < 1:
        raise ValueError(f"resume_interval must be positive, got {resume_interval}")
    if validation_epochs is None and patience is not None:
        raise ValueError("patience is meaningful only with validation data")
    if validation_epochs is not None and (patience is None or patience < 1):
        raise ValueError("validation training requires positive patience")
    if not len(train_epochs):
        raise ValueError("network training received no training trials")
    if validation_epochs is not None and not len(validation_epochs):
        raise ValueError("network training received an empty validation subject")

    device = ctx.device
    model.to(device)
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "init_from_source"):
        backbone.init_from_source(train_epochs)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("network model exposes no trainable parameters")
    optimizer = torch.optim.Adam(
        trainable,
        lr=ctx.cfg.lr,
        weight_decay=ctx.cfg.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    state_path = Path(resume_path) if resume_path is not None else None
    start_epoch = 0
    best_score = float("-inf")
    best_epoch = None
    best_state = None
    bad_epochs = 0
    optimizer_steps = 0

    if state_path is not None and state_path.exists():
        state = _load_training_state(state_path)
        if state.get("identity") != resume_identity:
            raise RuntimeError(
                f"{state_path} belongs to a different training request; preserve it "
                "and use another campaign root"
            )
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state["next_epoch"])
        best_score = float(state["best_score"])
        best_epoch = state["best_epoch"]
        best_state = state["best_state"]
        bad_epochs = int(state["bad_epochs"])
        optimizer_steps = int(state["optimizer_steps"])
        _restore_rng_state(state["rng_state"])

    resumed_from_epoch = start_epoch
    stop_epoch = start_epoch
    already_stopped = (
        validation_epochs is not None
        and patience is not None
        and bad_epochs >= patience
    )

    if not already_stopped:
        for epoch_index in range(start_epoch, epochs):
            model.train()
            for batch in iterate_batches(
                train_epochs,
                ctx.cfg.batch_size,
                shuffle=True,
                drop_last=True,
                seed=ctx.cfg.seed + epoch_index,
            ):
                if batch.x.size(0) <= 1:
                    continue
                batch = ctx.augmenter(batch).to(device)
                _, logits = model(batch.x)
                loss = criterion(logits, batch.y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_steps += 1

            stop_epoch = epoch_index + 1
            if validation_epochs is not None:
                logits = forward_logits(model, validation_epochs, device)
                score = float(accuracy(validation_epochs.y, logits.argmax(1)))
                if score > best_score:
                    best_score = score
                    best_epoch = stop_epoch
                    best_state = cpu_state_dict(model)
                    bad_epochs = 0
                    ctx.log(
                        f"  epoch {stop_epoch}/{epochs} inner_val_acc={score:.2f} *"
                    )
                else:
                    bad_epochs += 1

            should_stop = (
                validation_epochs is not None
                and patience is not None
                and bad_epochs >= patience
            )
            should_checkpoint = (
                stop_epoch % resume_interval == 0
                or stop_epoch == epochs
                or should_stop
            )
            if state_path is not None and should_checkpoint:
                atomic_torch_save(
                    {
                        "schema_version": 1,
                        "identity": resume_identity,
                        "next_epoch": stop_epoch,
                        "model_state": cpu_state_dict(model),
                        "optimizer_state": optimizer.state_dict(),
                        "best_score": best_score,
                        "best_epoch": best_epoch,
                        "best_state": best_state,
                        "bad_epochs": bad_epochs,
                        "optimizer_steps": optimizer_steps,
                        "rng_state": _rng_state(),
                    },
                    state_path,
                )

            if should_stop:
                ctx.log(
                    f"  early stop at epoch {stop_epoch} "
                    f"(best inner validation={best_score:.2f})"
                )
                break

    if optimizer_steps == 0:
        raise RuntimeError(
            f"network training performed 0 optimizer steps: epochs={epochs}, "
            f"batch_size={ctx.cfg.batch_size}, training_trials={len(train_epochs)}"
        )

    if validation_epochs is not None:
        if best_state is None or best_epoch is None or not np.isfinite(best_score):
            raise RuntimeError("validation training produced no finite best checkpoint")
        model.load_state_dict(best_state)
        best_validation = best_score
    else:
        best_validation = None
        best_epoch = None

    if state_path is not None and state_path.exists():
        state_path.unlink()

    return NetworkTrainingResult(
        model=model,
        best_validation=best_validation,
        best_epoch=None if best_epoch is None else int(best_epoch),
        stop_epoch=int(stop_epoch),
        optimizer_steps=int(optimizer_steps),
        resumed_from_epoch=int(resumed_from_epoch),
    )
