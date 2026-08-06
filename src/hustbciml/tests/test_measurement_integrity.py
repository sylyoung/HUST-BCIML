"""Focused guards for provenance, nested selection, and ensemble artifacts.

These tests use synthetic arrays and mocked training records. They exercise the
measurement firewall without downloading EEG data or launching long training runs.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from hustbciml.algorithms.ensembles import build_combiners, combiner_manifest
from hustbciml.algorithms.network_methods import NETWORK_METHOD_BY_NAME
from hustbciml.algorithms.strategies._common import split_train_val
from hustbciml.core.batch import EEGEpochs
from hustbciml.core.config import Config
from hustbciml.core.context import RunContext
from hustbciml.core.stages import VoteCombiner
from hustbciml.data_provider.datasets import MOABBAdapter, _MOABB_SPEC, _epochs_digest
from hustbciml.exp.exp_cross_subject import Exp_CrossSubject
from hustbciml.exp import network_training
from hustbciml.scripts import combined_ensemble, decentralized, ensemble, tune_networks
from hustbciml.utils.io import (
    atomic_json_dump,
    atomic_torch_save,
    atomic_write_text,
    file_sha256,
)
from hustbciml.utils import provenance as provenance_module
from hustbciml.utils.provenance import arrays_digest, runtime_provenance, source_tree_digest


def _epochs(domains=(0, 1, 2), trials_per_domain=4, n_chans=3, n_times=16):
    domain = np.repeat(np.asarray(domains), trials_per_domain)
    n_trials = len(domain)
    epochs = EEGEpochs(
        X=np.arange(n_trials * n_chans * n_times, dtype=np.float32).reshape(
            n_trials, n_chans, n_times
        ),
        y=np.arange(n_trials, dtype=np.int64) % 2,
        domain=domain,
        sfreq=128.0,
        n_classes=2,
        ch_names=[f"C{index}" for index in range(n_chans)],
        classes=["left_hand", "right_hand"],
    )
    epochs.provenance = {
        "schema_version": 1,
        "is_measurement": True,
        "dataset": "Toy",
        "content_sha256": _epochs_digest(epochs),
    }
    return epochs


def _runtime():
    return {
        "schema_version": 1,
        "hustbciml_version": "test",
        "source_sha256": "source-digest",
        "dependencies": {"numpy": np.__version__},
        "torch_runtime": {"cuda_available": False},
    }


def test_array_digest_covers_values_dtype_shape_and_metadata():
    arrays = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    digest = arrays_digest(arrays, {"band": [8, 32]})
    assert digest == arrays_digest({"x": arrays["x"].copy()}, {"band": [8, 32]})
    assert digest != arrays_digest({"x": arrays["x"].astype(np.float64)}, {"band": [8, 32]})
    changed = arrays["x"].copy()
    changed[0, 0] = 99
    assert digest != arrays_digest({"x": changed}, {"band": [8, 32]})
    assert digest != arrays_digest(arrays, {"band": [4, 40]})


def test_source_digest_excludes_tests_and_docs_but_includes_presets(tmp_path):
    root = tmp_path / "hustbciml"
    (root / "core").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "tests").mkdir()
    (root / "algorithms" / "presets").mkdir(parents=True)
    (root / "core" / "run.py").write_text("VALUE = 1\n")
    (root / "docs" / "card.yaml").write_text("text: one\n")
    (root / "tests" / "test_run.py").write_text("assert True\n")
    preset = root / "algorithms" / "presets" / "EA.yaml"
    preset.write_text("lr: 0.001\n")

    initial = source_tree_digest(root)
    (root / "docs" / "card.yaml").write_text("text: two\n")
    (root / "tests" / "test_run.py").write_text("assert False\n")
    assert source_tree_digest(root) == initial
    preset.write_text("lr: 0.003\n")
    assert source_tree_digest(root) != initial


def test_git_provenance_uses_working_directory_for_legacy_git(monkeypatch, tmp_path):
    root = tmp_path / "repository"
    (root / ".git").mkdir(parents=True)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        stdout = "clean-commit\n" if command[1] == "rev-parse" else ""
        return types.SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(provenance_module.shutil, "which", lambda name: "/usr/bin/git")
    monkeypatch.setattr(provenance_module.subprocess, "run", fake_run)

    assert provenance_module._git_info(root) == {
        "root": str(root),
        "commit": "clean-commit",
        "dirty": False,
    }
    assert [command for command, _ in calls] == [
        ["/usr/bin/git", "rev-parse", "HEAD"],
        ["/usr/bin/git", "status", "--porcelain"],
    ]
    assert all(kwargs["cwd"] == root for _, kwargs in calls)


def test_runtime_provenance_records_effective_numerical_backend():
    provenance = runtime_provenance()
    assert len(provenance["source_sha256"]) == 64
    assert provenance["python"]
    assert provenance["machine"]["machine"]
    assert provenance["dependencies"]["numpy"] == np.__version__
    numpy_build = provenance["numpy_build"]
    assert numpy_build
    assert (
        "blas" in numpy_build or "blas_opt_info" in numpy_build
    ), numpy_build
    assert isinstance(provenance["numerical_libraries"], list)
    assert provenance["nvidia_driver"] is None or isinstance(
        provenance["nvidia_driver"], (str, list)
    )
    assert provenance["torch_runtime"]["torch_version"] == torch.__version__
    assert isinstance(provenance["torch_runtime"]["devices"], list)
    lock_path = Path(__file__).resolve().parents[3] / "requirements-network-production.txt"
    lock = provenance["environment_lock"]
    assert lock["path"] == "requirements-network-production.txt"
    assert lock["sha256"] == file_sha256(lock_path)
    assert lock["size_bytes"] == lock_path.stat().st_size
    assert lock["expected_runtime"] == {
        "python": "3.11.13",
        "nvidia_driver": "525.60.13",
        "cuda_runtime": "12.1",
        "cudnn": 90100,
        "gpu": "NVIDIA GeForce RTX 3090",
    }
    assert lock["expected_packages"] == sum(
        bool(line.strip()) and not line.lstrip().startswith(("#", "-"))
        for line in lock_path.read_text().splitlines()
    )
    assert isinstance(lock["matches_installed"], bool)
    assert isinstance(lock["mismatches"], dict)


def test_atomic_json_is_strict_and_leaves_no_temporary_file(tmp_path):
    destination = tmp_path / "manifest.json"
    atomic_json_dump({"complete": True}, destination)
    assert json.loads(destination.read_text()) == {"complete": True}
    assert not list(tmp_path.glob(".*.tmp.*"))
    with pytest.raises(ValueError):
        atomic_json_dump({"bad": float("nan")}, destination)
    assert json.loads(destination.read_text()) == {"complete": True}
    assert not list(tmp_path.glob(".*.tmp.*"))

    text_path = tmp_path / "report.txt"
    atomic_write_text("complete\n", text_path)
    assert text_path.read_text() == "complete\n"
    checkpoint_path = tmp_path / "checkpoint.pt"
    atomic_torch_save({"weight": torch.arange(3)}, checkpoint_path)
    assert torch.equal(
        torch.load(checkpoint_path)["weight"],
        torch.arange(3),
    )
    assert len(file_sha256(checkpoint_path)) == 64
    assert not list(tmp_path.glob(".*.tmp.*"))


def _write_cache(path: Path, epochs: EEGEpochs, provenance=None):
    arrays = {
        "X": epochs.X,
        "y": epochs.y,
        "domain": epochs.domain,
        "sfreq": np.asarray(epochs.sfreq),
        "n_classes": np.asarray(epochs.n_classes),
        "ch_names": np.asarray(epochs.ch_names, dtype="U"),
        "classes": np.asarray(epochs.classes, dtype="U"),
    }
    if provenance is not None:
        arrays["provenance_json"] = np.asarray(
            json.dumps(provenance, sort_keys=True), dtype="U"
        )
    np.savez(path, **arrays)


def _bnci_2014_002_epochs():
    epochs = EEGEpochs(
        X=np.zeros((4, 15, 16), dtype=np.float32),
        y=np.array([0, 1, 0, 1]),
        domain=np.array([0, 0, 1, 1]),
        sfreq=512.0,
        n_classes=2,
        ch_names=[f"EEG{index}" for index in range(1, 16)],
        classes=["feet", "right_hand"],
    )
    return epochs


@pytest.fixture
def small_bnci_2014_002_spec(monkeypatch):
    specification = dict(_MOABB_SPEC["BNCI2014002"])
    specification.update(n_subjects=2, per_subject=2, n_times=16)
    monkeypatch.setitem(_MOABB_SPEC, "BNCI2014002", specification)


def test_legacy_cache_fails_by_default_and_is_marked_exploratory(
    tmp_path, small_bnci_2014_002_spec
):
    cache = tmp_path / "BNCI2014002_epochs.npz"
    _write_cache(cache, _bnci_2014_002_epochs())
    with pytest.raises(ValueError, match="no preprocessing provenance"):
        MOABBAdapter("BNCI2014002", data_dir=str(tmp_path)).load()

    loaded = MOABBAdapter(
        "BNCI2014002", data_dir=str(tmp_path), allow_legacy_cache=True
    ).load()
    assert loaded.provenance["is_measurement"] is False
    assert loaded.provenance["status"] == "legacy_unknown"
    assert loaded.provenance["content_sha256"] == _epochs_digest(loaded)


def test_annotated_cache_rejects_content_or_preprocessing_mismatch(
    tmp_path, small_bnci_2014_002_spec
):
    cache = tmp_path / "BNCI2014002_epochs.npz"
    epochs = _bnci_2014_002_epochs()
    provenance = {
        "schema_version": 2,
        "is_measurement": True,
        "dataset": "BNCI2014002",
        "content_sha256": "not-the-array-digest",
        "preprocessing": {
            "paradigm": "MotorImagery",
            "n_classes_requested": 2,
            "fmin": 8.0,
            "fmax": 32.0,
            "tmin": 0.0,
            "tmax": None,
        },
    }
    _write_cache(cache, epochs, provenance)
    with pytest.raises(ValueError, match="content digest"):
        MOABBAdapter("BNCI2014002", data_dir=str(tmp_path)).load()

    provenance["content_sha256"] = _epochs_digest(epochs)
    provenance["preprocessing"]["fmin"] = 4.0
    _write_cache(cache, epochs, provenance)
    with pytest.raises(ValueError, match="preprocessing"):
        MOABBAdapter("BNCI2014002", data_dir=str(tmp_path)).load()

    provenance["preprocessing"]["fmin"] = 8.0
    provenance["loader"] = "MOABBAdapter"
    provenance["dataset_class"] = "moabb.datasets.BNCI2014_002"
    provenance["selection"] = {
        "session_first": False,
        "run_contains": "train",
        "two_class": None,
    }
    provenance["selection_resolved"] = {
        "session_order": ["0"],
        "selected_sessions": ["0"],
        "run_order": ["0train", "1train"],
        "selected_runs": ["0train", "1train"],
        "subject_trial_counts": [2, 2],
        "class_trial_counts": [2, 2],
    }
    _write_cache(cache, epochs, provenance)
    with pytest.raises(ValueError, match="software versions"):
        MOABBAdapter("BNCI2014002", data_dir=str(tmp_path)).load()


def test_fresh_moabb_call_uses_explicit_preprocessing(
    monkeypatch, tmp_path, small_bnci_2014_002_spec
):
    captured = {}

    class FakeDataset:
        subject_list = [1, 2]

    class FakeMotorImagery:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def get_data(self, dataset, subjects):
            metadata = pd.DataFrame({
                "subject": [1, 1, 2, 2],
                "session": ["0", "0", "0", "0"],
                "run": ["0train", "1train", "0train", "1train"],
            })
            return (
                np.zeros((4, 15, 16), dtype=np.float32),
                np.array(["feet", "right_hand", "feet", "right_hand"]),
                metadata,
            )

    moabb_module = types.ModuleType("moabb")
    datasets_module = types.ModuleType("moabb.datasets")
    paradigms_module = types.ModuleType("moabb.paradigms")
    datasets_module.BNCI2014_002 = FakeDataset
    paradigms_module.MotorImagery = FakeMotorImagery
    moabb_module.datasets = datasets_module
    moabb_module.paradigms = paradigms_module
    moabb_module.set_log_level = lambda _: None
    monkeypatch.setitem(__import__("sys").modules, "moabb", moabb_module)
    monkeypatch.setitem(__import__("sys").modules, "moabb.datasets", datasets_module)
    monkeypatch.setitem(__import__("sys").modules, "moabb.paradigms", paradigms_module)

    loaded = MOABBAdapter("BNCI2014002", data_dir=str(tmp_path)).load()
    assert captured == {
        "n_classes": 2,
        "fmin": 8.0,
        "fmax": 32.0,
        "tmin": 0.0,
        "tmax": None,
    }
    assert loaded.provenance["preprocessing"] == {
        "paradigm": "MotorImagery",
        "n_classes_requested": 2,
        "fmin": 8.0,
        "fmax": 32.0,
        "tmin": 0.0,
        "tmax": None,
    }
    assert loaded.provenance["schema_version"] == 2
    assert loaded.provenance["selection_resolved"] == {
        "session_order": ["0"],
        "selected_sessions": ["0"],
        "run_order": ["0train", "1train"],
        "selected_runs": ["0train", "1train"],
        "subject_trial_counts": [2, 2],
        "class_trial_counts": [2, 2],
    }
    with np.load(tmp_path / "BNCI2014002_epochs.npz", allow_pickle=False) as archive:
        stored = json.loads(str(archive["provenance_json"].item()))
    assert stored["content_sha256"] == _epochs_digest(loaded)
    manifest_path = tmp_path / "BNCI2014002_epochs.npz.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 2
    assert manifest["file_sha256"] == file_sha256(tmp_path / "BNCI2014002_epochs.npz")
    assert manifest["content_sha256"] == stored["content_sha256"]


def test_base_metrics_and_predictions_share_one_atomic_write_identity(tmp_path):
    config = Config(
        dataset="Toy", algorithm="EA-EEGNet", results_dir=str(tmp_path), device="cpu"
    )
    config.data_provenance = {
        "is_measurement": True,
        "dataset": "Toy",
        "content_sha256": "data-digest",
    }
    experiment = Exp_CrossSubject(config)
    prediction = {
        "subject": 0,
        "y_true": np.array([0, 1]),
        "y_pred": np.array([0, 1]),
        "y_score": np.eye(2),
    }
    directory = Path(experiment.save_results(
        [{"primary": 100.0}],
        {"primary": {"mean": 100.0, "std": 0.0}},
        predictions=[prediction],
    ))
    metrics = json.loads((directory / "metrics.json").read_text())
    with np.load(directory / "predictions.npz", allow_pickle=True) as archive:
        assert str(archive["artifact_id"].item()) == metrics["artifact_id"]

    with np.load(directory / "predictions.npz", allow_pickle=True) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["artifact_id"] = np.asarray("different-write", dtype="U")
    np.savez(directory / "predictions.npz", **arrays)
    with pytest.raises(RuntimeError, match="different writes"):
        ensemble._validate_base_artifact(directory, "EA-EEGNet", "Toy", config.seed)


def test_subject_validation_never_falls_back_to_trials():
    domain = np.repeat([0, 1, 2, 3], 5)
    train, validation = split_train_val(
        len(domain), 0.25, seed=7, domain=domain, mode="subject"
    )
    assert set(domain[train]).isdisjoint(set(domain[validation]))
    with pytest.raises(ValueError, match="cannot split"):
        split_train_val(5, 0.2, seed=7, domain=np.zeros(5), mode="subject")


def test_selection_fold_never_constructs_or_scores_outer_target(monkeypatch):
    epochs = _epochs(domains=(0, 1, 2))
    target = 2
    observed = {}

    class FakeAligner:
        def fit(self, source):
            observed["fit_domains"] = set(source.domain.tolist())
            return self

        def transform(self, source):
            observed["transform_domains"] = set(source.domain.tolist())
            return source

    class FakeStrategy:
        mode = "gradient"
        uses_target = False

        def fit(self, model, source, context):
            assert context.target_unlabeled is None
            observed["strategy_domains"] = set(source.domain.tolist())
            model._val_score = 73.5
            return model

    fake_pipeline = types.SimpleNamespace(
        model=torch.nn.Linear(1, 1),
        aligner=FakeAligner(),
        augmenter=object(),
        strategy=FakeStrategy(),
    )
    monkeypatch.setattr(
        "hustbciml.exp.exp_cross_subject.build_pipeline", lambda cfg: fake_pipeline
    )
    monkeypatch.setattr(
        "hustbciml.exp.exp_cross_subject.cross_subject",
        lambda *_: (_ for _ in ()).throw(AssertionError("outer target was constructed")),
    )
    cfg = Config(
        dataset="Toy", device="cpu", strategy="ERM", n_chans=3,
        n_times=16, n_classes=2, sfreq=128.0, val_split="subject",
    )
    result = Exp_CrossSubject(cfg).run_fold(epochs, target, selection_only=True)
    assert result.val_primary == 73.5
    assert result.metrics is None and result.prediction is None
    assert observed == {
        "fit_domains": {0, 1},
        "transform_domains": {0, 1},
        "strategy_domains": {0, 1},
    }


def test_literal_nested_tuner_visits_every_inner_subject_and_stays_nonmeasurement(
    monkeypatch, tmp_path
):
    epochs = _epochs(domains=(0, 1, 2))

    def fake_get_data(self):
        self.cfg.n_chans = epochs.n_channels
        self.cfg.n_times = epochs.n_times
        self.cfg.n_classes = epochs.n_classes
        self.cfg.sfreq = epochs.sfreq
        self.cfg.ch_names = list(epochs.ch_names)
        self.cfg.classes = list(epochs.classes)
        self.cfg.data_provenance = dict(epochs.provenance)
        return epochs

    monkeypatch.setattr(Exp_CrossSubject, "_get_data", fake_get_data)
    monkeypatch.setattr(tune_networks, "runtime_provenance", _runtime)
    selection_calls = []

    def fake_selection(
        base_cfg, ep, runtime, method, lr, epochs_ceiling, batch_size,
        selection_seed, outer_target, inner_validation, method_root, mode,
    ):
        selection_calls.append((outer_target, inner_validation, lr))
        return {
            "val_primary": 80.0 if lr == 0.001 else 70.0,
            "best_epoch": 1,
        }

    def fake_final(
        base_cfg, ep, runtime, method, lr, selected_epochs, batch_size,
        final_seed, outer_target, method_root, mode,
    ):
        value = 60.0 + final_seed + outer_target
        return {
            "metrics": {"accuracy": value, "kappa": value / 100, "primary": value}
        }

    monkeypatch.setattr(tune_networks, "_selection_record", fake_selection)
    monkeypatch.setattr(tune_networks, "_final_record", fake_final)
    tune_networks.main([
        "--dataset", "Toy", "--device", "cpu",
        "--results_dir", str(tmp_path),
        "--methods", "EEGNet", "--targets", "0",
        "--lrs", "0.001,0.003", "--epochs", "1",
        "--batch_size", "2", "--seeds", "1", "--mode", "smoke",
    ])

    assert selection_calls == [
        (0, 1, 0.001), (0, 1, 0.003),
        (0, 2, 0.001), (0, 2, 0.003),
    ]
    archive = json.loads((tmp_path / "Toy" / "EA-EEGNet-Nested" / "summary.json").read_text())
    assert archive["is_measurement"] is False
    assert archive["requested_targets"] == [0]
    assert archive["requested_seeds"] == [1]
    assert archive["selected_by_target"] == {"0": {"lr": 0.001, "epochs": 1}}


def test_network_production_requires_exact_five_seed_contract():
    arguments = types.SimpleNamespace(
        dataset="BNCI2014001",
        epochs=tune_networks.PRODUCTION_EPOCHS,
        batch_size=tune_networks.PRODUCTION_BATCH_SIZE,
        selection_seed=tune_networks.PRODUCTION_SELECTION_SEED,
    )
    method = (NETWORK_METHOD_BY_NAME["EEGNet"],)
    runtime_specification = {
        "python": "3.11.13",
        "nvidia_driver": "525.60.13",
        "cuda_runtime": "12.1",
        "cudnn": 90100,
        "gpu": "NVIDIA GeForce RTX 3090",
    }
    runtime = {
        "git": {"commit": "clean-commit", "dirty": False},
        "environment_lock": {
            "sha256": "lock-digest", "expected_runtime": runtime_specification,
            "matches_installed": True, "mismatches": {},
        },
        "python": runtime_specification["python"],
        "nvidia_driver": runtime_specification["nvidia_driver"],
        "torch_runtime": {
            "cuda_runtime": runtime_specification["cuda_runtime"],
            "cudnn": runtime_specification["cudnn"],
            "devices": [{"name": runtime_specification["gpu"]}],
        },
    }
    with pytest.raises(ValueError, match="seeds must be exactly"):
        tune_networks._validate_production_request(
            arguments,
            method,
            list(tune_networks.PRODUCTION_LRS),
            [1],
            [0, 1],
            [0, 1],
            runtime,
        )
    tune_networks._validate_production_request(
        arguments,
        method,
        list(tune_networks.PRODUCTION_LRS),
        list(tune_networks.PRODUCTION_SEEDS),
        [0, 1],
        [0, 1],
        runtime,
    )
    with pytest.raises(RuntimeError, match="environment"):
        tune_networks._validate_production_request(
            arguments,
            method,
            list(tune_networks.PRODUCTION_LRS),
            list(tune_networks.PRODUCTION_SEEDS),
            [0, 1],
            [0, 1],
            {"git": {"commit": "clean-commit", "dirty": False}},
        )
    with pytest.raises(RuntimeError, match="differs"):
        tune_networks._validate_production_request(
            arguments,
            method,
            list(tune_networks.PRODUCTION_LRS),
            list(tune_networks.PRODUCTION_SEEDS),
            [0, 1],
            [0, 1],
            {
                "git": {"commit": "clean-commit", "dirty": False},
                "environment_lock": {
                    "sha256": "lock-digest",
                    "matches_installed": False,
                    "mismatches": {"numpy": {"expected": "1", "actual": "2"}},
                },
            },
        )


def test_nested_final_cache_requires_matching_triplet_write(tmp_path):
    identity = {"phase": "outer_final", "outer_target": 0, "final_seed": 1}
    directory = tmp_path / "final" / "outer0" / "seed1"
    directory.mkdir(parents=True)
    record_path = directory / "record.json"
    prediction_path = directory / "predictions.npz"
    checkpoint_path = directory / "checkpoint.pt"
    np.savez(
        prediction_path,
        artifact_id=np.asarray("prediction-write", dtype="U"),
        identity_json=np.asarray(json.dumps(identity, sort_keys=True), dtype="U"),
    )
    atomic_torch_save(
        {
            "artifact_id": "checkpoint-write",
            "identity": identity,
            "model_state": {},
        },
        checkpoint_path,
    )
    atomic_json_dump(
        {
            "identity": identity,
            "is_measurement": True,
            "artifact_id": "record-write",
            "predictions_file": prediction_path.name,
            "predictions_sha256": file_sha256(prediction_path),
            "checkpoint_file": checkpoint_path.name,
            "checkpoint_sha256": file_sha256(checkpoint_path),
        },
        record_path,
    )
    with pytest.raises(RuntimeError, match="different writes"):
        tune_networks._validate_final_triplet(record_path, identity)


def test_network_training_resumes_exact_interrupted_epoch(monkeypatch, tmp_path):
    epochs = _epochs(domains=(0,), trials_per_domain=8, n_chans=1, n_times=4)
    config = Config(seed=7, lr=0.01, batch_size=2, weight_decay=0.0)
    context = RunContext(
        cfg=config,
        device=torch.device("cpu"),
        augmenter=lambda batch: batch,
        aligner=None,
        log=lambda _: None,
    )
    identity = {"phase": "resume-test", "seed": 7}
    resume_path = tmp_path / "training.resume.pt"

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Linear(4, 2)

        def forward(self, x):
            features = x.flatten(1)
            return features, self.classifier(features)

    real_save = network_training.atomic_torch_save
    saves = {"count": 0}

    def interrupt_after_save(payload, path):
        real_save(payload, path)
        saves["count"] += 1
        if saves["count"] == 1:
            raise RuntimeError("simulated process interruption")

    torch.manual_seed(11)
    monkeypatch.setattr(network_training, "atomic_torch_save", interrupt_after_save)
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        network_training.train_network(
            TinyModel(),
            epochs,
            context,
            epochs=3,
            validation_epochs=None,
            patience=None,
            resume_path=resume_path,
            resume_identity=identity,
            resume_interval=1,
        )
    assert resume_path.exists()

    monkeypatch.setattr(network_training, "atomic_torch_save", real_save)
    torch.manual_seed(11)
    result = network_training.train_network(
        TinyModel(),
        epochs,
        context,
        epochs=3,
        validation_epochs=None,
        patience=None,
        resume_path=resume_path,
        resume_identity=identity,
    )
    assert result.resumed_from_epoch == 1
    assert result.stop_epoch == 3
    assert result.optimizer_steps == 12
    assert not resume_path.exists()


def test_vote_combiner_receives_declared_unobserved_class_count():
    class Capture(VoteCombiner):
        name = "capture"

        def __init__(self):
            self.received = None

        def aggregate(self, votes, n_classes):
            self.received = n_classes
            return np.zeros(votes.shape[1], dtype=int)

    combiner = Capture()
    scores = np.zeros((3, 5, 3), dtype=float)
    scores[:, :, 0] = 1.0  # classes 1 and 2 receive no vote
    assert combiner.combine(scores).shape == (5,)
    assert combiner.received == 3


def test_combiner_manifest_records_effective_parameters_and_backend():
    combiners = build_combiners(
        ["GLAD", "M-MSR", "MACE", "ZenCrowd", "PM"],
        settings={"ZenCrowd": {"n_iter": 7}, "PM": {"n_iter": 4}},
    )
    manifest = combiner_manifest(combiners)
    assert manifest["GLAD"]["parameters"] == {
        "n_iter": 100,
        "tol": 1e-5,
        "m_step_max_iter": 25,
        "m_step_tol": 0.01,
    }
    assert manifest["M-MSR"]["parameters"]["random_state"] == 0
    assert manifest["MACE"]["parameters"]["method"] == "vb"
    assert manifest["ZenCrowd"]["parameters"] == {"n_iter": 7}
    assert manifest["PM"]["parameters"] == {"n_iter": 4}
    assert manifest["GLAD"]["backend"] == "crowdkit.aggregation.GLAD"
    assert manifest["GLAD"]["backend_version"] is not None
    with pytest.raises(ValueError, match="at least one"):
        build_combiners(["ZenCrowd"], settings={"ZenCrowd": {"n_iter": 0}})


class _FailingCombiner:
    binary_only = False

    def __call__(self, scores):
        raise RuntimeError("combiner exploded")


def test_decentralized_combiner_failure_aborts(monkeypatch):
    epochs = _epochs(domains=(0, 1))

    def fake_base(cfg, aligned, subjects, class_count):
        ytrue = {target: np.array([0, 1]) for target in subjects}
        scores = {
            target: {
                f"worker-{1 - target}": np.eye(2)[np.array([0, 1])]
            }
            for target in subjects
        }
        return ytrue, scores

    monkeypatch.setattr(decentralized, "_base_tangent_lda", fake_base)
    with pytest.raises(RuntimeError, match="combiner exploded"):
        decentralized._seed_run(
            Config(seed=1), torch.device("cpu"), epochs, 2,
            {"broken": _FailingCombiner()}, "tangent_lda",
        )


def _fake_load_aligned(cfg):
    epochs = _epochs(domains=(0, 1))
    cfg.n_chans = epochs.n_channels
    cfg.n_times = epochs.n_times
    cfg.n_classes = epochs.n_classes
    cfg.sfreq = epochs.sfreq
    cfg.ch_names = list(epochs.ch_names)
    cfg.classes = list(epochs.classes)
    cfg.data_provenance = dict(epochs.provenance)
    return epochs, 2


def _fake_decentralized_seed(cfg, device, epochs, class_count, combiners, base):
    subjects = [0, 1]
    truth = {target: np.array([0, 1]) for target in subjects}
    names = list(combiners)
    predictions = {
        name: {target: np.array([0, 1]) for target in subjects} for name in names
    }
    return {
        "single_source": 75.0,
        "combiners": {name: 100.0 for name in names},
        "per_target": {
            name: {target: 100.0 for target in subjects} for name in names
        },
        "subjects": subjects,
        "worker_ids": {target: ["worker"] for target in subjects},
        "y_true": truth,
        "hard_votes": {target: np.array([[0, 1]]) for target in subjects},
        "predictions": predictions,
    }


def test_decentralized_archive_requires_complete_seed_set(monkeypatch, tmp_path):
    monkeypatch.setattr(decentralized, "_load_aligned", _fake_load_aligned)
    monkeypatch.setattr(decentralized, "runtime_provenance", _runtime)
    calls = {"count": 0}

    def interrupted(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("simulated seed failure")
        return _fake_decentralized_seed(*args, **kwargs)

    monkeypatch.setattr(decentralized, "_seed_run", interrupted)
    with pytest.raises(RuntimeError, match="simulated seed failure"):
        decentralized.main([
            "--dataset", "Toy", "--device", "cpu", "--base", "tangent_lda",
            "--seeds", "1,2", "--combiners", "voting",
            "--results_dir", str(tmp_path),
        ])
    manifest_path = tmp_path / "decentralized_Toy_tangent_lda_EA-EEGNet.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["is_measurement"] is False
    assert manifest["completed_seeds"] == [1]
    assert manifest["non_measurement_reason"] == "requested seed set is incomplete"


def test_combined_archive_records_complete_seed_and_prediction_identity(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(combined_ensemble, "_load_aligned", _fake_load_aligned)
    monkeypatch.setattr(combined_ensemble, "runtime_provenance", _runtime)

    def fake_seed(cfg, device, epochs, class_count, combiners, backbones):
        result = _fake_decentralized_seed(
            cfg, device, epochs, class_count, combiners, "unused"
        )
        result["single_model"] = result.pop("single_source")
        result.pop("worker_ids")
        return result

    monkeypatch.setattr(combined_ensemble, "_seed_run", fake_seed)
    combined_ensemble.main([
        "--dataset", "Toy", "--device", "cpu", "--backbones", "EEGNet",
        "--seeds", "1,2", "--combiners", "voting",
        "--results_dir", str(tmp_path),
    ])
    manifest_path = tmp_path / "combined_Toy_hetero_EA-EEGNet.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["is_measurement"] is True
    assert manifest["completed_seeds"] == [1, 2]
    assert manifest["identity"]["combiners"]["voting"]["parameters"] == {}
    prediction_path = tmp_path / manifest["seed_results"]["1"]["predictions_file"]
    with np.load(prediction_path, allow_pickle=False) as archive:
        prediction_identity = json.loads(str(archive["identity_json"].item()))
        prediction_artifact_id = str(archive["artifact_id"].item())
    assert prediction_identity["seed"] == 1
    assert prediction_identity["archive_identity"] == manifest["identity"]
    assert prediction_artifact_id == manifest["seed_results"]["1"]["artifact_id"]


def _base_measurement(seed):
    return {
        "setting": f"Toy_cross_subject_EA-EEGNet_seed{seed}",
        "dataset": "Toy",
        "algorithm": "EA-EEGNet",
        "is_measurement": True,
        "config": {
            "seed": seed,
            "lr": 0.001,
            "epochs": 1,
            "resolved_device": "cpu",
        },
        "provenance": {
            "runtime": _runtime(),
            "data": {
                "is_measurement": True,
                "content_sha256": "data-digest",
            },
        },
    }


def test_multiseed_ensemble_refuses_mismatched_existing_identity(monkeypatch, tmp_path):
    subjects = np.array([0, 1])
    truths = [np.array([0, 1]), np.array([1, 0])]

    def fake_ensure(algorithm, dataset, seed, device, results_dir, data_dir):
        return f"seed-{seed}"

    def fake_load(directory):
        seed = int(directory.split("-")[-1])
        scores = [
            np.eye(2)[truths[0]],
            np.eye(2)[truths[1]],
        ]
        return subjects, truths, scores, _base_measurement(seed)

    monkeypatch.setattr(ensemble, "_ensure_run", fake_ensure)
    monkeypatch.setattr(ensemble, "_load", fake_load)
    monkeypatch.setattr(ensemble, "runtime_provenance", _runtime)
    common = [
        "--algorithm", "EA-EEGNet", "--dataset", "Toy", "--device", "cpu",
        "--seeds", "1,2", "--results_dir", str(tmp_path),
        "--combiners", "voting,PM",
    ]
    ensemble.main(common + ["--pm_iters", "3"])
    manifest_path = tmp_path / "multiseed_Toy_EA-EEGNet_seeds1-2.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["is_measurement"] is True
    assert manifest["identity"]["combiners"]["PM"]["parameters"] == {"n_iter": 3}
    prediction_path = tmp_path / manifest["predictions_file"]
    with np.load(prediction_path, allow_pickle=False) as prediction_archive:
        assert str(prediction_archive["artifact_id"].item()) == manifest["artifact_id"]
    with pytest.raises(FileExistsError, match="different or legacy"):
        ensemble.main(common + ["--pm_iters", "4"])
