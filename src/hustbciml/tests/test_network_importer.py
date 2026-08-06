"""Fail-closed publication tests for the corrected Network campaign importer."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from hustbciml.algorithms.network_methods import NETWORK_METHODS
from hustbciml.scripts import import_network_campaign as importer
from hustbciml.scripts.import_network_campaign import (
    EXPECTED_TARGETS,
    _apply_publication_metadata,
    _apply_values,
)
from hustbciml.utils.io import atomic_json_dump


def _certificate():
    values = {}
    for dataset_index, dataset in enumerate(EXPECTED_TARGETS):
        values[dataset] = {}
        for method_index, method in enumerate(NETWORK_METHODS):
            mean = 60.0 + dataset_index + method_index / 10
            values[dataset][method.public_key] = {
                "mean": mean,
                "std": 0.25 + method_index / 100,
                "per_seed": [mean - 0.2, mean - 0.1, mean, mean + 0.1, mean + 0.2],
            }
    return {
        "schema_version": 1,
        "source_sha256": "a" * 64,
        "numerical_families": {
            dataset: {"machine": {"node": "measurement-host"}}
            for dataset in EXPECTED_TARGETS
        },
        "values": values,
    }


def _pending_benchmark(path: Path) -> None:
    rows = []
    for method in NETWORK_METHODS:
        rows.append({
            "name": method.display_name,
            "key": method.public_key,
            "na_reason": "measurement pending",
            "code": f"src/hustbciml/algorithms/models/{method.backbone}.py",
            "acc": {
                dataset: {"mean": None, "std": None}
                for dataset in EXPECTED_TARGETS
            },
        })
    path.write_text(yaml.safe_dump({"tables": [{"id": "network", "rows": rows}]}, sort_keys=False))


def test_validated_import_updates_all_publication_sources(tmp_path):
    certificate = _certificate()
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()
    atomic_json_dump(certificate, campaign_root / "validated_network_campaign.json")

    benchmark = tmp_path / "benchmark.yml"
    _pending_benchmark(benchmark)
    repro = tmp_path / "repro_targets.yaml"
    repro.write_text("# measured targets\n")
    cards = tmp_path / "_content.yaml"
    cards.write_text("\n".join(
        f"{method.public_key}:\n  status: pending\n  axis: backbone"
        for method in NETWORK_METHODS
    ) + "\n")
    origins = tmp_path / "cell_origin.tsv"
    origins.write_text(
        "# The 18-row Network table is intentionally absent while the corrected five-seed,\n"
        "# literal nested-LOSO campaign is pending. The validated importer restores exactly\n"
        "# 54 Network origin rows together with the new public measurements.\n"
        "# algorithm\tdataset\torigin\tconfidence\tbasis\n"
    )
    results = tmp_path / "RESULTS.md"
    results.write_text(
        "# Results\n\n## Network (backbone) — corrected campaign pending\nPending\n\n"
        "## Alignment\nMeasured alignment table.\n"
    )

    _apply_values(certificate, benchmark)
    _apply_publication_metadata(
        certificate,
        campaign_root,
        benchmark,
        repro,
        cards,
        origins,
        results,
    )

    network = yaml.safe_load(benchmark.read_text())["tables"][0]
    assert len(network["rows"]) == 18
    assert all("na_reason" not in row for row in network["rows"])
    assert all(
        cell["mean"] is not None and cell["std"] is not None
        for row in network["rows"]
        for cell in row["acc"].values()
    )

    targets = yaml.safe_load(repro.read_text())
    assert set(targets) == {method.public_key for method in NETWORK_METHODS}
    assert all(target["seeds"] == 5 for target in targets.values())
    assert all(target["measurement_runner"] == "tune_networks" for target in targets.values())
    assert cards.read_text().count("status: validated") == 18
    origin_rows = [line for line in origins.read_text().splitlines() if not line.startswith("#")]
    assert len(origin_rows) == 54
    assert all("measurement-host" in line for line in origin_rows)
    assert "validated five-seed nested LOSO" in results.read_text()
    assert "Pending" not in results.read_text()


def test_selection_validator_recomputes_policy_from_inner_leaves(monkeypatch, tmp_path):
    dataset = "Synthetic"
    target = 0
    method = NETWORK_METHODS[0]
    learning_rates = [0.001, 0.003]
    monkeypatch.setitem(importer.EXPECTED_TARGETS, dataset, [0, 1, 2])
    monkeypatch.setattr(importer, "EXPECTED_LRS", learning_rates)
    method_root = tmp_path / method.public_key
    common = {
        "mode": "production",
        "dataset": dataset,
        "method": method.__dict__,
        "outer_target": target,
        "source_sha256": "source",
        "data_sha256": "data",
    }
    values = {
        (1, 0.001): (70.0, 5),
        (2, 0.001): (80.0, 7),
        (1, 0.003): (90.0, 3),
        (2, 0.003): (60.0, 9),
    }
    for (inner, learning_rate), (validation, best_epoch) in values.items():
        leaf = {
            "identity": {
                **common,
                "phase": "inner_selection",
                "inner_validation": inner,
                "learning_rate": learning_rate,
            },
            "is_measurement": True,
            "val_primary": validation,
            "best_epoch": best_epoch,
        }
        path = (
            method_root / "selection" / "outer0" / f"inner{inner}"
            / f"lr{importer.tune_networks._tag(learning_rate)}.json"
        )
        atomic_json_dump(leaf, path)

    candidates = {
        "0.001": {
            "mean_validation": 75.0,
            "per_inner_subject": {"1": 70.0, "2": 80.0},
            "best_epoch_by_inner_subject": {"1": 5, "2": 7},
            "median_low_best_epoch": 5,
        },
        "0.003": {
            "mean_validation": 75.0,
            "per_inner_subject": {"1": 90.0, "2": 60.0},
            "best_epoch_by_inner_subject": {"1": 3, "2": 9},
            "median_low_best_epoch": 3,
        },
    }
    summary_path = method_root / "selection" / "outer0" / "summary.json"
    summary = {
        "identity": {
            **common,
            "phase": "outer_selection_summary",
            "inner_subjects": [1, 2],
            "learning_rates": learning_rates,
        },
        "is_measurement": True,
        "selected_lr": 0.003,
        "selected_epochs": 3,
        "candidates": candidates,
    }
    atomic_json_dump(summary, summary_path)
    assert importer._validate_selection(method_root, dataset, method, target) == {
        "selected_lr": 0.003,
        "selected_epochs": 3,
        "source_sha256": "source",
        "data_sha256": "data",
    }

    summary["selected_lr"] = 0.001
    atomic_json_dump(summary, summary_path)
    with pytest.raises(RuntimeError, match="does not follow policy"):
        importer._validate_selection(method_root, dataset, method, target)


def test_final_validator_rejects_a_partial_target_prediction(monkeypatch, tmp_path):
    dataset = "BNCI2014002"
    method = NETWORK_METHODS[0]
    target = 0
    seed = 1
    selection = {
        "selected_lr": 0.001,
        "selected_epochs": 7,
        "source_sha256": "source",
        "data_sha256": "data",
    }
    identity = {
        "phase": "outer_final",
        "mode": "production",
        "dataset": dataset,
        "outer_target": target,
        "final_seed": seed,
        "selected_lr": selection["selected_lr"],
        "selected_epochs": selection["selected_epochs"],
        "source_sha256": selection["source_sha256"],
        "data_sha256": selection["data_sha256"],
        "method": method.__dict__,
    }
    record_path = (
        tmp_path / "final" / f"outer{target}" / f"seed{seed}" / "record.json"
    )
    record = {
        "identity": identity,
        "is_measurement": True,
        "checkpoint_reload": {"passed": True},
        "predictions_file": "predictions.npz",
        "metrics": {},
    }
    atomic_json_dump(record, record_path)
    y_true = np.array([0, 1, 0, 1])
    logits = np.array([[2.0, 1.0], [1.0, 2.0], [2.0, 1.0], [1.0, 2.0]])
    y_score = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    np.savez(
        record_path.with_name("predictions.npz"),
        y_true=y_true,
        y_pred=logits.argmax(axis=1),
        y_score=y_score,
        logits=logits,
        subject=np.asarray(target),
        trial_index=np.arange(len(y_true)),
    )
    monkeypatch.setattr(
        importer.tune_networks,
        "_validate_final_triplet",
        lambda path, expected_identity: record,
    )
    with pytest.raises(RuntimeError, match="target trials"):
        importer._validate_final(
            tmp_path, dataset, method, target, seed, selection
        )
