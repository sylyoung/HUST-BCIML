# import_network_campaign.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Validate a complete Network campaign and optionally import its numeric cells."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

import numpy as np
import yaml

from hustbciml.algorithms.network_methods import NETWORK_METHODS, network_method_manifest
from hustbciml.exp.exp_basic import _json_safe
from hustbciml.scripts import tune_networks
from hustbciml.utils.io import (
    atomic_json_dump,
    atomic_write_text,
    file_sha256,
)
from hustbciml.utils.metrics import score

EXPECTED_TARGETS = {
    "BNCI2014001": list(range(9)),
    "BNCI2014002": list(range(14)),
    "BNCI2015001": list(range(12)),
}
EXPECTED_CLASS_COUNTS = {
    "BNCI2014001": [72, 72],
    "BNCI2014002": [50, 50],
    "BNCI2015001": [100, 100],
}
EXPECTED_SEEDS = list(tune_networks.PRODUCTION_SEEDS)
EXPECTED_LRS = list(tune_networks.PRODUCTION_LRS)


def _read_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"{path} is not readable strict JSON") from exc


def _assert_close(actual, expected, label: str, tolerance: float = 1e-9) -> None:
    if actual is None and expected is None:
        return
    if actual is None or expected is None:
        raise RuntimeError(f"{label} differs: {actual!r} != {expected!r}")
    if not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise RuntimeError(f"{label} differs: {actual!r} != {expected!r}")


def _validate_selection(method_root: Path, dataset: str, method, target: int) -> dict:
    expected_inner = [value for value in EXPECTED_TARGETS[dataset] if value != target]
    summary_path = method_root / "selection" / f"outer{target}" / "summary.json"
    summary = _read_json(summary_path)
    identity = summary.get("identity") or {}
    if summary.get("is_measurement") is not True or identity.get("mode") != "production":
        raise RuntimeError(f"{summary_path} is not a production measurement")
    if identity.get("phase") != "outer_selection_summary" or identity.get("dataset") != dataset:
        raise RuntimeError(f"{summary_path} has the wrong selection-summary identity")
    if identity.get("outer_target") != target or identity.get("inner_subjects") != expected_inner:
        raise RuntimeError(f"{summary_path} has incomplete literal inner LOSO coverage")
    if identity.get("learning_rates") != EXPECTED_LRS:
        raise RuntimeError(f"{summary_path} has the wrong learning-rate grid")
    if identity.get("method") != method.__dict__:
        raise RuntimeError(f"{summary_path} has the wrong method identity")

    records = {}
    for inner in expected_inner:
        for learning_rate in EXPECTED_LRS:
            leaf_path = (
                method_root
                / "selection"
                / f"outer{target}"
                / f"inner{inner}"
                / f"lr{tune_networks._tag(learning_rate)}.json"
            )
            leaf = _read_json(leaf_path)
            leaf_identity = leaf.get("identity") or {}
            if leaf.get("is_measurement") is not True:
                raise RuntimeError(f"{leaf_path} is not a production measurement")
            expected_fields = {
                "phase": "inner_selection",
                "mode": "production",
                "dataset": dataset,
                "outer_target": target,
                "inner_validation": inner,
                "learning_rate": learning_rate,
                "method": method.__dict__,
                "source_sha256": identity.get("source_sha256"),
                "data_sha256": identity.get("data_sha256"),
            }
            for key, value in expected_fields.items():
                if leaf_identity.get(key) != value:
                    raise RuntimeError(f"{leaf_path} has wrong {key}: {leaf_identity.get(key)!r}")
            forbidden = {"metrics", "prediction", "predictions_file", "y_true", "y_pred"}
            if forbidden.intersection(leaf):
                raise RuntimeError(f"{leaf_path} contains forbidden selection output")
            if leaf.get("best_epoch") is None or leaf.get("val_primary") is None:
                raise RuntimeError(f"{leaf_path} has no validation result")
            records[(inner, learning_rate)] = leaf

    candidates = {}
    for learning_rate in EXPECTED_LRS:
        leaves = [records[(inner, learning_rate)] for inner in expected_inner]
        scores = [float(leaf["val_primary"]) for leaf in leaves]
        best_epochs = [int(leaf["best_epoch"]) for leaf in leaves]
        candidates["%g" % learning_rate] = {
            "mean_validation": float(np.mean(scores)),
            "per_inner_subject": {
                str(inner): float(records[(inner, learning_rate)]["val_primary"])
                for inner in expected_inner
            },
            "best_epoch_by_inner_subject": {
                str(inner): int(records[(inner, learning_rate)]["best_epoch"])
                for inner in expected_inner
            },
            "median_low_best_epoch": int(statistics.median_low(best_epochs)),
        }
    selected_lr = min(
        EXPECTED_LRS,
        key=lambda value: (
            -candidates["%g" % value]["mean_validation"],
            candidates["%g" % value]["median_low_best_epoch"],
            value,
        ),
    )
    selected_epochs = candidates["%g" % selected_lr]["median_low_best_epoch"]
    if summary.get("candidates") != candidates:
        raise RuntimeError(f"{summary_path} candidates do not recompute from inner leaves")
    if summary.get("selected_lr") != selected_lr:
        raise RuntimeError(f"{summary_path} selected learning rate does not follow policy")
    if summary.get("selected_epochs") != selected_epochs:
        raise RuntimeError(f"{summary_path} selected epoch count does not follow policy")
    return {
        "selected_lr": selected_lr,
        "selected_epochs": selected_epochs,
        "source_sha256": identity.get("source_sha256"),
        "data_sha256": identity.get("data_sha256"),
    }


def _validate_final(
    method_root: Path,
    dataset: str,
    method,
    target: int,
    seed: int,
    selection: dict,
) -> dict:
    record_path = method_root / "final" / f"outer{target}" / f"seed{seed}" / "record.json"
    record = _read_json(record_path)
    identity = record.get("identity") or {}
    validated = tune_networks._validate_final_triplet(record_path, identity)
    if validated is None:
        raise RuntimeError(f"{record_path} disappeared during validation")
    expected_fields = {
        "phase": "outer_final",
        "mode": "production",
        "dataset": dataset,
        "outer_target": target,
        "final_seed": seed,
        "selected_lr": selection["selected_lr"],
        "selected_epochs": selection["selected_epochs"],
        "source_sha256": selection["source_sha256"],
        "data_sha256": selection["data_sha256"],
    }
    for key, value in expected_fields.items():
        if identity.get(key) != value:
            raise RuntimeError(f"{record_path} has wrong {key}: {identity.get(key)!r}")
    if identity.get("method") != method.__dict__:
        raise RuntimeError(f"{record_path} has the wrong method identity")
    if record.get("is_measurement") is not True:
        raise RuntimeError(f"{record_path} is not a production measurement")
    if (record.get("checkpoint_reload") or {}).get("passed") is not True:
        raise RuntimeError(f"{record_path} did not pass checkpoint reload validation")

    prediction_path = record_path.with_name(record["predictions_file"])
    with np.load(prediction_path, allow_pickle=False) as archive:
        y_true = np.asarray(archive["y_true"])
        y_pred = np.asarray(archive["y_pred"])
        y_score = np.asarray(archive["y_score"])
        logits = np.asarray(archive["logits"])
        stored_subject = int(np.asarray(archive["subject"]).item())
        trial_index = np.asarray(archive["trial_index"])
    if stored_subject != target:
        raise RuntimeError(f"{prediction_path} stores target {stored_subject}, expected {target}")
    if y_score.ndim != 2 or y_score.shape[1] != 2 or logits.shape != y_score.shape:
        raise RuntimeError(f"{prediction_path} must contain two-column scores and logits")
    if len(y_true) != len(y_pred) or len(y_true) != len(y_score):
        raise RuntimeError(f"{prediction_path} has inconsistent trial counts")
    if not np.array_equal(trial_index, np.arange(len(y_true))):
        raise RuntimeError(f"{prediction_path} has missing or reordered trial indices")
    if set(np.unique(y_true).tolist()) - {0, 1}:
        raise RuntimeError(f"{prediction_path} contains non-binary labels")
    expected_class_counts = EXPECTED_CLASS_COUNTS[dataset]
    if len(y_true) != sum(expected_class_counts):
        raise RuntimeError(
            f"{prediction_path} has {len(y_true)} target trials, expected "
            f"{sum(expected_class_counts)}"
        )
    class_counts = np.bincount(y_true.astype(int), minlength=2).tolist()
    if class_counts != expected_class_counts:
        raise RuntimeError(
            f"{prediction_path} class counts {class_counts} != {expected_class_counts}"
        )
    if not np.isfinite(y_score).all() or not np.isfinite(logits).all():
        raise RuntimeError(f"{prediction_path} contains non-finite outputs")
    if not np.array_equal(y_pred, logits.argmax(axis=1)):
        raise RuntimeError(f"{prediction_path} predictions disagree with logits")

    recomputed = score(y_true, y_pred, y_score, paradigm="MI", n_classes=2)
    for key, expected in recomputed.items():
        actual = record["metrics"].get(key)
        if expected is not None and isinstance(expected, float) and np.isnan(expected):
            expected = None
        _assert_close(actual, expected, f"{record_path}:{key}")
    return {
        "metrics": recomputed,
        "source_sha256": identity.get("source_sha256"),
        "data_sha256": identity.get("data_sha256"),
        "environment_lock": identity.get("environment_lock"),
        "machine": identity.get("machine"),
        "numpy_build": identity.get("numpy_build"),
        "numerical_libraries": identity.get("numerical_libraries"),
        "nvidia_driver": identity.get("nvidia_driver"),
        "dependencies": identity.get("dependencies"),
        "device": identity.get("device"),
        "torch_runtime": identity.get("torch_runtime"),
    }


def validate_campaign(root: str | Path) -> dict:
    root = Path(root).resolve()
    manifest_path = root / "network_method_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest != {"schema_version": 1, "methods": network_method_manifest()}:
        raise RuntimeError(f"{manifest_path} does not match the executable 18-method manifest")

    values = {}
    source_identities = set()
    environment_identities = set()
    dataset_identity = {}
    numerical_identity = {}
    for dataset, targets in EXPECTED_TARGETS.items():
        dataset_root = root / dataset
        expected_dirs = {method.public_key for method in NETWORK_METHODS}
        actual_dirs = {path.name for path in dataset_root.iterdir() if path.is_dir()}
        if actual_dirs != expected_dirs:
            raise RuntimeError(
                f"{dataset_root} method directories differ: expected {sorted(expected_dirs)}, "
                f"got {sorted(actual_dirs)}"
            )
        values[dataset] = {}
        for method in NETWORK_METHODS:
            method_root = dataset_root / method.public_key
            summary = _read_json(method_root / "summary.json")
            if summary.get("is_measurement") is not True:
                raise RuntimeError(f"{method_root / 'summary.json'} is not reportable")
            if summary.get("requested_targets") != targets or summary.get("completed_targets") != targets:
                raise RuntimeError(f"{method_root / 'summary.json'} has incomplete targets")
            if summary.get("requested_seeds") != EXPECTED_SEEDS or summary.get("completed_seeds") != EXPECTED_SEEDS:
                raise RuntimeError(f"{method_root / 'summary.json'} has incomplete seeds")

            selections = {}
            for target in targets:
                selections[target] = _validate_selection(
                    method_root, dataset, method, target
                )
            expected_selected = {
                str(target): {
                    "lr": selections[target]["selected_lr"],
                    "epochs": selections[target]["selected_epochs"],
                }
                for target in targets
            }
            if summary.get("selected_by_target") != expected_selected:
                raise RuntimeError(
                    f"{method_root / 'summary.json'} selected schedules differ from leaves"
                )

            per_seed = []
            for seed in EXPECTED_SEEDS:
                target_values = []
                for target in targets:
                    validated = _validate_final(
                        method_root,
                        dataset,
                        method,
                        target,
                        seed,
                        selections[target],
                    )
                    target_values.append(float(validated["metrics"]["primary"]))
                    source_identities.add(validated["source_sha256"])
                    environment_identities.add(json.dumps(
                        validated["environment_lock"], sort_keys=True
                    ))
                    dataset_identity.setdefault(dataset, set()).add(validated["data_sha256"])
                    numerical_identity.setdefault(dataset, set()).add(json.dumps({
                        "machine": validated["machine"],
                        "numpy_build": validated["numpy_build"],
                        "numerical_libraries": validated["numerical_libraries"],
                        "nvidia_driver": validated["nvidia_driver"],
                        "dependencies": validated["dependencies"],
                        "device": validated["device"],
                        "torch_runtime": validated["torch_runtime"],
                    }, sort_keys=True))
                per_seed.append(float(np.mean(target_values)))
            mean = float(np.mean(per_seed))
            std = float(np.std(per_seed, ddof=1))
            _assert_close(summary.get("primary_mean"), mean, f"{method_root}:primary_mean")
            _assert_close(summary.get("primary_std"), std, f"{method_root}:primary_std")
            values[dataset][method.public_key] = {
                "mean": mean,
                "std": std,
                "per_seed": per_seed,
            }

    if len(source_identities) != 1 or None in source_identities:
        raise RuntimeError(f"campaign mixes executable sources: {source_identities}")
    if len(environment_identities) != 1:
        raise RuntimeError("campaign mixes production environment locks")
    environment_lock = json.loads(next(iter(environment_identities)))
    if not isinstance(environment_lock, dict) or not environment_lock.get("sha256"):
        raise RuntimeError("campaign has no production environment-lock digest")
    if environment_lock.get("matches_installed") is not True or environment_lock.get(
        "mismatches"
    ):
        raise RuntimeError("campaign environment did not match its production lock")
    if not environment_lock.get("expected_runtime"):
        raise RuntimeError("campaign environment lock has no numerical runtime specification")
    for dataset in EXPECTED_TARGETS:
        if len(dataset_identity[dataset]) != 1 or None in dataset_identity[dataset]:
            raise RuntimeError(f"{dataset} mixes data identities: {dataset_identity[dataset]}")
        if len(numerical_identity[dataset]) != 1:
            raise RuntimeError(f"{dataset} mixes numerical families")

    certificate = {
        "schema_version": 1,
        "is_measurement": True,
        "methods": network_method_manifest(),
        "datasets": list(EXPECTED_TARGETS),
        "targets": EXPECTED_TARGETS,
        "seeds": EXPECTED_SEEDS,
        "learning_rates": EXPECTED_LRS,
        "n_classes": 2,
        "chance_percent": 50.0,
        "source_sha256": next(iter(source_identities)),
        "environment_lock": environment_lock,
        "data_sha256": {
            dataset: next(iter(identities))
            for dataset, identities in dataset_identity.items()
        },
        "numerical_families": {
            dataset: json.loads(next(iter(identities)))
            for dataset, identities in numerical_identity.items()
        },
        "values": values,
    }
    atomic_json_dump(_json_safe(certificate), root / "validated_network_campaign.json")
    return certificate


def _apply_values(certificate: dict, benchmark_path: Path) -> None:
    document = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
    tables = document["tables"] if isinstance(document, dict) else document
    network = next(table for table in tables if table.get("id") == "network")
    rows = {row.get("key"): row for row in network["rows"]}
    expected = {method.public_key for method in NETWORK_METHODS}
    if set(rows) != expected:
        raise RuntimeError(
            "Network table identities must be migrated to the corrected manifest before "
            "numeric import"
        )

    replacements = {}
    for dataset in EXPECTED_TARGETS:
        for method in NETWORK_METHODS:
            result = certificate["values"][dataset][method.public_key]
            replacements[(method.public_key, dataset)] = (
                round(float(result["mean"]), 2),
                round(float(result["std"]), 2),
            )

    lines = benchmark_path.read_text(encoding="utf-8").splitlines(keepends=True)
    current_table = None
    current_key = None
    current_dataset = None
    skipping_na_reason = False
    seen = set()
    output = []
    for line in lines:
        if skipping_na_reason:
            if line.startswith("      "):
                continue
            skipping_na_reason = False
        stripped = line.strip()
        if line.startswith("- id: "):
            current_table = stripped.split(":", 1)[1].strip()
            current_key = None
            current_dataset = None
        elif current_table == "network" and line.startswith("    na_reason:"):
            skipping_na_reason = True
            continue
        elif current_table == "network" and line.startswith("    key: "):
            current_key = stripped.split(":", 1)[1].strip()
            current_dataset = None
        elif current_table == "network" and line.startswith("      BNCI") and line.rstrip().endswith(":"):
            current_dataset = stripped[:-1]
        elif current_table == "network" and current_key and current_dataset:
            pair = (current_key, current_dataset)
            if pair in replacements and line.startswith("        mean: "):
                mean, _ = replacements[pair]
                line = f"        mean: {mean:.2f}\n"
                seen.add((current_key, current_dataset, "mean"))
            elif pair in replacements and line.startswith("        std: "):
                _, std = replacements[pair]
                line = f"        std: {std:.2f}\n"
                seen.add((current_key, current_dataset, "std"))
        output.append(line)

    expected_seen = {
        (method.public_key, dataset, field)
        for method in NETWORK_METHODS
        for dataset in EXPECTED_TARGETS
        for field in ("mean", "std")
    }
    if seen != expected_seen:
        raise RuntimeError(
            f"numeric importer matched {len(seen)} fields, expected {len(expected_seen)}"
        )
    atomic_write_text("".join(output), benchmark_path)


def _top_level_yaml_spans(text: str) -> dict[str, tuple[int, int]]:
    matches = list(re.finditer(r"(?m)^([^#\s][^:\n]*):\s*\n", text))
    return {
        match.group(1): (
            match.start(),
            matches[index + 1].start() if index + 1 < len(matches) else len(text),
        )
        for index, match in enumerate(matches)
    }


def _apply_repro_targets(
    certificate: dict,
    certificate_sha256: str,
    path: Path,
) -> None:
    text = path.read_text(encoding="utf-8")
    replacements = {}
    for method in NETWORK_METHODS:
        result = certificate["values"]["BNCI2014001"][method.public_key]
        per_seed = ", ".join(f"{value:.8f}" for value in result["per_seed"])
        replacements[method.public_key] = (
            f"{method.public_key}:\n"
            "  dataset: BNCI2014001\n"
            "  protocol: literal_nested_cross_subject_loso\n"
            "  metric: accuracy\n"
            f"  reproduced: {float(result['mean']):.2f}\n"
            f"  reproduced_std: {float(result['std']):.2f}\n"
            "  reference_range: null\n"
            "  seeds: 5\n"
            "  measurement_runner: tune_networks\n"
            f"  seed_subject_macro_accuracy: [{per_seed}]\n"
            f"  source: \"Validated Network campaign; executable source {certificate['source_sha256']}\"\n"
            "  note: >-\n"
            "    Five-seed subject-macro accuracy from literal target-isolated nested LOSO.\n"
            "    The complete campaign passed checkpoint reload, prediction, identity, cache,\n"
            "    numerical-family, and coverage validation before publication.\n"
            f"  certificate_sha256: {certificate_sha256}\n\n"
        )

    spans = _top_level_yaml_spans(text)
    for key in replacements:
        if key in spans:
            start, end = spans[key]
            text = text[:start] + replacements[key] + text[end:]
            spans = _top_level_yaml_spans(text)
        else:
            text = text.rstrip() + "\n\n" + replacements[key]
    atomic_write_text(text.rstrip() + "\n", path)


def _mark_network_cards_validated(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    spans = _top_level_yaml_spans(text)
    for method in NETWORK_METHODS:
        if method.public_key not in spans:
            raise RuntimeError(f"{path} has no card content for {method.public_key}")
        start, end = spans[method.public_key]
        block = text[start:end]
        if block.count("  status: pending\n") != 1:
            raise RuntimeError(
                f"{path}:{method.public_key} is not one explicit pending card"
            )
        block = block.replace("  status: pending\n", "  status: validated\n", 1)
        text = text[:start] + block + text[end:]
        spans = _top_level_yaml_spans(text)
    atomic_write_text(text, path)


def _apply_cell_origins(certificate: dict, path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    reportable_keys = {method.public_key for method in NETWORK_METHODS}
    lines = text.splitlines(keepends=True)
    existing = [
        line for line in lines
        if line and not line.startswith("#") and line.split("\t", 1)[0] in reportable_keys
    ]
    if existing:
        raise RuntimeError(
            f"{path} already contains {len(existing)} corrected Network origin rows"
        )
    marker = (
        "# The 18-row Network table is intentionally absent while the corrected five-seed,\n"
        "# literal nested-LOSO campaign is pending. The validated importer restores exactly\n"
        "# 54 Network origin rows together with the new public measurements.\n"
    )
    replacement = (
        "# The corrected 18-row Network table was imported only after the complete five-seed,\n"
        "# literal nested-LOSO campaign passed the validated campaign certificate.\n"
    )
    if marker not in text:
        raise RuntimeError(f"{path} does not carry the expected pending Network marker")
    text = text.replace(marker, replacement, 1).rstrip() + "\n"
    rows = []
    for method in NETWORK_METHODS:
        for dataset in EXPECTED_TARGETS:
            family = certificate["numerical_families"][dataset]
            machine = family.get("machine") or {}
            node = machine.get("node") or "unknown"
            rows.append(
                f"{method.public_key}\t{dataset}\t{node}\texact\t"
                "5-seed literal nested LOSO; validated_network_campaign.json\n"
            )
    if len(rows) != 54:
        raise AssertionError("Network origin import must write exactly 54 rows")
    atomic_write_text(text + "".join(rows), path)


def _apply_results_markdown(certificate: dict, benchmark_path: Path, path: Path) -> None:
    document = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
    network = next(table for table in document["tables"] if table.get("id") == "network")
    rows = []
    for row in network["rows"]:
        key = row["key"]
        label = row["name"] + (" **(lab)**" if row.get("lab") else "")
        cells = []
        for dataset in EXPECTED_TARGETS:
            value = certificate["values"][dataset][key]
            cells.append(f"{value['mean']:.2f} ± {value['std']:.2f}")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")

    section = "\n".join([
        "## Network (backbone) — validated five-seed nested LOSO",
        "_All 18 rows use the same two-class 8–32 Hz input, target EA from unlabeled trials, shared Linear",
        "head, cross-entropy ERM objective, literal target-isolated nested LOSO, and final seeds 1–5.",
        "Values are seed-level subject-macro accuracy means; ± is the sample standard deviation across",
        "the five seeds. The complete campaign passed checkpoint, prediction, source, cache, numerical-",
        "family, and coverage validation before import._",
        "",
        "| Backbone | BNCI2014001 | BNCI2014002 | BNCI2015001 |",
        "|---|--:|--:|--:|",
        *rows,
        "",
        "These are architecture transfers under one benchmark protocol, not reproductions of each paper's",
        "dataset, split, preprocessing, classifier, or optimizer. Target EA is transductive normalization",
        "from unlabeled target trials; no target label is used for alignment, selection, or training.",
        "",
    ])
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"(?ms)^## Network \(backbone\).*?(?=^## Alignment)")
    if len(pattern.findall(text)) != 1:
        raise RuntimeError(f"{path} does not contain one replaceable Network section")
    text = pattern.sub(section, text, count=1)
    atomic_write_text(text, path)


def _apply_publication_metadata(
    certificate: dict,
    campaign_root: Path,
    benchmark_path: Path,
    repro_targets_path: Path,
    cards_content_path: Path,
    cell_origin_path: Path,
    results_markdown_path: Path,
) -> None:
    certificate_path = campaign_root / "validated_network_campaign.json"
    certificate_sha256 = file_sha256(certificate_path)
    _apply_repro_targets(certificate, certificate_sha256, repro_targets_path)
    _mark_network_cards_validated(cards_content_path)
    _apply_cell_origins(certificate, cell_origin_path)
    _apply_results_markdown(certificate, benchmark_path, results_markdown_path)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hustbciml.scripts.import_network_campaign")
    parser.add_argument("--campaign_root", required=True)
    parser.add_argument(
        "--benchmark",
        default="gallery/data/benchmark.yml",
        help="source leaderboard YAML; unchanged unless --apply is passed",
    )
    parser.add_argument(
        "--repro_targets",
        default="src/hustbciml/tests/repro/repro_targets.yaml",
    )
    parser.add_argument(
        "--cards_content",
        default="src/hustbciml/docs/cards/_content.yaml",
    )
    parser.add_argument(
        "--cell_origin",
        default="src/hustbciml/scripts/cell_origin.tsv",
    )
    parser.add_argument(
        "--results_markdown",
        default="src/hustbciml/RESULTS.md",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    campaign_root = Path(args.campaign_root).resolve()
    certificate = validate_campaign(campaign_root)
    print(json.dumps(_json_safe(certificate["values"]), indent=2, allow_nan=False))
    if args.apply:
        benchmark_path = Path(args.benchmark).resolve()
        _apply_values(certificate, benchmark_path)
        _apply_publication_metadata(
            certificate,
            campaign_root,
            benchmark_path,
            Path(args.repro_targets).resolve(),
            Path(args.cards_content).resolve(),
            Path(args.cell_origin).resolve(),
            Path(args.results_markdown).resolve(),
        )
        print(f"updated validated Network publication sources from {campaign_root}")
    else:
        print("dry run: validation certificate written; benchmark YAML unchanged")


if __name__ == "__main__":
    main()
