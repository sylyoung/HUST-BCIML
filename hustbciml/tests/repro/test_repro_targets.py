"""Make the reproduction registry executable.

``repro_targets.yaml`` records, per method, what this benchmark measured and the
published range it should land in. Until now nothing read it except
``scripts/build_cards.py``, which copies its numbers into documentation — so the
mechanism the project relies on to guarantee that "the claimed method hits the
claimed number" never actually ran, and the file drifted out of step with
``gallery/data/benchmark.yml`` without anything noticing.

Two tiers here:

* **Consistency** (fast, runs on every commit). No training. Checks the registry
  against the rest of the repository: every row names a real, runnable
  composition; every ``reproduced`` value sits inside its own
  ``reference_range``; and every row agrees with the number the public
  leaderboard shows for the same method and dataset. That last check is the one
  that would have caught ten stale card values.

* **Reproduction** (``-m repro``, opt-in, slow, needs the real datasets). Runs
  each row's setting and asserts the measured metric falls in
  ``reference_range``. Intended for a nightly job, not a commit hook.
"""
from __future__ import annotations

import os

import pytest
import yaml

HERE = os.path.dirname(__file__)
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TARGETS_PATH = os.path.join(HERE, "repro_targets.yaml")
BENCHMARK_PATH = os.path.join(REPO, "gallery", "data", "benchmark.yml")
PRESET_DIR = os.path.join(REPO, "hustbciml", "algorithms", "presets")

def _ensemble_keys():
    """Keys in the ensemble table.

    These are post-hoc combiners applied by ``scripts/ensemble.py`` to several
    already-trained models, not pipeline compositions: there is no
    ``--algorithm <key>`` to run and no single-run reference range to reproduce,
    because their number is a property of the K-model ensemble, not of one
    method's LOSO sweep. They are therefore exempt from the preset and
    repro-registry checks — and held to their own provenance requirement instead
    (``test_ensemble_rows_carry_their_own_provenance``).
    """
    data = _load(BENCHMARK_PATH)
    keys = set()
    for table in data.get("tables", []):
        if table.get("id") != "ensemble":
            continue
        for group in table.get("groups", []) or [table]:
            for row in group.get("rows", []) or []:
                if row.get("key"):
                    keys.add(row["key"])
    return keys


def _load(path):
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


TARGETS = _load(TARGETS_PATH)
TARGET_IDS = sorted(TARGETS)


def _benchmark_rows():
    """{(key, dataset): accuracy_mean} from the public leaderboard source."""
    data = _load(BENCHMARK_PATH)
    out = {}
    for table in data.get("tables", []):
        for group in table.get("groups", []) or [table]:
            for row in group.get("rows", []) or []:
                key = row.get("key")
                if not key:
                    continue
                for ds, cell in (row.get("acc") or {}).items():
                    if isinstance(cell, dict) and cell.get("mean") is not None:
                        out[(key, ds)] = float(cell["mean"])
                    elif isinstance(cell, (int, float)):
                        out[(key, ds)] = float(cell)
    return out


@pytest.mark.parametrize("name", TARGET_IDS)
def test_target_names_a_runnable_algorithm(name):
    if name in _ensemble_keys():
        pytest.skip(f"{name} is produced by a runner script, not by --algorithm")
    assert os.path.exists(os.path.join(PRESET_DIR, f"{name}.yaml")), (
        f"repro target {name!r} has no preset, so `--algorithm {name}` — the "
        f"reproduction path the README advertises — raises FileNotFoundError")


@pytest.mark.parametrize("name", TARGET_IDS)
def test_reproduced_value_is_inside_its_own_reference_range(name):
    row = TARGETS[name]
    lo, hi = row["reference_range"]
    assert lo <= row["reproduced"] <= hi, (
        f"{name}: recorded {row['reproduced']} is outside its own reference range "
        f"[{lo}, {hi}]")


@pytest.mark.parametrize("name", TARGET_IDS)
def test_target_agrees_with_the_public_leaderboard(name):
    """The registry and ``benchmark.yml`` must not publish two different numbers
    for the same method, dataset and protocol.

    They are separate files maintained by hand, and a re-run that updated one and
    not the other is exactly how a reader ends up comparing a card against the
    leaderboard and finding two answers with no way to tell which is current.
    """
    row = TARGETS[name]
    published = _benchmark_rows().get((name, row["dataset"]))
    if published is None:
        pytest.skip(f"{name} has no {row['dataset']} row on the leaderboard")
    assert abs(published - row["reproduced"]) < 0.005, (
        f"{name} on {row['dataset']}: repro_targets.yaml says {row['reproduced']}, "
        f"benchmark.yml says {published}. One of them is stale; the cards are "
        f"generated from repro_targets.yaml and the website from benchmark.yml, so "
        f"the repository is publishing two different numbers for the same run.")


def test_every_leaderboard_key_has_a_provenance_entry():
    """Every published row should have a paper citation, reference range and
    reproduction note somewhere in the repository — which is what BUILD.md tells
    readers the ``key`` field links to."""
    missing = sorted({k for (k, _ds) in _benchmark_rows()} - set(TARGETS)
                     - _ensemble_keys())
    assert not missing, (
        f"{len(missing)} leaderboard keys have no repro_targets.yaml entry, so they "
        f"have no recorded provenance and no generated card: {missing}")


@pytest.mark.repro
@pytest.mark.parametrize("name", TARGET_IDS)
def test_reproduces_reference_range(name):
    """Run the row and assert the measured metric lands in its published range.

    Slow and dataset-dependent by nature: opt in with ``pytest -m repro`` and
    point ``HUSTBCIML_DATA_DIR`` at the prepared data.
    """
    if name in _ensemble_keys():
        pytest.skip(f"{name} is produced by a runner script, not by --algorithm")
    data_dir = os.environ.get("HUSTBCIML_DATA_DIR")
    if not data_dir:
        pytest.skip("set HUSTBCIML_DATA_DIR to the prepared dataset directory")

    from hustbciml.core.config import resolve_config
    from hustbciml.run import PROTOCOLS

    row = TARGETS[name]
    seeds = int(row.get("seeds", 1))
    results_dir = os.environ.get("HUSTBCIML_REPRO_RESULTS", "./results_repro")
    values = []
    for seed in range(1, seeds + 1):
        cfg, _ = resolve_config([
            "--algorithm", name, "--dataset", row["dataset"],
            "--protocol", row["protocol"], "--seed", str(seed), "--itr", "1",
            "--data_dir", data_dir, "--results_dir", results_dir,
        ])
        summary = PROTOCOLS[cfg.protocol](cfg).run()
        values.append(summary[row["metric"]]["mean"])

    measured = sum(values) / len(values)
    lo, hi = row["reference_range"]
    assert lo <= measured <= hi, (
        f"{name} measured {measured:.2f} over {seeds} seed(s), outside the published "
        f"reference range [{lo}, {hi}] (recorded: {row['reproduced']})")


def test_ensemble_rows_carry_their_own_provenance():
    """Every ensemble combiner row names its paper and its implementation file.

    They are exempt from the reproduction registry (no single-run reference range
    applies), so this is the check that keeps them from being the one family with
    no traceable provenance at all.
    """
    data = _load(BENCHMARK_PATH)
    bad = []
    for table in data.get("tables", []):
        if table.get("id") != "ensemble":
            continue
        for group in table.get("groups", []) or [table]:
            for row in group.get("rows", []) or []:
                if not row.get("key"):
                    continue                      # context/reference rows
                if not row.get("ref") or not row.get("code"):
                    bad.append(f"{row['key']}: ref={row.get('ref')!r} code={row.get('code')!r}")
    assert not bad, "ensemble rows missing a citation or a code path:\n  " + "\n  ".join(bad)
