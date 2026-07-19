#!/usr/bin/env python3
# build_cards.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Generate one Markdown card per benchmarked algorithm.

An algorithm card is the human-facing record of a ported method: what it does,
the exact stage composition that produced its number, its measured accuracy
against the reference range, its paper citation, and its vendored-code /
license posture. Cards are GENERATED (not hand-written) by merging two sources,
keyed by algorithm, so their numbers can never drift from the leaderboard:

  tests/repro/repro_targets.yaml  measured accuracy + std, reference range,
                                  paper citation, per-method note. The numeric
                                  registry (also consumed by the web app).
  docs/cards/_content.yaml        documentation prose: axis, stage config,
                                  mechanism, implementation/license note.

Every number originates in repro_targets.yaml; _content.yaml carries no numbers.
This mirrors the gallery/build_site.py pattern (YAML source of truth -> generated
artifact).

Writes:
  docs/cards/<key>.md    one card per method
  docs/cards/README.md   an index grouped by comparison axis

Run (from anywhere):
  python -m hustbciml.scripts.build_cards
  # or: python hustbciml/scripts/build_cards.py
"""
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))   # .../hustbciml/scripts
EEGDEC = os.path.dirname(HERE)                       # .../hustbciml

REPRO = os.path.join(EEGDEC, "tests", "repro", "repro_targets.yaml")
CONTENT = os.path.join(EEGDEC, "docs", "cards", "_content.yaml")
OUT = os.path.join(EEGDEC, "docs", "cards")

# axis key -> display title, and the order axes appear in the index
AXIS_TITLE = {
    "canonical": "Canonical reference",
    "backbone": "Network (backbone)",
    "alignment": "Alignment",
    "strategy": "Transfer / adaptation strategy",
    "augmentation": "Augmentation",
    "composite": "Composite (multi-stage)",
    "classical": "Classical (network-free)",
}
AXIS_ORDER = ["canonical", "backbone", "alignment", "strategy",
              "augmentation", "composite", "classical"]

# friendlier names for the two rows that serve as delta baselines
BASELINE_DISPLAY = {"EA-EEGNet": "EA-EEGNet", "NoAlign-EEGNet": "no-alignment EEGNet"}


def load(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def fmt_range(rng):
    if not rng:
        return "—"
    lo, hi = rng
    return f"{lo}–{hi}"


def fmt_acc(r):
    std = r.get("reproduced_std")
    if std is not None:
        return f"{r['reproduced']:.2f} ± {std:.2f}"
    return f"{r['reproduced']:.2f}"


def delta(key, c, repro):
    """Signed accuracy delta vs the row's baseline/reference.

    Prefers an explicit ``delta`` in _content.yaml — the authoritative value from
    compare.py, computed on the full-precision means — falling back to the
    difference of the displayed 2-decimal accuracies. They agree except for a few
    rows where full-precision rounding differs by 0.01; the override keeps the
    cards identical to RESULTS.md. Returns None when there is no reference.
    """
    ref_key = c.get("baseline") or c.get("reference")
    if not ref_key:
        return None
    if c.get("delta") is not None:
        return float(c["delta"])
    return round(repro[key]["reproduced"] - repro[ref_key]["reproduced"], 2)


def delta_value(key, c, repro):
    """(label, value) for the delta table row, or None when not applicable."""
    if c["role"] == "base":
        return ("Role", "canonical base — every controlled delta is measured against this row")
    if c["role"] == "control":
        return ("Role", "no-alignment control (baseline of the alignment and no-align augmentation axes)")
    d = delta(key, c, repro)
    if d is None:
        return None
    ref_key = c.get("baseline") or c.get("reference")
    kind = "baseline" if c.get("baseline") else "reference"
    ctx = " — context" if kind == "reference" else ""
    disp = BASELINE_DISPLAY.get(ref_key, ref_key)
    return (f"Δ vs {kind} ({disp}, {repro[ref_key]['reproduced']:.2f}){ctx}",
            f"{'+' if d >= 0 else ''}{d:.2f}")


def delta_short(key, c, repro):
    """Compact delta string for the index table."""
    if c["role"] == "base":
        return "base"
    if c["role"] == "control":
        return "control"
    d = delta(key, c, repro)
    if d is None:
        return ""
    return f"{'+' if d >= 0 else ''}{d:.2f}" + (" vs ref" if c.get("reference") else "")


def snippet(text, limit=130):
    """First sentence of the mechanism, capped, for the index table."""
    first = text.strip().split(". ")[0].rstrip(".") + "."
    return first if len(first) <= limit else first[: limit - 1].rstrip() + "…"


def card_md(key, c, repro):
    r = repro[key]
    axis = AXIS_TITLE.get(c["axis"], c["axis"])
    proto = r["protocol"].replace("_", "-")
    out = [
        f"# {key}",
        "",
        f"**Axis:** {axis} &nbsp;·&nbsp; **Paradigm:** {c['paradigm']} "
        f"&nbsp;·&nbsp; **Dataset:** {r['dataset']} ({proto} LOSO)",
        "",
        f"**Stage configuration:** `{c['config']}`",
        "",
        "## Mechanism",
        c["mechanism"].strip(),
        "",
        f"## Result — {r.get('seeds', 3)} seeds (1, 2, 3)",
        "",
        "| Metric | Value |",
        "|---|--:|",
        f"| Accuracy (mean ± std across seeds) | {fmt_acc(r)} |",
    ]
    dv = delta_value(key, c, repro)
    if dv:
        out.append(f"| {dv[0]} | {dv[1]} |")
    out += [
        f"| Reference range (expected band) | {fmt_range(r.get('reference_range'))} |",
        "",
        "Accuracy is the mean over seeds; ± is the standard deviation *across* "
        "seeds (a reproducibility figure, not the cross-subject spread). The "
        "reference range is the published / expected band on this dataset.",
        "",
        "## Provenance",
        f"- **Paper / source:** {r['source']}",
        f"- **Implementation:** {c['implementation'].strip()}",
        "",
        "## Note",
        r.get("note", "").strip(),
        "",
        "---",
        "_Generated by `scripts/build_cards.py` from `tests/repro/repro_targets.yaml` "
        "+ `docs/cards/_content.yaml`. Numbers mirror [RESULTS.md](../../RESULTS.md); "
        "terms are defined in [glossary.md](../glossary.md)._",
    ]
    return "\n".join(out) + "\n"


def index_md(content, repro):
    out = [
        "# Algorithm cards",
        "",
        "One card per benchmarked method, generated from the reproduction registry. "
        "Accuracy is the 3-seed mean ± the standard deviation across seeds on "
        "**BNCI2014001**, cross-subject leave-one-subject-out (9 subjects, 2-class, "
        "chance 50%). Single-axis methods show Δ against that axis's baseline; "
        "composite and classical methods show Δ against the EA-EEGNet reference.",
        "",
        "See [../../RESULTS.md](../../RESULTS.md) for the controlled-comparison tables, "
        "[../glossary.md](../glossary.md) for terms, and "
        "[../porting_guide.md](../porting_guide.md) to add a method.",
        "",
    ]
    by_axis = {}
    for k, c in content.items():
        by_axis.setdefault(c["axis"], []).append(k)
    for axis in AXIS_ORDER:
        keys = sorted(by_axis.get(axis, []), key=lambda k: -repro[k]["reproduced"])
        if not keys:
            continue
        out += [f"## {AXIS_TITLE.get(axis, axis)}", "",
                "| Method | Acc ± std | Δ | Mechanism |", "|---|--:|--:|---|"]
        for k in keys:
            out.append(f"| [{k}]({k}.md) | {fmt_acc(repro[k])} | "
                       f"{delta_short(k, content[k], repro)} | "
                       f"{snippet(content[k]['mechanism'])} |")
        out.append("")
    out += ["---", f"_Generated by `scripts/build_cards.py` — {len(content)} methods._"]
    return "\n".join(out) + "\n"


def main():
    content = load(CONTENT)
    repro = load(REPRO)
    missing, extra = set(repro) - set(content), set(content) - set(repro)
    if missing or extra:
        raise SystemExit(
            f"key mismatch — repro-only={sorted(missing)} content-only={sorted(extra)}")
    os.makedirs(OUT, exist_ok=True)
    for key, c in content.items():
        with open(os.path.join(OUT, f"{key}.md"), "w", encoding="utf-8") as f:
            f.write(card_md(key, c, repro))
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
        f.write(index_md(content, repro))
    print(f"wrote {len(content)} cards + README.md to docs/cards/")


if __name__ == "__main__":
    main()
