# Porting guide — add an algorithm

How to bring a published EEG-decoding method into the benchmark so that it is
composable, auto-registered, measured, and documented. The terms used here are
fixed in the [glossary](glossary.md); the finished record of a port is its
[algorithm card](cards/README.md).

The rule the whole benchmark rests on: **a ported method must reproduce a
published-range number, and any deviation from the paper must be disclosed.** A
port that silently underperforms, or quietly swaps in a different mechanism, is
worse than no port.

---

## The seven steps

### 1. Classify the method into stage(s)

Decide which of the five stages the method touches (see the glossary). Most
methods are a single stage:

- a new network → **Backbone** (`algorithms/models/`)
- a recentring transform → **Aligner** (`algorithms/aligners/`)
- a training-time data transform → **Augmenter** (`algorithms/augmenters/`)
- a training / adaptation procedure → **Strategy** (`algorithms/strategies/`)
- a classifier on features → **Head** (`algorithms/heads/`)

If it changes **more than one** stage it is a *composite* method (e.g. MVCNet =
IFNet backbone + a contrastive strategy). Composites are still ported, but they
do **not** go in a single-axis controlled table — they are shown against the
reference as a context number.

### 2. Vendor the core math into one ABC-conforming file

Create `algorithms/<group>/<Name>.py` with a single class `<Name>` that
subclasses the stage ABC from `core/stages.py`. **Transplant the method's math;
do not import the original lab/third-party repo** — the file must be
self-contained. The ABCs and their required members:

| Stage | Subclass | Must implement | Declares |
|---|---|---|---|
| Aligner | `Aligner` | `fit(epochs)`, `transform(epochs)` | `requires_labels`, `supports_online` |
| Augmenter | `Augmenter` | `__call__(batch)` | `train_only` |
| Backbone | `Backbone(nn.Module)` | `forward_features(x)`, set `self.out_features` | `task_name` |
| Head | `Head(nn.Module)` | `forward(feats)` | `is_gradient` |
| Strategy | `Strategy` | `fit(model, source, ctx)`, `predict(model, target, ctx)` | `mode`, `uses_target` |

Keep the tensor contract: backbones consume `(B, 1, C, T)` and return
`(B, out_features)`; aligners/augmenters operate on `EEGEpochs` / `EEGBatch`.
A `Strategy` owns its loop semantics through `mode`: `gradient` (train then
frozen), `tta` (keep adapting on the target stream), or `fit` (classical
fit/predict). Set `uses_target = True` if it reads the unlabeled target during
`fit` (transductive).

### 3. Auto-register — nothing to wire

The registry scans each stage directory, so `class <Name>` in
`algorithms/<group>/<Name>.py` is immediately available as `--<stage> <Name>`.
The invariant is **filename == class name == CLI key**. Confirm it appears:

```bash
python -m hustbciml.run --list
```

Add any new hyperparameters as argparse lines in `core/config.py`, so a
`.sh` script reproduces a run exactly.

### 4. Name the algorithm with a preset

Add `algorithms/presets/<Algorithm>.yaml` composing the stages into the named
unit the leaderboard uses. A single-axis method holds the canonical config and
changes one stage, e.g. a strategy port:

```yaml
# algorithms/presets/<Algorithm>.yaml
aligner: EA
augmenter: Identity
backbone: EEGNet
head: Linear
strategy: <Name>
```

Now it runs end-to-end:

```bash
python -m hustbciml.run --algorithm <Algorithm> --dataset BNCI2014001 --itr 3
```

### 5. Add a reproduction target (the integrity gate)

Add a row to `tests/repro/repro_targets.yaml` — the numeric registry that every
result and card draws from:

```yaml
<Algorithm>:
  dataset: BNCI2014001
  protocol: cross_subject
  metric: accuracy
  reproduced:            # fill after measuring on the server
  reproduced_std:        # std across seeds
  reference_range: [lo, hi]   # the published / expected band on this dataset
  seeds: 3
  source: "Author Year (Method); <origin repo or 'from-scratch'>"
  note: "one axis it varies; Δ vs its baseline; any caveat"
```

`reference_range` is the pass/fail band. If the measured number lands outside it,
the port is wrong (or the method genuinely differs on this dataset — investigate
before recording).

### 6. Measure, then record the number

Run all seeds on the GPU server (never locally — smoke tests only locally), then
aggregate with `compare.py`, which computes each table and the Δacc against the
canonical baseline on the full-precision means:

```bash
python -m hustbciml.scripts.compare /path/to/results --dataset BNCI2014001
```

Write the measured `reproduced` / `reproduced_std` back into the repro row, and
refresh [RESULTS.md](../RESULTS.md).

### 7. Write the card content, then generate

Add the method's prose to `docs/cards/_content.yaml` (axis, role, baseline,
paradigm, stage `config`, `mechanism`, and `implementation` — the vendored-code
and license posture). Then regenerate:

```bash
python -m hustbciml.scripts.build_cards
```

This writes `docs/cards/<Algorithm>.md` and refreshes the index. Numbers come
from the repro registry, so the card never drifts from RESULTS.md. Include a
`delta` override in `_content.yaml` only if the full-precision Δ from compare.py
differs from the difference of the displayed 2-decimal accuracies.

---

## The controlled-comparison rule

A single-axis method must be measured against the **canonical configuration**
(`EA · no-aug · EEGNet · Linear · ERM`) with exactly one stage changed. This is
what makes Δacc meaningful. Do not compare two methods that differ on two axes
and attribute the gap to one of them.

Choose the right baseline for the axis:

- backbone, alignment (from EA), strategy, EA-regime augmentation → baseline **EA-EEGNet**
- an electrode-space augmenter (Channel Reflection) that must precede spatial whitening → held at **no alignment**, baseline **NoAlign-EEGNet**

If your method needs a stage combination the canonical config does not allow
(like Channel Reflection needing no prior whitening), keep the *rest* canonical
and state the regime explicitly — do not change two things silently.

## Measurement integrity (non-negotiable)

- **Reproduce the range.** The measured number must fall in `reference_range`. No silent fallback to a weaker variant to hit a number.
- **Keep faithful negatives.** If a method lands at or below its baseline, record it and explain why (data-hungry backbone, unstable backprop-free TTA, a lab-disabled component). Do not drop it.
- **Disclose every deviation.** If the public method has a part you could not reproduce exactly (an unspecified learned module, an omitted variant), implement the faithful core, approximate the rest, and say so in the card's `implementation` and the repro `note`. BFT is the worked example: its learned learning-to-rank module is approximated by a confidence weight, disclosed in both places.
- **Never fabricate a number.** An unmeasured method has an empty `reproduced`, not a guess.
- **Audit the license.** State in the card whether the file is a from-scratch reimplementation (carries the repo's MIT license) or adapted from a specific repository (name it, and point to its original license).

## Checklist

```
[ ] classified into stage(s); composite noted if >1
[ ] algorithms/<group>/<Name>.py — self-contained, conforms to the ABC
[ ] appears in `python -m hustbciml.run --list`
[ ] new hyperparameters added to core/config.py
[ ] algorithms/presets/<Algorithm>.yaml composes it
[ ] tests/repro/repro_targets.yaml row with reference_range + source
[ ] measured on the server (all seeds); reproduced/std recorded; RESULTS.md refreshed
[ ] docs/cards/_content.yaml entry; `build_cards` regenerated
[ ] faithful negatives kept; every deviation disclosed; license audited
```

---

See also: [glossary](glossary.md) · [algorithm cards](cards/README.md) · [RESULTS.md](../RESULTS.md) · [../../README.md](../../README.md).
