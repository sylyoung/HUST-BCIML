# Glossary — the unified terms

The benchmark ports many EEG-decoding methods onto one architecture. That only
works if every method uses the same words for the same things. This page fixes
those words. It is the reference the [porting guide](porting_guide.md) and the
[algorithm cards](cards/README.md) assume.

All terms below reflect the actual code contracts in `core/` and
`data_provider/`; where a term maps to a field, the field is named.

---

## Data hierarchy

EEG recordings nest. From coarse to fine:

| Term | Meaning |
|---|---|
| **subject** | One person. The cross-subject protocol's domain unit. Field: `domain` (an integer id per trial). |
| **session** | One recording day for a subject. BNCI2014001 has two per subject (a train session `T` and an evaluation session `E`). |
| **run** | One continuous block within a session. |
| **trial** / **epoch** | One labelled segment cut around a cue — the classification unit. Both words mean the same object here; the code says *trial*. |
| **sample** | One time point of one channel — the finest unit. |

## Tensor conventions

Two containers carry the data, separating the numpy data provider from the
tensor `forward` signature.

| Container | Shape | Type | Where |
|---|---|---|---|
| `EEGEpochs.X` | `(N, C, T)` | float32 numpy | dataset level — what aligners, augmenters and splitters see |
| `EEGEpochs.y` | `(N,)` | int64 | class index in `[0, n_classes)`; **`-1` = unlabeled** (`UNLABELED`) |
| `EEGEpochs.domain` | `(N,)` | int64 | subject id per trial — the domain axis |
| `EEGBatch.x` | `(B, 1, C, T)` | float32 torch | the `forward` contract fed to backbones/heads/strategies |
| `EEGBatch.y` | `(B,)` | int64 | labels (`-1` where unknown) |
| `EEGBatch.domain` | `(B,)` | int64 | subject ids — lets DANN/CDAN and other domain methods work with no signature change |

- **N** trials, **C** channels, **T** time samples, **B** batch size.
- The singleton dim in `(B, 1, C, T)` is the image-style channel for 2-D convolutions; it is not the EEG channel (that is `C`).
- **Units:** amplitudes in microvolts (µV); sampling rate `sfreq` in Hz; time `T` is samples, so seconds = `T / sfreq`.

## Paradigm and primary metric

The **paradigm** is the mental task the EEG encodes; it selects the headline
metric (`utils/metrics.py`).

| Paradigm | Meaning | Primary metric |
|---|---|---|
| **MI** | motor imagery (imagined movement) | accuracy (kappa also reported) |
| **P300** / **ERP** | event-related potential to an oddball | AUC |
| **SSVEP** | steady-state visual evoked potential | accuracy |

- **chance** — accuracy of random guessing = `100 / n_classes`. BNCI2014001 is 2-class (left/right hand MI), so chance = 50%. Always quote it beside an accuracy.
- **kappa** — Cohen's κ, accuracy corrected for chance agreement; 0 = chance, 1 = perfect.
- **accuracy is a percentage** here (0–100), not a fraction.

## Transfer-learning terms

| Term | Meaning |
|---|---|
| **source** | Labelled data used to train — in the cross-subject protocol, every subject except the held-out one. |
| **target** | The held-out subject the model is evaluated on. |
| **calibration** | An optional labelled slice of the target used to adapt before testing (`--calib_ratio`). Zero for the strict cross-subject numbers in this benchmark. |
| **alignment** | A per-subject transform that recenters each subject's trials into a common reference frame so their distributions overlap — Euclidean Alignment (EA) and Riemannian Alignment (RA) here. Fit on the relevant domains only, never across the split. |
| **transductive** | A strategy that reads the *unlabeled* target during source training (DANN, CDAN, MCC …). Field: `Strategy.uses_target = True`. |
| **source-free** | Adapts to the target with no access to source data at adapt time (SHOT, ISFDA). |
| **test-time adaptation (TTA)** | The model keeps updating on the target stream at inference (T-TIME, Tent, SAR …). Field: `Strategy.mode = "tta"`. |

## Protocol (the evaluation split)

A **protocol** is the data-split rule — EEG's real "task" axis, independent of
the model. It is a subclass `Exp_<Protocol>`; the split logic lives in
`data_provider/splitters.py`.

- **cross_subject (LOSO)** — leave-one-subject-out. For each target subject: source = all other subjects, target = that subject. 9 subjects → 9 folds, averaged. This is the only protocol measured in the current benchmark.
- **within_subject**, **cross_session** — planned; not yet benchmarked.

**No-leak invariant (hard rule).** Nothing about the target may enter training:
the validation set is drawn from source domains only, and an aligner is fit on
its own domain's trials only (source aligners on source, a fresh target aligner
on the target at test time). Breaking this inflates accuracy and invalidates the
comparison.

## The five stages

An **algorithm** is a named composition of five plug-in stages
(`core/stages.py`):

```
Aligner → Augmenter → Backbone → Head , driven by a Strategy
```

| Stage | Contract (ABC) | Role |
|---|---|---|
| **Aligner** | numpy `fit(epochs)` / `transform(epochs)`; attrs `requires_labels`, `supports_online` | per-domain signal recentring, before tensors are formed |
| **Augmenter** | `__call__(batch)`, `train_only=True` | training-time data transform (Channel Reflection, CSDA) |
| **Backbone** | `nn.Module` with `forward_features`, `out_features`, verbatim `task_name` | feature extractor (EEGNet, DBConformer …) |
| **Head** | `forward(feats)`; `is_gradient` flag | maps features to logits; gradient (Linear) or classical (LDA/MDM) |
| **Strategy** | `fit(model, source, ctx)` / `predict(model, target, ctx)`; `mode ∈ {gradient, tta, fit}`, `uses_target` | owns the train / adapt / predict **procedure** |

**Strategy is separate from protocol.** `Exp_<Protocol>` owns the *data axis*
(which subjects are source vs target, folds, early stopping); the Strategy owns
the *procedure axis* (ERM / DANN / T-TIME …). One protocol runs with any
strategy.

- **`Strategy.mode`** — `gradient` (train then frozen inference), `tta` (keep adapting on the target stream), `fit` (classical fit/predict, no gradient loop, used by CSP-LDA and Riemann-MDM).
- **`Head.is_gradient`** — `True` heads train with the backbone by backprop; `False` heads (LDA, MDM) call a classical `fit`/`predict` on numpy features.

## The canonical configuration

Every controlled comparison holds four stages fixed and varies one:

```
EA · no-augmentation · EEGNet · Linear head · ERM strategy
```

- **Δacc** — a method's accuracy minus its table's baseline. It is only meaningful because the two rows differ on exactly one axis.
- A **controlled comparison** varies one stage at a time against this canonical config. A single flat ranking across all methods is deliberately *not* produced: rows would differ on different axes and the comparison would be apples-to-oranges. See [RESULTS.md](../RESULTS.md).
- **faithful negative result** — a method that lands at or below its baseline is kept and explained, not hidden (EEG Conformer here). Recording what a method actually does is the point.

## Reproducibility terms

- **seed** — a random-seed repeat of the same run. The benchmark uses 3 (seeds 1, 2, 3).
- **mean ± std** — accuracy is the mean over seeds; the ± is the standard deviation *across seeds* (a reproducibility figure), which is **not** the cross-subject spread. Deterministic fit-mode methods (CSP-LDA, Riemann-MDM) have across-seed std 0.00.
- **reference range** — the published / expected accuracy band for a method on this dataset; a ported method must land inside it (measurement integrity). Recorded per method in `tests/repro/repro_targets.yaml`.

---

See also: [porting guide](porting_guide.md) · [algorithm cards](cards/README.md) · [RESULTS.md](../RESULTS.md) · architecture in [../../README.md](../../README.md).
