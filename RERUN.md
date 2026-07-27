# Cells to re-measure at v1.2.0

Some of the v1.2.0 fixes change what a method computes, so the leaderboard cells they touch were
measured with the previous code. This file lists exactly which ones, and which are unaffected, so
a re-run can be scoped rather than blanket.

Everything not listed here is unchanged: the fix either could not fire on the three shipped
datasets (a guard on a condition they never hit), or was documentation only. In particular
**EA-EEGNet, the alignment table, and the whole transfer-learning table are unaffected** — the
canonical baseline every controlled delta is measured against does not move.

Until a cell is re-measured, the published number is the v1.1.x measurement. `RESULTS.md` and the
web leaderboard still show those numbers; they are not silently mixed with new ones.

---

## Must be re-measured

| Row(s) | Datasets | Why |
|---|---|---|
| **CR-EEGNet** (Channel Reflection) | BNCI2014001 | The reflection itself is unchanged on this dataset (real 10-20 montage, left/right classes), so the number should reproduce. Re-run to confirm, then the other two cells stay `n/a` — with a stated reason now, rather than a bare dash. |
| **FSurr-EEGNet** (Fourier Surrogate) | all three | The surrogate now rotates the original spectrum by a shared phase instead of rebuilding it from magnitude, and copies the DC/Nyquist bins verbatim. Different augmented trials, so a different number. |
| **CSDA-EEGNet** | all three | Forward and inverse DWT now share a boundary mode, and partners are drawn from a *different* subject. Different augmented trials. |
| **RA-EEGNet** | all three | Per-covariance ridge instead of a first-trial ridge. Changes the Riemannian mean, hence the alignment. |
| **MVCNet** | all three | Reads its loss weights from `hp` (defaults unchanged, so BNCI2014001 should reproduce). On **BNCI2014002 and BNCI2015001** the channel-reflection view is now dropped — it was generating mislabeled views there — so those two cells will move. |
| **MDMAML** | all three | ~~BatchNorm buffers are restored between inner-loop task pairs, so the meta-gradient no longer depends on the sampling order.~~ **Withdrawn — that change was a regression and has been reverted.** In train mode BatchNorm normalises with the *batch* statistics, so the running buffers never enter the forward pass and could not have affected the meta-gradient it was meant to protect. What restoring them did do was stop the buffers ever advancing past their initialisation, while inference scores in eval mode against exactly those buffers — costing 12.6 and 20.2 accuracy points on BNCI2014001 seeds 1 and 2. MDMAML now behaves as in v1.1.x and is re-run as a **control** — and it passed: all nine runs (3 datasets × 3 seeds) reproduce the v1.1.x runs to six decimals per subject. Its BNCI2015001 cell was re-run on 7002 and came back 0.85 below the published value, which this file previously explained as a published number no run reproduces. That explanation was wrong, and so was the one that replaced it: the cell was produced on 10022, so the 7002 run did compare across BLAS families — but re-running it on 10022's own family returned **the same 72.19**, and two BLAS families agreeing to two decimals is not a machine effect. The actual cause is that this cell is published from a `tune_algorithm.py` verdict, at a *selected* configuration, while the sweep runs the preset's defaults. Re-running the selection returns **73.06 ± 0.23** (per-seed 73.38 / 72.92 / 72.88) at the same selected configuration and with the same grid scores as the published verdict — identical, not merely close. **MDMAML therefore reproduces as a control on all three datasets and this release does not change any of its cells.** |
| **The whole decentralized ensemble table** (all 15 rows) | all three | Three combiners changed their own maths — SML's global-sign repair instead of element-wise `abs`, LAA's logits instead of double-softmaxed probabilities, PM's current-round normalisation and random tie-breaking. But the table cannot be re-measured three rows at a time: the decentralized ensemble's per-source learners are Tangent+LDA, Tangent+SVM, EEGNet, **ShallowConvNet** and CSP-Net, and ShallowConvNet is one of the shape-probing backbones below. Its initial weights move, so the hard votes every combiner consumes move, so every row moves — including majority voting, which is the table's own baseline. |
| **The multi-seed ensemble table** (K = 5 seeds of T-TIME, `RESULTS.md` §"Multi-seed ensemble") | all three | Affected by the *same* three combiner fixes, and easy to miss because "the ensemble table" above reads as if it covered it — it does not. This is a separate experiment with different base models, and the row above is justified entirely by a base learner this one does not use. Its own base models are five seeds of T-TIME on EEGNet, and EEGNet is the one backbone that does **not** probe its output shape, so the hard votes are unchanged and only the combiners can move it. Re-measured for that reason rather than argued about, by re-running the fusion over the cached predictions on the machine that holds each dataset's base runs. **All three datasets reproduce the published values exactly**, so this section is unchanged in v1.2.0. The fixes are no-ops in this regime by construction: SML's sign repair only bites on a below-chance base learner and PM's tie-breaking only on a tie, and five seeds of one strong algorithm supply neither. |
| **EA-CTNet** | all three | Temporal kernel now scales with `sfreq`, positional table sized from the real token count, and the paper's pre-classifier dropout restored. An earlier revision of this file predicted the 512 Hz cells would move most, on the strength of the kernel change alone. Measured, the largest gain is on BNCI2014001 (+3.50), which is 250 Hz — the other two fixes are not sampling-rate-dependent, and together they dominate. This is the largest movement in the release. |
| **Networks table** — every backbone that probes its output shape: ADFCNN, CTNet, DBConformer, DeepConvNet, EEGConformer, EEG-Deformer, EEGNeX, EEGWaveNet, FBMSNet, IFNet, MSCFormer, MSVTNet, ShallowConvNet, SlimSeiz, TMSA-Net | all three | The construction-time dummy forward no longer perturbs BatchNorm statistics or advances the RNG, so these start from a different (and now *consistent*) random state. Expect a shift within seed noise, not a systematic change. **EEGNet, TIE-EEGNet, CSP-Net and KDFNet do not probe and are unaffected** — which is why the baseline and every non-network row stand. |

`MVCNet` uses the IFNet backbone, so it is in the probe set too.

**Behaviour-changing fixes deliberately absent from this table.** A fix missing from the list above
is ambiguous — considered and cleared, or overlooked? — and this release found one that really had
been overlooked (the multi-seed ensemble table, now listed). So the cleared ones are named:

* **DELTA's class-balance EMA cadence.** The fix hoists the EMA update out of the inner `--steps`
  loop, so `--steps 3` no longer ages the DOT memory three times per batch. The batch-boundary
  condition that gates the update, `(i + 1) % test_batch == 0`, is unchanged. At `steps: 1` — the
  preset's value and the config default, and what every published cell uses — the inner loop runs
  exactly once, so the two versions execute the identical sequence of updates. This is an equality
  in the control flow, not a numerical tolerance, so no cell can move and none is re-measured.
* **Channel Reflection on BNCI2014002 and BNCI2015001.** The fail-closed rule makes the augmenter
  raise on both (no paired lateral montage on 2014002, no left/right class pair on either), so the
  right outcome is no cell at all rather than a re-measured one. `benchmark.yml` accordingly carries
  a Channel Reflection value for BNCI2014001 only.

Four of the probing backbones — ShallowConvNet, DeepConvNet, EEGConformer, DBConformer — have their
learning rate *selected* rather than fixed. Re-running only their reported seeds would be wrong: the
probe fix redraws the validation curve that picked the learning rate, so the selection has to be
redone from the whole grid (`scripts/rerun_v12_nettune.sh`).

Those twelve cells were produced by `tune_networks.py`, which records its verdict in
`<results_dir>/tuned_<dataset>.json` rather than in an ordinary run directory — so the provenance
resolver, which matches published values against result trees, initially found almost nothing for
them. Reading the verdict files instead resolves **all twelve** exactly, on one machine, and
`cell_origin.tsv` carries them with the selected learning rate recorded alongside.

An earlier revision of this file said only nine resolved, and that DeepConvNet, EEGConformer and
DBConformer on **BNCI2014001** reproduced on no surviving tree anywhere. That was wrong: their means
and stds are in `hustbciml_results_nettune/tuned_BNCI2014001.json` on 7002 — the same file, for the
same dataset, that resolved ShallowConvNet. Three affected cells were therefore skipped as
provenance gaps and would have shipped with pre-fix numbers while the rest of their table was
re-measured.

Re-selecting the learning rate is not a formality. Comparing the new verdicts against the v1.1.x
ones, **six of the eight pairs that have finished chose a different learning rate** — DBConformer on
BNCI2014002 moved 0.001 → 0.003, DeepConvNet 0.003 → 0.001, EEGConformer 0.0003 → 0.0001 on both of
its datasets, ShallowConvNet 0.0003 → 0.001 and 0.0001 → 0.0003. The probe fix perturbs the RNG that
seeds the validation runs, so the curve the selection reads is redrawn and its argmax moves. Had
these cells been re-measured at their old learning rates, the numbers would have been produced by a
selection procedure the release no longer performs.

The probe set is **fifteen** backbones. An earlier revision of this file counted fourteen and listed
SlimSeiz among the unaffected — that was wrong; `SlimSeiz.py` runs the same construction-time dummy
forward to infer its feature width.

## Explicitly unaffected

* **EA-EEGNet** and every row built on EEGNet at fixed hyperparameters — the alignment table, the
  augmentation table's other rows, the whole transfer-learning table, the privacy table.
* **MCC, MDD, ADFCNN, MSCFormer** — the four upstream-inherited defects were deliberately *not*
  changed (see the CHANGELOG's "Documented" section), so their numbers stand. The ADFCNN and
  MSCFormer cells move only through the shape-probe fix above.
* **CSP-LDA and Riemann-MDM** — the network-free classical pipelines. EA is untouched and there is no
  backbone, so nothing in this release can reach them. Verified rather than argued: re-run on their
  origin box by `rerun_v12_classical.sh` and **bit-identical to v1.1.x per subject on every seed**,
  which also re-confirms the exactly-zero across-seed std the table reports. These two rows join the
  leaderboard in v1.2.0 (see the CHANGELOG), so they are checked here for the first time.
* Every claim on this list was checked by re-running the row on the new code and requiring it to
  reproduce its v1.1.x result **per subject**, not just to the two published decimals. `EA-EEGNet`,
  `NoAlign-EEGNet`, `Noise-EEGNet` and `FShift-EEGNet` are run on all three datasets for exactly
  that purpose; the sweep calls them control rows. **Result: 10 of 10 identical** — 7 in the
  10022/20022 family, 2 on 7002, 1 on 60022, matching on all 9, 12 or 14 subjects as the dataset
  requires. A control is compared seed 1 against seed 1, so its mean is not the published
  three-seed figure and is not meant to be. Each control counts only against a baseline from its
  own numerical family; `compare_v12.py` answers "not checkable here" rather than reaching across
  families, since a cross-family difference would be a BLAS change reported as a regression.
* `--weight_decay` plumbing: the default is `0.0` everywhere and no preset or sweep sets it, so
  honouring it changes nothing that was run.
* Every guard added in this release fires only on inputs the three shipped datasets do not
  produce (a domain smaller than `batch_size`, a singular covariance, a failed aggregation).

## Re-measure a cell on the machine that produced it

The benchmark is bit-reproducible on a given machine but **not across machines**. Measured spread
between two of the lab's servers reaches **2.08 accuracy points on a single seed**.

It is not the GPU. Every machine used here runs a different NVIDIA driver, yet `NoAlign-EEGNet` —
a full GPU training run, and the one control that skips Euclidean Alignment — comes back identical
everywhere, while every aligned row differs. EA whitens with a covariance inverse square root: a
LAPACK eigendecomposition, on the CPU, whose last bits training then amplifies. The machines run
the same Python and torch versions, so what differs underneath is the BLAS: the two servers whose
results agree bit-for-bit are exactly the two linked against Intel MKL, while the two linked
against a reference build sit on different CPU microarchitectures and each form their own regime.
`hustbciml/RESULTS.md` has the table.

So re-measuring a cell on a different box than the one that produced its published value yields a
delta that mixes the code change with a BLAS change. `hustbciml/scripts/cell_origin.tsv` records the
producing machine for 162 published cells; `rerun_v12.sh`, `rerun_v12_nettune.sh` and
`rerun_v12_classical.sh` read it and take only the cells belonging to their own box (`JOB_ORIGIN`).
Two of the lab's servers are bit-identical to each other and can stand in for one another; the file's
header says which, and all three scripts fold that pair onto one family name before comparing, so a
worker told its own hostname behaves the same as one told the family's.

Cells with no recorded origin are skipped rather than measured somewhere arbitrary — they are
provenance gaps, and reporting them as such is more honest than giving them a number whose
comparability is unknown.

The file's fourth column separates two things that are easy to conflate. **`exact`** means surviving
evidence reproduces the published value, so the machine is known by measurement. **`inferred`** means
it does not, and the machine was assigned from the preset's own pattern. **Every one of the 162 rows
is now `exact`**, and the three that were `inferred` are the reason the distinction earns its place.

Those three — `CSDA-EEGNet` on BNCI2014001, `MDMAML` and `MVCNet` on BNCI2015001 — were described
here as published numbers that no surviving run reproduces anywhere, a broken provenance chain. They
are nothing of the kind. All three are reproduced exactly, mean and std, by `tune_algorithm.py`
verdict files on **10022**: `hustbciml_results_qual/strat/`, `.../priv/` and `.../mvcnet/`. The
resolver missed them because it read verdict files only under one directory and only for the four
learning-rate-tuned backbones, and matched everything else against ordinary run trees.

The consequence was not cosmetic. `inferred` had put two of them on 7002 and one on 60022, so all
three were scheduled for re-measurement on a **different BLAS family than produced them** — the
single error this whole apparatus exists to prevent, arrived at through the one column that was
supposed to flag uncertainty. They are now recorded on 10022 and re-measured there (on 20022, its
bit-identical twin, since 10022's GPU is unusable). Reading every verdict file on every reachable box
also resolved **14 further cells that had no entry at all**, none of them affected by this release.

### And a verdict file records a *selected* configuration, which changes how the cell is re-measured

Fixing the machine was not enough, and the reason is the same property that let these three be
resolved at all. A `tuned_<ds>.json` does not record a run of the preset; it records the
configuration a grid search picked, together with that configuration's three-seed mean and std:

| cell | published | selection signal | selected configuration |
|---|---|---|---|
| `CSDA-EEGNet` / BNCI2014001 | 72.74 ± 1.92 | source validation | `lr 0.001, batch_size 8` |
| `MDMAML` / BNCI2015001 | 73.06 ± 0.23 | dev subjects | `meta_lr 0.003, inner_lr 0.001` |
| `MVCNet` / BNCI2015001 | 72.21 ± 0.50 | dev subjects | `lr 0.0003` |

`rerun_v12.sh` runs `--algorithm <preset>`, which takes the preset's defaults. So re-running these
three the ordinary way measures a different configuration than the one published, and the delta it
reports mixes a code change with a configuration change. That is what the first attempt did, and
`MDMAML` is what exposed it: the 10022-family re-run returned **72.19** against the 7002 run's
**72.21**. Two BLAS families agreeing to two decimals is not a machine effect, and the published
73.06 was never going to appear from either of them, because neither was running the tuned
configuration.

These cells are therefore re-measured by **re-running the selection**, with
`tune_algorithm.py`, rather than by pinning the recorded hyperparameters. Pinning would answer
"what does this configuration do under the new code"; re-selecting answers "what does this method
publish", which is what the leaderboard states — the same reasoning the Networks table already
follows, where re-selection moved six of eight learning rates. The dev-subject choice is
deterministic (`_dev_spread`), so the re-run scores against the same held-out subjects the
published verdict used.

Only cells whose evidence is a verdict file need this. The other 43 affected cells are backed by
ordinary run trees, where the preset defaults *are* what was published.

What re-selection returned:

| cell | published | re-selected configuration | v1.2.0 |
|---|---|---|---|
| `MDMAML` / BNCI2015001 | 73.06 ± 0.23 | unchanged (`meta_lr 0.003, inner_lr 0.001`) | **73.06 ± 0.23** — identical, and with the same grid scores |
| `CSDA-EEGNet` / BNCI2014001 | 72.74 ± 1.92 | **moved** to `lr 0.003, batch_size 32` | **72.45 ± 1.87** |
| `MVCNet` / BNCI2015001 | 72.21 ± 0.50 | unchanged (`lr 0.0003`) | **74.75 ± 0.10** — same configuration, moved by the code fix |

That the search moved for one cell and not the other is the case for re-selecting rather than
pinning the recorded hyperparameters: pinning would have been right by luck on MDMAML and wrong on
CSDA-EEGNet. One coincidence worth recording, because it looks like a caching bug and is not:
CSDA-EEGNet's re-selected mean matches its plain-preset run to two decimals, but the per-seed values
differ (75.08 / 70.83 / 71.45 against 75.23 / 71.14 / 70.99) and so do the stds. Two configurations
landed within 0.0002 of each other.

`MVCNet` is a third pattern, and the one that shows why an unchanged configuration is not the same
as an unchanged cell. Its argmax held, but every score in its grid moved, and by far more than
MDMAML's did — MDMAML reproduced its grid to the digit:

| learning rate | v1.1.x dev | v1.2.0 dev |
|---|--:|--:|
| 1e-4 | 71.50 | 73.33 |
| **3e-4** (selected, both times) | **72.17** | **74.83** |
| 1e-3 (the preset) | 66.33 | 73.83 |
| 3e-3 | 58.67 | 69.50 |

Same held-out subjects, and 10022 and 20022 are the same BLAS family, so neither the split nor the
machine is the variable: this is the dropped channel-reflection view. The gain is largest exactly
where the old grid was worst, which is the shape a mislabeled-view bug should have — a view whose
labels are wrong hurts most when the learning rate lets the model fit it. It also means the cell's
published 72.21 was depressed by the bug at *every* learning rate, not only at the preset.

## How to re-run

```bash
# one cell
python -m hustbciml.run --algorithm CSDA-EEGNet --dataset BNCI2014002 \
    --seed 1 --itr 3 --device cuda --results_dir <dir> --data_dir <dir>

# the Networks table (per-architecture learning rates are selected, not fixed)
python -m hustbciml.scripts.tune_networks --dataset BNCI2014001 --device cuda \
    --results_dir <dir> --data_dir <dir>
```

Then update `gallery/data/benchmark.yml` **and** `hustbciml/tests/repro/repro_targets.yaml` in the
same commit, and run `pytest hustbciml/tests/repro` — it fails if the two disagree, which is the
check that would have caught the ten stale card values this release repaired.

### Turning a finished sweep into published numbers

Four steps, one per question, and the order matters — nothing should be published before the
controls are known to hold.

```bash
# 1. aggregate one machine's results tree into a report
python -m hustbciml.scripts.extract_v12 --results_dir <sweep dir> --out v12_report_<machine>.json

# 2. did the controls hold, and how far did the affected cells move?
python -m hustbciml.scripts.compare_v12 --report v12_report_<machine>.json \
    --baseline_dir <baselines> --benchmark gallery/data/benchmark.yml --origin <machine>

# 3. write the report into the two hand-maintained sources (dry run without --write)
python -m hustbciml.scripts.apply_v12 --report v12_report_7002.json \
    --report v12_report_60022.json --report v12_report_20022.json --write

# 4. correct RESULTS.md's tables from the leaderboard it now disagrees with
python -m hustbciml.scripts.sync_results_md --write
```

Step 4 exists because `RESULTS.md` restates the leaderboard in prose and tables, and a sweep this
size moves more cells than anyone will retype correctly — 122 in this release. It shares the guard's
own cell pattern and name mapping, so the fixer and the test cannot disagree about which leaderboard
row a table row refers to. It declines three things rather than doing them badly, and reports each:
cells carrying a delta annotation, whose reference row is moving in the same edit; row order, since
several tables are sorted by their first dataset column and a re-sort has to carry the prose that
reads a ranking off it; and bold, which marks a result worth the reader's eye rather than a computed
maximum. Those are the hand-edits a sweep leaves behind, and the script's output is the list of them.

Step 2 needs the v1.1.x runs arranged as `<baselines>/<machine>/<sweep>/<run>/metrics.json`, because
the machine has to be recoverable from the path — that is what keeps a control from being scored
against another BLAS family. The v1.1.x sweeps are already on the boxes under their original names,
so the tree is a directory of symlinks, one per sweep, under a directory named for the machine. Note
that a machine with a wedged GPU is still perfectly able to serve its baselines: the 10022-family
baselines were read off 10022 over SSH while its driver was unusable.

Step 3 reads `cell_origin.tsv` again and drops any cell whose published value came from a different
family, because a results directory accumulates runs from earlier cross-machine checks and the report
is only as trustworthy as the directory it summarises.
