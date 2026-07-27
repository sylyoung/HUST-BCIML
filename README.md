<div align="center">

# HUST-BCIML

**English** | [简体中文](README.zh-CN.md)

**The open-source code home of the Brain-Computer Interface and Machine Learning Laboratory**

Prof. Dongrui Wu &nbsp;·&nbsp; Huazhong University of Science and Technology

A unified, reproducible **EEG-decoding benchmark** &nbsp;+&nbsp; a searchable **paper-to-code gallery**.

### &nbsp;[🌐&nbsp; Open the live web app &nbsp;↗](https://sylyoung.github.io/HUST-BCIML/)&nbsp;

[![Open the live web app](https://img.shields.io/badge/sylyoung.github.io%2FHUST--BCIML-Open_the_live_web_app-2563EB?style=for-the-badge&labelColor=1e293b)](https://sylyoung.github.io/HUST-BCIML/)

<sub>searchable paper-to-code gallery&nbsp; ·&nbsp; interactive benchmark leaderboard&nbsp; ·&nbsp; runs in the browser, no install</sub>

![Python](https://img.shields.io/badge/python-3.10%2B-3776ab)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c)
![Approaches](https://img.shields.io/badge/approaches-58-4338ca)
![Datasets](https://img.shields.io/badge/datasets-3%20MOABB%20MI-059669)
![License](https://img.shields.io/badge/license-MIT-blue)

[**Official lab website**](https://lab.bciml.cn/) &nbsp;·&nbsp; [**Prof. Dongrui Wu**](https://sites.google.com/site/drwuhust/) &nbsp;·&nbsp; [**Google Scholar**](https://scholar.google.com/citations?user=UYGzCPEAAAAJ)

</div>

---

> **Scope.**
> The lab website and Prof. Wu's homepage linked above are the authoritative source for the
> laboratory profile, its members, news, and the complete publication list.
>
> **This repository is the laboratory's open-source _code_ home.** It provides a unified
> benchmark of the laboratory's EEG-decoding methods together with a map from its papers to their
> public code. It complements the laboratory pages rather than replacing them.

## Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Design principles](#design-principles)
- [Benchmark methodology](#benchmark-methodology)
- [Method inventory](#method-inventory)
- [Quickstart](#quickstart)
- [The paper-to-code gallery](#the-paper-to-code-gallery)
- [Repository layout](#repository-layout)
- [Reproduction and measurement integrity](#reproduction-and-measurement-integrity)
- [Extending the benchmark](#extending-the-benchmark)
- [Featured repositories](#featured-repositories)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)
- [License](#license)

<details>
<summary><b>What's new</b></summary>

The full version history is in [`CHANGELOG.md`](CHANGELOG.md). Recent highlights:

- **2026-07-27 (v1.2.5).** Withdrew the **Ensemble Learning** table from the leaderboard. An audit
of it found two problems. The five per-source learners are built by copying the `EA-EEGNet` preset
and swapping the backbone alone, so ShallowConvNet and CSP-Net trained at a learning rate the
Networks table's own grid search had selected against, and under a 100-epoch budget where that
table uses 300. Separately, M-MSR, GLAD and ZenCrowd sat roughly ten points below majority voting
because they emit a single class on most target subjects, which the table rendered as an ordinary
low accuracy. The combiner implementations, `scripts/decentralized.py` and the presets are
unchanged and remain runnable. No other table is affected.

- **2026-07-27 (v1.2.4).** Rewrote both READMEs in the same two registers, and corrected what
they claimed. The Chinese file still stated that runs persist model checkpoints and that
hyperparameter selection touches no target label, neither of which the code or the English text
supports, and claimed a license audit of the ported code that was never performed. Both files
reported 76 papers with public code (the count is 72) and a per-run `config.yaml` that nothing
writes. The Chinese "Design principles" also regained the two caveats it never had.

- **2026-07-27 (v1.2.3).** Completed the prose rewrite: twelve transfer and ensemble row
descriptions the v1.2.2 pass missed, which kept British spelling and four sentences that judged a
number rather than stating it.

- **2026-07-27 (v1.2.2).** Rewrote the web app's explanatory prose in both languages: the English in
the register of the lab's own papers, the Chinese in that of the lab's own 公众号 writing. Fixed the
anchor card's stale approach count, and closed a gap in `check_i18n.py` that left the library card
and the Overview prose unchecked (39 generated strings checked, now 64).

- **2026-07-27 (v1.2.1).** Bounded `scikit-learn` to `<1.8`: from 1.8 its `check_is_fitted` requires
`__sklearn_tags__`, which crowd-kit's `Wawa` does not provide, so that combiner raised before
aggregating and its leaderboard row could not be reproduced. Found by the CI added the same day, on
its first run. The other four crowd-kit combiners are unaffected.

- **2026-07-27 (v1.2.0).** Acted on a 176-finding external code review: fixed silent fallbacks and
missing guards throughout the measurement path, corrected several method implementations
(Channel Reflection, Fourier Surrogate, CSDA, RA, SML, LAA, PM, CTNet, backbone shape
probing), recorded four defects inherited verbatim from the reference implementations rather than
"fixing" them out of comparability, made the reproduction registry executable, added CI and a link
checker, and corrected the claims that did not match the code. Those fixes change what the code
computes, so **every leaderboard cell they touch was re-measured**, each on the machine that
produced the published value, because the same code gives different numbers on different BLAS
builds. [`RERUN.md`](RERUN.md) lists the cells and their provenance. The rows the release does not
touch were re-run as controls and come back identical to v1.1.x for every subject. The ensemble
table now carries mean ± std over three seeds, where before it was single-seed and mean-only. Also
adds a **Classical Pipelines** table for the two network-free rows (**58** approaches), which were
published in `RESULTS.md` but appeared on no leaderboard table and so were cross-checked by nothing.

- **2026-07-24 (v1.1.3).** Rewrote all 22 lab methods' in-source docs to match their published papers (documentation only; benchmark numbers unchanged).

- **2026-07-24 (v1.1.2).** Regrouped the transfer and ensemble families, renamed **privacy-preserving transfer**, and dropped Channel Symmetry as a benchmarked augmenter (**56** approaches).

- **2026-07-24 (v1.1.1).** Split the ensemble table, listed augmenters by full name, and de-duplicated the publication index (275 → 263).

- **2026-07-24 (v1.1.0).** Added ten network backbones, eight augmentation baselines, and five lab methods (CSP-Net, DJP-MMD, LSFT, MSDT, and a full MEKT), all benchmarked over three datasets × three seeds; launched the web app's leaderboard and paper-to-code gallery.

</details>

---

## Overview

This repository contains two deliverables.

**1. The EEG-decoding benchmark**, in directory [`hustbciml/`](hustbciml/).

A self-contained framework built around a single command-line entry point and an auto-scanning
plug-in registry. On one composable pipeline it re-implements **58 EEG-decoding approaches**,
spanning data alignment, data augmentation, network backbones and transfer learning. All of them
are evaluated under a **single controlled protocol**, and every reported number carries a
per-method reproduction record. Fourteen black-box ensemble combiners are implemented as well,
and are runnable, but are not on the leaderboard; see [Ensemble aggregation](#method-inventory).

**2. The paper-to-code web app**, in directory [`docs/`](docs/).

A static web application that presents the benchmark leaderboard alongside a searchable
**paper-to-code gallery** over the laboratory's **263 publications**, 72 of which have public
code. It opens as a local file and is served by GitHub Pages with **no build step**.

## Motivation

The laboratory has published extensively on EEG decoding, but the accompanying code is
distributed across many independent repositories with heterogeneous data handling, evaluation
splits, and hyperparameter conventions.

Reproducing a single result, or comparing two approaches on equal terms, therefore requires
re-deriving the preprocessing, the cross-subject split and the training schedule of each
approach by hand. That procedure is error-prone, and a published accuracy does not remove the
difficulty on its own.

This repository addresses the problem in two complementary ways.

- It **re-implements** the approaches on one shared pipeline and evaluates them under a single
  controlled protocol, so that two rows of a leaderboard table differ in **one** component,
  apart from the rows that are explicitly marked as varying more than one.

- It **maps** the laboratory's publications to their public code, so that a reader can move
  from a paper to a runnable implementation in one step.

## Design principles

The benchmark is organized around six principles, each of which is enforced by the code and the
reporting rather than left to convention.

1. **Composability.**
   An algorithm is a named composition of stage plug-ins. Adding a method is, in the common
   case, adding a single file that conforms to a stage interface, and the registry discovers it
   by filename.

2. **Controlled comparison.**
   Every comparison varies **one** pipeline stage while holding the rest at a fixed canonical
   configuration, so that two rows differing in one component isolate the effect of that
   component. Some rows cannot be reduced to a single axis: MVCNet changes the backbone, the
   objective and the batch size; PAT changes the augmenter as well as the objective; and MEKT,
   LSFT and MSDT are network-free Riemannian approaches that use no backbone at all. Each such
   row carries an explicit "also varies" note on the leaderboard rather than being presented as
   a single-stage change. Two further axes are shared across a whole table and are stated once:
   the ERM baseline is the best checkpoint on a held-out source split, whereas the domain
   adaptation rows are the last iterate of a fixed schedule, as in their reference
   implementations, and the Networks table selects a learning rate per architecture.

3. **Measurement integrity.**
   Every reported number is a **measured** mean over three random seeds. No number is ever
   hand-set to match a paper. Each is recorded in a machine-readable reproduction file, against
   the paper's own value where the protocol matches, or against an expected-behavior band where
   it differs.

4. **Honest reporting.**
   Negative and below-baseline results are kept and explained rather than hidden. Rankings are
   **dataset-dependent** and are reported as measured. A single flat ranking across all methods
   is deliberately **not** presented.

5. **Reproducibility.**
   Runs fix their seeds and persist their **full resolved configuration**, i.e., every
   optimization, architecture and method-specific `hp` value, alongside the per-subject
   predictions and scores, in `metrics.json` and `predictions.npz` under `results/<setting>/`.
   Re-running a *different* configuration into an existing result folder is refused rather than
   silently overwriting it. Model checkpoints are not persisted; those two artifacts are what a
   number is audited from. Hyperparameter selection, where used, is described in "Hyperparameter
   selection" below, including the respect in which it is **not** purely source-only.

6. **Self-containment and zero build.**
   The web app renders from a single file with no build step, and the benchmark runs end-to-end
   on a bundled synthetic dataset with no download, so that both are inspectable before any real
   data is fetched.

## Benchmark methodology

### The pipeline

An algorithm is a composition of stage plug-ins, trained under a strategy, i.e., a learning
objective together with the training or adaptation loop that optimizes it:

```
Aligner  →  Augmenter  →  Backbone  →  Head        (trained under a Strategy)
```

- **Aligner.** A per-domain signal normalization applied before learning, for example Euclidean
  or Riemannian alignment of the trial covariances.
- **Augmenter.** A train-time transform that expands the training set.
- **Backbone.** The neural feature extractor, or `Identity` for the classical network-free
  track.
- **Head.** The classifier on top of the backbone features.
- **Strategy.** The learning objective and its train or adapt loop, such as empirical risk
  minimization, a domain-adaptation objective, or a source-free or test-time adaptation
  procedure.

### Controlled comparison

Each stage table **varies exactly one axis** and holds the remaining stages at the canonical
configuration:

```
EA  ·  no augmentation  ·  EEGNet  ·  Linear head  ·  ERM
```

Consequently, every row differs from its table's baseline in one way only, and a row's reported
delta (Δ) is its accuracy minus that table's same-dataset baseline.

### Evaluation protocol

All results are **cross-subject, leave-one-subject-out (LOSO)**: the model is trained on all but
one subject and evaluated on the held-out subject, repeated over every subject.

Each configuration is run over **three random seeds** (1, 2, 3). Reported accuracy is the **mean
over seeds**. The reported `±` is the standard deviation **across seeds**, a reproducibility
measure rather than the cross-subject spread. Deterministic, network-free methods therefore
carry a standard deviation of `0.00` by construction.

### Datasets

The full benchmark runs on three MOABB motor-imagery EEG datasets. A bundled synthetic **Toy**
dataset reproduces the entire pipeline with no download and serves as the smoke test.

| Dataset | Subjects | Channels | Classes used in the benchmark | Chance |
|---|--:|--:|---|--:|
| **BNCI2014001** | 9 | 22 | two-class (left vs. right hand) throughout, including the privacy-preserving section. The native four-class variant (both hands, feet, tongue) stays available in code | 50% |
| **BNCI2014002** | 14 | 15 | two-class (right hand vs. feet) | 50% |
| **BNCI2015001** | 12 | 13 | two-class (right hand vs. feet) | 50% |

Every table is two-class (chance 50%) on all three datasets, so the columns are directly comparable
throughout. Each family is measured against its own same-dataset baseline. The transfer families
are measured against ERM, and the privacy-preserving family against Centralized Training.

### Metrics

Accuracy is the primary metric for the motor-imagery task and is reported throughout. The
benchmark code additionally computes Cohen's κ, macro-F1, and ROC-AUC where the paradigm calls
for it. Per-subject predictions are saved so that any additional metric can be recomputed
without re-running a model.

## Method inventory

Approaches proposed by the laboratory are marked **(lab)**. Each plug-in is listed under the one
pipeline stage it changes; the privacy-preserving methods span several stages and are listed
by role.

**Signal alignment (aligners).**
Euclidean Alignment (**EA (lab)**, the default), Riemannian Alignment (**RA**), and `Identity`
(no alignment). An aligner maps the trials of each subject into a shared statistical space prior
to the backbone, and requires no label.

**Data augmentation (augmenters).**
Two electrode-space transforms run before alignment: **Channel Reflection (lab)**, a
sagittal-midline mirror that swaps the left/right label, and **Half-Sample Recombination**. The
signal- and frequency-domain augmenters run on EA-aligned trials: **CSDA (lab)** (a wavelet
cross-subject detail-swap), **additive noise**, **amplitude flip**, **amplitude scaling**,
**frequency shift**, **Fourier surrogate**, and **frequency recombination**. `Identity` applies
none.

**Network backbones.**
On a fixed EA-aligned, ERM-trained setup, only the network changes. **EEGNet** is the canonical
baseline, alongside **ShallowConvNet**, **DeepConvNet**, **EEG Conformer**, **CSP-Net (lab)**,
**TIE-EEGNet (lab)**, **KDFNet (lab)**, **DBConformer (lab)**, **MVCNet (lab)**, and a set of
recent networks (**ADFCNN**, **CTNet**, **MSCFormer**, **MSVTNet**, **TMSA-Net**, **EEGWaveNet**,
**SlimSeiz**, **FBMSNet**, **EEGNeX**, **EEG-Deformer**). Each backbone keeps its own paper's
architecture; only its learning rate is tuned, and only on held-out source subjects.

**Transfer and adaptation strategies** (vary the learning objective on a fixed EA-aligned
EEGNet). The families differ in when the unlabeled target is used and whether the source data is
still on hand:

- **Source-only training, with unlabeled target alignment**: **ERM** (the no-transfer
  baseline), **MDMAML (lab)**, **ABAT (lab)**, **PAT (lab)**. No target label is used, and the
  target is not used during training. All four nevertheless compose `aligner: EA`, and Euclidean
  Alignment estimates the whitening reference of the held-out subject from the unlabeled trials
  of that subject before prediction. That is the standard EA protocol and it is leakage-free,
  but it is transductive rather than zero-shot, and is therefore not described here as using no
  target data at all.
- **Unsupervised domain adaptation** (replaces ERM with a joint source-plus-target objective):
  **MCC**, **CDAN**, **JAN**, **DAN**, **DANN**, **MDD**, **DJP-MMD (lab)**, and the network-free
  **MEKT (lab)**.
- **Source-free adaptation** (a second objective on the target after source ERM, source data
  gone): **ASFA (lab)**, **SHOT**, and the network-free **LSFT (lab)**.
- **Test-time adaptation** (online, one target batch at a time): **T-TIME (lab)**, **DELTA**,
  **ISFDA**, **SAR**, **PL** (pseudo-labeling), **BN-adapt**, **BFT (lab)**, **Tent**.

**Classical (network-free) baselines.**
**CSP-LDA** and **Riemann-MDM** are no-transfer baselines; the classical transfer methods
**MEKT (lab)** and **LSFT (lab)** above work on Riemannian tangent-space features.

**Privacy-preserving transfer.**
Cross-subject transfer that never pools the raw EEG, measured against **Centralized Training**,
which does pool it. The **federated** approaches run a server that averages the per-subject model
updates each round: **FedAvg**, and the lab's **FedBS (lab)** and **SAFE (lab)**. The
**decentralized** **MSDT (lab)** shares only the trained per-subject models, which are fused on
the target.

**Ensemble aggregation.** *Implemented and runnable, but withdrawn from the leaderboard.*
A decentralized, black-box setting: each source subject trains five learners on its own data and
shares only the hard predicted labels, and a combiner fuses the votes without any target label.
The combiners are majority **voting** (the baseline), the spectral meta-learners **SML** and the
lab's **SML-OVR (lab)**, the lab's **StackingNet (lab)**, and a set of crowd-labeling and
truth-discovery aggregators (**Dawid-Skene**, **EBCC**, **GLAD**, **ZenCrowd**, **MACE**, **PM**,
**LAA**, **LA**, **M-MSR**, **Wawa**). They are run through
[`scripts/decentralized.py`](hustbciml/scripts/decentralized.py).

The corresponding leaderboard table was published from v1.1.x to v1.2.4 and is withdrawn, for two
reasons. First, the five per-source learners are built by copying the `EA-EEGNet` preset and
swapping the backbone alone, so ShallowConvNet and CSP-Net train at a learning rate the Networks
table's own grid search selected against, and under a 100-epoch budget where that table uses 300.
Their accuracies are therefore not the accuracies the Networks table reports for the same
backbones. Second, M-MSR, GLAD and ZenCrowd fall roughly ten points below majority voting because
they emit a single class on most target subjects, and the table displayed that degenerate output
as an ordinary low accuracy. Neither is a defect of the combiners themselves, and the code is
unchanged; what is missing is a per-backbone configuration and a class-balance diagnostic, which
the table needs before it can be published again.

## Quickstart

### Browse the web app (no install, no server)

**Live site:** **[sylyoung.github.io/HUST-BCIML](https://sylyoung.github.io/HUST-BCIML/)**. Or run it locally:

```bash
open docs/index.html          # macOS, or just double-click the file
```

The data is inlined into the page, so it renders directly from the file system and identically
when served by GitHub Pages. The application has three tabs:

- **Overview.** What the repository is, the official-lab links, and the featured code
  repositories.
- **Benchmark.** The three-dataset leaderboard with per-family explanations.
- **Papers & Code.** Search and filter the paper-to-code gallery.

### Run the benchmark

```bash
pip install -r requirements.txt

# from the repository root, so that `hustbciml` is importable
python -m hustbciml.run --list                                                # every plug-in
python -m hustbciml.run --algorithm EA-EEGNet --dataset Toy --device cpu       # synthetic, no download
python -m hustbciml.run --algorithm EA-EEGNet --dataset BNCI2014001 --itr 3    # real data, via MOABB
```

Compose an algorithm on the fly instead of naming a preset:

```bash
python -m hustbciml.run --aligner EA --augmenter CSDA --backbone DBConformer \
                        --strategy ERM --head Linear --dataset BNCI2014001 --itr 3
```

Each run writes two files under `results/<setting>/`. `metrics.json` holds the per-subject
accuracies, the mean and the standard deviation, together with the entire resolved
configuration, so that a leaderboard cell is auditable back to the exact settings that produced
it from that file alone. `predictions.npz` holds the per-subject predictions and scores. See
[`hustbciml/RESULTS.md`](hustbciml/RESULTS.md) for the current numbers and
[`hustbciml/docs/`](hustbciml/docs/index.md) for the glossary, algorithm cards, and porting guide.

## The paper-to-code gallery

The web app is generated from human-curated YAML by a single script with no framework
dependency.

- **Source of truth.** The files under [`gallery/data/`](gallery/data/):
  `publications.yml` (the 263 papers), `lab.yml` (bio, anchor project, featured repos), and
  `benchmark.yml` (the controlled-comparison leaderboard).

- **Generator.** [`gallery/build_site.py`](gallery/build_site.py) compiles those YAML files
  into `docs/data/*.js`. It requires only PyYAML.

To regenerate the web-app data after editing any YAML under `gallery/data/`:

```bash
python3 gallery/build_site.py     # requires only PyYAML
```

## Repository layout

```
HUST-BCIML/
├── docs/                       # THE WEB APP (GitHub Pages source)
│   ├── index.html
│   ├── assets/                 # style.css, app.js  (vanilla JS, no framework)
│   └── data/                   # generated: lab.js, publications.js, benchmark.js
├── gallery/                    # source of truth for the web app's data
│   ├── data/
│   │   ├── publications.yml     # 263 papers (hand-curated)
│   │   ├── lab.yml              # lab bio, anchor project, featured repos
│   │   └── benchmark.yml        # controlled-comparison leaderboard
│   └── build_site.py           # YAML → docs/data/*.js   (requires only PyYAML)
├── hustbciml/                  # THE BENCHMARK
│   ├── run.py                  # python -m hustbciml.run --algorithm EA-EEGNet --dataset BNCI2014001
│   ├── core/                   # batch, stages (ABCs), registry, pipeline, config, context
│   ├── exp/                    # exp_basic + one Exp class per protocol
│   ├── algorithms/             # aligners / augmenters / models / heads / strategies / presets
│   ├── data_provider/          # datasets, data_factory, splitters, collate
│   ├── utils/                  # metrics, seed, tools
│   ├── scripts/                # ensemble, leaderboard, compare, tuning
│   ├── tests/repro/            # repro_targets.yaml: measured vs. published, per method
│   ├── docs/                   # glossary, porting guide, per-algorithm cards
│   └── RESULTS.md              # the full leaderboard, in Markdown
└── requirements.txt
```

## Reproduction and measurement integrity

Every number in the benchmark is a **measured** three-seed mean. None is ever hand-set to match
a paper.

Each number is recorded in
[`hustbciml/tests/repro/repro_targets.yaml`](hustbciml/tests/repro/repro_targets.yaml), against
the paper's own value where the protocol matches, or against an expected-behavior band where it
differs, together with a per-method note. `tests/repro/test_repro_targets.py` checks on every
commit that every leaderboard key has an entry, that each recorded value sits inside its own
reference range, and that the registry and the public leaderboard do not publish two different
numbers for the same run. The algorithm [cards](hustbciml/docs/cards/README.md) carry the
reported-vs-reproduced table and, for each method, the upstream source it was ported from.
Upstream *license* terms are recorded where the source repository states one; where it does not,
the card says so rather than implying an audit that was not performed.

#### Hyperparameter selection, and what it does not guarantee

A small grid over the learning rate, the training length and the loss trade-offs of each
approach was scored, and the three-seed test number of the winning configuration replaced the
previous one **only where it improved on it**. Two selection signals were used, and they do not
carry the same guarantee.

* **Source-validation selection** (`select="val"`, used for the source-model hyperparameters,
  i.e., ABAT, CSDA and the per-architecture learning rates in the Networks table). The score is
  the accuracy on a held-out split of the *source* subjects. No target data of any kind enters
  it. This is the clean case.

* **Dev-subject selection** (`select="dev"`, used for the adaptation-phase hyperparameters that
  do not move the source-validation signal, i.e., ASFA, Tent, BFT, DJP-MMD, MDMAML, MSDT, LSFT
  and MVCNet). Three subjects spread across the cohort are held out as pseudo-targets, each
  scored by its own leave-one-subject-out accuracy, **against its true labels**, and one global
  configuration is chosen from that. Those three subjects are then also part of the reported
  average, so for these eight approaches, three of the nine, fourteen or twelve reported folds
  also served as the selection signal. No target label is used at *training* time, and the
  selected value is a single global one rather than one per fold. This is nonetheless the common
  practice of choosing one hyperparameter on a subject subset, and not a source-only signal, so
  it is stated here rather than left implied.

A dev-subset run is not a reportable result and cannot be mistaken for one: the run identity
carries a `dev<ids>` tag, so it lands in its own results folder.

> **Disclaimer.**
> This benchmark **re-implements** both external baselines and the laboratory's own methods
> independently.
>
> The reported results, both baseline reproductions and lab-method numbers, **may differ from
> the original papers and can contain errors**. The cause may be a protocol mismatch, a faithful
> but imperfect port, or a hyperparameter choice.
>
> If you spot a discrepancy, please open an issue or contact the maintainer. Corrections are
> welcome.

## Extending the benchmark

Add `hustbciml/algorithms/<group>/<Name>.py` defining a class that conforms to the stage
abstract base class. It **auto-registers by filename**.

Then compose it with a preset YAML, add a reproduction target once real numbers exist, and write
an algorithm card. Each new file carries a standard header with the author, date, the exact IEEE
citation, and a link to the original authors' code where one exists.

The full workflow is in the
[porting guide](hustbciml/docs/porting_guide.md).

## Featured repositories

The laboratory's flagship repositories are pinned on the [Overview tab](docs/index.html),
beginning with:

- [**DeepTransferEEG**](https://github.com/sylyoung/DeepTransferEEG)
- [**TestEnsemble**](https://github.com/sylyoung/TestEnsemble)
- [**DBConformer**](https://github.com/wzwvv/DBConformer)
- [**EEG-FM-Benchmark**](https://github.com/Dingkun0817/EEG-FM-Benchmark)
- [**EEGAdversarialBenchmark**](https://github.com/xqchen914/EEGAdversarialBenchmark)
- [**NT-Benchmark**](https://github.com/chamwen/NT-Benchmark)
- [**TLBCI**](https://github.com/drwuHUST/TLBCI)

## Roadmap

The following directions are planned for future releases.

- **Evaluation protocols.** Within-subject and cross-session splits, and an online
  (streaming) protocol, alongside the current cross-subject LOSO.
- **Paradigm breadth.** ERP/P300 (with ROC-AUC as the primary metric) and SSVEP, beyond
  motor imagery.
- **Citable release.** A versioned, DOI-archived release once the results are frozen.

## Citation

If the benchmark or gallery is useful in your work, please cite the relevant laboratory papers
and link back to this repository. Each method's source file carries its exact IEEE citation in
its header.

A versioned, citable release with a DOI is planned.

## Contact

The benchmark and web app are built and maintained by **Siyang Li**.
[homepage](https://sylyoung.github.io/) &nbsp;·&nbsp; **lsyyoungll@gmail.com**

Prof. Dongrui Wu's email address is available in any of the laboratory's publications.

## Acknowledgements

Datasets are served through [MOABB](https://moabb.neurotechx.com/) (the Mother of All BCI
Benchmarks).

Ported methods credit their original authors in each file header and in the corresponding
algorithm card. The crowd-aggregation baselines used in the ensemble and privacy-preserving
sections are credited, with their references, in
[`hustbciml/RESULTS.md`](hustbciml/RESULTS.md).

## License

This project is released under the **MIT License**. See [`LICENSE`](LICENSE) for the
full text.

The benchmark reimplements or adapts a number of previously published methods. Each
[algorithm card](hustbciml/docs/cards/README.md) documents that method's code provenance:
from-scratch reimplementations are covered by this repository's MIT license, while
implementations adapted from a specific upstream repository retain that project's original
license terms. Datasets are obtained through their respective providers under their own
terms of use.

---

<div align="center"><sub>HUST-BCIML · MIT License · Brain-Computer Interface and Machine Learning Laboratory, HUST</sub></div>
