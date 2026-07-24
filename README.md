<div align="center">

# HUST-BCIML

**English** | [简体中文](README.zh-CN.md)

**The open-source code home of the Brain-Computer Interface and Machine Learning Laboratory**

Prof. Dongrui Wu &nbsp;·&nbsp; Huazhong University of Science and Technology

<br>

A unified, reproducible **EEG-decoding benchmark** &nbsp;+&nbsp; a searchable **paper-to-code gallery**.

<br>

### &nbsp;[🌐&nbsp; Open the live web app &nbsp;↗](https://sylyoung.github.io/HUST-BCIML/)&nbsp;

[![Open the live web app](https://img.shields.io/badge/sylyoung.github.io%2FHUST--BCIML-Open_the_live_web_app-2563EB?style=for-the-badge&labelColor=1e293b)](https://sylyoung.github.io/HUST-BCIML/)

<sub>searchable paper-to-code gallery&nbsp; ·&nbsp; interactive benchmark leaderboard&nbsp; ·&nbsp; runs in the browser, no install</sub>

<br>

![Python](https://img.shields.io/badge/python-3.10%2B-3776ab)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c)
![Approaches](https://img.shields.io/badge/approaches-56-4338ca)
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

<br>

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

<br>

<details>
<summary><b>What's new</b></summary>

<br>

The full version history is in [`CHANGELOG.md`](CHANGELOG.md). Recent highlights:

- **2026-07-24 (v1.1.3).** Rewrote the in-source documentation of all 22 lab-proposed methods to
  match their published papers — corrected citations, each paper's own terminology and equation
  references, and honest scoping to the variant each file implements. Documentation only; no
  benchmark numbers change.

- **2026-07-24 (v1.1.2).** The method inventory was rewritten and the transfer and ensemble
  families reorganized: transfer methods are now grouped by when they use the unlabeled target
  (source-only, unsupervised domain adaptation, source-free, test-time), the privacy-preserving
  family was renamed **privacy-preserving transfer** and its explanation expanded, and the
  ensemble section now spells out the decentralized black-box protocol. **Channel Symmetry** was
  removed as a benchmarked augmenter (its rationale now lives in the Channel Reflection source),
  bringing the count to **56** approaches, and MVCNet is presented as a plain network backbone.
  A `CHANGELOG.md` was added and the Chinese pages were made more idiomatic.

- **2026-07-24 (v1.1.1).** The Ensemble Learning table was split into two sub-families,
  non-ensemble references and ensemble learning, mirroring the Transfer Learning layout, and its
  former summary strip was dropped. Augmenters now appear by full name (Additive Noise, Amplitude
  Scaling, Frequency Shift, Fourier Surrogate, Frequency Recombination, Channel Symmetry, and
  Half-Sample Recombination). The benchmark and overview prose was rewritten for clarity in both
  English and Chinese, and the publication index was de-duplicated to the single official version
  of each paper (275 → 263).

- **2026-07.** The Networks axis gained ten more backbones (ADFCNN, CTNet, MSCFormer, MSVTNet,
  TMSA-Net, EEGWaveNet, SlimSeiz, FBMSNet, EEGNeX, and EEG-Deformer) and the augmentation axis an
  eighth baseline, amplitude scaling. All were benchmarked across the three datasets over three
  seeds, and their measured accuracies now appear on the leaderboard, each with a runnable preset.

- **2026-07.** Seven data-augmentation baselines from the lab's augmentation studies joined the
  augmentation axis: additive noise, amplitude flip, frequency shift, Fourier surrogate,
  frequency recombination, channel symmetry, and half-sample recombination. Each ships with a
  runnable preset.

- **2026-07.** Every in-code reference now gives the full journal or conference name. The Common
  Spatial Patterns, Euclidean Alignment, and MVCNet citations were corrected and expanded.

- **2026-07.** A faithful, full **MEKT** implementation (the Section III-C domain-adaptation
  projection, ported from the authors' code) now tops the classical-transfer results on two of
  the three datasets.

- **2026-07.** The benchmark package was consolidated as **`hustbciml`**. The
  privacy-preserving comparison was extended to **three** MOABB datasets. A
  held-out-source hyperparameter-selection pass refreshed the network, transfer, augmentation,
  and composite tables, replacing numbers **only** where a fairly selected configuration beat
  the previous one.

- **2026-07.** Four additional lab methods were ported (**CSP-Net, DJP-MMD, LSFT, MSDT**), and
  the transfer table was regrouped into source-only / unsupervised-DA / source-free / test-time
  families.

- **2026-07.** The web app gained a three-dataset leaderboard, lab-approach highlighting, and a
  searchable paper-to-code gallery over **263** publications.

</details>

---

## Overview

This repository bundles two deliverables, **code first**.

**1. The EEG-decoding benchmark**, in directory [`hustbciml/`](hustbciml/).

A self-contained framework built around a single command-line entry point and an
auto-scanning plug-in registry. On this one composable pipeline it re-implements **56
EEG-decoding approaches** that span data alignment, data augmentation, network backbones,
transfer learning, and ensemble aggregation. It compares them head-to-head under a **single
controlled evaluation protocol**, with a per-method reproduction record for every reported
number.

**2. The paper-to-code web app**, in directory [`docs/`](docs/).

A static web application that presents the benchmark leaderboard alongside a searchable
**paper-to-code gallery** over the laboratory's **263 publications** (76 of which have public
code). It opens directly as a local file and hosts on GitHub Pages with **no build step**.

<br>

## Motivation

The laboratory has published extensively on EEG decoding, but the accompanying code is
distributed across many independent repositories with heterogeneous data handling, evaluation
splits, and hyperparameter conventions.

Reproducing a single result, or comparing two methods on equal terms, therefore requires
re-deriving each method's preprocessing, cross-subject split, and training schedule by hand.
This is error-prone, and the published accuracy numbers alone do not remove the difficulty.

This repository addresses the problem in two complementary ways.

- It **re-implements** the methods on one shared pipeline and evaluates them under a single
  controlled protocol, so that any two leaderboard rows differ in **exactly one** component.

- It **maps** the laboratory's publications to their public code, so that a reader can move
  from a paper to a runnable implementation in one step.

<br>

## Design principles

The benchmark is organized around six principles, each of which is enforced by the code and the
reporting rather than left to convention.

1. **Composability.**
   An algorithm is a named composition of stage plug-ins. Adding a method is, in the common
   case, adding a single file that conforms to a stage interface, and the registry discovers it
   by filename.

2. **Controlled comparison.**
   Every comparison varies **one** pipeline stage while holding the rest at a fixed canonical
   configuration. Two rows that differ in one component isolate the effect of that component.

3. **Measurement integrity.**
   Every reported number is a **measured** mean over three random seeds. No number is ever
   hand-set to match a paper. Each is recorded in a machine-readable reproduction file, against
   the paper's own value where the protocol matches, or against an expected-behaviour band where
   it differs.

4. **Honest reporting.**
   Negative and below-baseline results are kept and explained rather than hidden. Rankings are
   **dataset-dependent** and are reported as measured. A single flat ranking across all methods
   is deliberately **not** presented.

5. **Reproducibility.**
   Runs fix their seeds and persist their resolved configuration, per-subject predictions, and
   checkpoints. Hyperparameter selection, where used, is performed on **held-out source
   subjects only** and never touches the target or test labels.

6. **Self-containment and zero build.**
   The web app renders from a single file with no build step, and the benchmark runs end-to-end
   on a bundled synthetic dataset with no download, so that both are inspectable before any real
   data is fetched.

<br>

## Benchmark methodology

### The pipeline

An algorithm is a composition of stage plug-ins, evaluated under a training or adaptation
procedure called the strategy, which is the learning objective:

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
delta (Δ) is its accuracy minus that table's same-dataset baseline. A separate **ensemble** axis
aggregates several models and is reported apart from the single-axis tables.

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
| **BNCI2014001** | 9 | 22 | two-class (left vs. right hand) throughout, including the privacy-preserving and ensemble sections. The native four-class variant (both hands, feet, tongue) stays available in code | 50% |
| **BNCI2014002** | 14 | 15 | two-class (right hand vs. feet) | 50% |
| **BNCI2015001** | 12 | 13 | two-class (right hand vs. feet) | 50% |

Every table is two-class (chance 50%) on all three datasets, so the columns are directly comparable
throughout. Each family is measured against its own same-dataset baseline. The transfer families
are measured against ERM, the privacy-preserving family against Centralized Training, and the
ensemble table against majority voting.

### Metrics

Accuracy is the primary metric for the motor-imagery task and is reported throughout. The
benchmark code additionally computes Cohen's κ, macro-F1, and ROC-AUC where the paradigm calls
for it. Per-subject predictions are saved so that any additional metric can be recomputed
without re-running a model.

<br>

## Method inventory

Approaches proposed by the laboratory are marked **(lab)**. Each plug-in is listed under the one
pipeline stage it changes; the privacy-preserving and ensemble methods span several stages and
are listed by role.

**Signal alignment (aligners).**
Euclidean Alignment (**EA (lab)**, the default), Riemannian Alignment (**RA**), and `Identity`
(no alignment). An aligner recenters each subject's trials into a shared statistical frame before
the backbone sees them, using no labels.

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

- **Source-only** (no target at all): **ERM** (the no-transfer baseline), **MDMAML (lab)**,
  **ABAT (lab)**, **PAT (lab)**.
- **Unsupervised domain adaptation** (replaces ERM with a joint source-plus-target objective):
  **MCC**, **CDAN**, **JAN**, **DAN**, **DANN**, **MDD**, **DJP-MMD (lab)**, and the network-free
  **MEKT (lab)**.
- **Source-free adaptation** (a second objective on the target after source ERM, source data
  gone): **ASFA (lab)**, **SHOT**, and the network-free **LSFT (lab)**.
- **Test-time adaptation** (online, one target batch at a time): **T-TIME (lab)**, **DELTA**,
  **ISFDA**, **SAR**, **PL** (pseudo-labelling), **BN-adapt**, **BFT (lab)**, **Tent**.

**Classical (network-free) baselines.**
**CSP-LDA** and **Riemann-MDM** are no-transfer baselines; the classical transfer methods
**MEKT (lab)** and **LSFT (lab)** above work on Riemannian tangent-space features.

**Privacy-preserving transfer.**
Cross-subject transfer that never pools raw EEG, measured against **Centralized Training** (which
does). **Federated** methods run a server that averages per-subject model updates each round —
**FedAvg**, and the lab's **FedBS (lab)** and **SAFE (lab)** — while **decentralized**
**MSDT (lab)** shares only trained per-subject models, fused on the target.

**Ensemble aggregation.**
A decentralized, black-box setting: each source subject trains five learners on its own data and
shares only hard predicted labels, and a combiner fuses the votes with no target labels. The
combiners are majority **voting** (the baseline), the spectral meta-learners **SML** and the
lab's **SML-OVR (lab)**, the lab's **StackingNet (lab)**, and a set of crowd-labelling and
truth-discovery aggregators (**Dawid-Skene**, **EBCC**, **GLAD**, **ZenCrowd**, **MACE**, **PM**,
**LAA**, **LA**, **M-MSR**, **Wawa**).

<br>

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

Each run writes `results/<setting>/metrics.json` (per-subject accuracies plus mean/std),
`predictions.npz`, and the resolved `config.yaml`. See
[`hustbciml/README.md`](hustbciml/README.md) for the full command reference and
[`hustbciml/RESULTS.md`](hustbciml/RESULTS.md) for the current numbers.

<br>

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

<br>

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
├── hustbciml/                  # THE BENCHMARK  (see hustbciml/README.md)
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
├── references.bib              # IEEE-style BibTeX for every benchmarked method
└── requirements.txt
```

<br>

## Reproduction and measurement integrity

Every number in the benchmark is a **measured** three-seed mean. None is ever hand-set to match
a paper.

Each number is recorded in
[`hustbciml/tests/repro/repro_targets.yaml`](hustbciml/tests/repro/repro_targets.yaml), against
the paper's own value where the protocol matches, or against an expected-behaviour band where it
differs, together with a per-method note. The algorithm
[cards](hustbciml/docs/cards/README.md) carry the reported-vs-reproduced table and a
vendored-code license and provenance audit for each method.

Where hyperparameters were selected, selection was performed on **held-out source subjects
only**. A small grid over learning rate, training length, and each method's own loss trade-offs
was scored on source-validation data that never includes the target or test labels. The winning
configuration's three-seed test number replaced the previous one **only where it improved on
it**. Selection never touches the reported cohort, so the guarantee that no number is tuned to
hit a target still holds.

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

<br>

## Extending the benchmark

Add `hustbciml/algorithms/<group>/<Name>.py` defining a class that conforms to the stage
abstract base class. It **auto-registers by filename**.

Then compose it with a preset YAML, add a reproduction target once real numbers exist, and write
an algorithm card. Each new file carries a standard header with the author, date, the exact IEEE
citation, and a link to the original authors' code where one exists.

The full workflow is in the
[porting guide](hustbciml/docs/porting_guide.md).

<br>

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

<br>

## Roadmap

The following directions are planned for future releases.

- **Evaluation protocols.** Within-subject and cross-session splits, and an online
  (streaming) protocol, alongside the current cross-subject LOSO.
- **Paradigm breadth.** ERP/P300 (with ROC-AUC as the primary metric) and SSVEP, beyond
  motor imagery.
- **Citable release.** A versioned, DOI-archived release once the results are frozen.

<br>

## Citation

If the benchmark or gallery is useful in your work, please cite the relevant laboratory papers
and link back to this repository. IEEE-style BibTeX for every benchmarked method is provided in
[`references.bib`](references.bib).

A versioned, citable release with a DOI is planned.

<br>

## Contact

The benchmark and web app are built and maintained by **Siyang Li**.
[homepage](https://sylyoung.github.io/) &nbsp;·&nbsp; **lsyyoungll@gmail.com**

Prof. Dongrui Wu's email address is available in any of the laboratory's publications.

<br>

## Acknowledgements

Datasets are served through [MOABB](https://moabb.neurotechx.com/) (the Mother of All BCI
Benchmarks).

Ported methods credit their original authors in each file header and in the corresponding
algorithm card. The crowd-aggregation baselines used in the ensemble and privacy-preserving
sections are credited, with their references, in
[`hustbciml/RESULTS.md`](hustbciml/RESULTS.md).

<br>

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
