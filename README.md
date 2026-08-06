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
![Approaches](https://img.shields.io/badge/approaches-59-4338ca)
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

- **2026-08-06 (v1.6.5).** Attribution fix: the BFT implementation credit now
  names its two actual implementers, Jiayi Ouyang & Siyang Li (Ouyang first),
  instead of the full paper author list. **No measured numbers change.**

- **2026-08-06 (v1.6.4).** Documentation fix: the "What's new" block in both
  READMEs is collapsible again. Its `<details>` opening tag was accidentally
  dropped in the v1.6.0 rewrite, which made the whole digest render expanded
  instead of collapsed by default. **No measured numbers change.**

- **2026-08-06 (v1.6.3).** Header layout simplified: the `Integrated by:` and
  `Author:` lines are removed from every algorithm header, leaving the credit chain
  as original authors → implementation → current code. **No measured numbers change.**

- **2026-08-06 (v1.6.2).** Header layout only: the credit-chain legend line is
  dropped and the `Author:` line moves below the chain, so headers are title, chain,
  author, references — no prose. **No measured numbers change.**

- **2026-08-06 (v1.6.1).** Documentation and attribution only — **no measured numbers
change.** Every algorithm source file now carries a complete credit chain in its header:
original authors (paper, venue, and official code when one exists), the implementation
author and repository, the author of the current port, and the integrator, with GitHub
links on every node. Co-first authorships are marked (†), including Channel Reflection
(Ziwei Wang† & Siyang Li†) and StackingNet. Ensemble chains were traced to their earliest
release: EBCC, ZenCrowd, PM, LA, LAA and StackingNet follow Chenhao Liu's
`Flashingcat/Golden_task-Ensemble` into the lab's TestEnsemble. The overview now also
features the lab's `wzwvv/EEGAug` data-augmentation repository.

- **2026-08-06 (v1.6.0).** All 18 Network rows were remeasured from scratch with
literal target-isolated nested LOSO and five final seeds; the validated campaign is imported and the
corrected values are published. Five rows whose identities were corrected — DeepConvNet,
ShallowConvNet, ADFCNN, EEGWaveNet, and FBMSNet — are explicit architecture transfers of the
cited references (Braindecode Deep4Net/ShallowFBCSPNet feature architectures, released-code
ADFCNN/EEGWaveNet topologies, and an 8–32 Hz adaptation of FBMSNet). MVCNet is reported on
the Network axis with its documented three-seed legacy values. Legacy values were
withdrawn rather than assigned to the corrected code. Checkpoints, predictions, selected settings,
source/data/software provenance, and resumable training state are recorded for every final fold; the
importer fails closed unless the complete campaign passes independent validation.

- **2026-07-31 (unreleased audit correction).** The Network table is now identified as a legacy
measurement pending clean remeasurement: its rows mixed fixed and per-backbone schedules, and the
old global learning-rate sweep reused reported subjects as source-validation data in other LOSO
folds. The new tuner performs target-isolated nested selection and never evaluates the outer target
during selection. Result/cache/ensemble artifacts now record source, software, preprocessing and
method parameters and fail closed on stale, partial or mismatched inputs. DeepConvNet and FBMSNet
are documented as legacy adaptations rather than paper-faithful ports. No leaderboard number was
changed in this correction.

- **2026-07-30 (v1.5.0).** Repository structure only. The library moved to `src/hustbciml/`, so the
root names what each directory holds; `hustbciml` is still the import name and every documented
command is unchanged, but the package is now installed with `pip install -e .`. Added
`pyproject.toml` (which absorbs `pytest.ini`) and removed 33 internal files — `RERUN.md` with the
seven scripts that carried out the finished v1.2.0 re-measurement, and 25 sweep and launch scripts
that hard-coded one contributor's home directory. No measured number changes.

- **2026-07-30 (v1.4.1).** Restored the "Ensemble Learning" group of approach chips on the
Overview, which v1.3.0 had dropped, so all 14 combiners are listed and the lab's SML-OVR and
StackingNet are visible there; the chip count rises from 60 to 74. The ensemble table's two
non-ensemble reference rows stay out of that list. No measured number changes and the leaderboard
renders identically.

- **2026-07-30 (v1.4.0).** Re-measured the Ensemble Learning table on a three-learner-per-source pool,
one per model family: tangent-space logistic regression, CSP-Net and EEGConformer, so each target
collects (N−1)×3 hard votes. All 45 combiner and reference cells are new. SML-OVR (lab) and binary
SML rise from a mean +0.47 over the three datasets to +1.61, positive on all three for the first
time; StackingNet (lab) from −0.08 to +0.31. `--base hetero` is removed.

- **2026-07-29 (v1.3.2).** Deleted both four-class BNCI2014001 appendices from `RESULTS.md`. The benchmark is
two-class throughout, as its header states, but two supplementary appendices still reported native
four-class results (chance 25%), and two four-class figures had leaked into the two-class per-dataset
discussion as the third entry of a triple. All of it is removed and the affected sentences restated
against the two-class evidence. No two-class number changes; the web leaderboard never carried
four-class content, and the `BNCI2014001-4` code path stays, since it is what SML-OVR's one-vs-rest
form exists for.

- **2026-07-28 (v1.3.1).** Corrected StackingNet's sum-to-one regularizer to the L1-norm form the
authors' released code uses. For non-negative weights the two agree in value but not in gradient,
so a weight the non-negativity clamp drives to zero now stays there instead of being revived, which
is the published method's own behaviour. No measured number changes: the two forms have identical
gradients while every weight stays positive, which is the case at the shipped hyperparameters, and
on cached decentralized votes over three datasets and three seeds they agree on 100.00% of trials.
`RESULTS.md`, the leaderboard and the web app are unchanged.

- **2026-07-27 (v1.3.0).** Restored the Ensemble Learning table, which v1.2.5 had briefly
withdrawn, with the numbers it carried in v1.2.4; nothing was re-measured. Removed the ensemble
block from the Overview instead: the "ensemble combiners" statistic and the "Ensemble Learning"
group of approach chips are gone, so the Overview describes the decoding pipeline alone and the
approach-chip count falls from 75 to 60. The Benchmark tab, the method inventory in both READMEs,
and every number in `RESULTS.md` are unchanged.

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
builds. Which machine produced each published cell is recorded in
[`cell_origin.tsv`](src/hustbciml/scripts/cell_origin.tsv). The rows the release does not
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

**1. The EEG-decoding benchmark**, in directory [`src/hustbciml/`](src/hustbciml/).

A self-contained framework built around a single command-line entry point and an auto-scanning
plug-in registry. On one composable pipeline it re-implements **59 EEG-decoding approaches**,
spanning data alignment, data augmentation, network backbones and transfer learning, together
with **14 ensemble combiners** counted separately, as they aggregate several trained models
rather than composing a pipeline. All of them are evaluated under a **single controlled
protocol**, and every reported number carries a per-method reproduction record.

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
   The default comparison design varies **one** pipeline stage while holding the rest at a fixed
   canonical configuration, so that two rows differing in one component isolate the effect of that
   component. Some rows cannot be reduced to a single axis: MVCNet changes the backbone, the
   objective and the batch size; PAT changes the augmenter as well as the objective; and MEKT,
   LSFT and MSDT are network-free Riemannian approaches that use no backbone at all. Each such
   row carries an explicit "also varies" note on the leaderboard rather than being presented as
   a single-stage change. The ERM baseline takes the best checkpoint on a held-out source split,
   whereas the domain-adaptation rows take the last iterate of their fixed reference schedules.
   The superseded Network table mixed fixed and tuned schedules, and its old sweep selected one
   dataset-wide learning rate from validation scores pooled across overlapping LOSO folds. Those
   values are withdrawn. Its corrected rows use literal target-isolated nested LOSO; the complete
   five-seed campaign passed validation on 2026-08-06 and its values are published.

3. **Measurement integrity.**
   Every displayed leaderboard number is a measured mean over three random seeds. The corrected
   Network table is pending and will use five seeds. No number is ever hand-set to match a paper.
   Every published value has a machine-readable reproduction record; validated Network values also
   require the complete campaign certificate.

4. **Honest reporting.**
   Negative and below-baseline results are kept and explained rather than hidden. Rankings are
   **dataset-dependent** and are reported as measured. A single flat ranking across all methods
   is deliberately **not** presented.

5. **Reproducibility.**
   New runs persist the full resolved configuration, source-tree and data digests, dependency and
   BLAS/LAPACK runtime identity, explicit preprocessing, method parameters, and per-subject
   predictions/scores. Reuse requires the complete measurement identity to match; partial,
   unreadable, legacy, or differently configured artifacts fail closed. Corrected Network runs
   additionally persist model checkpoints and epoch-level resume state. Other public values predate
   that complete artifact schema and remain labeled as historical measurements.

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

Most stage tables vary one axis and hold the remaining stages at the canonical configuration:

```
EA  ·  no augmentation  ·  EEGNet  ·  Linear head  ·  ERM
```

A row's delta (Δ) is its accuracy minus that table's same-dataset baseline. Rows that necessarily
change more than one stage state the extra changes under their names. The corrected Network table
uses its own nested-selection contract and distinct EEGNet baseline key. A separate **ensemble**
axis aggregates several models and is reported apart from the pipeline-stage tables.

### Evaluation protocol

All results are **cross-subject, leave-one-subject-out (LOSO)**: the model is trained on all but
one subject and evaluated on the held-out subject, repeated over every subject.

Displayed historical configurations use **three random seeds** (1, 2, 3); corrected Network rows
use **five seeds** (1–5). Reported accuracy is the mean over seed-level subject-macro accuracies.
The reported `±` is the sample standard deviation **across seeds**, a reproducibility measure rather
than the cross-subject spread. Deterministic, network-free methods therefore carry a standard
deviation of `0.00` by construction.

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

## Method inventory

Approaches proposed by the laboratory are marked **(lab)**. Each plug-in is listed under the one
pipeline stage it changes; the privacy-preserving and ensemble methods span several stages and
are listed by role.

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
On one fixed two-class setup, only the feature network changes: MOABB 8–32 Hz epochs, target EA
from unlabeled target trials, a shared Linear head, cross-entropy ERM, literal target-isolated nested
LOSO, and five final seeds. The corrected rows are **EEGNet**, **ShallowConvNet**,
**DeepConvNet**, **EEG Conformer**, **CSP-Net (lab)**, **TIE-EEGNet (lab)**,
**KDFNet (lab)**, **DBConformer (lab)**, **ADFCNN**, **CTNet**,
**MSCFormer**, **MSVTNet**, **TMSA-Net**, **EEGWaveNet**, **SlimSeiz**,
**FBMSNet**, **EEGNeX**, and **EEG-Deformer**, plus **MVCNet (lab)**, which keeps its
documented three-seed legacy values because it changes the backbone, the learning objective and
the batch size together. These are architecture transfers, not
reproductions of each paper's dataset, split, preprocessing, classifier, or optimizer. The five
corrected rows record material adaptation choices in their linked implementations: corrected
ADFCNN attention transpose, released-code EEGWaveNet topology, Braindecode
Deep4Net/ShallowFBCSPNet feature architectures, and six causal FBMSNet views restricted to
8–32 Hz.

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

**Ensemble aggregation.**
A decentralized, black-box setting: each source subject trains three learners on its own data — one
per model family, namely tangent-space logistic regression, CSP-Net and EEGConformer — and
shares only the hard predicted labels, and a combiner fuses the votes without any target label.
The combiners are majority **voting** (the baseline), the spectral meta-learners **SML** and the
lab's **SML-OVR (lab)**, the lab's **StackingNet (lab)**, and a set of crowd-labeling and
truth-discovery aggregators (**Dawid-Skene**, **EBCC**, **GLAD**, the simplified TestEnsemble
**ZenCrowd** EM baseline, **MACE**, three-round **PM/CRH**, **LAA**, **LA**, **M-MSR**, **Wawa**).
The ZenCrowd and PM iteration counts are method identity, not hidden runner defaults.

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
pip install -r requirements.txt   # dependencies
pip install -e .                  # the package itself, from src/

python -m hustbciml.run --list                                                # every plug-in
python -m hustbciml.run --algorithm EA-EEGNet --dataset Toy --device cpu       # synthetic, no download
python -m hustbciml.run --algorithm EA-EEGNet --dataset BNCI2014001 --itr 3    # real data, via MOABB
```

The exact Python 3.11 and CUDA package stack used for the corrected five-seed Network campaign is frozen in
[`requirements-network-production.txt`](requirements-network-production.txt). It is a measurement lock,
not a replacement for the general installation above.

Compose an algorithm on the fly instead of naming a preset:

```bash
python -m hustbciml.run --aligner EA --augmenter CSDA --backbone DBConformer \
                        --strategy ERM --head Linear --dataset BNCI2014001 --itr 3
```

Each run writes two files under `results/<setting>/`. `metrics.json` holds the per-subject
accuracies, the mean and the standard deviation, together with the entire resolved
configuration, so that a leaderboard cell is auditable back to the exact settings that produced
it from that file alone. `predictions.npz` holds the per-subject predictions and scores. See
[`src/hustbciml/RESULTS.md`](src/hustbciml/RESULTS.md) for the current numbers and
[`src/hustbciml/docs/`](src/hustbciml/docs/index.md) for the glossary, algorithm cards, and porting guide.

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
├── .github/workflows/ci.yml    # tests, generated-file checks, weekly link check
├── .gitignore                  # shared exclusions for generated/local artifacts
├── src/hustbciml/              # THE BENCHMARK  (the importable package)
│   ├── run.py                  # python -m hustbciml.run --algorithm EA-EEGNet --dataset BNCI2014001
│   ├── core/                   # batch, stages (ABCs), registry, pipeline, config, context
│   ├── exp/                    # exp_basic + one Exp class per protocol
│   ├── algorithms/             # aligners / augmenters / models / heads / strategies / presets
│   ├── data_provider/          # datasets, data_factory, splitters, collate
│   ├── utils/                  # metrics, seed, tools
│   ├── scripts/                # ensemble, leaderboard, compare, tuning, card generation
│   ├── tests/repro/            # repro_targets.yaml: measured vs. published, per method
│   ├── docs/                   # glossary, porting guide, per-algorithm cards
│   └── RESULTS.md              # the full leaderboard, in Markdown
├── docs/                       # THE WEB APP (GitHub Pages serves this folder by name)
│   ├── index.html
│   ├── assets/                 # style.css, app.js  (vanilla JS, no framework)
│   └── data/                   # generated: lab.js, publications.js, benchmark.js
├── gallery/                    # source of truth for the web app's data
│   ├── data/
│   │   ├── publications.yml     # 263 papers (hand-curated)
│   │   ├── lab.yml              # lab bio, anchor project, featured repos
│   │   └── benchmark.yml        # controlled-comparison leaderboard
│   └── build_site.py           # YAML → docs/data/*.js   (requires only PyYAML)
├── pyproject.toml              # packaging + pytest configuration
├── requirements.txt            # supported general-purpose dependencies
└── requirements-network-production.txt  # exact corrected-Network measurement lock
```

The package sits under `src/` rather than at the top level because `src/` is the package-search root
and `hustbciml/` is the actual import package. Flattening the two would remove the
`import hustbciml` namespace used by the CLI and plug-in registry. `pip install -e .` adds the
search root to Python's path.

The three small root infrastructure files are intentionally tracked. `.gitignore` is shared policy
that prevents local caches, results, and build products from entering commits. `pyproject.toml`
defines how the package is installed, its dependencies and package data, and the pytest settings.
`.github/workflows/ci.yml` tells GitHub Actions to run the non-reproduction tests and generated-file
checks on changes, plus the external-link check each week; it does not commit or push files.

## Reproduction and measurement integrity

Every displayed benchmark number is a measured multi-seed mean (three seeds for the historical
tables, five for the corrected Network table). The corrected Network values were imported only
after the complete campaign passed validation on 2026-08-06. No value is ever hand-set to match a paper.

Each published value is recorded in
[`src/hustbciml/tests/repro/repro_targets.yaml`](src/hustbciml/tests/repro/repro_targets.yaml), with a
per-method note. `tests/repro/test_repro_targets.py` checks that the leaderboard, registry, and
runnable presets agree. Corrected Network results are instead gated by the complete campaign
validator, which checks all checkpoints, predictions, five seeds, nested splits, and provenance;
a single preset run cannot replace that campaign. The algorithm
[cards](src/hustbciml/docs/cards/README.md) document each method and its upstream implementation.

#### Hyperparameter selection: legacy results and corrected procedure

The non-Network displayed values predate the corrected tuner. The historical sweep used two selection signals:

* **Global source-validation selection** (`select="val"`, including the old Network-table tuning).
  Despite the earlier documentation, this held out random source **trials**, not whole source
  subjects. It then pooled validation scores across overlapping LOSO folds to choose one
  dataset-wide value. A subject excluded as the outer target in one fold appeared with labels in
  source validation for the other folds, so every reported subject influenced the selected value.
  Candidate runs also computed and printed outer-target accuracy. This was not nested or blinded.
* **Dev-subject selection** (`select="dev"`, used for ASFA, Tent, BFT, DJP-MMD, MDMAML, MSDT,
  LSFT and MVCNet). Three cohort subjects were scored as pseudo-targets against their true labels
  and then remained in the reported average. This is target-label model selection, although no
  target label enters the training loss.

The historical publication process also retained a newly tuned value only when its test result
improved on the previous value. That adoption rule directly uses test performance and is not a
valid model-selection firewall. Related non-Network values remain displayed as historical evidence;
the superseded Network values were withdrawn.

The corrected `tune_networks.py` performs nested selection separately for each outer target. It
holds out whole source subjects, never aligns/predicts/scores the outer target during selection,
then evaluates that target with fresh models for seeds 1–5 at the selected learning rate and epoch
count. It refuses partial seeds, stale identities, legacy caches, and mixed numerical families. The
Network table's corrected values were imported after the complete campaign passed validation on
2026-08-06.

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

Add `src/hustbciml/algorithms/<group>/<Name>.py` defining a class that conforms to the stage
abstract base class. It **auto-registers by filename**.

Then compose it with a preset YAML, add a reproduction target once real numbers exist, and write
an algorithm card. Each new file carries a standard header with the author, date, the exact IEEE
citation, and a link to the original authors' code where one exists.

The full workflow is in the
[porting guide](src/hustbciml/docs/porting_guide.md).

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
- **Citable release.** The versioned 1.6.0 release is published; a DOI archive is planned.

## Citation

If the benchmark or gallery is useful in your work, please cite the relevant laboratory papers
and link back to this repository. Each method's source file carries its exact IEEE citation in
its header.

A DOI-archived release is planned; version 1.6.0 is published.

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
[`src/hustbciml/RESULTS.md`](src/hustbciml/RESULTS.md).

## License

This project is released under the **MIT License**. See [`LICENSE`](LICENSE) for the
full text.

The benchmark reimplements or adapts a number of previously published methods. Each
[algorithm card](src/hustbciml/docs/cards/README.md) documents that method's code provenance:
from-scratch reimplementations are covered by this repository's MIT license, while
implementations adapted from a specific upstream repository retain that project's original
license terms. Datasets are obtained through their respective providers under their own
terms of use.

---

<div align="center"><sub>HUST-BCIML · MIT License · Brain-Computer Interface and Machine Learning Laboratory, HUST</sub></div>
