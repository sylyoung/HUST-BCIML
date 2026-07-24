# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic-versioning](https://semver.org/)-style tags (`vMAJOR.MINOR.PATCH`).

A short "What's new" digest also appears in [`README.md`](README.md) and
[`README.zh-CN.md`](README.zh-CN.md); this file is the full history.

## [1.1.2] - 2026-07-24

### Changed
- Rewrote the **Method inventory** in both READMEs and reorganized how methods are
  presented on the web app.
- Transfer methods are now grouped by **when they use the unlabeled target** — source-only,
  unsupervised domain adaptation (replaces ERM), source-free (a second objective after source
  ERM), and test-time (online) — with the network configuration and per-method hyperparameter
  handling made explicit.
- Renamed the privacy family to **Privacy-preserving transfer** and expanded its explanation
  (federated vs. decentralized), reading the FedBS, SAFE, and MSDT papers.
- Rewrote the **Ensemble Learning** explanation to spell out the decentralized black-box
  protocol and the reliability-estimation families (majority voting, spectral meta-learners,
  crowd-labelling / truth-discovery aggregators, StackingNet).
- **MVCNet** is now presented as a plain network backbone; the multi-view contrastive details
  live in its source file.
- Moved **EEG-FM-Benchmark** up to the fourth featured repository, and made the featured-repo
  blurbs name the method collections each repository bundles.
- Made the Chinese web app and README more idiomatic and removed redundant English glosses on
  non-technical terms.

### Removed
- **Channel Symmetry** as a benchmarked augmenter; its rationale (the prior-art contrast that
  motivates Channel Reflection's label swap) now lives as comments in the Channel Reflection
  source. The benchmarked-approach count is now **56**.

### Added
- This `CHANGELOG.md`.

## [1.1.1] - 2026-07-24

### Changed
- Split the **Ensemble Learning** table into two sub-families, non-ensemble references and
  ensemble learning, mirroring the Transfer Learning layout, and dropped its former summary strip.
- Augmenters now appear by full name (Additive Noise, Amplitude Scaling, Frequency Shift,
  Fourier Surrogate, Frequency Recombination, and Half-Sample Recombination).
- Rewrote the benchmark and overview prose for clarity in both English and Chinese.

### Removed
- Duplicate publications: the index was de-duplicated to the single official version of each
  paper (275 → 263).

## [1.1.0] - 2026-07-24

### Added
- Ten network backbones (ADFCNN, CTNet, MSCFormer, MSVTNet, TMSA-Net, EEGWaveNet, SlimSeiz,
  FBMSNet, EEGNeX, EEG-Deformer) and an amplitude-scaling augmenter, all benchmarked across the
  three datasets over three seeds.
- Seven data-augmentation baselines from the lab's augmentation studies, each with a runnable
  preset.
- Four more lab methods (CSP-Net, DJP-MMD, LSFT, MSDT); the transfer table was regrouped into
  source-only / unsupervised-DA / source-free / test-time families.
- A faithful, full **MEKT** implementation (the Section III-C domain-adaptation projection).
- The web app's three-dataset leaderboard, lab-approach highlighting, and a searchable
  paper-to-code gallery.

### Changed
- Consolidated the benchmark package as `hustbciml` and extended the privacy-preserving
  comparison to three MOABB datasets.
- Every in-code reference now gives the full journal or conference name; the Common Spatial
  Patterns, Euclidean Alignment, and MVCNet citations were corrected and expanded.
- A held-out-source hyperparameter-selection pass refreshed the network, transfer, augmentation,
  and composite tables, replacing numbers only where a fairly selected configuration improved.

[1.1.2]: https://github.com/sylyoung/HUST-BCIML/releases/tag/v1.1.2
[1.1.1]: https://github.com/sylyoung/HUST-BCIML/releases/tag/v1.1.1
[1.1.0]: https://github.com/sylyoung/HUST-BCIML/releases/tag/v1.1.0
