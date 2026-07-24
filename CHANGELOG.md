# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic-versioning](https://semver.org/)-style tags (`vMAJOR.MINOR.PATCH`).

A short "What's new" digest also appears in [`README.md`](README.md) and
[`README.zh-CN.md`](README.zh-CN.md); this file is the full history.

## [1.1.3] - 2026-07-24

### Changed
- Rewrote the in-source documentation of all 22 lab-proposed methods to be faithful to their
  published papers: corrected short citations, adopted each paper's own terminology and equation
  references, scoped every file to the specific variant it implements, and removed inaccurate or
  unsupported descriptions. Documentation only — no method logic or benchmark numbers change (two
  local variables were renamed to match the papers' notation).

## [1.1.2] - 2026-07-24

### Changed
- Reorganized the method inventory (READMEs + web app): transfer methods grouped by when they use
  the unlabeled target; the privacy family renamed **privacy-preserving transfer**; the ensemble
  and MVCNet explanations rewritten.

### Removed
- **Channel Symmetry** as a benchmarked augmenter (its rationale moved into the Channel Reflection
  source); benchmarked-approach count now **56**.

### Added
- This `CHANGELOG.md`.

## [1.1.1] - 2026-07-24

### Changed
- Split the Ensemble Learning table into non-ensemble references and ensemble learning; augmenters
  now listed by full name; benchmark and overview prose clarified (English + Chinese).

### Removed
- De-duplicated the publication index to one official version per paper (275 → 263).

## [1.1.0] - 2026-07-24

### Added
- Ten network backbones, an amplitude-scaling augmenter, and seven further augmentation baselines,
  all benchmarked on three datasets over three seeds; four more lab methods (CSP-Net, DJP-MMD,
  LSFT, MSDT) and a full **MEKT** implementation.
- The web app's three-dataset leaderboard and searchable paper-to-code gallery.

### Changed
- Consolidated the benchmark package as `hustbciml`; extended the privacy-preserving comparison to
  three MOABB datasets and refreshed the tables via held-out-source hyperparameter selection.

[1.1.3]: https://github.com/sylyoung/HUST-BCIML/releases/tag/v1.1.3
[1.1.2]: https://github.com/sylyoung/HUST-BCIML/releases/tag/v1.1.2
[1.1.1]: https://github.com/sylyoung/HUST-BCIML/releases/tag/v1.1.1
[1.1.0]: https://github.com/sylyoung/HUST-BCIML/releases/tag/v1.1.0
