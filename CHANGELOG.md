# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic-versioning](https://semver.org/)-style tags (`vMAJOR.MINOR.PATCH`).

A short "What's new" digest also appears in [`README.md`](README.md) and
[`README.zh-CN.md`](README.zh-CN.md); this file is the full history.

## [1.6.6] - 2026-08-06

### Changed

- Attribution fixes: the StackingNet implementation credit now names Chenhao
  Liu & Siyang Li (Liu first), and the Channel Reflection credit Ziwei Wang &
  Siyang Li (Wang first), instead of the full paper author lists. No measured
  numbers change.
- The "What's new" digest in both READMEs now uses one-line labels per
  release; the full detail stays in this file.

## [1.6.5] - 2026-08-06

### Changed

- BFT header: the implementation credit now names its two actual implementers,
  Jiayi Ouyang & Siyang Li (Ouyang first), instead of the full paper author
  list. No measured numbers change.

## [1.6.4] - 2026-08-06

### Fixed

- The `<details>` opening tag was accidentally dropped from the "What's new"
  block in both READMEs during the v1.6.0 rewrite, which made the whole digest
  render expanded instead of collapsed. The tag is restored, so the block
  collapses by default again. No measured numbers change.

## [1.6.3] - 2026-08-06

### Changed

- Header layout simplified: the `Integrated by:` and `Author:` lines are removed
  from every algorithm header; the credit chain now reads original authors →
  implementation → current code only. No measured numbers change.

## [1.6.2] - 2026-08-06

### Changed

- Header layout refined: the explanatory "Credit chain († = ...)" legend line is
  removed, and the `Author:` line moves below the chain (after `Integrated by:`),
  so every header reads as title, chain, author, references — no prose.

## [1.6.1] - 2026-08-06

### Added

- Every algorithm source file (`aligners`, `augmenters`, `models`, `strategies`,
  `ensembles`) now carries a complete credit chain in its header: original authors
  (paper, venue, and official code when one exists), the implementation author and
  repository, the author of the current port, and the integrator, with GitHub links on
  every node except the integrator. Co-first authorships are marked with †:
  Channel Reflection (Ziwei Wang† & Siyang Li†) and StackingNet
  (Siyang Li† & Chenhao Liu†), verified against the journal pages.
- Ensemble chains were traced to their earliest release: EBCC, ZenCrowd, PM, LA, LAA
  and StackingNet follow Chenhao Liu's `Flashingcat/Golden_task-Ensemble` (2024–2025)
  into the lab's `sylyoung/TestEnsemble` and from there into this benchmark. PM is
  identified against its true source paper — "Resolving Conflicts in Heterogeneous
  Data by Truth Discovery and Source Reliability Estimation" (CRH, SIGMOD 2014) —
  rather than the earlier mis-attributed title.
- The overview now features the lab's `wzwvv/EEGAug` data-augmentation repository.

### Changed

- Header attribution only; **no measured numbers change.** The full re-injected set
  passed compile, header-verification, and header-only-diff checks.

## [1.6.0] - 2026-08-06

### Corrected

- The Network benchmark now has one declarative 18-method inventory and a literal target-isolated nested leave-one-subject-out selector. Selection excludes the outer target, validates on every remaining source subject, and final evaluation uses seeds 1–5.
- The complete five-seed campaign was run, validated, and imported: 18 methods × 3 datasets × all target subjects × 5 seeds, with complete inner-LOSO coverage. The importer independently recomputes metrics and is the only path that can replace Network values in the publication sources.
- Five rows whose identities were corrected — DeepConvNet, ShallowConvNet, ADFCNN, EEGWaveNet and FBMSNet — are explicit architecture transfers of the cited references (the Braindecode Deep4Net and ShallowFBCSPNet feature architectures, the released-code ADFCNN and EEGWaveNet topologies, and an 8–32 Hz adaptation of FBMSNet with exactly six causal views and no branch below 8 Hz). Their plug-ins are `Deep4NetAT.py`, `ShallowFBCSPNetAT.py`, `ADFCNNTransposeAT.py`, `EEGWaveNetReleaseAT.py`, and `FBMSNet8to32AT.py`.
- MVCNet is reported on the Network axis with its documented three-seed legacy values; the one-row Composite group was removed.
- Every final Network fold writes a matched JSON, prediction archive, and model checkpoint, reloads the checkpoint to verify its logits, and maintains an atomic resumable training state. Cache, source, software, method, seed, and numerical-family identities fail closed.
- The exact Python 3.11/CUDA package environment is frozen in `requirements-network-production.txt`; production refuses to run unless the lock digest is recorded and every installed distribution matches it.
- This release also carries the audit-phase corrections from the previously unreleased trunk: result/cache/ensemble artifacts record source, software, preprocessing and method parameters and fail closed on stale, partial or mismatched inputs; the ensemble combiners' parameters are serialized; and ZenCrowd is identified as the simplified TestEnsemble EM baseline and PM as the three-round PM/CRH port.

### Measurement status

- Legacy Network values from the superseded selection procedure were withdrawn and replaced by the corrected five-seed campaign values, which passed checkpoint, prediction, provenance, and coverage validation on 2026-08-06.
- Other tables retain their documented historical measurements and provenance; no four-class result is published.

## [1.5.0] - 2026-07-30

Repository structure only. **No measured number changes**, no algorithm changes, and the
leaderboard and the website render identically: the only edit to a generated file is the
`code:` path each leaderboard row links to.

### Changed
- **The library moved to `src/hustbciml/`.** The repository root now names what each
  directory is for — `src/` the code, `docs/` the site, `gallery/` the site's source.
  `hustbciml` remains the import name, so `python -m hustbciml.run …` is unchanged; what
  changes is that the package has to be installed to be importable, which is one added
  line in the documented setup:
  ```bash
  pip install -r requirements.txt   # dependencies
  pip install -e .                  # the package itself, from src/
  ```
- **Added `pyproject.toml`**, which makes that install possible, declares the core
  dependencies and four optional groups (`data`, `ensembles`, `augmenters`, `dev`), and
  absorbs the root `pytest.ini`. It reads the version from `hustbciml.__version__` rather
  than restating it.
- The four tests that located the repository by counting `..` segments now look for a
  marker file instead. That count was one level short after the move, and it is the kind
  of breakage that would recur at every layout change.
- On the website, the "Benchmark code" buttons and the `RESULTS.md` button follow the
  package to its new path — one constant in `app.js`. Every in-repo link the leaderboard
  renders is checked against the working tree by `tests/repro/test_leaderboard_links.py`,
  so a path that moved without its link would fail rather than 404 in a reader's browser.
- The `--data_dir` defaults of `tune_algorithm.py` and `tune_networks.py`, and the
  `compare` invocation documented in `RESULTS.md`, pointed at one contributor's home
  directory. They use `./data` and `./results` now, matching `Config.data_dir`.

### Removed
- **33 internal files**, none of which a reader of this repository could use.
  - `RERUN.md` and the seven scripts that executed it (`rerun_v12*.sh`, `rerun_fix.sh`,
    `extract_v12.py`, `compare_v12.py`, `apply_v12.py`). They planned and carried out the
    v1.2.0 re-measurement; that migration finished, its results are published in
    `benchmark.yml` and `RESULTS.md`, and nothing but those files referenced them.
  - Twenty-five launchers, sweeps and extractors (`sweep_*.sh`, `launch_*.sh`,
    `prod_sweep*.sh`, `server_launch.sh`, `fullrun_newmethods.sh`, `smoke_new_methods.sh`,
    `extract_augbb_3ds.py`, `extract_newmethods_3ds.py`). Each hard-coded a home
    directory, a personal conda environment or a personal log path, so none of them ran
    anywhere but on the machine that wrote them. The interface they wrapped —
    `python -m hustbciml.run --algorithm <name> --dataset <name>` — is what the README
    documents, and which machine produced each published cell is still recorded in
    `scripts/cell_origin.tsv`.

  Every script the repository's published claims depend on stayed: `build_cards.py`,
  `compare.py`, `leaderboard.py`, `ensemble.py`, `decentralized.py`, `combined_ensemble.py`,
  `tune_networks.py`, `tune_algorithm.py`, `sync_results_md.py`, `cell_origin.tsv`.

### Added
- `tests/test_packaging.py`. `pyproject.toml` restates two things written down elsewhere,
  and both now have a check: `__version__` must match the newest CHANGELOG heading — it had
  been left at `1.2.0` through four tagged releases — and where `pyproject.toml` and
  `requirements.txt` name the same package, the version range must be identical, so the
  measured `scikit-learn<1.8` bound cannot be relaxed in one file only.

### Fixed
- CI installs the package (`pip install -e .`) before running the tests and the card
  generator, which the `src/` layout requires. Its comments describe what each job checks
  rather than which past defect prompted it.

## [1.4.1] - 2026-07-30

### Changed
- **The Overview lists the ensemble combiners again.** v1.3.0 dropped the "Ensemble Learning"
  group of approach chips; it is back, so all 14 combiners appear under their own heading and
  the lab's **SML-OVR** and **StackingNet** are visible where the site lists the lab's methods.
  The approach-chip count rises from 60 to 74.
- The two rows of the ensemble table's *Non-ensemble references* sub-category, `single-source`
  and `Centralized Training`, are marked `reference: true` and stay out of that list, as every
  other table's references already do.
- **No measured number changes** and the leaderboard renders identically: rebuilding
  `docs/data/*.js` moves only those two flags and the count. The "ensemble combiners" statistic
  v1.3.0 also removed is deliberately not restored — the chips list them, and both READMEs
  state the figure.

## [1.4.0] - 2026-07-30

### Changed
- **The Ensemble Learning table is re-measured on a three-learner-per-source pool** — one
  learner per model family: tangent-space logistic regression, CSP-Net, EEGConformer — so each
  target collects (N−1)×3 hard votes. All 45 combiner and reference cells are new (3 datasets ×
  3 seeds, hust-gpu-7002); Centralized Training is unchanged. Members are individually stronger
  (single-source 61.32 / 57.94 / 58.45) and the combiners separate further from majority voting:
  **SML-OVR (lab)** and binary **SML** go from a mean +0.47 over the three datasets to **+1.61**,
  positive on all three for the first time, and **StackingNet (lab)** from −0.08 to **+0.31**.
  Dawid-Skene leads at +1.70.
- `--base hetero` is gone; `--base hetero3` is the only heterogeneous option, so an old script
  naming `hetero` fails with an invalid-choice error instead of measuring a different pool.

### Removed
- `scripts/rerun_v12_ensemble.sh`, which existed only to reproduce the five-learner table.

### Added
- The 45 ensemble cells in `scripts/cell_origin.tsv` — the one published family whose machine
  had never been written down.

### Known limitation
- The two neural members train at the `EA-EEGNet` preset's `lr = 1e-3` / 100 epochs, not the
  per-backbone values `tune_networks.py` selects (EEGConformer: `3e-4` on BNCI2014001, `1e-4`
  on the other two), so they are not the configuration published under those names in the
  Networks table. Recorded in `RESULTS.md` and the `_base_hetero3` docstring. It does not affect
  what the table compares: every combiner fuses identical votes.

## [1.3.2] - 2026-07-29

### Removed
- **Both four-class BNCI2014001 appendices are deleted from `RESULTS.md`.** The benchmark is
  two-class throughout, as its own header has always stated, but two supplementary appendices
  at the end of the file still reported native four-class results (left/right hand, feet,
  tongue; chance 25%): the privacy-preserving appendix and the heterogeneous
  decentralized-ensemble appendix, 22 table rows in total. They are gone, along with the two
  cross-references that pointed at them.

  Two four-class numbers had also leaked into the two-class narrative: the per-dataset
  discussion cited Dawid-Skene as `64.86 / 57.55 / 41.51` and EBCC as `64.53 / 58.38 / 40.34`,
  where the third entry of each triple was the four-class figure. Those triples are now pairs
  with their datasets named explicitly, and the sentence they supported has been restated
  against the two-class evidence that remains.

  The `BNCI2014001-4` code path is deliberately kept: it is what `SML-OVR`'s one-vs-rest form
  exists for, and removing it would be a functional change. **No two-class number changes.**
  The web leaderboard never carried four-class content — rebuilding `docs/data/*.js` after the
  edit produces a byte-identical result — and `repro_targets.yaml` has no four-class entries.

## [1.3.1] - 2026-07-28

### Fixed
- **StackingNet's sum-to-one regularizer is now written on the L1 norm, as in the authors'
  released code.** The combiner used `(1 - sum_j w_j)^2`; `StackingNet_classification` in
  [TestEnsemble](https://github.com/sylyoung/TestEnsemble) uses
  `(1 - l1_regularization(net, 1))^2`, and `l1_regularization` returns `torch.norm(weights, 1)`.
  For non-negative weights the two agree in value but not in gradient: `d||w||_1/dw_j` is
  `sgn(w_j)`, which is 0 at exactly zero, so a weight the non-negativity clamp drives to 0
  receives no restoring pull and stays dead, whereas the sum form revives it. The published
  objective and its clamp are together a sparsifier, and the benchmark now reproduces that
  rather than a more forgiving variant of it.

  **No measured number changes, and none were re-measured.** The two forms have identical
  gradients as long as every weight stays strictly positive, which is what happens at the
  shipped defaults (`lr = 1e-3`, `lambda_1 = 1e-3`, `lambda_2 = 100`, 200 epochs): no weight
  reaches the clamp. Checked on cached decentralized votes from the same five per-source
  learners the leaderboard uses — TangentLDA, TangentSVM, EEGNet, ShallowConvNet, CSP-Net over
  three datasets and three seeds, 105 target runs — the two forms give identical predictions on
  100.00% of trials, 71.81 balanced accuracy either way. `RESULTS.md`, the leaderboard and the
  web app are untouched.

  The forms diverge only once the hyperparameters are swept far enough for a weight to reach the
  clamp. At `lr = 0.1` every weight is clamped to zero and the faithful form returns chance
  (50.00) where the previous one returned 70.52 — the released method really does collapse
  there, and a benchmark that claims to implement it should show that.

### Added
- **Two sweeping notes in `StackingNet.py`**, since the constructor exposes the hyperparameters.
  Adam normalizes each coordinate's step by that coordinate's own gradient RMS, so only the
  ratio `lambda_1 / lambda_2` changes the trajectory rather than the two values separately; and
  once the unsupervised term dominates, every weight decreases by about `lr` per step regardless
  of its disagreement count, which makes `lr` and `epochs` one knob — their product — not two.

## [1.3.0] - 2026-07-27

### Added
- **The Ensemble Learning table is on the leaderboard again**, reversing the withdrawal of 1.2.5.
  The 14 combiners and the reference rows they are read against — majority voting, the
  single-source five-learner mean, and Centralized Training — carry the numbers they carried in
  1.2.4. Nothing was re-measured, and nothing in `hustbciml/` changed. The two findings 1.2.5
  recorded remain true of these numbers and are stated here so that withdrawing the table is not
  the only place they are written down: the five per-source learners are built by deep-copying
  the `EA-EEGNet` preset and swapping `backbone` alone, so ShallowConvNet and CSP-Net train at
  `lr = 1e-3` under a 100-epoch ceiling rather than at the per-backbone values
  `scripts/tune_networks.py` selects under a 300-epoch one; and M-MSR, GLAD and ZenCrowd fall
  five to ten points below majority voting by emitting a single class on most target subjects
  (8 of the 12 on BNCI2015001, seed 1), which an accuracy column does not distinguish from an
  ordinary low score.

### Removed
- **The ensemble block on the Overview.** Two things go: the "ensemble combiners" statistic
  beside the approach counts, and the "Ensemble Learning" group of approach chips below them.
  The Overview now describes the decoding pipeline and nothing else. The combiners are post-hoc,
  in that they fuse the predictions of models the other tables train, so listing them as a sixth
  stage of that pipeline overstated what they are, and a second headline count beside the first
  asked a reader of the landing page to hold two populations at once. The Benchmark tab is
  untouched: the table, its blurb and the reading guide that names it are all as they were.
- `n_approaches`, the count stated in the heading directly above those chips, falls from 75 to
  60 to match the chips it introduces. `approach_names()` in `build_site.py` and
  `benchApproaches()` in `app.js` now skip the same table, and each says so, because a count and
  a list computed over different populations is the defect that made the page disagree with
  itself by 18 once already.

### Unchanged
- The leaderboard, the Benchmark tab, both READMEs apart from the digest bullet, and every test:
  identical to 1.2.4. `n_ensemble_methods` is still emitted by the build, because both READMEs
  state the combiner count and `tests/repro/test_readme_counts.py` checks it against the build.
  **58** approaches and **20** lab approaches are unchanged.

## [1.2.5] - 2026-07-27

### Removed
- **The Ensemble Learning table, withdrawn from the leaderboard** (web app, both READMEs). An
  audit of it found two problems, neither of them in the combiner implementations. First, the
  configuration is not the one the Networks table reports. `_base_hetero` builds the five
  per-source learners by deep-copying the `EA-EEGNet` preset and swapping `backbone` alone, so
  ShallowConvNet and CSP-Net train at `lr = 1e-3` under a 100-epoch ceiling, whereas
  `scripts/tune_networks.py` grid-searches the learning rate per backbone under a 300-epoch
  ceiling and selects `1e-4` for ShallowConvNet on BNCI2015001, `3e-3` for it on BNCI2014001,
  and `3e-4` for CSP-Net on BNCI2014001. Those two learners are therefore not the configuration
  published for the same names one table above. (EEGNet's fixed `1e-3` does coincide with its
  tuned selection on the datasets whose tuning record is unambiguous, so the mismatch is confined
  to the other two backbones.) Second, three rows are degenerate rather than
  weak: M-MSR, GLAD and ZenCrowd sit five to ten points below majority voting because they emit
  a single class on most target subjects, and on BNCI2015001 seed 1 each collapses on 8 of the
  12 targets, while every combiner that beats voting collapses on none. An accuracy column
  cannot distinguish that from an ordinary low score. Republishing requires a per-backbone
  configuration and a class-balance diagnostic; both are absent, so the table is withdrawn
  rather than annotated.
- `n_ensemble_methods` from the generated site data, the "ensemble combiners" statistic from the
  Overview, and the ensemble-combiner count from both READMEs and the anchor card.

### Changed
- **`hustbciml/RESULTS.md` keeps the measurements** and states the withdrawal above the table.
  Its rows are declared in `NOT_ON_LEADERBOARD` in `tests/repro/test_results_md.py`, so their
  numbers are now checked by nothing, which the declaration records explicitly. 199 cells remain
  cross-checked against `benchmark.yml`.
- `test_ensemble_rows_carry_their_own_provenance` asserts the table's absence instead of looping
  over no rows, so it cannot pass vacuously, and fails if the table returns without provenance.

### Unchanged
- The 14 combiner implementations, `scripts/decentralized.py`, the ensemble presets, and the
  multi-seed ensemble experiment in `RESULTS.md`, which is a separate experiment over K seeds of
  one algorithm and is not affected. No other leaderboard table, and no reported number outside
  the withdrawn table, changes: **58** approaches as before.

## [1.2.4] - 2026-07-27

### Fixed
- **Three claims in `README.zh-CN.md` that the English text does not make.** The Chinese
  "Reproducibility" principle stated that runs persist model checkpoints (nothing calls
  `torch.save`; a run writes `metrics.json` and `predictions.npz` only) and that hyperparameter
  selection happens **only on held-out source subjects, never touching target or test labels**.
  That second claim is contradicted by the file's own "Hyperparameter selection" section further
  down, which describes dev-subject selection scoring three subjects against their true labels
  and then counting them in the reported average. It is the same measurement-integrity
  regression `check_i18n.py` was written to prevent recurring: retracted in English, left
  standing in Chinese. The heading-skeleton check cannot catch it, because the divergence is
  inside a section both files have. The Chinese "Reproduction" section also claimed a
  license/provenance *audit* of the ported code; the cards record what an upstream repository
  states and say so when it states nothing, which the English is careful about and the Chinese
  was not.
- **The paper-with-code count, in both READMEs.** Stated as 76; `build_site.py` counts 72
  non-empty `code_url` entries, which is what the web app has been rendering.
- **A results file that no code writes, in both READMEs.** Quickstart promised a resolved
  `config.yaml` per run. There is none: the resolved configuration is a field inside
  `metrics.json`, which is what makes a leaderboard cell auditable from one artifact.
- **Two sections missing from the Chinese "Design principles".** The "also varies" caveat for
  rows that change more than one stage (MVCNet, PAT, MEKT, LSFT, MSDT) and the two axes shared
  across a table (best-checkpoint baseline vs. last-iterate adaptation rows, per-architecture
  learning rates) existed in English only.

### Changed
- **Both READMEs rewritten in the registers set in 1.2.2** — English in that of the lab's own
  papers, Chinese in that of its own release writing. English: the remaining em-dashes and
  British spellings (`unlabelled`, `pseudo-labelling`, `crowd-labelling`, `optimisation`,
  `expected-behaviour`), "compares them head-to-head", "code first", and "before the backbone
  sees them" — the same essay-voice phrasing 1.2.2 removed from the web app. Chinese: the
  gratuitous English glosses on terms that need none, among them 生成器（Generator）,
  卡片（cards）, 免责声明（Disclaimer）, 冒烟测试（smoke test）, 预设（preset）,
  抽象基类（abstract base class）and 移植指南（porting guide）; and the vendor voice "我们用一个
  小网格搜索了…" where the English is impersonal. The gloss-once forms that carry information
  (脑电（EEG）, 经验风险最小化（ERM）) were kept. No headings were added or removed, so the
  skeleton check still passes.

## [1.2.3] - 2026-07-27

### Changed
- **The twelve row descriptions 1.2.2 missed.** The rewrite covered the alignment, augmentation,
  network and classical tables; the transfer and ensemble rows kept British spelling
  (unlabelled, labelled, pseudo-labelling), one em-dash, and four sentences that told the reader
  what to think of a number rather than stating it: "so privacy is nearly free here", "lifts the
  other two datasets clearly above centralized training", "The floor", "the vanilla federated
  baseline". Same register as 1.2.2. No number changes; these strings render in English only and
  have no i18n keys.

## [1.2.2] - 2026-07-27

### Changed
- **The web app's explanatory prose is rewritten in both languages.** Every stage blurb, dataset
  role, leaderboard reading guide, disclaimer and Overview paragraph, plus the per-row method
  descriptions. The English now follows the register of the lab's own papers: declarative, no
  em-dashes, terms introduced once with the acronym in parentheses, US spelling, `i.e.,` where a
  list is being named. What went, went for a reason — "before the backbone sees anything", "marks
  the floor", "the whole problem is estimating how far to trust each learner", "two further
  caveats worth stating" — an essay voice reads as unserious next to the numbers it is
  explaining. The Chinese now follows the lab's own 公众号 writing: function first, then
  mechanism, then what it is for, in noun-phrase compounds (变更/归因/参照/配置) rather than
  spoken paraphrase, still under the existing ban on ——, ；and 引号. An English term is glossed
  once as 中文（English, ABBR）and then kept short, so 欧氏对齐（Euclidean Alignment, EA）is
  expanded in the reading guide and appears bare in every blurb below it.
- **The anchor card counts what the page counts.** It claimed 56 approaches; the Overview beside
  it has read 58 pipeline approaches and 14 ensemble combiners since 1.2.0, when the two
  network-free classical rows were given their own table. It now states both numbers.

### Fixed
- **`check_i18n.py` was not checking the library card or the Overview prose.** It looked for
  `library` under `window.LAB`, but app.js reads it from `window.BENCH`, so the lookup returned
  nothing and all three fields were dropped silently — the headline of the Benchmark tab and its
  one-paragraph description of the pipeline had no coverage at all. The lab tagline, the repo
  intro, the anchor blurb and the flagship pillar labels were never listed either, though each is
  a `tr()` call exactly like a table blurb. Generated strings checked go from 39 to 64. Two dead
  half-sentence keys for the datasets intro, superseded by the whole-sentence template, are
  removed: nothing called them, and a stale entry beside a live one attracts the wrong edit.

## [1.2.1] - 2026-07-27

### Fixed
- **`scikit-learn` is bounded to `<1.8`, because one published leaderboard row does not run above
  it.** From scikit-learn 1.8, `check_is_fitted` consults `get_tags()`, which requires
  `__sklearn_tags__` in the estimator's MRO. crowd-kit's `Wawa` does not inherit from
  `BaseEstimator`, so the `check_is_fitted(self)` inside its own `_apply` raises `AttributeError`
  before aggregating anything — the Wawa row of the Ensemble Learning table cannot be reproduced at
  all. Found by the CI added in 1.2.0 on its first run: the Python 3.10 job resolved scikit-learn
  1.7.2 and passed, the 3.12 job resolved 1.9.0 and failed, which is the version split the new
  matrix exists to expose. The boundary was then measured rather than inferred — crowd-kit 1.4.2
  held fixed, 1.7.2 aggregates and 1.8.0 raises — and the other four crowd-kit combiners
  (Dawid-Skene, GLAD, MACE, M-MSR) are unaffected. crowd-kit 1.4.2 is the current release and
  declares an unbounded `scikit-learn`, so there is no newer version to require instead. The bound
  sits on the shared dependency because pip cannot express "only when the ensemble table is run";
  it is a measured incompatibility ceiling, not a hand-written pin, and `requirements.txt` says so
  and says when to remove it.

## [1.2.0] - 2026-07-27

An external code review (GPT-5.5, ten scoped passes over the library and the web app, then a
verification pass against the reference implementations on disk) produced 176 findings. This
release acts on them. Nothing here changes what a method *is*, but several fixes change what it
*computes*; those are listed separately below, and the leaderboard cells they touch are listed in
`RERUN.md`.

### Fixed — behaviour-changing (affected leaderboard cells must be re-measured; see RERUN.md)
- **Channel Reflection now fails closed.** It required only "two classes" and any channel names
  ending in a digit. On BNCI2014002, whose montage is exposed as `EEG1 … EEG15`, the 10-20
  odd/even hemisphere rule produced an arbitrary sensor permutation *and* swapped the label; on
  right-hand-vs-feet data (BNCI2014002, BNCI2015001) the label swap is invalid whatever the
  montage. It now requires a real electrode montage with every lateral channel paired, and a
  left/right class pair, and raises otherwise. Same rule applied to MVCNet's reflected view,
  which is dropped on the datasets where it does not apply (its contrastive losses are written
  over however many views survive).
- **Fourier Surrogate** rotated the spectrum by a shared random phase instead of rebuilding it
  from magnitude alone, which had forced every channel to identical phase at each frequency
  (perfect zero-lag coherence across the montage). DC and Nyquist bins are copied verbatim, so a
  negative channel mean no longer comes back positive.
- **CSDA** passed `mode` to the forward DWT (decomposition and reconstruction had used different
  boundary extensions, leaving an edge artifact on every augmented trial), and now pairs each
  trial with a same-class partner from a *different* subject — the "cross-subject" in its name.
- **RA** derives its ridge from each covariance's own trace instead of the first trial's, and
  raises on a non-negligible imaginary residual instead of silently taking the real part.
- **SML** repairs the leading eigenvector's *global* sign instead of taking an element-wise
  absolute value, which had turned anti-correlated (below-chance) base learners into
  positive-weight voters. **LAA** feeds logits to `CrossEntropyLoss` instead of probabilities
  (the warm-up had been softmaxed twice). **PM** normalises by the current round's maximum
  disagreement rather than a running one, and breaks ties uniformly instead of always choosing
  class 0.
- **CTNet** sizes its temporal kernel from `sfreq` (a fixed 64 samples is a quarter-second at
  250 Hz but an eighth at 512 Hz, so the "same" architecture spanned different durations across
  datasets), sizes its positional table from the actual token count instead of a fixed 100, and
  keeps the paper's pre-classifier `Dropout(0.5)`.
- **Backbone shape probes** run in eval mode with BatchNorm buffers and the RNG restored. A bare
  `torch.no_grad()` does not switch modules to eval, so 13 of 19 backbones had been perturbing
  their own BatchNorm statistics and advancing the RNG stream during construction — while EEGNet,
  which nearly every non-network row uses, did not. Only the Networks table is affected.
- **DELTA** advances its class-balance EMA once per online batch rather than once per adaptation
  step, so `--steps` no longer also changes the DOT memory schedule. *No published cell moves:* the
  batch-boundary condition gating the update is unchanged, and at the preset's `steps: 1` the inner
  loop runs once, so both versions execute the identical sequence of updates. The fix changes only
  `--steps > 1`, which no published cell uses. Listed here rather than below because the behaviour
  genuinely changes — just not at any configuration the leaderboard reports.

### Fixed — no effect on published numbers
- **Run identity.** `setting()` now includes the augmenter (two hand-composed augmentation runs
  had shared one results folder) and appends any stage flags that override a named preset (so
  `--algorithm EA-EEGNet --backbone ShallowConvNet` can no longer overwrite the genuine
  `EA-EEGNet` result and be labelled as it). `metrics.json` records the **full resolved config**,
  and `save_results` refuses to overwrite a result produced by a different configuration.
  `predictions.npz` stores `y_pred` alongside `y_score`.
- **Guards that were placeholders.** `build_pipeline` now rejects an unknown strategy mode, a
  label-requiring aligner under a held-out-target protocol, a non-online aligner under a
  test-time strategy, and `mode='tta'` with `uses_target`. The online TTA loop dispatches on the
  composed aligner instead of running EA for anything not named `Identity`. The target handed to
  the aligner has its labels masked.
- **Silent fallbacks removed** in measurement paths: unknown `--hp` keys, a non-zero
  `--calib_ratio` (whose protocol is unimplemented), zero-optimizer-step training, ASFA
  adaptation that never ran, a failed CrowdKit aggregation, a failed ensemble combiner,
  an unreadable `metrics.json` during aggregation, a learning-rate grid with no valid point, a
  FBMSNet filter-bank design failure, and a `RiemannMDM` `predict_proba` error. `metrics.json` is
  written with `allow_nan=False`, so it is valid JSON for non-Python readers.
- **Unequal source/target batches** no longer silently mis-split domains in MDD and CDAN, and
  MK-MMD says what is wrong instead of failing on a broadcast shape.
- `--weight_decay` is honoured by the transductive DA loop and by DANN (it was dropped, so a
  regularisation sweep over eleven methods returned identical numbers). `--verbose` prints the
  per-epoch progress that was being formatted and discarded. `--protocol` advertises only
  implemented protocols. Strategy coefficients (`dan_align`, `jan_align`, `mcc_temp`,
  `mdd_margin`, `mdd_trade_off`, `cdan_max_iter`, `mvc_lamda1/2`, `mvc_temp`, `mvc_f_shift`)
  are readable from `--hp` at their existing defaults.
- Three leaderboard keys with no preset (`NoAlign-EEGNet`, `EA-ShallowConvNet`,
  `EA-DeepConvNet`) are now runnable via `--algorithm`.
- **Re-measurement sweep scripts: three ways to measure nothing and call it done.** These run
  unattended for hours, so a silent no-op is worse than a crash. (i) The claim reaper released a
  claim whenever the worker loop that took it was gone — but killing a worker leaves its
  `python -m hustbciml.run` child running, reparented to init, still computing and still writing
  its own `metrics.json`, so a second GPU picked the cell up and both wrote it. It now checks for
  the running job, not just the loop. (ii) `rerun_v12_nettune.sh` resolved its origin map through
  `dirname "$0"` *after* `cd`-ing to the work directory, so a relative launch made every cell look
  out of scope and the sweep reported a clean finish having measured nothing; the script now
  resolves its own path first, and an unreadable map is a hard error rather than an empty job
  list. (iii) A missing interpreter was discovered once per dataset instead of once, after the job
  list had already been consumed. `rerun_v12_nettune.sh` also gained the `JOB_ORIGIN` filter the
  main sweep already had, so it can no longer measure a cell on a machine that did not produce it.
  (iv) A fourth, found by walking into it: the main sweep's origin filter matched `JOB_ORIGIN`
  against the map's machine column **literally**, while two of the servers are bit-identical and
  therefore interchangeable. Passing a worker its own hostname rather than its family's selected
  nothing, and the worker printed `0 jobs in scope` and exited 0 — indistinguishable from "the sweep
  is finished". All three sweep scripts now fold that pair onto one family name on both sides of the
  comparison, so the filter no longer depends on the caller and the map having independently agreed
  which of the two names to write.
- `compare_v12.py` walked the report's reserved `_nettune`/`_ensemble` keys as if they were method
  cells and died on the first one, which meant the ensemble table's comparison never printed. Those
  keys are now skipped, and the ensemble table gets its own section.
- **`compare_v12.py` compared against whichever machine's baseline it found first**, which is the
  one mistake this sweep is built to avoid: the difference it then reports is a BLAS change, not a
  code change. It called a passing control broken — `EA-EEGNet`/BNCI2014002 re-run on 20022, scored
  against 7002's baseline, "DIFFERS in 12/14 subjects" — and would just as readily have called a
  real regression fine. Baselines are now restricted to the re-measuring machine's numerical family
  (10022 and 20022 are one family), and a cell with no baseline in that family reports *not
  checkable here* rather than counting as a failure. It also excludes cells whose recorded origin is
  another machine: a results directory accumulates earlier ad-hoc runs — the cross-machine
  cross-checks were deliberately written into this one — and the extractor reports whatever is in
  the tree, so four such cells were being judged as if they belonged to the sweep.
- The generated site data carried a `generated` build date that nothing displayed and nothing read,
  and that the new CI job regenerates and compares against the committed file. A stamp that moves
  with the clock cannot survive that comparison: the build would have gone red on the date alone
  every day after a release, reporting the generated files as stale when nothing had changed —
  the exact signal the job exists to give, spent on noise. `SOURCE_DATE_EPOCH` support only narrows
  the window, since whatever the developer stamps before committing, CI recomputes after. The stamp
  is removed; if a "data as of" line is ever wanted, it should come from the commit at display time.
- The dataset table rendered two of its cells in English on the Chinese page: `2-class` under
  类别数 and `288 / session` under 试次/被试. The headers were translated and the values were not,
  which is the failure mode the i18n check exists to catch — but the check mirrors the `tr()` calls
  in `app.js`, and these two fields never went through `tr()`, so both the app and the checker
  agreed there was nothing to translate. `classes` is now the plain value `2` (the column header
  already says what it counts, in both languages), `trials` goes through `tr()`, and
  `check_i18n.py` covers it — for values carrying a word, since demanding a translation of `100`
  would be noise. Found by reading the rendered page, not by any test.
- `hustbciml.__version__` still read `0.0.1.dev0` four releases in. It is now set to the release
  being tagged. Nothing imports it — there is no packaging metadata for it to feed — so this changes
  no behaviour, but a version string that disagrees with the tag is worse than none at all.

### Documented — deviations that are kept deliberately
Four defects are inherited verbatim from the reference implementations this repository ports, and
were verified line-by-line against them. Correcting any of them moves the number away from the
published baseline it is meant to be comparable with, so each is kept and recorded in the source
and on its card: **MCC**'s missing `keepdim` in the confusion normalisation, **MDD**'s
initialisation of the BatchNorm rather than the Linear bottleneck (traced to a commented-out pool
layer upstream), **ADFCNN**'s `reshape` where a transpose is meant, and **MSCFormer**'s
non-learnable class token. Also recorded: EEGNet ships without the paper's max-norm constraints
(as its reference does, and it underpins nearly every row); MSVTNet is a backbone-only ablation,
not the full method; LSFT fits a tangent space per domain; MSDT's consistency loss is computed on
logits; BN-adapt is an EMA variant rather than recompute-from-batch.

A fifth is recorded because this release *tried* to change it and was wrong to. **MDMAML reverts
only the parameters between inner-loop task pairs, never the BatchNorm buffers**, and that is
deliberate. Restoring the buffers looks like the tidier choice — a pair would then not inherit the
previous pair's running statistics, and the meta-gradient would not depend on the order the pairs
were sampled in. The reasoning fails twice. In train mode BatchNorm normalises with the *batch*
statistics, so the running buffers never enter the forward pass and cannot influence any gradient,
which means there was no order dependence to remove. And because every pair restored them, the
buffers never advanced past their initialisation for the whole run, while `predict` scores in eval
mode against exactly those buffers (§IV-F) — the model was evaluated under a normalisation it had
never been trained with. Measured cost on BNCI2014001: **−12.58 and −20.22 accuracy points** on
seeds 1 and 2. `test_training_estimates_the_batchnorm_running_statistics` now fails if the buffers
stop moving during training.

### Fixed — claims that did not match the code
- "Source-only (no target at all)" → source-only *training*, with unlabelled target alignment:
  every one of those rows composes EA, which estimates the held-out subject's reference from that
  subject's own trials.
- The one-axis guarantee now names its exceptions, and rows that vary more than their table's axis
  (MVCNet, PAT, MEKT, LSFT, MSDT) carry an explicit "also varies" note on the leaderboard. Empty
  dataset cells carry an `na_reason`, so "n/a" no longer conflates "inapplicable" with "not run".
- **`README.zh-CN.md` still stated an integrity claim the English README had retracted.** It said
  that wherever hyperparameters were selected, selection ran on held-out source subjects alone, and
  that the selection process never touched the reported cohort. The English file had already
  replaced that with the honest version: two selection signals, and for eight methods (ASFA, Tent,
  BFT, DJP-MMD, MDMAML, MSDT, LSFT, MVCNet) the adaptation-phase knobs were chosen on three
  pseudo-target subjects scored against their true labels, subjects who are also part of the
  reported average. The Chinese section has been rewritten to match. Nothing failed while the two
  disagreed, which is the point: a reader of the Chinese page was given a stronger guarantee than
  the benchmark can make, in the language of the laboratory's own institution. `check_i18n.py` now
  compares the two READMEs' heading skeletons — levels, not text — and this divergence has exactly
  the shape it detects, one language gaining a subsection the other lacks.
- The federated module no longer claims "the server never sees raw EEG": it is a simulated
  in-process protocol, and what the privacy table measures is the accuracy cost of the federated
  *schedule*.
- Card prose corrected where it contradicted the code: SHOT (information maximization only), PL
  (no confidence threshold), BN-adapt (EMA), BFT (the learned ranker, and tau = 0.25 not 0.5).
- The **EA-EEGConformer** card asserted 70.14 ± 4.45 and built a "transformers are data-hungry"
  narrative on it, while the leaderboard showed 74.05 ± 0.58 — the two artefacts disagreed on the
  *sign* of the delta. Ten methods' card values were stale; all are re-synced from the production
  run, and a test now fails the build if the two sources diverge again.
- The Overview's "22 lab approaches / 56 benchmarked" were counted over different populations
  (the first included the ensemble table, the second did not). Both now come from one definition
  in `build_site.py`: **20** lab, **56** pipeline approaches, **14** ensemble combiners.
- `requirements.txt` was missing `crowd-kit` and `pandas`, so five of the fourteen aggregators
  failed to import after a documented install.

### Added
- `hustbciml/tests/test_plugins.py` — 113 fast tests covering every backbone at all three dataset
  shapes, every ensemble combiner, every augmenter, and each new guard.
- `hustbciml/tests/repro/test_results_md.py` — holds `RESULTS.md` to `benchmark.yml`. The web app
  is generated from the leaderboard and the cards are generated from the registry, each with a
  test behind it, but `RESULTS.md` carried the same 80-odd accuracies by hand with nothing
  checking it — the arrangement that let ten card values go stale, and worse here because
  `RESULTS.md` reads as the authoritative record. It compares **all 244** cells; names that are
  ambiguous across tables (`none` is the unaligned row in one and the un-augmented baseline in
  another) are pinned to the table they mean, and a floor on the number of comparisons stops a
  broken parser from passing by checking nothing. Two rounds of tightening, each of which found the
  check was covering less than it looked:
  - The tables are not written the same way, and the first version required `mean ± std` — which
    silently skipped every cell of the ensemble table, written as `mean (Δ vs voting)`, i.e. the
    largest block of hand-written numbers in the file. The std and the delta annotation are both
    optional now, and a cell that states no std is compared on its mean rather than ignored.
  - Matching by display name then still dropped 42 of 244 cells, and they were the worst possible
    42: `RESULTS.md` writes `_majority voting (baseline)_` where the leaderboard says
    `Majority voting`, `none (no alignment)` for `none`, `_EA-EEGNet (deep reference)_` for
    `EA (Euclidean)` — so the unmatched set was very nearly the set of baseline and reference rows,
    the values every Δ in the file is measured against, two of which move in this release. An
    unmatched row is now a failure unless it is declared, not a line of output nobody reads.
- `hustbciml/scripts/sync_results_md.py` — corrects `RESULTS.md`'s tables from `benchmark.yml`.
  The test above closed the detection half of the gap; this closes the correction half. This
  release moves 122 table cells, and retyping 122 numbers to satisfy a test is how the file went
  stale in the first place. It imports the test's own cell pattern and display-name mapping instead
  of reimplementing them, because a fixer with its own idea of which leaderboard row a table row
  refers to could "correct" a cell to a number from a different row and the test, agreeing with
  itself, would pass it. Three things it refuses to do rather than do badly, each reported instead:
  cells carrying a delta (`76.16 (+1.85)`), whose annotation is measured against a reference row
  moving in the same edit; row order, since several tables are sorted by their first dataset column
  and a re-sort has to carry the prose that reads a ranking off it; and bold, which marks a result
  worth the reader's eye rather than a computed maximum — the Networks table bolds one cell and
  leaves two other column leaders plain, so there is no rule to apply. It reports an order or a bold
  that its own edit broke, not every column that merely is not descending. A test asserts the fixer
  is a **byte-for-byte no-op** on a file that already agrees with the leaderboard: the guard only
  certifies the cells the fixer chose to touch, so a bug that reformatted an untouched cell or
  dropped a std would leave the numbers right and the file wrong with nothing to catch it.
- `hustbciml/tests/repro/test_leaderboard_links.py` — every leaderboard row carries a `code:` path
  that the web app turns into a `blob/main/…` URL, and **nothing checked those 74 paths**. The link
  checker reads the same file but follows only its DOIs, and it could not have helped anyway: the
  URL is built from a path in the repository, so a file renamed in the commit that publishes it is
  broken before anything could fetch it. The failure is silent and public — the page renders, the
  row keeps its number, and the reader who clicks through to see how a method is implemented gets a
  404, on the one link that exists to make the benchmark checkable. All 74 resolve. In passing,
  `check_links.py` now also walks each group's `reference` row, which it had been skipping; no
  reference row currently carries a DOI, so this closes a latent hole rather than a live one.
- `hustbciml/tests/repro/test_repro_targets.py` — makes the reproduction registry executable:
  a fast consistency tier (runnable preset, value inside its own range, agreement with the public
  leaderboard, no key without provenance) and an opt-in `-m repro` tier that runs each row.
- `.github/workflows/ci.yml` — the repository had no CI. Runs both test tiers on 3.10/3.12 and
  fails if the generated site data or cards are stale.
- `gallery/check_links.py` — resolves every paper and code link. Its first run found three dead
  links and two stale renames, all now repaired; all 66 code links and all 288 DOIs resolve. A
  `doi.org` link that redirects has *resolved*, so it counts as ok rather than as a warning: the
  redirect is the identifier doing its job, it fires for every registered DOI, and the advice a
  redirect carries — store the target instead — would trade the one permanent URL for the
  publisher URL it exists to insulate the gallery against. A DOI that has really rotted does not
  redirect, it 404s at `doi.org`. So the redirect list means only one thing, which is the thing
  worth reading: a code or paper URL moved. Today none have. A publication may
  carry `in_press: true`, for a DOI the publisher has assigned but not yet registered: the
  checker then resolves the **parent** DOI instead (an ACM article DOI is
  `<proceedings>.<article>`, and the proceedings DOI is registered when the volume is), so the
  entry is still verified against something real rather than exempted. The web app shows such a
  paper as "in press" and does not link its title, instead of sending a reader to a 404. One
  entry uses this today: the KDD 2026 dataset paper, whose proceedings DOI resolves while the
  conference is still ahead of us.
- `gallery/check_i18n.py` — checks the Chinese layer of the web app: every `tr()` string in
  `app.js` must have a key in `i18n.js`, and no translated value may carry an em-dash, a
  full-width semicolon or curly quotes. Editing an English string without updating its key makes
  `tr()` fall back to English, which renders exactly like a translation and is visible only to
  someone reading the Chinese page. Its first run caught the leaderboard's entire "how to read"
  guide and the Overview's approach blurb shipping that way. It now also covers the strings that are
  *not* in `app.js`: the table titles, their explanatory blurbs, the sub-category headers and the
  dataset roles all come from `benchmark.yml` through the build and are translated by the same
  `tr()`, so checking only the literal calls left the longest prose on the site unchecked — a new
  leaderboard table would have rendered its title and whole blurb in English on the Chinese page
  with nothing failing. That coverage then stopped at `benchmark.js` while `app.js` also runs the
  lab card's title, tagline and driver line, and every publication's research area, through the same
  `tr()` — so a new research area on the papers page would have rendered in English on the Chinese
  page, again with nothing failing. All three data files are covered now: **39 generated strings**
  resolve, up from 29. Its two prose conventions — no em-dashes, curly quotes or full-width
  semicolons, and no 伍老师 or the usual calques — now also apply to `README.zh-CN.md`, across 313
  lines of prose. Those are rules about writing Chinese, not about the web app, and the Chinese
  README is both the repo's other Chinese artifact and the first one most readers meet, yet it was
  held to them by nothing: the rules existed only inside a check that reads `i18n.js`. Fenced and
  inline code are skipped, so a semicolon in a shell command is not reported as a Chinese one.
  Finally, it compares the two READMEs' heading skeletons — levels, not text — which is what caught
  the retracted-guarantee paragraph above. Where a level mismatch pins the divergence it says so;
  where one file is merely shorter it says that instead, because dropping a `##` leaves every later
  level unchanged and pointing at the first difference would send the reader to the wrong section.
- `--val_split subject` holds out whole source subjects for early stopping (the default trial
  split measures within-subject generalisation); `--fold_seed` makes each LOSO fold independently
  reproducible. Both are opt-in because they change the published numbers.
- Provenance entries and generated cards for the 20 leaderboard keys that had neither (all ten
  external backbones, six augmentation baselines, the federated methods and SAFE); every ensemble
  combiner now carries a key. Stable ids for the 33 publications that had `id: null`.
- `hustbciml/scripts/apply_v12.py` — writes a finished re-measurement into `benchmark.yml` and
  `repro_targets.yaml` from the sweep's own report. The two files hold the same numbers twice by
  hand, and this release moves about a hundred cells across four machines; transcribing that by
  hand is the process that produced the ten stale card values. It edits only the matched
  `mean:`/`std:` lines so the files' comments survive, refuses any cell missing a seed (a
  single-seed std is 0.00 by definition, which reads as precision), refuses a cell that two
  machines both measured, and distinguishes a cell the leaderboard already agrees with from one
  that matched no row at all — the second being a measurement silently going nowhere. Dry run by
  default. It also reports which lines of `RESULTS.md` **prose** quote a value the run just replaced.
  The tables there are held to the leaderboard by a test, but the prose quotes the same numbers as
  claims — "MSCFormer (76.29) and MSVTNet (75.95) top the table", "climbs from 69.65 there to 74.79"
  — and no check can reach a claim about a ranking. Matching on the replaced value, rather than
  asking whether some number could be explained by some cell, is what keeps it precise: in a
  20-point range the second question matches almost anything.
  It also checks each cell against `cell_origin.tsv` and drops any whose published value came from
  a different machine family. The report is only an aggregate of a results directory, and a results
  directory accumulates strays: 20022's holds four cells left by earlier cross-machine checks, whose
  published values came from 60022 and 7002. Two guards already stood between those and the
  leaderboard, but only by accident — the strays were short a seed, so the partial-cell rule caught
  them. Had one finished, the same cell would have been measured on two machines and the run would
  have refused outright; had only the wrong machine finished it, its number would have been published
  with a BLAS change folded into the delta, which is the one error this release's provenance work
  exists to prevent. Verified by perturbation: marking a stray complete makes the filter exclude it
  by name and publish the right machine's value instead.
- The provenance map is now **162 cells, every one `exact`**, up from 145 with three `inferred`. The
  three that were `inferred` — `CSDA-EEGNet` on BNCI2014001, `MDMAML` and `MVCNet` on BNCI2015001 —
  were documented as published numbers no surviving run reproduces anywhere, a broken provenance
  chain. They are reproduced exactly, mean and std, by `tune_algorithm.py` verdict files on 10022.
  The resolver matched published values against ordinary run trees and read verdict files only under
  one directory, for the four learning-rate-tuned backbones; every other row produced by a tuning run
  was invisible to it. Reading every `tuned_<dataset>.json` on every reachable box resolved all three
  plus 14 cells that had no entry at all.
  This was not bookkeeping. `inferred` had placed two of those cells on 7002 and one on 60022, so all
  three were re-measured on a **different BLAS family than produced them** — the exact error the
  provenance work exists to prevent, reached through the one column meant to flag uncertainty. One of
  them, MVCNet on BNCI2015001, produced the most quotable result in the release (a reversal from
  below the deep reference to above it) on a machine that never produced its baseline. All three are
  re-measured on 10022's family, and the value from the wrong machine is discarded by the same filter
  described above rather than by anyone remembering. The re-selection on the right family returns
  **74.75 ± 0.10** against the 73.19 reference, so the reversal survives being measured properly —
  which is the point: it is now the same claim, but from the machine that produced the baseline it
  is compared against.
- **What a verdict file records also changes how its cell must be re-measured**, and this cost a
  second full pass to find. `rerun_v12.sh` runs `--algorithm <preset>` with the preset's defaults,
  but a cell published from a verdict file is published at the configuration a *search* selected.
  For those three cells the two are different runs, so the delta between them mixes a code change
  with a configuration change and none of it is attributable to the release. The tell was MDMAML on
  BNCI2015001 returning 72.19 against a published 73.06 on 7002, being diagnosed as a cross-machine
  artefact, and returning **72.19 again on the right machine** — two BLAS families agreeing to two
  decimals, which no machine effect explains. Re-running the *selection* returns 73.06 ± 0.23, the
  published value exactly. So all three are re-measured with `tune_algorithm.py`, and `apply_v12.py`
  now refuses a plain preset run for any cell whose provenance is a verdict file rather than
  publishing its delta. The distinction is confined to those three: the other 43 affected cells are
  backed by ordinary run trees, where running the preset's defaults is the right comparison.
  Re-selecting rather than pinning the recorded hyperparameters is what the published value claims —
  not "this method at these settings" but "this method at the settings the tuner picks" — and it is
  not cosmetic: the CSDA-EEGNet search moved from `lr 0.001, batch_size 8` to `lr 0.003,
  batch_size 32`, as six of the eight Networks-table pairs also moved.
  `extract_v12.py` gained `--algtune_dir`, a second tuning channel alongside the backbone one, so
  a re-selected cell reaches the report the same way every other cell does instead of being pasted
  in. The two channels stay separate because their keys mean different things: a backbone-tuning
  key is a bare backbone whose row is `EA-<backbone>`, while an algorithm-tuning key is already the
  preset name. `apply_v12.py` reports a rejected preset run only while nothing else has supplied
  that cell — once the re-tuned value arrives, both channels deliver the same cell and announcing
  the rejection would report finished work as outstanding.
- **A second ensemble table was affected and was not on the re-measurement list.** `RESULTS.md` has
  two ensemble experiments — the decentralized one (weak per-subject base models) and the multi-seed
  one (K = 5 seeds of T-TIME, strong correlated base models) — and `RERUN.md`'s entry, written as
  "the whole ensemble table", is justified entirely by a base learner only the decentralized table
  uses. The multi-seed table uses the same three fixed combiners (SML, LAA, PM) and appeared nowhere
  in the affected list. It is now re-measured on all three datasets, by re-running the fusion over
  the cached base predictions on the machine holding each dataset's runs, and **reproduces every
  published value exactly** — so that section is unchanged. The fixes are no-ops there by
  construction: SML's global-sign repair only matters for a below-chance base learner and PM's
  uniform tie-breaking only for a tie, and five seeds of one strong algorithm supply neither. That
  is the same property, read the other way round, that makes those fixes move every row of the
  decentralized table. `RERUN.md` now lists the two tables separately.
- **`apply_v12.py` refuses a cell whose `key` identifies more than one row.** `key` is the identity
  the report, the provenance map and the writer all match on, but nothing made one key mean one
  measurement: `EA-EEGNet` is carried both by the reference row every table is measured against
  (72.07 on BNCI2014001) and by the Networks table's EEGNet row, which is the same architecture at a
  *grid-searched* learning rate (72.53). Writing one report value into both would leave one row
  wrong and still plausible — they differ by less than half a point. Nothing had gone wrong only
  because EA-EEGNet is a control that reproduced, which is luck rather than a guard. The check keys
  on the rows *disagreeing*, so `EA-EEGNet` on BNCI2014002, where all four rows genuinely read
  74.40, is not flagged. Verified by perturbation: feeding it a fabricated EA-EEGNet cell makes it
  name all four rows and drop the write.
- Three more affected cells were being **skipped entirely**: DeepConvNet, EEGConformer and
  DBConformer on BNCI2014001. They were recorded as reproducing on no surviving tree anywhere, but
  their numbers sit in the same verdict file, for the same dataset, that resolved ShallowConvNet.
  They would have shipped with pre-fix numbers while the rest of the Networks table moved.
- Re-selecting the learning rate turns out to matter on its own: **six of the eight finished pairs
  chose a different learning rate than v1.1.x**. The shape probe perturbed the RNG that seeds the
  validation runs, so the curve the selection reads is redrawn and its argmax moves. Re-measuring
  those cells at their old learning rates would have reported numbers produced by a selection
  procedure the release no longer performs.
- `rerun_v12.sh` releases a failed job's claim with `rm -rf` rather than `rmdir`. The claim directory
  holds the owner's `pid` file, so `rmdir` could never succeed, and its error went to `/dev/null` —
  the claim outlived the failure, every later worker in that run skipped the job, and only the reaper
  at the *next* launch could free it. The comment above the line already said the opposite. Surfaced
  when 37 jobs failed on GPU memory at once and left 120 claims against 77 results.
- `rerun_v12_nettune.sh` folds 10022/20022 onto one family name, which it alone among the three sweep
  scripts did not do, and takes `DS_LIST`/`BB_LIST` so a subset can be re-run without shard
  arithmetic — the shard index counts pairs before the already-tuned check, so a pair still being
  tuned by a live worker would have been picked up a second time into the same results directory.
- The control comparison in `compare_v12.py` was **never actually able to run.** It expects the
  v1.1.x baselines under `<dir>/<machine>/<sweep>/`, no such tree existed on any box, and its answer
  for a baseline it cannot find is "not checkable here" — deliberately distinct from "failed", but
  indistinguishable from it if nobody reads the output. So RERUN.md's central claim, that the blast
  radius is correctly scoped, had no evidence behind it. The trees are now assembled per machine
  (10022's had to be read off the box with the wedged GPU, which is still fine over SSH) and the
  controls checked: **10 of 10 in-family controls reproduce v1.1.x identically, per subject** — 7 in
  the 10022 family, 2 on 7002, 1 on 60022, across `EA-EEGNet`, `NoAlign-EEGNet`, `Noise-EEGNet` and
  `FShift-EEGNet`. Not to two decimals: the same accuracy for every one of 9, 12 or 14 subjects.
- A `schedule:` trigger on the CI workflow. Its link-check job was documented as weekly but was
  reachable only by hand, so the weekly run it promised had never happened once.
- A **Classical Pipelines** leaderboard table: `CSP-LDA` and `Riemann-MDM`, the two network-free
  rows, on all three datasets. They were already in `RESULTS.md` and in the reproduction registry
  but on no leaderboard table, and the registry's cross-check against `benchmark.yml` *skips* a row
  the leaderboard lacks — so those two were the only skipped tests in the suite, and their six
  published numbers were the only ones nothing cross-checked. The web app never showed the
  benchmark's one network-free track either. Both rows were re-run on their origin box before being
  published and came back bit-identical to v1.1.x per subject on every seed, which also confirms the
  exactly-zero across-seed std they report. The suite now runs with **no skips**.
- `hustbciml/scripts/rerun_v12_classical.sh` — re-runs those two rows on the box that produced them.
  CPU only, so it does not compete with the GPU sweeps for a device.
- `hustbciml/tests/repro/test_readme_counts.py` — the approach count is stated by hand six times
  across the two READMEs (a shields.io badge and the opening sentence in each), so adding one
  leaderboard row made all of them wrong at once, and a stale badge is a number inside a picture
  that nothing about looks wrong. Checked against the built site data, which is derived from
  `benchmark.yml`. The version-history bullets are deliberately excluded: "(**56** approaches)"
  under v1.1.2 is a true statement about that release and must not be rewritten.

### Removed
- The SAFE special case that rewrote `BNCI2014001` to `BNCI2014001-4` in the committed sweep and
  extractor scripts. The published cell is two-class and correct; re-running the documented
  pipeline would have replaced it with a four-class result (chance 25% against chance 50%).

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

[1.3.0]: https://github.com/sylyoung/HUST-BCIML/releases/tag/v1.3.0
[1.2.5]: https://github.com/sylyoung/HUST-BCIML/releases/tag/v1.2.5
[1.2.4]: https://github.com/sylyoung/HUST-BCIML/releases/tag/v1.2.4
[1.2.3]: https://github.com/sylyoung/HUST-BCIML/releases/tag/v1.2.3
[1.2.2]: https://github.com/sylyoung/HUST-BCIML/releases/tag/v1.2.2
[1.2.1]: https://github.com/sylyoung/HUST-BCIML/releases/tag/v1.2.1
[1.2.0]: https://github.com/sylyoung/HUST-BCIML/releases/tag/v1.2.0
[1.1.3]: https://github.com/sylyoung/HUST-BCIML/releases/tag/v1.1.3
[1.1.2]: https://github.com/sylyoung/HUST-BCIML/releases/tag/v1.1.2
[1.1.1]: https://github.com/sylyoung/HUST-BCIML/releases/tag/v1.1.1
[1.1.0]: https://github.com/sylyoung/HUST-BCIML/releases/tag/v1.1.0
