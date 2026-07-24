<div align="center">

# `hustbciml`: a unified EEG-decoding benchmark

**Controlled, reproducible head-to-head comparison of the HUST-BCIML lab's EEG-decoding methods**

![Python](https://img.shields.io/badge/python-3.10%2B-3776ab)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c)
![Approaches](https://img.shields.io/badge/approaches-56-4338ca)
![Seeds](https://img.shields.io/badge/protocol-cross--subject%20LOSO%20%C3%97%203%20seeds-059669)

</div>

A unified rebuild of
DeepTransferEEG whose value is **algorithm coverage**. It ports ~30 of the lab's EEG-decoding
methods, plus the standard external baselines, onto **one stage architecture** and measures
them head-to-head under a **single controlled protocol**. The full 3-seed benchmark runs on three
MOABB motor-imagery EEG datasets, all cross-subject, leave-one-subject-out: **BNCI2014001**
(9 subjects, 22 ch), **BNCI2014002** (14, 15 ch) and **BNCI2015001** (12, 13 ch). A
bundled synthetic **Toy** dataset runs everything with no download, as a smoke test.

> **Status:** actively developed · released under the MIT License (see [`LICENSE`](../LICENSE)).
> **Docs:** [index](docs/index.md) · [glossary](docs/glossary.md) ·
> [algorithm cards](docs/cards/README.md) · [porting guide](docs/porting_guide.md)

## The idea: an algorithm is a composition of plug-in stages

```
Aligner  →  Augmenter  →  Backbone  →  Head        driven by a  Strategy  (learning objective)
```

Each **controlled comparison varies exactly one stage** and holds the rest at the canonical
configuration (**EA · no-aug · EEGNet · Linear · ERM**). Every row differs from its baseline
in one way, so the comparison is fair. A separate **ensemble** axis aggregates several models.

### Registry (`python -m hustbciml.run --list`)

| Stage | Plug-ins |
|---|---|
| **Aligner**   | `EA`, `RA`, `Identity` |
| **Augmenter** | `ChannelReflection`, `CSDA`, `Noise`, `Flip`, `Scaling`, `FShift`, `FSurr`, `FComb`, `HS`, `Identity` |
| **Backbone**  | `EEGNet`, `ShallowConvNet`, `DeepConvNet`, `EEGConformer`, `DBConformer`, `IFNet`, `CSPNet`, `TIEEEGNet`, `KDFNet`, `ADFCNN`, `CTNet`, `MSCFormer`, `MSVTNet`, `TMSANet`, `EEGWaveNet`, `SlimSeiz`, `FBMSNet`, `EEGNeX`, `EEGDeformer` |
| **Head**      | `Linear` |
| **Strategy**  | `ERM`, `MEKT`, `MDMAML`, `ABAT`, `DJPMMD`, `MCC`, `CDAN`, `JAN`, `DAN`, `DANN`, `MDD`, `ASFA`, `LSFT`, `SHOT`, `TTIME`, `DELTA`, `ISFDA`, `SAR`, `PL`, `BNAdapt`, `Tent`, `BFT`, `FedBS`, `SAFE`, `FedAvg`, `MSDT`, `MVCNet`, `CSP_LDA`, `RiemannMDM` |
| **Protocol**  | `cross_subject` (leave-one-subject-out) |

**Presets** in `algorithms/presets/*.yaml` name the composite algorithms, the units on the
leaderboard: `EA-EEGNet`, `T-TIME`, `MEKT`, `MCC`, `CSP-Net`, `EA-DBConformer`, `MVCNet`,
`FedBS`, `SAFE`, … (`--list` for all, or pass `--algorithm <preset>`).

## Run it

```bash
pip install -r ../requirements.txt        # from this directory; or -r requirements.txt from repo root

# from the repo root (so `hustbciml` is importable)
python -m hustbciml.run --list                                          # every plug-in
python -m hustbciml.run --algorithm EA-EEGNet --dataset Toy --device cpu # synthetic, no download
python -m hustbciml.run --algorithm T-TIME    --dataset Toy --device cpu
python -m hustbciml.tests.test_smoke                                    # integration checks
```

Real data, fetched via MOABB on first run, then cached:

```bash
python -m hustbciml.run --algorithm EA-EEGNet --dataset BNCI2014001 --itr 3   # 3 seeds
python -m hustbciml.run --algorithm MEKT      --dataset BNCI2014002 --device cpu
```

Each run writes `results/<setting>/metrics.json` (per-subject accuracies + mean/std),
`predictions.npz`, and the resolved `config.yaml`. Compose your own algorithm on the fly:

```bash
python -m hustbciml.run --aligner EA --augmenter CSDA --backbone DBConformer \
                        --strategy ERM --head Linear --dataset BNCI2014001 --itr 3
```

**Ensembles** are black-box multi-seed combiners over K seeds of a base algorithm (hard majority
voting, ten crowd-label aggregators, and the lab's SML / SML-OVR / StackingNet), fusing hard
votes only:

```bash
python -m hustbciml.scripts.ensemble --algorithm T-TIME --dataset BNCI2014001 --seeds 1,2,3,4,5
```

## Results

The canonical base `EA · no-aug · EEGNet · Linear · ERM` scores **72.07 ± 1.58** on
BNCI2014001 (3-seed mean ± std, cross-subject LOSO). The **full three-dataset leaderboard**,
per-method notes, and the ensemble section are in **[RESULTS.md](RESULTS.md)**, with the
interactive version in the **[web app](https://sylyoung.github.io/HUST-BCIML/)**. Per-method
mechanism, exact stage configuration, and reproduced-vs-reference numbers are in the
[algorithm cards](docs/cards/README.md).

Rankings are **dataset-dependent** and reported as measured. No single method wins everywhere.
Faithful **negative** results are kept and explained.

## Reproduction integrity

Every leaderboard number is a **measured** 3-seed mean, never hand-set to hit a paper's value.
Each is recorded in [`tests/repro/repro_targets.yaml`](tests/repro/repro_targets.yaml) against
the paper's own value where the protocol matches, or an expected behaviour band where it
differs, with a per-method note. The algorithm [cards](docs/cards/README.md) carry the
reported-vs-reproduced table and a vendored-code license/provenance audit per method.

## Data

`data_provider/` has three adapters behind a hard-coded `DATA_DICT`:

- **`ToyDataset`**: synthetic, bundled, deterministic (used by the smoke test, needs no download).
- **`NumpyDataset`**: reads a DeepTransferEEG-style `data/<name>/X.npy` + `labels.npy` layout.
- **`MOABBAdapter`**: downloads BNCI via MOABB into that same layout, then loads it. This is the path
  used to produce the 3-seed benchmark numbers.

## Architecture

```
hustbciml/
  run.py                # CLI: preset + CLI → protocol → itr loop
  core/                 # batch, stages (ABCs), registry (auto-scan), pipeline, config, context
  exp/                  # exp_basic + exp_cross_subject (one class per protocol)
  algorithms/           # aligners / augmenters / models / heads / strategies / presets
  data_provider/        # datasets, data_factory, splitters, collate
  utils/                # metrics, seed, tools (EarlyStopping)
  scripts/              # ensemble, leaderboard, compare, tuning
  tests/                # test_smoke (integration, no download) + repro targets
  docs/                 # index, glossary, porting guide, per-algorithm cards
```

## Add an algorithm

Drop `algorithms/<group>/<Name>.py` defining `class <Name>` that conforms to the stage ABC. It
**auto-registers by filename** (`--<stage> <Name>`). Add a preset YAML to compose it, a `repro`
target once real numbers exist, and a card. Each new file carries a standard header with the author,
date, the exact IEEE citation, and a link to the original authors' code. Full workflow in the
[porting guide](docs/porting_guide.md).

## Disclaimer & contact

This benchmark **reimplements** both external baselines and the lab's own methods independently.
The reported numbers may differ from the original papers and can contain errors. Corrections are
welcome. Maintained by **Siyang Li**, [homepage](https://sylyoung.github.io/) ·
**lsyyoungll@gmail.com**. See the top-level [README](../README.md) for the paper-to-code gallery
and the web app.
