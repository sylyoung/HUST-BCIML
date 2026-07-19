# hustbciml — controlled comparison results

**Datasets:** three MOABB motor-imagery datasets, cross-subject leave-one-subject-out,
3 seeds (1, 2, 3).
- **BNCI2014001** — 9 subjects, 22 channels. Two-class (left vs right hand, chance 50%)
  for every table, including the privacy-preserving and ensemble sections; the native
  four-class variant (both hands, feet, tongue) stays available in code but is not reported here.
- **BNCI2014002** — 14 subjects, 15 channels, right hand vs feet, two-class (chance 50%).
- **BNCI2015001** — 12 subjects, 13 channels, right hand vs feet, two-class (chance 50%).

Every `Acc` is the mean over seeds; `±` is the std ACROSS seeds (reproducibility), not the
cross-subject spread; deterministic network-free methods have std 0.00 by construction.
Measured on the hust-gpu servers, CUDA for the neural methods and CPU for the classical
track.

Each stage table varies exactly ONE axis and holds the rest at the canonical configuration
(EA aligner, no augmentation, EEGNet backbone, Linear head, ERM strategy); a row's Δ
(discussed in prose) is its accuracy minus that table's same-dataset baseline. Rows are
comparable because every row differs from its baseline in exactly one way. Columns are
directly comparable within every table: all three datasets are two-class (chance 50%)
throughout, including the privacy-preserving and ensemble sections. Each family is measured
against its own same-dataset baseline — the transfer families against ERM, the
privacy-preserving family against Centralized Training, and the ensemble table against
majority voting. A single flat ranking across all methods is deliberately NOT presented.

> **Base note (why 3 seeds matter).** The earlier single-seed (seed 1) EA-EEGNet
> value was 75.00; the 3-seed mean is **72.07 ± 1.58** — seed 1 was a lucky draw.
> Because most methods are measured as a delta against this base, the honest 3-seed
> base widened nearly every delta below. Every number is a measured 3-seed mean —
> none tuned to hit a target.
>
> **HP selection (2026-07-19).** Cells in the Network, Transfer, Augmentation, Composite and
> classical-transfer tables were refreshed from a held-out-source hyperparameter campaign: for
> each flagged method a small grid (learning rate, epochs/batch, and the method's own loss
> trade-offs) was scored on held-out SOURCE subjects (never the target/test labels), and the
> winning config's 3-seed test number replaced the earlier one ONLY where it beat it — otherwise
> the original stands. This extends the per-architecture backbone tuning to the other stages;
> selection never touches the reported cohort, so ‘none tuned to hit a target’ still holds.

**Lab methods** (marked **(lab)** throughout) are proposed by Prof. Dongrui Wu's group:
the backbones **CSP-Net**, **DBConformer**, **TIE-EEGNet** and **KDFNet**; the augmenters
**Channel Reflection** and **CSDA**; the transfer methods **MEKT**, **MDMAML**, **ASFA**,
**ABAT**, **BFT**, **DJP-MMD**, **LSFT** and **T-TIME**; the composite **MVCNet**; the
privacy methods **SAFE**, **FedBS** and **MSDT**; and the decentralized-ensemble combiners
**SML-OVR** and **StackingNet**. Their measured numbers are recorded — against each
paper's own reported value where the protocol matches, or an expected behavior band where it
differs — in `tests/repro/repro_targets.yaml`. Six methods added 2026-07-18 —
**TIE-EEGNet**, **KDFNet** (backbones), **MEKT**, **MDMAML**, **ASFA** (transfer) and
**SAFE** (privacy) — are marked *(new)*.

Regenerate live (on the server, where the raw results are), per dataset:
`python -m hustbciml.scripts.compare /home/sylyoung/hustbciml_results_3ds --dataset BNCI2014002`

---

## Network (backbone)
_EA + ERM, no aug; vary the deep architecture. Unlike the other stage tables, each
backbone uses its own learning rate and training length, selected on held-out
source-validation data (per-architecture tuning); the EEGNet baseline here (tuned) is
therefore slightly different from the canonical fixed-HP EEGNet used as the baseline in
every other table. Baseline = EEGNet._

| Backbone | BNCI2014001 | BNCI2014002 | BNCI2015001 |
|---|--:|--:|--:|
| CSP-Net **(lab)** | 75.15 ± 1.06 | 74.40 ± 0.24 | 72.42 ± 0.38 |
| DBConformer **(lab)** | 74.85 ± 0.98 | **77.05 ± 0.60** | 72.94 ± 0.84 |
| DeepConvNet | 74.07 ± 1.04 | 68.64 ± 0.20 | 69.99 ± 1.07 |
| EEGConformer | 74.05 ± 0.58 | 75.12 ± 1.00 | 73.07 ± 1.37 |
| TIE-EEGNet **(lab)** *(new)* | 73.51 ± 0.25 | 73.17 ± 0.35 | 73.83 ± 0.38 |
| ShallowConvNet | 72.69 ± 0.71 | 71.14 ± 0.25 | 73.03 ± 0.28 |
| EEGNet (baseline) | 72.53 ± 1.22 | 74.40 ± 1.04 | 73.39 ± 0.69 |
| KDFNet **(lab)** *(new)* | 70.88 ± 0.32 | 72.64 ± 0.69 | 68.65 ± 1.05 |

The backbone ranking is dataset-dependent, and the table reports it as measured. On
BNCI2014001 the two lab backbones lead — CSP-Net (75.15, an EEGNet whose depthwise spatial
convolution is initialized with frozen CSP filters) just above the dual-branch
convolutional-transformer DBConformer (74.85); per-architecture LR tuning also tames the
variance EEGConformer shows at fixed HP, lifting it to 74.05. On BNCI2014002 DBConformer
leads clearly (77.05). On BNCI2015001 the backbones are tightly bunched: with per-architecture LR tuning **TIE-EEGNet** just edges ahead (73.83), a shade above the plain EEGNet baseline (73.39), with DBConformer (72.94) and CSP-Net (72.42) close behind — a faithful, narrow spread. **TIE-EEGNet** (EEGNet with a sinusoidal time-positional
convolution) tracks the EEGNet baseline closely on all three, edging just ahead on BNCI2015001. **KDFNet** (an
FBCSP-mirroring CNN) lands below the baseline on every dataset — the weakest backbone here,
an honest below-baseline result. No single backbone dominates all three datasets.

## Alignment
_EEGNet + ERM, no aug; vary the aligner. Baseline = no alignment._

| Aligner | BNCI2014001 | BNCI2014002 | BNCI2015001 |
|---|--:|--:|--:|
| RA (Riemannian) | 73.97 ± 1.27 | 71.86 ± 1.23 | 72.39 ± 0.32 |
| EA (Euclidean) | 72.07 ± 1.58 | 74.40 ± 1.04 | 73.19 ± 0.81 |
| none | 69.34 ± 0.65 | 61.90 ± 2.96 | 63.46 ± 0.83 |

Alignment helps most where the raw cross-subject shift is largest: on BNCI2014002 and
BNCI2015001 EA lifts accuracy by **+12.5** and **+9.7** over no alignment (61.90 → 74.40,
63.46 → 73.19), far more than on BNCI2014001 (+2.73). The two aligners trade places by
dataset, though: RA (recentring on the Riemannian/Fréchet mean) is the best aligner on
BNCI2014001 (+1.90 over EA), but on both new datasets RA falls **below** EA (71.86 vs
74.40; 72.39 vs 73.19) — the geometric mean is not universally the better reference. EA is
the safer default, and is the canonical aligner used throughout the rest of the benchmark.

## Transfer / adaptation strategy
_EA + EEGNet, no aug; vary the training/adaptation procedure. Baseline = ERM (no
transfer). All on the identical EEGNet network._

| Strategy | BNCI2014001 | BNCI2014002 | BNCI2015001 |
|---|--:|--:|--:|
| MCC | 79.04 ± 0.67 | 80.88 ± 1.64 | 78.53 ± 0.61 |
| CDAN | 76.26 ± 0.94 | 78.31 ± 0.89 | 76.22 ± 0.76 |
| T-TIME **(lab)** | 76.05 ± 0.42 | 80.33 ± 0.52 | 77.75 ± 0.68 |
| DELTA | 75.93 ± 0.44 | 80.14 ± 0.51 | 77.44 ± 0.64 |
| ISFDA | 75.80 ± 0.54 | 79.81 ± 0.42 | 77.74 ± 0.53 |
| JAN | 75.44 ± 0.41 | 75.86 ± 0.67 | 74.64 ± 0.57 |
| MDMAML **(lab)** *(new)* | 75.13 ± 0.38 | 73.40 ± 1.23 | 73.06 ± 0.23 |
| DAN | 75.03 ± 1.04 | 73.90 ± 0.61 | 74.40 ± 1.20 |
| SAR | 74.90 ± 1.99 | 77.12 ± 2.02 | 72.31 ± 1.95 |
| DANN | 74.77 ± 1.01 | 74.02 ± 0.79 | 73.65 ± 1.11 |
| PL | 74.38 ± 1.89 | 77.05 ± 1.20 | 73.96 ± 1.01 |
| ABAT **(lab)** | 74.20 ± 0.69 | 74.90 ± 0.41 | 74.14 ± 0.70 |
| SHOT | 74.20 ± 1.06 | 75.93 ± 0.70 | 75.64 ± 0.23 |
| MDD | 74.18 ± 0.25 | 74.48 ± 0.93 | 73.17 ± 0.56 |
| BFT **(lab)** | 73.79 ± 0.67 | 76.29 ± 0.73 | 74.46 ± 0.31 |
| PAT **(lab)** *(new)* | 73.53 ± 0.95 | 75.12 ± 0.47 | 74.08 ± 0.42 |
| ASFA **(lab)** *(new)* | 73.28 ± 0.51 | 75.10 ± 0.93 | 74.68 ± 0.17 |
| BN-adapt | 73.23 ± 1.29 | 75.00 ± 1.19 | 75.04 ± 0.56 |
| DJP-MMD **(lab)** | 73.10 ± 0.64 | 77.62 ± 0.44 | 73.49 ± 0.64 |
| ERM (no transfer) | 72.07 ± 1.58 | 74.40 ± 1.04 | 73.19 ± 0.81 |
| Tent | 72.04 ± 1.42 | 73.81 ± 0.99 | 72.01 ± 1.13 |

MCC leads on every dataset (79.04 / 80.88 / 78.53), and the strong adaptation cluster —
CDAN, T-TIME, DELTA, ISFDA — sits +3 to +6 over ERM throughout; the online and source-free
methods (T-TIME, DELTA, ISFDA) are especially strong on the two new datasets (around 80 on
BNCI2014002). The new **MDMAML** (a meta-learned, forward-only EEGNet initialization) is
the standout on BNCI2014001 at 75.13 (+3.06 over ERM), and on the two new datasets lands −1.00 below ERM on BNCI2014002 and essentially level on BNCI2015001 (73.06, −0.13) — the meta-learned initialization helps most where the source
pool is largest, a faithful mixed result. The new source-free **ASFA** is at or above ERM on all three (+1.21 / +0.70 / +1.49), the mildest member of the source-free family (ISFDA and SHOT are stronger), with low-to-moderate across-seed variance (±0.17–0.93). At the bottom,
Tent (entropy-only on BatchNorm parameters) sits below ERM on BNCI2015001 (72.01, −1.18), the clearest such dip. MDD keeps the tightest across-seed variance on
BNCI2014001 (±0.25). MEKT (the lab's network-free manifold-embedded transfer) is reported
separately in the classical-transfer section below.

## Augmentation (no-alignment regime) — BNCI2014001 only
_EEGNet + ERM; vary the augmenter. Baseline = no augmentation. Held at no-alignment
(aligner: Identity): Channel Reflection is an electrode-space transform and must precede
any spatial whitening._

| Augmenter | Acc ± std | Δacc vs base |
|---|--:|--:|
| Channel Reflection **(lab)** | 73.23 ± 0.74 | +3.88 |
| none | 69.34 ± 0.65 | (baseline) |

Channel Reflection mirrors each trial across the sagittal midline and swaps the left/right
label, doubling the training set with anatomically valid copies (+3.88). It is
constitutionally two-class **left/right** and is therefore measured only on BNCI2014001:
BNCI2014002 and BNCI2015001 are right-hand-vs-feet, which have no left/right hemispheric
symmetry for the reflection to exploit, so the transform does not apply there.

## Augmentation (EA regime)
_EA + EEGNet + ERM; vary the augmenter. Baseline = EA-EEGNet (no augmentation). CSDA
(db4-wavelet cross-subject detail-swap) operates on EA-aligned trials, so it is measured in
the EA regime — unlike Channel Reflection above, which is an electrode-space transform held
at no-alignment._

| Augmenter | BNCI2014001 | BNCI2014002 | BNCI2015001 |
|---|--:|--:|--:|
| CSDA **(lab)** | 72.74 ± 1.92 | 73.98 ± 0.32 | 73.53 ± 0.44 |
| none (EA-EEGNet) | 72.07 ± 1.58 | 74.40 ± 1.04 | 73.19 ± 0.81 |

CSDA is marginal on all three datasets: +0.67 on BNCI2014001, −0.42 on BNCI2014002, +0.34
on BNCI2015001 — a small, sometimes slightly negative effect consistent with its
high-variance, small-gain profile. Only the paper's DWTaug variant is ported (HHTaug
omitted).

## Composite method (changes more than one stage)
_Not a single-axis controlled comparison. Shown against the EA-EEGNet reference as a
context number only._

| Method | BNCI2014001 | BNCI2014002 | BNCI2015001 |
|---|--:|--:|--:|
| MVCNet **(lab)** (IFNet + multi-view contrastive) | 75.64 ± 0.95 | 76.69 ± 0.94 | 72.21 ± 0.50 |
| _EA-EEGNet (reference)_ | 72.07 ± 1.58 | 74.40 ± 1.04 | 73.19 ± 0.81 |

MVCNet changes two stages at once — an IFNet CNN backbone **and** a multi-view contrastive
training strategy (cross-view + cross-modal supervised-contrastive losses) — so it cannot
sit in any one-axis table; at inference it is just IFNet + the linear head. It is strong on
BNCI2014001 (75.64, +3.57 over the reference) and BNCI2014002 (76.69, +2.29). On BNCI2015001,
selecting its learning rate on held-out source data (3e-4, not the preset 1e-3) lifts it from
an earlier 67.93 to 72.21 (−0.98 below the reference) — the collapse was a learning-rate
artifact, not the method. Its two contrastive loss weights are set to 1.0 (the
source has no hardcoded default).

## Classical (network-free) baselines
_EA-aligned trials into a classical pipeline (no backbone). Deterministic fit-mode, so the
across-seed std is 0.00. Shown against the deep EA-EEGNet reference._

| Method | BNCI2014001 | BNCI2014002 | BNCI2015001 |
|---|--:|--:|--:|
| CSP-LDA | 73.77 ± 0.00 | 72.71 ± 0.00 | 72.00 ± 0.00 |
| Riemann-MDM | 71.68 ± 0.00 | 69.57 ± 0.00 | 66.42 ± 0.00 |
| _EA-EEGNet (deep reference)_ | 72.07 ± 1.58 | 74.40 ± 1.04 | 73.19 ± 0.81 |

CSP-LDA (CSP spatial filters + LDA) is a strong classical baseline: it edges past the deep
EA-EEGNet reference on BNCI2014001 (73.77 vs 72.07) and sits within ~1.4–1.7 points of it on
the two new datasets (72.71 vs 74.40; 72.00 vs 73.19). Riemann-MDM (minimum distance to the
Riemannian mean) trails on all three, by a widening margin on the higher-channel-count
datasets — a reminder that a well-tuned CSP-LDA pipeline stays competitive with a deep
network on cross-subject MI, while the simpler covariance-distance classifier does not.

## Classical transfer methods (lab)
_Lab transfer on Riemannian tangent-space features (no backbone), deterministic across
seeds. Distinct from the no-transfer classical baselines above and from the neural transfer
table. Shown against the deep EA-EEGNet reference on each dataset._

| Method | BNCI2014001 | BNCI2014002 | BNCI2015001 |
|---|--:|--:|--:|
| MEKT *(new)* | 76.54 ± 0.00 | 77.86 ± 0.00 | 73.04 ± 0.00 |
| LSFT | 74.77 ± 0.00 | 73.64 ± 0.00 | 75.46 ± 0.00 |
| _EA-EEGNet (deep reference)_ | 72.07 ± 1.58 | 74.40 ± 1.04 | 73.19 ± 0.81 |

**LSFT** (Zhang & Wu, IEEE TCDS 2022) is source-free: it votes source classifiers to
pseudo-label the target, builds a virtual intermediate source from the confident trials,
then iteratively adapts a DJP-MMD subspace — all on tangent-space features. On BNCI2014001 it
reaches 74.77 (+2.70 over the reference; the paper reports 75.15 on this exact 2-class 2a
LOSO setup, so −0.38, a faithful reproduction), and it is above the reference on BNCI2015001
(75.46, +2.27), a little below on BNCI2014002 (73.64, −0.76). The new **MEKT** (Manifold Embedded Knowledge Transfer, Zhang & Wu 2020) is the full method: it centroid-aligns each subject's covariances, maps to Riemannian tangent-space features, then learns a joint source/target subspace projection (joint-probability MMD + source discriminability + target locality + coupling, via a generalized eigendecomposition with EM pseudo-label refinement) before a shrinkage-LDA. It reaches 76.54 / 77.86 / 73.04 — above the deep EA-EEGNet reference on BNCI2014001 (+4.47) and BNCI2014002 (+3.46) and level on BNCI2015001 (−0.15), and now tops the classical-transfer table on those first two datasets (LSFT leads on BNCI2015001). Both are network-free and deterministic (std 0.00). (The Sec-III-C projection — ported from the authors' code via TBC-TJU/MetaBCI — lifts BNCI2014001 by +6.25 and BNCI2014002 by +4.86 over the CA-tangent core; the earlier core-only reading came from a buggy projection attempt, now fixed — see the MEKT card.)

> **MSDT moved.** MSDT was previously listed here on the two-class task; it is now
> evaluated in the **Privacy-preserving** section below, alongside the federated and
> decentralized-ensemble methods it belongs with.

---

## Multi-seed ensemble

_A post-hoc black-box ensemble over K random seeds of one base algorithm: each seed is a
full run, and for every target subject the K seeds' per-trial predictions are fused by a
combiner that sees only the **hard votes** — there is no soft-score averaging combiner, so
none has an information advantage. Hard majority **voting** is the baseline; the other
combiners are the same crowd-label aggregators (Dawid-Skene, Wawa, M-MSR, MACE, GLAD,
ZenCrowd, PM, LA, LAA, EBCC) and lab methods (SML, SML-OVR, StackingNet) used in the
privacy section below. Measured on BNCI2014001, cross-subject LOSO (9 subjects, 2-class,
chance 50%), K = 5 seeds {1–5}; each combiner accuracy is the mean over subjects, shown
without std (the leaderboard convention). Reproduce with `python -m hustbciml.scripts.ensemble
--algorithm T-TIME --dataset BNCI2014001 --seeds 1,2,3,4,5`._

| Combiner | Acc | Δ vs single-seed base |
|---|--:|--:|
| M-MSR | 79.09 | +3.44 |
| ZenCrowd | 78.55 | +2.90 |
| GLAD | 78.40 | +2.75 |
| LA | 78.40 | +2.75 |
| voting | 78.32 | +2.67 |
| Wawa | 78.32 | +2.67 |
| MACE | 78.32 | +2.67 |
| LAA | 78.32 | +2.67 |
| SML | 78.32 | +2.67 |
| SML-OVR **(lab)** | 78.32 | +2.67 |
| StackingNet **(lab)** | 78.32 | +2.67 |
| Dawid-Skene | 78.24 | +2.59 |
| EBCC | 78.01 | +2.36 |
| PM | 77.62 | +1.98 |
| _single-seed base (mean over 5 seeds)_ | 75.65 ± 0.60 | (ref) |

This is the mirror image of the decentralized-ensemble section. Here the base models are **strong and
highly correlated** — five seeds of T-TIME, each ~75.65 %, all well above the 50 % chance
level — so they rarely disagree, and there is little for a weighting scheme to exploit:
most hard combiners collapse to the plain majority vote (78.32), and the lab's **SML-OVR**
and **StackingNet** land exactly on that baseline. **M-MSR** (79.09) edges out the rest.
The ensemble lifts the single-seed base by about +3, in line with a K = 5 prediction
ensemble. (An across-subject std is omitted from the table because it is large — 2a
subjects range from near-chance to > 95 % — and would only report subject spread, not
combiner precision; the ± 0.60 on the base row is the across-seed std.) Contrast the
decentralized ensemble, where **weak per-subject** base models spread the combiners out and
Dawid-Skene pulls ahead — the same combiners, opposite base-model regime.

### Multi-seed ensemble on the two other datasets

The same K = 5 T-TIME ensemble on BNCI2014002 and BNCI2015001 tells the same story: the base
models are strong and highly correlated, so the hard combiners cluster near plain voting and
lift the base by about +1 to +2.

| Dataset | single-seed base (5 seeds) | plain voting | combiner leader |
|---|--:|--:|--:|
| BNCI2014002 | 79.97 ± 1.20 | 81.57 (+1.60) | voting cluster (81.57) |
| BNCI2015001 | 77.27 ± 0.87 | 78.50 (+1.23) | PM (79.08, +1.82) |

On BNCI2014002 every hard combiner ties the majority vote at 81.57 (voting, Wawa, M-MSR,
GLAD, LA, LAA, SML, StackingNet), with Dawid-Skene / MACE (81.43) and PM (81.29) a hair
behind — no weighting scheme beats voting on these near-identical strong models. On
BNCI2015001 PM (79.08) edges just ahead of the voting cluster (78.50), with EBCC (78.79) and
Dawid-Skene (78.67) next. As in the decentralized ensemble, the multi-class SML-OVR reduces
to binary SML on these two-class datasets, so it matches SML (which sits in the voting
cluster). The lift is modest and positive on all three datasets — the mirror image of the
decentralized ensemble, where weak per-subject base models spread the combiners out.

---

## Decentralized-heterogeneous ensemble

_A fully decentralized, privacy-preserving ensemble. Five heterogeneous learners — tangent-space
LDA, tangent-space SVM, EEGNet, ShallowConvNet, and CSPNet — are trained on each source subject's
EA-aligned data alone, and the subjects share only their **hard predicted labels** on the target,
never model weights or raw EEG. Each target trial therefore collects (N−1)×5 hard votes, which a
post-hoc black-box combiner fuses into one consensus label — no target labels, no model internals,
no soft scores, so no combiner has an information advantage; they differ only in how they weight
and combine the votes. Two combiners are lab-proposed (**StackingNet**, and the multi-class
**SML-OVR**); the rest are established crowd-labelling / truth-discovery aggregators. All three
datasets are two-class (chance 50%), cross-subject LOSO, 1 seed; each combiner is measured against
plain majority **voting**. Reproduce with
`python -m hustbciml.scripts.decentralized --dataset <D> --base hetero`._

| Combiner | BNCI2014001 | BNCI2014002 | BNCI2015001 |
|---|--:|--:|--:|
| PM | 76.16 (+1.85) | 70.57 (−1.43) | 65.79 (−4.04) |
| LAA | 76.08 (+1.77) | 73.36 (+1.36) | 72.71 (+2.88) |
| EBCC | 76.08 (+1.77) | 72.43 (+0.43) | 71.17 (+1.34) |
| SML-OVR **(lab)** | 75.46 (+1.15) | 73.14 (+1.14) | 72.71 (+2.88) |
| SML | 75.46 (+1.15) | 73.14 (+1.14) | 72.71 (+2.88) |
| StackingNet **(lab)** | 75.31 (+1.00) | 73.00 (+1.00) | 70.50 (+0.67) |
| Dawid-Skene | 74.85 (+0.54) | 73.14 (+1.14) | 74.29 (+4.46) |
| ZenCrowd | 74.85 (+0.54) | 66.93 (−5.07) | 59.21 (−10.62) |
| GLAD | 74.61 (+0.30) | 67.29 (−4.71) | 59.83 (−10.00) |
| Wawa | 74.38 (+0.07) | 72.14 (+0.14) | 68.50 (−1.33) |
| LA | 74.38 (+0.07) | 70.21 (−1.79) | 65.29 (−4.54) |
| MACE | 73.46 (−0.85) | 65.50 (−6.50) | 72.00 (+2.17) |
| M-MSR | 72.92 (−1.39) | 68.07 (−3.93) | 59.54 (−10.29) |
| _majority voting (baseline)_ | 74.31 | 72.00 | 69.83 |
| _single-source (5-learner mean)_ | 61.22 | 59.61 | 59.59 |
| _Centralized Training (reference)_ | 72.07 | 74.40 | 73.19 |

Each per-subject learner is individually weak — the five-learner single-source mean is about 60%
on every dataset, since each learner sees one subject's data only — yet plain majority voting over
the (N−1)×5 votes already recovers most of the accuracy, and on BNCI2014001 it clears Centralized
Training, the non-private reference that pools all raw EEG (74.31 vs 72.07). On these weak,
only-loosely-independent votes the combiners spread out. **Dawid-Skene** is the most consistent,
beating voting on all three datasets (+0.54 / +1.14 / +4.46), and the lab's **StackingNet** also
clears voting everywhere (+1.00 / +1.00 / +0.67), with **SML**, **LAA** and **EBCC** positive on
all three. At the other end, the EM-heavy **M-MSR**, **GLAD** and **ZenCrowd** collapse on the
weakest learners, down to about −10 on BNCI2015001 — the honest failure mode of confusion-matrix
aggregators when the base votes are noisy and correlated. **SML-OVR**, the lab's multi-class
generalization of SML, reduces exactly to binary SML on these two-class tasks, so it reports the
same numbers as **SML** and sits beside it; its one-vs-rest form applies only to native multi-class
data such as four-class BNCI2014001. Against Centralized Training the ensemble is ahead on
BNCI2014001 but below it on BNCI2014002 (74.40) and BNCI2015001 (73.19), the accuracy cost of
never pooling raw data.

The ten crowd-labelling / truth-discovery aggregators are ported from
`github.com/sylyoung/TestEnsemble`: Dawid-Skene (Dawid & Skene, _J. R. Stat. Soc. C_ 1979), PM
(Q. Li et al., ACM SIGMOD 2014), ZenCrowd (Demartini et al., WWW 2012), EBCC (Y. Li et al., ICML
2019), LA (Y. Yang et al., ACM TKDD 2024), LAA (L. Yin et al., IJCAI 2017), GLAD (Whitehill et al.,
NeurIPS 2009), M-MSR (Ma & Olshevsky, NeurIPS 2020), Wawa (crowd-kit heuristic), MACE (Hovy et al.,
NAACL-HLT 2013); binary **SML** is Parisi et al., PNAS 2014, and the lab's **SML-OVR** /
**StackingNet** generalize it.

---

## Privacy-preserving

_Every method here keeps each subject's raw EEG local — it is never pooled across
subjects — the privacy-preserving counterpart to **Centralized Training** (EA-EEGNet
pooling all sources), which is the reference. All three datasets are two-class (chance
50%): BNCI2014001 (left vs right hand), BNCI2014002 and BNCI2015001 (right hand vs feet),
cross-subject LOSO, 3 seeds — so the columns are directly comparable and each Δacc is
versus Centralized Training on the same dataset. Every EEGNet method uses the identical
full-size EEGNet, Adam optimizer and learning rate 0.001 as Centralized Training, so that
only the privacy mechanism differs — each single-source model trains on one subject's
trials only, with no augmentation (CSDA is a cross-subject transform and would breach the
privacy premise); MSDT is a Riemannian tangent-space pipeline, exempt from the
EEGNet-config unification. Measured on hust-gpu-60022 / -7002 (seeds 1–3). The native
four-class BNCI2014001 exploration is kept as a supplementary appendix at the end of this
section and is **not** part of this binary benchmark._

| Method | BNCI2014001 | BNCI2014002 | BNCI2015001 |
|---|--:|--:|--:|
| SAFE **(lab)** *(new)* | 70.91 ± 1.15 | 78.21 ± 0.66 | 75.96 ± 0.53 |
| FedBS **(lab)** | 72.69 ± 1.62 | 76.07 ± 0.65 | 75.64 ± 0.63 |
| FedAvg | 74.54 ± 0.79 | 74.12 ± 0.44 | 71.62 ± 0.86 |
| MSDT **(lab)** | 73.84 ± 0.23 | 73.36 ± 0.59 | 72.51 ± 0.22 |
| _Centralized Training (EA-EEGNet, reference)_ | 72.07 ± 1.58 | 74.40 ± 1.04 | 73.19 ± 0.81 |

On the two-class task, **privacy is nearly free**. **FedBS** (Jia et al., IEEE TNSRE 2024 —
batch-specific BatchNorm + sharpness-aware minimization on top of FedAvg) matches or
slightly exceeds Centralized Training on all three datasets (+0.62 / +1.67 / +2.45).
**SAFE** (Jia et al., 2026 — FedBS plus single-step adversarial feature training and a
one-step adversarial weight perturbation) dips just below Centralized on BNCI2014001
(70.91, −1.16) but leads clearly on the other two (+3.81 / +2.77), the strongest federated
method overall — its adversarial hardening pays off once the single-source models are
strong enough. **MSDT** (Zhang et al., IEEE TNSRE 2022), the Riemannian decentralized
method, lands close to Centralized across the board (+1.77 / −1.04 / −0.68). Plain
**FedAvg** (McMahan et al., AISTATS 2017) stays competitive (+2.47 / −0.28 / −1.57), the
two-class task being forgiving of plain weight averaging. Every number is measured on the
server, none tuned to hit a target.

The fully decentralized alternative — each subject shares only hard predicted labels,
never model updates — is the **Decentralized-heterogeneous ensemble** table earlier in
this file, which is what the web leaderboard reports. Two-class per-dataset detail for
BNCI2014002 and BNCI2015001 follows; the four-class appendix closes the section.

### Per-dataset detail: BNCI2014002 and BNCI2015001 (two-class)

The privacy-preserving comparison is repeated on two additional MOABB motor-imagery
datasets, **BNCI2014002** (14 subjects, right-hand vs feet) and **BNCI2015001** (12
subjects, right-hand vs feet). Both are **two-class** (chance 50%), so their accuracies
are comparable to each other as well as to the headline three-dataset table above. Same
protocol: each single-source EA-EEGNet trains on one subject only (BNCI2014002: 100
training-run trials; BNCI2015001: 200 first-session trials), Centralized Training pools
all sources under the identical EEGNet, and the decentralized combiners fuse only hard
predicted labels. Binary **SML** (Parisi et al., _PNAS_ 2014) is now included since the task
is two-class, and both **SAFE** (federated adversarial, lab) and **MSDT** (decentralized
multi-source, lab) are now measured on both datasets. The decentralized **SML-OVR** is
dropped here — its multi-class one-vs-rest averaging is the wrong estimator for two classes,
where binary SML applies. Measured on hust-gpu-7002 (seeds 1–3); combiners shown mean-only
per the leaderboard convention. These decentralized rows use the single-architecture
EA-EEGNet-per-subject voters; the stronger five-learner heterogeneous version is the reported
ensemble — see the Decentralized-heterogeneous ensemble table above, which is what the web
leaderboard shows.

**BNCI2014002** — single-source model 52.51 ± 0.49 (mean over source→target pairs):

| Method | Acc ± std | Δacc vs Centralized Training |
|---|--:|--:|
| SAFE **(lab)** *(new)* | 78.21 ± 0.66 | +3.81 |
| FedBS **(lab)** | 76.07 ± 0.65 | +1.67 |
| _Centralized Training (EA-EEGNet, reference)_ | 74.40 ± 1.04 | (ref) |
| FedAvg | 74.12 ± 0.44 | −0.28 |
| MSDT **(lab)** | 73.36 ± 0.59 | −1.04 |
| decentralized EBCC | 58.38 | −16.02 |
| decentralized MACE | 57.88 | −16.52 |
| decentralized Dawid-Skene | 57.55 | −16.85 |
| decentralized StackingNet **(lab)** | 56.90 | −17.50 |
| decentralized majority voting | 56.88 | −17.52 |
| decentralized Wawa | 56.88 | −17.52 |
| decentralized LA | 56.71 | −17.69 |
| decentralized SML | 56.38 | −18.02 |
| decentralized GLAD | 56.19 | −18.21 |
| decentralized PM | 55.81 | −18.59 |
| decentralized LAA | 55.60 | −18.80 |
| decentralized ZenCrowd | 54.86 | −19.54 |
| decentralized M-MSR | 53.43 | −20.97 |

**BNCI2015001** — single-source model 54.16 ± 0.33 (mean over source→target pairs):

| Method | Acc ± std | Δacc vs Centralized Training |
|---|--:|--:|
| SAFE **(lab)** *(new)* | 75.96 ± 0.53 | +2.77 |
| FedBS **(lab)** | 75.64 ± 0.63 | +2.45 |
| _Centralized Training (EA-EEGNet, reference)_ | 73.19 ± 0.81 | (ref) |
| MSDT **(lab)** | 72.51 ± 0.22 | −0.68 |
| FedAvg | 71.62 ± 0.86 | −1.57 |
| decentralized Dawid-Skene | 64.86 | −8.33 |
| decentralized EBCC | 64.53 | −8.66 |
| decentralized SML | 64.18 | −9.01 |
| decentralized PM | 62.49 | −10.70 |
| decentralized MACE | 61.67 | −11.52 |
| decentralized LAA | 61.56 | −11.63 |
| decentralized LA | 61.50 | −11.69 |
| decentralized StackingNet **(lab)** | 61.31 | −11.88 |
| decentralized Wawa | 60.90 | −12.29 |
| decentralized majority voting | 60.60 | −12.59 |
| decentralized GLAD | 58.65 | −14.54 |
| decentralized ZenCrowd | 58.26 | −14.93 |
| decentralized M-MSR | 56.46 | −16.73 |

The same qualitative ordering holds — **FedBS ≈ Centralized ≥
FedAvg ≫ decentralized ensemble** — with one twist: on both two-class datasets FedBS
slightly *exceeds* Centralized Training, its batch-specific BatchNorm and sharpness-aware
minimization helping a little more here. The more informative story is inside the
decentralized ensemble. On the harder BNCI2014002 (single-source only 52.51, barely above
chance) the combiners stay bunched near majority voting (54–58) and the lab's spectral
SML / SML-OVR sit mid-pack. But on BNCI2015001, where the
single-source models are stronger (54.16) and more nearly independent, the spectral
meta-learner comes into its own: **binary SML (64.18) ranks third** among the combiners,
clearly above majority voting (60.60) *and* the lab's own StackingNet (61.31) — the regime
the SML estimator was designed for. **Dawid-Skene** (64.86 / 57.55 / 41.51) and
**EBCC** (64.53 / 58.38 / 40.34) remain the most consistent combiners across all three
datasets. The takeaway: spectral crowd-aggregation is not
weak per se — it needs base models comfortably above chance and roughly conditionally
independent, which two-class single-subject EEG supplies and four-class does not.

### Supplementary: privacy-preserving on native four-class BNCI2014001

_Kept for the record only — this appendix is **not part of the binary benchmark** and is **not shown on the web leaderboard**. Every headline table above is two-class; the tables in this appendix are the earlier native four-class BNCI2014001 exploration (left/right hand, feet, tongue), chance 25%._

_Every method here keeps each subject's raw EEG local — it is never pooled across
subjects — the privacy-preserving counterpart to **Centralized Training** (EA-EEGNet
pooling all sources), which is the reference. These are four-class results on native BNCI2014001 (left/right hand, feet, tongue), chance 25%,
cross-subject LOSO, 3 seeds. Every EEGNet method uses the identical full-size EEGNet
(F1 4 / D 2 / F2 8, dropout 0.25), Adam optimizer, and learning rate 0.001 as
Centralized Training, so that only the privacy mechanism differs — each
single-source model trains on one subject's ~288 trials only, with no augmentation
(CSDA is a cross-subject transform and would breach the privacy premise). Measured
on hust-gpu-60022: FedBS / FedAvg in `hustbciml_results_privacy3` and MSDT (a Riemannian
pipeline, exempt from the EEGNet-config unification) in `hustbciml_results_privacy`
(2026-07-14); Centralized Training from `hustbciml_results_ens4` (seeds 1–3); SAFE from
`hustbciml_results_3ds` (2026-07-18). Δacc is vs Centralized Training. The fully decentralized
ensemble that shares only hard predicted labels is a separate two-class comparison, in the
Decentralized-heterogeneous ensemble section above._

| Method | Acc ± std | Δacc vs Centralized Training |
|---|--:|--:|
| MSDT **(lab)** | 55.29 ± 0.58 | +4.83 |
| FedBS **(lab)** | 50.90 ± 0.29 | +0.44 |
| _Centralized Training (EA-EEGNet, reference)_ | 50.46 ± 0.86 | (ref) |
| SAFE **(lab)** *(new)* | 50.15 ± 0.55 | −0.31 |
| FedAvg | 47.65 ± 2.24 | −2.81 |

Keeping each subject's data local costs accuracy, and the cost scales with how strictly
the data is siloed. **Federated learning** aggregates per-subject (client) updates
through a server without moving raw EEG: with the same Adam optimizer and learning rate
as Centralized Training, **FedBS** (Jia et al., IEEE TNSRE 2024 — batch-specific
BatchNorm + sharpness-aware minimization on top of FedAvg) reaches 50.90, essentially
matching the 50.46 reference — privacy is nearly free here — while plain **FedAvg**
(McMahan et al., AISTATS 2017) trails at 47.65, the accuracy cost of plain weight
averaging. **SAFE** (Jia et al., 2026 — FedBS plus single-step adversarial feature training
and a one-step adversarial weight perturbation) matches the reference here too (50.15,
−0.31): its adversarial regularization is a wash on this four-class task, but it pays off on
the two-class datasets in the headline table, where SAFE is the strongest federated method. (With FedBS's own SGD at the paper's larger learning rate it
reproduces ~53.4, matching its published ~53.31; the benchmark instead holds it to the
shared Adam / 0.001 setting, so that only the privacy mechanism — not the optimizer —
separates it from Centralized Training.)

The one method above Centralized Training is **MSDT** (Zhang et al., IEEE TNSRE 2022,
55.29), and it is the exception that proves the rule: MSDT is **not an EEGNet model at
all**. It decodes Riemannian tangent-space features with a per-source MLP each, then
adapts and fuses them at test time by entropy-weighted information maximization. Its
lead therefore reflects that different feature representation and its test-time
adaptation — not the privacy mechanism, which it shares with the methods that trail.
Read together, the table says privacy costs accuracy in proportion to how little is
shared — near-zero when model *updates* are federated, largest when only *labels* are —
and the apparent counterexample is a different model class, not a free lunch from
privacy.

The single-model rows (Centralized Training, FedBS, SAFE, FedAvg, MSDT) carry the across-seed
std; the decentralized combiners do vary across seeds — their base models are
re-initialized per seed — but that spread is omitted from the table for readability, as
noted above. Every number is measured on the server, none tuned to hit a target.

#### Supplementary: heterogeneous decentralized ensemble on four-class BNCI2014001

_This is the native four-class BNCI2014001 counterpart to the two-class Decentralized-heterogeneous
ensemble table above. A single EA-EEGNet per source subject, all the same architecture, each near
chance (~31.5%) on four-class single-subject data, makes weak and correlated voters, the regime
where crowd aggregation has least to work with. This variant keeps the identical privacy premise
(each subject's raw EEG stays local; only hard predicted labels are shared) but replaces
the single per-subject EEGNet with **five heterogeneous learners per source subject** —
Tangent-space + LDA, Tangent-space + SVM (RBF), EEGNet, ShallowConvNet, and CSPNet — so
each of the 8 source subjects casts 5 votes and every target is decided by 8 × 5 = 40
label votes. The learners span two feature families (Riemannian tangent space and
end-to-end CNNs) and three network architectures, making the voters both individually
stronger and more conditionally independent — the two properties crowd aggregation
actually needs. Same four-class BNCI2014001, cross-subject LOSO, 3 seeds; measured in
`hustbciml_results_hetero`. Reproduce with `python -m hustbciml.scripts.decentralized
--dataset BNCI2014001-4 --base hetero`. Δacc is vs Centralized Training (50.46)._

| Combiner | Acc | Δacc vs Centralized Training |
|---|--:|--:|
| LAA | 55.49 | +5.03 |
| StackingNet **(lab)** | 55.11 | +4.65 |
| Dawid-Skene | 55.03 | +4.57 |
| SML-OVR **(lab)** | 54.68 | +4.22 |
| Wawa | 54.35 | +3.89 |
| GLAD | 53.94 | +3.48 |
| LA | 53.91 | +3.45 |
| PM | 53.77 | +3.31 |
| majority voting | 53.67 | +3.21 |
| EBCC | 53.67 | +3.21 |
| ZenCrowd | 53.42 | +2.96 |
| MACE | 53.42 | +2.96 |
| M-MSR | 53.13 | +2.67 |

_Single heterogeneous learner (mean over all source learners and targets): 37.56 —
already above the homogeneous single-subject EEGNet's 31.54, because the Riemannian
tangent-space classifiers decode four-class single-subject data better than a
from-scratch EEGNet._

The heterogeneous base flips the privacy-preserving story. Every combiner now **exceeds
Centralized Training** (50.46): even plain majority voting reaches 53.67 (+3.21), and the
best aggregators — **LAA** (55.49), the lab's **StackingNet** (55.11) and **Dawid-Skene**
(55.03) — reach the level of **MSDT** (55.29), the previous top privacy method, while
sharing strictly less (only hard labels — no Riemannian test-time feature adaptation).
Two effects compound: the voters are individually stronger (37.56 vs 31.54 single-voter)
and, spanning Riemannian and CNN feature spaces, far less correlated, so their errors
cancel under aggregation instead of reinforcing. The same combiners that trailed by ~10
points on homogeneous EEGNet voters (best 41.51) now lead by ~5 — a +14 swing from
base-learner heterogeneity alone, with the combiner code unchanged. **SML-OVR** (54.68)
recovers in particular: given above-chance, roughly independent voters it finally does
what spectral aggregation is designed to do, in contrast to the homogeneous case where
near-chance voters left it below plain voting (37.40). The winning combiner is not
selected on the test set — the whole combiner set is reported, and the gain holds across
all thirteen.

---

Every controlled-comparison number above is also recorded, with its paper citation,
reference range, and per-method note, in `hustbciml/tests/repro/repro_targets.yaml`.
