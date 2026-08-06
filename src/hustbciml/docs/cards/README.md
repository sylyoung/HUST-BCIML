# Algorithm cards

One card per benchmarked method — every key on the public leaderboard except the ensemble combiners, which are post-hoc aggregators over several trained models rather than pipeline compositions and are documented on the leaderboard row itself. `tests/repro/test_repro_targets.py` fails if a key is missing. Generated from the reproduction registry. Measured cards show their recorded multi-seed mean ± standard deviation on **BNCI2014001**. The Network rows come from the validated five-seed nested-LOSO campaign; MVCNet retains its documented three-seed legacy values. Single-axis measured methods show Δ against that axis's baseline; classical methods show Δ against the EA-EEGNet reference.

See [../../RESULTS.md](../../RESULTS.md) for the controlled-comparison tables, [../glossary.md](../glossary.md) for terms, and [../porting_guide.md](../porting_guide.md) to add a method.

## Canonical reference

| Method | Acc ± std | Δ | Mechanism |
|---|--:|--:|---|
| [EA-EEGNet](EA-EEGNet.md) | 72.07 ± 1.58 | base | Euclidean Alignment recenters each subject's trials by the inverse square root of their mean spatial covariance, so every subject… |

## Network (backbone)

| Method | Acc ± std | Δ | Mechanism |
|---|--:|--:|---|
| [EA-MSCFormer](EA-MSCFormer.md) | 76.11 ± 1.53 | +3.35 | Three parallel multi-scale temporal-convolution branches whose features are fused and passed to a transformer encoder. |
| [EA-DBConformer](EA-DBConformer.md) | 75.76 ± 1.47 | +3.00 | A dual-branch convolutional transformer. |
| [MVCNet](MVCNet.md) | 75.75 ± 0.56 | +2.99 | Multi-View Contrastive Network is a dual-branch network: an IFNet backbone (an interactive frequency-domain CNN) trained with mul… |
| [CSP-Net](CSP-Net.md) | 75.54 ± 0.84 | +2.78 | A standard EEGNet whose depthwise spatial convolution is initialized with Common Spatial Pattern filters estimated from the EA-al… |
| [EA-EEGNeX](EA-EEGNeX.md) | 75.28 ± 1.15 | +2.52 | A purely convolutional EEGNet variant that replaces the separable temporal convolutions with a stack of dilated convolutions for… |
| [EA-MSVTNet](EA-MSVTNet.md) | 74.41 ± 1.34 | +1.65 | Several parallel multi-scale EEGNet-style convolution branches followed by a transformer that mixes their tokens. |
| [EA-EEGConformer](EA-EEGConformer.md) | 74.04 ± 0.74 | +1.28 | EEG Conformer is a convolutional tokenizer (temporal then spatial convolution producing patch tokens) feeding a transformer self-… |
| [EA-Deep4Net-AT](EA-Deep4Net-AT.md) | 73.94 ± 1.59 | +1.18 | Four temporal/spatial convolution-and-pooling blocks use width-10 temporal kernels, 3/3 pooling, and 25→50→100→200 feature maps b… |
| [EA-ShallowFBCSPNet-AT](EA-ShallowFBCSPNet-AT.md) | 73.84 ± 0.39 | +1.08 | A temporal convolution and whole-channel spatial convolution are followed by square, average pooling, safe logarithm, and dropout… |
| [EA-ADFCNN-Transpose-AT](EA-ADFCNN-Transpose-AT.md) | 73.64 ± 0.78 | +0.88 | Two spectral-spatial pathways at different temporal scales are fused by self-attention; the corrected attention output is transpo… |
| [EA-TMSANet](EA-TMSANet.md) | 73.47 ± 0.99 | +0.71 | Sums two parallel multi-scale temporal convolutions, then applies a temporal multi-scale self-attention module. |
| [EA-TIEEEGNet](EA-TIEEEGNet.md) | 72.95 ± 1.17 | +0.19 | TIE-EEGNet is EEGNet with its first temporal convolution replaced by a time-information-enhanced (TIE) convolution: a fixed sinus… |
| [EA-EEGNet-Nested](EA-EEGNet-Nested.md) | 72.76 ± 1.14 | base | A compact temporal convolution, depthwise spatial convolution, and separable convolution produce features for the shared Linear h… |
| [EA-CTNet](EA-CTNet.md) | 72.70 ± 0.93 | -0.06 | An EEGNet-style convolutional patch embedding feeding a transformer encoder. |
| [EA-FBMSNet-8-32-AT](EA-FBMSNet-8-32-AT.md) | 71.93 ± 0.48 | -0.83 | Six causal sub-band views feed shared mixed-scale temporal and depthwise spatial filters; four remainder-preserving temporal log-… |
| [EA-KDFNet](EA-KDFNet.md) | 71.91 ± 1.26 | -0.85 | KDFNet (knowledge-data fusion network) mirrors the FBCSP pipeline inside a CNN: a windowed-sinc FIR filter bank supplies fixed, d… |
| [EA-EEGDeformer](EA-EEGDeformer.md) | 70.96 ± 1.82 | -1.80 | A dense convolutional transformer that interleaves shallow CNN encoders with coarse-to-fine transformer stages. |
| [EA-SlimSeiz](EA-SlimSeiz.md) | 67.75 ± 1.14 | -5.01 | A lightweight multi-branch 1D-convolution feature extractor paired with a single Mamba selective-state-space mixer; originally a… |
| [EA-EEGWaveNet-Release-AT](EA-EEGWaveNet-Release-AT.md) | 61.53 ± 3.30 | -11.23 | Five retained scales from a depthwise downsampling cascade are processed by two-convolution feature blocks, pooled, concatenated,… |

## Alignment

| Method | Acc ± std | Δ | Mechanism |
|---|--:|--:|---|
| [RA-EEGNet](RA-EEGNet.md) | 73.69 ± 1.09 | +4.35 | Riemannian Alignment recenters each subject by the affine-invariant (Fréchet) geometric mean of their trial covariances instead o… |
| [NoAlign-EEGNet](NoAlign-EEGNet.md) | 69.34 ± 0.65 | control | The no-alignment control: identical to the canonical composition but with the aligner replaced by Identity, so EEGNet sees raw pe… |

## Transfer / adaptation strategy

| Method | Acc ± std | Δ | Mechanism |
|---|--:|--:|---|
| [MCC](MCC.md) | 79.04 ± 0.67 | +6.97 | Minimum Class Confusion adds a loss that minimizes the off-diagonal class-confusion of the temperature-rescaled prediction correl… |
| [CDAN](CDAN.md) | 76.26 ± 0.94 | +4.19 | Conditional Domain-Adversarial Network is DANN with the domain discriminator conditioned on the multilinear (outer-product) combi… |
| [T-TIME](T-TIME.md) | 76.05 ± 0.42 | +3.99 | Test-Time Information Maximization for online motor imagery: the source-trained model adapts on the streaming target by minimizin… |
| [DELTA](DELTA.md) | 75.93 ± 0.44 | +3.86 | DELTA performs entropy minimization with a class-imbalance-corrected diversity term (dynamic online reweighting of classes), whic… |
| [ISFDA](ISFDA.md) | 75.80 ± 0.54 | +3.73 | Imbalanced Source-Free Domain Adaptation combines temperature-scaled information maximization with intra-class feature tightening… |
| [JAN](JAN.md) | 75.44 ± 0.41 | +3.37 | Joint Adaptation Network extends the MMD idea by aligning the joint distribution across multiple layers (feature and softmax) wit… |
| [MDMAML](MDMAML.md) | 75.13 ± 0.38 | +3.06 | MDMAML meta-learns an EEGNet initialization across the source subjects with domain-paired first-order model-agnostic meta-learnin… |
| [DAN](DAN.md) | 75.03 ± 1.04 | +2.96 | Deep Adaptation Network matches source and target feature distributions by minimizing a multi-kernel Maximum Mean Discrepancy (MK… |
| [SAR](SAR.md) | 74.90 ± 1.99 | +2.83 | Sharpness-Aware and Reliable test-time adaptation minimizes temperature-scaled prediction entropy over all parameters using a Sha… |
| [EA-DANN](EA-DANN.md) | 74.77 ± 1.01 | +2.70 | Domain-Adversarial Neural Network trains a domain classifier to separate source from unlabeled-target features while a gradient-r… |
| [FedAvg](FedAvg.md) | 74.54 ± 0.79 |  | Federated averaging: each subject trains locally and the server averages the model weights. |
| [PL](PL.md) | 74.38 ± 1.89 | +2.31 | Pseudo-Label self-training at test time turns the model's own argmax predictions on target trials into training targets for a few… |
| [ABAT](ABAT.md) | 74.20 ± 0.69 | +2.13 | Alignment-Based Adversarial Training, after a clean warmup, perturbs each batch with channel-standard-deviation-scaled projected… |
| [SHOT](SHOT.md) | 74.20 ± 1.06 | +2.13 | Source Hypothesis Transfer is source-free: the source classifier head is frozen and only the feature extractor is adapted on the… |
| [MDD](MDD.md) | 74.18 ± 0.25 | +2.11 | Margin Disparity Discrepancy bounds the target error by a margin-based disparity between the main classifier and an adversarial a… |
| [BFT](BFT.md) | 73.79 ± 0.67 | +1.72 | Backpropagation-Free Transformations (the BFT-A variant) pass each trial through K label-preserving transforms at test time and a… |
| [PAT](PAT.md) | 73.53 ± 0.95 | +1.46 | Privacy-preserving Adversarial Transfer extends ABAT: after a clean warmup, each Euclidean-aligned batch is first amplitude-scale… |
| [ASFA](ASFA.md) | 73.28 ± 0.51 | +1.21 | ASFA is source-free adaptation: after ERM source training the classifier head is frozen and only the feature extractor is adapted… |
| [BN-adapt](BN-adapt.md) | 73.23 ± 1.29 | +1.16 | BatchNorm adaptation re-estimates the BatchNorm statistics from the target stream — no gradient step and no parameter update — co… |
| [DJP-MMD](DJP-MMD.md) | 73.10 ± 0.64 | +1.03 | Discriminative Joint Probability Maximum Mean Discrepancy adds a discrepancy over the joint P(X, Y) — using source labels and tar… |
| [FedBS](FedBS.md) | 72.69 ± 1.62 |  | Federated learning with batch-specific BatchNorm and sharpness-aware minimization, aggregating per-subject model updates through… |
| [Tent](Tent.md) | 72.04 ± 1.42 | -0.03 | Test-time entropy minimization updates only the BatchNorm affine parameters on the target stream to minimize prediction entropy,… |
| [SAFE](SAFE.md) | 70.91 ± 1.15 |  | Federated learning that adds single-step adversarial feature training and a one-step adversarial weight perturbation on top of ba… |

## Augmentation

| Method | Acc ± std | Δ | Mechanism |
|---|--:|--:|---|
| [FShift-EEGNet](FShift-EEGNet.md) | 73.28 ± 0.51 |  | Translates a trial's whole spectrum by a small frequency offset using the analytic (Hilbert) signal. |
| [FSurr-EEGNet](FSurr-EEGNet.md) | 73.28 ± 1.25 |  | Draws a surrogate trial with the same power spectrum as the original but randomized Fourier phase. |
| [CR-EEGNet](CR-EEGNet.md) | 73.23 ± 0.74 | +3.88 | Channel Reflection augmentation mirrors each trial across the sagittal midline (a left/right electrode swap) and swaps the left/r… |
| [Scale-EEGNet](Scale-EEGNet.md) | 72.97 ± 0.83 |  | Copies each trial with its amplitude multiplied by a coefficient close to one; the augmentation half of the PAT pipeline. |
| [CSDA-EEGNet](CSDA-EEGNet.md) | 72.45 ± 1.87 | +0.38 | Cross-Subject Detail-swap Augmentation applies a db4 discrete wavelet transform to split each EA-aligned trial into approximation… |
| [FComb-EEGNet](FComb-EEGNet.md) | 72.30 ± 2.11 |  | Splits each trial's cosine spectrum into contiguous bands and rebuilds a new trial by taking each band from a different same-clas… |
| [Noise-EEGNet](Noise-EEGNet.md) | 71.94 ± 0.66 |  | Copies each trial once with zero-mean Gaussian noise added, scaled to the trial's own amplitude; the simplest label-preserving au… |
| [Flip-EEGNet](Flip-EEGNet.md) | 70.60 ± 0.77 |  | Mirrors each channel vertically about its own maximum, adding one label-preserving copy per trial. |
| [HS-EEGNet](HS-EEGNet.md) | 64.99 ± 0.61 |  | Splices the left- and right-hemisphere channels from two same-class trials into a new trial, exploiting motor-imagery lateralizat… |

## Classical (network-free)

| Method | Acc ± std | Δ | Mechanism |
|---|--:|--:|---|
| [MEKT](MEKT.md) | 76.54 ± 0.00 | +4.47 vs ref | Manifold Embedded Knowledge Transfer is a network-free classical transfer method. |
| [LSFT](LSFT.md) | 74.77 ± 0.00 | +2.70 vs ref | Lightweight Source-Free Transfer keeps no raw source data at transfer time: pretrained source classifiers vote to pseudo-label th… |
| [MSDT](MSDT.md) | 73.84 ± 0.23 | +1.77 vs ref | Multi-Source Decentralized Transfer trains one small MLP per source subject on Riemannian tangent-space features (decentralized —… |
| [CSP-LDA](CSP-LDA.md) | 73.77 ± 0.00 | +1.70 vs ref | Common Spatial Patterns plus Linear Discriminant Analysis. |
| [Riemann-MDM](Riemann-MDM.md) | 71.68 ± 0.00 | -0.39 vs ref | Minimum Distance to Riemannian Mean represents each trial by its spatial covariance matrix and classifies by the smallest affine-… |

---
_Generated by `scripts/build_cards.py` — 59 methods._
