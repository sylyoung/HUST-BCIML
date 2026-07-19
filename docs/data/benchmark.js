window.BENCHMARK = {
  "meta": {
    "dataset": "BNCI2014001",
    "protocol": "cross-subject leave-one-subject-out",
    "subjects": 9,
    "classes": 2,
    "chance": 50.0,
    "datasets": [
      {
        "name": "BNCI2014001",
        "subjects": 9,
        "channels": 22,
        "sfreq": 250,
        "classes": "2-class",
        "chance": "50%",
        "trials": "288 / session",
        "role": "Left vs right hand (two-class, chance 50%) for every table, including the privacy-preserving and ensemble families. The native dataset is four-class (both hands, feet, tongue); the benchmark uses its two-class left/right subset throughout, and the four-class variant stays available in code."
      },
      {
        "name": "BNCI2014002",
        "subjects": 14,
        "channels": 15,
        "sfreq": 512,
        "classes": "2-class",
        "chance": "50%",
        "trials": 100,
        "role": "Right hand vs feet, 14 subjects, 100 training-run trials each. Two-class (chance 50%) throughout."
      },
      {
        "name": "BNCI2015001",
        "subjects": 12,
        "channels": 13,
        "sfreq": 512,
        "classes": "2-class",
        "chance": "50%",
        "trials": 200,
        "role": "Right hand vs feet, 12 subjects, 200 first-session trials each. Two-class (chance 50%) throughout."
      }
    ]
  },
  "library": {
    "title": "A unified, reproducible EEG-decoding benchmark",
    "tagline": "An algorithm is a named composition of plug-in stages — an aligner, an augmenter and a backbone, trained under a chosen learning objective, with an optional ensemble that aggregates several models. Every controlled comparison varies one stage while the rest stay fixed.",
    "pipeline": [
      "Aligner",
      "Augmenter",
      "Backbone"
    ],
    "driver": "Learning objective"
  },
  "datasets": [
    "BNCI2014001",
    "BNCI2014002",
    "BNCI2015001"
  ],
  "tables": [
    {
      "id": "alignment",
      "title": "Data Alignment",
      "blurb": "The aligner stage. Each aligner normalizes a subject's trials into a common statistical frame before the backbone sees them, shrinking the between-subject covariance shift that otherwise dominates cross-subject decoding. It is label-free and applied per subject; the backbone and its training stay fixed, and the baseline applies no alignment.",
      "references": null,
      "context": null,
      "groups": [
        {
          "subcat": null,
          "blurb": "",
          "baseline": "none",
          "reference": null,
          "rows": [
            {
              "name": "EA (Euclidean)",
              "acc": {
                "BNCI2014001": {
                  "mean": 72.07,
                  "std": 1.58
                },
                "BNCI2014002": {
                  "mean": 74.4,
                  "std": 1.04
                },
                "BNCI2015001": {
                  "mean": 73.19,
                  "std": 0.81
                }
              },
              "delta": {
                "BNCI2014001": 2.73,
                "BNCI2014002": 12.5,
                "BNCI2015001": 9.73
              },
              "isBaseline": false,
              "isReference": false,
              "key": "EA-EEGNet",
              "lab": true,
              "code": "hustbciml/algorithms/aligners/EA.py",
              "desc": "Whitens each subject's trials by the inverse square root of their mean spatial covariance, so every subject's average covariance becomes the identity. The benchmark's default aligner.",
              "ref": "H. He, D. Wu*, IEEE Trans. Biomed. Eng., 2020",
              "doi": "10.1109/TBME.2019.2913914",
              "pinAfter": null
            },
            {
              "name": "RA (Riemannian)",
              "acc": {
                "BNCI2014001": {
                  "mean": 73.97,
                  "std": 1.27
                },
                "BNCI2014002": {
                  "mean": 71.86,
                  "std": 1.23
                },
                "BNCI2015001": {
                  "mean": 72.39,
                  "std": 0.32
                }
              },
              "delta": {
                "BNCI2014001": 4.63,
                "BNCI2014002": 9.96,
                "BNCI2015001": 8.93
              },
              "isBaseline": false,
              "isReference": false,
              "key": "RA-EEGNet",
              "lab": false,
              "code": "hustbciml/algorithms/aligners/RA.py",
              "desc": "Normalizes each subject's trials against the affine-invariant Riemannian (Fréchet) mean of their spatial covariances — recentring in the curved covariance geometry rather than the Euclidean one.",
              "ref": "P. Zanini et al., IEEE Trans. Biomed. Eng., 2018",
              "doi": "10.1109/TBME.2017.2742541",
              "pinAfter": null
            },
            {
              "name": "none",
              "acc": {
                "BNCI2014001": {
                  "mean": 69.34,
                  "std": 0.65
                },
                "BNCI2014002": {
                  "mean": 61.9,
                  "std": 2.96
                },
                "BNCI2015001": {
                  "mean": 63.46,
                  "std": 0.83
                }
              },
              "delta": {
                "BNCI2014001": null,
                "BNCI2014002": null,
                "BNCI2015001": null
              },
              "isBaseline": true,
              "isReference": false,
              "key": "NoAlign-EEGNet",
              "lab": false,
              "code": "hustbciml/algorithms/aligners/Identity.py",
              "desc": "No alignment; trials are fed to the backbone as recorded.",
              "ref": null,
              "doi": null,
              "pinAfter": null
            }
          ]
        }
      ]
    },
    {
      "id": "augmentation",
      "title": "Data Augmentation",
      "blurb": "The augmenter stage. Each augmenter synthesizes extra training trials to regularize the same backbone, which is otherwise trained identically; each augmenter is compared against that backbone trained without it. Channel Reflection is an electrode-space transform and must precede any spatial whitening, so it runs on unaligned trials; CSDA operates on the Euclidean-aligned trials.",
      "references": null,
      "context": null,
      "groups": [
        {
          "subcat": null,
          "blurb": "",
          "baseline": "none",
          "reference": null,
          "rows": [
            {
              "name": "CSDA",
              "acc": {
                "BNCI2014001": {
                  "mean": 72.74,
                  "std": 1.92
                },
                "BNCI2014002": {
                  "mean": 73.98,
                  "std": 0.32
                },
                "BNCI2015001": {
                  "mean": 73.53,
                  "std": 0.44
                }
              },
              "delta": {
                "BNCI2014001": 0.67,
                "BNCI2014002": -0.42,
                "BNCI2015001": 0.34
              },
              "isBaseline": false,
              "isReference": false,
              "key": "CSDA-EEGNet",
              "lab": true,
              "code": "hustbciml/algorithms/augmenters/CSDA.py",
              "desc": "Cross-subject wavelet detail-swap — mixes the high-frequency wavelet detail of same-class trials from different subjects to synthesize new trials.",
              "ref": "Z. Wang, ..., D. Wu*, Knowl.-Based Syst., 2025",
              "doi": "10.1016/j.knosys.2025.113074",
              "pinAfter": null
            },
            {
              "name": "Channel Reflection",
              "acc": {
                "BNCI2014001": {
                  "mean": 73.23,
                  "std": 0.74
                },
                "BNCI2014002": null,
                "BNCI2015001": null
              },
              "delta": {
                "BNCI2014001": 3.89,
                "BNCI2014002": null,
                "BNCI2015001": null
              },
              "isBaseline": false,
              "isReference": false,
              "key": "CR-EEGNet",
              "lab": true,
              "code": "hustbciml/algorithms/augmenters/ChannelReflection.py",
              "desc": "Mirrors each trial across the sagittal midline and swaps its left/right label, adding anatomically valid copies that double the training set for two-class left/right motor imagery.",
              "ref": "Z. Wang†, S. Li†, ..., D. Wu*, Neural Networks, 2024",
              "doi": "10.1016/j.neunet.2024.106351",
              "pinAfter": null
            },
            {
              "name": "none",
              "acc": {
                "BNCI2014001": {
                  "mean": 72.07,
                  "std": 1.58
                },
                "BNCI2014002": {
                  "mean": 74.4,
                  "std": 1.04
                },
                "BNCI2015001": {
                  "mean": 73.19,
                  "std": 0.81
                }
              },
              "delta": {
                "BNCI2014001": null,
                "BNCI2014002": null,
                "BNCI2015001": null
              },
              "isBaseline": true,
              "isReference": false,
              "key": "EA-EEGNet",
              "lab": false,
              "code": null,
              "desc": "EA-aligned EEGNet trained without augmentation — the baseline CSDA is measured against (aligned regime; Channel Reflection is measured against the unaligned baseline, since it must run before whitening).",
              "ref": null,
              "doi": null,
              "pinAfter": null
            }
          ]
        }
      ]
    },
    {
      "id": "network",
      "title": "Networks",
      "blurb": "The backbone stage. Vary the deep network on the same Euclidean-aligned trials, holding alignment and the learning objective fixed; each backbone is trained with its own learning rate and schedule, selected on held-out source-subject validation data. Baseline: EEGNet. MVCNet is a composite — an IFNet backbone trained with a multi-view contrastive objective — shown here among the backbones.",
      "references": null,
      "context": null,
      "groups": [
        {
          "subcat": null,
          "blurb": "",
          "baseline": "EEGNet",
          "reference": null,
          "rows": [
            {
              "name": "DBConformer",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.85,
                  "std": 0.98
                },
                "BNCI2014002": {
                  "mean": 77.05,
                  "std": 0.6
                },
                "BNCI2015001": {
                  "mean": 72.94,
                  "std": 0.84
                }
              },
              "delta": {
                "BNCI2014001": 2.32,
                "BNCI2014002": 2.65,
                "BNCI2015001": -0.45
              },
              "isBaseline": false,
              "isReference": false,
              "key": "EA-DBConformer",
              "lab": true,
              "code": "hustbciml/algorithms/models/DBConformer.py",
              "desc": "Dual-branch convolutional transformer with parallel temporal and spatial branches whose features are fused before classification.",
              "ref": "Z. Wang, ..., D. Wu*, IEEE J. Biomed. Health Inform., 2026",
              "doi": "10.1109/JBHI.2025.3622725",
              "pinAfter": null
            },
            {
              "name": "MVCNet",
              "acc": {
                "BNCI2014001": {
                  "mean": 75.64,
                  "std": 0.95
                },
                "BNCI2014002": {
                  "mean": 76.69,
                  "std": 0.94
                },
                "BNCI2015001": {
                  "mean": 72.21,
                  "std": 0.5
                }
              },
              "delta": {
                "BNCI2014001": 3.11,
                "BNCI2014002": 2.29,
                "BNCI2015001": -1.18
              },
              "isBaseline": false,
              "isReference": false,
              "key": "MVCNet",
              "lab": true,
              "code": "hustbciml/algorithms/strategies/MVCNet.py",
              "desc": "Composite of an IFNet backbone and a multi-view contrastive training objective; at inference only the backbone and linear head are used.",
              "ref": "Z. Wang, ..., D. Wu*, Knowl.-Based Syst., 2025",
              "doi": "10.1016/j.knosys.2025.114205",
              "pinAfter": null
            },
            {
              "name": "CSP-Net",
              "acc": {
                "BNCI2014001": {
                  "mean": 75.15,
                  "std": 1.06
                },
                "BNCI2014002": {
                  "mean": 74.4,
                  "std": 0.24
                },
                "BNCI2015001": {
                  "mean": 72.42,
                  "std": 0.38
                }
              },
              "delta": {
                "BNCI2014001": 2.62,
                "BNCI2014002": 0.0,
                "BNCI2015001": -0.97
              },
              "isBaseline": false,
              "isReference": false,
              "key": "CSP-Net",
              "lab": true,
              "code": "hustbciml/algorithms/models/CSPNet.py",
              "desc": "EEGNet whose depthwise spatial convolution is initialized with Common Spatial Pattern filters and then frozen.",
              "ref": "X. Jiang, ..., D. Wu*, Knowl.-Based Syst., 2024",
              "doi": "10.1016/j.knosys.2024.112668",
              "pinAfter": null
            },
            {
              "name": "TIE-EEGNet",
              "acc": {
                "BNCI2014001": {
                  "mean": 73.51,
                  "std": 0.25
                },
                "BNCI2014002": {
                  "mean": 73.17,
                  "std": 0.35
                },
                "BNCI2015001": {
                  "mean": 73.83,
                  "std": 0.38
                }
              },
              "delta": {
                "BNCI2014001": 0.98,
                "BNCI2014002": -1.23,
                "BNCI2015001": 0.44
              },
              "isBaseline": false,
              "isReference": false,
              "key": "EA-TIEEEGNet",
              "lab": true,
              "code": "hustbciml/algorithms/models/TIEEEGNet.py",
              "desc": "EEGNet whose first temporal convolution is replaced by a time-information-enhanced convolution that injects a fixed sinusoidal positional embedding into the signal. ⚠ Note: originally developed for seizure detection (Peng et al. 2022); this time-positional design targets seizure EEG and may not be well-suited to motor imagery.",
              "ref": "R. Peng, ..., D. Wu*, IEEE Trans. Neural Syst. Rehabil. Eng., 2022",
              "doi": "10.1109/TNSRE.2022.3204540",
              "pinAfter": null
            },
            {
              "name": "KDFNet",
              "acc": {
                "BNCI2014001": {
                  "mean": 70.88,
                  "std": 0.32
                },
                "BNCI2014002": {
                  "mean": 72.64,
                  "std": 0.69
                },
                "BNCI2015001": {
                  "mean": 68.65,
                  "std": 1.05
                }
              },
              "delta": {
                "BNCI2014001": -1.65,
                "BNCI2014002": -1.76,
                "BNCI2015001": -4.74
              },
              "isBaseline": false,
              "isReference": false,
              "key": "EA-KDFNet",
              "lab": true,
              "code": "hustbciml/algorithms/models/KDFNet.py",
              "desc": "Knowledge-data fusion CNN mirroring FBCSP — a windowed-sinc FIR filter bank and per-band CSP spatial filters are knowledge-initialized on the aligned source, then fine-tuned end-to-end.",
              "ref": "X. Jiang, ..., D. Wu*, Inf. Sci., 2026",
              "doi": "10.1016/j.ins.2025.123001",
              "pinAfter": null
            },
            {
              "name": "EEGConformer",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.05,
                  "std": 0.58
                },
                "BNCI2014002": {
                  "mean": 75.12,
                  "std": 1.0
                },
                "BNCI2015001": {
                  "mean": 73.07,
                  "std": 1.37
                }
              },
              "delta": {
                "BNCI2014001": 1.52,
                "BNCI2014002": 0.72,
                "BNCI2015001": -0.32
              },
              "isBaseline": false,
              "isReference": false,
              "key": "EA-EEGConformer",
              "lab": false,
              "code": "hustbciml/algorithms/models/EEGConformer.py",
              "desc": "Convolutional tokenizer followed by a transformer encoder.",
              "ref": "Y. Song et al., IEEE Trans. Neural Syst. Rehabil. Eng., 2023",
              "doi": "10.1109/TNSRE.2022.3230250",
              "pinAfter": null
            },
            {
              "name": "ShallowConvNet",
              "acc": {
                "BNCI2014001": {
                  "mean": 72.69,
                  "std": 0.71
                },
                "BNCI2014002": {
                  "mean": 71.14,
                  "std": 0.25
                },
                "BNCI2015001": {
                  "mean": 73.03,
                  "std": 0.28
                }
              },
              "delta": {
                "BNCI2014001": 0.16,
                "BNCI2014002": -3.26,
                "BNCI2015001": -0.36
              },
              "isBaseline": false,
              "isReference": false,
              "key": "EA-ShallowConvNet",
              "lab": false,
              "code": "hustbciml/algorithms/models/ShallowConvNet.py",
              "desc": "Shallow convolution-and-pooling network modelled on band-power features.",
              "ref": "R. T. Schirrmeister et al., Hum. Brain Mapp., 2017",
              "doi": "10.1002/hbm.23730",
              "pinAfter": null
            },
            {
              "name": "DeepConvNet",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.07,
                  "std": 1.04
                },
                "BNCI2014002": {
                  "mean": 68.64,
                  "std": 0.2
                },
                "BNCI2015001": {
                  "mean": 69.99,
                  "std": 1.07
                }
              },
              "delta": {
                "BNCI2014001": 1.54,
                "BNCI2014002": -5.76,
                "BNCI2015001": -3.4
              },
              "isBaseline": false,
              "isReference": false,
              "key": "EA-DeepConvNet",
              "lab": false,
              "code": "hustbciml/algorithms/models/DeepConvNet.py",
              "desc": "Deeper four-block convolutional network for EEG decoding.",
              "ref": "R. T. Schirrmeister et al., Hum. Brain Mapp., 2017",
              "doi": "10.1002/hbm.23730",
              "pinAfter": null
            },
            {
              "name": "EEGNet",
              "acc": {
                "BNCI2014001": {
                  "mean": 72.53,
                  "std": 1.22
                },
                "BNCI2014002": {
                  "mean": 74.4,
                  "std": 1.04
                },
                "BNCI2015001": {
                  "mean": 73.39,
                  "std": 0.69
                }
              },
              "delta": {
                "BNCI2014001": null,
                "BNCI2014002": null,
                "BNCI2015001": null
              },
              "isBaseline": true,
              "isReference": false,
              "key": "EA-EEGNet",
              "lab": false,
              "code": "hustbciml/algorithms/models/EEGNet.py",
              "desc": "Compact convolutional network: a temporal convolution, a depthwise spatial convolution, and a separable convolution. The benchmark's default backbone.",
              "ref": "V. J. Lawhern et al., J. Neural Eng., 2018",
              "doi": "10.1088/1741-2552/aace8c",
              "pinAfter": null
            }
          ]
        }
      ]
    },
    {
      "id": "transfer",
      "title": "Transfer Learning",
      "blurb": "The learning-objective stage. Every row is the identical Euclidean-aligned EEGNet; only the training or adaptation objective changes. Approaches are grouped by how much of the target they use and when — source-only, unsupervised domain adaptation, source-free, and test-time — all two-class on the three datasets and measured against the same no-transfer baseline (ERM). The privacy-preserving family keeps each subject's raw EEG local and is measured against Centralized Training rather than ERM (see its note).",
      "references": null,
      "context": null,
      "groups": [
        {
          "subcat": "Source-only",
          "blurb": "Trained on the labelled source subjects only; the target is never used for adaptation, and inference is a plain forward pass. Baseline: ERM.",
          "baseline": "ERM",
          "reference": null,
          "rows": [
            {
              "name": "ABAT",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.2,
                  "std": 0.69
                },
                "BNCI2014002": {
                  "mean": 74.9,
                  "std": 0.41
                },
                "BNCI2015001": {
                  "mean": 74.14,
                  "std": 0.7
                }
              },
              "delta": {
                "BNCI2014001": 2.13,
                "BNCI2014002": 0.5,
                "BNCI2015001": 0.95
              },
              "isBaseline": false,
              "isReference": false,
              "key": "ABAT",
              "lab": true,
              "code": "hustbciml/algorithms/strategies/ABAT.py",
              "desc": "Replaces each training batch with a channel-scaled adversarial batch after a short clean warm-up, hardening the source-trained EEGNet against distribution shift; the target is not used during training.",
              "ref": "X. Chen, ..., D. Wu*, IEEE Trans. Neural Syst. Rehabil. Eng., 2024",
              "doi": "10.1109/TNSRE.2024.3391936",
              "pinAfter": null
            },
            {
              "name": "PAT",
              "acc": {
                "BNCI2014001": {
                  "mean": 73.53,
                  "std": 0.95
                },
                "BNCI2014002": {
                  "mean": 75.12,
                  "std": 0.47
                },
                "BNCI2015001": {
                  "mean": 74.08,
                  "std": 0.42
                }
              },
              "delta": {
                "BNCI2014001": 1.46,
                "BNCI2014002": 0.72,
                "BNCI2015001": 0.89
              },
              "isBaseline": false,
              "isReference": false,
              "key": "PAT",
              "lab": true,
              "code": "hustbciml/algorithms/strategies/PAT.py",
              "desc": "Extends adversarial training for privacy-preserving (source-only) transfer: after a clean warm-up each Euclidean-aligned batch is amplitude-scaled (×(1±0.05)) then perturbed by a global-ε L∞ PGD attack (noisy-initialized, eps 0.03, 10 steps), hardening the source-trained EEGNet against distribution shift; the target is never used in training.",
              "ref": "X. Chen, ..., D. Wu*, Fundamental Research, 2026",
              "doi": "10.1016/j.fmre.2026.04.034",
              "pinAfter": null
            },
            {
              "name": "MDMAML",
              "acc": {
                "BNCI2014001": {
                  "mean": 75.13,
                  "std": 0.38
                },
                "BNCI2014002": {
                  "mean": 73.4,
                  "std": 1.23
                },
                "BNCI2015001": {
                  "mean": 73.06,
                  "std": 0.23
                }
              },
              "delta": {
                "BNCI2014001": 3.06,
                "BNCI2014002": -1.0,
                "BNCI2015001": -0.13
              },
              "isBaseline": false,
              "isReference": false,
              "key": "MDMAML",
              "lab": true,
              "code": "hustbciml/algorithms/strategies/MDMAML.py",
              "desc": "Domain-paired first-order MAML across the source subjects — meta-learns an initialization so that one adaptation step on any source subject lowers the loss on the others, then applies the meta-learned EEGNet to the target with no target fine-tuning.",
              "ref": "S. Li, ..., D. Wu*, IEEE Comput. Intell. Mag., 2022",
              "doi": "10.1109/MCI.2022.3199622",
              "pinAfter": null
            },
            {
              "name": "ERM",
              "acc": {
                "BNCI2014001": {
                  "mean": 72.07,
                  "std": 1.58
                },
                "BNCI2014002": {
                  "mean": 74.4,
                  "std": 1.04
                },
                "BNCI2015001": {
                  "mean": 73.19,
                  "std": 0.81
                }
              },
              "delta": {
                "BNCI2014001": null,
                "BNCI2014002": null,
                "BNCI2015001": null
              },
              "isBaseline": true,
              "isReference": false,
              "key": "EA-EEGNet",
              "lab": false,
              "code": "hustbciml/algorithms/strategies/ERM.py",
              "desc": "Standard supervised training on the source subjects with no adaptation — the no-transfer baseline every transfer family is measured against.",
              "ref": null,
              "doi": null,
              "pinAfter": null
            }
          ]
        },
        {
          "subcat": "Unsupervised domain adaptation",
          "blurb": "Trained jointly on the labelled source and the unlabelled target, aligning the two distributions during source training; no target labels are used. Measured against the no-transfer baseline.",
          "baseline": null,
          "reference": {
            "name": "EEGNet baseline (ERM, no transfer)",
            "acc": {
              "BNCI2014001": {
                "mean": 72.07,
                "std": 1.58
              },
              "BNCI2014002": {
                "mean": 74.4,
                "std": 1.04
              },
              "BNCI2015001": {
                "mean": 73.19,
                "std": 0.81
              }
            }
          },
          "rows": [
            {
              "name": "MEKT",
              "acc": {
                "BNCI2014001": {
                  "mean": 76.54,
                  "std": 0.0
                },
                "BNCI2014002": {
                  "mean": 77.86,
                  "std": 0.0
                },
                "BNCI2015001": {
                  "mean": 73.04,
                  "std": 0.0
                }
              },
              "delta": {
                "BNCI2014001": 4.47,
                "BNCI2014002": 3.46,
                "BNCI2015001": -0.15
              },
              "isBaseline": false,
              "isReference": false,
              "key": "MEKT",
              "lab": true,
              "code": "hustbciml/algorithms/strategies/MEKT.py",
              "desc": "Network-free manifold transfer (not an EEGNet model): per-subject covariance centroid alignment and Riemannian tangent-space features, then a jointly learned source/target subspace that minimizes the joint-distribution shift while preserving source discriminability and target locality, refined by EM pseudo-labelling, into a shrinkage-LDA. Deterministic.",
              "ref": "W. Zhang, D. Wu*, IEEE Trans. Neural Syst. Rehabil. Eng., 2020",
              "doi": "10.1109/TNSRE.2020.2985996",
              "pinAfter": null
            },
            {
              "name": "DJP-MMD",
              "acc": {
                "BNCI2014001": {
                  "mean": 73.1,
                  "std": 0.64
                },
                "BNCI2014002": {
                  "mean": 77.62,
                  "std": 0.44
                },
                "BNCI2015001": {
                  "mean": 73.49,
                  "std": 0.64
                }
              },
              "delta": {
                "BNCI2014001": 1.03,
                "BNCI2014002": 3.22,
                "BNCI2015001": 0.3
              },
              "isBaseline": false,
              "isReference": false,
              "key": "DJP-MMD",
              "lab": true,
              "code": "hustbciml/algorithms/strategies/DJPMMD.py",
              "desc": "Matches the joint probability across domains with a discriminative joint-probability maximum mean discrepancy.",
              "ref": "W. Zhang, D. Wu*, IJCNN, 2020",
              "doi": "10.1109/IJCNN48605.2020.9207365",
              "pinAfter": null
            },
            {
              "name": "MCC",
              "acc": {
                "BNCI2014001": {
                  "mean": 79.04,
                  "std": 0.67
                },
                "BNCI2014002": {
                  "mean": 80.88,
                  "std": 1.64
                },
                "BNCI2015001": {
                  "mean": 78.53,
                  "std": 0.61
                }
              },
              "delta": {
                "BNCI2014001": 6.97,
                "BNCI2014002": 6.48,
                "BNCI2015001": 5.34
              },
              "isBaseline": false,
              "isReference": false,
              "key": "MCC",
              "lab": false,
              "code": "hustbciml/algorithms/strategies/MCC.py",
              "desc": "Minimizes class confusion in the target predictions during source training.",
              "ref": "Y. Jin et al., ECCV, 2020",
              "doi": "10.1007/978-3-030-58589-1_28",
              "pinAfter": null
            },
            {
              "name": "CDAN",
              "acc": {
                "BNCI2014001": {
                  "mean": 76.26,
                  "std": 0.94
                },
                "BNCI2014002": {
                  "mean": 78.31,
                  "std": 0.89
                },
                "BNCI2015001": {
                  "mean": 76.22,
                  "std": 0.76
                }
              },
              "delta": {
                "BNCI2014001": 4.19,
                "BNCI2014002": 3.91,
                "BNCI2015001": 3.03
              },
              "isBaseline": false,
              "isReference": false,
              "key": "CDAN",
              "lab": false,
              "code": "hustbciml/algorithms/strategies/CDAN.py",
              "desc": "Domain-adversarial training conditioned on the classifier's predictions.",
              "ref": "M. Long et al., NeurIPS, 2018",
              "doi": null,
              "pinAfter": null
            },
            {
              "name": "JAN",
              "acc": {
                "BNCI2014001": {
                  "mean": 75.44,
                  "std": 0.41
                },
                "BNCI2014002": {
                  "mean": 75.86,
                  "std": 0.67
                },
                "BNCI2015001": {
                  "mean": 74.64,
                  "std": 0.57
                }
              },
              "delta": {
                "BNCI2014001": 3.37,
                "BNCI2014002": 1.46,
                "BNCI2015001": 1.45
              },
              "isBaseline": false,
              "isReference": false,
              "key": "JAN",
              "lab": false,
              "code": "hustbciml/algorithms/strategies/JAN.py",
              "desc": "Matches the joint distribution of features and predictions across domains (joint maximum mean discrepancy).",
              "ref": "M. Long et al., ICML, 2017",
              "doi": null,
              "pinAfter": null
            },
            {
              "name": "DAN",
              "acc": {
                "BNCI2014001": {
                  "mean": 75.03,
                  "std": 1.04
                },
                "BNCI2014002": {
                  "mean": 73.9,
                  "std": 0.61
                },
                "BNCI2015001": {
                  "mean": 74.4,
                  "std": 1.2
                }
              },
              "delta": {
                "BNCI2014001": 2.96,
                "BNCI2014002": -0.5,
                "BNCI2015001": 1.21
              },
              "isBaseline": false,
              "isReference": false,
              "key": "DAN",
              "lab": false,
              "code": "hustbciml/algorithms/strategies/DAN.py",
              "desc": "Matches feature distributions across domains with a multi-kernel maximum mean discrepancy.",
              "ref": "M. Long et al., ICML, 2015",
              "doi": null,
              "pinAfter": null
            },
            {
              "name": "DANN",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.77,
                  "std": 1.01
                },
                "BNCI2014002": {
                  "mean": 74.02,
                  "std": 0.79
                },
                "BNCI2015001": {
                  "mean": 73.65,
                  "std": 1.11
                }
              },
              "delta": {
                "BNCI2014001": 2.7,
                "BNCI2014002": -0.38,
                "BNCI2015001": 0.46
              },
              "isBaseline": false,
              "isReference": false,
              "key": "EA-DANN",
              "lab": false,
              "code": "hustbciml/algorithms/strategies/DANN.py",
              "desc": "Adversarial feature learning through a gradient-reversal domain discriminator.",
              "ref": "Y. Ganin et al., J. Mach. Learn. Res., 2016",
              "doi": null,
              "pinAfter": null
            },
            {
              "name": "MDD",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.18,
                  "std": 0.25
                },
                "BNCI2014002": {
                  "mean": 74.48,
                  "std": 0.93
                },
                "BNCI2015001": {
                  "mean": 73.17,
                  "std": 0.56
                }
              },
              "delta": {
                "BNCI2014001": 2.11,
                "BNCI2014002": 0.08,
                "BNCI2015001": -0.02
              },
              "isBaseline": false,
              "isReference": false,
              "key": "MDD",
              "lab": false,
              "code": "hustbciml/algorithms/strategies/MDD.py",
              "desc": "Bounds the domain gap with a margin disparity discrepancy between source and target.",
              "ref": "Y. Zhang et al., ICML, 2019",
              "doi": null,
              "pinAfter": null
            }
          ]
        },
        {
          "subcat": "Source-free adaptation",
          "blurb": "Adapts a source-trained model to the target while keeping no source data at transfer time. Measured against the no-transfer baseline.",
          "baseline": null,
          "reference": {
            "name": "EEGNet baseline (ERM, no transfer)",
            "acc": {
              "BNCI2014001": {
                "mean": 72.07,
                "std": 1.58
              },
              "BNCI2014002": {
                "mean": 74.4,
                "std": 1.04
              },
              "BNCI2015001": {
                "mean": 73.19,
                "std": 0.81
              }
            }
          },
          "rows": [
            {
              "name": "LSFT",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.77,
                  "std": 0.0
                },
                "BNCI2014002": {
                  "mean": 73.64,
                  "std": 0.0
                },
                "BNCI2015001": {
                  "mean": 75.46,
                  "std": 0.0
                }
              },
              "delta": {
                "BNCI2014001": 2.7,
                "BNCI2014002": -0.76,
                "BNCI2015001": 2.27
              },
              "isBaseline": false,
              "isReference": false,
              "key": "LSFT",
              "lab": true,
              "code": "hustbciml/algorithms/strategies/LSFT.py",
              "desc": "Classical source-free transfer on Riemannian tangent-space features: source classifiers vote to pseudo-label the target, then an iterative subspace adaptation relabels it. No raw source data at transfer time.",
              "ref": "W. Zhang, D. Wu*, IEEE Trans. Cogn. Devel. Syst., 2023",
              "doi": "10.1109/TCDS.2022.3193731",
              "pinAfter": null
            },
            {
              "name": "ASFA",
              "acc": {
                "BNCI2014001": {
                  "mean": 73.28,
                  "std": 0.51
                },
                "BNCI2014002": {
                  "mean": 75.1,
                  "std": 0.93
                },
                "BNCI2015001": {
                  "mean": 74.68,
                  "std": 0.17
                }
              },
              "delta": {
                "BNCI2014001": 1.21,
                "BNCI2014002": 0.7,
                "BNCI2015001": 1.49
              },
              "isBaseline": false,
              "isReference": false,
              "key": "ASFA",
              "lab": true,
              "code": "hustbciml/algorithms/strategies/ASFA.py",
              "desc": "Freezes the source classifier head and adapts the feature extractor by minimizing a Tsallis-entropy prediction-uncertainty objective with a consistency-regularized auxiliary head; no source data at transfer time.",
              "ref": "K. Xia, ..., D. Wu*, IEEE Trans. Biomed. Eng., 2022",
              "doi": "10.1109/TBME.2022.3168570",
              "pinAfter": null
            },
            {
              "name": "SHOT",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.2,
                  "std": 1.06
                },
                "BNCI2014002": {
                  "mean": 75.93,
                  "std": 0.7
                },
                "BNCI2015001": {
                  "mean": 75.64,
                  "std": 0.23
                }
              },
              "delta": {
                "BNCI2014001": 2.13,
                "BNCI2014002": 1.53,
                "BNCI2015001": 2.45
              },
              "isBaseline": false,
              "isReference": false,
              "key": "SHOT",
              "lab": false,
              "code": "hustbciml/algorithms/strategies/SHOT.py",
              "desc": "Freezes the source classifier and adapts the feature extractor by information maximization with pseudo-labels.",
              "ref": "J. Liang, D. Hu, and J. Feng, ICML, 2020",
              "doi": null,
              "pinAfter": null
            }
          ]
        },
        {
          "subcat": "Test-time adaptation",
          "blurb": "Adapts online as the target trials arrive at test time, updating the source-trained model without target labels. Measured against the no-transfer baseline.",
          "baseline": null,
          "reference": {
            "name": "EEGNet baseline (ERM, no transfer)",
            "acc": {
              "BNCI2014001": {
                "mean": 72.07,
                "std": 1.58
              },
              "BNCI2014002": {
                "mean": 74.4,
                "std": 1.04
              },
              "BNCI2015001": {
                "mean": 73.19,
                "std": 0.81
              }
            }
          },
          "rows": [
            {
              "name": "T-TIME",
              "acc": {
                "BNCI2014001": {
                  "mean": 76.05,
                  "std": 0.42
                },
                "BNCI2014002": {
                  "mean": 80.33,
                  "std": 0.52
                },
                "BNCI2015001": {
                  "mean": 77.75,
                  "std": 0.68
                }
              },
              "delta": {
                "BNCI2014001": 3.98,
                "BNCI2014002": 5.93,
                "BNCI2015001": 4.56
              },
              "isBaseline": false,
              "isReference": false,
              "key": "T-TIME",
              "lab": true,
              "code": "hustbciml/algorithms/strategies/TTIME.py",
              "desc": "Online test-time adaptation — for each incoming target batch it updates an incremental Euclidean-Alignment reference and minimizes an information-maximization loss (conditional-entropy minimization with a marginal-diversity regularizer), then predicts. Plug-and-play, no target labels.",
              "ref": "S. Li, ..., D. Wu*, IEEE Trans. Biomed. Eng., 2024",
              "doi": "10.1109/TBME.2023.3303289",
              "pinAfter": null
            },
            {
              "name": "BFT",
              "acc": {
                "BNCI2014001": {
                  "mean": 73.79,
                  "std": 0.67
                },
                "BNCI2014002": {
                  "mean": 76.29,
                  "std": 0.73
                },
                "BNCI2015001": {
                  "mean": 74.46,
                  "std": 0.31
                }
              },
              "delta": {
                "BNCI2014001": 1.72,
                "BNCI2014002": 1.89,
                "BNCI2015001": 1.27
              },
              "isBaseline": false,
              "isReference": false,
              "key": "BFT",
              "lab": true,
              "code": "hustbciml/algorithms/strategies/BFT.py",
              "desc": "Backpropagation-free test-time adaptation: averages the model's predictions over label-preserving augmentations of each target trial, gaining robustness with no gradient updates — aimed at lightweight, low-power BCI hardware.",
              "ref": "S. Li†, J. Ouyang†, Z. Cui†, ..., D. Wu*, arXiv:2601.07556, 2026",
              "doi": "10.48550/arXiv.2601.07556",
              "pinAfter": null
            },
            {
              "name": "DELTA",
              "acc": {
                "BNCI2014001": {
                  "mean": 75.93,
                  "std": 0.44
                },
                "BNCI2014002": {
                  "mean": 80.14,
                  "std": 0.51
                },
                "BNCI2015001": {
                  "mean": 77.44,
                  "std": 0.64
                }
              },
              "delta": {
                "BNCI2014001": 3.86,
                "BNCI2014002": 5.74,
                "BNCI2015001": 4.25
              },
              "isBaseline": false,
              "isReference": false,
              "key": "DELTA",
              "lab": false,
              "code": "hustbciml/algorithms/strategies/DELTA.py",
              "desc": "Test-time entropy minimization with class-imbalance-corrected prediction diversity.",
              "ref": "B. Zhao, C. Chen, and S.-T. Xia, ICLR, 2023",
              "doi": null,
              "pinAfter": null
            },
            {
              "name": "ISFDA",
              "acc": {
                "BNCI2014001": {
                  "mean": 75.8,
                  "std": 0.54
                },
                "BNCI2014002": {
                  "mean": 79.81,
                  "std": 0.42
                },
                "BNCI2015001": {
                  "mean": 77.74,
                  "std": 0.53
                }
              },
              "delta": {
                "BNCI2014001": 3.73,
                "BNCI2014002": 5.41,
                "BNCI2015001": 4.55
              },
              "isBaseline": false,
              "isReference": false,
              "key": "ISFDA",
              "lab": false,
              "code": "hustbciml/algorithms/strategies/ISFDA.py",
              "desc": "Online test-time adaptation by information maximization, with intra-class tightening and inter-class separation on pseudo-labelled target features; adapts the whole network over the target stream.",
              "ref": "X. Li et al., ACM MM, 2021",
              "doi": "10.1145/3474085.3475487",
              "pinAfter": null
            },
            {
              "name": "PL",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.38,
                  "std": 1.89
                },
                "BNCI2014002": {
                  "mean": 77.05,
                  "std": 1.2
                },
                "BNCI2015001": {
                  "mean": 73.96,
                  "std": 1.01
                }
              },
              "delta": {
                "BNCI2014001": 2.31,
                "BNCI2014002": 2.65,
                "BNCI2015001": 0.77
              },
              "isBaseline": false,
              "isReference": false,
              "key": "PL",
              "lab": false,
              "code": "hustbciml/algorithms/strategies/PL.py",
              "desc": "Online self-training on the model's own pseudo-labels.",
              "ref": "D.-H. Lee, ICML Workshop Challenges Represent. Learn., 2013",
              "doi": null,
              "pinAfter": null
            },
            {
              "name": "SAR",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.9,
                  "std": 1.99
                },
                "BNCI2014002": {
                  "mean": 77.12,
                  "std": 2.02
                },
                "BNCI2015001": {
                  "mean": 72.31,
                  "std": 1.95
                }
              },
              "delta": {
                "BNCI2014001": 2.83,
                "BNCI2014002": 2.72,
                "BNCI2015001": -0.88
              },
              "isBaseline": false,
              "isReference": false,
              "key": "SAR",
              "lab": false,
              "code": "hustbciml/algorithms/strategies/SAR.py",
              "desc": "Sharpness-aware, reliable test-time entropy minimization.",
              "ref": "S. Niu et al., ICLR, 2023",
              "doi": null,
              "pinAfter": null
            },
            {
              "name": "BN-adapt",
              "acc": {
                "BNCI2014001": {
                  "mean": 73.23,
                  "std": 1.29
                },
                "BNCI2014002": {
                  "mean": 75.0,
                  "std": 1.19
                },
                "BNCI2015001": {
                  "mean": 75.04,
                  "std": 0.56
                }
              },
              "delta": {
                "BNCI2014001": 1.16,
                "BNCI2014002": 0.6,
                "BNCI2015001": 1.85
              },
              "isBaseline": false,
              "isReference": false,
              "key": "BN-adapt",
              "lab": false,
              "code": "hustbciml/algorithms/strategies/BNAdapt.py",
              "desc": "Re-estimates BatchNorm statistics on the target, with no gradient step.",
              "ref": "S. Schneider et al., NeurIPS, 2020",
              "doi": null,
              "pinAfter": null
            },
            {
              "name": "Tent",
              "acc": {
                "BNCI2014001": {
                  "mean": 72.04,
                  "std": 1.42
                },
                "BNCI2014002": {
                  "mean": 73.81,
                  "std": 0.99
                },
                "BNCI2015001": {
                  "mean": 72.01,
                  "std": 1.13
                }
              },
              "delta": {
                "BNCI2014001": -0.03,
                "BNCI2014002": -0.59,
                "BNCI2015001": -1.18
              },
              "isBaseline": false,
              "isReference": false,
              "key": "Tent",
              "lab": false,
              "code": "hustbciml/algorithms/strategies/Tent.py",
              "desc": "Test-time entropy minimization over the BatchNorm affine parameters.",
              "ref": "D. Wang et al., ICLR, 2021",
              "doi": null,
              "pinAfter": null
            }
          ]
        },
        {
          "subcat": "Privacy-preserving",
          "blurb": "These approaches never pool raw EEG across subjects — each subject's data stays local — the privacy-preserving counterpart to Centralized Training (the reference). FedBS, SAFE and FedAvg share model updates through a server (federated); MSDT shares only per-source models fused at test time (decentralized). All three datasets are two-class (chance 50%), so the columns are directly comparable. Δ is versus Centralized Training on the same dataset.",
          "baseline": "Centralized Training",
          "reference": null,
          "rows": [
            {
              "name": "SAFE",
              "acc": {
                "BNCI2014001": {
                  "mean": 70.91,
                  "std": 1.15
                },
                "BNCI2014002": {
                  "mean": 78.21,
                  "std": 0.66
                },
                "BNCI2015001": {
                  "mean": 75.96,
                  "std": 0.53
                }
              },
              "delta": {
                "BNCI2014001": -1.16,
                "BNCI2014002": 3.81,
                "BNCI2015001": 2.77
              },
              "isBaseline": false,
              "isReference": false,
              "key": "SAFE",
              "lab": true,
              "code": "hustbciml/algorithms/strategies/SAFE.py",
              "desc": "Federated learning that adds single-step adversarial feature training and a one-step adversarial weight perturbation on top of batch-specific BatchNorm, hardening the shared model without pooling raw EEG. The adversarial regularization costs a little clean accuracy on BNCI2014001 but lifts the other two datasets clearly above centralized training.",
              "ref": "T. Jia, ..., D. Wu*, arXiv:2601.05789, 2026",
              "doi": "10.48550/arXiv.2601.05789",
              "pinAfter": null
            },
            {
              "name": "FedBS",
              "acc": {
                "BNCI2014001": {
                  "mean": 72.69,
                  "std": 1.62
                },
                "BNCI2014002": {
                  "mean": 76.07,
                  "std": 0.65
                },
                "BNCI2015001": {
                  "mean": 75.64,
                  "std": 0.63
                }
              },
              "delta": {
                "BNCI2014001": 0.62,
                "BNCI2014002": 1.67,
                "BNCI2015001": 2.45
              },
              "isBaseline": false,
              "isReference": false,
              "key": "FedBS",
              "lab": true,
              "code": "hustbciml/algorithms/strategies/FedBS.py",
              "desc": "Federated learning with batch-specific BatchNorm and sharpness-aware minimization, aggregating per-subject model updates through a server without sharing raw EEG. Under the same optimizer and learning rate as Centralized Training it recovers essentially all of the centralized accuracy — privacy is nearly free here.",
              "ref": "T. Jia, ..., D. Wu*, IEEE Trans. Neural Syst. Rehabil. Eng., 2024",
              "doi": "10.1109/TNSRE.2024.3457504",
              "pinAfter": null
            },
            {
              "name": "MSDT",
              "acc": {
                "BNCI2014001": {
                  "mean": 73.84,
                  "std": 0.23
                },
                "BNCI2014002": {
                  "mean": 73.36,
                  "std": 0.59
                },
                "BNCI2015001": {
                  "mean": 72.51,
                  "std": 0.22
                }
              },
              "delta": {
                "BNCI2014001": 1.77,
                "BNCI2014002": -1.04,
                "BNCI2015001": -0.68
              },
              "isBaseline": false,
              "isReference": false,
              "key": "MSDT",
              "lab": true,
              "code": "hustbciml/algorithms/strategies/MSDT.py",
              "desc": "Decentralized multi-source transfer on Riemannian tangent-space features (not an EEGNet model): each source subject trains its own classifier and the target adapts and fuses them at test time, with no source data pooled. It lands close to Centralized Training across the three datasets — a little above on BNCI2014001, a little below on the other two — reflecting that Riemannian representation and test-time adaptation rather than the privacy mechanism.",
              "ref": "W. Zhang, ..., D. Wu*, IEEE Trans. Neural Syst. Rehabil. Eng., 2022",
              "doi": "10.1109/TNSRE.2022.3207494",
              "pinAfter": null
            },
            {
              "name": "FedAvg",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.54,
                  "std": 0.79
                },
                "BNCI2014002": {
                  "mean": 74.12,
                  "std": 0.44
                },
                "BNCI2015001": {
                  "mean": 71.62,
                  "std": 0.86
                }
              },
              "delta": {
                "BNCI2014001": 2.47,
                "BNCI2014002": -0.28,
                "BNCI2015001": -1.57
              },
              "isBaseline": false,
              "isReference": false,
              "key": "FedAvg",
              "lab": false,
              "code": "hustbciml/algorithms/strategies/FedAvg.py",
              "desc": "Federated averaging: each subject trains locally and the server averages the model weights — the vanilla federated baseline that isolates FedBS's two additions.",
              "ref": "B. McMahan et al., AISTATS, 2017",
              "doi": null,
              "pinAfter": null
            },
            {
              "name": "Centralized Training",
              "acc": {
                "BNCI2014001": {
                  "mean": 72.07,
                  "std": 1.58
                },
                "BNCI2014002": {
                  "mean": 74.4,
                  "std": 1.04
                },
                "BNCI2015001": {
                  "mean": 73.19,
                  "std": 0.81
                }
              },
              "delta": {
                "BNCI2014001": null,
                "BNCI2014002": null,
                "BNCI2015001": null
              },
              "isBaseline": true,
              "isReference": false,
              "key": null,
              "lab": false,
              "code": null,
              "desc": "EA-EEGNet trained on all source subjects pooled together — the non-private reference every privacy-preserving approach is measured against.",
              "ref": null,
              "doi": null,
              "pinAfter": null
            }
          ]
        }
      ]
    },
    {
      "id": "ensemble",
      "title": "Ensemble Learning",
      "blurb": "The aggregation stage, in a fully decentralized privacy setting. Five heterogeneous learners — tangent-space LDA, tangent-space SVM, EEGNet, ShallowConvNet, and CSPNet — are trained on each source subject's data alone, and the subjects share only their hard predicted labels on the target, never model weights or raw EEG. A post-hoc combiner fuses the (N-1)×5 label votes into a consensus prediction. Every combiner sees the same hard votes, so none has an information advantage; they differ only in how they weight and combine the votes. StackingNet is lab-proposed; SML-OVR is the lab's multi-class combiner, a one-vs-rest generalization of the binary SML that on these two-class tasks reduces exactly to it, so the two report the same accuracy and are placed together. The others (the binary SML and the crowd-labelling and truth-discovery aggregators) are established baselines. All three datasets are two-class (chance 50%), so the columns are directly comparable, and each combiner is measured against plain majority voting on the same dataset. Rows are ordered lab-proposed first, then the remaining combiners by accuracy, with plain majority voting (the baseline) last.",
      "references": null,
      "context": {
        "BNCI2014001": {
          "classes": "2-class",
          "chance": 50.0,
          "single_source": 61.22,
          "centralized": 72.07,
          "voting": 74.31
        },
        "BNCI2014002": {
          "classes": "2-class",
          "chance": 50.0,
          "single_source": 59.61,
          "centralized": 74.4,
          "voting": 72.0
        },
        "BNCI2015001": {
          "classes": "2-class",
          "chance": 50.0,
          "single_source": 59.59,
          "centralized": 73.19,
          "voting": 69.83
        }
      },
      "groups": [
        {
          "subcat": null,
          "blurb": "",
          "baseline": "Majority voting",
          "reference": null,
          "rows": [
            {
              "name": "SML-OVR",
              "acc": {
                "BNCI2014001": {
                  "mean": 75.46,
                  "std": null
                },
                "BNCI2014002": {
                  "mean": 73.14,
                  "std": null
                },
                "BNCI2015001": {
                  "mean": 72.71,
                  "std": null
                }
              },
              "delta": {
                "BNCI2014001": 1.15,
                "BNCI2014002": 1.14,
                "BNCI2015001": 2.88
              },
              "isBaseline": false,
              "isReference": false,
              "key": "SML-OVR",
              "lab": true,
              "code": "hustbciml/scripts/_ensembles.py",
              "desc": "The lab's one-vs-rest spectral meta-learner, the multi-class generalization of the binary SML: for each class it runs the binary SML weight estimation on the one-hot votes and sums the per-class weightings, so it also handles more than two classes (for example the native four-class BNCI2014001, which the code still supports). On these two-class tasks it reduces exactly to the binary SML directly below, so the two report the identical accuracy here; the multi-class advantage shows only on native multi-class data.",
              "ref": "S. Li, ..., D. Wu*, IEEE Comput. Intell. Mag., 2026",
              "doi": "10.1109/MCI.2025.3624194",
              "pinAfter": null
            },
            {
              "name": "SML",
              "acc": {
                "BNCI2014001": {
                  "mean": 75.46,
                  "std": null
                },
                "BNCI2014002": {
                  "mean": 73.14,
                  "std": null
                },
                "BNCI2015001": {
                  "mean": 72.71,
                  "std": null
                }
              },
              "delta": {
                "BNCI2014001": 1.15,
                "BNCI2014002": 1.14,
                "BNCI2015001": 2.88
              },
              "isBaseline": false,
              "isReference": false,
              "key": null,
              "lab": false,
              "code": "hustbciml/scripts/_ensembles.py",
              "desc": "Binary spectral meta-learner: weights each source model by the principal eigenvector of the models' ±1 vote-covariance, an unsupervised accuracy estimate valid for two classes. It is the binary base that the lab's SML-OVR above generalizes to more classes; on these two-class tasks the two coincide, which is why they report the same accuracy and sit together.",
              "ref": "F. Parisi et al., Proc. Natl. Acad. Sci. USA, 2014",
              "doi": "10.1073/pnas.1219097111",
              "pinAfter": "SML-OVR"
            },
            {
              "name": "StackingNet",
              "acc": {
                "BNCI2014001": {
                  "mean": 75.31,
                  "std": null
                },
                "BNCI2014002": {
                  "mean": 73.0,
                  "std": null
                },
                "BNCI2015001": {
                  "mean": 70.5,
                  "std": null
                }
              },
              "delta": {
                "BNCI2014001": 1.0,
                "BNCI2014002": 1.0,
                "BNCI2015001": 0.67
              },
              "isBaseline": false,
              "isReference": false,
              "key": "StackingNet",
              "lab": true,
              "code": "hustbciml/scripts/_ensembles.py",
              "desc": "Unsupervised transductive meta-combiner over the source models' hard labels: learns per-model weights on the unlabelled target by consensus agreement (no target labels), initialized from each model's balanced accuracy against the majority vote.",
              "ref": "S. Li†, C. Liu†, D. Wu*, Advanced Science, 2026",
              "doi": "10.1002/advs.76488",
              "pinAfter": null
            },
            {
              "name": "Dawid-Skene",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.85,
                  "std": null
                },
                "BNCI2014002": {
                  "mean": 73.14,
                  "std": null
                },
                "BNCI2015001": {
                  "mean": 74.29,
                  "std": null
                }
              },
              "delta": {
                "BNCI2014001": 0.54,
                "BNCI2014002": 1.14,
                "BNCI2015001": 4.46
              },
              "isBaseline": false,
              "isReference": false,
              "key": null,
              "lab": false,
              "code": "hustbciml/scripts/_ensemble_baselines.py",
              "desc": "Classic EM crowd-labelling aggregator: jointly estimates each source model's full confusion matrix and the consensus label from the models' hard votes alone (no target labels).",
              "ref": "A. P. Dawid and A. M. Skene, J. R. Stat. Soc. C, 1979",
              "doi": "10.2307/2346806",
              "pinAfter": null
            },
            {
              "name": "LAA",
              "acc": {
                "BNCI2014001": {
                  "mean": 76.08,
                  "std": null
                },
                "BNCI2014002": {
                  "mean": 73.36,
                  "std": null
                },
                "BNCI2015001": {
                  "mean": 72.71,
                  "std": null
                }
              },
              "delta": {
                "BNCI2014001": 1.77,
                "BNCI2014002": 1.36,
                "BNCI2015001": 2.88
              },
              "isBaseline": false,
              "isReference": false,
              "key": null,
              "lab": false,
              "code": "hustbciml/scripts/_ensemble_baselines.py",
              "desc": "Label-aware autoencoder: an unsupervised neural aggregator that encodes the per-trial votes into a consensus label and reconstructs each source model's vote from it.",
              "ref": "L. Yin, ..., IJCAI, 2017",
              "doi": "10.24963/ijcai.2017/184",
              "pinAfter": null
            },
            {
              "name": "EBCC",
              "acc": {
                "BNCI2014001": {
                  "mean": 76.08,
                  "std": null
                },
                "BNCI2014002": {
                  "mean": 72.43,
                  "std": null
                },
                "BNCI2015001": {
                  "mean": 71.17,
                  "std": null
                }
              },
              "delta": {
                "BNCI2014001": 1.77,
                "BNCI2014002": 0.43,
                "BNCI2015001": 1.34
              },
              "isBaseline": false,
              "isReference": false,
              "key": null,
              "lab": false,
              "code": "hustbciml/scripts/_ensemble_baselines.py",
              "desc": "Enhanced Bayesian classifier combination: variational inference over low-rank worker-correlation groups — the most expressive of the crowd-aggregation baselines.",
              "ref": "Y. Li, B. Rubinstein, and T. Cohn, ICML, 2019",
              "doi": null,
              "pinAfter": null
            },
            {
              "name": "Wawa",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.38,
                  "std": null
                },
                "BNCI2014002": {
                  "mean": 72.14,
                  "std": null
                },
                "BNCI2015001": {
                  "mean": 68.5,
                  "std": null
                }
              },
              "delta": {
                "BNCI2014001": 0.07,
                "BNCI2014002": 0.14,
                "BNCI2015001": -1.33
              },
              "isBaseline": false,
              "isReference": false,
              "key": null,
              "lab": false,
              "code": "hustbciml/scripts/_ensemble_baselines.py",
              "desc": "Worker-agreement-with-aggregate heuristic: weight each source model by its agreement with the plain majority vote, then re-vote. A crowd-kit heuristic with no separate paper.",
              "ref": "Worker Agreement With Aggregate — crowd-kit heuristic",
              "doi": null,
              "pinAfter": null
            },
            {
              "name": "PM",
              "acc": {
                "BNCI2014001": {
                  "mean": 76.16,
                  "std": null
                },
                "BNCI2014002": {
                  "mean": 70.57,
                  "std": null
                },
                "BNCI2015001": {
                  "mean": 65.79,
                  "std": null
                }
              },
              "delta": {
                "BNCI2014001": 1.85,
                "BNCI2014002": -1.43,
                "BNCI2015001": -4.04
              },
              "isBaseline": false,
              "isReference": false,
              "key": null,
              "lab": false,
              "code": "hustbciml/scripts/_ensemble_baselines.py",
              "desc": "Truth-discovery aggregator: iteratively weights each source model by how much its votes agree with the current consensus (weight = -log of normalized disagreement), then re-estimates the consensus.",
              "ref": "Q. Li, ..., ACM SIGMOD, 2014",
              "doi": "10.1145/2588555.2610509",
              "pinAfter": null
            },
            {
              "name": "MACE",
              "acc": {
                "BNCI2014001": {
                  "mean": 73.46,
                  "std": null
                },
                "BNCI2014002": {
                  "mean": 65.5,
                  "std": null
                },
                "BNCI2015001": {
                  "mean": 72.0,
                  "std": null
                }
              },
              "delta": {
                "BNCI2014001": -0.85,
                "BNCI2014002": -6.5,
                "BNCI2015001": 2.17
              },
              "isBaseline": false,
              "isReference": false,
              "key": null,
              "lab": false,
              "code": "hustbciml/scripts/_ensemble_baselines.py",
              "desc": "Variational aggregator that separates competent labelling from per-model spamming, to down-weight unreliable source models.",
              "ref": "D. Hovy, ..., NAACL-HLT, 2013",
              "doi": null,
              "pinAfter": null
            },
            {
              "name": "LA",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.38,
                  "std": null
                },
                "BNCI2014002": {
                  "mean": 70.21,
                  "std": null
                },
                "BNCI2015001": {
                  "mean": 65.29,
                  "std": null
                }
              },
              "delta": {
                "BNCI2014001": 0.07,
                "BNCI2014002": -1.79,
                "BNCI2015001": -4.54
              },
              "isBaseline": false,
              "isReference": false,
              "key": null,
              "lab": false,
              "code": "hustbciml/scripts/_ensemble_baselines.py",
              "desc": "Lightweight two-pass aggregator: one online pass estimates each source model's ability under a Beta prior, a second pass re-votes weighted by that ability.",
              "ref": "Y. Yang, ..., ACM Trans. Knowl. Discov. Data, 2024",
              "doi": "10.1145/3630102",
              "pinAfter": null
            },
            {
              "name": "GLAD",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.61,
                  "std": null
                },
                "BNCI2014002": {
                  "mean": 67.29,
                  "std": null
                },
                "BNCI2015001": {
                  "mean": 59.83,
                  "std": null
                }
              },
              "delta": {
                "BNCI2014001": 0.3,
                "BNCI2014002": -4.71,
                "BNCI2015001": -10.0
              },
              "isBaseline": false,
              "isReference": false,
              "key": null,
              "lab": false,
              "code": "hustbciml/scripts/_ensemble_baselines.py",
              "desc": "EM aggregator that jointly infers the consensus label, each source model's ability, and each trial's difficulty.",
              "ref": "J. Whitehill, ..., NeurIPS, 2009",
              "doi": null,
              "pinAfter": null
            },
            {
              "name": "ZenCrowd",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.85,
                  "std": null
                },
                "BNCI2014002": {
                  "mean": 66.93,
                  "std": null
                },
                "BNCI2015001": {
                  "mean": 59.21,
                  "std": null
                }
              },
              "delta": {
                "BNCI2014001": 0.54,
                "BNCI2014002": -5.07,
                "BNCI2015001": -10.62
              },
              "isBaseline": false,
              "isReference": false,
              "key": null,
              "lab": false,
              "code": "hustbciml/scripts/_ensemble_baselines.py",
              "desc": "EM aggregator with a single reliability scalar per source model, inferred from vote agreement alone (no target labels).",
              "ref": "G. Demartini, ..., WWW, 2012",
              "doi": "10.1145/2187836.2187900",
              "pinAfter": null
            },
            {
              "name": "M-MSR",
              "acc": {
                "BNCI2014001": {
                  "mean": 72.92,
                  "std": null
                },
                "BNCI2014002": {
                  "mean": 68.07,
                  "std": null
                },
                "BNCI2015001": {
                  "mean": 59.54,
                  "std": null
                }
              },
              "delta": {
                "BNCI2014001": -1.39,
                "BNCI2014002": -3.93,
                "BNCI2015001": -10.29
              },
              "isBaseline": false,
              "isReference": false,
              "key": null,
              "lab": false,
              "code": "hustbciml/scripts/_ensemble_baselines.py",
              "desc": "Recovers each source model's skill from the pairwise inter-model agreement matrix by robust rank-one matrix completion, then weights the vote by it.",
              "ref": "Q. Ma and A. Olshevsky, NeurIPS, 2020",
              "doi": null,
              "pinAfter": null
            },
            {
              "name": "Majority voting",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.31,
                  "std": null
                },
                "BNCI2014002": {
                  "mean": 72.0,
                  "std": null
                },
                "BNCI2015001": {
                  "mean": 69.83,
                  "std": null
                }
              },
              "delta": {
                "BNCI2014001": null,
                "BNCI2014002": null,
                "BNCI2015001": null
              },
              "isBaseline": true,
              "isReference": false,
              "key": null,
              "lab": false,
              "code": "hustbciml/scripts/_ensembles.py",
              "desc": "Plain majority vote over the hard predicted labels of the five per-subject learners across all source subjects — the label-only baseline every combiner is measured against.",
              "ref": "S. Li, ..., D. Wu*, IEEE Comput. Intell. Mag., 2026",
              "doi": "10.1109/MCI.2025.3624194",
              "pinAfter": null
            }
          ]
        }
      ]
    }
  ]
};
