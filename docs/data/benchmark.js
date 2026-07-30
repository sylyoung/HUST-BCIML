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
        "classes": 2,
        "chance": "50%",
        "trials": "288 / session",
        "role": "Left hand versus right hand (two-class, chance 50%) in every table, including the privacy-preserving and the ensemble families. The original dataset is four-class (left hand, right hand, both feet, and tongue). The benchmark uses its two-class left/right subset throughout, and the four-class variant remains available in the code."
      },
      {
        "name": "BNCI2014002",
        "subjects": 14,
        "channels": 15,
        "sfreq": 512,
        "classes": 2,
        "chance": "50%",
        "trials": 100,
        "role": "Right hand versus both feet, 14 subjects, 100 training-run trials per subject. Two-class (chance 50%) throughout."
      },
      {
        "name": "BNCI2015001",
        "subjects": 12,
        "channels": 13,
        "sfreq": 512,
        "classes": 2,
        "chance": "50%",
        "trials": 200,
        "role": "Right hand versus both feet, 12 subjects, 200 first-session trials per subject. Two-class (chance 50%) throughout."
      }
    ]
  },
  "library": {
    "title": "A unified and reproducible EEG decoding benchmark",
    "tagline": "Every approach is composed of the same modular stages, i.e., an aligner, an augmenter and a backbone, trained under a single learning objective and optionally aggregated by an ensemble. A controlled comparison varies one stage and fixes the rest, so that any change in the accuracy is attributable to that stage alone.",
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
      "blurb": "The aligner stage. An aligner maps the trials of each subject into a shared statistical space prior to the backbone, reducing the between-subject covariance shift that otherwise dominates cross-subject decoding. Alignment requires no label, and is performed separately for each subject. The backbone and its training configuration are identical in every row, and the baseline performs no alignment.",
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
              "desc": "Whitens the trials of each subject by the inverse square root of their mean spatial covariance, so that the average covariance of every subject becomes the identity matrix. The default aligner of the benchmark.",
              "ref": "H. He, D. Wu*, IEEE Trans. Biomed. Eng., 2020",
              "doi": "10.1109/TBME.2019.2913914",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "RA (Riemannian)",
              "acc": {
                "BNCI2014001": {
                  "mean": 73.69,
                  "std": 1.09
                },
                "BNCI2014002": {
                  "mean": 72.17,
                  "std": 0.93
                },
                "BNCI2015001": {
                  "mean": 71.97,
                  "std": 0.07
                }
              },
              "delta": {
                "BNCI2014001": 4.35,
                "BNCI2014002": 10.27,
                "BNCI2015001": 8.51
              },
              "isBaseline": false,
              "isReference": false,
              "key": "RA-EEGNet",
              "lab": false,
              "code": "hustbciml/algorithms/aligners/RA.py",
              "desc": "Normalizes the trials of each subject by the affine-invariant Riemannian (Fréchet) mean of their spatial covariances. The recentering is performed in the curved covariance geometry instead of the Euclidean one.",
              "ref": "P. Zanini et al., IEEE Trans. Biomed. Eng., 2018",
              "doi": "10.1109/TBME.2017.2742541",
              "naReason": null,
              "alsoVaries": null,
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
              "desc": "No alignment. The trials are passed to the backbone as recorded.",
              "ref": null,
              "doi": null,
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            }
          ]
        }
      ]
    },
    {
      "id": "augmentation",
      "title": "Data Augmentation",
      "blurb": "The augmenter stage. An augmenter synthesizes additional training trials to regularize an otherwise identical backbone, and is measured against the same backbone trained without augmentation. The augmenters operate in two different spaces. The electrode-space transforms, i.e., Channel Reflection and Half-Sample Recombination, rearrange the channels, so they are applied to unaligned trials, before any spatial whitening, and are compared with the unaligned baseline. The signal-domain and frequency-domain augmenters are applied to Euclidean-aligned trials, and are compared with the aligned baseline.",
      "groups": [
        {
          "subcat": null,
          "blurb": "",
          "baseline": "none",
          "reference": null,
          "rows": [
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
              "desc": "Mirrors each trial across the sagittal midline and swaps its left/right label, generating anatomically valid copies that double the training set in two-class left/right motor imagery.",
              "ref": "Z. Wang†, S. Li†, ..., D. Wu*, Neural Networks, 2024",
              "doi": "10.1016/j.neunet.2024.106351",
              "naReason": "Channel Reflection requires a two-class left/right task. BNCI2014002 and BNCI2015001 are right hand versus both feet, and BNCI2014002 provides no anatomical montage.",
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "CSDA",
              "acc": {
                "BNCI2014001": {
                  "mean": 72.45,
                  "std": 1.87
                },
                "BNCI2014002": {
                  "mean": 73.55,
                  "std": 0.29
                },
                "BNCI2015001": {
                  "mean": 73.42,
                  "std": 1.1
                }
              },
              "delta": {
                "BNCI2014001": 0.38,
                "BNCI2014002": -0.85,
                "BNCI2015001": 0.23
              },
              "isBaseline": false,
              "isReference": false,
              "key": "CSDA-EEGNet",
              "lab": true,
              "code": "hustbciml/algorithms/augmenters/CSDA.py",
              "desc": "Cross-subject wavelet detail swap. It mixes the high-frequency wavelet details of same-class trials from different subjects, to synthesize new trials.",
              "ref": "Z. Wang, ..., D. Wu*, Knowl.-Based Syst., 2025",
              "doi": "10.1016/j.knosys.2025.113074",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "Frequency Shift",
              "acc": {
                "BNCI2014001": {
                  "mean": 73.28,
                  "std": 0.51
                },
                "BNCI2014002": {
                  "mean": 75.0,
                  "std": 0.38
                },
                "BNCI2015001": {
                  "mean": 74.14,
                  "std": 0.31
                }
              },
              "delta": {
                "BNCI2014001": 1.21,
                "BNCI2014002": 0.6,
                "BNCI2015001": 0.95
              },
              "isBaseline": false,
              "isReference": false,
              "key": "FShift-EEGNet",
              "lab": false,
              "code": "hustbciml/algorithms/augmenters/FShift.py",
              "desc": "Translates a trial's whole spectrum by a small frequency offset using the analytic (Hilbert) signal.",
              "ref": "D. Freer, G.-Z. Yang, J. Neural Eng., 2020",
              "doi": "10.1088/1741-2552/ab57c0",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "Fourier Surrogate",
              "acc": {
                "BNCI2014001": {
                  "mean": 73.28,
                  "std": 1.25
                },
                "BNCI2014002": {
                  "mean": 75.17,
                  "std": 0.58
                },
                "BNCI2015001": {
                  "mean": 72.88,
                  "std": 0.74
                }
              },
              "delta": {
                "BNCI2014001": 1.21,
                "BNCI2014002": 0.77,
                "BNCI2015001": -0.31
              },
              "isBaseline": false,
              "isReference": false,
              "key": "FSurr-EEGNet",
              "lab": false,
              "code": "hustbciml/algorithms/augmenters/FSurr.py",
              "desc": "Draws a surrogate trial with the same power spectrum as the original but randomized Fourier phase.",
              "ref": "J. T. C. Schwabedal et al., arXiv:1806.08675, 2018",
              "doi": null,
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "Frequency Recombination",
              "acc": {
                "BNCI2014001": {
                  "mean": 72.3,
                  "std": 2.11
                },
                "BNCI2014002": {
                  "mean": 73.81,
                  "std": 0.07
                },
                "BNCI2015001": {
                  "mean": 73.58,
                  "std": 0.88
                }
              },
              "delta": {
                "BNCI2014001": 0.23,
                "BNCI2014002": -0.59,
                "BNCI2015001": 0.39
              },
              "isBaseline": false,
              "isReference": false,
              "key": "FComb-EEGNet",
              "lab": false,
              "code": "hustbciml/algorithms/augmenters/FComb.py",
              "desc": "Splits each trial's cosine spectrum into contiguous bands and rebuilds a new trial by taking each band from a different same-class trial.",
              "ref": "X. Zhao et al., J. Neural Eng., 2022",
              "doi": "10.1088/1741-2552/aca04f",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "Additive Noise",
              "acc": {
                "BNCI2014001": {
                  "mean": 71.94,
                  "std": 0.66
                },
                "BNCI2014002": {
                  "mean": 74.14,
                  "std": 0.81
                },
                "BNCI2015001": {
                  "mean": 73.18,
                  "std": 1.02
                }
              },
              "delta": {
                "BNCI2014001": -0.13,
                "BNCI2014002": -0.26,
                "BNCI2015001": -0.01
              },
              "isBaseline": false,
              "isReference": false,
              "key": "Noise-EEGNet",
              "lab": false,
              "code": "hustbciml/algorithms/augmenters/Noise.py",
              "desc": "Copies each trial once, with zero-mean Gaussian noise added and scaled to the amplitude of the trial itself. The simplest label-preserving augmentation.",
              "ref": "D. Freer, G.-Z. Yang, J. Neural Eng., 2020",
              "doi": "10.1088/1741-2552/ab57c0",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "Amplitude Scaling",
              "acc": {
                "BNCI2014001": {
                  "mean": 72.97,
                  "std": 0.83
                },
                "BNCI2014002": {
                  "mean": 72.83,
                  "std": 0.38
                },
                "BNCI2015001": {
                  "mean": 72.72,
                  "std": 1.53
                }
              },
              "delta": {
                "BNCI2014001": 0.9,
                "BNCI2014002": -1.57,
                "BNCI2015001": -0.47
              },
              "isBaseline": false,
              "isReference": false,
              "key": "Scale-EEGNet",
              "lab": false,
              "code": "hustbciml/algorithms/augmenters/Scaling.py",
              "desc": "Copies each trial with its amplitude multiplied by a coefficient close to one. It is the augmentation component of the PAT pipeline.",
              "ref": "X. Chen, ..., D. Wu*, Fundamental Research, 2026",
              "doi": "10.1016/j.fmre.2026.04.034",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "Amplitude Flip",
              "acc": {
                "BNCI2014001": {
                  "mean": 70.6,
                  "std": 0.77
                },
                "BNCI2014002": {
                  "mean": 74.24,
                  "std": 0.03
                },
                "BNCI2015001": {
                  "mean": 73.28,
                  "std": 0.93
                }
              },
              "delta": {
                "BNCI2014001": -1.47,
                "BNCI2014002": -0.16,
                "BNCI2015001": 0.09
              },
              "isBaseline": false,
              "isReference": false,
              "key": "Flip-EEGNet",
              "lab": false,
              "code": "hustbciml/algorithms/augmenters/Flip.py",
              "desc": "Mirrors each channel vertically about its own maximum, adding one label-preserving copy per trial.",
              "ref": "D. Freer, G.-Z. Yang, J. Neural Eng., 2020",
              "doi": "10.1088/1741-2552/ab57c0",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "Half-Sample Recombination",
              "acc": {
                "BNCI2014001": {
                  "mean": 64.99,
                  "std": 0.61
                },
                "BNCI2014002": {
                  "mean": 61.07,
                  "std": 2.33
                },
                "BNCI2015001": {
                  "mean": 64.53,
                  "std": 1.14
                }
              },
              "delta": {
                "BNCI2014001": -4.35,
                "BNCI2014002": -0.83,
                "BNCI2015001": 1.07
              },
              "isBaseline": false,
              "isReference": false,
              "key": "HS-EEGNet",
              "lab": false,
              "code": "hustbciml/algorithms/augmenters/HS.py",
              "desc": "Splices the left-hemisphere and right-hemisphere channels of two same-class trials into a new trial, exploiting the lateralization of motor imagery. An electrode-space transform, applied to unaligned trials.",
              "ref": "Y. Pei et al., Front. Hum. Neurosci., 2021",
              "doi": "10.3389/fnhum.2021.645952",
              "naReason": null,
              "alsoVaries": null,
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
              "desc": "EA-aligned EEGNet trained without augmentation, which is the baseline for the augmenters applied to aligned trials. Channel Reflection is instead measured against the unaligned baseline, as it must be applied before whitening.",
              "ref": null,
              "doi": null,
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            }
          ]
        }
      ]
    },
    {
      "id": "network",
      "title": "Networks",
      "blurb": "The backbone stage. Only the deep network varies. The input remains Euclidean-aligned, and the objective remains supervised empirical risk minimization (ERM). All backbones share one training configuration, i.e., Adam with batch size 32 for at most 100 epochs, early-stopped on a 20% held-out split of the source subjects, and each network retains the architecture hyperparameters of its original paper. The learning rate is the only tuned hyperparameter. It is grid-searched for each backbone and selected by that held-out source validation accuracy, never on the target, so that no configuration is fitted to the test data. The baseline is EEGNet.",
      "groups": [
        {
          "subcat": null,
          "blurb": "",
          "baseline": "EEGNet",
          "reference": null,
          "rows": [
            {
              "name": "MVCNet",
              "acc": {
                "BNCI2014001": {
                  "mean": 75.75,
                  "std": 0.56
                },
                "BNCI2014002": {
                  "mean": 77.86,
                  "std": 1.07
                },
                "BNCI2015001": {
                  "mean": 74.75,
                  "std": 0.1
                }
              },
              "delta": {
                "BNCI2014001": 3.22,
                "BNCI2014002": 3.46,
                "BNCI2015001": 1.36
              },
              "isBaseline": false,
              "isReference": false,
              "key": "MVCNet",
              "lab": true,
              "code": "hustbciml/algorithms/strategies/MVCNet.py",
              "desc": "Multi-View Contrastive Network. An IFNet convolutional backbone trained with a multi-view contrastive objective. At the inference time, only the backbone and the linear head are used.",
              "ref": "Z. Wang, ..., D. Wu*, Knowl.-Based Syst., 2025",
              "doi": "10.1016/j.knosys.2025.114205",
              "naReason": null,
              "alsoVaries": "strategy (multi-view contrastive objective) and batch size 64, not the backbone alone.",
              "pinAfter": null
            },
            {
              "name": "DBConformer",
              "acc": {
                "BNCI2014001": {
                  "mean": 76.26,
                  "std": 0.84
                },
                "BNCI2014002": {
                  "mean": 77.19,
                  "std": 1.28
                },
                "BNCI2015001": {
                  "mean": 71.86,
                  "std": 0.23
                }
              },
              "delta": {
                "BNCI2014001": 3.73,
                "BNCI2014002": 2.79,
                "BNCI2015001": -1.53
              },
              "isBaseline": false,
              "isReference": false,
              "key": "EA-DBConformer",
              "lab": true,
              "code": "hustbciml/algorithms/models/DBConformer.py",
              "desc": "Dual-branch convolutional transformer with parallel temporal and spatial branches whose features are fused before classification.",
              "ref": "Z. Wang, ..., D. Wu*, IEEE J. Biomed. Health Inform., 2026",
              "doi": "10.1109/JBHI.2025.3622725",
              "naReason": null,
              "alsoVaries": null,
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
              "naReason": null,
              "alsoVaries": null,
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
              "desc": "EEGNet whose first temporal convolution is replaced by a time-information-enhanced convolution, which injects a fixed sinusoidal positional embedding into the signal. ⚠ Note: it was originally developed for seizure detection (Peng et al. 2022). This time-positional design targets seizure EEG, and may not be well suited to motor imagery.",
              "ref": "R. Peng, ..., D. Wu*, IEEE Trans. Neural Syst. Rehabil. Eng., 2022",
              "doi": "10.1109/TNSRE.2022.3204540",
              "naReason": null,
              "alsoVaries": null,
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
              "desc": "Knowledge-data fusion CNN mirroring FBCSP. A windowed-sinc FIR filter bank and per-band CSP spatial filters are knowledge-initialized on the aligned source, then fine-tuned end-to-end.",
              "ref": "X. Jiang, ..., D. Wu*, Inf. Sci., 2026",
              "doi": "10.1016/j.ins.2025.123001",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "MSCFormer",
              "acc": {
                "BNCI2014001": {
                  "mean": 75.67,
                  "std": 0.26
                },
                "BNCI2014002": {
                  "mean": 76.14,
                  "std": 1.21
                },
                "BNCI2015001": {
                  "mean": 73.44,
                  "std": 1
                }
              },
              "delta": {
                "BNCI2014001": 3.14,
                "BNCI2014002": 1.74,
                "BNCI2015001": 0.05
              },
              "isBaseline": false,
              "isReference": false,
              "key": "EA-MSCFormer",
              "lab": false,
              "code": "hustbciml/algorithms/models/MSCFormer.py",
              "desc": "Three parallel multi-scale temporal-convolution branches whose features are fused and passed to a transformer encoder.",
              "ref": "W. Zhao et al., Sci. Rep., 2025",
              "doi": "10.1038/s41598-025-96611-5",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "MSVTNet",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.82,
                  "std": 0.74
                },
                "BNCI2014002": {
                  "mean": 75.9,
                  "std": 1.07
                },
                "BNCI2015001": {
                  "mean": 73.17,
                  "std": 1.04
                }
              },
              "delta": {
                "BNCI2014001": 2.29,
                "BNCI2014002": 1.5,
                "BNCI2015001": -0.22
              },
              "isBaseline": false,
              "isReference": false,
              "key": "EA-MSVTNet",
              "lab": false,
              "code": "hustbciml/algorithms/models/MSVTNet.py",
              "desc": "Several parallel multi-scale EEGNet-style convolution branches followed by a transformer that mixes their tokens.",
              "ref": "K. Liu et al., IEEE J. Biomed. Health Inform., 2024",
              "doi": "10.1109/JBHI.2024.3450753",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "EEG-Deformer",
              "acc": {
                "BNCI2014001": {
                  "mean": 73.79,
                  "std": 1.78
                },
                "BNCI2014002": {
                  "mean": 75.02,
                  "std": 0.8
                },
                "BNCI2015001": {
                  "mean": 73.26,
                  "std": 0.09
                }
              },
              "delta": {
                "BNCI2014001": 1.26,
                "BNCI2014002": 0.62,
                "BNCI2015001": -0.13
              },
              "isBaseline": false,
              "isReference": false,
              "key": "EA-EEGDeformer",
              "lab": false,
              "code": "hustbciml/algorithms/models/EEGDeformer.py",
              "desc": "A dense convolutional transformer that interleaves shallow CNN encoders with coarse-to-fine transformer stages.",
              "ref": "Y. Ding et al., IEEE J. Biomed. Health Inform., 2025",
              "doi": "10.1109/JBHI.2024.3504604",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "EEGConformer",
              "acc": {
                "BNCI2014001": {
                  "mean": 72.84,
                  "std": 0.76
                },
                "BNCI2014002": {
                  "mean": 74.88,
                  "std": 0.59
                },
                "BNCI2015001": {
                  "mean": 73.43,
                  "std": 0.79
                }
              },
              "delta": {
                "BNCI2014001": 0.31,
                "BNCI2014002": 0.48,
                "BNCI2015001": 0.04
              },
              "isBaseline": false,
              "isReference": false,
              "key": "EA-EEGConformer",
              "lab": false,
              "code": "hustbciml/algorithms/models/EEGConformer.py",
              "desc": "Convolutional tokenizer followed by a transformer encoder.",
              "ref": "Y. Song et al., IEEE Trans. Neural Syst. Rehabil. Eng., 2023",
              "doi": "10.1109/TNSRE.2022.3230250",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "CTNet",
              "acc": {
                "BNCI2014001": {
                  "mean": 73.97,
                  "std": 0.8
                },
                "BNCI2014002": {
                  "mean": 74.79,
                  "std": 0.48
                },
                "BNCI2015001": {
                  "mean": 72.33,
                  "std": 0.37
                }
              },
              "delta": {
                "BNCI2014001": 1.44,
                "BNCI2014002": 0.39,
                "BNCI2015001": -1.06
              },
              "isBaseline": false,
              "isReference": false,
              "key": "EA-CTNet",
              "lab": false,
              "code": "hustbciml/algorithms/models/CTNet.py",
              "desc": "An EEGNet-style convolutional patch embedding feeding a transformer encoder.",
              "ref": "W. Zhao et al., Sci. Rep., 2024",
              "doi": "10.1038/s41598-024-71118-7",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "EEGNeX",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.61,
                  "std": 0.92
                },
                "BNCI2014002": {
                  "mean": 73.6,
                  "std": 0.59
                },
                "BNCI2015001": {
                  "mean": 72.32,
                  "std": 0.64
                }
              },
              "delta": {
                "BNCI2014001": 2.08,
                "BNCI2014002": -0.8,
                "BNCI2015001": -1.07
              },
              "isBaseline": false,
              "isReference": false,
              "key": "EA-EEGNeX",
              "lab": false,
              "code": "hustbciml/algorithms/models/EEGNeX.py",
              "desc": "A purely convolutional EEGNet variant that replaces the separable temporal convolutions with a stack of dilated convolutions for a wider receptive field.",
              "ref": "X. Chen et al., Biomed. Signal Process. Control, 2024",
              "doi": "10.1016/j.bspc.2023.105475",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "SlimSeiz",
              "acc": {
                "BNCI2014001": {
                  "mean": 69.65,
                  "std": 0.42
                },
                "BNCI2014002": {
                  "mean": 74.79,
                  "std": 1.61
                },
                "BNCI2015001": {
                  "mean": 72.94,
                  "std": 1.04
                }
              },
              "delta": {
                "BNCI2014001": -2.88,
                "BNCI2014002": 0.39,
                "BNCI2015001": -0.45
              },
              "isBaseline": false,
              "isReference": false,
              "key": "EA-SlimSeiz",
              "lab": false,
              "code": "hustbciml/algorithms/models/SlimSeiz.py",
              "desc": "A lightweight multi-branch 1D convolution feature extractor, paired with a single Mamba selective state space mixer. It was originally a seizure prediction network.",
              "ref": "G. Lu et al., IEEE Int. Symp. Circuits Syst., 2025",
              "doi": "10.1109/ISCAS56072.2025.11043364",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "TMSA-Net",
              "acc": {
                "BNCI2014001": {
                  "mean": 71.84,
                  "std": 1.26
                },
                "BNCI2014002": {
                  "mean": 73.67,
                  "std": 0.24
                },
                "BNCI2015001": {
                  "mean": 70.92,
                  "std": 0.78
                }
              },
              "delta": {
                "BNCI2014001": -0.69,
                "BNCI2014002": -0.73,
                "BNCI2015001": -2.47
              },
              "isBaseline": false,
              "isReference": false,
              "key": "EA-TMSANet",
              "lab": false,
              "code": "hustbciml/algorithms/models/TMSANet.py",
              "desc": "Sums two parallel multi-scale temporal convolutions, then applies a temporal multi-scale self-attention module.",
              "ref": "Q. Zhao, W. Zhu, Biomed. Signal Process. Control, 2025",
              "doi": "10.1016/j.bspc.2024.107189",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "ADFCNN",
              "acc": {
                "BNCI2014001": {
                  "mean": 72.17,
                  "std": 1.53
                },
                "BNCI2014002": {
                  "mean": 71.81,
                  "std": 0.38
                },
                "BNCI2015001": {
                  "mean": 71.62,
                  "std": 0.27
                }
              },
              "delta": {
                "BNCI2014001": -0.36,
                "BNCI2014002": -2.59,
                "BNCI2015001": -1.77
              },
              "isBaseline": false,
              "isReference": false,
              "key": "EA-ADFCNN",
              "lab": false,
              "code": "hustbciml/algorithms/models/ADFCNN.py",
              "desc": "Two parallel spectral-spatial pathways at different temporal scales, fused by a self-attention module.",
              "ref": "W. Tao et al., IEEE Trans. Neural Syst. Rehabil. Eng., 2024",
              "doi": "10.1109/TNSRE.2023.3342331",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "ShallowConvNet",
              "acc": {
                "BNCI2014001": {
                  "mean": 71.12,
                  "std": 1.05
                },
                "BNCI2014002": {
                  "mean": 70.88,
                  "std": 1.54
                },
                "BNCI2015001": {
                  "mean": 72.35,
                  "std": 0.65
                }
              },
              "delta": {
                "BNCI2014001": -1.41,
                "BNCI2014002": -3.52,
                "BNCI2015001": -1.04
              },
              "isBaseline": false,
              "isReference": false,
              "key": "EA-ShallowConvNet",
              "lab": false,
              "code": "hustbciml/algorithms/models/ShallowConvNet.py",
              "desc": "Shallow convolution-and-pooling network modeled on band-power features.",
              "ref": "R. T. Schirrmeister et al., Hum. Brain Mapp., 2017",
              "doi": "10.1002/hbm.23730",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "FBMSNet",
              "acc": {
                "BNCI2014001": {
                  "mean": 70.91,
                  "std": 0.95
                },
                "BNCI2014002": {
                  "mean": 71.9,
                  "std": 0.41
                },
                "BNCI2015001": {
                  "mean": 69.72,
                  "std": 0.96
                }
              },
              "delta": {
                "BNCI2014001": -1.62,
                "BNCI2014002": -2.5,
                "BNCI2015001": -3.67
              },
              "isBaseline": false,
              "isReference": false,
              "key": "EA-FBMSNet",
              "lab": false,
              "code": "hustbciml/algorithms/models/FBMSNet.py",
              "desc": "Decomposes the signal into a filter bank of narrow sub-bands, then applies mixed-scale depthwise temporal convolutions.",
              "ref": "K. Liu et al., IEEE Trans. Biomed. Eng., 2023",
              "doi": "10.1109/TBME.2022.3193277",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "DeepConvNet",
              "acc": {
                "BNCI2014001": {
                  "mean": 73.79,
                  "std": 0.46
                },
                "BNCI2014002": {
                  "mean": 69.05,
                  "std": 0.44
                },
                "BNCI2015001": {
                  "mean": 69.29,
                  "std": 0.47
                }
              },
              "delta": {
                "BNCI2014001": 1.26,
                "BNCI2014002": -5.35,
                "BNCI2015001": -4.1
              },
              "isBaseline": false,
              "isReference": false,
              "key": "EA-DeepConvNet",
              "lab": false,
              "code": "hustbciml/algorithms/models/DeepConvNet.py",
              "desc": "Deeper four-block convolutional network for EEG decoding.",
              "ref": "R. T. Schirrmeister et al., Hum. Brain Mapp., 2017",
              "doi": "10.1002/hbm.23730",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "EEGWaveNet",
              "acc": {
                "BNCI2014001": {
                  "mean": 66.64,
                  "std": 1.44
                },
                "BNCI2014002": {
                  "mean": 68.93,
                  "std": 1.49
                },
                "BNCI2015001": {
                  "mean": 68.94,
                  "std": 1.3
                }
              },
              "delta": {
                "BNCI2014001": -5.89,
                "BNCI2014002": -5.47,
                "BNCI2015001": -4.45
              },
              "isBaseline": false,
              "isReference": false,
              "key": "EA-EEGWaveNet",
              "lab": false,
              "code": "hustbciml/algorithms/models/EEGWaveNet.py",
              "desc": "A cascade of depthwise Conv1d layers repeatedly halves the sampling rate, to extract multi-scale temporal features. It was originally a seizure detector.",
              "ref": "P. Thuwajit et al., IEEE Trans. Ind. Inform., 2022",
              "doi": "10.1109/TII.2021.3133307",
              "naReason": null,
              "alsoVaries": null,
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
              "desc": "Compact convolutional network, consisting of a temporal convolution, a depthwise spatial convolution and a separable convolution. The default backbone of the benchmark.",
              "ref": "V. J. Lawhern et al., J. Neural Eng., 2018",
              "doi": "10.1088/1741-2552/aace8c",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            }
          ]
        }
      ]
    },
    {
      "id": "classical",
      "title": "Classical Pipelines",
      "blurb": "No backbone. These rows replace the deep network with a classical decoding pipeline, fitted on the same Euclidean-aligned trials without any gradient-based training. There is hence neither early stopping nor random initialization, and each row is deterministic, with an across-seed standard deviation of exactly zero. They vary more than one stage simultaneously, and hence are not a controlled comparison of a single stage. They are reported as a reference for the deep rows, and are compared with EA-EEGNet on each dataset.",
      "groups": [
        {
          "subcat": null,
          "blurb": "",
          "baseline": null,
          "reference": {
            "name": "EA-EEGNet",
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
              "name": "CSP-LDA",
              "acc": {
                "BNCI2014001": {
                  "mean": 73.77,
                  "std": 0.0
                },
                "BNCI2014002": {
                  "mean": 72.71,
                  "std": 0.0
                },
                "BNCI2015001": {
                  "mean": 72.0,
                  "std": 0.0
                }
              },
              "delta": {
                "BNCI2014001": 1.7,
                "BNCI2014002": -1.69,
                "BNCI2015001": -1.19
              },
              "isBaseline": false,
              "isReference": false,
              "key": "CSP-LDA",
              "lab": false,
              "code": "hustbciml/algorithms/strategies/CSP_LDA.py",
              "desc": "Common Spatial Pattern filters (10 components) followed by Linear Discriminant Analysis. The classical motor imagery baseline, which remains competitive with a deep network in cross-subject decoding.",
              "ref": "H. Ramoser et al., IEEE Trans. Rehabil. Eng., 2000",
              "doi": "10.1109/86.895946",
              "naReason": null,
              "alsoVaries": "the backbone and the head are both gone, and training is a direct fit with no gradient loop.",
              "pinAfter": null
            },
            {
              "name": "Riemann-MDM",
              "acc": {
                "BNCI2014001": {
                  "mean": 71.68,
                  "std": 0.0
                },
                "BNCI2014002": {
                  "mean": 69.57,
                  "std": 0.0
                },
                "BNCI2015001": {
                  "mean": 66.42,
                  "std": 0.0
                }
              },
              "delta": {
                "BNCI2014001": -0.39,
                "BNCI2014002": -4.83,
                "BNCI2015001": -6.77
              },
              "isBaseline": false,
              "isReference": false,
              "key": "Riemann-MDM",
              "lab": false,
              "code": "hustbciml/algorithms/strategies/RiemannMDM.py",
              "desc": "Each trial is represented by its spatial covariance matrix, and assigned to the nearest class mean under the affine-invariant Riemannian metric. No filter and no feature beyond the covariance is used.",
              "ref": "A. Barachant et al., IEEE Trans. Biomed. Eng., 2012",
              "doi": "10.1109/TBME.2011.2172210",
              "naReason": null,
              "alsoVaries": "the backbone and the head are both gone, and the trial is represented by its covariance matrix rather than by the time series.",
              "pinAfter": null
            }
          ]
        }
      ]
    },
    {
      "id": "transfer",
      "title": "Transfer Learning",
      "blurb": "The learning-objective stage. Every row uses the same Euclidean-aligned EEGNet, and only the training or adaptation objective varies. The families differ in when the unlabeled target data are used, and whether the source data are still available. Unsupervised domain adaptation replaces ERM with a joint objective, trained on the labeled source and the unlabeled target together. Source-free adaptation first trains an ERM source model, and then optimizes a second objective on the target alone, without access to the source data. Test-time adaptation also starts from an ERM source model, but updates it online, one incoming target batch at a time. Source-only approaches do not use the target at all. Each strategy retains the shared EA-EEGNet training configuration (Adam, batch size 32, learning rate 1e-3), and adds only its own loss trade-offs and adaptation steps, which are read from its preset. All are two-class on the three datasets, and measured against the same no-transfer baseline, ERM. Privacy-preserving transfer is the exception. It keeps the raw EEG of each subject local, and hence is measured against Centralized Training instead, as its own note describes.",
      "groups": [
        {
          "subcat": "Source-only",
          "blurb": "Trained on the labeled source subjects only. The target is never used for adaptation, and the inference is a single forward pass. The baseline is ERM.",
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
              "desc": "Replaces each training batch with a channel-scaled adversarial batch after a short clean warm-up, hardening the source-trained EEGNet against distribution shift. The target is not used during training.",
              "ref": "X. Chen, ..., D. Wu*, IEEE Trans. Neural Syst. Rehabil. Eng., 2024",
              "doi": "10.1109/TNSRE.2024.3391936",
              "naReason": null,
              "alsoVaries": null,
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
              "desc": "Extends adversarial training for privacy-preserving (source-only) transfer: after a clean warm-up each Euclidean-aligned batch is amplitude-scaled (×(1±0.05)) then perturbed by a global-ε L∞ PGD attack (noisy-initialized, eps 0.03, 10 steps), hardening the source-trained EEGNet against distribution shift. The target is never used in training.",
              "ref": "X. Chen, ..., D. Wu*, Fundamental Research, 2026",
              "doi": "10.1016/j.fmre.2026.04.034",
              "naReason": null,
              "alsoVaries": "augmenter (amplitude Scaling) as well as the training objective.",
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
              "desc": "Domain-paired first-order MAML across the source subjects. It meta-learns an initialization so that one adaptation step on any source subject lowers the loss on the others, then applies the meta-learned EEGNet to the target with no target fine-tuning.",
              "ref": "S. Li, ..., D. Wu*, IEEE Comput. Intell. Mag., 2022",
              "doi": "10.1109/MCI.2022.3199622",
              "naReason": null,
              "alsoVaries": null,
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
              "desc": "Standard supervised training on the source subjects with no adaptation. This is the no-transfer baseline every transfer family is measured against.",
              "ref": null,
              "doi": null,
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            }
          ]
        },
        {
          "subcat": "Unsupervised domain adaptation",
          "blurb": "Trained jointly on the labeled source and the unlabeled target, aligning the two distributions during the source training. No target label is used. Measured against the no-transfer baseline.",
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
              "desc": "Network-free manifold transfer, i.e., not an EEGNet model: per-subject covariance centroid alignment and Riemannian tangent-space features, then a jointly learned source/target subspace that minimizes the joint distribution shift while preserving the source discriminability and the target locality, refined by EM pseudo-labeling, into a shrinkage LDA. Deterministic.",
              "ref": "W. Zhang, D. Wu*, IEEE Trans. Neural Syst. Rehabil. Eng., 2020",
              "doi": "10.1109/TNSRE.2020.2985996",
              "naReason": null,
              "alsoVaries": "the whole neural pipeline: this is a network-free Riemannian and tangent-space approach, so the aligner is Identity, and the EEGNet backbone and the Linear head are unused. It is a context row rather than a one-stage change to the EA-EEGNet baseline.",
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
              "naReason": null,
              "alsoVaries": null,
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
              "naReason": null,
              "alsoVaries": null,
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
              "naReason": null,
              "alsoVaries": null,
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
              "naReason": null,
              "alsoVaries": null,
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
              "naReason": null,
              "alsoVaries": null,
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
              "naReason": null,
              "alsoVaries": null,
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
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            }
          ]
        },
        {
          "subcat": "Source-free adaptation",
          "blurb": "Adapts a source-trained model to the target, retaining no source data at the transfer time. Measured against the no-transfer baseline.",
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
              "naReason": null,
              "alsoVaries": "the whole neural pipeline: this is a network-free Riemannian and tangent-space approach, so the aligner is Identity, and the EEGNet backbone and the Linear head are unused. It is a context row rather than a one-stage change to the EA-EEGNet baseline.",
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
              "desc": "Freezes the source classifier head and adapts the feature extractor by minimizing a Tsallis-entropy prediction-uncertainty objective with a consistency-regularized auxiliary head. No source data at transfer time.",
              "ref": "K. Xia, ..., D. Wu*, IEEE Trans. Biomed. Eng., 2022",
              "doi": "10.1109/TBME.2022.3168570",
              "naReason": null,
              "alsoVaries": null,
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
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            }
          ]
        },
        {
          "subcat": "Test-time adaptation",
          "blurb": "Adapts online as the target trials arrive at the test time, updating the source-trained model without any target label. Measured against the no-transfer baseline.",
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
              "desc": "Online test-time adaptation. For each incoming target batch it updates an incremental Euclidean-Alignment reference and minimizes an information-maximization loss (conditional-entropy minimization with a marginal-diversity regularizer), then predicts. Plug-and-play, no target labels.",
              "ref": "S. Li, ..., D. Wu*, IEEE Trans. Biomed. Eng., 2024",
              "doi": "10.1109/TBME.2023.3303289",
              "naReason": null,
              "alsoVaries": null,
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
              "desc": "Backpropagation-free test-time adaptation: averages the model's predictions over label-preserving augmentations of each target trial, gaining robustness with no gradient updates. It is aimed at lightweight, low-power BCI hardware.",
              "ref": "S. Li†, J. Ouyang†, Z. Cui†, ..., D. Wu*, arXiv:2601.07556, 2026",
              "doi": "10.48550/arXiv.2601.07556",
              "naReason": null,
              "alsoVaries": null,
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
              "naReason": null,
              "alsoVaries": null,
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
              "desc": "Online test-time adaptation by information maximization, with intra-class tightening and inter-class separation on pseudo-labeled target features. The whole network is adapted over the target stream.",
              "ref": "X. Li et al., ACM MM, 2021",
              "doi": "10.1145/3474085.3475487",
              "naReason": null,
              "alsoVaries": null,
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
              "naReason": null,
              "alsoVaries": null,
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
              "naReason": null,
              "alsoVaries": null,
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
              "naReason": null,
              "alsoVaries": null,
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
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            }
          ]
        },
        {
          "subcat": "Privacy-preserving transfer",
          "blurb": "Cross-subject transfer that never pools the raw EEG. The data of each subject remain on their own device, so these approaches trade a small amount of accuracy for privacy, relative to Centralized Training, which pools all data. Two mechanisms are included. The federated approaches (FedAvg, and the lab's FedBS and SAFE) use a central server, which averages the per-subject model updates in each round and returns the shared model, so that only the model weights, and never the EEG, are transmitted. FedBS additionally keeps the batch normalization of each client local and seeks a flat minimum, and SAFE further adds adversarial robustness. Decentralized MSDT uses no server. Each source subject trains its own classifier, and only the trained models are shared and then fused on the target. All three datasets are two-class (chance 50%), so the columns are directly comparable. Δ is computed against Centralized Training on the same dataset.",
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
              "desc": "Federated learning that adds single-step adversarial feature training and a one-step adversarial weight perturbation on top of batch-specific BatchNorm, hardening the shared model without pooling the raw EEG. The adversarial regularization reduces the clean accuracy slightly on BNCI2014001, and raises the other two datasets above Centralized Training.",
              "ref": "T. Jia, ..., D. Wu*, arXiv:2601.05789, 2026",
              "doi": "10.48550/arXiv.2601.05789",
              "naReason": null,
              "alsoVaries": null,
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
              "desc": "Federated learning with batch-specific BatchNorm and sharpness-aware minimization, aggregating the per-subject model updates through a server without sharing the raw EEG. Under the same optimizer and learning rate as Centralized Training, it recovers essentially all of the centralized accuracy, so the privacy mechanism costs almost nothing here.",
              "ref": "T. Jia, ..., D. Wu*, IEEE Trans. Neural Syst. Rehabil. Eng., 2024",
              "doi": "10.1109/TNSRE.2024.3457504",
              "naReason": null,
              "alsoVaries": null,
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
              "desc": "Decentralized multi-source transfer on Riemannian tangent-space features, i.e., not an EEGNet model: each source subject trains its own classifier, and the target adapts and fuses them at the test time, with no source data pooled. It is close to Centralized Training on all three datasets, slightly above on BNCI2014001 and slightly below on the other two. The difference reflects the Riemannian representation and the test-time adaptation, rather than the privacy mechanism.",
              "ref": "W. Zhang, ..., D. Wu*, IEEE Trans. Neural Syst. Rehabil. Eng., 2022",
              "doi": "10.1109/TNSRE.2022.3207494",
              "naReason": null,
              "alsoVaries": "the whole neural pipeline: this is a network-free Riemannian and tangent-space approach, so the aligner is Identity, and the EEGNet backbone and the Linear head are unused. It is a context row rather than a one-stage change to the EA-EEGNet baseline.",
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
              "desc": "Federated averaging: each subject trains locally, and the server averages the model weights. The plain federated baseline, which isolates the two additions of FedBS.",
              "ref": "B. McMahan et al., AISTATS, 2017",
              "doi": null,
              "naReason": null,
              "alsoVaries": null,
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
              "desc": "EA-EEGNet trained on all source subjects pooled together. This is the non-private reference every privacy-preserving approach is measured against.",
              "ref": null,
              "doi": null,
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            }
          ]
        }
      ]
    },
    {
      "id": "ensemble",
      "title": "Ensemble Learning",
      "blurb": "The aggregation stage, in a fully decentralized and privacy-preserving setting. Each source subject trains three different learners, i.e., tangent-space logistic regression, CSP-Net and EEGConformer, on its own data alone, and shares only its hard predicted labels on the target, never the model weights or the raw EEG. A combiner then fuses the resulting (N−1)×3 label votes into a single prediction, without any target label. One learner is taken from each of three model families, i.e., a Riemannian linear model, a convolutional network and a self-attention network, so that the votes a single subject contributes are as mutually decorrelated as this menu allows. Because only the hard votes are observed, the task reduces to estimating the reliability of each learner in the absence of the ground truth. Two non-ensemble references bound the task, and the combiners are grouped below them.",
      "groups": [
        {
          "subcat": "Non-ensemble references",
          "blurb": "Decoding without any aggregation, to bound the ensemble approaches below. A single source learner applied to the target gives the lower reference. One model trained on all source subjects pooled together, i.e., Centralized Training, gives the non-private upper reference, which the privacy-preserving combiners approach without ever sharing the raw EEG.",
          "baseline": null,
          "reference": null,
          "rows": [
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
              "isBaseline": false,
              "isReference": false,
              "key": null,
              "lab": false,
              "code": null,
              "desc": "EA-aligned EEGNet trained on all source subjects pooled into a single model. The non-private, non-ensemble reference. It uses the raw data that the decentralized combiners are denied.",
              "ref": null,
              "doi": null,
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "single-source",
              "acc": {
                "BNCI2014001": {
                  "mean": 61.32,
                  "std": 0.35
                },
                "BNCI2014002": {
                  "mean": 57.94,
                  "std": 0.77
                },
                "BNCI2015001": {
                  "mean": 58.45,
                  "std": 0.23
                }
              },
              "delta": {
                "BNCI2014001": null,
                "BNCI2014002": null,
                "BNCI2015001": null
              },
              "isBaseline": false,
              "isReference": false,
              "key": null,
              "lab": false,
              "code": null,
              "desc": "Mean accuracy of one source learner applied to the target, averaged over all (N−1)×3 individual learners. The lower reference, before any cross-subject aggregation.",
              "ref": null,
              "doi": null,
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            }
          ]
        },
        {
          "subcat": "Ensemble learning",
          "blurb": "All combiners observe identical hard votes, and hence none of them has an information advantage. They differ only in how the reliability of each learner is estimated without labels. Majority voting weights all learners equally, and is the baseline. The spectral meta-learners weight each learner by the leading eigenvector of the vote agreement, which is an unsupervised estimate of the accuracy. SML is the binary form, and the lab's SML-OVR extends it to an arbitrary number of classes, so the binary SML is listed immediately below SML-OVR, as the two coincide on these two-class tasks. The crowd-labeling and truth-discovery aggregators (Dawid-Skene, EBCC, GLAD, and others) instead infer the confusion matrix or the skill of each learner from the agreement among the votes. StackingNet, another lab approach, learns the per-learner weights directly on the unlabeled target. Each combiner is measured against majority voting on the same dataset. All three datasets are two-class (chance 50%), so the columns are directly comparable.",
          "baseline": "Majority voting",
          "reference": null,
          "rows": [
            {
              "name": "SML-OVR",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.95,
                  "std": 0.41
                },
                "BNCI2014002": {
                  "mean": 73.38,
                  "std": 0.64
                },
                "BNCI2015001": {
                  "mean": 72.62,
                  "std": 0.15
                }
              },
              "delta": {
                "BNCI2014001": 1.39,
                "BNCI2014002": 1.43,
                "BNCI2015001": 2.01
              },
              "isBaseline": false,
              "isReference": false,
              "key": "SML-OVR",
              "lab": true,
              "code": "hustbciml/algorithms/ensembles/SMLOVR.py",
              "desc": "The lab's one-vs-rest spectral meta-learner, the multi-class generalization of the binary SML: for each class it runs the binary SML weight estimation on the one-hot votes and sums the per-class weightings, so it also handles more than two classes (for example the native four-class BNCI2014001, which the code still supports). On these two-class tasks it reduces exactly to the binary SML directly below, so the two report the identical accuracy here. The multi-class advantage shows only on native multi-class data.",
              "ref": "S. Li, ..., D. Wu*, IEEE Comput. Intell. Mag., 2026",
              "doi": "10.1109/MCI.2025.3624194",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "SML",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.95,
                  "std": 0.41
                },
                "BNCI2014002": {
                  "mean": 73.38,
                  "std": 0.64
                },
                "BNCI2015001": {
                  "mean": 72.62,
                  "std": 0.15
                }
              },
              "delta": {
                "BNCI2014001": 1.39,
                "BNCI2014002": 1.43,
                "BNCI2015001": 2.01
              },
              "isBaseline": false,
              "isReference": false,
              "key": "SML",
              "lab": false,
              "code": "hustbciml/algorithms/ensembles/SML.py",
              "desc": "Binary spectral meta-learner: weights each source model by the principal eigenvector of the ±1 vote covariance, which is an unsupervised estimate of the accuracy, valid for two classes. It is the binary form that the lab's SML-OVR above generalizes to an arbitrary number of classes. The two coincide on these two-class tasks, which is why they report the same accuracy and are listed together.",
              "ref": "F. Parisi et al., Proc. Natl. Acad. Sci. USA, 2014",
              "doi": "10.1073/pnas.1219097111",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": "SML-OVR"
            },
            {
              "name": "StackingNet",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.61,
                  "std": 0.71
                },
                "BNCI2014002": {
                  "mean": 72.55,
                  "std": 1.16
                },
                "BNCI2015001": {
                  "mean": 69.9,
                  "std": 0.74
                }
              },
              "delta": {
                "BNCI2014001": 1.05,
                "BNCI2014002": 0.6,
                "BNCI2015001": -0.71
              },
              "isBaseline": false,
              "isReference": false,
              "key": "StackingNet",
              "lab": true,
              "code": "hustbciml/algorithms/ensembles/StackingNet.py",
              "desc": "Unsupervised transductive meta-combiner over the hard labels of the source models: it learns per-model weights on the unlabeled target by consensus agreement, without any target label, initialized from the balanced accuracy of each model against the majority vote.",
              "ref": "S. Li†, C. Liu†, D. Wu*, Advanced Science, 2026",
              "doi": "10.1002/advs.76488",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "Dawid-Skene",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.92,
                  "std": 0.6
                },
                "BNCI2014002": {
                  "mean": 73.48,
                  "std": 1.3
                },
                "BNCI2015001": {
                  "mean": 72.82,
                  "std": 0.43
                }
              },
              "delta": {
                "BNCI2014001": 1.36,
                "BNCI2014002": 1.53,
                "BNCI2015001": 2.21
              },
              "isBaseline": false,
              "isReference": false,
              "key": "DawidSkene",
              "lab": false,
              "code": "hustbciml/algorithms/ensembles/DawidSkene.py",
              "desc": "Classic EM crowd-labeling aggregator: jointly estimates the full confusion matrix of each source model and the consensus label from the hard votes alone, without any target label.",
              "ref": "A. P. Dawid and A. M. Skene, J. R. Stat. Soc. C, 1979",
              "doi": "10.2307/2346806",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "LAA",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.97,
                  "std": 0.85
                },
                "BNCI2014002": {
                  "mean": 73.36,
                  "std": 0.38
                },
                "BNCI2015001": {
                  "mean": 72.62,
                  "std": 0.71
                }
              },
              "delta": {
                "BNCI2014001": 1.41,
                "BNCI2014002": 1.41,
                "BNCI2015001": 2.01
              },
              "isBaseline": false,
              "isReference": false,
              "key": "LAA",
              "lab": false,
              "code": "hustbciml/algorithms/ensembles/LAA.py",
              "desc": "Label-aware autoencoder: an unsupervised neural aggregator that encodes the per-trial votes into a consensus label and reconstructs each source model's vote from it.",
              "ref": "L. Yin, ..., IJCAI, 2017",
              "doi": "10.24963/ijcai.2017/184",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "EBCC",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.33,
                  "std": 0.74
                },
                "BNCI2014002": {
                  "mean": 71.17,
                  "std": 0.98
                },
                "BNCI2015001": {
                  "mean": 71.32,
                  "std": 0.56
                }
              },
              "delta": {
                "BNCI2014001": 0.77,
                "BNCI2014002": -0.78,
                "BNCI2015001": 0.71
              },
              "isBaseline": false,
              "isReference": false,
              "key": "EBCC",
              "lab": false,
              "code": "hustbciml/algorithms/ensembles/EBCC.py",
              "desc": "Enhanced Bayesian classifier combination: variational inference over low-rank worker-correlation groups. It is the most expressive of the crowd-aggregation baselines.",
              "ref": "Y. Li, B. Rubinstein, and T. Cohn, ICML, 2019",
              "doi": null,
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "Wawa",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.31,
                  "std": 0.82
                },
                "BNCI2014002": {
                  "mean": 71.26,
                  "std": 1.98
                },
                "BNCI2015001": {
                  "mean": 68.6,
                  "std": 1.25
                }
              },
              "delta": {
                "BNCI2014001": 0.75,
                "BNCI2014002": -0.69,
                "BNCI2015001": -2.01
              },
              "isBaseline": false,
              "isReference": false,
              "key": "Wawa",
              "lab": false,
              "code": "hustbciml/algorithms/ensembles/Wawa.py",
              "desc": "Worker-agreement-with-aggregate heuristic: weight each source model by its agreement with the plain majority vote, then re-vote. A crowd-kit heuristic with no separate paper.",
              "ref": "Worker Agreement With Aggregate, a crowd-kit heuristic",
              "doi": null,
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "PM",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.43,
                  "std": 0.63
                },
                "BNCI2014002": {
                  "mean": 67.88,
                  "std": 1.84
                },
                "BNCI2015001": {
                  "mean": 64.24,
                  "std": 1.13
                }
              },
              "delta": {
                "BNCI2014001": 0.87,
                "BNCI2014002": -4.07,
                "BNCI2015001": -6.37
              },
              "isBaseline": false,
              "isReference": false,
              "key": "PM",
              "lab": false,
              "code": "hustbciml/algorithms/ensembles/PM.py",
              "desc": "Truth-discovery aggregator: iteratively weights each source model by how much its votes agree with the current consensus (weight = -log of normalized disagreement), then re-estimates the consensus.",
              "ref": "Q. Li, ..., ACM SIGMOD, 2014",
              "doi": "10.1145/2588555.2610509",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "MACE",
              "acc": {
                "BNCI2014001": {
                  "mean": 70.76,
                  "std": 3.42
                },
                "BNCI2014002": {
                  "mean": 65.33,
                  "std": 2.49
                },
                "BNCI2015001": {
                  "mean": 68.82,
                  "std": 0.43
                }
              },
              "delta": {
                "BNCI2014001": -2.8,
                "BNCI2014002": -6.62,
                "BNCI2015001": -1.79
              },
              "isBaseline": false,
              "isReference": false,
              "key": "MACE",
              "lab": false,
              "code": "hustbciml/algorithms/ensembles/MACE.py",
              "desc": "Variational aggregator that separates competent labeling from per-model spamming, to down-weight unreliable source models.",
              "ref": "D. Hovy, ..., NAACL-HLT, 2013",
              "doi": null,
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "LA",
              "acc": {
                "BNCI2014001": {
                  "mean": 74.51,
                  "std": 0.62
                },
                "BNCI2014002": {
                  "mean": 66.26,
                  "std": 2.06
                },
                "BNCI2015001": {
                  "mean": 61.1,
                  "std": 1.76
                }
              },
              "delta": {
                "BNCI2014001": 0.95,
                "BNCI2014002": -5.69,
                "BNCI2015001": -9.51
              },
              "isBaseline": false,
              "isReference": false,
              "key": "LA",
              "lab": false,
              "code": "hustbciml/algorithms/ensembles/LA.py",
              "desc": "Lightweight two-pass aggregator: one online pass estimates each source model's ability under a Beta prior, a second pass re-votes weighted by that ability.",
              "ref": "Y. Yang, ..., ACM Trans. Knowl. Discov. Data, 2024",
              "doi": "10.1145/3630102",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "ZenCrowd",
              "acc": {
                "BNCI2014001": {
                  "mean": 71.5,
                  "std": 1.96
                },
                "BNCI2014002": {
                  "mean": 62.29,
                  "std": 1.34
                },
                "BNCI2015001": {
                  "mean": 56.39,
                  "std": 0.22
                }
              },
              "delta": {
                "BNCI2014001": -2.06,
                "BNCI2014002": -9.66,
                "BNCI2015001": -14.22
              },
              "isBaseline": false,
              "isReference": false,
              "key": "ZenCrowd",
              "lab": false,
              "code": "hustbciml/algorithms/ensembles/ZenCrowd.py",
              "desc": "EM aggregator with a single reliability scalar per source model, inferred from vote agreement alone (no target labels).",
              "ref": "G. Demartini, ..., WWW, 2012",
              "doi": "10.1145/2187836.2187900",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "M-MSR",
              "acc": {
                "BNCI2014001": {
                  "mean": 71.71,
                  "std": 0.86
                },
                "BNCI2014002": {
                  "mean": 60.02,
                  "std": 1.61
                },
                "BNCI2015001": {
                  "mean": 55.96,
                  "std": 0.24
                }
              },
              "delta": {
                "BNCI2014001": -1.85,
                "BNCI2014002": -11.93,
                "BNCI2015001": -14.65
              },
              "isBaseline": false,
              "isReference": false,
              "key": "MMSR",
              "lab": false,
              "code": "hustbciml/algorithms/ensembles/MMSR.py",
              "desc": "Recovers each source model's skill from the pairwise inter-model agreement matrix by robust rank-one matrix completion, then weights the vote by it.",
              "ref": "Q. Ma and A. Olshevsky, NeurIPS, 2020",
              "doi": null,
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "GLAD",
              "acc": {
                "BNCI2014001": {
                  "mean": 70.78,
                  "std": 3.23
                },
                "BNCI2014002": {
                  "mean": 59.5,
                  "std": 1.07
                },
                "BNCI2015001": {
                  "mean": 55.44,
                  "std": 1.29
                }
              },
              "delta": {
                "BNCI2014001": -2.78,
                "BNCI2014002": -12.45,
                "BNCI2015001": -15.17
              },
              "isBaseline": false,
              "isReference": false,
              "key": "GLAD",
              "lab": false,
              "code": "hustbciml/algorithms/ensembles/GLAD.py",
              "desc": "EM aggregator that jointly infers the consensus label, each source model's ability, and each trial's difficulty.",
              "ref": "J. Whitehill, ..., NeurIPS, 2009",
              "doi": null,
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            },
            {
              "name": "Majority voting",
              "acc": {
                "BNCI2014001": {
                  "mean": 73.56,
                  "std": 1.03
                },
                "BNCI2014002": {
                  "mean": 71.95,
                  "std": 1.48
                },
                "BNCI2015001": {
                  "mean": 70.61,
                  "std": 0.73
                }
              },
              "delta": {
                "BNCI2014001": null,
                "BNCI2014002": null,
                "BNCI2015001": null
              },
              "isBaseline": true,
              "isReference": false,
              "key": "Voting",
              "lab": false,
              "code": "hustbciml/algorithms/ensembles/Voting.py",
              "desc": "Plain majority vote over the hard predicted labels of the three per-subject learners across all source subjects. This is the label-only baseline every combiner is measured against.",
              "ref": "S. Li, ..., D. Wu*, IEEE Comput. Intell. Mag., 2026",
              "doi": "10.1109/MCI.2025.3624194",
              "naReason": null,
              "alsoVaries": null,
              "pinAfter": null
            }
          ]
        }
      ]
    }
  ]
};
