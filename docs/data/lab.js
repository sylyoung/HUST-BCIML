window.LAB = {
  "name": "HUST BCIML",
  "full_name": "Brain-Computer Interface & Machine Learning Laboratory",
  "pi": "Prof. Dongrui Wu",
  "institution": "Huazhong University of Science and Technology",
  "tagline": "Transfer learning, robustness, privacy, and fuzzy systems for EEG-based BCIs.",
  "links": {
    "lab_site": "https://lab.bciml.cn/",
    "pi_homepage": "https://sites.google.com/site/drwuhust/",
    "scholar": "https://scholar.google.com/citations?user=UYGzCPEAAAAJ"
  },
  "pi_photo": "assets/dongrui-wu.png",
  "maintainer": {
    "name": "Siyang Li",
    "homepage": "https://sylyoung.github.io/",
    "email": "lsyyoungll@gmail.com"
  },
  "repo_intro": "This repository is the lab's open-source code home: a unified, reproducible EEG-decoding benchmark, plus a paper-to-code gallery mapping the lab's publications to their released code.",
  "anchor": {
    "name": "HUST-BCIML: unified EEG-decoding benchmark",
    "owner": "sylyoung",
    "url": "https://github.com/sylyoung/HUST-BCIML",
    "blurb": "This repository. A unified, self-contained framework that reimplements 57 EEG-decoding approaches, spanning data alignment, data augmentation, network backbones, transfer learning, and ensemble aggregation, on one composable pipeline. It compares them head-to-head under a single controlled protocol on three MOABB motor-imagery EEG datasets, with per-method reproduction records."
  },
  "flagships": [
    {
      "pillar": "Transfer Learning",
      "name": "DeepTransferEEG",
      "owner": "sylyoung",
      "url": "https://github.com/sylyoung/DeepTransferEEG",
      "stars": 51,
      "blurb": "T-TIME test-time adaptation, Euclidean Alignment, and ~15 domain-adaptation methods for cross-subject EEG. This is the transfer-learning library the benchmark grew out of."
    },
    {
      "pillar": "Ensemble & Aggregation",
      "name": "TestEnsemble",
      "owner": "sylyoung",
      "url": "https://github.com/sylyoung/TestEnsemble",
      "stars": 2,
      "blurb": "Black-box test-time ensembling: majority voting, crowd-label aggregators, and the lab's SML-OVR / StackingNet combiners, fusing hard predictions across independent models."
    },
    {
      "pillar": "Deep Architectures",
      "name": "DBConformer",
      "owner": "wzwvv",
      "url": "https://github.com/wzwvv/DBConformer",
      "stars": 70,
      "blurb": "Dual-branch convolutional transformer for EEG decoding, covering motor imagery, seizure, and SSVEP."
    },
    {
      "pillar": "Robustness & Security",
      "name": "EEGAdversarialBenchmark",
      "owner": "xqchen914",
      "url": "https://github.com/xqchen914/EEGAdversarialBenchmark",
      "stars": 4,
      "blurb": "Benchmark of adversarial attacks and defenses for EEG-based BCIs."
    },
    {
      "pillar": "Transfer Learning",
      "name": "NT-Benchmark",
      "owner": "chamwen",
      "url": "https://github.com/chamwen/NT-Benchmark",
      "stars": 17,
      "blurb": "Negative-transfer benchmark: detecting and mitigating when transfer hurts in domain adaptation."
    },
    {
      "pillar": "Transfer Learning",
      "name": "TLBCI",
      "owner": "drwuHUST",
      "url": "https://github.com/drwuHUST/TLBCI",
      "stars": 72,
      "blurb": "Transfer learning for BCIs with non-deep (classical) machine learning, providing alignment and domain-adaptation baselines."
    },
    {
      "pillar": "Foundation Models",
      "name": "EEG-FM-Benchmark",
      "owner": "Dingkun0817",
      "url": "https://github.com/Dingkun0817/EEG-FM-Benchmark",
      "stars": 125,
      "blurb": "Benchmarking EEG foundation models."
    },
    {
      "pillar": "Foundation Models",
      "name": "MIRepNet",
      "owner": "staraink",
      "url": "https://github.com/staraink/MIRepNet",
      "stars": 122,
      "blurb": "Pre-trained EEG motor-imagery foundation model and pipeline."
    },
    {
      "pillar": "Fuzzy Systems & CWW",
      "name": "pytsk",
      "owner": "YuqiCui",
      "url": "https://github.com/YuqiCui/pytsk",
      "stars": 53,
      "blurb": "TSK fuzzy-system toolbox, scikit-learn compatible."
    },
    {
      "pillar": "Transfer Learning",
      "name": "MEKT",
      "owner": "chamwen",
      "url": "https://github.com/chamwen/MEKT",
      "stars": 47,
      "blurb": "Manifold Embedded Knowledge Transfer. The reference implementation of the classical MEKT method."
    },
    {
      "pillar": "Transfer Learning",
      "name": "EA",
      "owner": "hehe03",
      "url": "https://github.com/hehe03/EA",
      "stars": 27,
      "blurb": "Euclidean Alignment. The field-standard cross-subject alignment."
    },
    {
      "pillar": "Data Augmentation",
      "name": "CSDA",
      "owner": "wzwvv",
      "url": "https://github.com/wzwvv/CSDA",
      "stars": 20,
      "blurb": "Time-frequency EEG data augmentation."
    },
    {
      "pillar": "Fuzzy Systems & CWW",
      "name": "MBGD_RDA",
      "owner": "drwuHUST",
      "url": "https://github.com/drwuHUST/MBGD_RDA",
      "stars": 15,
      "blurb": "Mini-batch gradient descent with regularization for TSK fuzzy-system training."
    },
    {
      "pillar": "Active Learning",
      "name": "iGS",
      "owner": "drwuHUST",
      "url": "https://github.com/drwuHUST/iGS",
      "stars": 10,
      "blurb": "Improved greedy sampling for pool-based active learning."
    },
    {
      "pillar": "Speech (SEEG)",
      "name": "SACM",
      "owner": "WangHongbinary",
      "url": "https://github.com/WangHongbinary/SACM",
      "stars": 8,
      "blurb": "SEEG-audio contrastive matching for Chinese speech decoding."
    },
    {
      "pillar": "Privacy-Preserving BCI",
      "name": "FedBS",
      "owner": "TianwangJia",
      "url": "https://github.com/TianwangJia/FedBS",
      "stars": 7,
      "blurb": "Federated motor-imagery classification."
    },
    {
      "pillar": "Robustness & Security",
      "name": "bci_adv_defense",
      "owner": "lbinmeng",
      "url": "https://github.com/lbinmeng/bci_adv_defense",
      "stars": 4,
      "blurb": "Adversarial robustness harness for EEG BCIs."
    },
    {
      "pillar": "Biometrics",
      "name": "BrainprintNet",
      "owner": "hustmx721",
      "url": "https://github.com/hustmx721/BrainprintNet",
      "stars": 4,
      "blurb": "EEG-based brainprint recognition."
    },
    {
      "pillar": "Intracortical iBCI",
      "name": "SNN_iBCIs",
      "owner": "SongYang008",
      "url": "https://github.com/SongYang008/SNN_iBCIs",
      "stars": 3,
      "blurb": "Spiking neural networks for intra-cortical decoding."
    }
  ]
};
window.SITE = {"generated": "2026-07-24", "n_papers": 275, "n_code": 80, "n_methods": 57};
