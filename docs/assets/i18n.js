/* HUST BCIML — internationalization (i18n) dictionary.
   Loaded as a plain <script> BEFORE app.js. Defines window.I18N: a flat map
   from the EXACT English source string to its Simplified-Chinese translation.
   app.js looks a string up here via tr(s) when LANG === 'zh'; any string absent
   from this map falls back to the original English, by design.

   Style: natural, plain academic Chinese. No em-dashes (——), guillemet/quotation
   marks around terms, or semicolons; sentences are split with periods or joined
   with commas instead. Essential English terms and abbreviations are kept in
   parentheses on first use.

   Scope note: UI chrome + structural prose + controlled-vocabulary labels are
   translated here. Publication titles/authors/venues/TL;DRs, per-method one-line
   descriptions and citations, per-repo blurbs, repo names, URLs, file paths,
   DOIs, numbers and method names/keys are intentionally kept in English. */
(function () {
  "use strict";
  window.I18N = {
    zh: {
      /* ============ UI chrome: tabs, header/footer, buttons, search ============ */
      "Overview": "概览",
      "Benchmark": "基准测试",
      "Papers & Code": "论文与代码",
      "Lab site": "实验室主页",
      "Prof. Wu": "伍冬睿教授",
      "Scholar": "学术主页",
      "GitHub": "GitHub",
      "Maintainer": "维护者",
      "Lab website": "实验室主页",
      "Prof. Wu's homepage": "伍冬睿教授主页",
      "Repository": "代码仓库",
      "Benchmark and web app built and maintained by ": "基准测试与网页应用由 ",
      ". Prof. Wu's email is available in any of the lab's publications.":
        " 构建并维护。伍冬睿教授的邮箱可在实验室的任一篇论文中找到。",
      "Disclaimer: this benchmark reimplements both external baselines and the lab's own methods independently. The reported numbers — baseline reproductions and lab-method results alike — may differ from the original papers and can contain errors. Corrections are welcome; please contact the maintainer.":
        "免责声明。本基准独立地重新实现了外部基线方法和实验室自研方法。所报告的数值都可能与原论文有出入，也可能存在错误，基线复现结果和实验室方法结果都是如此。欢迎指正，也欢迎联系维护者。",

      /* ============ overview: official links, stats, section titles ============ */
      "Official lab presence": "实验室官方渠道",
      "Prof. Dongrui Wu": "伍冬睿教授",
      /* proper nouns rendered from lab.js (LAB.full_name / LAB.institution /
         maintainer.name); English in English mode, Chinese here. */
      "Brain-Computer Interface & Machine Learning Laboratory": "脑机接口与机器学习实验室",
      "Huazhong University of Science and Technology": "华中科技大学",
      "Siyang Li": "李思扬",
      "Google Scholar": "Google 学术",
      "Research area": "研究方向",
      "Papers": "论文数",
      "With code": "含代码",
      "lab approaches": "实验室方法",
      "approaches benchmarked": "已纳入基准的方法",
      "papers with code": "含代码的论文",
      "papers indexed": "已收录论文",
      "research areas": "研究方向",
      "BCI paradigms": "脑机接口范式",
      "Approaches in the benchmark": "基准中的全部方法",
      "Every approach evaluated in the benchmark, grouped by pipeline stage. The lab's own methods (Prof. Wu's group) are highlighted; the external baselines they are compared against are shown alongside.":
        "基准中评测的每一个方法，按流水线阶段分组。实验室（伍冬睿教授课题组）自己提出的方法以高亮样式显示，与它们对比的外部基线也一并列出。",
      "lab-proposed": "实验室提出",
      "external baseline": "外部基线",
      "Anchor project": "核心项目",
      "View the benchmark": "查看基准测试",
      "Benchmark code": "基准测试代码",
      "stars": "颗星",
      "citations": "次引用",
      "Featured code repositories": "精选代码仓库",
      "Browse the lab's work by area": "按研究方向浏览实验室成果",
      "Publications grouped by research area, with how many have released code. Open Papers & Code to search and filter; the official sites above hold the full publication list.":
        "论文按研究方向分组，并标注每个方向已开源代码的数量。打开论文与代码标签页可以检索和筛选，完整论文列表见上方官方网站。",

      /* ============ papers & code gallery ============ */
      "Papers & code gallery": "论文与代码总览",
      "The lab's publications, each linked to its released code where available. Showing the ":
        "实验室的论文，凡有开源代码的都已给出链接。当前显示 ",
      " with public code": " 篇含公开代码的论文",
      "; untick “has code” for all ": "，取消勾选含代码即可查看全部 ",
      ". The complete, authoritative publication list is on the ":
        " 篇。完整且权威的论文列表见",
      "lab website": "实验室主页",
      " and ": "与",
      ".": "。",
      "Search title, authors, venue, summary…": "检索标题、作者、发表期刊或会议、摘要…",
      "has code": "含代码",
      "Show all": "显示全部",
      "BCI paradigm": "脑机接口范式",
      "No papers match these filters.": "没有符合当前筛选条件的论文。",
      /* dynamic count: "showing " + n + " of " + m */
      "showing": "显示",
      "of": "/ 共",
      /* paper card link labels */
      "code": "代码",
      "paper": "论文",
      "no code": "无代码",
      "more": "展开",
      "less": "收起",

      /* ============ benchmark: library intro, datasets, guide ============ */
      "The benchmark": "基准测试",
      "A unified, reproducible EEG-decoding benchmark": "统一、可复现的脑电（EEG）解码基准",
      "An algorithm is a named composition of plug-in stages — an aligner, an augmenter and a backbone, trained under a chosen learning objective, with an optional ensemble that aggregates several models. Every controlled comparison varies one stage while the rest stay fixed.":
        "一个算法是若干可插拔阶段的具名组合，包括一个对齐器、一个数据增强器和一个骨干网络，在选定的学习目标下训练，还可以选择加入一个融合多个模型的集成模块。每一次受控对比只改变其中一个阶段，其余阶段保持固定。",
      /* pipeline-diagram stage labels + connector (benchmark library intro) */
      "Aligner": "对齐器",
      "Augmenter": "数据增强器",
      "Backbone": "骨干网络",
      "Learning objective": "学习目标",
      "trained under": "训练目标为",
      /* benchmark table link titles + inline connector */
      "Open ": "打开 ",
      "Open the paper": "打开论文",
      "chance": "随机水平",
      "README": "README",
      "RESULTS.md": "RESULTS.md",
      "References (BibTeX)": "参考文献（BibTeX）",
      "Datasets": "数据集",
      /* dynamic datasets intro: "The benchmark spans " + N + " MOABB…" */
      "The benchmark spans ": "本基准涵盖 ",
      " MOABB motor-imagery EEG datasets, all evaluated cross-subject (leave-one-subject-out). Accuracies are comparable only within the same dataset and class count.":
        " 个 MOABB 运动想象脑电数据集，全部采用跨被试的留一被试交叉验证（leave-one-subject-out）来评估。准确率只有在同一数据集且类别数相同时才可比较。",
      "Dataset": "数据集",
      "Subjects": "被试数",
      "Channels": "通道数",
      "Rate": "采样率",
      "Classes": "类别数",
      "Chance": "随机水平",
      "Trials/subj": "试次/被试",
      /* per-dataset `role` descriptions (meta.datasets[i].role) */
      "Left vs right hand (two-class, chance 50%) for every table, including the privacy-preserving and ensemble families. The native dataset is four-class (both hands, feet, tongue); the benchmark uses its two-class left/right subset throughout, and the four-class variant stays available in code.":
        "在所有表格中，任务均为左手对右手的二分类，随机水平 50%，隐私保护方法族和集成方法族也不例外。该数据集原生为四分类（双手、双脚和舌头），但本基准全程只使用其左右手的二分类子集，四分类版本仍保留在代码中可供使用。",
      "Right hand vs feet, 14 subjects, 100 training-run trials each. Two-class (chance 50%) throughout.":
        "右手对双脚，14 名被试，每名被试 100 个训练轮次试次。全程为二分类，随机水平 50%。",
      "Right hand vs feet, 12 subjects, 200 first-session trials each. Two-class (chance 50%) throughout.":
        "右手对双脚，12 名被试，每名被试 200 个首次会话试次。全程为二分类，随机水平 50%。",
      "Controlled-comparison leaderboard": "受控对比排行榜",
      "How to read this leaderboard": "如何阅读本排行榜",
      "Each table varies one stage of the pipeline and holds the rest at the default — Euclidean-aligned trials, an EEGNet backbone, standard supervised training — so every row differs from its baseline in exactly one way. The three columns are the three datasets; beneath each accuracy, Δ is the change versus that dataset's baseline. Every table is two-class (chance 50%) on all three datasets — the pipeline-stage tables, the source-only, unsupervised-adaptation, source-free and test-time transfer families, the privacy-preserving family, and the ensemble-learning table — so the columns are directly comparable throughout. Each family is measured against its own baseline: the transfer families against ERM, the privacy-preserving family against Centralized Training, and the ensemble table against majority voting. Every row links to its implementation and its paper.":
        "每张表只改变流水线中的一个阶段，其余保持默认配置，也就是经欧氏对齐（Euclidean alignment）的试次、EEGNet 骨干网络和标准监督训练，因此每一行与它的基线之间恰好只有一处不同。三列对应三个数据集，每个准确率下方的 Δ 表示相对该数据集基线的变化量。在全部三个数据集上，每张表都是二分类，随机水平 50%，包括各流水线阶段表，仅源域、无监督自适应、无源域和测试时这几类迁移方法族，隐私保护方法族，以及集成学习表，因此各列在全程都可以直接比较。每个方法族都以各自的基线来衡量，迁移各方法族以经验风险最小化（ERM）为基线，隐私保护方法族以集中式训练为基线，集成表以多数投票为基线。每一行都链接到它的实现代码和对应论文。",

      /* ---- ensemble per-dataset context cards ---- */
      "single-source": "单源",
      "Centralized Training": "集中式训练",
      "majority voting": "多数投票",

      /* ---- reference / baseline row labels inside leaderboard tables ---- */
      "Approach": "方法",
      "baseline": "基线",
      "reference": "参照",
      "lab": "实验室",
      "n/a": "不适用",

      /* ============ benchmark table titles & blurbs ============ */
      "Data Alignment": "数据对齐",
      "The aligner stage. Each aligner normalizes a subject's trials into a common statistical frame before the backbone sees them, shrinking the between-subject covariance shift that otherwise dominates cross-subject decoding. It is label-free and applied per subject; the backbone and its training stay fixed, and the baseline applies no alignment.":
        "对齐器阶段。每个对齐器在骨干网络处理之前，把某一名被试的试次归一化到共同的统计框架中，从而缩减被试之间的协方差偏移。如果不做处理，这种偏移会成为跨被试解码困难的主要来源。对齐不需要标签，按被试逐一施加。此时骨干网络及其训练保持固定，而基线不做任何对齐。",
      "Data Augmentation": "数据增强",
      "The augmenter stage. Each augmenter synthesizes extra training trials to regularize the same backbone, which is otherwise trained identically; each augmenter is compared against that backbone trained without it. Channel Reflection is an electrode-space transform and must precede any spatial whitening, so it runs on unaligned trials; CSDA operates on the Euclidean-aligned trials.":
        "数据增强器阶段。每个增强器合成额外的训练试次来正则化同一个骨干网络，除此之外训练过程完全相同。每个增强器都与没有使用它的同一骨干网络作比较。通道反射（Channel Reflection）是一种电极空间的变换，必须在任何空间白化之前进行，所以它作用于未对齐的试次。CSDA 则作用于经欧氏对齐的试次。",
      "Networks": "网络骨干",
      "The backbone stage. Vary the deep network on the same Euclidean-aligned trials, holding alignment and the learning objective fixed; each backbone is trained with its own learning rate and schedule, selected on held-out source-subject validation data. Baseline: EEGNet. MVCNet is a composite — an IFNet backbone trained with a multi-view contrastive objective — shown here among the backbones.":
        "骨干网络阶段。在相同的经欧氏对齐试次上更换深度网络，同时固定对齐方式和学习目标。每个骨干网络使用各自的学习率和训练计划，都在留出的源被试验证数据上选定。基线是 EEGNet。MVCNet 是一个复合方法，以 IFNet 为骨干、在多视图对比目标下训练，这里把它列在各骨干网络当中。",
      "Transfer Learning": "迁移学习",
      "The learning-objective stage. Every row is the identical Euclidean-aligned EEGNet; only the training or adaptation objective changes. Approaches are grouped by how much of the target they use and when — source-only, unsupervised domain adaptation, source-free, and test-time — all two-class on the three datasets and measured against the same no-transfer baseline (ERM). The privacy-preserving family keeps each subject's raw EEG local and is measured against Centralized Training rather than ERM (see its note).":
        "学习目标阶段。每一行都是完全相同的、经欧氏对齐的 EEGNet，只改变训练或自适应目标。各方法按照使用目标域数据的多少和时机来分组，分为仅源域、无监督域自适应（unsupervised domain adaptation）、无源域（source-free）和测试时（test-time）。它们在三个数据集上都是二分类，并以同一个无迁移基线（经验风险最小化 ERM）为基准来衡量。隐私保护方法族把每名被试的原始脑电保留在本地，以集中式训练而非 ERM 为基准来衡量，详见其说明。",
      "Ensemble Learning": "集成学习",
      "The aggregation stage, in a fully decentralized privacy setting. Five heterogeneous learners — tangent-space LDA, tangent-space SVM, EEGNet, ShallowConvNet, and CSPNet — are trained on each source subject's data alone, and the subjects share only their hard predicted labels on the target, never model weights or raw EEG. A post-hoc combiner fuses the (N-1)×5 label votes into a consensus prediction. Every combiner sees the same hard votes, so none has an information advantage; they differ only in how they weight and combine the votes. StackingNet is lab-proposed; SML-OVR is the lab's multi-class combiner, a one-vs-rest generalization of the binary SML that on these two-class tasks reduces exactly to it, so the two report the same accuracy and are placed together. The others (the binary SML and the crowd-labelling and truth-discovery aggregators) are established baselines. All three datasets are two-class (chance 50%), so the columns are directly comparable, and each combiner is measured against plain majority voting on the same dataset. Rows are ordered lab-proposed first, then the remaining combiners by accuracy, with plain majority voting (the baseline) last.":
        "聚合阶段，在完全去中心化的隐私场景下进行。为每一名源被试单独训练五个异构学习器，分别是切空间 LDA、切空间 SVM、EEGNet、ShallowConvNet 和 CSPNet，每个学习器只用该被试自己的数据训练。各被试之间只共享其在目标域上的硬预测标签，绝不共享模型权重或原始脑电。每个目标试次因此得到 (N-1)×5 个硬投票，由一个事后组合器把这些投票融合成一个共识预测。每个组合器看到的硬投票完全相同，因此没有谁具有信息优势，它们的区别只在于如何对投票加权和组合。StackingNet 由实验室提出。SML-OVR 是实验室的多分类组合器，是二分类 SML 的一对多（one-vs-rest）推广，在这些二分类任务上会精确退化为二分类 SML，因此两者报告相同的准确率并相邻排列。其余方法，包括二分类的 SML 以及群体标注和真值发现类的聚合方法，都是成熟的基线。三个数据集都是二分类，随机水平 50%，因此各列可以直接比较，每个组合器都以同数据集上的普通多数投票为基准来衡量。行的排序是实验室方法在前，其余组合器按准确率排列，普通多数投票作为基线放在最后。",

      /* ============ benchmark transfer sub-category (subcat) headers & blurbs ============ */
      "Source-only": "仅源域",
      "Trained on the labelled source subjects only; the target is never used for adaptation, and inference is a plain forward pass. Baseline: ERM.":
        "只在有标签的源被试上训练，目标域始终不用于自适应，推断就是一次普通的前向传播。基线是经验风险最小化（Empirical Risk Minimization，ERM）。",
      "Unsupervised domain adaptation": "无监督域自适应",
      "Trained jointly on the labelled source and the unlabelled target, aligning the two distributions during source training; no target labels are used. Measured against the no-transfer baseline.":
        "在有标签的源域和无标签的目标域上联合训练，在源域训练过程中对齐两个分布，不使用任何目标域标签。以无迁移基线为基准来衡量。",
      "Source-free adaptation": "无源域自适应",
      "Adapts a source-trained model to the target while keeping no source data at transfer time. Measured against the no-transfer baseline.":
        "在迁移阶段不保留任何源域数据的前提下，把一个已在源域训练的模型自适应到目标域。以无迁移基线为基准来衡量。",
      "Test-time adaptation": "测试时自适应",
      "Adapts online as the target trials arrive at test time, updating the source-trained model without target labels. Measured against the no-transfer baseline.":
        "在测试时随着目标试次的到来进行在线自适应，在不使用目标域标签的情况下更新已在源域训练的模型。以无迁移基线为基准来衡量。",
      "Privacy-preserving": "隐私保护",
      "These approaches never pool raw EEG across subjects — each subject's data stays local — the privacy-preserving counterpart to Centralized Training (the reference). FedBS, SAFE and FedAvg share model updates through a server (federated); MSDT shares only per-source models fused at test time (decentralized). All three datasets are two-class (chance 50%), so the columns are directly comparable. Δ is versus Centralized Training on the same dataset.":
        "这些方法从不跨被试汇集原始脑电，每名被试的数据都保留在本地，是集中式训练（作为参照）的隐私保护版本。FedBS、SAFE 和 FedAvg 通过服务器共享模型更新，属于联邦式（federated）。MSDT 只共享各源域模型，并在测试时融合，属于去中心化（decentralized）。三个数据集都是二分类，随机水平 50%，因此各列可以直接比较。Δ 表示相对同数据集上集中式训练的差值。",

      /* ============ lab.js structural prose ============ */
      /* lab.tagline */
      "Transfer learning, robustness, privacy, and fuzzy systems for EEG-based BCIs.":
        "面向脑电（EEG）脑机接口（BCI）的迁移学习、鲁棒性、隐私保护与模糊系统研究。",
      /* lab.repo_intro */
      "This repository is the lab's open-source code home: a unified, reproducible EEG-decoding benchmark, plus a paper-to-code gallery mapping the lab's publications to their released code.":
        "本仓库是实验室的开源代码主页，包含一个统一、可复现的脑电解码基准，以及一个把实验室论文映射到其已发布代码的论文到代码总览。",
      /* anchor.blurb */
      "This repository. A unified, self-contained framework that reimplements 39 EEG-decoding approaches — data alignment, data augmentation, network backbones, transfer learning, and ensemble aggregation — on one composable pipeline, and compares them head-to-head under a single controlled protocol on three MOABB motor-imagery EEG datasets, with per-method reproduction records.":
        "即本仓库。这是一个统一、自包含的框架，在同一条可组合流水线上重新实现了 39 种脑电解码方法，涵盖数据对齐、数据增强、网络骨干、迁移学习和集成聚合，并在三个 MOABB 运动想象脑电数据集上依据单一受控协议对它们正面比较，同时提供逐方法的复现记录。",

      /* ---- flagship repo `pillar` labels (controlled vocabulary) ----
         "Transfer Learning" and "Data Augmentation" reuse the benchmark-table
         titles above; "Deep Architectures", "Robustness & Security",
         "Privacy-Preserving BCI" and "Active Learning" reuse the topic keys below. */
      "Ensemble & Aggregation": "集成与聚合",
      "Foundation Models": "基础模型",
      "Fuzzy Systems & CWW": "模糊系统与词计算",
      "Speech (SEEG)": "言语解码（SEEG）",
      "Biometrics": "生物特征识别",
      "Intracortical iBCI": "皮层内植入式脑机接口（iBCI）",

      /* ============ publications controlled vocabulary: research-area topics ============
         (exact `topic` values as they appear in docs/data/publications.js) */
      "Transfer Learning & Alignment": "迁移学习与对齐",
      "Robustness & Security": "鲁棒性与安全",
      "Privacy-Preserving BCI": "隐私保护脑机接口",
      "Deep Architectures": "深度网络架构",
      "Data Augmentation & Generation": "数据增强与生成",
      "Foundation & Self-Supervised Models": "基础模型与自监督模型",
      "Fuzzy Systems & Computing-with-Words": "模糊系统与词计算",
      "General ML & Optimization": "通用机器学习与优化",
      "Active Learning": "主动学习",

      /* ============ publications controlled vocabulary: BCI paradigm tags ============ */
      "MI": "运动想象（MI）",
      "P300": "P300",
      "SSVEP": "稳态视觉诱发电位（SSVEP）",
      "Seizure": "癫痫发作",
      "Affect": "情感",
      "Speech": "言语",
      "Biometric": "生物特征",
      "Drowsy": "疲劳驾驶",
      "Sleep": "睡眠",
      "iBCI": "植入式脑机接口（iBCI）"
    }
  };

  /* Convenience helpers (optional; app.js keeps its own tr()/LANG for robustness). */
  window.LANG = (function () {
    try { return localStorage.getItem("lang") || "en"; } catch (e) { return "en"; }
  })();
})();
