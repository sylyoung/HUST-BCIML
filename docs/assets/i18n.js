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
      "Disclaimer: this benchmark reimplements both external baselines and the lab's own methods independently. The reported numbers, both baseline reproductions and lab-method results, may differ from the original papers and can contain errors. Corrections are welcome, so please contact the maintainer.":
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
      "Every approach evaluated in the benchmark, grouped by pipeline stage. The lab's own methods (Prof. Wu's group) are highlighted, and the external baselines they are compared against are shown alongside.":
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
      "Publications grouped by research area, with how many have released code. Open Papers & Code to search and filter. The official sites above hold the full publication list.":
        "论文按研究方向分组，并标注每个方向已开源代码的数量。打开论文与代码标签页可以检索和筛选，完整论文列表见上方官方网站。",

      /* ============ papers & code gallery ============ */
      "Papers & code gallery": "论文与代码总览",
      "The lab's publications, each linked to its released code where available. Showing the ":
        "实验室的论文，凡有开源代码的都已给出链接。当前显示 ",
      " with public code": " 篇含公开代码的论文",
      ". Untick the code filter to see all ": "。取消勾选代码筛选即可查看全部 ",
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
      "Every algorithm here is built from the same plug-in stages: an aligner, an augmenter, and a backbone, trained under one learning objective and optionally wrapped in an ensemble. A controlled comparison changes a single stage and holds the rest fixed, so any shift in accuracy traces back to that one change.":
        "这里每个算法都由同一组可组合的模块搭成，依次是一个对齐器、一个数据增强器和一个骨干网络，在某个学习目标下训练，需要时再加一层集成。受控对比每次只替换其中一个模块，其余保持不变，于是准确率的任何变化都能归到这一处改动上。",
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
        " 个 MOABB 运动想象脑电数据集，全部采用跨被试的留一被试交叉验证（leave-one-subject-out）评估。准确率只能在同一个数据集、且类别数相同的前提下相互比较。",
      "Dataset": "数据集",
      "Subjects": "被试数",
      "Channels": "通道数",
      "Rate": "采样率",
      "Classes": "类别数",
      "Chance": "随机水平",
      "Trials/subj": "试次/被试",
      /* per-dataset `role` descriptions (meta.datasets[i].role) */
      "Left vs right hand (two-class, chance 50%) for every table, including the privacy-preserving and ensemble families. The native dataset is four-class (both hands, feet, tongue). The benchmark uses its two-class left/right subset throughout, and the four-class variant stays available in code.":
        "在所有表格中，任务均为左手对右手的二分类，随机水平 50%，隐私保护方法族和集成方法族也不例外。该数据集原生为四分类（双手、双脚和舌头），但本基准全程只使用其左右手的二分类子集，四分类版本仍保留在代码中可供使用。",
      "Right hand vs feet, 14 subjects, 100 training-run trials each. Two-class (chance 50%) throughout.":
        "右手对双脚，14 名被试，每名被试 100 个训练轮次试次。全程为二分类，随机水平 50%。",
      "Right hand vs feet, 12 subjects, 200 first-session trials each. Two-class (chance 50%) throughout.":
        "右手对双脚，12 名被试，每名被试 200 个首次会话试次。全程为二分类，随机水平 50%。",
      "Controlled-comparison leaderboard": "受控对比排行榜",
      "How to read this leaderboard": "如何阅读本排行榜",
      "Read each row against its table's baseline. A table changes just one stage of the pipeline and holds the rest at the default: Euclidean-aligned trials, an EEGNet backbone, plain supervised training. So every row differs from the baseline in exactly one way, and any change in accuracy is down to that stage. The three columns are the three datasets. Under each accuracy, Δ is the gain or loss against that dataset's baseline. Every table is two-class (chance 50%) on all three datasets, so the columns stay comparable throughout. Each family has its own baseline: the transfer families use ERM, the privacy-preserving family uses Centralized Training, and the ensemble table uses majority voting. Every row links to its code and its paper.":
        "每一行都要对照它所在表的基线来看。一张表只改动流水线中的一个阶段，其余保持默认配置，也就是经欧氏对齐的试次、EEGNet 骨干网络和普通监督训练。于是每一行与基线只有一处不同，准确率的任何变化都来自这个阶段。三列对应三个数据集。每个准确率下方的 Δ，是相对该数据集基线的增减。三个数据集上每张表都是二分类，随机水平 50%，因此各列自始至终可比。每一类方法都有自己的基线。迁移各方法族用经验风险最小化（ERM），隐私保护方法族用集中式训练，集成表用多数投票。每一行都链接到它的代码和论文。",

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
      "The aligner stage. Before the backbone sees anything, the aligner recenters each subject's trials into a shared statistical frame. This shrinks the between-subject covariance shift that otherwise dominates cross-subject decoding. Alignment needs no labels and runs per subject. The backbone and its training stay fixed, and the baseline aligns nothing.":
        "对齐器阶段。在骨干网络接触数据之前，对齐器先把每名被试的试次重新对齐到一个共同的统计框架里。这样就压低了被试之间的协方差偏移。不做对齐时，这种偏移正是跨被试解码困难的主因。对齐不需要标签，按被试逐一进行。此时骨干网络和它的训练保持不变，基线则完全不做对齐。",
      "Data Augmentation": "数据增强",
      "The augmenter stage. Each augmenter synthesizes extra training trials to regularize an otherwise-identical backbone, and is measured against that same backbone trained without it. The augmenters fall into two regimes by where they act. The electrode-space transforms (Channel Reflection and Half-Sample Recombination) rearrange channels, so they must run before any spatial whitening, on unaligned trials, and are compared to the unaligned baseline. The signal- and frequency-domain augmenters act on Euclidean-aligned trials, and are compared to the aligned baseline.":
        "数据增强器阶段。每个增强器都会合成额外的训练试次，对骨干网络起到正则化作用。作为对照的骨干网络除了不做增强，其余设置完全一致。这些增强器按作用位置分成两类。电极空间变换会重排通道，包括通道反射（Channel Reflection）和半样本重组（Half-Sample Recombination）。它们作用于未对齐的试次，且必须在任何空间白化之前进行，因此以未对齐基线为参照。信号域和频率域的增强器则作用于经欧氏对齐的试次，以对齐后的基线为参照。",
      "Networks": "网络骨干",
      "The backbone stage. Only the deep network changes; the input stays Euclidean-aligned and the objective stays plain supervised ERM. Every backbone shares one training setup, Adam with batch size 32 for up to 100 epochs, stopped early on a 20% held-out slice of the source subjects, and each network keeps its own architecture hyperparameters from its original paper. The one tuned knob is the learning rate: it is grid-searched per backbone and chosen by that held-out-source validation accuracy, never the target, so no configuration is fit to the test set. The baseline is EEGNet.":
        "骨干网络阶段。这里只更换深度网络，输入始终经欧氏对齐，学习目标始终是普通的监督式经验风险最小化（ERM）。所有骨干网络共用同一套训练设置，即 Adam 优化器、批大小 32、最多训练 100 轮，并在留出的 20% 源被试上按验证结果提前停止。每个网络自身的结构超参数则沿用其原论文的取值。唯一需要调的旋钮是学习率，它对每个骨干网络在一个网格上搜索，并按刚才那部分留出源被试的验证准确率来选，绝不看目标域，因此没有任何配置是照着测试集调出来的。基线是 EEGNet。",
      "Transfer Learning": "迁移学习",
      "The learning-objective stage. Every row is the same Euclidean-aligned EEGNet; only the training or adaptation objective changes. The families differ in when the unlabelled target is used and whether the source data is still on hand. Unsupervised domain adaptation replaces plain ERM with a joint objective trained on the labelled source and the unlabelled target together. Source-free adaptation first trains an ERM source model, then optimizes a second objective on the target alone, with the source data gone. Test-time adaptation also starts from an ERM source model but adapts it online, one incoming target batch at a time. Source-only methods use no target at all. Each strategy keeps the shared EA-EEGNet training setup (Adam, batch 32, learning rate 1e-3) and adds only its own loss trade-offs and adaptation steps, read from its preset; all are two-class on the three datasets and measured against the same no-transfer baseline, ERM. Privacy-preserving transfer is the exception: it keeps each subject's raw EEG local and is measured against Centralized Training instead (see its note).":
        "学习目标阶段。每一行都是同一个经欧氏对齐的 EEGNet，只改动训练或自适应的目标。各族方法的区别在于何时用到无标签的目标域，以及那时是否还留着源域数据。无监督域自适应（unsupervised domain adaptation）把普通的 ERM 换成一个联合目标，在有标签的源域和无标签的目标域上一起训练。无源域自适应（source-free）先训练一个 ERM 源模型，再只在目标域上优化第二个目标，此时源域数据已经不在。测试时自适应（test-time）同样从 ERM 源模型出发，但一边接收目标试次一边在线更新，每次只用一小批。仅源域方法则完全不用目标域。每种策略都沿用同一套 EA-EEGNet 训练设置，即 Adam、批大小 32、学习率 1e-3，只在此之上加入各自的损失权衡和自适应步数，这些都来自它的预设（preset）。它们在三个数据集上都是二分类，都以同一个无迁移基线，也就是 ERM，作为基准。隐私保护迁移是个例外，它把每名被试的原始脑电留在本地，改以集中式训练为基准，详见该组说明。",
      "Ensemble Learning": "集成学习",
      "The aggregation stage, in a fully decentralized, privacy-preserving setting. Each source subject trains five different learners — tangent-space LDA, tangent-space SVM, EEGNet, ShallowConvNet, and CSP-Net — on its own data alone, and shares only its hard predicted labels on the target, never model weights or raw EEG. A combiner then fuses those (N−1)×5 label votes into one prediction, with no target labels to learn from. Since it sees only hard votes, the whole problem is estimating how far to trust each learner without any ground truth. Two non-ensemble references bracket the task, and the combiners are grouped beneath them.":
        "聚合阶段，在完全去中心化的隐私保护场景下进行。每名源被试只用自己的数据各训练五个不同的学习器，分别是切空间 LDA、切空间 SVM、EEGNet、ShallowConvNet 和 CSP-Net，并且只共享它在目标域上的硬预测标签，绝不共享模型权重或原始脑电。之后由一个组合器把这 (N−1)×5 个标签投票融合成一个预测，其间没有任何目标域标签可供学习。由于只看得到硬投票，整个问题就变成在没有真值的情况下，判断每个学习器该被信任到什么程度。两个非集成的参照给出问题的上下界，各个组合器分组列在它们下面。",
      "Decoding without any aggregation, to bracket the ensemble methods below. A single source learner applied to the target marks the floor; one model trained on all source subjects pooled together, Centralized Training, marks the non-private ceiling that the privacy-preserving combiners aim to match without ever sharing raw EEG.":
        "不做任何聚合时的解码效果，用来给下面的集成方法划定上下界。单个源学习器直接用到目标域，代表下界。把所有源被试的数据汇集起来训练的单一模型，也就是集中式训练，代表非隐私的上界。去中心化的隐私组合器要在不共享原始脑电的前提下追平这个上界。",
      "Every combiner sees the identical hard votes, so none has an information advantage; they differ only in how they estimate each learner's reliability with no labels. Plain majority voting trusts all learners equally and is the baseline. The spectral meta-learners weight each learner by the leading eigenvector of the vote agreement, an unsupervised accuracy estimate: SML is the binary form, and the lab's SML-OVR extends it to any number of classes, so the binary SML is pinned beneath SML-OVR because the two coincide on these two-class tasks. The crowd-labelling and truth-discovery aggregators (Dawid-Skene, EBCC, GLAD, and the rest) instead infer each learner's confusion or skill from how the votes agree. StackingNet, another lab method, learns per-learner weights directly on the unlabelled target. Each is measured against majority voting on the same dataset. All three datasets are two-class (chance 50%), so the columns compare directly.":
        "每个组合器看到的硬投票完全一样，谁都没有信息上的优势，区别只在于如何在没有标签的情况下估计每个学习器的可靠程度。普通多数投票对所有学习器一视同仁，是基线。谱元学习器按投票一致程度的主特征向量给每个学习器加权，这是一种无监督的准确率估计。SML 是二分类形式，实验室的 SML-OVR 把它推广到任意类别数。在这些二分类任务上两者结果一致，所以二分类的 SML 紧跟在 SML-OVR 后面。群体标注和真值发现类聚合方法（Dawid-Skene、EBCC、GLAD 等）则改从各条投票之间如何相互吻合，来推断每个学习器的混淆或能力。StackingNet 也是实验室方法，它直接在无标签的目标域上学习每个学习器的权重。每个组合器都以同一数据集上的多数投票为基准。三个数据集都是二分类，随机水平 50%，各列可以直接比较。",

      /* ============ benchmark transfer sub-category (subcat) headers & blurbs ============ */
      "Source-only": "仅源域",
      "Trained on the labelled source subjects only. The target is never used for adaptation, and inference is a plain forward pass. Baseline: ERM.":
        "只在有标签的源被试上训练，目标域始终不用于自适应，推断就是一次普通的前向传播。基线是经验风险最小化（Empirical Risk Minimization，ERM）。",
      "Unsupervised domain adaptation": "无监督域自适应",
      "Trained jointly on the labelled source and the unlabelled target, aligning the two distributions during source training. No target labels are used. Measured against the no-transfer baseline.":
        "在有标签的源域和无标签的目标域上联合训练，在源域训练过程中对齐两个分布，不使用任何目标域标签。以无迁移基线为基准来衡量。",
      "Source-free adaptation": "无源域自适应",
      "Adapts a source-trained model to the target while keeping no source data at transfer time. Measured against the no-transfer baseline.":
        "在迁移阶段不保留任何源域数据的前提下，把一个已在源域训练的模型自适应到目标域。以无迁移基线为基准来衡量。",
      "Test-time adaptation": "测试时自适应",
      "Adapts online as the target trials arrive at test time, updating the source-trained model without target labels. Measured against the no-transfer baseline.":
        "在测试时随着目标试次的到来进行在线自适应，在不使用目标域标签的情况下更新已在源域训练的模型。以无迁移基线为基准来衡量。",
      "Privacy-preserving transfer": "隐私保护迁移",
      "Non-ensemble references": "非集成参照",
      "Ensemble learning": "集成学习",
      "Cross-subject transfer that never pools raw EEG. Each subject's data stays on their own device, so these methods trade a little accuracy for privacy against Centralized Training, which pools everything. Two mechanisms appear. Federated methods (FedAvg, and the lab's FedBS and SAFE) run a central server that averages the per-subject model updates each round and sends the shared model back, so only weights, never EEG, are exchanged. FedBS additionally keeps each client's batch normalization local and seeks a flat minimum, and SAFE adds adversarial robustness on top. Decentralized MSDT uses no server at all: each source subject trains its own classifier, and only those trained models are shared and then fused on the target. All three datasets are two-class (chance 50%), so the columns are directly comparable. Δ is versus Centralized Training on the same dataset.":
        "跨被试迁移，但从不汇集原始脑电。每名被试的数据都留在自己的设备上，因此这些方法是以牺牲一点准确率来换取隐私，对照的是把所有数据汇到一起的集中式训练。这里有两种机制。联邦式方法（FedAvg，以及实验室的 FedBS 和 SAFE）由一个中心服务器在每一轮对各被试的模型更新做加权平均，再把共享模型发回，因此来回传递的只有权重，绝不是脑电。FedBS 还让每个客户端的批归一化（batch normalization）留在本地，并去寻找一个平坦的极小值。SAFE 在此之上再加入对抗鲁棒性。去中心化的 MSDT 则完全不用服务器，每名源被试训练自己的分类器，只把训练好的模型共享出去，再在目标域上融合。三个数据集都是二分类，随机水平 50%，因此各列可以直接比较。Δ 表示相对同数据集上集中式训练的差值。",

      /* ============ lab.js structural prose ============ */
      /* lab.tagline */
      "Transfer learning, robustness, privacy, and fuzzy systems for EEG-based BCIs.":
        "面向脑电（EEG）脑机接口（BCI）的迁移学习、鲁棒性、隐私保护与模糊系统研究。",
      /* lab.repo_intro */
      "The lab's open-source home. It holds two things: a unified, reproducible EEG-decoding benchmark, and a paper-to-code gallery that links every lab publication to its released code.":
        "实验室的开源主页。这里放着两样东西。一个是统一、可复现的脑电解码基准，另一个是把实验室每一篇论文都链接到其开源代码的论文到代码总览。",
      /* anchor.blurb */
      "This repository. A single, self-contained framework that reimplements 56 EEG-decoding approaches, covering data alignment, data augmentation, network backbones, transfer learning, and ensemble aggregation, on one composable pipeline. It puts them head-to-head under one controlled protocol on three MOABB motor-imagery EEG datasets. Every number on the leaderboard is a measured reproduction, recorded per method.":
        "即本仓库。这是一个自包含的统一框架，在同一条可组合流水线上重新实现了 56 种脑电解码方法，涵盖数据对齐、数据增强、网络骨干、迁移学习和集成聚合。它在三个 MOABB 运动想象脑电数据集上，依据单一受控协议把这些方法正面比较。排行榜上的每个数字都是实测复现，逐方法记录在案。",

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
