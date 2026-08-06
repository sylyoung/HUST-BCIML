/* HUST BCIML — internationalization (i18n) dictionary.
   Loaded as a plain <script> BEFORE app.js. Defines window.I18N: a flat map
   from the EXACT English source string to its Simplified-Chinese translation.
   app.js looks a string up here via tr(s) when LANG === 'zh'; any string absent
   from this map falls back to the original English, by design.

   Style: the formal, compressed Chinese of a technical release note, matching the
   lab's own 公众号 writing. Function first, then mechanism, then what it is for;
   noun-phrase compounds over spoken paraphrase (变更/归因/参照/配置, not 改动一下/
   看成/拿来比). An English term is glossed once as 中文（English, ABBR） and then
   used in its short form, never re-expanded. No em-dashes (——), quotation marks
   around terms, or semicolons (；): sentences are split with periods or joined
   with commas instead. gallery/check_i18n.py enforces the punctuation rules.

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
      "Disclaimer: both the external baselines and the lab's own approaches are independently reimplemented in this benchmark. The reported results, for the baseline reproductions and for the lab approaches alike, may differ from the original papers, and may contain errors. Corrections are welcome, and can be sent to the maintainer.":
        "免责声明。库内的外部基线方法和实验室自研方法，均由本基准独立重新实现。所报告的结果，无论是基线复现还是实验室方法，都可能与原论文存在偏差，也可能存在错误。欢迎指正，可联系维护者。",

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
      "Approaches in the benchmark": "基准中的全部方法",
      "Every approach evaluated in the benchmark, grouped by the stage of the decoding pipeline that it varies. The ensemble combiners form a group of their own, as they fuse the predictions of the models that the other groups train. The lab's own approaches, i.e., those proposed by Prof. Wu's group, are highlighted, and the external baselines they are compared with are listed alongside.":
        "基准测试所评测的全部方法，按各自所改变的解码流水线模块分组。集成聚合方法自成一组，因为它们融合的是其他各组所训练模型的预测结果。实验室自研方法，即伍冬睿教授课题组提出的方法，以高亮显示，与之对比的外部基线一并列出。",
      "lab-proposed": "实验室提出",
      "external baseline": "外部基线",
      "Anchor project": "核心项目",
      "View the benchmark": "查看基准测试",
      "Benchmark code": "基准测试代码",
      "stars": "颗星",
      "citations": "次引用",
      "Featured code repositories": "精选代码仓库",
      "Browse the lab's work by area": "按研究方向浏览实验室成果",
      "Publications grouped by research area, with the number of them that have released code. Use Papers & Code to search and filter. The official sites above hold the complete publication list.":
        "论文按研究方向分组，并标注其中已开源代码的数量。检索和筛选请使用论文与代码标签页，完整论文列表见上方官方网站。",

      /* ============ papers & code gallery ============ */
      "Papers & code gallery": "论文与代码总览",
      "The lab's publications, each linked to its released code where available. Showing the ":
        "实验室论文列表，已开源代码的均给出链接。当前显示 ",
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
      "in press": "即将发表",
      "more": "展开",
      "less": "收起",

      /* ============ benchmark: library intro, datasets, guide ============ */
      "The benchmark": "基准测试",
      "A unified and reproducible EEG decoding benchmark":
        "统一、可复现的脑电（EEG）解码基准",
      "Every approach is composed of the same modular stages, i.e., an aligner, an augmenter and a backbone, trained under a single learning objective and optionally aggregated by an ensemble. A controlled comparison varies one stage and fixes the rest, so that any change in the accuracy is attributable to that stage alone.":
        "库内全部方法均由同一组可组合模块构成，依次为对齐器、数据增强器和骨干网络，在单一学习目标下训练，需要时叠加一层集成。受控对比仅变更其中一个模块，其余配置保持不变，准确率的任何变化均可归因至该模块。",
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
      "RESULTS.md": "RESULTS.md",
      "Datasets": "数据集",
      /* The datasets intro is one whole-sentence template further down, so that
         Chinese controls its own word order. The two half-sentence keys it
         replaced are gone: nothing called them, and a dead entry sitting beside
         the live one invites an edit to the copy that never renders. */
      "Dataset": "数据集",
      "Subjects": "被试数",
      "Channels": "通道数",
      "Rate": "采样率",
      "Classes": "类别数",
      "Chance": "随机水平",
      "Trials/subj": "试次/被试",
      /* dataset `trials` values that carry a word rather than a bare count */
      "288 / session": "288 / 会话",
      /* per-dataset `role` descriptions (meta.datasets[i].role) */
      "Left hand versus right hand (two-class, chance 50%) in every table, including the privacy-preserving and the ensemble families. The original dataset is four-class (left hand, right hand, both feet, and tongue). The benchmark uses its two-class left/right subset throughout, and the four-class variant remains available in the code.":
        "全部表格中的任务均为左手对右手的二分类，随机水平 50%，隐私保护方法族和集成方法族同样如此。原始数据集为四分类，包括左手、右手、双脚和舌头。本基准全程使用其左右手二分类子集，四分类版本仍保留在代码中。",
      "Right hand versus both feet, 14 subjects, 100 training-run trials per subject. Two-class (chance 50%) throughout.":
        "右手对双脚，14 名被试，每名被试 100 个训练轮次试次。全程为二分类，随机水平 50%。",
      "Right hand versus both feet, 12 subjects, 200 first-session trials per subject. Two-class (chance 50%) throughout.":
        "右手对双脚，12 名被试，每名被试 200 个首次会话试次。全程为二分类，随机水平 50%。",
      "Controlled-comparison leaderboard": "受控对比排行榜",
      "How to read this leaderboard": "如何阅读本排行榜",
      "Each row should be read against the baseline of its own table. A table varies one stage of the pipeline and holds the rest at the default configuration, i.e., Euclidean-aligned trials, an EEGNet backbone, and supervised training. Most rows hence differ from the baseline in exactly one respect. A row that differs in more than one respect states so beneath its name, so that a Δ is not read as the effect of a single stage when it is not. The three columns are the three datasets. Under each accuracy, mean ± std is the mean over three random seeds and the standard deviation across those seeds. It quantifies the reproducibility, and not the spread across subjects, which is roughly ten times larger. Δ is the gain or the loss with respect to the baseline of the same dataset. Every table is two-class (chance 50%) on all three datasets, so the columns remain comparable throughout. Each family has its own baseline: the transfer families use ERM, the privacy-preserving family uses Centralized Training, the ensemble table uses majority voting, and the network-free classical pipelines are compared with EA-EEGNet. Two caveats apply. First, the baseline is the best checkpoint on a held-out source split, whereas the domain adaptation rows are the last iterate of a fixed schedule, as in their reference implementations. Second, every EA row estimates the alignment reference of the held-out subject from the unlabeled trials of that subject, which uses no label, but is transductive rather than zero-shot. Each row links to its code, and to its paper where a DOI is recorded.":
        "每一行均应对照其所在表的基线阅读。一张表只变更流水线中的一个模块，其余保持默认配置，即经欧氏对齐（Euclidean Alignment, EA）的试次、EEGNet 骨干网络和监督训练。因此绝大多数行与基线仅有一处差异。若某一行的差异不止一处，会在方法名下方注明，以免将 Δ 误读为单一模块带来的效果。三列对应三个数据集。每个准确率下方的均值 ± 标准差，为三个随机种子的均值和这三个种子之间的标准差，衡量的是可复现性，而非被试之间的差异，后者约为其十倍。Δ 为相对同一数据集基线的增减。三个数据集上每张表均为二分类，随机水平 50%，各列自始至终可比。各方法族均有自身的基线，迁移方法族以经验风险最小化（ERM）为基线，隐私保护方法族以集中式训练为基线，集成表以多数投票为基线，不含网络的经典流程与 EA-EEGNet 比较。另有两点需要说明。其一，基线取自留出源域划分上的最优检查点，而域自适应各行取自固定训练计划的最后一次迭代，与其参考实现的训练方式一致。其二，每一个使用欧氏对齐的行，均以留出被试自身的无标签试次估计对齐参考，该步骤不使用任何标签，但属于直推式设置而非零样本设置。各行均链接至其代码，记录了 DOI 的还链接至论文。",

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
      "The aligner stage. An aligner maps the trials of each subject into a shared statistical space prior to the backbone, reducing the between-subject covariance shift that otherwise dominates cross-subject decoding. Alignment requires no label, and is performed separately for each subject. The backbone and its training configuration are identical in every row, and the baseline performs no alignment.":
        "对齐器模块。对齐器在骨干网络之前，将每名被试的试次映射至共享统计空间，压低被试间协方差偏移，该偏移是跨被试解码性能受限的主要来源。对齐无需标签，按被试逐一完成。各行的骨干网络和训练配置完全一致，基线不做任何对齐。",
      "Data Augmentation": "数据增强",
      "The augmenter stage. An augmenter synthesizes additional training trials to regularize an otherwise identical backbone, and is measured against the same backbone trained without augmentation. The augmenters operate in two different spaces. The electrode-space transforms, i.e., Channel Reflection and Half-Sample Recombination, rearrange the channels, so they are applied to unaligned trials, before any spatial whitening, and are compared with the unaligned baseline. The signal-domain and frequency-domain augmenters are applied to Euclidean-aligned trials, and are compared with the aligned baseline.":
        "数据增强器模块。数据增强器合成额外的训练试次，对其余配置完全相同的骨干网络起正则化作用，并以同一骨干网络在无增强条件下的结果为参照。各增强器按作用空间分为两类。电极空间变换包括通道反射（Channel Reflection）和半样本重组（Half-Sample Recombination），需重排通道，因此作用于未对齐试次，且必须在空间白化之前完成，以未对齐基线为参照。信号域和频率域增强器作用于经欧氏对齐的试次，以对齐后的基线为参照。",
      "Networks": "网络骨干",
      "All 18 rows are measured from scratch under literal target-isolated nested LOSO with five final seeds, on the same two-class 8–32 Hz input with target EA from unlabeled target trials, shared Linear head and ERM objective. The five corrected rows — DeepConvNet, ShallowConvNet, ADFCNN, EEGWaveNet and FBMSNet — are explicit architecture transfers of the cited references; the complete campaign passed checkpoint, prediction, provenance and seed-coverage validation before import. MVCNet is a lab-proposed network whose row retains its documented three-seed legacy measurements because it changes the IFNet backbone, the multi-view contrastive objective and the batch size together.":
        "全部 18 行均从头重测，采用逐目标被试的严格嵌套留一被试交叉验证和 5 个最终种子，输入为同一套二分类 8–32 Hz 试次，目标域欧氏对齐只用目标被试的无标签试次，分类头与 ERM 目标统一。五个订正行 DeepConvNet、ShallowConvNet、ADFCNN、EEGWaveNet、FBMSNet 均为对所引文献的显式架构移植，完整实验通过模型检查点、预测、来源和种子覆盖校验后才导入。MVCNet 是本实验室提出的网络，它同时改变 IFNet 骨干、多视图对比目标和批大小，因此该行保留记录在案的三种子历史数值。",
      "Classical Pipelines": "经典流程",
      "No backbone. These rows replace the deep network with a classical decoding pipeline, fitted on the same Euclidean-aligned trials without any gradient-based training. There is hence neither early stopping nor random initialization, and each row is deterministic, with an across-seed standard deviation of exactly zero. They vary more than one stage simultaneously, and hence are not a controlled comparison of a single stage. They are reported as a reference for the deep rows, and are compared with EA-EEGNet on each dataset.":
        "不使用骨干网络。这些行以经典解码流程替代深度网络，输入为同一批经欧氏对齐的试次，拟合过程不含梯度训练，因此既无提前停止也无随机初始化，每一行的结果均为确定值，跨种子标准差恒为零。它们同时变更了多个模块，不构成针对单一模块的受控对比，列于此处是作为深度方法值得对照的参照，各数据集上均与 EA-EEGNet 比较。",
      "Transfer Learning": "迁移学习",
      "The learning-objective stage. Every row uses the same Euclidean-aligned EEGNet, and only the training or adaptation objective varies. The families differ in when the unlabeled target data are used, and whether the source data are still available. Unsupervised domain adaptation replaces ERM with a joint objective, trained on the labeled source and the unlabeled target together. Source-free adaptation first trains an ERM source model, and then optimizes a second objective on the target alone, without access to the source data. Test-time adaptation also starts from an ERM source model, but updates it online, one incoming target batch at a time. Source-only approaches do not use the target at all. Each strategy retains the shared EA-EEGNet training configuration (Adam, batch size 32, learning rate 1e-3), and adds only its own loss trade-offs and adaptation steps, which are read from its preset. All are two-class on the three datasets, and measured against the same no-transfer baseline, ERM. Privacy-preserving transfer is the exception. It keeps the raw EEG of each subject local, and hence is measured against Centralized Training instead, as its own note describes.":
        "学习目标模块。每一行均为同一个经欧氏对齐的 EEGNet，仅变更训练或自适应目标。各方法族的区别在于何时使用无标签的目标域数据，以及此时源域数据是否仍然可用。无监督域自适应以联合目标替代 ERM，在有标签源域和无标签目标域上共同训练。无源域自适应先训练一个 ERM 源模型，再在无法访问源域数据的条件下，仅在目标域上优化第二个目标。测试时自适应同样以 ERM 源模型为起点，但按批在线更新，每次仅使用一批到达的目标数据。仅源域方法完全不使用目标域。各策略沿用共享的 EA-EEGNet 训练配置，即 Adam、批大小 32、学习率 1e-3，仅在此基础上加入自身的损失权衡和自适应步数，取值来自各自的预设文件。全部方法在三个数据集上均为二分类，并以同一个无迁移基线 ERM 为参照。隐私保护迁移是例外，它将每名被试的原始脑电保留在本地，因此改以集中式训练为参照，详见该组说明。",
      "Ensemble Learning": "集成学习",
      "The aggregation stage, in a fully decentralized and privacy-preserving setting. Each source subject trains three different learners, i.e., tangent-space logistic regression, CSP-Net and EEGConformer, on its own data alone, and shares only its hard predicted labels on the target, never the model weights or the raw EEG. A combiner then fuses the resulting (N−1)×3 label votes into a single prediction, without any target label. One learner is taken from each of three model families: a Riemannian linear model, a convolutional network and a self-attention network. The displayed values are legacy measurements whose artifacts did not serialize combiner parameters or backend versions. The audited generating code used the simplified TestEnsemble ZenCrowd EM implementation for 20 passes and PM/CRH for three rounds; these settings are part of method identity. New runners record them and fail if any requested seed or combiner is incomplete.":
        "聚合模块，运行于完全去中心化的隐私保护场景。每名源被试只用自身数据训练三个不同的学习器，即切空间逻辑回归、CSP-Net 和 EEGConformer，并且只共享其在目标域上的硬预测标签，不共享模型权重和原始脑电。组合器随后将 (N−1)×3 个标签投票融合为单一预测，全程不使用目标域标签。三个学习器分别取自黎曼线性模型、卷积网络和自注意力网络。表中为历史测量，其结果文件没有记录组合器参数和后端版本。经审计，生成这些结果的代码使用 TestEnsemble 中简化的 ZenCrowd EM 实现，运行 20 轮，PM/CRH 运行 3 轮。这些设置属于方法身份。新运行脚本会记录这些参数，任何种子或组合器没有完整完成时都会终止。",
      "Decoding without any aggregation, to bound the ensemble approaches below. A single source learner applied to the target gives the lower reference. One model trained on all source subjects pooled together, i.e., Centralized Training, gives the non-private upper reference, which the privacy-preserving combiners approach without ever sharing the raw EEG.":
        "不做任何聚合的解码结果，用于界定下方集成方法的范围。单个源学习器直接作用于目标域，给出下参照。将全部源被试数据汇集后训练的单一模型，即集中式训练，给出非隐私的上参照，隐私保护组合器在不共享原始脑电的前提下逼近该参照。",
      "All combiners observe identical hard votes, and hence none of them has an information advantage. They differ only in how the reliability of each learner is estimated without labels. Majority voting weights all learners equally, and is the baseline. The spectral meta-learners weight each learner by the leading eigenvector of the vote agreement, which is an unsupervised estimate of the accuracy. SML is the binary form, and the lab's SML-OVR extends it to an arbitrary number of classes, so the binary SML is listed immediately below SML-OVR, as the two coincide on these two-class tasks. The crowd-labeling and truth-discovery aggregators (Dawid-Skene, EBCC, GLAD, and others) instead infer the confusion matrix or the skill of each learner from the agreement among the votes. StackingNet, another lab approach, learns the per-learner weights directly on the unlabeled target. Each combiner is measured against majority voting on the same dataset. All three datasets are two-class (chance 50%), so the columns are directly comparable.":
        "全部组合器观测到的硬投票完全相同，因而不存在信息上的优势，区别仅在于如何在无标签条件下估计各学习器的可靠程度。多数投票对所有学习器等权，作为基线。谱元学习器以投票一致性的主特征向量为各学习器加权，该权重是准确率的一种无监督估计。SML 为二分类形式，实验室的 SML-OVR 将其推广至任意类别数，两者在这些二分类任务上结果一致，因此二分类的 SML 紧列于 SML-OVR 之下。群体标注和真值发现类聚合方法，包括 Dawid-Skene、EBCC、GLAD 等，则从各投票之间的一致程度推断各学习器的混淆矩阵或能力水平。StackingNet 同为实验室方法，直接在无标签目标域上学习各学习器的权重。各组合器均以同一数据集上的多数投票为参照。三个数据集均为二分类，随机水平 50%，各列可直接比较。",

      /* ============ benchmark transfer sub-category (subcat) headers & blurbs ============ */
      "Source-only": "仅源域",
      "Trained on the labeled source subjects only. The target is never used for adaptation, and the inference is a single forward pass. The baseline is ERM.":
        "仅在有标签的源被试上训练，目标域始终不参与自适应，推断为一次前向传播。基线为 ERM。",
      "Unsupervised domain adaptation": "无监督域自适应",
      "Trained jointly on the labeled source and the unlabeled target, aligning the two distributions during the source training. No target label is used. Measured against the no-transfer baseline.":
        "在有标签源域和无标签目标域上联合训练，于源域训练过程中对齐两个分布，不使用任何目标域标签。以无迁移基线为参照。",
      "Source-free adaptation": "无源域自适应",
      "Adapts a source-trained model to the target, retaining no source data at the transfer time. Measured against the no-transfer baseline.":
        "在迁移阶段不保留任何源域数据的条件下，将已在源域训练的模型自适应至目标域。以无迁移基线为参照。",
      "Test-time adaptation": "测试时自适应",
      "Adapts online as the target trials arrive at the test time, updating the source-trained model without any target label. Measured against the no-transfer baseline.":
        "在测试阶段随目标试次到达进行在线自适应，在不使用任何目标域标签的条件下更新已在源域训练的模型。以无迁移基线为参照。",
      "Privacy-preserving transfer": "隐私保护迁移",
      "Non-ensemble references": "非集成参照",
      "Ensemble learning": "集成学习",
      "Cross-subject transfer that never pools the raw EEG. The data of each subject remain on their own device, so these approaches trade a small amount of accuracy for privacy, relative to Centralized Training, which pools all data. Two mechanisms are included. The federated approaches (FedAvg, and the lab's FedBS and SAFE) use a central server, which averages the per-subject model updates in each round and returns the shared model, so that only the model weights, and never the EEG, are transmitted. FedBS additionally keeps the batch normalization of each client local and seeks a flat minimum, and SAFE further adds adversarial robustness. Decentralized MSDT uses no server. Each source subject trains its own classifier, and only the trained models are shared and then fused on the target. All three datasets are two-class (chance 50%), so the columns are directly comparable. Δ is computed against Centralized Training on the same dataset.":
        "跨被试迁移，全程不汇集原始脑电。每名被试的数据均保留在其本地设备上，因此相对于汇集全部数据的集中式训练，这类方法以少量准确率换取隐私。这里包含两种机制。联邦式方法包括 FedAvg 以及实验室的 FedBS 和 SAFE，由中心服务器在每一轮对各被试的模型更新取平均并回传共享模型，传输的只有模型权重，不含脑电。FedBS 另将各客户端的批归一化保留在本地，并搜索平坦极小点，SAFE 在此基础上进一步引入对抗鲁棒性。去中心化的 MSDT 不使用服务器，每名源被试训练各自的分类器，仅共享训练完成的模型，再在目标域上融合。三个数据集均为二分类，随机水平 50%，各列可直接比较。Δ 为相对同一数据集上集中式训练的差值。",

      /* ============ lab.js structural prose ============ */
      /* lab.tagline */
      "Transfer learning, robustness, privacy, and fuzzy systems for EEG-based BCIs.":
        "面向脑电（EEG）脑机接口（BCI）的迁移学习、鲁棒性、隐私保护与模糊系统研究。",
      /* lab.repo_intro */
      "The lab's open-source home. It contains two components: a unified and reproducible EEG decoding benchmark, and a paper-to-code gallery that links each lab publication to its released code.":
        "实验室的开源主页，包含两部分内容，一是统一、可复现的脑电（EEG）解码基准，二是将实验室每篇论文链接至其开源代码的论文到代码总览。",
      /* anchor.blurb */
      "This repository. A self-contained framework that reimplements 59 pipeline approaches and 14 ensemble combiners on a single composable pipeline, covering data alignment, data augmentation, network backbones, transfer learning and ensemble aggregation. All of them are compared under one controlled protocol on three MOABB motor imagery EEG datasets. Every result on the leaderboard is a measured reproduction, recorded for each approach.":
        "即本仓库。一个自包含的统一框架，在同一条可组合流水线上重新实现了 59 种流水线方法和 14 种集成聚合方法，涵盖数据对齐、数据增强、网络骨干、迁移学习和集成聚合。全部方法在三个 MOABB 运动想象 EEG 数据集上依据同一受控协议对比，排行榜上的每个结果均为实测复现，并逐方法记录在案。",

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
      "iBCI": "植入式脑机接口（iBCI）",

      /* added with the 2026-07 review pass: whole-sentence templates (so
         Chinese controls its own word order), the split Overview counts, and
         the per-row leaderboard caveats. */
      "showing {n} of {total}": "显示 {n} 篇，共 {total} 篇",
      "The benchmark covers {n} MOABB motor imagery EEG datasets, all evaluated cross-subject under leave-one-subject-out. The accuracies are comparable only within the same dataset and the same number of classes.":
        "本基准覆盖 {n} 个 MOABB 运动想象 EEG 数据集，全部按跨被试的留一被试方式评测。准确率仅在同一数据集且类别数相同的条件下可比。",
      "pipeline approaches benchmarked": "已评测的流水线方法",
      "Also varies: ": "此外还改变了：",
      "Not applicable on some datasets: ": "在部分数据集上不适用：",
      "Filter by this topic": "按此方向筛选",
      "Filter by this paradigm": "按此范式筛选",
      "Approaches in the benchmark ({n})": "基准中的方法（共 {n} 种）",
      "HUST BCIML: EEG-Decoding Benchmark & Paper-to-Code Gallery": "华中科技大学 BCIML：EEG 解码基准与论文-代码索引",
      "Brain-Computer Interface & Machine Learning Laboratory (Prof. Dongrui Wu, HUST): the lab's open-source code home, holding a unified EEG-decoding benchmark and a paper-to-code gallery.": "华中科技大学脑机接口与机器学习实验室（伍冬睿教授）的开源代码主页，包含统一的 EEG 解码基准与论文-代码索引。"
    }
  };

  /* Convenience helpers (optional; app.js keeps its own tr()/LANG for robustness). */
  window.LANG = (function () {
    try { return localStorage.getItem("lang") || "en"; } catch (e) { return "en"; }
})();
})();
