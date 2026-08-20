<div align="center">

# HUST-BCIML

[English](README.md) | **简体中文**

**脑机接口与机器学习实验室的开源代码主页**

伍冬睿教授 &nbsp;·&nbsp; 华中科技大学

一个统一、可复现的**脑电（EEG）解码基准** &nbsp;+&nbsp; 一个可检索的**论文到代码总览**。

### &nbsp;[🌐&nbsp; 打开在线网页应用 &nbsp;↗](https://sylyoung.github.io/HUST-BCIML/)&nbsp;

[![Open the live web app](https://img.shields.io/badge/sylyoung.github.io%2FHUST--BCIML-Open_the_live_web_app-2563EB?style=for-the-badge&labelColor=1e293b)](https://sylyoung.github.io/HUST-BCIML/)

<sub>可检索的论文到代码总览&nbsp; ·&nbsp; 交互式基准排行榜&nbsp; ·&nbsp; 在浏览器中运行，无需安装</sub>

![Python](https://img.shields.io/badge/python-3.10%2B-3776ab)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c)
![Approaches](https://img.shields.io/badge/approaches-59-4338ca)
![Datasets](https://img.shields.io/badge/datasets-3%20MOABB%20MI-059669)
![License](https://img.shields.io/badge/license-MIT-blue)

[**实验室官方网站**](https://lab.bciml.cn/) &nbsp;·&nbsp; [**伍冬睿教授**](https://sites.google.com/site/drwuhust/) &nbsp;·&nbsp; [**Google 学术**](https://scholar.google.com/citations?user=UYGzCPEAAAAJ)

</div>

---

> **范围说明。**
> 上方链接的实验室官方网站与伍冬睿教授的个人主页，是了解本实验室概况、成员、动态及完整论文列表的权威来源。
>
> **本仓库是实验室的开源*代码*主页**，包含一个统一的脑电解码方法基准，以及一份从论文到其公开代码的映射。它与实验室各官方页面互为补充，而不是取而代之。

## 目录

- [概览](#概览)
- [研究动机](#研究动机)
- [设计原则](#设计原则)
- [基准测试方法](#基准测试方法)
- [方法清单](#方法清单)
- [快速开始](#快速开始)
- [论文到代码总览](#论文到代码总览)
- [仓库结构](#仓库结构)
- [复现与测量完整性](#复现与测量完整性)
- [扩展基准](#扩展基准)
- [精选仓库](#精选仓库)
- [路线图](#路线图)
- [引用](#引用)
- [联系方式](#联系方式)
- [致谢](#致谢)
- [许可证](#许可证)

<details>

<summary><b>更新日志</b></summary>

完整版本历史见 [`CHANGELOG.md`](CHANGELOG.md)。近期要点：

- **2026-08-19（v1.6.8）** 网页应用头部新增实验室标识，并用作浏览器页签图标。
- **2026-08-06（v1.6.7）** 仓库只发布一个依赖文件，测量环境锁不再随仓库发布。

- **2026-08-06（v1.6.6）** 署名修正（StackingNet、Channel Reflection），更新日志精简。

- **2026-08-06（v1.6.5）** 署名修正。

- **2026-08-06（v1.6.4）** 更新日志恢复默认折叠。

- **2026-08-06（v1.6.3）** 文件头布局精简。

- **2026-08-06（v1.6.2）** 文件头排版调整。

- **2026-08-06（v1.6.1）** 补全署名链文件头。

- **2026-08-06（v1.6.0）** 18 个网络骨干全部以嵌套 LOSO 与五个随机种子重测。

- **2026-07-31（未发布，审计订正）** 旧版网络数值标注为待重测的历史测量。

- **2026-07-30（v1.5.0）** 代码包移入 `src/hustbciml/`。

- **2026-07-30（v1.4.1）** 概览页恢复集成学习方法组。

- **2026-07-30（v1.4.0）** 集成学习表按每源三学习器重测。

- **2026-07-29（v1.3.2）** 删除 `RESULTS.md` 中的四分类附录。

- **2026-07-28（v1.3.1）** StackingNet 正则项改为 L1 形式。

- **2026-07-27（v1.3.0）** 恢复集成学习表，概览页移除集成部分。

- **2026-07-27（v1.2.4）** 重写并订正两份 README。

- **2026-07-27（v1.2.3）** 补齐遗漏的说明文字改写。

- **2026-07-27（v1.2.2）** 重写网页应用两种语言的说明文字。

- **2026-07-27（v1.2.1）** scikit-learn 上限定为 1.8 以下。

- **2026-07-27（v1.2.0）** 按外部审查修订，受影响的排行榜单元格全部重测。

- **2026-07-24（v1.1.3）** 按论文重写实验室方法的源码文档。

- **2026-07-24（v1.1.2）** 重组方法族，隐私保护迁移更名。

- **2026-07-24（v1.1.1）** 拆分集成学习表，增强器改用全称。

- **2026-07-24（v1.1.0）** 新增骨干、增强器与实验室方法，上线网页应用。

</details>

---

## 概览

本仓库包含两项内容。

**1. 脑电解码基准**，位于目录 [`src/hustbciml/`](src/hustbciml/)。

一个自包含的框架，围绕单一命令行入口和自动扫描的插件注册表构建。在同一条可组合流水线上重新实现了 **59 种脑电解码方法**，覆盖数据对齐、数据增强、网络骨干与迁移学习，另有 **14 种集成聚合方法**单独计数，它们聚合多个已训练模型，而不构成一条流水线。全部方法在**单一受控协议**下评估，每个报告数值都附有逐方法的复现记录。

**2. 论文到代码网页应用**，位于目录 [`docs/`](docs/)。

一个静态网页应用，并列呈现基准排行榜和覆盖实验室 **263 篇论文**的可检索**论文到代码总览**，其中 72 篇有公开代码。它可以作为本地文件直接打开，也可以由 GitHub Pages 托管，**无需构建步骤**。

## 研究动机

本实验室在脑电解码方向发表了大量成果，但相应的代码分散在众多相互独立的仓库中，各自的数据处理、评估划分与超参数约定都不一样。

因此，要复现任何单个结果，或者在同等条件下比较两种方法，都需要逐一手工重新推导每种方法的预处理、跨被试划分与训练计划。该过程容易出错，仅凭已发表的准确率数值也不能消除这一困难。

本仓库用两种互补的方式来解决这个问题。

- 它在同一条共享流水线上**重新实现**这些方法，并在单一受控协议下评估，使同一张排行榜表中的两行之间**只有一个**组件不同，明确标注了额外变动的行除外。

- 它把实验室的论文**映射**到其公开代码，让读者能够一步从一篇论文抵达可运行的实现。

## 设计原则

本基准围绕六条原则组织，每一条都由代码和报告方式来强制约束，而不是只靠惯例。

1. **可组合性。**
   一个*算法*是若干阶段插件的具名组合。多数情况下，添加一种方法就是添加一个符合某阶段接口的单一文件，注册表会按文件名发现它。

2. **受控比较。**
   基准原则上只改变**一个**流水线阶段，其余阶段保持在固定的规范配置上，因此只在一个组件上有差异的两行，可以把该组件的作用单独分离出来。有几行无法归结为单一维度：MVCNet 同时改变骨干网络、学习目标和批大小，PAT 在改变学习目标之外还改变数据增强器，MEKT、LSFT 与 MSDT 则是完全不含网络的黎曼方法。这些行在排行榜上带有明确的额外变动标注，而不作为单阶段改动呈现。ERM 基线取源域留出集上的最优检查点，域自适应各行则取参考实现所用固定训练计划的最后一次迭代。旧网络骨干表混用了固定与调参训练计划，旧脚本还汇总相互重叠的 LOSO 折验证分数，选出全数据集共用的学习率。相关数值已经撤下。订正后的网络表采用逐目标被试严格嵌套选择，完整五种子实验已通过校验，数值已经发布。

3. **测量完整性。**
   当前仍显示的排行榜数值都是在三个随机种子上实测得到的均值。网络骨干表的旧数值已经撤下，订正后的实验固定使用五个随机种子。没有任何数值是为了对上某篇论文而手工设定的。每一个已发布数值都记录在机器可读的复现文件里，协议匹配时对照论文自身的数值，协议不同时对照预期行为区间。

4. **诚实报告。**
   负面结果和低于基线的结果都予以保留并加以说明，而不是隐藏。排名**按数据集分别给出**，并如实报告实测值。本仓库刻意**不**提供横跨所有方法的单一扁平排名。

5. **可复现性。**
   新产生的测量会保存解析后的完整配置、源码与数据摘要、依赖版本、BLAS/LAPACK 数值后端、明确的预处理、方法参数，以及逐被试的预测与得分。完整测量身份不一致时不会复用结果。文件不全、无法读取、没有来源记录或配置不同，程序都会直接终止。订正后的网络骨干实验还会保存模型检查点和逐轮断点状态。其他公开数值早于这套文件格式，保留为历史测量。

6. **自包含与零构建。**
   网页应用从单一文件渲染，无需构建步骤。基准则在一个内置的合成数据集上端到端运行，无需下载。因此在获取任何真实数据之前，两者都可以先行审阅。

## 基准测试方法

### 流水线

一个算法是若干阶段插件的组合，在某一*学习策略*下训练。学习策略指学习目标，以及优化该目标的训练或自适应循环：

```
Aligner  →  Augmenter  →  Backbone  →  Head        (trained under a Strategy)
```

- **Aligner（对齐器）**，在学习之前施加的逐域信号归一化，例如对试次协方差做欧氏或黎曼对齐。
- **Augmenter（数据增强器）**，一种在训练时扩充训练集的变换。
- **Backbone（骨干网络）**，神经特征提取器，经典的无网络路径则用 `Identity`。
- **Head（分类头）**，位于骨干特征之上的分类器。
- **Strategy（学习策略）**，学习目标及其训练或自适应循环，例如经验风险最小化（ERM）、某种域自适应目标、无源域或测试时自适应过程等。

### 受控比较

多数阶段表只改变一个维度，其余阶段保持在规范配置上：

```
EA  ·  no augmentation  ·  EEGNet  ·  Linear head  ·  ERM
```

某一行报告的差值（Δ）就是它的准确率减去该表在同一数据集上的基线。必须同时改变多个阶段的行会在方法名下写明额外变动。订正后的网络骨干表采用独立的嵌套选择协议和单独的 EEGNet 基线键。另有一个独立的**集成**维度用于聚合多个模型，与各阶段表分开报告。

### 评估协议

所有结果都采用**跨被试的留一被试交叉验证（leave-one-subject-out, LOSO）**：模型在除一名被试之外的所有被试上训练，在留出的那名被试上评估，对每名被试重复进行。

当前显示的历史配置使用**三个随机种子**（1、2、3），订正后的网络骨干使用**五个随机种子**（1 至 5）。报告准确率先按每个种子计算逐被试宏平均，再计算跨种子均值。`±` 为跨种子的样本标准差，用于衡量可复现性，而不是被试之间的离散度。因此，确定性的无网络方法在构造上标准差为 `0.00`。

### 数据集

完整的基准在三个 MOABB 运动想象脑电数据集上运行。一个内置的合成 **Toy** 数据集可以在无需下载的情况下复现整条流水线，用作冒烟测试。

| 数据集 | 被试数 | 通道数 | 基准中使用的类别 | 随机水平 |
|---|--:|--:|---|--:|
| **BNCI2014001** | 9 | 22 | 全程为二分类，即左手对右手，隐私保护与集成部分也不例外。原生的四分类版本（双手、双脚、舌头）仍保留在代码中可供使用 | 50% |
| **BNCI2014002** | 14 | 15 | 二分类，即右手对双脚 | 50% |
| **BNCI2015001** | 12 | 13 | 二分类，即右手对双脚 | 50% |

在全部三个数据集上，每张表都是二分类（随机水平 50%），因此各列在全程都可以直接比较。每个方法族都以它在同一数据集上的基线来衡量，迁移各方法族以 ERM 为基线，隐私保护方法族以集中式训练为基线，集成表以多数投票为基线。

### 评估指标

准确率是运动想象任务的主要指标，并在全文中报告。此外，基准代码在范式需要时还会计算 Cohen's κ、macro-F1 与 ROC-AUC。逐被试预测都会保存下来，因此任何额外指标都可以在不重新运行模型的情况下重新计算。

## 方法清单

由实验室提出的方法标记为 **(lab)**。每个插件都归在它所改动的那一个流水线阶段之下，隐私保护与集成方法跨越多个阶段，按角色列出。

**信号对齐（对齐器）。**
欧氏对齐（**EA (lab)**，默认）、黎曼对齐（**RA**），以及 `Identity`（不做对齐）。对齐器在骨干网络看到数据之前，先把每名被试的试次重新对齐到一个共同的统计框架里，整个过程不需要标签。

**数据增强（增强器）。**
两种电极空间变换在对齐之前进行，即 **Channel Reflection (lab)**（把左右标签互换的矢状正中面镜像）和 **Half-Sample Recombination**。信号域和频率域的增强器则作用于经欧氏对齐的试次，包括 **CSDA (lab)**（一种小波跨被试细节互换）、**加性噪声**、**幅度翻转**、**幅度缩放**、**频率平移**、**傅里叶替代**和**频率重组**。`Identity` 不做任何增强。

**网络骨干。**
固定设置为二分类、MOABB 8–32 Hz 试次、用目标被试无标签试次计算目标域欧氏对齐、统一 Linear 分类头、交叉熵 ERM、逐目标被试严格嵌套留一被试交叉验证，以及 5 个最终随机种子。表中只更换特征网络。18 个订正网络为 **EEGNet**、**ShallowConvNet**、**DeepConvNet**、**EEG Conformer**、**CSP-Net (lab)**、**TIE-EEGNet (lab)**、**KDFNet (lab)**、**DBConformer (lab)**、**ADFCNN**、**CTNet**、**MSCFormer**、**MSVTNet**、**TMSA-Net**、**EEGWaveNet**、**SlimSeiz**、**FBMSNet**、**EEGNeX** 和 **EEG-Deformer**，另加 **MVCNet (lab)**，后者同时改变骨干网络、学习目标和批大小，保留其记录在案的三种子历史数值。这里比较的是统一基准协议下移植后的网络结构，不是复现各篇论文的数据集、划分、预处理、分类器和优化器。五个订正行的实现记录了关键改动：订正 ADFCNN 的注意力转置，采用 EEGWaveNet 公开代码的拓扑，采用 Braindecode 的 Deep4Net 与 ShallowFBCSPNet 特征结构，以及仅保留 8–32 Hz 六个因果分支的 FBMSNet。完整实验已通过模型文件、预测、来源和种子覆盖校验，网络表数值已恢复。

**迁移与自适应学习策略**（在固定的经欧氏对齐 EEGNet 上改变学习目标）。各族方法的区别在于何时用到无标签的目标域，以及那时是否还留着源域数据。

- **仅源域训练，配合目标域无标签对齐**：**ERM**（无迁移基线）、**MDMAML (lab)**、**ABAT (lab)**、**PAT (lab)**。训练阶段不使用目标域数据，也不使用任何目标标签。但这四种方法都组合了 `aligner: EA`，而欧氏对齐会先用留出被试自己的无标签试次估计该被试的白化参考矩阵，再做预测。这是标准的 EA 流程，不构成标签泄漏，但它属于直推式而非零样本，因此本文不把它描述为完全不使用目标域数据。
- **无监督域自适应**（把 ERM 换成一个源域加目标域的联合目标）：**MCC**、**CDAN**、**JAN**、**DAN**、**DANN**、**MDD**、**DJP-MMD (lab)**，以及无网络的 **MEKT (lab)**。
- **无源域自适应**（在源域 ERM 之后，只在目标域上再优化第二个目标，此时源域数据已不在）：**ASFA (lab)**、**SHOT**，以及无网络的 **LSFT (lab)**。
- **测试时自适应**（在线进行，每次只用一小批目标试次）：**T-TIME (lab)**、**DELTA**、**ISFDA**、**SAR**、**PL**（伪标签）、**BN-adapt**、**BFT (lab)**、**Tent**。

**经典（无网络）基线。**
**CSP-LDA** 与 **Riemann-MDM** 是无迁移基线，上面的经典迁移方法 **MEKT (lab)** 和 **LSFT (lab)** 作用于黎曼切空间特征。

**隐私保护迁移。**
从不汇集原始脑电的跨被试迁移，以**集中式训练**（会汇集数据）为对照。**联邦式**方法由一个服务器每一轮对各被试的模型更新做平均，包括 **FedAvg** 以及实验室的 **FedBS (lab)** 和 **SAFE (lab)**。去中心化的 **MSDT (lab)** 则只共享训练好的各被试模型，再在目标域上融合。

**集成聚合。**
一个去中心化的黑箱场景。每名源被试只用自己的数据训练三个学习器，三者分属不同的模型族，即切空间逻辑回归、CSP-Net 和 EEGConformer，并且只共享硬预测标签，再由一个组合器在没有目标域标签的情况下把这些投票融合起来。组合器包括多数**投票**（基线）、谱元学习器 **SML** 和实验室的 **SML-OVR (lab)**、实验室的 **StackingNet (lab)**，以及一批群体标注和真值发现类聚合方法（**Dawid-Skene**、**EBCC**、**GLAD**、TestEnsemble 中的简化 EM 基线 **ZenCrowd**、**MACE**、三轮 **PM/CRH**、**LAA**、**LA**、**M-MSR**、**Wawa**）。ZenCrowd 与 PM 的迭代次数属于方法身份，不再作为运行脚本中未记录的默认值。

## 快速开始

### 浏览网页应用（无需安装，无需服务器）

**在线站点：** **[sylyoung.github.io/HUST-BCIML](https://sylyoung.github.io/HUST-BCIML/)**，也可以在本地运行：

```bash
open docs/index.html          # macOS, or simply double-click the file
```

数据已经内联进页面，因此它可以直接从文件系统渲染，在由 GitHub Pages 提供服务时表现一致。该应用有三个标签页：

- **概览（Overview）**，介绍本仓库是什么、实验室官方链接，以及精选代码仓库。
- **基准测试（Benchmark）**，三数据集排行榜，附各方法族的说明。
- **论文与代码（Papers & Code）**，检索并筛选论文到代码总览。

### 运行基准

```bash
pip install -r requirements.txt   # 依赖
pip install -e .                  # 安装位于 src/ 下的本包

python -m hustbciml.run --list                                                # every plug-in
python -m hustbciml.run --algorithm EA-EEGNet --dataset Toy --device cpu       # synthetic, no download
python -m hustbciml.run --algorithm EA-EEGNet --dataset BNCI2014001 --itr 3    # real data, via MOABB
```

本仓库只发布一个依赖文件：`requirements.txt`。订正后的五种子网络骨干实验实际使用的 Python 3.11
与 CUDA 软件包版本，只在测量机器上锁定、不随仓库发布。其摘要与运行时身份记录在实验溯源数据中。
复现榜单数字需要重建该环境，做法见 `requirements.txt` 末尾的说明。

也可以即时组合一个算法，而不必指定某个预设：

```bash
python -m hustbciml.run --aligner EA --augmenter CSDA --backbone DBConformer \
                        --strategy ERM --head Linear --dataset BNCI2014001 --itr 3
```

每次运行在 `results/<setting>/` 下写入两个文件。`metrics.json` 记录逐被试准确率、均值与标准差，以及解析后的完整配置，因此仅凭这一个文件，就能把排行榜上的一个单元格追溯到产出它的确切设置。`predictions.npz` 记录逐被试的预测与得分。当前数值见 [`src/hustbciml/RESULTS.md`](src/hustbciml/RESULTS.md)，术语表、算法卡片与移植指南见 [`src/hustbciml/docs/`](src/hustbciml/docs/index.md)。

## 论文到代码总览

网页应用由人工整理的 YAML 经过单一脚本生成，不依赖任何框架。

- **唯一权威数据源**，位于 [`gallery/data/`](gallery/data/)，包括 `publications.yml`（263 篇论文）、`lab.yml`（简介、核心项目、精选仓库）与 `benchmark.yml`（受控比较排行榜）。

- **生成器**，即 [`gallery/build_site.py`](gallery/build_site.py)，把这些 YAML 文件编译为 `docs/data/*.js`，只需要 PyYAML。

在编辑 `gallery/data/` 下任何 YAML 之后，重新生成网页应用数据：

```bash
python3 gallery/build_site.py     # requires only PyYAML
```

## 仓库结构

```
HUST-BCIML/
├── .github/workflows/ci.yml    # 测试、生成文件校验与每周链接检查
├── .gitignore                  # 本地结果、缓存与构建文件的共享排除规则
├── src/hustbciml/              # THE BENCHMARK  (可导入的 Python 包)
│   ├── run.py                  # python -m hustbciml.run --algorithm EA-EEGNet --dataset BNCI2014001
│   ├── core/                   # batch, stages (ABCs), registry, pipeline, config, context
│   ├── exp/                    # exp_basic + one Exp class per protocol
│   ├── algorithms/             # aligners / augmenters / models / heads / strategies / presets
│   ├── data_provider/          # datasets, data_factory, splitters, collate
│   ├── utils/                  # metrics, seed, tools
│   ├── scripts/                # ensemble, leaderboard, compare, tuning, card generation
│   ├── tests/repro/            # repro_targets.yaml, measured vs. published, per method
│   ├── docs/                   # glossary, porting guide, per-algorithm cards
│   └── RESULTS.md              # the full leaderboard, in Markdown
├── docs/                       # THE WEB APP (GitHub Pages 按此目录名发布站点)
│   ├── index.html
│   ├── assets/                 # style.css, app.js  (vanilla JS, no framework)
│   └── data/                   # generated: lab.js, publications.js, benchmark.js
├── gallery/                    # source of truth for the web app's data
│   ├── data/
│   │   ├── publications.yml     # 263 papers (hand-curated)
│   │   ├── lab.yml              # lab bio, anchor project, featured repos
│   │   └── benchmark.yml        # controlled-comparison leaderboard
│   └── build_site.py           # YAML → docs/data/*.js   (requires only PyYAML)
├── pyproject.toml              # 打包与 pytest 配置
└── requirements.txt            # 仓库唯一的依赖文件
```

`src/` 是 Python 查找代码包的根目录，`hustbciml/` 才是实际导入包。把二者压成一层会破坏命令行入口和插件注册表所用的 `import hustbciml` 命名空间。`pip install -e .` 负责把这个查找根目录加入 Python 路径。

仓库根目录的三个基础设施文件也需要纳入版本控制。`.gitignore` 是共享排除规则，用来防止本地缓存、实验结果和构建产物进入提交。`pyproject.toml` 定义安装方式、依赖、包内数据和 pytest 配置。`.github/workflows/ci.yml` 让 GitHub Actions 在代码变更后运行非复现实验测试和生成文件校验，并每周检查外部链接。它不会提交或推送文件。

## 复现与测量完整性

基准中当前仍显示的数值都是三种子实测均值，没有任何数值是为了对上某篇论文而手工设定。网络骨干表的旧数值已经撤下，订正后的实验固定使用五个种子。

每一个已发布数值都记录在 [`src/hustbciml/tests/repro/repro_targets.yaml`](src/hustbciml/tests/repro/repro_targets.yaml) 中。协议匹配时对照论文自身的数值，协议不同时对照预期行为区间，并附有逐方法注记。`tests/repro/test_repro_targets.py` 在每次提交时检查排行榜、登记表和可运行预设是否一致。订正后的网络实验另由完整实验校验器检查模型文件、预测、五个种子、嵌套划分和来源信息，不能用单次预设运行代替。算法[卡片](src/hustbciml/docs/cards/README.md)说明方法机制和上游实现。上游仓库声明了许可条款的，卡片如实记录，未作声明的也照实写明。

#### 超参数选择：历史结果与订正后的流程

网络表以外的现有数值早于订正后的调参脚本。旧流程使用过两种信号。

* **全局源域验证选择**（`select="val"`，包括旧网络骨干调参）。旧说明写成留出源被试，但代码实际留出随机源域**试次**，再汇总相互重叠的 LOSO 折验证分数，选出全数据集共用的参数。某名被试作为一折的外层目标时，会在其他折中带标签进入源域验证，因此每名被试都间接参与了自身测试折的参数选择。候选运行还会计算并打印外层目标准确率。这不是嵌套、盲化的选择。
* **开发被试选择**（`select="dev"`，用于 ASFA、Tent、BFT、DJP-MMD、MDMAML、MSDT、LSFT 与 MVCNet）。三名被试以真实标签作为伪目标参与评分，随后仍计入最终平均值。训练损失虽不使用目标标签，这仍属于用目标标签选模型。

旧发布流程还有一条规则：新调参数值只有在测试结果优于旧值时才替换。该规则直接使用测试表现，也没有形成有效的模型选择隔离。相关非网络数值仍作为历史测量显示，网络表数值则已经撤下。

订正后的 `tune_networks.py` 对每名外层目标被试分别做嵌套选择。选参时留出完整的源被试，不对外层目标做对齐、预测或评分。确定学习率后，再用新模型对该目标评估一次。种子不全、结果身份不符或缓存缺少来源记录时，脚本会终止。完整五种子实验已于 2026-08-06 通过校验，网络表数值已经发布。

> **免责声明。**
> 本基准**独立地重新实现**了外部基线与实验室自研方法。
>
> 所报告的结果**都可能与原论文存在差异，也可能包含错误**，无论是基线复现值还是实验室方法数值。原因可能是协议不匹配、忠实但不完美的移植，或者某个超参数选择。
>
> 若您发现任何不一致之处，请提交 issue 或联系维护者。欢迎指正。

## 扩展基准

添加 `src/hustbciml/algorithms/<group>/<Name>.py`，在其中定义一个符合相应阶段抽象基类的类，它会**按文件名自动注册**。

随后用一个预设 YAML 把它组合进来，在有了真实数值之后添加一个复现目标，并撰写一张算法卡片。每个新文件都带有一个标准文件头，包含作者、日期、确切的 IEEE 引用，以及在有原作者代码时指向该代码的链接。

完整工作流见[移植指南](src/hustbciml/docs/porting_guide.md)。

## 精选仓库

实验室的代表性仓库被置顶展示在[概览标签页](docs/index.html)，起始于：

- [**DeepTransferEEG**](https://github.com/sylyoung/DeepTransferEEG)
- [**TestEnsemble**](https://github.com/sylyoung/TestEnsemble)
- [**DBConformer**](https://github.com/wzwvv/DBConformer)
- [**EEG-FM-Benchmark**](https://github.com/Dingkun0817/EEG-FM-Benchmark)
- [**EEGAdversarialBenchmark**](https://github.com/xqchen914/EEGAdversarialBenchmark)
- [**NT-Benchmark**](https://github.com/chamwen/NT-Benchmark)
- [**TLBCI**](https://github.com/drwuHUST/TLBCI)

## 路线图

以下方向计划在未来版本中实现。

- **评估协议**，在当前的跨被试 LOSO 之外，增加被试内与跨会话划分，以及一个在线流式协议。
- **范式广度**，在运动想象之外，增加 ERP/P300（以 ROC-AUC 为主要指标）与 SSVEP。
- **可引用发布**。带版本号的 1.6.0 版本已经发布，DOI 存档仍在规划中。

## 引用

若本基准或其中的论文到代码总览对您的工作有帮助，请引用相关的实验室论文，也请链接回本仓库。每个方法的源文件头部都写有其对应的 IEEE 引用。

一个经 DOI 存档的可引用发布正在计划中，版本 1.6.0 已经发布。

## 联系方式

基准与网页应用由 **李思扬（Siyang Li）** 构建并维护，[个人主页](https://sylyoung.github.io/) &nbsp;·&nbsp; **lsyyoungll@gmail.com**。

伍冬睿教授的邮箱地址可在实验室的任一篇论文中找到。

## 致谢

数据集通过 [MOABB](https://moabb.neurotechx.com/)（Mother of All BCI Benchmarks）提供。

移植的方法在各自的文件头以及对应的算法卡片中标注其原作者。集成与隐私保护部分所用的群体聚合基线，连同其参考文献，在 [`src/hustbciml/RESULTS.md`](src/hustbciml/RESULTS.md) 中致谢。

## 许可证

本项目以 **MIT 许可证** 发布，完整条款见 [`LICENSE`](LICENSE)。

本基准重新实现或改编了若干先前已发表的方法。每张[算法卡片](src/hustbciml/docs/cards/README.md)记录了对应方法的代码来源。从零重新实现的部分受本仓库的 MIT 许可证覆盖，改编自某个特定上游仓库的实现则保留该项目原有的许可证条款。数据集依各自提供方的使用条款获取。

---

<div align="center"><sub>HUST-BCIML · MIT License · Brain-Computer Interface and Machine Learning Laboratory, HUST</sub></div>
