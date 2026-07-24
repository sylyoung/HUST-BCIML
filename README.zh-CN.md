<div align="center">

# HUST-BCIML

[English](README.md) | **简体中文**

**脑机接口与机器学习实验室的开源代码主页**

伍冬睿教授 &nbsp;·&nbsp; 华中科技大学

<br>

一个统一、可复现的**脑电（EEG）解码基准** &nbsp;+&nbsp; 一个可检索的**论文到代码总览**。

<br>

### &nbsp;[🌐&nbsp; 打开在线网页应用 &nbsp;↗](https://sylyoung.github.io/HUST-BCIML/)&nbsp;

[![Open the live web app](https://img.shields.io/badge/sylyoung.github.io%2FHUST--BCIML-Open_the_live_web_app-2563EB?style=for-the-badge&labelColor=1e293b)](https://sylyoung.github.io/HUST-BCIML/)

<sub>可检索的论文到代码总览&nbsp; ·&nbsp; 交互式基准排行榜&nbsp; ·&nbsp; 在浏览器中运行，无需安装</sub>

<br>

![Python](https://img.shields.io/badge/python-3.10%2B-3776ab)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c)
![Approaches](https://img.shields.io/badge/approaches-56-4338ca)
![Datasets](https://img.shields.io/badge/datasets-3%20MOABB%20MI-059669)
![License](https://img.shields.io/badge/license-MIT-blue)

[**实验室官方网站**](https://lab.bciml.cn/) &nbsp;·&nbsp; [**伍冬睿教授**](https://sites.google.com/site/drwuhust/) &nbsp;·&nbsp; [**Google 学术**](https://scholar.google.com/citations?user=UYGzCPEAAAAJ)

</div>

---

> **范围说明。**
> 上方链接的实验室官方网站与伍冬睿教授的个人主页，是了解本实验室概况、成员、动态及完整论文列表的权威来源。
>
> **本仓库是实验室的开源*代码*主页**，包含一个统一的脑电解码方法基准，以及一份从论文到其公开代码的映射。它与实验室各官方页面互为补充，而不是取而代之。

<br>

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

<br>

<details>
<summary><b>更新日志</b></summary>

<br>

完整版本历史见 [`CHANGELOG.md`](CHANGELOG.md)。近期要点：

- **2026-07-24（v1.1.2）** 重写了方法清单，并重新组织了迁移与集成两族。迁移方法现在按何时用到无标签目标域来分组，依次是仅源域、无监督域自适应、无源域和测试时。隐私保护一族更名为隐私保护迁移，说明也做了扩充。集成部分则讲清了去中心化的黑箱协议。通道对称（Channel Symmetry）不再作为基准增强器，其原理改放到通道反射的源码注释里，方法数因此变为 56。MVCNet 也改按普通网络骨干呈现。新增了 `CHANGELOG.md`，中文页面也改得更地道。

- **2026-07-24（v1.1.1）** 集成学习表拆分为两个子族，即非集成参照与集成学习，与迁移学习的版式一致，原先表格上方的汇总条也已移除。数据增强器改用全称显示，包括加性噪声、幅度缩放、频率平移、傅里叶替代、频率重组、通道对称与半样本重组。基准与概览的文字在中英文两版都做了重写，以提升清晰度。论文索引也做了去重，每篇论文只保留正式发表的版本（275 篇减至 263 篇）。

- **2026-07** 网络骨干轴新增十种骨干网络：ADFCNN、CTNet、MSCFormer、MSVTNet、TMSA-Net、EEGWaveNet、SlimSeiz、FBMSNet、EEGNeX 与 EEG-Deformer。数据增强轴新增第八种基线，即幅度缩放（amplitude scaling）。以上方法均在三个数据集、三个随机种子上完成基准测试，测得的准确率已列入排行榜，各自附带可直接运行的预设（preset）。

- **2026-07** 来自实验室数据增强研究的七种数据增强基线加入了增强轴：加性噪声、幅度翻转、频率平移、傅里叶替代（Fourier surrogate）、频率重组、通道对称与半样本重组。每一种都附带可运行的预设。

- **2026-07** 代码中的每一条参考文献现在都给出了完整的期刊或会议名称。共空间模式（CSP）、欧氏对齐（EA）与 MVCNet 的引用得到订正与补全。

- **2026-07** 一个忠实、完整的 **MEKT** 实现（第 III-C 节的域自适应投影，移植自原作者代码）现已在三个数据集中的两个上，位列经典迁移（classical transfer）结果之首。

- **2026-07** 基准测试软件包整合为 **`hustbciml`**。隐私保护比较扩展到 **三个** MOABB 数据集。一次基于留出源域（held-out source）的超参数选择流程刷新了网络、迁移、增强与复合方法各表，只有当公平选出的配置优于此前配置时才替换数值。

- **2026-07** 又移植了四种实验室方法（**CSP-Net、DJP-MMD、LSFT、MSDT**），迁移表重新分组为仅源域、无监督域自适应、无源域、测试时几个族。

- **2026-07** 网页应用新增了三数据集排行榜、实验室方法高亮显示，以及一份覆盖 **263** 篇论文的可检索论文到代码总览。

</details>

---

## 概览

本仓库打包了两项交付成果，**代码优先**。

**1. 脑电解码基准**，位于目录 [`hustbciml/`](hustbciml/)。

这是一个自包含的框架，围绕单一命令行入口和一个自动扫描的插件注册表构建。在这条可组合的流水线上，它重新实现了 **56 种脑电解码方法**，涵盖数据对齐、数据增强、网络骨干、迁移学习与集成聚合，并在**单一受控评估协议**下对它们正面比较，同时为每一个报告数值配上逐方法的复现记录。

**2. 论文到代码网页应用**，位于目录 [`docs/`](docs/)。

这是一个静态网页应用，把基准排行榜和一份覆盖实验室 **263 篇论文**（其中 76 篇有公开代码）的可检索**论文到代码总览**并列呈现。它可以作为本地文件直接打开，也可以托管在 GitHub Pages 上，且**无需构建步骤**。

<br>

## 研究动机

本实验室在脑电解码方向发表了大量成果，但相应的代码分散在众多相互独立的仓库中，各自的数据处理、评估划分与超参数约定都不一样。

因此，要复现任何单个结果，或者在同等条件下比较两种方法，都得逐一手工重新推导每种方法的预处理、跨被试划分与训练计划。这个过程容易出错，而仅凭已发表的准确率数值并不能消除这一困难。

本仓库用两种互补的方式来解决这个问题。

- 它在同一条共享流水线上**重新实现**这些方法，并在单一受控协议下评估，使排行榜中任意两行之间**恰好只有一个**组件不同。

- 它把实验室的论文**映射**到其公开代码，让读者能够一步从一篇论文抵达可运行的实现。

<br>

## 设计原则

本基准围绕六条原则组织，每一条都由代码和报告方式来强制约束，而不是只靠惯例。

1. **可组合性。**
   一个*算法*是若干阶段插件的具名组合。多数情况下，添加一种方法就是添加一个符合某阶段接口的单一文件，注册表会按文件名发现它。

2. **受控比较。**
   每一次比较都只改变**一个**流水线阶段，其余阶段保持在固定的规范配置上。只在一个组件上有差异的两行，可以把该组件的作用单独分离出来。

3. **测量完整性。**
   每一个报告数值都是在三个随机种子上**实测**得到的均值。没有任何数值是为了对上某篇论文而手工设定的。每一个数值都记录在一个机器可读的复现文件里，协议匹配时对照论文自身的数值，协议不同时对照一个预期行为区间（expected-behaviour band）。

4. **诚实报告。**
   负面结果和低于基线的结果都予以保留并加以说明，而不是隐藏。排名**按数据集分别给出**，并如实报告实测值。本仓库刻意**不**提供横跨所有方法的单一扁平排名。

5. **可复现性。**
   每次运行都固定随机种子，并持久化解析后的配置、逐被试预测与检查点（checkpoint）。凡是用到超参数选择的地方，都**只在留出的源被试上**进行，绝不触及目标或测试标签。

6. **自包含与零构建。**
   网页应用从单一文件渲染，无需构建步骤。基准则在一个内置的合成数据集上端到端运行，无需下载。因此在获取任何真实数据之前，两者都可以先行审阅。

<br>

## 基准测试方法

### 流水线

一个算法是若干阶段插件的组合，在某一训练或自适应过程（也就是*学习策略（strategy）*，亦即学习目标）下评估：

```
Aligner  →  Augmenter  →  Backbone  →  Head        (trained under a Strategy)
```

- **Aligner（对齐器）**，在学习之前施加的逐域信号归一化，例如对试次协方差做欧氏或黎曼对齐。
- **Augmenter（数据增强器）**，一种在训练时扩充训练集的变换。
- **Backbone（骨干网络）**，神经特征提取器，经典的无网络路径则用 `Identity`。
- **Head（分类头）**，位于骨干特征之上的分类器。
- **Strategy（学习策略）**，学习目标及其训练或自适应循环，例如经验风险最小化（ERM）、某种域自适应目标、无源域或测试时自适应过程等。

### 受控比较

每张阶段表都**恰好只改变一个维度**，其余阶段保持在规范配置上：

```
EA  ·  no augmentation  ·  EEGNet  ·  Linear head  ·  ERM
```

因此，每一行与它所在表的基线之间只有一处不同，某一行报告的差值（Δ）就是它的准确率减去该表在同一数据集上的基线。另有一个独立的**集成（ensemble）**维度用于聚合多个模型，它与各单维度表分开报告。

### 评估协议

所有结果都采用**跨被试的留一被试交叉验证（leave-one-subject-out, LOSO）**：模型在除一名被试之外的所有被试上训练，在留出的那名被试上评估，对每名被试重复进行。

每种配置在**三个随机种子**（1、2、3）上运行。报告的准确率为**跨种子均值**，报告的 `±` 为**跨种子的标准差**，它衡量可复现性，而不是跨被试的离散度。因此，确定性的、无网络的方法在构造上标准差为 `0.00`。

### 数据集

完整的基准在三个 MOABB 运动想象脑电数据集上运行。一个内置的合成 **Toy** 数据集可以在无需下载的情况下复现整条流水线，用作冒烟测试（smoke test）。

| 数据集 | 被试数 | 通道数 | 基准中使用的类别 | 随机水平 |
|---|--:|--:|---|--:|
| **BNCI2014001** | 9 | 22 | 全程为二分类，即左手对右手，隐私保护与集成部分也不例外。原生的四分类版本（双手、双脚、舌头）仍保留在代码中可供使用 | 50% |
| **BNCI2014002** | 14 | 15 | 二分类，即右手对双脚 | 50% |
| **BNCI2015001** | 12 | 13 | 二分类，即右手对双脚 | 50% |

在全部三个数据集上，每张表都是二分类（随机水平 50%），因此各列在全程都可以直接比较。每个方法族都以它在同一数据集上的基线来衡量，迁移各方法族以 ERM 为基线，隐私保护方法族以集中式训练为基线，集成表以多数投票为基线。

### 评估指标

准确率是运动想象任务的主要指标，并在全文中报告。此外，基准代码在范式需要时还会计算 Cohen's κ、macro-F1 与 ROC-AUC。逐被试预测都会保存下来，因此任何额外指标都可以在不重新运行模型的情况下重新计算。

<br>

## 方法清单

由实验室提出的方法标记为 **(lab)**。每个插件都归在它所改动的那一个流水线阶段之下，隐私保护与集成方法跨越多个阶段，按角色列出。

**信号对齐（对齐器）。**
欧氏对齐（**EA (lab)**，默认）、黎曼对齐（**RA**），以及 `Identity`（不做对齐）。对齐器在骨干网络看到数据之前，先把每名被试的试次重新对齐到一个共同的统计框架里，整个过程不需要标签。

**数据增强（增强器）。**
两种电极空间变换在对齐之前进行，即 **Channel Reflection (lab)**（把左右标签互换的矢状正中面镜像）和 **Half-Sample Recombination**。信号域和频率域的增强器则作用于经欧氏对齐的试次，包括 **CSDA (lab)**（一种小波跨被试细节互换）、**加性噪声**、**幅度翻转**、**幅度缩放**、**频率平移**、**傅里叶替代**和**频率重组**。`Identity` 不做任何增强。

**网络骨干。**
在固定的经欧氏对齐、ERM 训练设置上，只更换网络。**EEGNet** 是规范基线，此外还有 **ShallowConvNet**、**DeepConvNet**、**EEG Conformer**、**CSP-Net (lab)**、**TIE-EEGNet (lab)**、**KDFNet (lab)**、**DBConformer (lab)**、**MVCNet (lab)**，以及一批较新的网络（**ADFCNN**、**CTNet**、**MSCFormer**、**MSVTNet**、**TMSA-Net**、**EEGWaveNet**、**SlimSeiz**、**FBMSNet**、**EEGNeX**、**EEG-Deformer**）。每个骨干网络都沿用其原论文的结构，只调学习率，而且只在留出的源被试上调。

**迁移与自适应学习策略**（在固定的经欧氏对齐 EEGNet 上改变学习目标）。各族方法的区别在于何时用到无标签的目标域，以及那时是否还留着源域数据。

- **仅源域**（完全不用目标域）：**ERM**（无迁移基线）、**MDMAML (lab)**、**ABAT (lab)**、**PAT (lab)**。
- **无监督域自适应**（把 ERM 换成一个源域加目标域的联合目标）：**MCC**、**CDAN**、**JAN**、**DAN**、**DANN**、**MDD**、**DJP-MMD (lab)**，以及无网络的 **MEKT (lab)**。
- **无源域自适应**（在源域 ERM 之后，只在目标域上再优化第二个目标，此时源域数据已不在）：**ASFA (lab)**、**SHOT**，以及无网络的 **LSFT (lab)**。
- **测试时自适应**（在线进行，每次只用一小批目标试次）：**T-TIME (lab)**、**DELTA**、**ISFDA**、**SAR**、**PL**（伪标签）、**BN-adapt**、**BFT (lab)**、**Tent**。

**经典（无网络）基线。**
**CSP-LDA** 与 **Riemann-MDM** 是无迁移基线，上面的经典迁移方法 **MEKT (lab)** 和 **LSFT (lab)** 作用于黎曼切空间特征。

**隐私保护迁移。**
从不汇集原始脑电的跨被试迁移，以**集中式训练**（会汇集数据）为对照。**联邦式**方法由一个服务器每一轮对各被试的模型更新做平均，包括 **FedAvg** 以及实验室的 **FedBS (lab)** 和 **SAFE (lab)**。去中心化的 **MSDT (lab)** 则只共享训练好的各被试模型，再在目标域上融合。

**集成聚合。**
一个去中心化的黑箱场景。每名源被试只用自己的数据训练五个学习器，并且只共享硬预测标签，再由一个组合器在没有目标域标签的情况下把这些投票融合起来。组合器包括多数**投票**（基线）、谱元学习器 **SML** 和实验室的 **SML-OVR (lab)**、实验室的 **StackingNet (lab)**，以及一批群体标注和真值发现类聚合方法（**Dawid-Skene**、**EBCC**、**GLAD**、**ZenCrowd**、**MACE**、**PM**、**LAA**、**LA**、**M-MSR**、**Wawa**）。

<br>

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
pip install -r requirements.txt

# from the repository root, so that `hustbciml` is importable
python -m hustbciml.run --list                                                # every plug-in
python -m hustbciml.run --algorithm EA-EEGNet --dataset Toy --device cpu       # synthetic, no download
python -m hustbciml.run --algorithm EA-EEGNet --dataset BNCI2014001 --itr 3    # real data, via MOABB
```

也可以即时组合一个算法，而不必指定某个预设（preset）：

```bash
python -m hustbciml.run --aligner EA --augmenter CSDA --backbone DBConformer \
                        --strategy ERM --head Linear --dataset BNCI2014001 --itr 3
```

每次运行会写入 `results/<setting>/metrics.json`（逐被试准确率以及均值和标准差）、`predictions.npz` 与解析后的 `config.yaml`。完整命令参考见 [`hustbciml/README.md`](hustbciml/README.md)，当前数值见 [`hustbciml/RESULTS.md`](hustbciml/RESULTS.md)。

<br>

## 论文到代码总览

网页应用由人工整理的 YAML 经过单一脚本生成，不依赖任何框架。

- **唯一权威数据源（source of truth）**，位于 [`gallery/data/`](gallery/data/)，包括 `publications.yml`（263 篇论文）、`lab.yml`（简介、核心项目、精选仓库）与 `benchmark.yml`（受控比较排行榜）。

- **生成器（Generator）**，即 [`gallery/build_site.py`](gallery/build_site.py)，把这些 YAML 文件编译为 `docs/data/*.js`。它只需要 PyYAML。

在编辑 `gallery/data/` 下任何 YAML 之后，重新生成网页应用数据：

```bash
python3 gallery/build_site.py     # requires only PyYAML
```

<br>

## 仓库结构

```
HUST-BCIML/
├── docs/                       # THE WEB APP (GitHub Pages source)
│   ├── index.html
│   ├── assets/                 # style.css, app.js  (vanilla JS, no framework)
│   └── data/                   # generated: lab.js, publications.js, benchmark.js
├── gallery/                    # source of truth for the web app's data
│   ├── data/
│   │   ├── publications.yml     # 263 papers (hand-curated)
│   │   ├── lab.yml              # lab bio, anchor project, featured repos
│   │   └── benchmark.yml        # controlled-comparison leaderboard
│   └── build_site.py           # YAML → docs/data/*.js   (requires only PyYAML)
├── hustbciml/                  # THE BENCHMARK  (see hustbciml/README.md)
│   ├── run.py                  # python -m hustbciml.run --algorithm EA-EEGNet --dataset BNCI2014001
│   ├── core/                   # batch, stages (ABCs), registry, pipeline, config, context
│   ├── exp/                    # exp_basic + one Exp class per protocol
│   ├── algorithms/             # aligners / augmenters / models / heads / strategies / presets
│   ├── data_provider/          # datasets, data_factory, splitters, collate
│   ├── utils/                  # metrics, seed, tools
│   ├── scripts/                # ensemble, leaderboard, compare, tuning
│   ├── tests/repro/            # repro_targets.yaml, measured vs. published, per method
│   ├── docs/                   # glossary, porting guide, per-algorithm cards
│   └── RESULTS.md              # the full leaderboard, in Markdown
├── references.bib              # IEEE-style BibTeX for every benchmarked method
└── requirements.txt
```

<br>

## 复现与测量完整性

基准中的每一个数值都是**实测**的三种子均值。没有任何数值是为了对上某篇论文而手工设定的。

每一个数值都记录在 [`hustbciml/tests/repro/repro_targets.yaml`](hustbciml/tests/repro/repro_targets.yaml) 里，协议匹配时对照论文自身的数值，协议不同时对照一个预期行为区间，并附有逐方法的注记。算法[卡片（cards）](hustbciml/docs/cards/README.md)为每种方法提供了报告值与复现值的对照表，以及对所用外部代码的许可与来源审计（license/provenance audit）。

凡是进行了超参数选择的地方，选择都**只在留出的源被试上**进行。一个覆盖学习率、训练时长以及各方法自身损失权衡（loss trade-offs）的小网格，在从不包含目标或测试标签的源域验证数据上打分，获胜配置的三种子测试数值**只有在优于此前数值时**才替换。选择过程从不触及所报告的受试群体，因此没有任何数值被调整到刚好命中某个目标，这一保证依然成立。

> **免责声明（Disclaimer）。**
> 本基准**独立地重新实现**了外部基线与实验室自研方法。
>
> 所报告的结果**都可能与原论文存在差异，也可能包含错误**，无论是基线复现值还是实验室方法数值。原因可能是协议不匹配、忠实但不完美的移植，或者某个超参数选择。
>
> 若您发现任何不一致之处，请提交 issue 或联系维护者。欢迎指正。

<br>

## 扩展基准

添加 `hustbciml/algorithms/<group>/<Name>.py`，在其中定义一个符合相应阶段抽象基类（abstract base class）的类，它会**按文件名自动注册**。

随后用一个预设 YAML 把它组合进来，在有了真实数值之后添加一个复现目标（reproduction target），并撰写一张算法卡片。每个新文件都带有一个标准文件头，包含作者、日期、确切的 IEEE 引用，以及在有原作者代码时指向该代码的链接。

完整工作流见[移植指南（porting guide）](hustbciml/docs/porting_guide.md)。

<br>

## 精选仓库

实验室的代表性仓库被置顶展示在[概览标签页](docs/index.html)，起始于：

- [**DeepTransferEEG**](https://github.com/sylyoung/DeepTransferEEG)
- [**TestEnsemble**](https://github.com/sylyoung/TestEnsemble)
- [**DBConformer**](https://github.com/wzwvv/DBConformer)
- [**EEG-FM-Benchmark**](https://github.com/Dingkun0817/EEG-FM-Benchmark)
- [**EEGAdversarialBenchmark**](https://github.com/xqchen914/EEGAdversarialBenchmark)
- [**NT-Benchmark**](https://github.com/chamwen/NT-Benchmark)
- [**TLBCI**](https://github.com/drwuHUST/TLBCI)

<br>

## 路线图

以下方向计划在未来版本中实现。

- **评估协议**，在当前的跨被试 LOSO 之外，增加被试内（within-subject）与跨会话（cross-session）划分，以及一个在线（流式 streaming）协议。
- **范式广度**，在运动想象之外，增加 ERP/P300（以 ROC-AUC 为主要指标）与 SSVEP。
- **可引用发布**，在结果冻结之后，发布一个带版本号、经 DOI 存档的版本。

<br>

## 引用

若本基准或其中的论文到代码总览对您的工作有帮助，请引用相关的实验室论文，[`references.bib`](references.bib) 中提供了每一种基准方法的 IEEE 风格 BibTeX，也请链接回本仓库。

一个带 DOI 和版本号的可引用发布正在计划中。

<br>

## 联系方式

基准与网页应用由 **李思扬（Siyang Li）** 构建并维护，[个人主页](https://sylyoung.github.io/) &nbsp;·&nbsp; **lsyyoungll@gmail.com**。

伍冬睿教授的邮箱地址可在实验室的任一篇论文中找到。

<br>

## 致谢

数据集通过 [MOABB](https://moabb.neurotechx.com/)（Mother of All BCI Benchmarks）提供。

移植的方法在各自的文件头以及对应的算法卡片中标注其原作者。集成与隐私保护部分所用的群体聚合基线，连同其参考文献，在 [`hustbciml/RESULTS.md`](hustbciml/RESULTS.md) 中致谢。

<br>

## 许可证

本项目以 **MIT 许可证** 发布，完整条款见 [`LICENSE`](LICENSE)。

本基准重新实现或改编了若干先前已发表的方法。每张[算法卡片](hustbciml/docs/cards/README.md)记录了对应方法的代码来源。从零重新实现的部分受本仓库的 MIT 许可证覆盖，改编自某个特定上游仓库的实现则保留该项目原有的许可证条款。数据集依各自提供方的使用条款获取。

---

<div align="center"><sub>HUST-BCIML · MIT License · Brain-Computer Interface and Machine Learning Laboratory, HUST</sub></div>
