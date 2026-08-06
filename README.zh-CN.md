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

<summary><b>更新日志</b></summary>

完整版本历史见 [`CHANGELOG.md`](CHANGELOG.md)。近期要点：

- **2026-08-06（v1.6.3）** 文件头进一步精简：删去 `Integrated by:` 与 `Author:` 两行，
署名链仅保留「原始作者 → 实现者 → 当前代码」三段，**所有测量数值不变**。

- **2026-08-06（v1.6.2）** 仅调整文件头排版：删去 credit chain 的说明行，`Author:` 行移
到链之后，文件头按「标题、署名链、作者、文献」排列，**所有测量数值不变**。

- **2026-08-06（v1.6.1）** 仅文档与署名说明更新，**所有测量数值不变**。每个算法源文件
头部现在带有完整的署名链：原始作者（论文、期刊与官方代码）、实现者及其仓库、当前移植
的整理者、以及集成者，每个环节都附 GitHub 链接。共同一作以 † 标注，包括 Channel
Reflection（Ziwei Wang† 与 Siyang Li†）与 StackingNet。集成器部分追溯到了最早发布：
EBCC、ZenCrowd、PM、LA、LAA 与 StackingNet 均源自 Chenhao Liu 的
`Flashingcat/Golden_task-Ensemble`，再进入实验室的 TestEnsemble。总览页也新增了实验室
的 `wzwvv/EEGAug` 数据增强仓库。

- **2026-08-06（v1.6.0）** 18 个网络骨干已全部从头重测，订正后的数值已经写入基准表。
新流程逐目标被试执行严格的嵌套留一被试交叉验证，最终固定使用 5 个随机种子。五个订正行
DeepConvNet、ShallowConvNet、ADFCNN、EEGWaveNet、FBMSNet 均为对所引文献的显式架构移植：
Braindecode 的 Deep4Net 与 ShallowFBCSPNet 特征结构、ADFCNN 与 EEGWaveNet 的公开代码拓扑、
以及仅保留 8–32 Hz 六个因果分支的 FBMSNet。MVCNet 以记录在案的三种子历史数值列入网络骨干表。
旧版数值已经撤下，没有沿用到订正后的代码。每个最终折都保存了模型检查点、预测、入选配置、
源码与数据摘要、软件环境和断点状态，完整实验通过独立校验后，导入器才写入新数值。

- **2026-07-31（未发布，审计订正）** 网络骨干表现表的数值改标为历史测量，待按新流程重测。旧流程混用了固定训练计划和
逐骨干调参，全局学习率搜索还会让某名被试在其他 LOSO 折中进入源域验证，从而间接参与自身测试折的参数选择。
新脚本改为逐目标被试做嵌套选择，选参阶段不对外层目标被试做对齐、预测或评分。结果文件和缓存新增源码、软件版本、
预处理、数据摘要与方法参数，身份不符、种子不全或组合器报错时直接终止。DeepConvNet 与 FBMSNet 也改按实际情况标为
历史移植版本，而不是论文忠实复现。本次没有改动任何排行榜数值。

- **2026-07-30（v1.5.0）** 只调整仓库结构。代码包移入 `src/hustbciml/`，仓库根目录下每一项的用途一目了然。
导入名仍是 `hustbciml`，文档中的所有命令都不变，只是现在需要用 `pip install -e .` 安装本包。新增
`pyproject.toml`（同时接管原先的 `pytest.ini`），并删除 33 个内部文件：`RERUN.md` 及执行那次已完成的
v1.2.0 重测的七个脚本，以及 25 个写死了某位贡献者主目录路径的批量运行与启动脚本。测量结果没有任何变化。

- **2026-07-30（v1.4.1）** 恢复概览页上被 v1.3.0 移除的集成学习方法组，14 种集成聚合方法重新列出，
实验室自研的 SML-OVR 和 StackingNet 也重新出现在这份方法清单中，方法标签数由 60 增至 74。集成学习表中
两行非集成参照不计入该清单。测量结果没有任何变化，排行榜的渲染结果完全一致。

- **2026-07-30（v1.4.0）** 集成学习表改用每名源被试三个学习器的配置重新测量，三者分属不同的模型族，
即切空间逻辑回归、CSP-Net 和 EEGConformer，因此每个目标试次收到 (N−1)×3 个硬投票。表中 45 个组合器
和参照单元格全部为新测量值。SML-OVR (lab) 和二分类 SML 在三个数据集上的平均增益从 +0.47 升至 +1.61，
并且首次在三个数据集上全部为正，StackingNet (lab) 从 −0.08 升至 +0.31。`--base hetero` 已移除。

- **2026-07-29（v1.3.2）** 删除 `RESULTS.md` 末尾关于 BNCI2014001 四分类的两个附录。本基准全程为二分类，
文件开头即已声明，但两个附录仍在报告原生四分类结果（随机水平 25%），并且有两个四分类数值混入了
二分类的分数据集讨论，出现在三元组的第三项。以上内容全部删除，相关论述改用现有的二分类证据重新表述。
二分类数值没有任何变化，网页排行榜本来就不包含四分类内容，`BNCI2014001-4` 的代码路径保留，
因为 SML-OVR 的一对多形式正是为多分类而设。

- **2026-07-28（v1.3.1）** 将 StackingNet 中权重和为一的正则项改写为 L1 范数形式，与原作者公开代码一致。
在权重非负时，两种写法取值相同，梯度不同，因此被非负约束截断到零的权重此后保持为零，不再被重新拉起，
这正是该方法本身的行为。测量结果没有变化：只要权重全部为正，两种写法的梯度完全相同，
而在本仓库所用的超参数下正是如此，在三个数据集和三个随机种子的去中心化缓存投票上，两者预测完全一致。
`RESULTS.md`、排行榜和网页应用均未改动。

- **2026-07-27（v1.3.0）** 恢复了集成学习表。v1.2.5 曾短暂将其撤下，此次按 v1.2.4 的数值原样恢复，
未重新测量。改为移除概览页上的集成部分：集成聚合方法的统计数字，以及集成学习一组方法标签，均已去除，
概览页只呈现解码流水线，方法标签总数由 75 降至 60。基准测试页、两份 README 的方法清单，
以及 `RESULTS.md` 中的全部数值均保持不变。

- **2026-07-27（v1.2.4）** 用同样的两种行文方式重写了两份 README，并订正了其中的表述。
中文版此前仍写着每次运行会保存模型检查点，以及超参数选择完全不触及目标标签，
这两点既不符合代码，也与英文版不一致，另外还写了一次对移植代码的许可审计，而这项审计并未进行。
两份文件都把有公开代码的论文数写成 76 篇，实际为 72 篇，也都提到每次运行会写出一个 `config.yaml`，
而代码并不生成该文件。中文版的设计原则一节还补上了此前只有英文版才有的两处说明。

- **2026-07-27（v1.2.3）** 补齐上一版遗漏的说明文字。迁移和集成两张表中有十二条方法说明未纳入 v1.2.2 的改写，
仍保留英式拼写，另有四处是在替读者对数值下判断，而不是把数值本身讲清楚。

- **2026-07-27（v1.2.2）** 重写了网页应用中的全部说明文字。英文改用实验室论文的行文方式，
中文改用实验室公众号稿件的行文方式。同时订正了核心项目卡片上已过时的方法计数，
并补上 `check_i18n.py` 的一处覆盖缺口，此前基准介绍卡片和概览页的说明文字都不在检查范围内，
纳入检查的生成字符串由 39 条增加到 64 条。

- **2026-07-27（v1.2.1）** 把 `scikit-learn` 的版本上限定在 `1.8` 以下。从 1.8 起，
`check_is_fitted` 会去读取 `__sklearn_tags__`，而 crowd-kit 的 Wawa 没有这个属性，
于是该融合方法在真正开始聚合之前就抛出异常，它在排行榜上的那一行也就无法复现。
这个问题是当天新加的持续集成在第一次运行时发现的。另外四个来自 crowd-kit 的融合方法不受影响。

- **2026-07-27（v1.2.0）** 依据一次包含 176 条结论的外部代码审查做了修订：清理了测量路径上的静默回退与缺失校验，
修正了若干方法实现（通道反射、傅里叶替代、CSDA、RA、SML、LAA、PM、CTNet、骨干网络的形状探测），
对四处**逐字继承自参考实现**的缺陷选择如实记录而非直接改掉（改动会使数值失去与已发表基线的可比性），
让复现登记表变成可执行的测试，补上了持续集成与链接检查，并订正了与代码不符的表述。
这些修改改变了代码算出的结果，因此**凡是受影响的排行榜单元格都重新测量了一遍**，
而且每个单元格都在当初产出已发表数值的那台机器上重测，因为同一份代码在不同的 BLAS 实现上会给出不同的数字。
每个已发表单元格由哪台机器产出，记录在 [`cell_origin.tsv`](src/hustbciml/scripts/cell_origin.tsv)。
本次不涉及的方法作为对照组一并重跑，结果与 v1.1.x 逐被试完全一致。
集成学习表改为三个随机种子的均值加标准差，此前只有单个种子、且不给标准差。
另外新增**经典流程**表，收录两行不含网络的方法，方法数变为 58。
这两行此前只写在 `RESULTS.md` 里，却不在任何排行榜表中，于是没有任何检查能核对它们。

- **2026-07-24（v1.1.3）** 重写了全部 22 个实验室方法的源码文档，使其与各自论文一致，仅改注释，基准数值不变。

- **2026-07-24（v1.1.2）** 重新组织了迁移与集成两族，隐私保护一族更名为隐私保护迁移，通道对称不再作为基准增强器，方法数变为 56。

- **2026-07-24（v1.1.1）** 拆分了集成学习表，数据增强器改用全称，论文索引去重（275 篇减至 263 篇）。

- **2026-07-24（v1.1.0）** 新增十种网络骨干、八种数据增强基线，以及五种实验室方法（CSP-Net、DJP-MMD、LSFT、MSDT 与完整的 MEKT），均在三个数据集、三个随机种子上完成基准测试，并上线了网页应用的排行榜与论文到代码总览。

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

订正后的五种子网络骨干实验实际使用的 Python 3.11 与 CUDA 软件包版本，完整冻结在
[`requirements-network-production.txt`](requirements-network-production.txt) 中。它用于锁定测量环境，不替代上面的通用安装方式。

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
├── requirements.txt            # 通用安装依赖
└── requirements-network-production.txt  # 订正后网络实验的精确环境锁
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
