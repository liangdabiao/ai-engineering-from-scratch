# Jamba — 混合SSM-Transformer

> 状态空间模型(State Space Models, SSMs)和Transformer追求不同的目标。Transformer通过二次成本注意力(Qudratic Cost Attention)换取质量；SSM通过循环(Recurrence)实现线性时间推理和恒定内存，但质量落后。AI21的Jamba（2024年3月）和Jamba 1.5（2024年8月）将两者置于同一模型中：每7个Mamba层搭配1个Transformer层，每隔一个块使用MoE，以及一个可容纳于单个80GB GPU的256k上下文窗口。Mamba-3（ICLR 2026）通过复值状态空间和MIMO投影加强了SSM侧。本课从头到尾阅读这两种架构，并解释为何混合配方在纯SSM和纯Transformer的长上下文尝试均未成功的情况下，能够经受三年规模的考验。

**类型：** 学习
**语言：** Python（标准库，层混合计算器）
**先修知识：** 阶段10·14（开源模型架构），阶段10·17（原生稀疏注意力）
**时间：** 约60分钟

## 学习目标

- 解释Jamba块中的三个原语——Transformer层、Mamba层、MoE——以及1:7:偶数的交错配方。
- 概述SSM循环的高层形式，以及为何它能实现恒定内存推理。
- 计算Jamba模型在256k上下文下的KV缓存占用，并与纯Transformer模型所需量进行比较。
- 列举Mamba-3的三项创新（指数梯形离散化、复值状态更新、MIMO）以及每项创新针对的问题。

## 问题

注意力与序列长度呈二次关系。状态空间模型是线性的。这种差异会累积：在256k token时，Transformer注意力图每头有650亿条目；而SSM的循环状态大小固定，与序列长度无关。

纯SSM模型（Mamba、Mamba-2）在小规模下与Transformer的困惑度相当，但在状态追踪任务上落后，并在某些类别的上下文检索中失败。直觉：SSM将历史压缩为固定状态，当历史很长时，信息会泄漏。注意力精确记住所有内容，但付出二次代价。

显而易见的解决方案：两者都用。在精确回忆重要的地方使用Transformer层；其他地方使用SSM层。调整比例。Jamba是首个大规模部署这种混合配方的生产级模型（520亿参数总量，120亿活跃参数，256k上下文，单个80GB GPU）。Jamba 1.5将家族扩展到3980亿总量/940亿活跃参数。Mamba-3（ICLR 2026）是目前最佳的纯SSM基准，混合模型可以围绕它重建。

本课阅读所有三篇论文，并形成“选择正确比例”的心智模型。

## 核心概念

### 一页纸上的SSM

状态空间模型通过固定大小的状态`h`处理序列`x_1, ..., x_N`：

```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

每一步中，状态通过线性动力学`A`演化，接受输入`B x_t`，并产生输出`C h_t`。`A, B, C`是可学习的。注意关键性质：计算`y_t`仅需`h_{t-1}`和`x_t`，而不需要任何更早的`x`。内存恒定。推理是每个token O(1)。

建模质量的技巧在于`A`的结构。S4（Gu 2021）使用高度结构化的矩阵，可在训练期间作为长卷积高效计算。Mamba（Gu, Dao 2023）将固定的`A, B, C`替换为依赖于数据的矩阵（“选择性”部分）。Mamba-2（2024）进一步简化结构。Mamba-3（2026）在特定位置重新增加了复杂性。

关键性质：对于解码器LLM，SSM层可以替代注意力层，每层状态大小固定，而不是增长的KV缓存。

### Jamba块

Jamba块根据两个数字交错排列层：

- `l`：注意力与Mamba的比例。Jamba使用`l = 8`，意味着每7个Mamba层搭配1个Transformer层（7 Mamba + 1 Attention = 每组8层）。
- `l`：MoE频率。Jamba使用`l = 8`，意味着每隔一层应用MoE。

块内的层序列：

```
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (where | marks MoE applied)
```

每个Jamba块有8层。在4个块深度（共32层）下，得到28个Mamba层和4个注意力层。其中有16层使用MoE。

### 为什么是1:7比例

AI21进行了消融实验：在长上下文评估中，注意力与Mamba的何种比例能在每个参数的困惑度和上下文召回之间取得最佳平衡？

- 注意力过多（1:1）：质量上升但内存和速度下降。
- 注意力过少（1:15）：内存优秀但上下文检索失败。
- 最佳点：1:7或1:8。

直觉：Transformer层处理精确召回和状态追踪；Mamba层处理廉价的批量处理。

### 位置编码

Mamba层本身具有位置感知能力（通过循环）。最初基于Mamba的混合模型中的注意力层没有使用RoPE——SSM层提供了位置信息。Jamba 1.5为注意力层添加了RoPE，以更好地泛化长上下文，这是基于经验长上下文评估的事后改进。

### 内存预算

对于Jamba-1形状（32层：28 Mamba + 4 Attention，隐藏4096，32个注意力头）：

- KV缓存（仅注意力层）：`2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB` at 256k BF16。只有4个注意力层贡献。
- SSM状态：`2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB` per token前缀，但每层大小固定，不随序列长度增长。典型的Mamba状态为每个特征16，隐藏4096：`28 * hidden * state_size`总计。

与纯Transformer对比：32层，相同隐藏，全MHA 32头：`2 * 32 * 32 * 128 * 256k * 2 = 128 GB` at 256k BF16。KV缓存减少8倍。即使与大多数2024模型使用的GQA(8)基线（`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`）相比，Jamba的1:7混合模型在16 GB下仍然小2倍。

这就是AI21所说的“256k上下文在单个80GB GPU上”。全MHA纯Transformer的KV缓存无法容纳；即使是GQA基线也没有为权重和激活留下空间；而Jamba可以。

### Mamba-3：2026年的纯SSM基线

Mamba-3（ICLR 2026，arXiv:2603.15569）在纯SSM一侧引入了三项创新：

1. **指数梯形离散化。** 将Mamba-2中的欧拉方法离散化替换为更具表达力的递推关系。在核心递推中对状态输入应用类似卷积的操作，而不是作为`x_t`上的外层卷积。

2. **复数值状态更新。** 先前的Mamba将状态矩阵从复数（S4）缩减为实数对角矩阵（Mamba），再缩减为缩放单位矩阵（Mamba-2）。Mamba-3重新引入了复数值——相当于在状态上应用数据相关的旋转嵌入。这恢复了之前实数简化所损失的状态追踪能力。

3. **多输入多输出(MIMO)投影。** 使用矩阵值投影代替每特征标量投影。在不增加解码延迟的情况下提高了建模能力和推理时的硬件利用率。

在1.5B参数规模下，Mamba-3的平均下游准确率比Gated DeltaNet提高了0.6个百分点；MIMO变体额外增加了1.2个百分点，总共提升1.8个百分点。在相同状态大小下，Mamba-3以一半的状态大小匹配Mamba-2的性能。

Mamba-3尚未在量产规模的混合模型中部署——但它是下一代Jamba级别模型中SSM一侧的明显候选。

### 何时采用混合模型

混合模型在以下情况胜出：

- 上下文足够长，纯Transformer的KV缓存变得难以承受（64k以上）。
- 任务混合了短程结构（适合SSM）和长程回忆（需要Transformer）。
- 您希望在单GPU内存预算下部署，而纯Transformer的KV缓存单独就无法容纳。

混合模型在以下情况失利：

- 上下文较短（16k以下）。SSM开销被浪费；纯Transformer即可。
- 任务需要处处注意（深层推理、多文档交叉引用）。混合模型中注意力层的稀疏性会带来损害。
- 您正在扩展到万亿参数的前沿模型。纯Transformer + MLA + MoE（DeepSeek-V3风格）目前正在能力竞赛中获胜。

### 竞争格局

|  模型  |  系列  |  规模  |  独特主张  |
|-------|--------|------|-------------|
|  Mamba-2  |  纯SSM  |  3B  |  线性时间，恒定内存  |
|  Jamba  |  混合  |  52B/12B  |  80GB上256k  |
|  Jamba 1.5 Large  |  混合  |  398B/94B  |  企业级长上下文  |
|  Mamba-3  |  纯SSM  |  1.5B（论文）  |  恢复状态追踪  |
|  DeepSeek-V3  |  纯Transformer + MoE  |  671B/37B  |  前沿能力  |

2026年的格局：纯Transformer MoE主导前沿，但混合模型占据256k以上上下文的小众领域。Mamba-3的状态追踪优势可能推动下一代混合模型中的比例降低（更多SSM，更少注意力）。

```figure
swiglu-ffn
```

## 使用它

`code/main.py`是一个用于混合架构的内存计算器。给定SSM-Transformer比例和隐藏大小/层数配置，它计算：

- 目标上下文下的KV缓存。
- SSM状态内存。
- 一系列模型形状在上下文N下的总内存。

该计算器支持：

- 纯Transformer基线（KV缓存随N增长）。
- Jamba风格的1:7混合。
- 纯SSM（完全没有KV缓存）。

这些数字直接来自Jamba-1和Jamba-1.5论文中已发布的形状，并针对假设变体进行了外推。

实际部署的集成考虑：

- 大多数生产推理服务器（vLLM、SGLang）支持Jamba和Mamba。请检查具体版本。
- 在256k上下文下，Jamba的内存优势体现在并发请求吞吐量上。在相同的VRAM上，您能容纳比Transformer序列更多的Jamba序列。
- Mamba-3作为独立模型尚未在生产中部署——目前仅为1.5B的研究预览。

## 发布

本课生成`outputs/skill-hybrid-picker.md`。给定工作负载规格（上下文长度分布、任务混合、内存预算），它会在纯Transformer、Jamba风格混合模型和纯SSM之间进行推荐，并明确解释内存与质量的权衡。

## 练习

1. 运行`code/main.py`计算一个32层纯Transformer（隐藏大小4096，32头）和一个相同形状的Jamba-1混合模型在256k上下文下的KV缓存。验证AI21论文声称的约8倍内存缩减。

2. 修改计算器以建模1:3混合（4 Mamba : 1 Attention）和1:15混合（14 Mamba : 1 Attention）。绘制KV缓存与比例的关系图。在什么比例下KV缓存等于SSM状态内存？

3. 阅读Jamba论文（arXiv:2403.19887）第3节。解释为什么AI21使用Mamba-1而不是Mamba-2，尽管Mamba-2更快。提示：混合消融部分对此有说明。

4. 计算Jamba 1.5 Large（总计398B，活跃94B）中每隔一层使用MoE的参数开销。比较其活跃比率与DeepSeek-V3（37B/671B），并解释为什么Jamba的架构使活跃比率更高。

5. 阅读Mamba-3论文（arXiv:2603.15569）第3节。用三句话解释为什么复数值状态更新等价于数据相关的旋转嵌入。将答案与第7阶段·第04课的RoPE推导联系起来。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
| 状态空间模型(SSM) | "固定状态的循环" | 具有学习到的循环的层 `h_t = A h_{t-1} + B x_t`；每个token的常数内存 |
| 选择性SSM | "Mamba的诀窍" | 数据相关的A、B、C参数，使模型在线性时间内具有类似门控的选择性 |
| 注意力与Mamba比率 | "注意力层的数量" | 在Jamba中，`l = 8` 表示每7个Mamba层有1个注意力层 |
| Jamba块 | "8层组" | 一个注意力 + 七个Mamba + 交替位置的MoE |
| SSM状态 | "隐藏缓冲区" | 每层固定大小的状态，替代Mamba层的KV缓存 |
| 256k上下文 | "Jamba的标志性数字" | Jamba-1在单个80GB GPU上适用的序列长度；纯Transformer在该规模下无法做到 |
| Mamba-3 | "2026年纯SSM" | 当前最佳的纯SSM架构，具有复数状态和MIMO；混合模型重建的基线 |
| MIMO | "多输入多输出" | Mamba-3创新，使用矩阵值投影替代每个特征的标量 |
| 指数梯形离散化 | "Mamba-3的循环" | 更具表达力的循环，涵盖了Mamba-2的欧拉方法离散化 |
| 混合架构 | "混合注意力和SSM" | 任何交替使用Transformer和SSM层的模型；Jamba是生产环境中的原型 |

## 延伸阅读

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — 原始Jamba论文，比率消融实验，256k上下文声明
- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — 扩展系列，398B/94B和12B/52B公开版本
- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — Jamba所基于的选择性SSM论文
- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — 简化的结构化状态空间后继
- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — 复数值状态，MIMO，2026年纯SSM前沿
- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — S4论文，SSM谱系在LLM中的起点
