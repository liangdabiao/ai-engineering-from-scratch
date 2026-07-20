# 原生稀疏注意力(Native Sparse Attention, DeepSeek NSA)

> 在64k token下，注意力(Attention)消耗了解码延迟的70-80%。每个开源模型实验室都有改进方案。DeepSeek的NSA（ACL 2025最佳论文）是其中脱颖而出的：三个并行注意力分支——压缩的粗粒度token、选择性保留的细粒度token以及用于局部上下文的滑动窗口——通过一个可学习的门控(Gate)组合。该算法与硬件对齐（对内核友好），原生可训练（可在预训练中使用，而非推理时附加），在64k解码上运行速度比FlashAttention更快，同时匹配或超越全注意力质量。本课将从头构建这三个分支，并展示为何稀疏性(Sparsity)是端到端可微分的。

**类型：** 构建
**语言：** Python (stdlib)
**前置要求：** 第7阶段·12课 (KV缓存, flash-attention)，第7阶段·15课 (注意力变体)，第10阶段·16课 (微分注意力(Differential Attention))
**时长：** ~60分钟

## 学习目标

- 阐述NSA的三个注意力分支及其各自捕获的内容。
- 解释为何NSA是“原生可训练的”，而先前的稀疏注意力(Sparse Attention)方法仅适用于推理。
- 计算在64k上下文下，NSA相较于全注意力(Full Attention)的注意力计算节省量，作为压缩块大小(Compression Block Size)和选择top-k的函数。
- 使用stdlib Python在短合成序列上实现三分支组合，并验证门控权重的行为。

## 问题

序列长度为N的全注意力每个层成本为`O(N^2)`时间和`O(N)` KV缓存。在64k token下，计算和内存带宽的数字是灾难性的。根据NSA论文的测量理论估计：在64k下，注意力占解码总延迟的70-80%。所有下游指标——TTFT、tokens/sec、每百万token成本——都受注意力成本主导。

稀疏注意力是显而易见的答案。先前的尝试可分为两类。固定模式稀疏性(Fixed-pattern sparsity)（滑动窗口(Sliding-window)、条带化(Strided)、块局部(Block-local)）会丢弃信息，并且在长程检索任务上失败。推理时稀疏性(Inference-time sparsity)（KV缓存剪枝、H2O、StreamingLLM）应用于在密集注意力(Dense Attention)上预训练的模型，仅能恢复潜在加速的一小部分，因为模型从未被要求在稀疏模式下路由信息。

原生稀疏注意力(Native Sparse Attention)（Yuan等，DeepSeek + 北大 + 华盛顿大学，ACL 2025最佳论文，arXiv:2502.11089）两者兼顾：模型在预训练期间学习的稀疏模式，通过一个与内核对齐的算法实现，该算法在推理时真正带来计算节省。两年后，NSA或其直接后代将成为每个前沿长上下文模型的默认注意力机制。

## 核心概念

### 三个并行分支

对于每个查询(Query)，NSA对KV缓存的三种不同视图分别运行三次注意力：

1. **压缩分支(Compressed branch)。** Token被分组为大小为`l`（通常为32或64）的块。每个块通过一个小型可学习MLP压缩为单个摘要token。查询在这些压缩token上执行注意力，获得整个序列的粗粒度视图。

2. **选择分支(Selected branch)。** 利用压缩分支的注意力分数，识别出与当前查询最相关的前k个块。读取这些块中的细粒度（未压缩）token，查询对所有细粒度token执行注意力。可以将压缩分支注意力视为选择的路由信号。

3. **滑动窗口分支(Sliding-window branch)。** 查询对最近的`W`个token（通常为512）执行注意力，以获取局部上下文。该分支捕获结构密集的短距离模式（语法、局部共指），这些模式可能被其他两个分支遗漏。

三个分支的输出通过一个可学习的按位置门控(Position-wise Gate)组合：

```
out = g_cmp * out_cmp + g_sel * out_sel + g_win * out_win
```

`g_cmp, g_sel, g_win`是来自查询的小型MLP的门控权重。它们不必和为1——可以独立地对分支进行加权。

### 为何这是“原生可训练的”

选择步骤（top-k块）是离散的。离散操作会阻断梯度流。先前的稀疏注意力工作要么跳过了通过选择的反向传播（限制了训练），要么使用了在推理时无法实现真正稀疏性的连续松弛方法。

NSA规避了这个问题：压缩分支注意力本身就是对整个序列的可微分粗粒度注意力。top-k操作仅复用来自压缩分支的顶部注意力分数，以选择要加载的细粒度块。梯度通过压缩分支分数（这些分数既影响压缩输出，也影响选择逻辑）流动，而所选块对最终输出的贡献也是可微分的。非可微分的`top_k`操作在前向计算图中是无操作的——它仅控制哪些块从内存加载。

这就是为何NSA可以在预训练中端到端使用的原因。模型学习通过三个分支联合路由信息，产生一个在推理时真正实现承诺加速的稀疏模式。

### 与硬件对齐的内核

NSA的内核专为现代GPU内存层次结构设计。内核按GQA组加载查询（外层循环），为该组获取对应的稀疏KV块（内层循环），并在SRAM上运行注意力。由于每个查询组看到相同选中的块（选择是按查询组而非按查询头），KV加载在组内被分摊。算术强度保持较高。

论文报告，在64k解码上，Triton内核比FlashAttention快9倍，且加速比随序列长度增长。前向和反向内核均已提供。

### 计算预算

设`N`为序列长度，`l`为压缩块大小，`k`为top-k选择数量，`w`为滑动窗口大小，`b`为选择的块大小（通常等于`l`）。

- 压缩分支：每个查询`O(N/l)`个键，总计`O(N * N / l)`。
- 选择分支：每个查询`O(N/l)`个键，总计`O(N * N / l)`。
- 滑动分支：每个查询`O(N/l)`个键，总计`O(N * N / l)`。

总计：`O(N * (N/l + k*b + w))`。

当`N = 64k, l = 64, k = 16, b = 64, w = 512`时：每查询成本为`1000 + 1024 + 512 = 2536 keys`。全注意力为`64000 keys`。计算减少25倍。

当`N = 128k, l = 64, k = 16, b = 64, w = 512`时：每查询成本为`2000 + 1024 + 512 = 3536 keys`。全注意力为`128000 keys`。减少36倍。收益随序列长度增加，这正是关键所在。

### 比较

|  方法  |  可微分  |  真实推理加速  |  长程检索  |
|--------|---------------|----------------------|-------------------|
|  仅滑动窗口  |  是  |  是  |  失败  |
| 步进/块稀疏 | 是 | 是 | 部分 |
| KV剪枝(H2O, StreamingLLM) | 不适用(推理时) | 是 | 部分 |
| MoBA(Moonshot) | 部分 | 是 | 良好 |
| NSA | 是(原生支持) | 是(在64k时9倍) | 匹配全注意力 |

MoBA(Moonshot, arXiv:2502.13189)是同期发表的，采用了类似的三比一更好的方法，将MoE原理应用于注意力块。NSA和MoBA是2026年长上下文预训练需要了解的两种架构。

```figure
sliding-window-attention
```

## 动手构建

`code/main.py` 在一个短合成序列上实现三个分支并展示：

- 压缩MLP（为教学清晰起见使用简单的均值池化基线；真正的NSA使用学习的MLP）。
- 由压缩分支分数驱动的top-k块选择。
- 对最后`w`个token的滑动窗口注意力。
- 门控组合。
- 与全注意力比较的计算计数打印输出。

### 步骤1：将token压缩成块

```python
def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
```

### 步骤2：压缩分支注意力

对查询和压缩后的键运行softmax注意力。压缩分支分数同时作为top-k选择的信号。

### 步骤3：top-k块选择

选取`k`个得分最高的压缩块的索引。从这些块中加载原始的未压缩token，并对它们运行注意力。

### 步骤4：滑动窗口注意力

取最后`w`个token并对它们运行标准注意力。

### 步骤5：门控+组合

查询上的一个小MLP产生三个门控权重。最终输出是三个分支输出的加权和。

### 步骤6：计算计数

打印每个分支每个查询关注的键的数量以及总数。与`N`（全注意力）比较。在一个1024 token的合成数据上，`l = 32, k = 4, w = 128`，NSA每个查询看到`32 + 128 + 128 = 288`个键，而全注意力是1024个——减少了3.5倍。

## 使用它

NSA已部署在DeepSeek自己的长上下文预训练流程中。截至2026年4月，在公共推理栈中的集成状态：

- **DeepSeek内部**：原生支持，已发布权重使用NSA或其继任者DSA（Deepseek稀疏注意力）。
- **vLLM**：正在为DeepSeek-V3.x权重开发实验性NSA支持。
- **SGLang**：已发布NSA基准测试；生产路径跟随vLLM。
- **llama.cpp / CPU**：不支持；在CPU吞吐量下内核分解的开销不值得。

何时使用NSA：

- 针对64k以上上下文且有大量计算预算的预训练或持续训练运行。
- DeekSeek自身长上下文检查点的推理。权重是NSA原生的。

何时不使用：

- 服务于现有的密集注意力预训练模型。没有持续训练就无法改造NSA。
- 上下文低于16k。三个分支的开销主导了节省。
- Batch-1交互式聊天。延迟敏感的解码有收益，但仅限长上下文。

## 发布

本节课产生`outputs/skill-nsa-integrator.md`。给定一个长上下文预训练运行规范，它产生一个NSA集成计划：压缩块大小、top-k、滑动窗口、门控MLP宽度、内核选择，以及证明架构更改合理的特定长上下文评估。

## 练习

1. 在1024 token的合成数据上运行`code/main.py`。在三个预设中扫描`(l, k, w)`并打印计算计数。识别出在针堆测试中每个查询达到最低键数同时保持对全注意力95%召回率的预设。

2. 用一个小型学习MLP（2层，隐藏层32）替换均值池化压缩器。在一个信号是块平均值的合成任务上训练它。在保留数据上测量与均值池化基线的困惑度差距。

3. 实现门控MLP。它以查询为输入，输出三个标量。展示门控行为合理：在随机查询上接近均匀加权，当查询命中一个遥远的块时，在选中的分支上权重很大。

4. 计算启用NSA的70B模型在128k上下文下的KV缓存内存预算。KV头数为8，头维度128，BF16。与全注意力和MLA（阶段10·14显示了MLA的数字）比较。识别NSA细粒度分支KV缓存等于全注意力时的序列长度。

5. 阅读NSA论文（arXiv:2502.11089）第4节，用三句话解释为什么压缩分支的注意力分数被重用于top-k选择，而不是计算单独的路由分数。将答案与梯度流联系起来。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
| 压缩分支 | "粗视图" | 对块平均键的注意力，以每个查询O(N/l)个键提供全局上下文 |
| 选中分支 | "Top-k 块" | 对具有最高压缩分支分数的 `k` 块进行细粒度注意力 |
| 滑动窗口 | "局部上下文" | 对最近的 `W` 个 token 进行注意力以捕获短程模式 |
| 原生可训练性 | "预训练时即启用稀疏性" | 稀疏模式在预训练期间学习，而非在推理时附加 |
| 压缩块大小 l | "粗略视图的组大小" | 多少个 token 合并为一个摘要；通常为 32-64 |
| Top-k | "保留的块数" | 未被压缩的 token 被读取的压缩块数量；通常为 16 |
| 滑动窗口 W | "局部注意力半径" | 通常为 512；过短损害局部连贯性，过长浪费算力 |
| 分支门 | "如何混合三者" | 每个位置上的 MLP 输出，用于加权三个分支的贡献 |
| 硬件对齐 | "对内核友好的稀疏性" | 选择的稀疏模式使得实际 GPU 内核达到理论加速 |
| DSA | "NSA 的后继者" | Deepseek Sparse Attention，在 DeepSeek 系列中紧随 NSA 的架构 |

## 延伸阅读

- [Yuan et al. — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089) — 论文
- [Yuan et al. — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089) — NSA 所针对的架构家族
- [Yuan et al. — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089) — 同步工作，基于块的 MoE 风格注意力
- [Yuan et al. — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089) — 滑动窗口起源
- [Yuan et al. — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089) — NSA 改进的推理时稀疏性基线
- [Yuan et al. — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089) — NSA 内核在 64k 长度下超越的全注意力基线
