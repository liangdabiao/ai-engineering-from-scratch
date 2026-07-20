# 差分注意力(V2)

> Softmax注意力会将少量概率分散到每个不匹配的词元上。在超过10万个词元时，这种噪声会累积并淹没信号。差分Transformer（Ye等人，ICLR 2025）通过计算两个Softmax的差来修复此问题，从而减去共享的噪声底限。DIFF V2（微软，2026年1月）是对生产栈的重写：解码延迟与基线Transformer匹配，无需自定义内核，兼容FlashAttention。本课将从V1到V2进行端到端讲解，并提供一个可在stdlib Python中运行的差操作玩具实现。

**类型：** 构建
**语言：** Python (stdlib)
**前置知识：** 阶段7·02（自注意力）、阶段7·15（注意力变体）、阶段10·14（架构详解）
**时间：** 约60分钟

## 学习目标

- 精确说明为什么Softmax注意力存在噪声底限，以及它为何随上下文长度增长。
- 推导差分注意力公式，并解释减法如何抵消共享的噪声成分同时保留信号。
- 逐步分析V1到V2的区别：哪些部分更快、更简单、更稳定，以及每个变化为何对生产预训练是必要的。
- 用纯Python从头实现差分注意力，并在合成信号加噪声查询上实证验证噪声抵消特性。

## 问题

标准Softmax注意力有一个数学性质，在规模放大时会变成操作上的麻烦。对于查询`q`，注意力权重为`softmax(qK^T / sqrt(d))`。Softmax永远无法产生精确的零——每个不匹配的词元都会获得一些正质量。这种剩余质量就是噪声，并且它随上下文长度缩放。在128k个词元时，即使每个不匹配词元只获得0.001%的概率，127,999个词元合计贡献约12%的总和。模型必须学会绕过随上下文增长的噪声底限。

实证上，这表现为注意力头干扰：长上下文RAG中的幻觉引用、100k词元检索任务中的中间丢失失败，以及超过32k的“大海捞针”基准上的细微精度下降。差分Transformer论文（arXiv:2410.05258，ICLR 2025）测量了差距：DIFF Transformer在相同规模下比基线具有更低的困惑度、更高的长上下文准确率和更少的幻觉。

DIFF V1存在三个问题，使其无法进入前沿预训练流程。它的值缓存每次解码步需要加载两次，需要自定义的CUDA内核（破坏FlashAttention兼容性），并且其逐头RMSNorm在70B以上规模的长时训练中导致不稳定。DIFF V2（微软unilm博客，2026年1月20日）修复了这三个问题。本课将讲解两个版本，构建差分算子，并在玩具查询上基准测试噪声抵消。

## 核心概念

### Softmax的噪声底限

对于查询`q`和键`K = [k_1, ..., k_N]`，注意力权重为：

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

没有`w_i`是零。如果`k_i`与`q`完全无关，则分数`q . k_i`不为0——它在零附近波动，方差为`||q||^2 / d`。经过Softmax归一化后，每个无关词元仍对加权和贡献`O(1/N)`。无关词元的总贡献为`O((N-1)/N) = O(1)`——这不是一个小量。

模型真正想要的是类似硬top-k的东西：匹配词元上权重高，其他地方权重接近零。Softmax过于平滑，无法直接实现这一点。

### 差分思想

将每个头的Q和K投影分为两组：Q = (Q_1, Q_2) 和 K = (K_1, K_2)。计算两个注意力图：

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

输出：

```
DiffAttn = (A_1 - lambda * A_2) V
```

减法抵消了两个注意力图共享的任何噪声分布。如果两个图在127k个无关词元上都有大致均匀的权重（随机初始化时确实如此），这些部分会相互抵消。而信号——在少数真正相关词元上的尖峰权重——只有在两个图中以相同幅度出现时才会被抵消，而这在模型训练后不会发生。

`lambda`是一个每头可学习的标量，参数化为`lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`。它可以为负。`lambda_init`默认为一个小的正数，如0.8。

### 为何这符合定向降噪

想象两个有噪声的麦克风录制同一声音。两者都拾取说话者加上相关的背景噪声。将一个减去另一个，共享的噪声就会消失。语音之所以保留，是因为两个信号在相位或幅度上差异足够大，从而避免了完全抵消。每头的`lambda`恰好学习了这种平衡。

### V1与V2的区别

V1保持参数数量与基线Transformer相等。为了每头获得两个查询，它将头维度减半。这牺牲了头的表达能力，并且更痛苦的是，每头的值缓存减半。解码步必须加载两次值缓存（每个Softmax分支一次）。结果是：尽管参数数量匹配，解码速度却比基线慢。

V2将查询头数量加倍，并保持KV头不变（从上投影借用参数）。头维度与基线相同。减法后，额外的维度投影回与基线Transformer的O_W投影匹配。同时发生三件事：

1. 解码速度与基线匹配（KV缓存只加载一次）。
2. FlashAttention无需更改即可运行（无需自定义内核）。
3. 解码时的算术强度增加（每字节从HBM加载的计算量更多）。

V2还移除了V1用于稳定减法的逐头RMSNorm。在70B级预训练规模下，该RMSNorm导致后期训练不稳定。V2用一个更简单的初始化方案替代，无需额外模块即可保持训练稳定。

### 何时使用

|  工作负载  |  收益  |
|----------|---------|
|  长上下文RAG（64k以上）  |  更清晰的注意力图，更少的幻觉引用  |
|  “大海捞针”基准  |  超过32k后精度显著提升  |
|  多文档问答  |  更少的跨文档干扰  |
|  8k处的代码补全  |  边际收益，不值得改动架构  |
|  短对话（<4k）  |  与基线基本无区别  |

数值随上下文长度增长。在4k token时，噪声基底足够小，标准注意力即可胜任。在128k时，它会损害性能。

### 与其他2026年技术的结合方式

|  特性  |  是否兼容DIFF V2?  |
|---------|------------------------|
|  GQA  |  是（V2增加Q头数，而非KV头数）  |
|  MLA（DeepSeek）  |  原则上可行，但尚无结合两者的发表论文  |
|  MoE  |  是（注意力与MLP块独立）  |
|  RoPE  |  是（保持不变）  |
|  YaRN / 长上下文缩放  |  是（正是DIFF最有帮助的场景）  |
|  FlashAttention  |  V2中可兼容（V1中不可）  |
|  推测解码  |  是（注意力变化对推测解码循环不可见）  |

```figure
differential-attention
```

## 动手构建

`code/main.py` 使用纯Python实现微分注意力。一个具有已知信号加噪声结构的玩具查询可以让你直接测量噪声消除比率。

### 步骤1：标准softmax注意力

标准库矩阵运算：列表的列表，手动矩阵乘法，通过减去最大值实现数值稳定性的softmax。

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### 步骤2：将Q、K拆分为两半

V1风格：将头维度减半。V2风格：保持头维度不变，将头数加倍。本玩具实现使用V1风格以利于教学清晰——数学上相同，仅记账方式不同。

### 步骤3：两个softmax分支并相减

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

注意：输出权重可以为负。这没问题——值缓存仍然处理带符号的贡献。后续的V投影会吸收符号。

### 步骤4：噪声消除测量

构建一个长度为1024的合成序列。将信号token放在已知位置，其余位置填充噪声。计算（a）标准softmax注意力在信号位置上的权重和（b）微分注意力的权重。测量各自的信噪比。DIFF注意力可靠地产生更高的信噪比，依据两个分支被训练差异的程度，可达到3倍到10倍。

### 步骤5：V1与V2参数统计

给定配置（隐藏层=4096，头数=32，d_head=128），打印：

- 基准Transformer：Q、K、V各自尺寸为`hidden * hidden`，MLP为4 * 隐藏层。
- DIFF V1：Q、K各自尺寸为`hidden * hidden`，V尺寸为`hidden * hidden`（不变），内部头维度减半。每个头增加`hidden * hidden`参数（O(头数 * d_head)）。
- DIFF V2：Q尺寸为`hidden * hidden`，K尺寸为`hidden * hidden`，V尺寸为`hidden * hidden`。额外维度在输入O_W之前投影回原尺度。增加相同数量的`lambda`参数。

该玩具测量V2的额外参数开销（每个注意力块约`hidden * hidden`）并打印。

## 使用它

截至2026年4月，DIFF V2尚未部署到每个生产推理服务器中，但集成工作正在vLLM和SGLang中进行。同时，该模式出现在：

- 微软内部的长上下文生产模型中。
- 多个开放模型训练运行的研究复现，目标上下文长度超过256k。
- 混合架构，将DIFF注意力与滑动窗口注意力交替使用于不同层。

在2026年，你何时会选择使用它：

- 从头训练一个目标有效上下文超过64k的新模型。从开始就加入微分注意力；后期重新训练成本高昂。
- 微调一个长上下文模型，其中“lost-in-the-middle”失败主导了评估。对Q投影进行LoRA可以近似DIFF结构。

你何时不会使用：

- 你正在部署一个预训练好的稠密模型，且其长上下文性能稳定。重新训练的成本通常无法在现有权重上收回。
- 你的上下文长度始终低于16k。噪声基底可忽略不计。

## 发布

本课产生`outputs/skill-diff-attention-integrator.md`。给定模型架构、目标上下文长度、幻觉概况和训练预算，它会生成一个将微分注意力添加到新的预训练运行或LoRA微调中的集成计划。

## 练习

1. 运行`code/main.py`。验证差分注意力(Differential Attention)在合成查询上的信噪比报告是否高于标准softmax注意力。改变噪声幅度，并展示标准注意力变得不可用的交叉点。

2. 计算一个7B类模型(hidden=4096, heads=32, d_head=128, 32 layers)从baseline到DIFF V1以及从baseline到DIFF V2的参数数量变化。展示哪些组件增加了参数，哪些保持不变。

3. 阅读DIFF V1论文(arXiv:2410.05258)的第3节和DIFF V2的Hugging Face博客第2节。用两句话解释为什么V1的每个头的RMSNorm(Per-head RMSNorm)是必要的，以及为什么V2可以移除它而不导致训练发散。

4. 实现一个消融实验：使用`lambda = 0`（纯第一个softmax）和`lambda = 1`（完整减法）计算差分注意力。在合成查询上，测量信号噪声比如何随扫描变化。识别出最大化信号噪声比的`lambda`。

5. 将toy扩展到GQA + DIFF V2。选择8个KV头和32个Q头。证明KV缓存大小与具有相同(8, 32)配置的基线GQA模型匹配。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
| 差分注意力(Differential Attention)  |  "两个softmax相减"  |  将Q、K分成两半，计算两个softmax映射，从第一个中减去第二个（乘以lambda缩放），然后乘以V |
| 噪声底限(Noise floor)  |  "softmax的非零尾部"  |  softmax赋予每个无关token的O(1/N)权重，在长上下文中总和为O(1) |
| lambda  |  "减法尺度"  |  每个头可学习的标量，参数化为`exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`；可以为负 |
| DIFF V1  |  "ICLR 2025版本"  |  原始的差分Transformer；将heads维度减半以保持参数数量，需要自定义内核，解码较慢 |
| DIFF V2  |  "2026年1月的修复"  |  增加Q heads数量同时保持KV heads不变；匹配基线解码速度，并与FlashAttention兼容 |
| 每个头的RMSNorm(Per-head RMSNorm)  |  "V1的稳定器"  |  V1在差分后应用的额外归一化；V2移除了它以阻止后期训练不稳定 |
| 信噪比(Signal-to-noise ratio)  |  "注意力浪费了多少"  |  真实信号位置上的权重与无关位置平均权重的比值 |
| 中间丢失(Lost in the middle)  |  "长上下文失效模式"  |  一种经验现象，长上下文中间文档的检索准确性下降——差分注意力减轻了这一问题 |
| 算术强度(Arithmetic intensity)  |  "每加载字节的FLOPs"  |  V2通过每次KV加载加倍查询数量来提高解码时的比率；对于内存受限的解码很重要 |

## 延伸阅读

- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) — 原始论文，包含噪声消除理论和长上下文消融实验
- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) — 生产堆栈重写，匹配基线解码，与FlashAttention兼容
- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) — 理论分析为何减法恢复了预训练注意力结构
- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) — 参数共享变体
- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) — DIFF减去的基线Transformer
- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) — DIFF注意力瞄准的长上下文基准
