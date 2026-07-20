# 注意力变体——滑动窗口、稀疏、差分

> 全注意力是一个圆。每个词元看到每个词元，内存付出代价。四种变体弯曲了圆的形状，收回了一半成本。

**类型:** 构建
**语言:** Python
**前提条件:** 第7阶段·02 (自注意力), 第7阶段·03 (多头), 第7阶段·12 (KV缓存/Flash Attention)
**时间:** 约60分钟

## 问题

全注意力在序列长度上付出`O(N²)`内存和`O(N²)`计算的成本。对于一个128K上下文的Llama 3 70B，每层有160亿个注意力条目，乘以80层。Flash Attention（第12课）隐藏了`O(N²)`激活内存，但没有改变算术成本——每个词元仍然关注所有其他词元。

三类变体改变了注意力矩阵本身的拓扑结构：

1. **滑动窗口注意力（SWA）。** 每个词元关注一个固定的邻居窗口，而非完整前缀。内存和计算降至`O(N · W)`，其中`W`是窗口。Gemma 2/3, Mistral 7B的前几层, Phi-3-Long.
2. **稀疏/块注意力。** 仅选定的对`O(N · W)`被评分；其余被强制为零权重。Longformer, BigBird, OpenAI稀疏变换器。
3. **差分注意力。** 使用独立的Q/K投影计算两个注意力图，一个减去另一个。消除了将权重渗入前几个词元的“注意力沉没”。微软的DIFF Transformer（2024年）。

这些共存。一个2026年的前沿模型通常混合它们：大多数层是SWA-1024，每五层有一个全局全注意力层，还有少数是用于清理检索的差分头。Gemma 3的5:1 SWA与全局比率是当前的教科书默认值。

## 核心概念

### 滑动窗口注意力（SWA）

每个位于位置`i`的查询只关注位置在`[i - W, i]`（因果SWA）或`[i - W/2, i + W/2]`（双向）内的词元。窗口外的词元在得分矩阵中得到`-inf`。

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

对于`N = 8192`和`W = 1024`，得分矩阵期望有1024×8192个非零行——减少了8倍。

**KV缓存随SWA缩减。** 每层只需要保留K和V的最后`W`个词元。对于一个类似Gemma-3的配置（1024窗口，128K上下文），KV缓存减少128倍。

**质量代价。** 仅使用SWA的变换器难以处理长程检索。解决方法：将SWA层与全注意力层交错。Gemma 3使用5:1 SWA:全局。Mistral 7B使用了因果SWA堆叠，其中信息通过重叠窗口“向前流动”——每层将有效感受野扩展`W`，经过`L`层后，模型可以关注回`L × W`个词元。

### 稀疏/块注意力

预先选择一个`N × N`稀疏模式。三种经典形状：

- **局部+步进（OpenAI稀疏变换器）。** 关注最后`W`个词元加上之前的每第`stride`个词元。以`O(N · sqrt(N))`计算量捕获局部和长程信息。
- **Longformer / BigBird。** 局部窗口 + 一小部分全局词元（例如`W`），这些词元关注所有词元并被所有词元关注 + 随机稀疏链接。经验上匹配质量时拥有2倍上下文。
- **原生稀疏注意力（DeepSeek, 2025）。** 学习哪些`W`块重要；在内核级别跳过零块。兼容FlashAttention。

稀疏注意力是一个内核工程的故事。数学很简单（掩码得分矩阵）；优势在于从不将零条目加载到SRAM中。FlashAttention-3和2026年的FlexAttention API使得自定义稀疏模式在PyTorch中成为一等公民。

### 差分注意力（DIFF Transformer, 2024年）

常规注意力存在“注意力沉没”问题：softmax强制每一行求和为1，因此那些不想特别关注任何内容的词元将权重倾泻到第一个词元（或前几个）上。这窃取了本应用于真实内容的能力。

差分注意力通过计算**两**个注意力图并相减来解决这个问题：

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

其中`λ`是一个学习的标量（通常0.5-0.8）。A1捕获真实内容权重；A2捕获沉没量。减法抵消了沉没，将权重重新分配给相关词元。

报告的结果（微软2024年）：困惑度降低5-10%，相同训练长度下有效上下文长度增加1.5-2倍，大海捞针检索更精准。

### 变体比较

|  变体  |  计算  |  KV缓存  |  与全注意力相比的质量  |  生产使用  |
|---------|---------|----------|-----------------|----------------|
|  全注意力  |  O(N²)  |  每层O(N)  |  基线  |  每个模型的默认层  |
|  SWA（窗口1024） |  O(N·W)  |  每层O(W)  |  -0.1困惑度，配合全局层效果好  |  Gemma 2/3, Phi-3-Long  |
|  局部+步进稀疏  |  O(N·√N)  |  混合  |  与SWA相似  |  OpenAI稀疏变换器, Longformer  |
|  BigBird（局部+全局+随机） |  约O(N)  |  混合  |  在2倍上下文下匹配全注意力  |  早期长上下文BERT  |
|  原生稀疏（DeepSeek-V3.2） |  O(N · 活跃比例)  |  O(N)  |  困惑度在0.05以内  |  DeepSeek-V3.2, 2025  |
|  差分  |  O(2·N²)  |  O(2N)  |  困惑度降低5-10%  |  DIFF Transformer, 2026年早期模型  |

```figure
gqa-kv-sharing
```

## 动手构建

参见`code/main.py`。我们实现了一个因果掩码比较器，在一个玩具序列上并排显示全注意力、SWA、局部+步进和差分注意力。

### 步骤 1: 全因果掩码（baseline）

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

第 07 课的基线模型。下三角矩阵；对角线以上的权重为零。

### 步骤 2: 滑动窗口因果掩码

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

一个参数——`window`。当 `window >= n` 时，恢复全因果注意力。当 `window = 1` 时，每个词元只关注自身。

### 步骤 3: 局部 + 步进稀疏掩码

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

密集局部窗口加上每隔 `stride` 个词元回溯到序列起始位置。感受野随层数增加以对数步长增长。

### 步骤 4: 差分注意力（Differential Attention）

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

两个注意力通路，用一个可学习的混合系数相减。在代码中我们比较单注意力与差分注意力的注意力下沉热力图，观察下沉现象消失。

### 步骤 5: KV 缓存大小

打印每个变体在 `N = 131072` 时每层的缓存大小。SWA 和稀疏变体下降 10–100 倍。差分注意力加倍。明智地支付内存账单。

## 使用它

2026 年的生产模式：

```python
from transformers import AutoModelForCausalLM
# Gemma 3 mixes SWA (window=1024) and global layers at 5:1.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+ 中的 FlexAttention 接受一个掩码函数：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

这会被编译为自定义 Triton 内核。常见模式的速度在 FlashAttention-3 的 10% 以内，且掩码函数是 Python 可调用对象。

**如何选择每种模式：**

- **纯全注意力（Pure Full Attention）**——每一层，上下文长度最多约 16K，或检索质量要求极高时。
- **SWA + 全局混合**——长上下文（>32K），训练和推理受内存限制。2026 年 >32K 时的默认选择。
- **稀疏块注意力（Sparse Block Attention）**——自定义内核，自定义模式。保留给专门任务（检索、音频）。
- **差分注意力（Differential Attention）**——任何注意力下沉污染造成影响的场景（长上下文 RAG、大海捞针任务）。

## 发布

请参阅 `outputs/skill-attention-variant-picker.md`。该技能根据目标上下文长度、检索需求以及训练/推理计算配置，为新模型选择注意力拓扑结构。

## 练习

1. **简单。** 运行 `code/main.py`。验证 SWA 在 `window=4` 时每行最后 4 个词元以外的所有位置均为零。验证 `window=n` 在比特级别上完全复现全因果注意力。
2. **中等。** 在第 07 课终期项目之上实现带有 `code/main.py` 的因果 SWA。在 tinyshakespeare 上训练 1000 步。验证损失相对于全注意力回归了多少？峰值内存下降了多少？
3. **困难。** 在终期项目中实现 Gemma-3 风格的 5:1 层混合（5 层 SWA，1 层全局）。在参数量匹配的条件下，比较损失、内存和生成质量与纯 SWA 和纯全局基线的差异。
4. **困难。** 实现带有可学习 `code/main.py` 每头参数的差分注意力。在合成检索任务（1 个目标，2000 个干扰项）上训练。测量检索准确率并与参数量匹配的单注意力基线比较。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  滑动窗口注意力（SWA）  |  "局部注意力（Local Attention）"  |  每个查询关注其最近的 `W` 个词元；KV 缓存缩减为 `O(W)`。  |
|  有效感受野  |  "模型能看到多远的过去"  |  在一个 `L` 层 SWA 堆叠中，窗口大小为 `W`，最多可覆盖 `L × W` 个词元。  |
|  Longformer / BigBird  |  "局部 + 全局 + 随机"  |  具有少量始终关注全局词元的稀疏模式；早期的长上下文方法。  |
|  原生稀疏注意力（Native Sparse Attention）  |  "DeepSeek 的内核技巧"  |  学习块级稀疏性；在内核层面跳过零块同时保持质量。  |
|  差分注意力（Differential Attention）  |  "两个映射，一个减去另一个"  |  DIFF Transformer：从第一个注意力映射中减去可学习的 `λ` 乘以第二个注意力映射，以消除注意力下沉。  |
|  注意力下沉（Attention Sink）  |  "权重泄漏到词元 0"  |  Softmax 归一化强制每行和为 1；无信息的查询将权重倾倒在位置 0。  |
|  FlexAttention  |  "掩码即 Python"  |  PyTorch 2.5+ API，将任意掩码函数编译为 FlashAttention 形状的内核。  |
|  层类型混合  |  "5:1 的 SWA 到全局比例"  |  在堆叠中交错稀疏和全注意力层，以在更低内存下保持质量。  |

## 延伸阅读

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)——经典的滑动窗口+全局词元论文。
- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)——局部+全局+随机。
- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)——OpenAI 的局部+步进模式。
- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)——1:1 的 SWA:全局混合。
- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)——5:1 混合，窗口=1024，现已成为教科书默认。
- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)——DIFF Transformer 论文。
- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)——DeepSeek-V3.2 的可学习稀疏注意力。
- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)——"掩码即可调用对象"模式在 Use It 中的 API 参考。
