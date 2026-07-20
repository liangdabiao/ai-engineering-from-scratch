# 多头注意力(Multi-Head Attention)

> 一个注意力头一次学习一种关系。八个头学习八种。头是自由的，多取一些。

**类型：** 构建
**语言：** Python
**先修知识：** 阶段7 · 02（从头实现自注意力）
**时间：** ~75分钟

## 问题

单个自注意力头计算一个注意力矩阵。该矩阵捕捉一种关系——通常是使训练信号损失最小化的那种。如果数据中主语-动词一致、共指、长距离话语和句法分块交织在一起，单个头会将它们混合成一个单一的softmax分布，从而丢失一半的信号。

2017年Vaswani论文的修复方案：并行运行多个注意力函数，每个函数有自己的Q、K、V投影，并将输出拼接起来。每个头在维度为`d_model / n_heads`的较小子空间中操作。总参数量保持不变，表达能力提升。

多头注意力是2026年每个Transformer的默认配置。唯一的争论在于*多少个头*以及键和值是否共享投影（分组查询注意力、多查询注意力、多头潜在注意力）。

## 核心概念

![Multi-head attention splits, attends, concatenates](../assets/multi-head-attention.svg)

**拆分。** 取形状为`X`的`(N, d_model)`。投影到Q、K、V，每个形状为`(N, d_model)`。重塑为`(N, n_heads, d_head)`，其中`d_head = d_model / n_heads`。转置为`(n_heads, N, d_head)`。

**并行注意力。** 在每个头内部运行缩放点积注意力。每个头生成`(N, d_head)`。这些头在嵌入的不同子空间上操作，在注意力计算期间彼此不通信。

**拼接并投影。** 将头堆叠回`(N, d_model)`，并与形状为`W_o`的可学习输出矩阵`(d_model, d_model)`相乘。`W_o`是头混合的地方。

**为何有效。** 每个头可以专业化，而无需与其他头竞争表示预算。2019–2024年的探测研究显示了不同的头角色：位置头、关注前一个词元的头、复制头、命名实体头、归纳头（这是上下文学习的基础）。

**2026年的变体谱系：**

|  变体(Variant)  |  Q头  |  K/V头  | 使用方 |
|---------|---------|-----------|---------|
|  多头(MHA)  |  N  |  N  | GPT-2, BERT, T5 |
|  多查询(MQA)  |  N  |  1  | PaLM, Falcon |
|  分组查询(GQA)  |  N  | G (例如 N/8)  | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
|  多头潜在(MLA)  |  N  | 压缩为低秩  | DeepSeek-V2, V3 |

GQA是现代默认选择，因为它将KV缓存内存减少了`N/G`倍，同时保持了几乎完整的质量。MLA更进一步，将K/V压缩到潜在空间中，然后在计算时投影回来——消耗FLOPs，但节省更多内存。

```figure
multihead-split
```

## 动手构建

### 第1步：从已有的单头注意力中拆分出头

取第02课中的`SelfAttention`，并用拆分/拼接对包裹它。参见`code/main.py`的numpy实现；逻辑是：

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

一次重塑和一次转置。没有循环。这正是PyTorch在`nn.MultiheadAttention`下所做的。

### 第2步：每个头运行缩放点积注意力

每个头获得Q、K、V的独立切片。注意力变为批量矩阵乘法：

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

在真实硬件上，`Qh @ Kh.transpose(...)`是一个`bmm`。GPU看到一个形状为`(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`的批量矩阵乘法。增加头是免费的。

### 第3步：分组查询注意力变体

只有键和值的投影发生变化。Q获得`n_heads`组；K和V获得`n_kv_heads < n_heads`组并重复以匹配：

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

推理时这节省内存，因为只有`n_kv_heads`份副本存在于KV缓存中，而不是`n_heads`。Llama 3 70B使用64个查询头和8个KV头——缓存缩小了8倍。

### 第4步：探测每个头学到了什么

在一个有4个头的短句子上运行MHA。对于每个头，打印`(N, N)`注意力矩阵。你会看到不同的头即使随机初始化也会挑选出不同的结构——这部分是信号，部分是子空间中的旋转对称性。

## 使用它

在PyTorch中，单行版本：

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

截至PyTorch 2.5+的GQA：

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**多少个头？** 2026年生产模型的经验法则：

|  模型大小  |  d_model  |  n_heads  |  d_head  |
|------------|---------|---------|--------|
|  小型（~125M）  |  768  |  12  |  64  |
|  基础型（~350M）  |  1024  |  16  |  64  |
|  大型（~1B）  |  2048  |  16  |  128  |
|  前沿型（~70B）  |  8192  |  64  |  128  |

`d_head` 几乎总是落在 64 或 128。它是一个头能“看到”多少的单位。低于 32 时，头开始与缩放因子 `sqrt(d_head)` 冲突；高于 256 时，你会失去“许多小型专家”的好处。

## 发布

参见 `outputs/skill-mha-configurator.md`。该技能根据参数预算、序列长度和部署目标，为新 Transformer 推荐头数、KV 头数和投影策略。

## 练习

1. **简单。** 从 `code/main.py` 中取出 MHA，将 `n_heads` 从 1 改为 16，`d_model=64` 固定。在一个合成复制任务上绘制一个单层小模型的损失。更多的头有帮助、趋于平稳还是有害？
2. **中等。** 实现 MQA（所有查询头共享一个 KV 头）。测量参数数量相对于完整 MHA 下降了多少。计算推理时 N=2048 的 KV 缓存大小缩小了多少。
3. **困难。** 实现一个微型版本的多头潜在注意力：将 K,V 压缩到秩为 `code/main.py` 的潜在变量，将该潜在变量存储在 KV 缓存中，在注意力时解压缩。在什么 `n_heads` 下，缓存内存下降到完整 MHA 的 1/8 以下，同时质量保持在验证困惑度的 1 位以内？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  头  |  "单个注意力回路"  |  维度为 `d_head = d_model / n_heads` 的一个 Q/K/V 投影，带有其自身的注意力矩阵。 |
|  d_head  |  "头维度"  |  每头隐藏宽度；在生产中几乎总是 64 或 128。 |
|  拆分/合并  |  "重塑技巧"  |  注意力前后的 `(N, d_model) ↔ (n_heads, N, d_head)` 重塑+转置。 |
|  W_o  |  "输出投影"  |  在拼接头之后应用的 `(d_model, d_model)` 矩阵；头在此处混合。 |
|  MQA  |  "一个 KV 头"  |  多查询注意力(Multi-Query Attention): 单个共享的 K/V 投影。最小的 KV 缓存，一些质量损失。 |
|  GQA  |  "自 Llama 2 以来的默认设置"  |  分组查询注意力(Grouped-Query Attention) 使用 `n_kv_heads < n_heads`；重复以匹配 Q。 |
|  MLA  |  "DeepSeek 的技巧"  |  多头潜在注意力(Multi-head Latent Attention)：将 K,V 压缩到低秩潜在变量，在注意力时解压缩。 |
|  归纳头(Induction head)  |  "上下文学习背后的回路"  |  一对检测先前出现并复制其后续内容的头。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — 原始多头规范。
- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — MQA 论文。
- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — 训练后如何将 MHA 转换为 GQA。
- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — MLA 及其在缓存内存上优于 MHA/GQA 的原因。
- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — 对头实际功能的机制性审视。
