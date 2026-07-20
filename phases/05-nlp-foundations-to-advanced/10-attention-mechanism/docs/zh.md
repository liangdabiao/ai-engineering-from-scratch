# 注意力机制 — 突破

> 解码器不再眯着眼睛看压缩的摘要，而是开始审视整个源。此后的一切都是注意力机制加上工程实现。

**类型：** 构建
**语言：** Python
**预备知识：** 第5阶段·09（序列到序列模型）
**时间：** 约45分钟

## 问题

第9课以一次有意义的失败告终。在玩具复制任务上训练的GRU编码器-解码器，在长度为5时准确率为89%，到长度为80时几乎接近随机水平。原因是结构性的，而非训练错误：编码器收集到的每一点信息都必须塞进一个固定大小的隐藏状态中，而解码器永远看不到其他任何东西。

Bahdanau、Cho和Bengio在2014年发表了一个三行代码的修正方案。不再只给解码器最终的编码器状态，而是保留每一个编码器状态。在每个解码步骤中，计算编码器状态的加权平均值，其中权重表示“解码器当前需要关注编码器位置`i`多少？”这个加权平均值就是上下文，它会在每个解码步骤变化。

这就是全部思想。Transformer扩展了它。自注意力将其应用于单个序列。多头注意力并行运行。但2014年的版本已经打破了瓶颈，一旦你掌握了它，转向Transformer就只是工程问题，而非概念问题。

## 核心概念

![Bahdanau attention: decoder queries all encoder states](../assets/attention.svg)

在每个解码步骤`t`：

1. 使用前一个解码器隐藏状态`s_{t-1}`作为**查询**。
2. 对每个编码器隐藏状态`s_{t-1}`进行评分。每个编码器位置一个标量。
3. 对分数进行Softmax得到注意力权重`s_{t-1}`，权重之和为1。
4. 上下文向量`s_{t-1}`。编码器状态的加权平均值。
5. 解码器接收`s_{t-1}`加上前一个输出词元，生成下一个词元。

加权平均值是关键。当解码器需要将“Je”翻译成“I”时，它给“Je”对应的编码器状态高权重，给其他低权重。当需要翻译“not”时，它给“pas”高权重。上下文向量每一步都在重塑。

## 形状（最容易出问题的地方）

这是每个注意力机制实现第一次都会出错的地方。请慢慢阅读。

|  元素  |  形状  |  备注  |
|-------|-------|-------|
|  编码器隐藏状态`H`  |  `(T_enc, d_h)`  |  如果是BiLSTM，则`d_h = 2 * d_hidden`  |
|  解码器隐藏状态`s_{t-1}`  |  `(d_s,)`  |  一个向量  |
|  注意力分数`e_{t,i}`  |  标量  |  每个编码器位置一个  |
|  注意力权重`α_{t,i}`  |  标量  |  对所有`i`进行softmax后  |
|  上下文向量`c_t`  |  `(d_h,)`  |  与编码器状态形状相同  |

**Bahdanau（加性）分数。** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`。

- `s_{t-1}`的形状为`(d_s,)`，`h_i`的形状为`(d_h,)`。
- `s_{t-1}`的形状为`(d_s,)`。`h_i`的形状为`(d_h,)`。
- 它们在tanh内部的加和的形状为`s_{t-1}`。
- `s_{t-1}`的形状为`(d_s,)`。与`h_i`的内积坍缩为一个标量。**这就是`(d_h,)`的作用。** 这不是魔法，而是将注意力维度向量转化为标量分数的投影。

**Luong（乘法）分数。** 三种变体：

- `dot`：`e_{t,i} = s_t^T * h_i`。要求`d_s == d_h`。严格约束。如果你的编码器是双向的，请跳过。
- `dot`：`e_{t,i} = s_t^T * h_i`，其中`d_s == d_h`的形状为`general`。消除了等维约束。
- `dot`：本质上是Bahdanau形式。由于前两种更高效，很少使用。

**一个值得指出的Bahdanau/Luong陷阱。** Bahdanau使用`s_{t-1}`（生成当前词*之前*的解码器状态）。Luong使用`s_t`（生成当前词*之后*的状态）。混淆它们会产生难以调试的微妙梯度错误。选择一篇论文并坚持其约定。

```figure
attention-heatmap
```

## 动手构建

### 步骤1：加性（Bahdanau）注意力

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

对照上表检查形状。`encoder_states`的形状为`(T_enc, d_h)`。`projected_enc`的形状为`(T_enc, d_attn)`。`projected_dec`的形状为`(d_attn,)`并进行广播。`combined`的形状为`(T_enc, d_attn)`。`scores`的形状为`(T_enc,)`。`weights`的形状为`(T_enc,)`。`context`的形状为`(d_h,)`。搞定。

### 步骤2：Luong点积和通用注意力

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

每个三行代码。这就是Luong论文获得成功的原因。在大多数任务上准确率相同，代码量却少得多。

### 步骤3：一个带数值的示例

给定三个编码器状态（大致对应“cat”、“sat”、“mat”）和一个解码器状态（与第一个最对齐），注意力分布集中在位置0。如果解码器状态偏移到与最后一个对齐，注意力则移动到位置2。上下文向量随之变化。

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

第一行胜出。然后将解码器状态移近第三个编码器状态，观察权重的变化。这就是注意力机制——显式地对齐。

### 步骤4：为什么这是通往Transformer的桥梁

将上述语言转换为Q/K/V：

- **查询(Query)** = 解码器状态 `s_{t-1}`
- **键(Key)** = 编码器状态（我们评分的对象）
- **值(Value)** = 编码器状态（我们加权求和的项）

在经典注意力机制中，键和值是相同的。自注意力(Self-attention)将它们分离：您可以对一个序列自身进行查询，使用不同的学习投影来生成K和V。多头注意力(Multi-head attention)并行运行多个不同的学习投影。变换器(Transformer)将整个过程堆叠多次并舍弃循环神经网络(RNN)。

数学是一样的。形状是一样的。从Bahdanau注意力到缩放点积注意力的教学跳跃主要在于符号表示。

## 使用它

PyTorch和TensorFlow直接内置了注意力机制。

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

那是一个变换器(transformer)的注意力层。查询批次有5个位置，键/值批次有10个位置，每个维度128，8个头。`output` 是新的上下文增强查询。`weights` 是你可以可视化的5x10对齐矩阵。

### 经典注意力何时仍然重要

- 教学用途。单头、单层、基于RNN的版本使每个概念清晰可见。
- 设备上不适用变换器(transformer)的序列任务。
- 2014-2017年的任何论文。如果不了解Bahdanau的惯例，你会误读。
- 机器翻译中的细粒度对齐分析。原始注意力权重即使对于变换器模型也是一种可解释性工具，解读它们需要了解它们是什么。

### 注意力权重作为解释的陷阱

注意力权重看起来是可解释的。它们是在各个位置上总和为1的权重；你可以绘制它们；高值意味着“关注了这里”。评审人喜欢它们。

它们并不像看起来那样可解释。Jain和Wallace (2019) 表明，对于某些任务，注意力分布可以被置换并替换为任意替代，而不改变模型预测。切勿在没有消融或反事实检查的情况下将注意力权重报告为推理的证据。

## 发布

保存为 `outputs/prompt-attention-shapes.md`：

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## 练习

1. **简单。** 实现`softmax`掩码，使编码器中的填充标记的注意力权重为零。在包含变长序列的批次上进行测试。
2. **中等。** 为Luong `softmax`形式添加多头注意力。将`general`分成`d_h`组，对每个头运行注意力，然后拼接。验证单头情况与之前的实现匹配。
3. **困难。** 在第09课的玩具复制任务上训练带有Bahdanau注意力的GRU编码器-解码器。绘制准确率与序列长度的关系图。与无注意力的基线进行比较。你应该能看到随着长度增加差距扩大，确认注意力缓解了瓶颈。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  注意力(Attention)  |  关注事物  |  值序列的加权平均，权重由查询-键相似度计算。 |
|  查询、键、值  |  QKV  |  三个投影：Q提问，K是被匹配的对象，V是要返回的内容。 |
|  加性注意力(Additive attention)  |  Bahdanau  |  前馈分数：`v^T tanh(W q + U k)`。 |
|  乘法注意力(Multiplicative attention)  |  Luong点积/通用  |  分数是`q^T k`或`q^T W k`。更便宜，在大多数任务上精度相同。 |
|  对齐矩阵(Alignment matrix)  |  漂亮的图片  |  注意力权重作为一个`(T_dec, T_enc)`网格。阅读它以查看模型关注了什么。 |

## 延伸阅读

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 论文。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 三种分数变体及其比较。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 可解释性注意事项。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 使用PyTorch的可运行指南。
