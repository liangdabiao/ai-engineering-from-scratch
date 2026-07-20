# 位置编码 — 正弦型(Sinusoidal)、RoPE、ALiBi

> 注意力机制具有置换不变性。没有位置信号时，“The cat sat on the mat”和“mat the on sat cat the”会产生相同的输出。三种算法解决了这一问题——它们对“位置”的含义各有不同的假设。

**类型：** 实践
**语言：** Python
**前置知识：** 阶段7·02（自注意力）、阶段7·03（多头注意力）
**时长：** 约45分钟

## 问题

缩放点积注意力对顺序不敏感。注意力矩阵`softmax(Q K^T / √d) V`由两两相似度计算得出。打乱`X`的行，输出的行也会以相同方式被打乱。注意力内部没有任何机制关心位置。

在词袋模型中这并非缺陷。但对于语言、代码、音频、视频——任何顺序承载意义的领域——这是致命的。

解决方法是以某种方式将位置注入到嵌入中。三个时代的答案如下：

1. **绝对正弦编码**（Vaswani 2017）。将位置的`sin/cos`加到嵌入上。简单、无需学习，但在超出训练长度时外推能力差。
2. **RoPE——旋转位置编码**（Su 2021）。将Q和K向量旋转与位置成比例的角度。直接在点积中编码*相对*位置。到2026年成为主流。
3. **ALiBi——线性偏置注意力**（Press 2022）。完全跳过嵌入；根据距离向注意力分数添加每头线性惩罚。长度外推能力优秀。

截至2026年，几乎所有前沿开源模型都使用RoPE：Llama 2/3/4、Qwen 2/3、Mistral、Mixtral、DeepSeek-V3、Kimi。少数长上下文模型使用ALiBi或其现代变体。绝对正弦编码已成为历史。

## 核心概念

![Sinusoidal absolute vs RoPE rotations vs ALiBi distance bias](../assets/positional-encoding.svg)

### 绝对正弦编码

预计算一个形状为`(max_len, d_model)`的固定矩阵`PE`：

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

然后在注意力之前执行`X' = X + PE[:N]`。每个维度是不同频率的正弦波。模型学会从相位模式中读取位置。在超出`max_len`时失败：当模型只见过位置0–2047时，没有任何信息告诉它在位置2048处会怎样。

### RoPE

旋转Q和K向量（而非嵌入）。对于一对维度`(2i, 2i+1)`：

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

对位置为`pos_k`的键应用相同的旋转。点积`q'_m · k'_n`成为仅关于`(m - n)`的函数。也就是说：**注意力分数仅依赖于相对距离**，尽管旋转是基于绝对位置进行的。精妙的技巧。

扩展RoPE：`base`可以被缩放（NTK-aware、YaRN、LongRoPE）以在不重新训练的情况下外推到更长的上下文。Llama 3通过这种方式从8K上下文扩展到了128K。

### ALiBi

跳过嵌入技巧。直接对注意力分数加偏置：

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

其中`m_h`是特定于头的斜率（例如`1 / 2^(8·h/H)`）。更近的标记得到增强，更远的标记受到惩罚。没有训练时间开销。论文表明长度外推能力优于正弦编码，并在其原始训练长度上与RoPE相当。

### 2026年如何选择

|  变体  |  外推能力  |  训练成本  |  使用者  |
|---------|---------------|---------------|---------|
|  绝对正弦编码  |  差  |  零成本  |  原始Transformer、早期BERT  |
|  可学习绝对编码  |  无  |  微小  |  GPT-2、GPT-3  |
|  RoPE  |  通过缩放表现良好  |  零成本  |  Llama 2/3/4、Qwen 2/3、Mistral、DeepSeek-V3、Kimi  |
|  RoPE + YaRN  |  优秀  |  微调阶段  |  Qwen2-1M、Llama 3.1 128K  |
|  ALiBi  |  优秀  |  零成本  |  BLOOM、MPT、百川  |

RoPE胜出的原因是它可以无缝嵌入注意力机制而不改变架构，编码了相对位置，并且其`base`超参数为长上下文微调提供了一个简洁的调节旋钮。

```figure
rope-explorer
```

## 动手构建

### 第一步：正弦编码

参见`code/main.py`。一个4行的计算：

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

在第一个注意力层之前将其加到嵌入矩阵上。

### 第二步：对Q、K应用RoPE

RoPE 对 Q 和 K 进行原地操作。对于每一对维度：

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

关键：对位置 `m` 的 Q 和位置 `n` 的 K 应用相同的函数。它们的点积在每个坐标对上获得一个 `cos((m-n)·θ_i)` 因子。注意力机制无需额外代价即可学习相对位置。

### 步骤 3：ALiBi 斜率和偏置

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

将 `bias[h]` 加到头部 `h` 的 `(seq_len, seq_len)` 注意力分数矩阵上，然后进行 softmax。

### 步骤 4：验证 RoPE 的相对距离特性

选取两个随机向量 `a, b`。旋转 `(pos_a, pos_b)`。然后再旋转 `(pos_a + k, pos_b + k)`。两个点积必须在浮点误差范围内匹配。这一性质正是 RoPE 的核心所在——它对绝对偏移不敏感，只有相对间隔重要。

## 使用它

PyTorch 2.5+ 在 `torch.nn.functional` 中内置了 RoPE 工具函数。大多数生产代码使用 `flash_attn` 或 `xformers`，在注意力内核内部应用 RoPE。

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026 年的长上下文技巧：**

- **NTK-aware 插值。** 当从 4K 扩展到 16K+ 时，将 `base` 重新缩放到 `base * (scale_factor)^(d/(d-2))`。
- **YaRN。** 更智能的插值方法，在长上下文中保持注意力熵。Llama 3.1 128K 使用了它。
- **LongRoPE。** 微软 2024 年提出的方法，使用进化搜索为每个维度选择缩放因子。Phi-3-Long 使用了它。
- **位置插值 + 微调。** 只需按扩展因子缩小位置，并在 1–5B 个 token 上进行微调。效果出奇地好。

## 发布

参见 `outputs/skill-positional-encoding-picker.md`。该技能根据目标上下文长度、外推需求和训练预算，为新模型选择编码策略。

## 练习

1. **简单。** 将正弦 `PE` 矩阵绘制为 `max_len=512, d=128` 的热力图。确认"条纹随维度索引增加而变宽"的模式。
2. **中等。** 实现 NTK-aware RoPE 缩放。在长度为 256 的序列上训练一个小型语言模型，然后在长度为 1024 的序列上分别测试有缩放和无缩放的情况。测量困惑度。
3. **困难。** 在同一注意力模块中实现 ALiBi 和 RoPE。在长度为 512 的序列上训练一个 4 层 Transformer 执行复制任务。测试时外推到 2048。比较性能下降程度。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  位置编码(Positional Encoding)  |  "告诉注意力关于顺序"  |  添加到嵌入或注意力中用于编码位置的任何信号。  |
|  正弦(Sinusoidal)  |  "最初的版本"  |  以几何频率的 `sin/cos` 添加到嵌入中；不能外推。  |
|  RoPE  |  "旋转嵌入"  |  根据位置相关角度旋转 Q 和 K；点积编码相对距离。  |
|  ALiBi  |  "线性偏置技巧"  |  将 `-m·\ | i-j\ | ` 加到注意力分数上；无需嵌入，外推能力强。  |
|  base  |  "RoPE 的旋钮"  |  RoPE 中的频率缩放器；增加它以在推理时扩展上下文。  |
|  NTK-aware  |  "一种 RoPE 缩放技巧"  |  重新缩放 `base`，使得上下文扩展时高频维度不被压缩。  |
|  YaRN  |  "花哨的那个"  |  逐维度插值+外推，保持注意力熵。  |
|  外推(Extrapolation)  |  "在训练长度之外工作"  |  位置方案能否在训练中看到的 `max_len` 之外提供正确的输出？  |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — 原始正弦。
- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — RoPE 论文。
- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — ALiBi。
- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — 最先进的 RoPE 缩放。
- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — Meta 的 Llama 2 长上下文论文。
- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — Phi-3-Long 使用并在"使用它"部分引用的微软方法。
- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — 每种 RoPE 缩放方案的生产级实现（默认、线性、动态、YaRN、LongRoPE、Llama-3）。
