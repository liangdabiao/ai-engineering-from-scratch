# 完整Transformer——编码器+解码器

> 注意力机制是核心。其他一切——残差连接、归一化、前馈网络、交叉注意力——都是让你能够堆叠深度的脚手架。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段7·02（自注意力 Self-Attention），阶段7·03（多头注意力 Multi-Head Attention），阶段7·04（位置编码 Positional Encoding）
**时间：** 约75分钟

## 问题

单层注意力是特征提取器，不是模型。每层仅一次矩阵乘法不足以处理语言。你需要深度——而没有正确的管道，深度就会崩溃。

2017年Vaswani的论文打包了六项设计决策，将单层注意力变成了可堆叠的模块。此后的每个Transformer——编码器专用（BERT）、解码器专用（GPT）、编码器-解码器（T5）——都继承了相同的骨架。到了2026年，模块已经优化（RMSNorm、SwiGLU、前归一化、RoPE），但骨架相同。

本课介绍骨架。后续课程进行细化——第06课讲编码器，第07课讲解码器，第08课讲编码器-解码器。

## 核心概念

![Encoder and decoder block internals, wired](../assets/full-transformer.svg)

### 六个组成部分

1. **嵌入+位置信号。** 词元→向量。通过RoPE（现代）或正弦波（经典）注入位置。
2. **自注意力。** 每个位置关注所有其他位置。解码器中带掩码。
3. **前馈网络（FFN）。** 逐位置的两层MLP：`W_2 · activation(W_1 · x)`。默认扩展比4倍。
4. **残差连接。** `W_2 · activation(W_1 · x)`。没有它，梯度在大约6层后消失。
5. **层归一化。** `W_2 · activation(W_1 · x)`或`x + sublayer(x)`（现代）。稳定残差流。
6. **交叉注意力（仅解码器）。** 查询来自解码器，键和值来自编码器输出。

观察向量流经一个模块：注意力跨位置混合，残差将其向前传递，FFN对其进行变换，归一化保持流稳定。

```figure
transformer-block
```

### 编码器模块（用于BERT、T5编码器）

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

编码器是双向的。无掩码。所有位置都能看到所有位置。

### 解码器模块（用于GPT、T5解码器）

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

解码器每个模块有三个子层。中间层——交叉注意力——是信息从编码器流向解码器的唯一位置。在纯解码器专用架构（GPT）中，交叉注意力被省略，只有掩码自注意力+FFN。

### 前归一化 vs 后归一化

原始论文：`x + sublayer(LN(x))` vs `LN(x + sublayer(x))`。后归一化在2019年左右失宠——没有仔细的预热很难训练深层网络。前归一化（`LN` *在子层之前*）是2026年的默认配置：Llama、Qwen、GPT-3+、Mistral都使用它。

### 2026年现代化模块

Vaswani 2017使用了LayerNorm+ReLU。现代堆叠替换了二者。生产环境中的模块实际样子：

|  组件  |  2017  |  2026  |
|-----------|------|------|
|  归一化  |  LayerNorm  |  RMSNorm  |
|  FFN激活函数  |  ReLU  |  SwiGLU  |
|  FFN扩展比  |  4×  |  2.6× (SwiGLU使用三个矩阵，总参数匹配)  |
|  位置编码  |  正弦绝对位置  |  RoPE  |
|  注意力  |  全MHA  |  GQA (或MLA)  |
|  偏置项  |  有  |  无  |

RMSNorm去掉了LayerNorm的均值中心化（少一次减法），节省了计算量，且经验上至少同样稳定。SwiGLU（`Swish(W1 x) ⊙ W3 x`）在Llama、PaLM和Qwen论文中一致优于ReLU/GELU FFN约0.5个点的困惑度。

### 参数数量

对于一个模块，输入维度`d_model = d`且FFN扩展比`r`：

- MHA: `4 · d²` (Q、K、V、O投影)
- FFN (SwiGLU): `4 · d²` ≈ `3 · d · (r · d)`
- 归一化: 可忽略

当`d = 4096, r = 2.6, layers = 32`时（约Llama 3 8B），总参数：`32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B`（加上嵌入和输出头）。与已发布的数量一致。

## 动手构建

### 步骤1：构建模块

使用第3课中的小型`Matrix`类（为独立起见，已复制到本文件）：

- `layer_norm(x, eps=1e-5)` — 减去均值，除以标准差。
- `layer_norm(x, eps=1e-5)` — 除以RMS。不减去均值。
- `layer_norm(x, eps=1e-5)`和`rms_norm(x, eps=1e-6)` (SwiGLU)。
- `layer_norm(x, eps=1e-5)`。
- `layer_norm(x, eps=1e-5)`和`rms_norm(x, eps=1e-6)`。

完整接线请参见`code/main.py`。

### 步骤2：连接一个2层编码器和一个2层解码器

将它们堆叠。将编码器输出传递给每个解码器交叉注意力。在输出投影前添加最终的LN。

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### 步骤3：在小示例上运行前向传播

输入6个词的源代码和5个词的目标代码。确认输出形状为`(5, vocab)`。不进行训练——本课关注的是架构，而非损失。

### 步骤4：替换为RMSNorm + SwiGLU

用RMSNorm和SwiGLU替换LayerNorm和ReLU-FFN。确认形状仍然匹配。这是通过一次函数替换实现的2026年现代化。

## 使用它

PyTorch/TF参考实现：`nn.TransformerEncoderLayer`，`nn.TransformerDecoderLayer`。但大多数2026年生产代码都自行编写block，因为：

- Flash Attention在注意力内部调用，而不是通过`nn.MultiheadAttention`。
- GQA/MLA不在标准库参考中。
- RoPE、RMSNorm、SwiGLU不是PyTorch默认值。

HF `transformers`有清晰的参考块，你应该阅读：`modeling_llama.py`是标准的2026年仅解码器块。它约有500行，值得仔细过一遍。

**编码器 vs 解码器 vs 编码器-解码器 — 如何选择：**

|  需求  |  选择  |  示例  |
|------|------|---------|
|  分类、嵌入、文本问答  |  仅编码器  |  BERT, DeBERTa, ModernBERT  |
|  文本生成、聊天、代码、推理  |  仅解码器  |  GPT, Llama, Claude, Qwen  |
|  结构化输入 → 结构化输出（翻译、摘要）  |  编码器-解码器  |  T5, BART, Whisper  |

仅解码器在语言任务中胜出，因为它的扩展最干净，并且同时处理理解和生成。当输入具有明确的“源序列”身份（翻译、语音识别、结构化任务）时，编码器-解码器仍然是最佳选择。

## 发布

请参见`outputs/skill-transformer-block-reviewer.md`。该技能检查新的Transformer块实现是否符合2026年默认设置，并标记缺失的部分（前归一化、RoPE、RMSNorm、GQA、FFN扩展比例）。

## 练习

1. **简单。** 统计`d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`处encoder_block中的参数。通过实现该块并使用`sum(p.numel() for p in block.parameters())`进行验证。
2. **中等。** 从后归一化切换到前归一化。初始化两者，并在随机输入上测量12个堆叠层后的激活范数。后归一化的激活应该爆炸；前归一化的应该保持有界。
3. **困难。** 在小规模复制任务（复制颠倒的`d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`）上实现4层编码器-解码器。训练100步。报告损失。替换为RMSNorm + SwiGLU + RoPE — 损失是否下降？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  Block  |  “一个Transformer层”  |  归一化 + 注意力 + 归一化 + FFN的堆叠，包裹在残差连接中。  |
|  残差  |  “跳跃连接”  |  `x + f(x)`输出；使得梯度能够流经深层堆叠。  |
|  前归一化  |  “先归一化，而非后归一化”  |  现代：`x + sublayer(LN(x))`。无需预热技巧即可训练更深网络。  |
|  RMSNorm  |  “没有均值的LayerNorm”  |  除以RMS；少一次运算，经验稳定性相同。  |
|  SwiGLU  |  “人人都切换到的FFN”  |  `Swish(W1 x) ⊙ W3 x → W2`。在LM困惑度上击败ReLU/GELU。  |
|  交叉注意力  |  “解码器如何看到编码器”  |  MHA，Q来自解码器，K/V来自编码器输出。  |
|  FFN扩展  |  “中间MLP有多宽”  |  隐藏大小与d_model的比率，通常为4（LayerNorm）或2.6（SwiGLU）。  |
|  无偏置  |  “去掉+b项”  |  现代堆叠省略线性层中的偏置；轻微的困惑度改善，模型更小。  |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始block规范。
- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 为什么前归一化在深层优于后归一化。
- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — RMSNorm。
- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — SwiGLU论文。
- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 标准的2026年仅解码器块。
