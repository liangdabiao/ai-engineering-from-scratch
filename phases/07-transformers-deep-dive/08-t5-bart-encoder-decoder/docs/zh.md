# T5, BART —— 编码器-解码器模型

> 编码器负责理解。解码器负责生成。将它们组合在一起，就得到了一个专为输入→输出任务而构建的模型：翻译、摘要、改写、转录。

**类型:** 学习
**语言:** Python
**前置知识:** 阶段7 · 05（完整Transformer）、阶段7 · 06（BERT）、阶段7 · 07（GPT）
**时间:** ~45分钟

## 问题

仅解码器的GPT和仅编码器的BERT各自为了不同的目标简化了2017年的架构。但许多任务本质上是输入-输出的：

- 翻译：英语 → 法语。
- 摘要：5,000个令牌的文章 → 200个令牌的摘要。
- 语音识别：音频令牌 → 文本令牌。
- 结构化提取：散文 → JSON。

对于这些任务，编码器-解码器是最清晰的匹配。编码器生成源的密集表示（Dense Representation）。解码器生成输出，并在每一步交叉注意力（Cross-Attending）到该表示上。训练是在输出侧进行偏移一位（Shift-by-One）的操作。与GPT相同的损失函数，只是以编码器输出为条件。

两篇论文定义了现代的实践蓝图：

1. **T5** (Raffel et al. 2019)。“文本到文本迁移Transformer”（Text-to-Text Transfer Transformer）。每个NLP任务都被重构为文本输入、文本输出。单一架构、单一词汇表、单一损失函数。预训练采用掩码跨度预测（Masked Span Prediction，在输入中破坏跨度，在输出中解码它们）。
2. **BART** (Lewis et al. 2019)。“双向和自回归Transformer”（Bidirectional and Auto-Regressive Transformer）。去噪自编码器（Denoising Autoencoder）：以多种方式破坏输入（打乱、掩码、删除、旋转），要求解码器重建原始内容。

到2026年，编码器-解码器格式仍然存在于输入结构重要的地方：

- Whisper（语音 → 文本）。
- 谷歌的翻译栈。
- 某些具有不同上下文和编辑结构的代码补全/修复模型。
- Flan-T5及其变体用于结构化推理任务。

仅解码器赢得了聚光灯，但编码器-解码器从未消失。

## 核心概念

![Encoder-decoder with cross-attention](../assets/encoder-decoder.svg)

### 前向循环

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

关键的是，编码器每个输入只运行一次。解码器自回归运行，但在每一步交叉注意力到相同的编码器输出。缓存编码器输出对于长输入来说是一个免费的加速。

### T5预训练 —— 跨度破坏（Span Corruption）

从输入中选取随机跨度（平均长度3个令牌，占15%）。用唯一哨兵（Sentinel）替换每个跨度：`<extra_id_0>`，`<extra_id_1>`等。解码器只输出带有其哨兵前缀的破坏跨度：

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

比预测整个序列更廉价的信号。在T5论文的消融实验中，与MLM（BERT）和前缀LM（UniLM）相比具有竞争力。

### BART预训练 —— 多噪声去噪

BART尝试了五种噪声函数：

1. 令牌掩码（Token Masking）。
2. 令牌删除（Token Deletion）。
3. 文本填充（Text Infilling，掩码一个跨度，解码器插入正确长度）。
4. 句子排列（Sentence Permutation）。
5. 文档旋转（Document Rotation）。

文本填充 + 句子排列的组合产生了最好的下游指标。解码器始终重建原始序列。BART的输出是完整序列，而不仅仅是破坏跨度——因此预训练计算量高于T5。

### 推理

与GPT相同的自回归生成。贪婪/束搜索/Top-p采样适用。束搜索（宽度4-5）是翻译和摘要的标准，因为输出分布比聊天更窄。

### 2026年如何选择每种变体

|  任务  |  编码器-解码器？  |  原因  |
|------|------------------|-----|
|  翻译  |  通常使用  |  清晰的源序列；固定的输出分布；束搜索有效  |
|  语音转文本  |  使用（Whisper）  |  输入模态与输出不同；编码器塑造音频特征  |
|  聊天/推理  |  不使用，仅解码器  |  没有持久的“输入”——对话本身就是序列  |
|  代码补全  |  通常不使用  |  仅解码器配合长上下文胜出；像Qwen 2.5 Coder这样的代码模型是仅解码器  |
|  摘要  |  两者皆可  |  BART、PEGASUS超越了早期的仅解码器基线；现代仅解码器LLM能与之匹敌  |
|  结构化提取  |  两者皆可  |  T5很简洁，因为“文本→文本”可以吸收任何输出格式  |

自2022年左右的趋势：解码器专用模型接管了编码器-解码器模型曾经负责的任务，因为(a) 经过指令微调的解码器专用LLM通过提示可以泛化到任何任务，(b) 单一架构比双架构更容易扩展，(c) RLHF假设使用解码器。编码器-解码器模型在输入模态不同（如语音、图像）或束搜索质量至关重要时仍保持优势。

## 动手构建

参见`code/main.py`。我们为一个玩具语料库实现T5风格的跨度损坏——本课中最有用的部分，因为它出现在此后每一个编码器-解码器预训练方案中。

### 第1步：跨度损坏

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

目标格式是T5惯例：`<sent0> span0 <sent1> span1 ...`。损坏后的输入将未更改的令牌与跨度位置处的哨兵令牌交错排列。

### 第2步：验证往返

给定损坏后的输入和目标，重建原始句子。如果损坏是可逆的，前向传播就是良定义的。这是一个完整性检查——实际训练从不这样做，但该测试代价低且能捕获跨度账本中的差一错误。

### 第3步：BART加噪

五个函数：`token_mask`，`token_delete`，`text_infill`，`sentence_permute`，`document_rotate`。组合其中两个函数并展示结果。

## 使用它

HuggingFace参考：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5的技巧：任务名称被放入输入文本中。同一个模型处理数十个任务，因为每个任务都是文本输入、文本输出。在2026年，这一模式已被指令微调的解码器专用模型泛化，但T5首先将其规范化。

## 发布

参见`outputs/skill-seq2seq-picker.md`。该技巧根据输入输出结构、延迟和质量目标，为新任务在编码器-解码器和解码器专用模型之间做出选择。

## 练习

1. **简单。** 运行`code/main.py`，对一个30令牌的句子应用跨度损坏，验证将非哨兵源令牌与解码后的目标跨度连接起来能否复现原句。
2. **中等。** 实现BART的`code/main.py`噪声：用一个`text_infill`令牌替换随机跨度，解码器必须推断出正确的跨度长度和内容。展示一个示例。
3. **困难。** 在小型英语→猪拉丁语语料库（200对）上微调`code/main.py`。在保留的50对集上测量BLEU。与在相同数据和相同计算量下微调`text_infill`进行比较。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
| 编码器-解码器  |  "序列到序列Transformer"  |  两个堆栈：用于输入的双向编码器，用于输出的带交叉注意力的因果解码器。 |
| 交叉注意力  |  "源与目标对话之处"  |  解码器的Q × 编码器的K/V。编码器信息进入解码器的唯一位置。 |
| 跨度损坏  |  "T5的预训练技巧"  |  用哨兵令牌替换随机跨度；解码器输出跨度。 |
| 去噪目标  |  "BART的游戏"  |  对输入应用噪声函数，训练解码器重建干净序列。 |
| 哨兵令牌  |  "`<extra_id_N>`占位符"  |  在源中标记损坏跨度并在目标中重新标记它们的特殊令牌。 |
| Flan  |  "指令微调的T5"  |  在超过1，800个任务上微调的T5；使编码器-解码器在指令遵循方面具有竞争力。 |
| 束搜索  |  "解码策略"  |  每一步保留前k个部分序列；翻译/摘要的标准方法。 |
| 教师强制  |  "训练时输入"  |  训练时，向解码器馈送真实的前一个输出令牌，而不是采样的令牌。 |

## 延伸阅读

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5。
- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — BART。
- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — Flan-T5。
- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — Whisper，2026年规范的编码器-解码器。
- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — 参考实现。
