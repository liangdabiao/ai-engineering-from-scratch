# 序列到序列模型

> 两个RNN假装成翻译器。它们遇到的瓶颈正是注意力机制存在的原因。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段5·08（用于文本的CNN+RNN）、阶段3·11（PyTorch入门）
**时间：** 约75分钟

## 问题

分类将变长序列映射为单个标签。翻译将变长序列映射为另一个变长序列。输入和输出处于不同的词汇表中，可能是不同语言，且长度不一定对等。

seq2seq架构（Sutskever, Vinyals, Le, 2014）用一个刻意简单的配方解决了这个问题。两个RNN。一个读取源句子并生成固定大小的上下文向量。另一个读取该向量并逐词生成目标句子。和你为第8课编写的代码相同，但以不同方式组合在一起。

这值得学习有两个原因。首先，上下文向量瓶颈是NLP中最具教学价值的失败。它促使了注意力机制和Transformer所擅长的一切。其次，训练方法（教师强制、计划采样、推理时的束搜索）仍然适用于包括LLM在内的每一个现代生成系统。

## 核心概念

**编码器。** 一个读取源句子的RNN。其最终隐藏状态是**上下文向量**——整个输入的固定大小摘要。据说除了源句子外什么都不丢失。

**解码器。** 另一个从上下文向量初始化的RNN。在每一步，它把先前生成的词元作为输入，并产生目标词汇表上的分布。采样或argmax来选择下一个词元。将其反馈回去。重复直到生成`<EOS>`词元或达到最大长度。

**训练：** 每个解码器步骤的交叉熵损失，对序列求和。通过时间反向传播的标准算法同时作用于两个网络。

**教师强制。** 在训练期间，解码器在步骤`t`的输入是位置`t-1`的*真实*词元，而不是解码器自身的先前预测。这稳定了训练；没有它，早期错误会级联放大，模型永远学不会。在推理时，你必须使用模型自身的预测，因此总存在训练/推理分布差距。这个差距称为**曝光偏差**。

**瓶颈。** 编码器关于源句子学到的一切都必须压缩进那一个上下文向量中。长句子丢失细节。罕见词变得模糊。词序重排（chat noir vs. black cat）只能被记忆而非计算。

注意力机制（第10课）通过让解码器查看*每一个*编码器隐藏状态（而不仅仅是最后一个）来解决这个问题。这就是全部要点。

```figure
lstm-gates
```

## 动手构建

### 步骤1：编码器

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs`的形状为`[batch, seq_len, hidden_dim]`——每个输入位置一个隐藏状态。`hidden`的形状为`[1, batch, hidden_dim]`——最后一步。第8课说“对输出进行池化以进行分类”。这里我们将最后一个隐藏状态保留为上下文向量，并忽略每步输出。

### 步骤2：解码器

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

解码器一次调用一步。输入：一批单个词元和当前隐藏状态。输出：下一个词元的词汇logits和更新后的隐藏状态。

### 步骤3：带教师强制训练的训练循环

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

两个值得命名的旋钮。`ignore_index=0`跳过填充词元上的损失。`teacher_forcing_ratio`是在每一步使用真实词元相对于模型预测的概率。从1.0（完全教师强制）开始，在训练过程中退火至约0.5，以缩小曝光偏差差距。

### 步骤4：推理循环（贪婪）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

贪婪解码在每一步选择概率最高的词元。它可能走偏：一旦你选定一个词元，就无法撤回。**束搜索**保持前`k`个部分序列存活，并在最后选择得分最高的完整序列。束宽3-5是标准。

### 步骤5：瓶颈演示

在玩具复制任务上训练模型：源`[a, b, c, d, e]`，目标`[a, b, c, d, e]`。增加序列长度。观察准确率。

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

单个GRU隐藏状态无法无损记忆40个词元的输入。信息存在于每个编码器步骤，但解码器只看到最后一个状态。注意力机制直接修复了这一点。

## 使用它

PyTorch有基于`nn.Transformer`和`nn.LSTM`的seq2seq模板。Hugging Face的`transformers`库提供了在数十亿词元上训练的完整编码器-解码器模型（BART、T5、mBART、NLLB）。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

现代编码器-解码器放弃了RNN，改用Transformer。高级结构（编码器、解码器、逐词生成）与2014年seq2seq论文完全相同。每个模块内部的机制不同。

### 何时仍应使用基于RNN的seq2seq

对于新项目，几乎从不。具体例外情况：

- 流式翻译，其中你一次消费一个词元，内存有界。
- 设备上文本生成，其中Transformer内存成本过高。
- 教学。理解编码器-解码器瓶颈是理解Transformer为何获胜的最快途径。

### 曝光偏差及其缓解措施

- **计划采样。** 在训练期间退火教师强制比例，使模型学会从自身错误中恢复。
- **最小风险训练。** 在句子级BLEU分数上训练，而不是词元级交叉熵。更接近你真正想要的。
- **强化学习微调。** 用某个指标奖励序列生成器。用于现代LLM的RLHF。

这三种方法仍然适用于基于Transformer的生成。

## 发布

保存为 `outputs/prompt-seq2seq-design.md`：

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## 练习

1. **简单（Easy）。** 实现玩具复制任务。在目标等于源的输入输出对上训练GRU序列到序列模型（GRU seq2seq）。测量长度为5、10、20时的准确率。重现瓶颈。
2. **中等（Medium）。** 添加束宽为3的束搜索解码。在小平行语料库上衡量与贪心解码相比的BLEU值。记录束搜索在哪些情况下表现更好（通常是最后几个令牌）以及哪些情况无差别。
3. **困难（Hard）。** 在1万对释义数据集上微调`facebook/bart-base`。将微调模型的束宽为4的输出与基础模型在留出输入上的输出进行比较。报告BLEU并挑选10个定性示例。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  编码器（Encoder） | 输入RNN（Input RNN） | 读取源序列。生成每个时间步的隐藏状态和最终的上下文向量（context vector）。 |
|  解码器（Decoder） | 输出RNN（Output RNN） | 从上下文向量初始化，逐个生成目标令牌。 |
|  上下文向量（Context vector） | 摘要 | 编码器最终隐藏状态。固定大小。这是注意力机制（Attention）所解决的瓶颈。 |
|  教师强制（Teacher forcing） | 使用真实令牌 | 在训练时输入真实的前一个令牌。稳定学习过程。 |
|  暴露偏差（Exposure bias） | 训练/测试差距 | 在真实令牌上训练的模型从未练习过从自身错误中恢复。 |
|  束搜索（Beam search） | 更好的解码 | 每一步保留top-k个部分序列，而不是贪心地确定。 |

## 延伸阅读

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — 最初的序列到序列论文。共四页。
- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — 引入了GRU和编码器-解码器框架。
- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — 注意力机制论文。课后立即阅读。
- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — 可构建的序列到序列+注意力代码。
