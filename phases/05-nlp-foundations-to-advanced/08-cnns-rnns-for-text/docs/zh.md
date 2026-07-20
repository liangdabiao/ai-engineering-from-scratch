# 用于文本的CNN和RNN

> 卷积学习n-gram。循环负责记忆。两者都被注意力取代。但在受限硬件上两者仍然重要。

**类型：** 构建
**语言：** Python
**先修知识：** 阶段3·11（PyTorch入门）、阶段5·03（词嵌入）、阶段4·02（从零开始的卷积）
**时间：** 约75分钟

## 问题

TF-IDF和Word2Vec生成忽略词序的扁平向量。基于它们的分类器无法区分`dog bites man`和`man bites dog`。词序有时携带信号。

在Transformer出现之前，两类架构填补了这一空白。

**用于文本的卷积网络（TextCNN）。** 对词嵌入序列应用一维卷积。宽度为3的滤波器是一个可学习的三元组检测器：它跨越三个词并输出一个分数。堆叠不同宽度（2、3、4、5）以检测多尺度模式。最大池化到固定大小的表示。扁平、并行、快速。

**循环网络（RNN、LSTM、GRU）。** 一次处理一个词元，维护一个携带信息的隐藏状态向前传递。顺序、有记忆、输入长度灵活。从2014年到2017年主导序列建模，然后注意力出现了。

本课构建两者，然后指出促使注意力产生的失败之处。

## 核心概念

**TextCNN**（Kim，2014）。对词元进行嵌入。一个宽度为`k`的一维卷积在嵌入的连续`k`-gram上滑动滤波器，生成特征图。对该图进行全局最大池化，选取最强激活。将多个滤波器宽度的最大池化输出拼接起来。送入分类器头部。

为什么有效。一个滤波器是一个可学习的n-gram。最大池化是位置不变的，因此"not good"在评论开头或中间会触发相同的特征。三个滤波器宽度各100个滤波器，给你300个学习的n-gram检测器。训练是并行的，没有顺序依赖。

**RNN。** 在每个时间步`t`，隐藏状态`h_t = f(W * x_t + U * h_{t-1} + b)`。在时间上共享`W`、`U`、`b`。时间`T`的隐藏状态是整个前缀的摘要。对于分类，跨`h_1 ... h_T`进行池化（最大、平均或最后）。

普通RNN存在梯度消失问题。**LSTM**增加了门控，决定遗忘什么、存储什么和输出什么，通过长序列稳定梯度。**GRU**将LSTM简化为两个门；参数更少，性能相似。

**双向RNN**运行一个前向RNN和一个后向RNN，拼接隐藏状态。每个词元的表示都能看到左右上下文。对于标注任务至关重要。

```figure
rnn-unroll
```

## 动手构建

### 第1步：PyTorch中的TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)`将`[batch, seq_len, embed_dim]`重塑为`[batch, embed_dim, seq_len]`，因为`nn.Conv1d`将中间轴视为通道。池化输出是固定大小的，与输入长度无关。

### 第2步：LSTM分类器

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

对序列进行最大池化，而不是最后状态池化。对于分类，最大池化通常优于取最后一个隐藏状态，因为长序列末尾的信息往往主导最后状态。

### 第3步：梯度消失演示（直觉）

没有门控的普通RNN无法学习长期依赖。考虑一个玩具任务：预测词元`A`是否出现在序列中的任何位置。如果`A`在位置1且序列长度为100个词元，则来自损失的梯度必须通过99次循环权重乘法回流。如果权重小于1，梯度消失。如果大于1，梯度爆炸。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTM通过**细胞状态**解决了这个问题，该状态仅通过加性交互贯穿网络（遗忘门对其进行乘法缩放，但梯度仍沿着“高速公路”流动）。GRU用更少的参数做了类似的事情。两者都能在100多步的序列上实现稳定训练。

### 第4步：为什么这仍然不够

即使使用LSTM，三个问题仍然存在。

1. **顺序瓶颈。** 在长度为1000的序列上训练RNN需要1000个串行的前向/后向步骤。无法跨时间并行化。
2. **编码器-解码器设置中的固定大小上下文向量。** 解码器仅看到编码器的最终隐藏状态，该状态压缩了整个输入。长输入会丢失细节。第09课将直接讨论这一点。
3. **远距离依赖准确率上限。** LSTM优于普通RNN，但仍然难以在200多步中传播特定信息。

注意力解决了所有三个问题。Transformer完全摒弃了循环。第10课是转折点。

## 使用它

PyTorch的`nn.LSTM`、`nn.GRU`和`nn.Conv1d`已可用于生产。训练代码是标准的。

Hugging Face提供预训练嵌入，您可以直接将其插入作为输入层：

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

根据约束使用的检查清单。

- **边缘/设备端推理。** 使用GloVe嵌入的TextCNN比Transformer小10-100倍。如果您的部署目标是手机，这就是合适的方案。
- **流式/在线分类。** RNN一次处理一个词元；Transformer需要完整的序列。对于实时传入的文本，LSTM仍然胜出。
- **用于基线的小型模型。** 在新任务上快速迭代。在CPU上5分钟内训练一个TextCNN。
- **有限数据下的序列标注。** BiLSTM-CRF（第06课）仍然是1000-10000条标注句子的生产级NER架构。

其他所有情况都使用Transformer。

## 发布

保存为 `outputs/prompt-text-encoder-picker.md`：

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## 练习

1. **简单。** 在一个3类玩具数据集（自己创建数据）上训练TextCNN。验证滤波器宽度（2、3、4）在平均F1上优于单个宽度（3）。
2. **中等。** 为LSTM分类器实现最大池化、平均池化和最后状态池化。在小数据集上比较，记录哪种池化胜出并假设原因。
3. **困难。** 构建一个BiLSTM-CRF NER标注器（结合第06课和本课）。在CoNLL-2003上训练。与第06课中仅CRF的基线以及BERT微调进行比较。报告训练时间、内存和F1。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  TextCNN  |  文本卷积神经网络  |  在词嵌入上堆叠一维卷积，并采用全局最大池化。Kim（2014）。 |
|  RNN  |  循环网络  |  隐藏状态在每个时间步更新：`h_t = f(W x_t + U h_{t-1})`。 |
|  LSTM  |  门控循环神经网络  |  增加输入门、遗忘门、输出门以及细胞状态，通过长序列能够稳定训练。 |
|  GRU  |  简化的LSTM  |  两个门代替三个门，精度相似，参数更少。 |
|  Bidirectional  |  两个方向  |  前向和后向RNN拼接，每个词元都能看到其上下文的两侧。 |
|  Vanishing gradient  |  训练信号衰减  |  在标准RNN中，反复乘以<1的权重使得早期步长的梯度实际为零。 |

## 延伸阅读

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — TextCNN论文。八页。可读性强。
- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — LSTM论文。异常清晰。
- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — 使LSTM为大众所理解的图表。
