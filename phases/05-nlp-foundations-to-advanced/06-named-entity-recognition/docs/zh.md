# 命名实体识别

> 提取名称。听起来容易，直到你处理模糊边界、嵌套实体和领域术语。

**类型：** 构建
**语言：** Python
**先决条件：** 阶段5 · 02（BoW + TF-IDF），阶段5 · 03（词嵌入）
**时间：** 约75分钟

## 问题

"苹果在美国就iPhone搜索交易起诉谷歌。"五个实体：Apple（组织）、Google（组织）、iPhone（产品）、search deal（可能是）、US（地缘政治实体）。一个好的NER系统能提取所有实体并正确分类。一个差的系统会遗漏iPhone，混淆苹果（水果）和苹果公司，并将“US”标记为人物。

NER是每个结构化提取管道背后的主力。简历解析、合规日志扫描、医疗记录匿名化、搜索查询理解、聊天机器人响应的接地、法律合同提取。你很少直接看到它，但总是依赖它。

本课沿着经典路径（基于规则、隐马尔可夫模型、条件随机场）走向现代路径（BiLSTM-CRF，然后是Transformer）。每一步都解决了前一步的特定限制。模式本身就是课程。

## 核心概念

**BIO标注**（或BILOU）将实体提取转化为序列标注问题。使用`B-TYPE`（实体开头）、`I-TYPE`（实体内部）或`O`（实体外部）标记每个词元。

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

多词元实体链：`New B-GPE`, `York I-GPE`, `City I-GPE`。理解BIO的模型可以提取任意跨度。

架构演变：

- **基于规则。** 正则表达式 + 地名词典查找。对已知实体高精度，对新实体零覆盖。
- **隐马尔可夫模型。** 发射概率：给定标签下词元的概率；转移概率：标签到标签的概率。维特比解码。在标注数据上训练。
- **条件随机场。** 与HMM类似但更判别式，因此可以混合任意特征（词形、大写、相邻词）。在2026年仍是低资源部署的经典生产主力。
- **BiLSTM-CRF。** 神经特征代替手工特征。LSTM双向读取句子，顶部的CRF层强制一致标签序列。
- **基于Transformer。** 使用token分类头微调BERT。准确率最高。算力需求最大。

```figure
ner-bio-tagging
```

## 动手构建

### 步骤1：BIO标注辅助工具

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### 步骤2：手工特征

对于经典（非神经）NER，特征至关重要。有用的特征包括：

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` 返回 `xXxxxx`。`word_shape("USA-2024")` 返回 `XXX-dddd`。大写模式对专有名词是强信号。

### 步骤3：一个简单的基于规则+词典基线

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

生产环境的地名词典有数百万个从维基百科和DBpedia抓取的条目。覆盖面很好。但消歧（`Apple` 公司 vs 水果）很糟糕。这就是统计模型胜出的原因。

### 步骤4：CRF步骤（草图，非完整实现）

在没有概率论基础的情况下，用50行从头实现完全CRF并不具有启发性。改用`sklearn-crfsuite`：

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` 和 `c2` 是L1和L2正则化。`all_possible_transitions=True` 让模型学习到非法序列（例如，`I-ORG` 跟在 `O` 之后）是不太可能的，这就是CRF无需你编写约束即可强制执行BIO一致性的方式。

### 步骤5：BiLSTM-CRF新增了什么

特征变为学习得到。输入：词元嵌入（GloVe或fastText）。LSTM从左到右和从右到左读取。拼接后的隐藏状态经过CRF输出层。CRF仍然强制标签序列一致性；LSTM用学习到的特征替代手工特征。

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

对于CRF层，使用`torchcrf.CRF`（pip install pytorch-crf）。与手工CRF相比的增益是可衡量的，但除非你有数万条标注句子，否则比预期要小。

## 使用它

spaCy开箱即用地提供生产级NER。

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

注意`iPhone`被标注为`ORG`而不是`PRODUCT`——spaCy的小模型对产品实体的覆盖较弱。大模型（`en_core_web_lg`）效果更好。Transformer模型（`en_core_web_trf`）效果更好。

Hugging Face用于基于BERT的NER：

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` 将连续的B-X、I-X词元合并为一个跨度。没有它，你得到词元级别的标签，必须自己合并。

### 基于LLM的NER（2026年方案）

零样本和少样本LLM NER现在在许多领域与微调模型具有竞争力，并且在标注数据稀缺时显著更好。

- **零样本提示。** 给LLM一个实体类型列表和一个示例模式。要求输出JSON。开箱即用；在新领域准确率中等。
- **ZeroTuneBio风格提示。** 将任务分解为候选提取→含义解释→判断→再检查。多阶段提示（非一次性提示）在生物医学NER上大幅提升准确率。同样的模式适用于法律、金融和科学领域。
- **使用RAG的动态提示。** 针对每次推理调用，从少量标注种子集中检索最相似的标注示例；实时构建少样本提示。在2026年的基准测试中，这使GPT-4在生物医学NER上的F1值比静态提示提升11-12%。
- **按实体类型分解。** 对于长文档，一次性提取所有实体类型的调用会随着长度增长而降低召回率。每种实体类型运行一次提取。更高的推理成本，但准确率大幅提升。这是临床笔记和法律合同的标准模式。

截至2026年的生产建议：在收集训练数据之前，从LLM零样本基线开始。通常F1值足够好，你永远不需要微调。

### 经典NER仍然胜出的领域

即使有了LLM，经典NER在以下情况仍占优势：

- 延迟预算低于50毫秒。
- 你有数千个标注样本，需要98%以上的F1值。
- 领域具有稳定的本体，预训练的CRF或BiLSTM能很好地迁移。
- 监管要求使用本地非生成式模型。

### 它的不足之处

- **领域偏移。** 在CoNLL上训练的NER在法律合同上的表现不如地名录。在你的领域上进行微调。
- **嵌套实体。** "Bank of America Tower"同时是组织(ORG)和设施(FACILITY)。标准的BIO无法表示重叠跨度。你需要嵌套NER（多遍或基于跨度的模型）。
- **长实体。** "United States Federal Deposit Insurance Corporation." 基于令牌的模型有时会将其拆分。使用`aggregation_strategy`或后处理。
- **稀疏类型。** 医学NER标签如DRUG_BRAND、ADVERSE_EVENT、DOSE。通用模型对此一无所知。Scispacy和BioBERT是起点。

## 发布

保存为 `outputs/skill-ner-picker.md`：

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## 练习

1. **简单。** 实现`bio_to_spans`（`spans_to_bio`的逆操作）并在10个句子上验证往返一致性。
2. **中等。** 在上面提到的sklearn-crfsuite CRF上训练CoNLL-2003英文NER数据集。使用`bio_to_spans`报告每个实体的F1值。典型结果：约84 F1。
3. **困难。** 在特定领域的NER数据集（医学、法律或金融）上微调`bio_to_spans`。与spaCy小型模型进行比较。记录数据泄露检查，并写出让你惊讶的地方。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
| NER  |  提取名称  |  使用类型（PERSON, ORG, GPE, DATE等）标注令牌跨度。 |
| BIO  |  标注方案  |  `B-X`开始，`I-X`继续，`O`外部。 |
| BILOU  |  更好的BIO  |  增加`L-X`（最后一个）、`U-X`（单元）以得到更清晰的边界。 |
| CRF  |  结构化分类器  |  对标签之间的转移进行建模，而不仅仅是发射。强制有效序列。 |
| 嵌套NER  |  重叠实体  |  一个跨度与其子跨度是不同的实体。BIO无法表示这一点。 |
| 实体级F1  |  合适的NER指标  |  预测跨度必须与真实跨度完全匹配。令牌级F1会夸大准确性。 |

## 延伸阅读

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360)——BiLSTM-CRF论文。经典之作。
- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360)——引入了成为标准的令牌分类模式。
- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360)——有关[Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)和[spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities)上每个属性的实用参考。
- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360)——正确的指标库。始终使用它。
