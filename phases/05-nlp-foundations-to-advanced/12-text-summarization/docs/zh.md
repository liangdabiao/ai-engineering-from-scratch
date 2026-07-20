# 文本摘要

> 抽取式系统告诉你文档说了什么。生成式系统告诉你作者想表达什么。不同的任务，不同的陷阱。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段5·02 (词袋法 + TF-IDF)，阶段5·11 (机器翻译)
**时间：** 约75分钟

## 问题

一篇2000词的新闻文章出现在你的信息流中。你需要120词来概括它。你可以从文章中挑选三个最重要的句子（extractive），或者用自己的话重写内容（abstractive）。两者都称为摘要。它们是完全不同的问题。

抽取式摘要是一个排序问题。对每个句子打分，返回前`k`个。输出总是语法正确的，因为它是逐字摘录的。风险在于遗漏分布在文章各处的信息。

生成式摘要是一个生成问题。Transformer根据输入条件生成新文本。输出流畅且压缩，但可能会产生源文本中没有的事实幻觉。风险在于自信地编造。

本课将构建这两种方法，以及各自拥有的失败模式。

## 核心概念

![Extractive TextRank vs abstractive transformer](../assets/summarization.svg)

**抽取式。** 将文章视为一个图，节点是句子，边是相似度。在图上运行PageRank（或类似算法），根据句子与其他所有句子的连接程度对其进行评分。得分最高的句子就是摘要。经典实现是**TextRank**（Mihalcea 和 Tarau，2004）。

**生成式。** 在文档-摘要对上微调Transformer编码器-解码器（BART、T5、Pegasus）。推理时，模型读取文档，通过交叉注意力逐token生成摘要。特别是Pegasus，它使用间隔句子(gap-sentence)预训练目标，使其在没有大量微调的情况下就能很好地完成摘要任务。

使用**ROUGE**（面向召回的摘要评估替身）进行评估。ROUGE-1和ROUGE-2评估一元组和二元组重叠。ROUGE-L评估最长公共子序列。越高越好，但ROUGE-L 40为"良好"，50为"优秀"。每篇论文都报告这三个指标。使用`rouge-score`包。

## 动手构建

### 步骤1：TextRank（抽取式）

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

有两件事值得一提。相似度函数使用对数归一化的词重叠，这是原始TextRank变体。TF-IDF向量的余弦相似度也可以。阻尼因子0.85和迭代次数是PageRank的默认值。

### 步骤2：使用BART进行生成式摘要

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN在CNN/DailyMail语料库上微调。它可以开箱即用地生成新闻风格的摘要。对于其他领域（科学论文、对话、法律），使用相应的Pegasus检查点或在目标数据上微调。

### 步骤3：ROUGE评估

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

始终使用词干提取。否则，"running"和"run"会被计为不同单词，导致ROUGE计数不足。

### 超越ROUGE（2026年摘要评估）

ROUGE作为主导的摘要指标已有二十年，在2026年它本身已不足够。一项对NLG论文的大规模元分析显示：

- **BERTScore**（上下文嵌入相似度）到2023年逐渐流行，现在大多数摘要论文中与ROUGE一起报告。
- **BARTScore** 将评估视为生成：通过预训练BART在给定源文本下分配摘要的可能性来评分。
- **MoverScore**（上下文嵌入上的推土机距离）在2025年摘要基准测试中达到最高点，因为它比ROUGE更好地捕捉了语义重叠。
- **FactCC** 和 **基于QA的忠实度** 在2021-2023年常见，现在常被 **G-Eval**（一条GPT-4提示链，通过思维链推理对连贯性、一致性、流畅性、相关性进行评分）所取代。
- **G-Eval** 和类似的LLM评委方法在规则设计良好时，与人类判断的一致性约为80%。

生产建议：报告ROUGE-L用于传统比较，BERTScore用于语义重叠，G-Eval用于连贯性和事实性。针对50-100个人工标注摘要进行校准。

### 步骤4：事实性问题

生成式摘要容易产生幻觉。抽取式摘要的幻觉风险要低得多，因为输出是从源文本逐字摘录的，但如果源句子脱离上下文、过时或引用顺序混乱，它们仍然可能误导。这是生产系统在合规相关内容中仍然偏好抽取方法的唯一最大原因。

需要指出的幻觉类型：

- **实体交换。** 源文本说"John Smith。"摘要说"John Brown。"
- **数字漂移。** 源文本说"25,000。"摘要说"2500万。"
- **极性翻转。** 源文本说"拒绝了提议。"摘要说"接受了提议。"
- **事实编造。** 源文本未提及CEO。摘要说CEO批准了。

有效的评估方法：

- **FactCC。** 一个在源句和摘要句之间的蕴含关系上训练的二元分类器。预测事实/非事实。
- **基于QA的事实性。** 向QA模型提问，答案在源文本中。如果摘要支持不同的答案，则标记。
- **实体级F1。** 比较源文本和摘要中的命名实体。仅出现在摘要中的实体可疑。

对于任何面向用户且事实性重要的内容（新闻、医疗、法律、金融），抽取式是更安全的默认选择。生成式需要在流程中加入事实性检查。

## 使用它

2026年技术栈：

|  用例  |  推荐  |
|---------|-------------|
|  新闻，3-5句摘要，英语  |  `facebook/bart-large-cnn`  |
| 科学论文 | `google/pegasus-pubmed` 或微调后的 T5 |
| 多文档长文本 | 任何具有32k+上下文的LLM，经过提示 |
| 对话摘要 | `philschmid/bart-large-cnn-samsum` |
| 抽取式，结构上低幻觉风险 | TextRank 或 `sumy` 的 LSA / LexRank |

具有长上下文的LLM在2026年往往胜过专门模型，当计算资源不受限制时。权衡在于成本和可复现性；专门模型的输出更一致。

## 发布

保存为 `outputs/skill-summary-picker.md`：

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## 练习

1. **简单。** 对5篇新闻文章运行TextRank。将前3个句子与参考摘要进行比较。测量ROUGE-L。在CNN/DailyMail风格的新闻上，你应该看到30-45的ROUGE-L。
2. **中等。** 实现实体级别的事实性：从源文档和摘要中提取命名实体（使用spaCy），计算源实体在摘要中的召回率以及摘要实体相对于源文档的精确率。高精确率和低召回率意味着安全但简洁；低精确率意味着存在幻觉实体。
3. **困难。** 在50篇CNN/DailyMail文章上比较BART-large-CNN与一个LLM（Claude或GPT-4）。报告ROUGE-L、事实性（通过实体F1分数）和每篇摘要的成本。记录每个模型在哪方面更优。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
| 抽取式 | 选择句子 | 从源文档逐字返回句子。绝不产生幻觉。 |
| 生成式 | 重写 | 基于源文档生成新文本。可能产生幻觉。 |
| ROUGE | 摘要指标 | 系统输出与参考之间的n-gram / LCS重叠。 |
| TextRank | 基于图的抽取式 | 在句子相似图上的PageRank。 |
| 事实性 | 是否正确 | 摘要中的声称是否得到源文档支持。 |
| 幻觉 | 虚构内容 | 摘要中源文档不支持的内容。 |

## 延伸阅读

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — 抽取式经典论文。
- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — BART论文。
- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — Pegasus与间隙句子目标。
- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — ROUGE论文。
- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — 事实性领域论文。
