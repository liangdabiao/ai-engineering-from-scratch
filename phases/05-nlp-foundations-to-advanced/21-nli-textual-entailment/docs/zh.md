# 自然语言推理 — 文本蕴涵

> "t 蕴涵 h" 意味着人类读者阅读 t 后会得出 h 为真。NLI 是预测蕴涵/矛盾/中立的任务。表面枯燥，生产中至关重要。

**类型：** 学习
**语言：** Python
**前置条件：** 第5阶段·05（情感分析），第5阶段·13（问答）
**时间：** 约60分钟

## 问题

你构建了一个摘要器。它生成了一个摘要。如何知道摘要中不包含幻觉？

你构建了一个聊天机器人。它回答了“是”。如何知道该答案得到了检索段落支持？

你需要将10,000篇新闻文章按主题分类。没有训练标签。能否复用模型？

这三个问题都归结为自然语言推理。NLI 要求：给定前提 `t` 和假设 `h`，判断 `h` 是否被 `t` 蕴涵、矛盾还是无关？

- **幻觉检查：** `t` = 源文档，`h` = 摘要陈述。非蕴涵 = 幻觉。
- **有依据的问答：** `t` = 检索段落，`h` = 生成答案。非蕴涵 = 捏造。
- **零样本分类：** `t` = 文档，`h` = 词语化标签（“这是关于体育的”）。蕴涵 = 预测标签。

一个任务，三种生产用途。这就是每个 RAG 评估框架在底层都搭载一个 NLI 模型的原因。

## 核心概念

![NLI: three-way classification, premise vs hypothesis](../assets/nli.svg)

**三个标签。**

- **蕴涵。** `t` → `h`。“猫在垫子上”蕴涵“有一只猫”。
- **矛盾。** `t` → ¬`h`。“猫在垫子上”与“没有猫”矛盾。
- **中立。** 无法推断任何方向。“猫在垫子上”与“猫饿了”中立。

**不是逻辑蕴涵。** NLI 是*自然*语言推理——典型人类读者会推断出的内容，而非严格逻辑。“John遛了他的狗”在 NLI 中蕴涵“John有一只狗”，但严格的一阶逻辑只有在将占有公理化后才承认。

**数据集。**

- **SNLI**（2015年）。57万个人工标注对，以图像描述为前提。领域狭窄。
- **MultiNLI**（2017年）。43.3万个对，涵盖10种体裁。2026年的标准训练语料。
- **ANLI**（2019年）。对抗性NLI。人类编写专门用于攻破现有模型的示例。更难。
- **DocNLI、ConTRoL**（2020–2021年）。文档级前提。测试多跳和长程推理。

**架构。** 一个 transformer 编码器（BERT、RoBERTa、DeBERTa）读取 `[CLS] premise [SEP] hypothesis [SEP]`。`[CLS]` 的表示送入一个3路softmax。在MNLI上训练，在留出基准上评估，分布内对准确率达90%以上。

**通过NLI的零样本分类。** 给定一个文档和候选标签，将每个标签转化为一个假设（“这篇文本是关于体育的”）。计算每个标签的蕴涵概率。选择最大值。这是 Hugging Face 的 `zero-shot-classification` 管道背后的机制。

## 动手构建

### 步骤1：运行预训练的NLI模型

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

对于生产级NLI，`facebook/bart-large-mnli` 和 `microsoft/deberta-v3-large-mnli` 是开放的默认选项。DeBERTa-v3 在排行榜上领先。

### 步骤2：零样本分类

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

默认模板是“这个示例是关于 {label} 的。”使用 `hypothesis_template` 自定义。无需训练数据。无需微调。开箱即用。

### 步骤3：RAG的忠实度检查

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

这是 RAGAS 忠实度的核心。将生成的答案拆分为原子化声明。对照检索到的上下文检查每个声明。报告蕴涵的比例。

### 步骤4：手写NLI分类器（概念性）

参见 `code/main.py` 获取仅用标准库的玩具示例：前提和假设通过词汇重叠+否定检测进行比较。无法与 transformer 模型竞争——但展示了任务的形态：两个文本输入，3路标签输出，损失 = 在 `{entail, contradict, neutral}` 上的交叉熵。

## 陷阱

- **仅假设的捷径。** 模型仅从假设即可预测标签，在SNLI上约60%准确率，因为“不”、“没有人”、“从不”与矛盾相关。用于检测标签泄漏的强基线。
- **词汇重叠启发式。** 子序列启发式（“每个子序列都被蕴涵”）在SNLI上有效，但在HANS/ANLI上失败。使用对抗性基准。
- **文档级退化。** 单句NLI模型在文档级前提上F1下降20以上。对长上下文使用DocNLI训练的模型。
- **零样本模板敏感性。** “这个示例是关于 {label} 的” vs “{label}” vs “主题是 {label}” 可使准确率波动10个百分点以上。调优模板。
- **领域不匹配。** MNLI在通用英语上训练。法律、医学和科学文本需要特定领域的NLI模型（例如SciNLI、MedNLI）。

## 使用它

2026年技术栈：

|  用例  |  模型  |
|---------|-------|
|  通用NLI  |  `microsoft/deberta-v3-large-mnli`  |
|  快速/边缘  |  `cross-encoder/nli-deberta-v3-base`  |
|  零样本分类（轻量级）  |  `facebook/bart-large-mnli`  |
|  文档级NLI  |  `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`  |
|  Multilingual  |  `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`  |
|  RAG中的幻觉检测 | RAGAS / DeepEval 中的 NLI 层 |

2026年的元模式：NLI是文本理解的万能胶带。每当你需要“A是否支持B？”或“A是否矛盾B？”时——在调用另一个LLM之前，请先使用NLI。

## 发布

保存为 `outputs/skill-nli-picker.md`：

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## 练习

1. **简单。** 在20个手工制作的（前提，假设，标签）三元组上运行`facebook/bart-large-mnli`，覆盖所有三个类别。测量准确率。添加对抗性的“子序列启发式”陷阱（“我没吃蛋糕” vs “我吃了蛋糕”），看看它是否失败。
2. **中等。** 在100个AG新闻标题上比较零样本模板`facebook/bart-large-mnli`与`"This text is about {label}"`和`"The topic is {label}"`。报告准确率波动。
3. **困难。** 构建一个RAG忠实度检查器：原子声明分解 + 每个声明的NLI。在50个RAG生成的答案上进行评估，使用黄金上下文。测量与人工标签相比的假阳性和假阴性率。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  NLI | 自然语言推理(Natural Language Inference) | 前提-假设关系的三分类。  |
|  RTE | 识别文本蕴含(Recognizing Textual Entailment) | NLI的旧名称；相同任务。  |
|  蕴含(Entailment) | “t蕴含h” | 一个典型读者会认为给定t时h为真。  |
|  矛盾(Contradiction) | “t排除h” | 一个典型读者会认为给定t时h为假。  |
|  中立(Neutral) | “未决定” | 从t无法推断h。  |
|  零样本分类(Zero-shot classification) | 作为分类器的NLI | 将标签表达为假设，选择最大蕴含。  |
|  忠实度(Faithfulness) | 答案是否得到支持？ | 对（检索到的上下文，生成的答案）进行NLI。  |

## 延伸阅读

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI。
- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — MultiNLI。
- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — ANLI基准。
- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — 作为分类器的NLI。
- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — 2026年的NLI工作马。
