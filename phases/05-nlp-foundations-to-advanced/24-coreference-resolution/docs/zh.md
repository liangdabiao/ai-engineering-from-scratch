# 共指消解

> “她打电话给他。他没有接。医生正在吃午饭。”三个指代指向两个人，但没有人被提及姓名。共指消解就是确定谁是谁。

**类型：** 学习
**语言：** Python
**前置要求：** 阶段5·06（命名实体识别），阶段5·07（词性标注与句法分析）
**时间：** 约60分钟

## 问题

从一篇300字的文章中提取所有提及苹果公司(Apple Inc.)的地方。当文章说“苹果(Apple)”时很容易，但当文章说“该公司(the company)”、“他们(they)”、“库比蒂诺的科技巨头(Cupertino's technology giant)”或“乔布斯的公司(Jobs's firm)”时就很难了。如果不将这些提及解析到同一实体，你的命名实体识别流水线(NER pipeline)会漏掉60-80%的提及。

共指消解将每个指向同一现实世界实体的表达式链接到一个簇(cluster)中。它是表层自然语言处理(NLP)（命名实体识别、句法分析）与下游语义（信息抽取、问答、摘要、知识图谱）之间的粘合剂。

为何在2026年重要：

- 摘要生成(Summarization)：“CEO宣布……”对比“蒂姆·库克宣布……”——摘要应命名CEO。
- 问答(Question answering)：“她打电话给谁？”需要解析“她”。
- 信息抽取(Information extraction)：一个知识图谱中如果“PER1创立了苹果”和“乔布斯创立了苹果”作为独立条目就是错误的。
- 多文档信息抽取(Multi-document IE)：合并关于同一事件的不同文章中的提及，即跨文档共指(Cross-document coreference)。

## 核心概念

![Coreference clustering: mentions → entities](../assets/coref.svg)

**任务。** 输入：一个文档。输出：提及（文本跨度）的聚类，每个簇指向一个实体。

**提及类型。**

- **命名实体(Named entity)。** “蒂姆·库克(Tim Cook)”
- **名词性短语(Nominal)。** “CEO”、“该公司”
- **代词(Pronominal)。** “他”、“她”、“他们”、“它”
- **同位语(Appositive)。** “蒂姆·库克，苹果的CEO，”

**架构。**

1. **基于规则（Hobbs, 1978）。** 基于语法树的代词解析，使用语法规则。很好的基线。在代词上出乎意料地难以超越。
2. **提及对分类器(Mention-pair classifier)。** 对每对提及(m_i, m_j)，预测它们是否共指。通过传递闭包(transitive closure)进行聚类。2016年之前的标准方法。
3. **提及排序(Mention-ranking)。** 对每个提及，给候选先行语(candidate antecedents)（包括“无先行语”）排序。选择排名最高的。
4. **基于跨度的端到端模型（Lee等人, 2017）。** 使用Transformer编码器。枚举所有长度上限内的候选跨度。预测提及分数。为每个跨度预测先行语概率。贪心聚类。现代的默认方法。
5. **生成式（2024年以后）。** 提示大语言模型(LLM)：“列出本文中每个代词及其先行语。”在简单情况下表现良好，但在长文档和罕见指代上困难。

**评估指标。** 五个标准指标（MUC、B³、CEAF、BLANC、LEA），因为没有一个单一指标能捕捉聚类质量。报告前三个指标的平均值作为CoNLL F1。2026年在CoNLL-2012上的最优水平：约83 F1。

**已知困难情况。**

- 指向前几页引入的实体的确定性描述(Definite descriptions)。
- 搭桥回指(Bridging anaphora)（“车轮”指向前文提到的汽车）。
- 中文和日语等语言中的零回指(Zero anaphora)。
- 后指(Cataphora)（代词出现在指代对象之前）：“当**她**走进来时，玛丽笑了。”

## 动手构建

### 第一步：预训练的神经共指解析（AllenNLP / spaCy-experimental）

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

在较长文档上，你会得到类似这样的结果：
- 簇1：[苹果(Apple), 该公司(The company), 它们(they)]
- 簇2：[新产品(new products)]

### 第二步：基于规则的代词解析器（教学用）

参见`code/main.py`获取仅使用标准库(stdlib)的实现：

1. 提取提及：命名实体（大写跨度）、代词（字典查找）、确定性描述（“the X”）。
2. 对每个代词，查看前K个提及，并根据以下条件评分：
   - 性别/数一致性（启发式）
   - 近因性（更近的优先）
   - 句法角色（主语优先）
3. 链接得分最高的先行语。

无法与神经模型竞争。但它展示了搜索空间以及端到端模型必须做出的决策。

### 第三步：使用LLM进行共指消解

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

注意两种失败模式。首先，LLM过度合并（“他”和“她”指向两个不同的人）。第二，LLM在长文档中静默地丢弃提及。始终使用跨度偏移检查(span-offset checks)进行验证。

### 第四步：评估

标准的CoNLL-2012脚本计算MUC、B³、CEAF-φ4并报告平均值。对于内部评估，先从带标注的测试集上的跨度级别精确率和召回率开始，然后添加提及链接F1。

## 陷阱

- **单例膨胀(Singleton explosion)。** 一些系统将每个提及报告为单独的簇。B³对此宽松，而MUC会惩罚。始终检查所有三个指标。
- **长上下文中的代词。** 在超过2000个令牌的文档上，性能下降约15 F1。小心地进行分块。
- **性别假设。** 硬编码的性别规则在非二元指代、组织、动物上会失效。使用学习模型或中性评分。
- **LLM在长文档上的漂移。** 单次API调用无法可靠地对50多个段落中的提及进行聚类。使用滑动窗口+合并。

## 使用它

2026年技术栈：

|  情况  |  选择  |
|-----------|------|
|  英语，单文档  |  `en_coreference_web_trf` (spaCy-experimental) 或 AllenNLP 神经共指解析  |
|  多语言  |  在OntoNotes或多语言CoNLL上训练的SpanBERT / XLM-R  |
|  跨文档事件共指  |  专门的端到端模型（2025–26年最优）  |
| 快速LLM基线  |  使用结构化输出指代消解提示的GPT-4o / Claude |
| 生产对话系统  |  基于规则的备选 + 神经主模型 + 关键槽位的人工审核 |

2026年发布的集成模式：先运行命名实体识别，再运行指代消解，将指代簇合并到NER实体中。下游任务每个簇看到一个实体，而不是每个提及看到一个实体。

## 发布

保存为 `outputs/skill-coref-picker.md`：

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## 练习

1. **简单.** 在`code/main.py`中对5个手工编写的段落运行基于规则的解析器。测量提及链接准确率与真实标签的对比。
2. **中等.** 在一篇新闻文章上使用预训练的神经指代消解模型。将簇与你自己的手动标注进行比较。它在哪里失败了？
3. **困难.** 构建一个指代增强的NER流水线：先NER，然后通过指代簇合并。在100篇文章上测量实体覆盖改进与仅NER的对比。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
| 提及  |  一个引用  |  指代某个实体的文本片段（名称、代词、名词短语）。 |
| 先行词  |  “它”所指的内容  |  后续提及所指的前面提及。 |
| 簇  |  实体的提及集合  |  所有指代同一真实世界实体的提及的集合。 |
| 回指  |  后向指代  |  后续提及指向前面的（“他”→“约翰”）。 |
| 前指  |  前向指代  |  前面提及指向后面的（“当他到达时，约翰...”）。 |
| 桥指  |  隐式指代  |  “我买了一辆车。轮子坏了。”（指那辆车的轮子。） |
| CoNLL F1  |  排行榜上的数字  |  MUC、B³、CEAF-φ4 F1分数的平均值。 |

## 延伸阅读

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 经典教材章节。
- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 基于跨度（Span）的端到端。
- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 提升指代消解的预训练。
- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 基准测试。
- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 基于规则的经典方法。
