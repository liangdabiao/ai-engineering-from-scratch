# 实体链接与消歧(Entity Linking & Disambiguation)

> NER找到了"Paris"。实体链接决定：法国巴黎？帕丽斯·希尔顿？德克萨斯州巴黎？帕里斯（特洛伊王子）？没有链接，知识图谱仍然模糊不清。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段5 · 06（命名实体识别），阶段5 · 24（共指消解）
**时间：** 约60分钟

## 问题

一个句子写道："Jordan beat the press." 你的NER将"Jordan"标记为人物。很好。但是*哪个*乔丹？

- 迈克尔·乔丹（篮球）？
- 迈克尔·B·乔丹（演员）？
- 迈克尔·I·乔丹（伯克利机器学习教授——是的，这种混淆在机器学习论文中真实存在）？
- 约旦（国家）？
- 乔丹（希伯来语名字）？

实体链接(Entity Linking, EL)将每个提及解析为知识库中的唯一条目：维基数据、维基百科、DBpedia或你的领域知识库。两个子任务：

1. **候选生成(Candidate generation)。** 给定"Jordan"，哪些知识库条目是合理的？
2. **消歧(Disambiguation)。** 给定上下文，哪个候选是正确的？

这两个步骤都是可学习的。它们都有基准测试。组合管道已经稳定了十年——变化的是消歧器的质量。

## 核心概念

![Entity linking pipeline: mention → candidates → disambiguated entity](../assets/entity-linking.svg)

**候选生成。** 给定提及的表层形式("Jordan")，在别名索引中查找候选。维基百科别名词典覆盖大多数命名实体："JFK" → 约翰·F·肯尼迪、杰奎琳·肯尼迪、JFK机场、JFK（电影）。典型索引每个提及返回10-30个候选。

**消歧：三种方法。**

1. **先验+上下文(Prior + context)（Milne & Witten, 2008）。** `P(entity | mention) × context-similarity(entity, text)`。效果好，速度快，无需训练。
2. **基于嵌入(Embedding-based)（ESS / REL / Blink）。** 编码提及+上下文。编码每个候选的描述。取最大余弦。2020-2024年的默认方法。
3. **生成式(Generative)（GENRE, 2021; 基于LLM, 2023+）。** 逐token解码实体的规范名称。约束为有效实体名称的trie，确保输出是有效的知识库ID。

**端到端 vs 管道。** 现代模型（ELQ, BLINK, ExtEnD, GENRE）一步完成NER+候选生成+消歧。管道系统在生产中仍占主导地位，因为你可以交换组件。

### 两个度量指标

- **提及召回率（候选生成）。** 正确知识库条目出现在候选列表中的黄金提及比例。整个管道的下限。
- **消歧准确率/F1。** 给定正确候选，top-1正确的频率。

始终报告两者。候选召回率80%上的消歧准确率99%的系统是80%的管道。

## 动手构建

### 步骤1：从维基百科重定向构建别名索引

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

维基百科别名数据：约1800万（别名，实体）对。从维基数据转储下载。存储为倒排索引。

### 步骤2：基于上下文的消歧

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard重叠是一个玩具。用嵌入上的余弦相似度替换（参见`code/main.py`步骤2了解Transformer版本）。

### 步骤3：基于嵌入（BLINK风格）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

在索引时，一次嵌入每个知识库实体。在查询时，一次嵌入提及+上下文，与候选池点积，取最大值。

### 步骤4：生成式实体链接（概念）

GENRE逐字符解码实体的维基百科标题。约束解码（参见第20课）确保只能输出有效标题。与基于知识库的trie紧密结合。现代后代是REL-GEN和带有结构化输出的LLM提示EL。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

结合白名单（Outlines `choice`），这是2026年最简单可部署的EL管道。

### 步骤5：在AIDA-CoNLL上评估

AIDA-CoNLL是标准的EL基准：1,393篇路透社文章，34,000个提及，维基百科实体。报告知识库内准确率（`P@1`）和知识库外NIL检测率。

## 陷阱

- **NIL处理。** 有些提及不在知识库中（新兴实体、不知名人物）。系统必须预测NIL而不是猜测错误实体。单独测量。
- **提及边界错误。** 上游NER遗漏部分跨度（"Bank of America"被标记为仅"Bank"）。EL召回率下降。
- **流行度偏差。** 训练过的系统过度预测频繁实体。机器学习论文中提及"Michael I. Jordan"经常链接到篮球乔丹。
- **跨语言EL。** 将中文文本中的提及映射到英文维基百科实体。需要多语言编码器或翻译步骤。
- **知识库过时。** 新公司、事件、人物不在去年的维基百科转储中。生产管道需要刷新循环。

## 使用它

2026年技术栈：

|  情况  |  选择  |
|-----------|------|
|  通用英文 + 维基百科  |  BLINK 或 REL  |
|  跨语言，知识库 = 维基百科  |  mGENRE  |
|  适合LLM，每天少量提及  |  用候选列表+约束JSON提示Claude/GPT-4  |
| 领域特定知识库(医学、法律) |||| 使用自定义BERT进行知识库感知检索 + 在领域AIDA风格数据集上微调 |  |
| 极低延迟 |||| 仅精确匹配先验(Milne-Witten基线) |  |
| 研究级SOTA |||| GENRE / ExtEnD / 生成式LLM实体链接 |  |

2026年上线的生产模式：命名实体识别(NER) → 共指消解 → 对每个提及进行实体链接(EL) → 将每个簇压缩为一个规范实体。输出：文档中每个实体一个知识库ID，而非每个提及一个。

## 发布

保存为 `outputs/skill-entity-linker.md`：

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## 练习

1. **简单。** 在`code/main.py`中实现先验+上下文消歧器，处理10个歧义提及(Paris, Jordan, Apple)。手工标注正确实体。测量准确率。
2. **中等。** 用句子编码器编码50个歧义提及。嵌入每个候选实体的描述。比较基于嵌入的消歧与Jaccard上下文重叠。
3. **困难。** 构建一个1k实体的领域知识库(例如你公司的员工+产品)。实现端到端的NER+EL。在100个保留句子上测量精确率和召回率。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
| 实体链接(EL) |||| 链接到维基百科 |||| 将提及映射到唯一的KB条目。 |  |  |
| 候选生成 |||| 可能是谁？ |||| 返回一个提及的可能KB条目候选列表。 |  |  |
| 消歧 |||| 选择正确的那个 |||| 使用上下文对候选打分，选出胜者。 |  |  |
| 别名索引 |||| 查找表 |||| 从表面形式映射到候选实体。 |  |  |
| NIL |||| 不在知识库中 |||| 明确预测没有KB条目匹配。 |  |  |
| KB |||| 知识库 |||| Wikidata, Wikipedia, DBpedia, 或你的领域知识库。 |  |  |
| AIDA-CoNLL |||| 基准测试集 |||| 1393篇带有黄金实体链接的路透社文章。 |  |  |

## 延伸阅读

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — 基础的先验+上下文方法。
- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — 基于嵌入的主力模型。
- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — 带约束解码的生成式EL。
- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — 基准论文。
- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — 开源生产栈。
