# 主题建模 — LDA 和 BERTopic

> LDA: 文档是主题的混合，主题是词上的分布。BERTopic: 文档在嵌入空间中聚类，聚类就是主题。目标相同，分解方式不同。

**类型:** 学习
**语言:** Python
**先修知识:** 阶段5·02 (词袋+TF-IDF)，阶段5·03 (Word2Vec)
**时间:** 约45分钟

## 问题

你有10,000份客户支持工单、50,000篇新闻文章或200,000条推文。你需要在不阅读的情况下知道这些集合是关于什么的。你没有标注的类别。你甚至不知道有多少个类别存在。

主题建模在无监督的情况下回答了这个问题。输入一个语料库，返回一组连贯的主题以及每个文档在这些主题上的分布。

两种算法族占主导地位。LDA（2003）将每个文档视为潜在主题的混合，每个主题视为词上的分布。推理是贝叶斯的。它仍然部署在生产中，当需要混合成员主题分配和可解释的词级概率分布时。

BERTopic（2020）使用BERT编码文档，使用UMAP降维，使用HDBSCAN聚类，并通过基于类别的TF-IDF提取主题词。它在短文本、社交媒体以及任何语义相似性比词重叠更重要的场景中表现出色。每个文档获得一个主题，这对于长文本来说是一个限制。

本节课为两者建立直觉，并说明针对给定语料库应选择哪一个。

## 核心概念

![LDA mixture model vs BERTopic clustering](../assets/topic-modeling.svg)

**LDA生成故事。** 每个主题是词上的分布。每个文档是主题的混合。为了生成文档中的一个词，先从文档的混合中采样一个主题，然后从该主题的分布中采样一个词。推理反过来：给定观察到的词，推断每个文档的主题分布和每个主题的词分布。折叠吉布斯采样或变分贝叶斯完成数学计算。

LDA的关键输出：

- `doc_topic`: 矩阵 `(n_docs, n_topics)`，每行和为1（文档的主题混合）。
- `doc_topic`: 矩阵 `(n_docs, n_topics)`，每行和为1（主题的词分布）。

**BERTopic流程。**

1. 使用句子转换器（如`all-MiniLM-L6-v2`）编码每个文档。384维向量。
2. 使用UMAP降维至约5维。BERT嵌入对于聚类而言维度过高。
3. 使用HDBSCAN聚类。基于密度，产生可变大小的聚类和一个“离群”标签。
4. 对于每个聚类，计算基于类别的TF-IDF以提取顶部词。

输出是每个文档一个主题（加上一个-1离群标签）。可选地，通过HDBSCAN的概率向量获得软成员关系。

## 动手构建

### 步骤1: 通过scikit-learn使用LDA

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

注意：已去除停用词，min_df和max_df过滤罕见和普遍词，使用CountVectorizer（而非TfidfVectorizer），因为LDA期望原始计数。

### 步骤2: BERTopic（生产环境）

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

对`Topic != -1`的过滤去掉了BERTopic的离群桶（HDBSCAN无法聚类的文档）。`min_topic_size`控制HDBSCAN的最小聚类大小；BERTopic库的默认值为10。此示例为了课程规模明确设为15。对于超过10,000个文档的语料库，增加到50或100。

### 步骤3: 评估

两种方法都输出主题词。问题是这些词是否连贯。

- **主题连贯性 (c_v)。** 结合顶部词对的NPMI（归一化逐点互信息），在滑动窗口上下文中聚合得分到主题向量，并通过余弦相似度比较这些向量。越高越好。使用`gensim.models.CoherenceModel`和`coherence="c_v"`。
- **主题多样性。** 所有主题顶部词中唯一词的比例。越高越好（主题不重叠）。
- **定性检查。** 阅读每个主题的顶部词。它们是否命名了一个真实的事物？人的判断仍然是最后一道防线。

## 如何选择

|  情况  |  选择  |
|-----------|------|
|  短文本（推文、评论、标题）  |  BERTopic  |
|  具有主题混合的长文档  |  LDA  |
|  无GPU/有限计算  |  LDA 或 NMF  |
|  需要文档级多主题分布  |  LDA  |
|  需要LLM集成进行主题标注  |  BERTopic（直接支持）  |
|  资源受限的边缘部署  |  LDA  |
|  最大语义连贯性  |  BERTopic  |

最大的实际考虑是文档长度。BERT嵌入会截断；LDA计数适用于任何长度。对于比嵌入模型上下文更长的文档，要么分块聚合，要么使用LDA。

## 使用它

2026年技术栈：

- **BERTopic。** 短文本以及语义重要的任何场景的默认选择。
- **`gensim.models.LdaModel`。** 经典LDA用于生产，成熟，久经考验。
- **`gensim.models.LdaModel`。** 用于实验的简易LDA。
- **NMF。** 非负矩阵分解。LDA的快速替代方案，在短文本上质量相当。
- **Top2Vec。** 与BERTopic设计相似。社区较小，但在某些基准上表现良好。
- **FASTopic。** 较新，在非常大的语料库上比BERTopic更快。
- **基于LLM的标注。** 运行任意聚类，然后提示模型为每个聚类命名。

## 发布

保存为 `outputs/skill-topic-picker.md`：

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## 练习

1. **简单.** 在20 Newsgroups数据集上用5个主题拟合LDA。打印每个主题的前10个单词。手动标注每个主题。算法是否找到了真实类别？
2. **中等.** 在同一20 Newsgroups子集上拟合BERTopic。比较找到的主题数量、前几个单词以及与LDA的定性一致性。哪个更清晰地揭示了真实类别？
3. **困难.** 计算语料库上LDA和BERTopic的c_v一致性。分别用5、10、20、50个主题运行。绘制一致性随主题数量的变化图。报告哪种方法在不同主题数量下更稳定。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  主题 | 语料库所涉及的事物 | 词上的概率分布（LDA）或相似文档的聚类（BERTopic）。  |
|  混合隶属度 | 文档属于多个主题 | LDA为每个文档分配一个在所有主题上的分布。  |
|  UMAP | 降维 | 保留局部结构的流形学习；用于BERTopic。  |
|  HDBSCAN | 密度聚类 | 找到可变大小的聚类；为异常值生成“噪声”标签(-1)。  |
|  c_v一致性 | 主题质量度量 | 滑动窗口内顶部主题词的平均点互信息。  |

## 延伸阅读

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — LDA论文。
- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — BERTopic论文。
- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — 介绍c_v及其相关指标的论文。
- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — 生产参考。优秀的示例。
