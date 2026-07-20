# 词袋模型、TF-IDF 与文本表示

> 先计数，再思考。到2026年，在定义明确的任务上，TF-IDF仍然胜过嵌入模型。

**类型：** 构建
**语言：** Python
**前提条件：** 阶段5·01（文本处理），阶段2·02（从零实现线性回归）
**时间：** 约75分钟

## 问题

模型需要数字，而你只有字符串。

每个NLP流水线都必须回答同一个问题：如何将可变长度的词元流转换为分类器可以接收的固定大小向量？该领域第一个答案是最简单但有效的：统计单词数量，构建一个向量。

这个向量支撑的生产级NLP系统比任何嵌入模型都多。垃圾邮件过滤器、主题分类器、日志异常检测、搜索排名（在BM25之前）、第一波情感分析、第一个十年的学术NLP基准测试。到2026年，从业者在狭窄的分类任务上仍然首选它。速度快、可解释性强，且在单词出现与否起决定作用的任务上，往往与4亿参数的嵌入模型表现无差别。

本节课从零构建词袋模型，然后是TF-IDF。接着展示scikit-learn用三行代码实现相同功能。最后指出让你转向嵌入模型的失败模式。

## 核心概念

**词袋模型(Bag of Words, BoW)** 丢弃顺序。对每个文档，统计每个词汇表中单词出现的次数。向量长度等于词汇表大小。位置`i`是单词`i`出现的次数。

**TF-IDF** 对BoW进行重新加权。出现在每个文档中的单词信息量低，因此将其降权。在整个语料库中罕见但在单个文档中频繁出现的单词是信号，因此将其升权。

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

其中`TF`是文档中的词频(Term Frequency)，`df`是文档频率(Document Frequency，即包含该词的文档数)，`N`是文档总数。`log`确保常见单词的权重有界。

关键特性：两者都生成具有可解释维度的稀疏向量。你可以查看训练好的分类器的权重，并读出哪些单词将文档推向每个类别。而768维的BERT嵌入则无法做到这一点。

```figure
bow-tfidf
```

## 动手构建

### 第一步：构建词汇表

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

输入：分词后的文档列表（任何词级分词器均可；本节课的`code/main.py`使用简化的小写变体）。输出：`{word: index}`字典。稳定的插入顺序意味着单词索引0是第一个文档中遇到的第一个单词。惯例不同；scikit-learn按字母顺序排序。

### 第二步：词袋模型

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

行是文档，列是词汇表索引。条目`[i][j]`表示“单词`j`在文档`i`中出现的次数”。文档1中`cat`出现了两次，因为确实如此。文档0中`ran`出现零次，因为未出现。

### 第三步：词频和文档频率

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

两个值得命名的平滑技巧。`(n+1)/(d+1)`避免了`log(x/0)`。末尾的`+1`确保出现在每个文档中的单词的IDF仍为1（而非0），与scikit-learn的默认值一致。其他实现使用原始`log(N/df)`。两者都有效；平滑版本更友好。

### 第四步：TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

三个文档，五个词汇表单词（`the`, `cat`, `sat`, `dog`, `ran`）。`the`出现在所有三个文档中，因此其IDF较低。`dog`只出现在一个文档中，因此其IDF较高。向量是稀疏的（大多数条目很小），并且区分性单词突出。

### 第五步：L2归一化行

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

如果不进行归一化，较长的文档会得到更大的向量，从而主导相似度分数。L2归一化将每个文档放在单位超球面上。行之间的余弦相似度现在只是点积。

## 使用它

scikit-learn提供了生产版本。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer`在一次调用中完成分词、词汇表和BoW。`TfidfVectorizer`添加IDF加权和L2归一化。两者都返回稀疏矩阵。对于10万文档，密集版本无法装入内存；请保持稀疏，直到分类器要求密集。

改变一切的参数：

|  参数  |  效果  |
|-----|--------|
|  `ngram_range=(1, 2)`  |  包含二元词组。通常提升分类效果。  |
|  `min_df=2`  |  丢弃出现在少于2个文档中的词。在噪声数据上裁剪词汇表。  |
|  `max_df=0.95`  |  丢弃出现在超过95%文档中的词。近似于硬编码停用词列表的去除效果。  |
|  `stop_words="english"`  |  scikit-learn内置停用词列表。任务相关——情感分析不应丢弃否定词。  |
|  `sublinear_tf=True`  |  使用`1 + log(tf)`代替原始`tf`。当单词在单个文档中重复多次时有用。  |

### TF-IDF仍然胜出的场景（截至2026年）

- 垃圾邮件检测、主题标注、日志异常标记。单词的出现与否至关重要；语义上的细微差别并不重要。
- 低数据量场景（数百个标注样本）。TF-IDF加逻辑回归没有预训练成本。
- 任何对延迟敏感的地方。TF-IDF加线性模型可在微秒级响应。通过Transformer嵌入文档需要10-100毫秒。
- 必须解释其预测的系统。检查分类器的系数。权重最高的正面词汇就是原因。

### 当TF-IDF失效时

语义盲点失效。考虑以下两个文档：

- "这部电影一点也不好看。"
- "这部电影太棒了。"

一个是负面评论，一个是正面评论。它们的TF-IDF重叠度恰好是`{the, movie, was}`。词袋分类器必须记住，单词`not`出现在`good`附近会翻转标签。在足够多的数据上它可以学会这一点，但永远比不上一个理解句法结构的模型那么优雅。

另一个失效点：推理时的未登录词。一个在IMDb评论上训练的词袋模型，如果`Zoomer-approved`这个token从未在训练中出现过，它就会不知所措。子词嵌入（第04课）可以处理这种情况，而TF-IDF不行。

### 混合方法：TF-IDF加权嵌入

对于中等数据量分类任务，2026年的实用默认方案是：使用TF-IDF权重作为对词嵌入的注意力机制。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

你从嵌入中获得语义能力，从TF-IDF中获得对罕见词的强调。分类器在池化向量上训练。在标记样本少于约5万的情感、主题和意图分类任务中，这种方法优于单独使用任何一种。

## 发布

保存为 `outputs/prompt-vectorization-picker.md`：

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## 练习

1. **简单.** 在L2归一化的TF-IDF输出上实现`cosine_similarity(doc_vec_a, doc_vec_b)`。验证相同的文档得分为1.0，词汇不重叠的文档得分为0.0。
2. **中等.** 为`n-gram`添加`cosine_similarity(doc_vec_a, doc_vec_b)`支持。参数`bag_of_words`生成`n`-gram的计数。测试在`n=2`上应用`n`是否能为`["the", "cat", "sat"]`生成二元组计数。
3. **困难.** 使用GloVe 100维向量（下载一次并缓存）构建上述TF-IDF加权嵌入混合模型。在20 Newsgroups数据集上，比较其分类准确率与纯TF-IDF和纯均值池化嵌入。报告哪种方法在哪些任务上胜出。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  BoW  |  词频向量  |  一个文档中词汇表中单词的计数。忽略顺序。 |
|  TF  |  词频  |  一个单词在文档中的计数，可选地按文档长度归一化。 |
|  DF  |  文档频率  |  至少包含该单词一次的文档计数。 |
|  IDF  |  逆文档频率  |  `log(N / df)`平滑。降低出现频率高的单词的权重。 |
|  稀疏向量  |  大部分为零  |  词汇表通常包含1万到10万个单词；大多数单词在任意给定文档中都不存在。 |
|  余弦相似度  |  向量夹角  |  L2归一化向量的点积。1表示相同，0表示正交。 |

## 延伸阅读

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — 标准API参考，以及每个参数的说明。
- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — 使TF-IDF成为十年默认选择的那篇论文。
- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — 2026年对旧方法何时胜出及其原因的见解。
