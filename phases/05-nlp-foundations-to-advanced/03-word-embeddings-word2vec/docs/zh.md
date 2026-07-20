# 词嵌入 — 从零实现 Word2Vec

> 一个词由其周围的词决定。训练一个浅层网络基于这个想法，几何关系自然浮现。

**类型：** 构建
**语言：** Python
**先修要求：** 阶段5·02 (词袋模型+TF-IDF), 阶段3·03 (从零实现反向传播)
**时间：** 约75分钟

## 问题

TF-IDF知道`dog`和`puppy`是不同的词。但它不知道它们几乎表示相同的意思。一个在`dog`上训练的分类器无法泛化到关于`puppy`的评论。你可以通过列出同义词来掩盖这一点，但这在稀有术语、领域术语和你未预料到的每种语言上都会失败。

你想要一个表示，其中`dog`和`puppy`在空间中靠近。`king - man + woman`靠近`queen`。一个在`dog`上训练的模型可以免费将一些信号迁移到`puppy`。

Word2Vec给了我们那个空间。两层神经网络，万亿token的训练，发表于2013年。这个架构简单得令人尴尬。其结果重塑了NLP达十年之久。

## 核心概念

**分布假说** (Firth, 1957): “一个词由其周围的词决定。”如果两个词出现在相似的上下文中，它们可能意味着相似的东西。

Word2Vec有两种变体，都利用了那个想法。

- **Skip-gram.** 给定中心词，预测周围的词。窗口大小为2的`cat -> (the, sat, on)`。
- **CBOW（连续词袋模型）。** 给定周围的词，预测中心词。`cat -> (the, sat, on)`。

Skip-gram训练较慢但处理稀有词更好。它成为了默认选择。

该网络有一个无非线性的隐藏层。输入是词表上的one-hot向量。输出是词表上的softmax。训练后，你丢弃输出层。隐藏层的权重就是嵌入向量。

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

技巧：对10万个词做softmax代价高昂。Word2Vec使用**负采样**将其转化为二分类任务。预测“这个上下文词是否出现在这个中心词附近，是或否”。每个训练对采样几个负（非共现）词，而不是计算整个词表的softmax。

```figure
word-vector-arithmetic
```

## 动手构建

### 步骤1：从语料库中生成训练对

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

窗口中的每个（中心词，上下文词）对都是一个正训练样本。

### 步骤2：嵌入表

两个矩阵。`W`是中心词嵌入表（你保留的那个）。`W'`是上下文词表（通常丢弃，有时与`W`平均）。

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

小随机初始化。词表大小10k、维度100是现实的；用于教学，50词表x16维足以看到几何结构。

### 步骤3：负采样目标

对于每个正对`(center, context)`，从词表中采样`k`个随机词作为负样本。训练模型使得点积`W[center] · W'[context]`对于正样本高，对于负样本低。

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

神奇公式：正对上的逻辑损失（希望sigmoid接近1）加上负对上的逻辑损失（希望sigmoid接近0）。梯度流向两个表。完整推导在原始论文中；如果你想记住它，用纸笔走一遍。

### 步骤4：在玩具语料库上训练

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

在大语料库上经过足够多的轮次后，共享上下文的词具有相似的中心嵌入。在玩具语料库上，你会隐约看到效果。在数十亿token上，效果显著。

### 步骤5：类比技巧

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

在预训练的300维Google News向量上：

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`。不是因为模型知道什么是皇室。而是因为向量`(king - man)`捕捉到了类似“皇室的”的东西，将其加到`woman`上会落在皇室-女性区域附近。

## 使用它

从零编写Word2Vec是教学。生产级NLP使用`gensim`。

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

对于实际工作，你几乎从不自己训练Word2Vec。你会下载预训练向量。

- **GloVe** — 斯坦福的共现矩阵分解方法。50d、100d、200d、300d检查点。覆盖广泛。第04课专门介绍GloVe。
- **fastText** — Facebook的Word2Vec扩展，嵌入字符n-gram。通过组合子词处理词汇表外的词。第04课。
- **在Google News上预训练的Word2Vec** — 300d，300万词词汇，2013年发布。至今仍每日下载。

### 当Word2Vec在2026年仍然胜出时

- 轻量级领域特定检索。在笔记本电脑上一小时内在医学摘要上训练，获得通用模型无法捕捉的专门向量。
- 类比式特征工程。`gender_vector = mean(man - woman pairs)`。将其从其他词中减去得到性别中立的轴。仍在公平性研究中使用。
- 可解释性。100维足够小，可以通过PCA或t-SNE绘图，实际看到聚类形成。
- 任何需要在无GPU设备上运行推理的地方。Word2Vec查找是一行读取。

### Word2Vec的失败之处

多义词问题。`bank` 只有一个向量。`river bank` 和 `financial bank` 共用它。`table`（电子表格 vs. 家具）共用它。下游分类器无法从向量中区分不同含义。

上下文嵌入（ELMo、BERT以及此后的所有Transformer）通过根据周围上下文为单词的每次出现生成不同向量解决了这个问题。这就是从Word2Vec到BERT的飞跃：从静态到上下文。第7阶段涵盖Transformer部分。

词汇外问题（out-of-vocabulary）是另一个失败。如果`Zoomer-approved`不在训练数据中，Word2Vec从未见过。没有后备方案。fastText通过子词组合（第04课）解决了这个问题。

## 发布

保存为 `outputs/skill-embedding-probe.md`：

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## 练习

1. **简单。**在小型语料库（20个关于猫和狗的句子）上运行训练循环。经过200轮后，验证`nearest(vocab, W, W[vocab["cat"]])`在其前3个结果中返回`dog`。如果没有，增加轮数或词汇量。
2. **中等。**添加高频词子采样。频率高于`nearest(vocab, W, W[vocab["cat"]])`的词以与其频率成比例的概率从训练对中丢弃。衡量对稀有词相似度的影响。
3. **困难。**在20个新闻组语料库上训练模型。计算两个偏差轴：`nearest(vocab, W, W[vocab["cat"]])`和`dog`。将职业词投影到两个轴上。报告哪些职业的偏差差距最大。这是研究人员使用的探针公平性类型。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  词嵌入(Word embedding)  |  词作为向量  |  从上下文中学习到的稠密、低维（通常100-300）表示。 |
|  Skip-gram  |  Word2Vec技巧  |  从中心词预测上下文词。比CBOW慢，但更适合稀有词。 |
|  负采样(Negative sampling)  |  训练捷径  |  用对`k`个随机词的二分类替换对整个词汇表的softmax。 |
|  静态嵌入(Static embedding)  |  每个词一个向量  |  无论上下文如何，向量相同。在多义词问题上失败。 |
|  上下文嵌入(Contextual embedding)  |  上下文敏感向量  |  根据周围词，每次出现生成不同向量。Transformer所产生。 |
|  OOV  |  词汇外(Out of vocabulary)  |  训练中未出现的词。Word2Vec无法为这些词生成向量。 |

## 延伸阅读

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)——负采样论文。简短易读。
- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)——梯度最清晰的推导，如果原论文的数学显得密集。
- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)——实际有效的生产训练设置。
