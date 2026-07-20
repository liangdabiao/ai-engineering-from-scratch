# GloVe、FastText 与子词嵌入

> Word2Vec为每个词训练了一个嵌入。GloVe直接分解了共现矩阵。FastText对子词进行了嵌入。BPE架起了通往Transformer的桥梁。

**类型：** 构建
**语言：** Python
**前置知识：** 第5阶段·03（从零实现Word2Vec）
**时间：** 约45分钟

## 问题

Word2Vec留下了两个悬而未决的问题。

首先，存在一条平行研究路线，直接分解共现矩阵（LSA、HAL），而不是进行在线skip-gram更新。Word2Vec的迭代方法本质上更好吗？还是差异只是两种方法处理计数方式的人为产物？**GloVe**给出了答案：通过精心选择的损失函数进行矩阵分解，效果可与Word2Vec媲美甚至更优，且训练成本更低。

其次，两种方法都无法处理未见过的词。`Zoomer-approved`、`dogecoin`、任何上周新造的专有名词、以及每个罕见词根的屈折形式。**FastText**通过嵌入字符n-gram解决了这个问题：一个词是其组成部分（包括词素）之和，因此即使词汇表外的词也能得到合理的向量。

第三，Transformer出现后，问题再次转变。词级词汇表最多约一百万个条目；真实语言远不止于此。**字节对编码（BPE）**及其变体通过学习覆盖所有内容的常见子词单元词汇表解决了这个问题。每个现代大型语言模型的现代分词器都是子词分词器。

本节课将逐一介绍这三种方法，然后解释在何种情况下应选用哪一种。

## 核心概念

**GloVe（全局向量）。** 构建词-词共现矩阵`X`，其中`X[i][j]`表示词`j`出现在词`i`上下文中的频率。训练向量使得`v_i · v_j + b_i + b_j ≈ log(X[i][j])`。对损失进行加权，使频繁共现的词对不占主导地位。完成。

**FastText。** 一个词是其字符n-gram加上该词本身之和。`where`变为`<wh, whe, her, ere, re>, <where>`。词向量是这些分量向量的和。像Word2Vec一样训练。优点：未见过的词（`whereupon`）可由已知的n-gram组合而成。

**BPE（字节对编码）。** 从单个字节（或字符）的词汇表开始。统计语料库中每对相邻对的出现次数。将最频繁的对合并为一个新token。重复执行`k`次迭代。结果：得到一个包含`k + 256`个token的词汇表，其中常见序列（`ing`、`tion`、`the`）是单个token，罕见词则被分解为熟悉的部分。每个句子都能被分词为某种形式。

## 动手构建

### GloVe：分解共现矩阵

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

有两个值得命名的移动部件。权重函数`f(x) = (x/x_max)^alpha`降低非常频繁的共现对（如`(the, and)`）的权重，以免它们主导损失。最终嵌入是`W`（中心词）和`W_tilde`（上下文）表之和。将两者相加是已发表的技巧，通常优于仅使用其中之一。

### FastText：子词感知的嵌入

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

每个词由一组n-gram（通常为3到6个字符）表示。词嵌入是其所有n-gram嵌入之和。对于skip-gram训练，将其替换Word2Vec中使用的单一向量。

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

对于未见过的词，只要其某些n-gram已知，你仍然可以得到一个向量。`whereupon`与`<wh`、`her`、`ere`和`<where`共享，因此两者在向量空间中彼此接近。

### BPE：学习的子词词汇表

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

第一次迭代合并最常见的相邻对。经过足够多的迭代后，频繁的子串（`low`、`est`、`tion`）成为单个token，而罕见词则被清晰地分解。

真正的GPT/BERT/T5分词器会学习3万到10万个合并。结果：任何文本都能被分词成一个长度有限的已知ID序列，永远不会有OOV。

## 使用它

在实践中，你很少自己训练这些。而是加载预训练的检查点。

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

对于Transformer时代的BPE风格子词分词：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ`前缀标记词边界（GPT-2的惯例）。每个现代分词器都是BPE变体、WordPiece（BERT）或SentencePiece（T5、LLaMA）。

### 如何选择

|  情况  |  选择  |
|-----------|------|
|  预训练的通用词向量，无需OOV容忍度  |  GloVe 300d  |
|  预训练的通用词向量，必须处理拼写错误/新词/形态丰富的语言  |  FastText  |
|  任何进入Transformer的数据（训练或推理）  |  随模型提供的任何分词器。切勿更换。  |
|  从头训练自己的语言模型  |  首先在自己的语料库上训练BPE或SentencePiece分词器  |
|  使用线性模型的生产级文本分类  |  仍然是TF-IDF。第02课。  |

## 发布

保存为 `outputs/skill-embeddings-picker.md`：

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## 练习

1. **简单。** 运行`char_ngrams("playing")`和`char_ngrams("played")`。计算两个n-gram集合的Jaccard重叠度。你应该会看到大量共享部分（`pla`、`lay`、`play`），这就是FastText在形态变体之间迁移良好的原因。
2. **中等。** 扩展`char_ngrams("playing")`以跟踪词汇表增长。绘制每个语料库字符的token数作为合并次数的函数。你应该会看到初始快速压缩，然后渐近接近每token约2-3个字符。
3. **困难。** 在莎士比亚全集上训练一个包含1000次合并的BPE。比较常见词和罕见专有名词的分词结果。测量每个词分得的平均token数在分词前后的变化。写下让你惊讶的地方。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  共现矩阵  |  词-词频率表  |  `X[i][j]` = 词`j`出现在词`i`周围窗口中的频率。  |
|  子词  |  词的一部分  |  一个字符n-gram（FastText）或学习的token（BPE/WordPiece/SentencePiece）。  |
|  BPE  |  字节对编码  |  迭代合并最频繁的相邻对，直到词汇表达到目标大小。  |
|  OOV  |  词汇外(Out of vocabulary)  |  模型从未见过的词。Word2Vec/GloVe会失效。FastText和BPE可以处理。  |
|  字节级BPE(Byte-level BPE)  |  基于原始字节的BPE  |  GPT-2的方案。词汇表从256个字节开始，因此不会有任何词是OOV。  |

## 延伸阅读

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — GloVe论文，七页，仍然是对损失函数的最佳推导。
- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — FastText。
- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — 将BPE引入现代NLP的论文。
- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — BPE、WordPiece和SentencePiece在实际中究竟有何不同。
