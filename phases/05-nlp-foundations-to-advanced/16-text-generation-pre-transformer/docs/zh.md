# Transformer之前的文本生成——N-gram语言模型

> 如果一个词令人惊讶，那么模型就很差。困惑度(Perplexity)将惊讶量化。平滑(Smoothing)使其保持有限。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段5·01（文本处理），阶段2·14（朴素贝叶斯）
**时间：** 约45分钟

## 问题

在Transformer、RNN、词嵌入(Word Embedding)出现之前，语言模型通过统计一个词出现在前`n-1`个词之后的频率来预测下一个词。统计“the cat”后接“sat”47次，“the cat”后接“jumped”12次，“the cat”后接“refrigerator”0次。归一化后得到概率分布。

这就是N-gram语言模型。从1980年到2015年，它支撑着每一个语音识别器、拼写检查器和基于短语的机器翻译系统。当需要廉价的设备端语言建模时，它仍然被使用。

有趣的问题是如何处理未见过的N-gram。基于原始计数的模型会将任何未见过的序列概率设为零，这是灾难性的，因为句子很长，几乎每个长句子都包含至少一个未见过的序列。五十年的平滑研究解决了这个问题。Kneser-Ney平滑(Kneser-Ney smoothing)是其成果，现代深度学习继承了它的经验传统。

## 核心概念

![N-gram model: count, smooth, generate](../assets/ngram.svg)

**N-gram概率：** `P(w_i | w_{i-n+1}, ..., w_{i-1})`。固定`n`（通常三克(trigrams)取3，四克(4-grams)取4）。根据计数计算：

```text
P(w | context) = count(context, w) / count(context)
```

**零计数问题。** 任何在训练中未出现的N-gram概率为零。2007年一项关于Brown语料库的研究发现，即使是4-gram模型，也有30%的保留4-gram在训练中未见。没有平滑，无法在真实文本上进行评估。

**平滑方法，按复杂度排序：**

1. **拉普拉斯(Laplace，加一平滑)。** 每个计数加1。简单，但对罕见事件效果差。
2. **Good-Turing。** 根据频次-频次分布，将高频事件的概率质量重新分配给未见事件。
3. **插值(Interpolation)。** 结合N-gram、(N-1)-gram等估计，使用可调权重。
4. **回退(Backoff)。** 如果N-gram计数为零，则回退到(N-1)-gram。Katz回退对此进行归一化。
5. **绝对折扣(Absolute discounting)。** 从所有计数中减去固定折扣`D`，重新分配给未见事件。
6. **Kneser-Ney。** 绝对折扣加上对低阶模型的巧妙选择：使用*延续概率(Continuation probability)*（一个词出现在多少种上下文中）而非原始频率。

Kneser-Ney的洞察深刻。“San Francisco”是一个常见的二元组。一元组“Francisco”大多出现在“San”之后。朴素的绝对折扣会赋予“Francisco”很高的单字概率（因为计数高）。Kneser-Ney注意到“Francisco”只出现在一种上下文中，因此降低了它的延续概率。结果：一个以“Francisco”结尾的新二元组获得适当的低概率。

**评估：困惑度。** 在保留测试集上每个词的平均负对数似然的指数。越低越好。困惑度为100意味着模型在从100个词中均匀选择时一样困惑。

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

```figure
ngram-backoff
```

## 动手构建

### 步骤1：三克计数

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

输入是分词后的句子列表。输出是N-gram计数和上下文计数。`<s>`和`</s>`是句子边界。

### 步骤2：拉普拉斯平滑

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

每个计数加1。平滑但给未见事件分配过多概率质量，也会损害已知的罕见事件。

### 步骤3：Kneser-Ney（二元组，插值）

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

三个活动部分。`continuation_prob`捕捉“这个词出现在多少种不同的上下文中？”（Kneser-Ney的创新）。`lambda_prev`是折扣释放的质量，用于加权回退。最终概率是折扣后的主要项加上加权的延续项。

### 步骤4：通过采样生成文本

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

按概率比例采样。每次使用不同种子得到不同输出。对于类似beam-search的输出，每一步选argmax（贪婪）并添加小的随机性旋钮（温度(temperature)）。

### 步骤5：困惑度

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

越低越好。对于Brown语料库，一个调优好的4-gram KN模型困惑度约为140。一个Transformer语言模型在同一测试集上可达15-30。差距约10倍。这就是该领域向前发展的原因。

## 使用它

- **经典NLP教学。** 你能得到的最清晰的平滑、最大似然估计和困惑度的讲解。
- **KenLM。** 生产级N-gram库。用于需要低延迟的语音和机器翻译系统中的重新评分器。
- **设备端自动补全。** 键盘中的三克模型。至今仍在用。
- **基线。** 在宣称你的神经语言模型优秀之前，总是先计算一个N-gram语言模型的困惑度。如果你的Transformer不能大幅超越KN，那肯定有问题。

## 发布

保存为 `outputs/prompt-lm-baseline.md`：

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## 练习

1. **简单。** 在1000句莎士比亚语料上训练一个三克语言模型。生成20个句子。它们会在局部合理但全局不连贯。这是经典演示。
2. **中等。** 在保留的莎士比亚测试集上实现你的KN模型的困惑度，与拉普拉斯对比。你应该看到KN困惑度低30-50%。
3. **困难。** 构建一个三克拼写校正器：给定一个拼写错误的词及其上下文，生成候选词并按该上下文下的语言模型概率排序。在Birkbeck拼写语料库（公开）上评估。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  N-gram  |  词序列  |  连续的`n`个token构成的序列。  |
|  平滑  |  避免零概率  |  重新分配概率质量，使未见事件获得非零概率。  |
|  困惑度  |  语言模型质量指标  |  在保留数据上的`exp(-average log-prob)`。越低越好。  |
|  回退  |  回退到更短上下文  |  如果三克计数为零，则使用二元组。Katz回退将其形式化。  |
|  Kneser-Ney  |  最佳的N-gram平滑  |  绝对折扣 + 低阶模型的延续概率。  |
|  延续概率  |  Kneser-Ney特有  |  `P(w)`按上下文数量`w`加权，而非按原始计数。  |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf)——N-gram语言模型和平滑的经典论述。
- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf)——确定Kneser-Ney为最佳N-gram平滑器的论文。
- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf)——原始KN论文。
- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf)——快速的生产级N-gram语言模型，在2026年仍用于延迟敏感的应用。
