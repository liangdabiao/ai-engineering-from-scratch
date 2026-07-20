# 情感分析

> 标准的NLP任务。经典文本分类所需的大部分知识都在这里体现。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段5·02（词袋+TF-IDF），阶段2·14（朴素贝叶斯）
**时间：** 约75分钟

## 问题

"食物不太棒。" 正面还是负面？

情感分析听起来很简单。评论者说他们喜欢或不喜欢某物。给句子打标签。它之所以成为标准的NLP任务，是因为每个看似简单的案例背后都隐藏着一个难题。否定会翻转含义。讽刺会颠倒它。"一点都不差"虽然是正面，但包含两个负面词汇。表情符号比周围文本携带更多信号。领域词汇很重要（音乐评论中的`tight`与时尚评论中的`tight`不同）。

情感分析是经典NLP的实战实验室。如果你理解为什么每个朴素的基线都有特定的失败模式，你就理解为什么每个更丰富的模型被发明出来。本课从头构建了一个朴素贝叶斯基线，添加了逻辑回归，并指出了那些使生产环境下的情感分析成为合规级别问题的陷阱。

## 核心概念

经典情感分析是一个两步配方。

1. **表示。** 将文本转换为特征向量。词袋、TF-IDF或n-gram。
2. **分类。** 在带标签的示例上拟合线性模型（朴素贝叶斯、逻辑回归、支持向量机）。

朴素贝叶斯是能工作的最简单的模型。假设在给定标签的情况下每个特征都是独立的。从计数中估计`P(word | positive)`和`P(word | negative)`。在推理时，将概率相乘。这种"朴素"的独立性假设错得可笑，但结果却惊人地强大。原因在于：对于稀疏的文本特征和中等数据量，分类器更关心每个单词倾向于哪一边，而不是程度如何。

逻辑回归修复了独立性假设。它为每个特征学习一个权重，包括负权重。作为双词特征的`not good`得到负权重。朴素贝叶斯无法对从未标记过的双词做到这一点。

```figure
sentiment-logits
```

## 动手构建

### 步骤1：一个真实的小型数据集

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

刻意很小。实际工作使用数万条示例（IMDb、SST-2、Yelp极性）。数学原理是相同的。

### 步骤2：从头实现多项式朴素贝叶斯

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

加法平滑（alpha=1.0）是拉普拉斯平滑。没有它，一个在某个类中未出现的单词概率为零，导致对数爆炸。`alpha=0.01`在实践中很常见。`alpha=1.0`是教学默认值。

### 步骤3：从头实现逻辑回归

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2正则化在这里很重要。文本特征是稀疏的；没有L2，模型会记忆训练示例。从`0.01`开始并调整。

### 步骤4：处理否定（失败模式）

考虑"不好"和"不错"。词袋分类器看到`{not, good}`和`{not, bad}`，并从训练中出现更多的那个中学习。双词分类器看到`not_good`和`not_bad`，并将它们作为不同特征学习。这通常就足够了。

一个更粗暴但有效的修复方法，当你没有双词时：**否定范围**。将否定词之后的词元加上前缀`NOT_`，直到下一个标点符号。

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

现在`good`和`NOT_good`是不同的特征。分类器可以对它们赋予相反的权重。三行预处理，情感基准测试上准确率明显提升。

### 步骤5：重要的评估指标

如果类别不平衡，仅靠准确率会产生误导。真实的情感语料库通常70-80%是正面或70-80%是负面；一个常预测多数类的分类器获得80%的准确率，但毫无价值。报告以下每一项：

- **每个类别的精确率和召回率。** 每个类别一对。宏平均它们得到一个尊重类别平衡的单一数字。
- **宏F1（不平衡数据的主要指标）。** 每个类别F1得分的均值，等权重。当类别不平衡时使用这个而不是准确率。
- **加权F1（备选）。** 与宏F1相同，但按类别频率加权。当不平衡本身具有业务意义时，与宏F1一起报告。
- **混淆矩阵。** 原始计数。在相信任何标量指标之前始终检查；它揭示了模型混淆了哪一对类别。
- **每个类别的错误样本。** 每个类别抽取5个错误预测。阅读它们。没有什么能替代阅读实际错误。

对于严重不平衡的数据（>95-5比例），报告**AUROC**和**AUPRC**而不是准确率。AUPRC对少数类更敏感，而这正是你通常关心的（垃圾邮件、欺诈、罕见情感）。

**常见要避免的错误。** 在不平衡数据上报告微F1而非宏F1会给出一个看起来很高的数字，因为它主要受多数类影响。宏F1迫使你看到少数类的表现。

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## 使用它

scikit-learn用六行代码正确完成。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

需要注意三件事。`stop_words=None`保留否定。`ngram_range=(1, 2)`添加双词，因此`not_good`成为一个特征。`sublinear_tf=True`抑制重复词。这三个标志是在SST-2上75%准确率基线与85%准确率基线之间的差异。

### 何时使用Transformer

- 讽刺检测。经典模型在这里会失败。句号。
- 长评论中情感在文档中途转变。
- 基于方面的情感。"相机很好但电池很差。"你需要将情感归因于方面。只有Transformer或结构化输出模型。
- 非英语、低资源语言。多语言BERT免费为你提供零样本基线。

如果你需要以上任何一项，直接跳到阶段7（Transformer深度剖析）。否则，基于TF-IDF加双词加否定处理的朴素贝叶斯或逻辑回归是你2026年的生产基线。

### 可重复性陷阱（再次）

重新训练情感模型是常规操作，但重新评估它们则不然。论文中报告的准确率数值使用了特定的划分、特定的预处理和特定的分词器。如果你没有使用完全相同的流水线就将新模型与基线进行比较，你会得到误导性的差异。始终要在你自己的流水线上重新生成基线，而不是使用论文中的数值。

## 发布

保存为 `outputs/prompt-sentiment-baseline.md`：

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## 练习

1. **简单.** 在scikit-learn流水线中添加`apply_negation`作为预处理步骤，并在一个小型情感数据集上测量F1差异。
2. **中等.** 实现类别加权逻辑回归（将`apply_negation`传递给scikit-learn，或者自己推导梯度）。在合成90-10类别不平衡数据上测量效果。
3. **困难.** 通过在情感模型的残差上训练第二个分类器来构建讽刺检测器。记录你的实验设置。当准确率低于随机水平时警告读者（二类讽刺的随机水平约为50%，大多数初次尝试都会落在这个水平）。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  极性  |  正面或负面  |  二分类标签；有时扩展到中性或细粒度（5星）。 |
|  基于方面的情感  |  每个方面的极性  |  将情感归因于文本中提到的特定实体或属性。 |
|  否定范围  |  反转附近的词元  |  在"not"之后的词元前加上`NOT_`，直到标点符号。 |
|  拉普拉斯平滑  |  计数加1  |  防止朴素贝叶斯中的零概率特征。 |
|  L2正则化  |  收缩权重  |  在损失中添加`lambda * sum(w^2)`。对稀疏文本特征至关重要。 |

## 延伸阅读

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — 基础性综述。篇幅较长，但前四节涵盖了所有经典内容。
- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — 该论文展示了二元组+朴素贝叶斯在短文本上很难被击败。
- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — 有关[Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/)、[scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)以及你将调节的每一个参数的参考。
