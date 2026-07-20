# 词性标注与句法分析

> 语法曾一度不受欢迎。后来每个大语言模型(LLM)流水线都需要验证结构化提取，它便重新流行起来。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段5·01（文本处理），阶段2·14（朴素贝叶斯）
**时间：** 约45分钟

## 问题

第01课曾承诺词形还原需要词性标注(part-of-speech tag)。若不知道`running`是动词，词形还原器就无法将其还原为`run`。若不知道`better`是形容词，就无法还原为`good`。

这个承诺背后隐藏着一个完整的子领域。词性标注(Part-of-speech tagging)分配语法类别。句法分析(Syntactic parsing)恢复句子的树状结构：哪个词修饰哪个词，哪个动词支配哪些论元。经典自然语言处理(NLP)花了二十年时间完善这两项技术。随后深度学习将它们简化为基于预训练Transformer的词元分类任务，研究社区便转向了其他方向。

但应用社区并非如此。每个结构化提取流水线底层仍在使用词性标注(POS)和依存树(Dependency trees)。大语言模型(LLM)生成的JSON需依据语法约束进行验证。问答系统利用依存句法分析分解查询。机器翻译质量评估器检查语法树的对应关系。

值得了解。本课介绍标记集、基线方法，以及何时你应停止从零实现并调用spaCy。

## 核心概念

**词性标注(POS tagging)** 为每个词元(token)分配一个语法类别。**宾州树库(Penn Treebank, PTB)** 标记集是英语的默认标记集。包含36个标签，其区分在普通读者看来颇为繁琐：`NN` 单数名词，`NNS` 复数名词，`NNP` 专有名词单数，`VBD` 动词过去时，`VBZ` 动词第三人称单数现在时，等等。**通用依存关系(Universal Dependencies, UD)** 标记集更粗略（17个标签），且与语言无关，已成为跨语言工作的默认选择。

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**句法分析(Syntactic parsing)** 生成一棵树。两种主要风格：

- **成分句法分析(Constituency parsing)。** 名词短语、动词短语、介词短语层层嵌套。输出是一棵以非终结符类别（NP、VP、PP）为内部节点、以单词为叶节点的树。
- **依存句法分析(Dependency parsing)。** 每个单词有一个它依赖的父词(head word)，并用语法关系标注。输出是一棵树，其中每条边是一个（父词，子词，关系）三元组。

依存句法分析在2010年代胜出，因为它跨语言泛化能力强，尤其适用于自由语序的语言。

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## 动手构建

### 步骤1：最常见标签基线

这是最简单但有效的词性标注器。对于每个单词，预测其在训练集中出现频率最高的标签。

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

在Brown语料库上，该基线达到约85%的准确率。不算好，但这是任何严谨模型不应低于的下限。

### 步骤2：双元隐马尔可夫模型(Bigram HMM)标注器

对序列的联合概率建模：

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

两个表：转移概率（给定前一个标签下的标签概率），发射概率（给定标签下的单词概率）。两者均通过计数并应用拉普拉斯平滑(Laplace smoothing)进行估计。使用维特比算法(Viterbi)（对标签网格进行动态规划）解码。

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

在Brown语料库上，双元HMM达到约93%的准确率。从85%到93%的提升主要得益于转移概率——模型学到了`DET NOUN`是常见的，而`NOUN DET`是罕见的。

### 步骤3：现代标注器为何能超越此方法

转移+发射概率是局部的。它们无法捕捉到`saw`在"I bought a saw"中是名词，而在"I saw the movie"中却是动词。一个具有任意特征（后缀、词形、前后词、词本身）的条件随机场(CRF)能达到约97%的准确率。双向LSTM-CRF(BiLSTM-CRF)或Transformer则可达到98%以上。

这项任务的上限由标注者之间的不一致性决定。在宾州树库(Penn Treebank)上，人类标注者的一致性约为97%。超过98%的模型可能是在测试集上过拟合了。

### 步骤4：依存句法分析概述

从零实现完整的依存句法分析超出了本课范围；经典的教材处理参见Jurafsky和Martin的著作。需了解两个经典家族：

- **基于转移(Transition-based)** 的解析器（arc-eager、arc-standard）类似于移进-归约(shift-reduce)解析器：它读取词元，将其压入栈中，然后执行创建弧(arc)的归约动作。贪心解码速度快。经典实现是MaltParser。现代神经网络版本：Chen和Manning的基于转移的解析器。
- **基于图(Graph-based)** 的解析器（Eisner算法、Dozat-Manning双仿射）为每个可能的父子边评分，并选择最大生成树。速度较慢但更准确。

对于大多数应用工作，调用spaCy：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

从下往上阅读`dep`列，句子的语法结构便一目了然。

## 使用它

每个生产级自然语言处理(NLP)库都将词性标注和依存句法分析器作为标准流水线的一部分提供。

- **spaCy** (`en_core_web_sm` / `md` / `lg` / `trf`)。速度快、准确率高，与分词(Tokenization)、命名实体识别(NER)和词形还原(Lemmatization)集成在一起。`token.tag_` (Penn)，`token.pos_` (UD)，`token.dep_` (依存关系)。
- **Stanford NLP (stanza)**。斯坦福大学CoreNLP的继任者。在60多种语言上达到最优水平。
- **trankit**。基于Transformer，在UD上准确率高。
- **NLTK**。`en_core_web_sm`。可用，但速度慢，较陈旧。适合教学。

### 2026年这些技术为何仍然重要

- **词形还原(Lemmatization)。** 第01课需要词性标注(POS)才能正确进行词形还原。这一点始终成立。
- **从大语言模型(LLM)输出中进行结构化提取。** 验证生成的句子是否遵守语法约束（例如主谓一致、必需的修饰语）。
- **基于方面的情感分析(Aspect-based sentiment)。** 依存句法分析告诉你哪个形容词修饰哪个名词。
- **查询理解(Query understanding)。** “movies directed by Wes Anderson starring Bill Murray”通过句法分析分解为结构化约束。
- **跨语言迁移(Cross-lingual transfer)。** 通用依存关系(UD)标签和依存关系与语言无关，因此能够对新语言进行零样本结构化分析。
- **低计算量流水线。** 如果你无法部署Transformer，词性标注+依存句法分析+地名词典(gazetteer)就能带来令人惊讶的效果。

## 发布

保存为 `outputs/skill-grammar-pipeline.md`：

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## 练习

1. **简单.** 使用最频繁标签基线在小型标注语料库（例如NLTK的Brown子集）上，衡量保留句子上的准确率。验证约85%的结果。
2. **中等.** 训练上述二元隐马尔可夫模型(HMM)，并报告每个标签的精确率/召回率。隐马尔可夫模型最容易混淆哪些标签？
3. **困难.** 使用spaCy的依存句法分析从1000句样本中提取主谓宾三元组。在50个手动标注的三元组上评估。记录提取失败的情况（通常是被动语态、并列结构和省略主语）。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  词性标注  |  词的类别  |  语法范畴。宾州树库(PTB)有36个；通用依存关系(UD)有17个。  |
|  宾州树库(PTB)  |  标准标签集  |  英语专用。细粒度的动词时态和名词数。  |
|  通用依存关系(UD)  |  多语言标签集  |  比宾州树库(PTB)更粗；语言中立；跨语言工作的默认选项。  |
|  依存句法分析  |  句子树  |  每个词有一个中心词，每条边有一个语法关系。  |
|  维特比算法  |  动态规划  |  在给定发射概率和转移概率下，找到概率最大的标签序列。  |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — 关于词性标注和句法分析的权威教材。
- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — 每个多语言解析器使用的跨语言标签集和树库集合。
- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — 关于[Universal Dependencies project](https://universaldependencies.org/)所暴露的每个属性的实用参考。
- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — 将神经解析器带入主流的论文。
