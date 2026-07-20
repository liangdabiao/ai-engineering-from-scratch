# 文本处理 —— 分词、词干提取、词形还原

> 语言是连续的。模型是离散的。预处理是桥梁。

**类型：** 构建
**语言：** Python
**前提条件：** 阶段2 · 14（朴素贝叶斯）
**时间：** ~45分钟

## 问题

模型无法读取"The cats were running."它读取的是整数。

每个NLP系统都以相同的三个问题开始。单词从哪里开始。单词的词根是什么。我们如何将"run"、"running"、"ran"在有助于时视作相同，在不必要时视作不同。

分词出错，模型就会从垃圾中学习。如果你的分词器将`don't`视为一个词元而`do n't`视为两个，训练分布就会分裂。如果你的词干提取器将`organization`和`organ`合并为同一词干，主题建模就会失效。如果你的词形还原器需要词性上下文但你未传入，动词就会被当作名词处理。

本课从零构建三个预处理步骤，然后展示NLTK和spaCy如何完成相同工作，以便你看到权衡。

## 核心概念

三个操作。每个都有其职责和失败模式。

**分词(Tokenization)** 将字符串分割为词元。"词元"故意定义模糊，因为正确的粒度取决于任务。词级用于经典NLP。子词用于Transformer。字符级用于没有空格的语言。

**词干提取(Stemming)** 用规则去掉后缀。快速、激进、愚蠢。`running -> run`。`organization -> organ`。第二个就是失败模式。

**词形还原(Lemmatization)** 利用语法知识将单词还原为词典形式。较慢、准确，需要查找表或形态分析器。`ran -> run`（需要知道"ran"是"run"的过去式）。`better -> good`（需要知道比较级形式）。

经验法则。当速度重要且能容忍噪声（搜索索引、粗略分类）时使用词干提取。当意义重要（问答、语义搜索、任何用户会阅读的内容）时使用词形还原。

```figure
edit-distance
```

## 动手构建

### 步骤1：一个正则表达式词元分词器

最简单实用的分词器在非字母数字字符上分割，同时将标点保留为单独词元。不完美，不最终，但一行代码就能运行。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

三个模式按优先级排序。带可选内部撇号的单词（`don't`, `it's`）。纯数字。任何单个非空白、非字母数字字符作为独立词元（标点）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

需要注意的失败模式。`3pm` 分割成 `['3', 'pm']`，因为我们在字母序列和数字序列之间交替。对大多数任务来说足够好。URL、邮箱、话题标签全部失效。生产环境中，在通用模式之前添加模式。

### 步骤2：一个Porter词干提取器（仅步骤1a）

完整的Porter算法有五个阶段的规则。仅步骤1a就涵盖了最常见的英语后缀，并展示了模式。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

自上而下阅读规则。`ies -> i` 规则就是为什么 `ponies -> poni`，而不是 `pony`。真正的Porter有步骤1b来修复它。规则相互竞争。更早的规则获胜。顺序比任何单一规则更重要。

### 步骤3：基于查找的词形还原器

真正的词形还原需要形态学。一个适合教学的小版本使用一个小的词形表和回退机制。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最后一个案例是关键的教学时刻。`watched` 不在我们的表中，而我们的回退仅处理 `ing`。真正的词形还原涵盖 `ed`、不规则动词、比较级形容词、带音变的复数形式（`children -> child`）。这就是为什么生产系统使用WordNet、spaCy的形态分析器或完整的形态分析器。

### 步骤4：将它们串联起来

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺少的部分是词性标注器。阶段5 · 07（词性标注）会构建一个。现在，将所有默认设为 `NOUN` 并承认其局限性。

## 使用它

NLTK和spaCy提供了生产版本。每样只需几行。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` 处理缩写、Unicode、你的正则表达式遗漏的边缘情况。`PorterStemmer` 运行所有五个阶段。`WordNetLemmatizer` 需要将词性标签从NLTK的Penn Treebank方案转换为WordNet的缩写集。上面的转换连接代码是大多数教程忽略的部分。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy将整个流程隐藏在 `nlp(text)` 之后。分词、词性标注和词形还原全部运行。比NLTK更快，开箱即用更准确。权衡是你不能轻易替换单个组件。

### 如何选择

|  情况  |  选择  |
|-----------|------|
|  教学、研究、组件替换  |  NLTK  |
|  生产环境、多语言、速度重要  |  spaCy  |
|  Transformer 流水线（反正你会用模型的分词器进行分词）||| 使用 @@SKIP0000@@ / @@SKIP0001@@ 并跳过经典预处理 |  |

### 没人警告过你的两种失败模式

大多数教程只教算法就结束了。有两件事会咬住真实预处理流水线，而且几乎从未被覆盖。

**可重现性漂移。** NLTK 和 spaCy 在不同版本之间改变了分词和词形还原器的行为。在 spaCy 2.x 中产生 `['do', "n't"]` 的代码可能在 3.x 中产生 `["don't"]`。你的模型在一个分布上训练，推理却在另一个分布上运行。准确率悄悄下降，没人知道原因。在 `requirements.txt` 中固定库版本。编写一个预处理回归测试，冻结 20 个样本句子的预期分词。每次升级时运行它。

**训练/推理不匹配。** 使用激进预处理（小写化、停用词移除、词干提取）训练，部署到原始用户输入上，看着性能崩塌。这是生产环境中 NLP 最常见的失败。如果你在训练时预处理，你必须在推理时运行相同的函数。将预处理作为模型包内的一个函数交付，而不是作为服务团队重写的笔记本单元格。

## 发布

一个可复用的提示，帮助工程师在不阅读三本教科书的情况下选择预处理策略。

保存为 `outputs/prompt-preprocessing-advisor.md`：

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## 练习

1. **简单。** 扩展 `tokenize` 以将 URL 保留为单个词元。测试：`tokenize("Visit https://example.com today.")` 应产生一个 URL 词元。
2. **中等。** 实现 Porter 第 1b 步。如果一个单词包含元音并以 `tokenize` 或 `tokenize("Visit https://example.com today.")` 结尾，则移除它。处理双辅音规则（`ed`，而不是 `ing`）。
3. **困难。** 构建一个使用 WordNet 作为查找表，但在 WordNet 没有条目时回退到你的 Porter 词干提取器的词形还原器。在有标注的语料库上测量准确率，与纯 WordNet 和纯 Porter 比较。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  词元 | 一个词 | 模型消费的任何单位。可以是词、子词、字符或字节。  |
|  词干 | 词根 | 基于规则的后缀去除的结果。不总是一个真正的词。  |
|  词元 | 字典形式 | 你会查找的形式。需要语法上下文才能正确计算。  |
|  词性标注 | 词性 | 如 NOUN、VERB、ADJ 等类别。需要准确进行词形还原。  |
|  词法 | 词形规则 | 单词如何根据时态、数、格改变形式。词形还原依赖于它。  |

## 延伸阅读

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 原始论文，五页，仍然是最清晰的解释。
- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 真实流水线如何连接。
- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 你还没想到的分词边缘情况。
