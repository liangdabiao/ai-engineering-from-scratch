# 子词分词 — BPE、WordPiece、Unigram、SentencePiece

> 词级分词器遇到未见过的词会卡住。字符级分词器会使序列长度膨胀。子词分词器则取折中方案。每个现代的大型语言模型都搭载一种子词分词器。

**类型:** 学习
**语言:** Python
**前置要求:** 阶段5·01 (文本处理), 阶段5·04 (GloVe / FastText / Subword)
**时长:** ~60分钟

## 问题

你的词汇表有50,000个词。用户输入了"untokenizable"。你的分词器返回了`[UNK]`。模型现在对该词没有任何信息。更糟的是：你语料库中第90百分位的文档有40个罕见词，这意味着每个文档丢失了40位信息。

子词分词解决了这个问题。常见词保持为单个词元。罕见词分解为有意义的片段：`untokenizable` → `un`, `token`, `izable`。训练数据覆盖所有情况，因为任何字符串归根结底都是字节序列。

2026年的每个前沿LLM都搭载三种算法之一（BPE、Unigram、WordPiece），并封装在三个库之一（tiktoken、SentencePiece、HF Tokenizers）中。不选择一个，你就无法发布语言模型。

## 核心概念

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE（字节对编码）。** 从字符级词汇表开始。统计每个相邻对。将最频繁的对合并成一个新词元。重复直到达到目标词汇量大小。主导算法：GPT-2/3/4、Llama、Gemma、Qwen2、Mistral。

**字节级BPE。** 相同算法但基于原始字节（256个基本词元）而非Unicode字符。保证零个`[UNK]`词元——任何字节序列都能编码。GPT-2使用50,257个词元（256个字节 + 50,000次合并 + 1个特殊词元）。

**Unigram。** 从一个庞大的词汇表开始。为每个词元分配一个Unigram概率。迭代剪枝那些移除后对语料库对数似然增加最小的词元。推理时是概率性的：可以对分词结果进行采样（通过子词正则化进行数据增强时很有用）。由T5、mBART、ALBERT、XLNet、Gemma使用。

**WordPiece。** 合并那些使训练语料库似然最大化而非原始频率的对。由BERT、DistilBERT、ELECTRA使用。

**SentencePiece vs tiktoken。** SentencePiece是一个直接在原始Unicode文本上*训练*词汇表（BPE或Unigram）的库，将空白编码为`▁`。tiktoken是OpenAI的快速*编码器*，针对预建词汇表；它不进行训练。

经验法则：

- **训练新词汇表：** SentencePiece（多语言，无需预分词）或HF Tokenizers。
- **针对GPT词汇表的快速推理：** tiktoken（cl100k_base, o200k_base）。
- **两者：** HF Tokenizers —— 一个库，训练+服务。

```figure
bpe-merge
```

## 动手构建

### 第一步：从零实现BPE

参见`code/main.py`。循环：

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

该算法编码的三个事实。`</w>`标记词尾，因此"low"（后缀）和"lower"（前缀）保持区分。频率加权使高频对早期获胜。合并列表是有序的——推理时按训练顺序应用合并。

### 第二步：使用学习到的合并进行编码

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

朴素实现为O(n·|merges|)。生产级实现（tiktoken、HF Tokenizers）使用带有优先级队列的合并排名查找，运行时间接近线性。

### 第三步：实践中的SentencePiece

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注意：无需预分词，空格编码为`▁`，`character_coverage`控制罕见字符是被保留还是映射到`<unk>`的激进程度。

### 第四步：用于OpenAI兼容词汇表的tiktoken

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

仅编码。快速（Rust后端）。与GPT-4/5的分词精确匹配，用于字节计数、成本估算、上下文窗口预算。

## 2026年仍存在的陷阱

- **分词器漂移。** 使用词汇表A训练，针对词汇表B部署。词元ID不同；模型输出垃圾。在CI中检查`tokenizer.json`哈希。
- **空白歧义。** BPE对"hello"和" hello"生成不同词元。始终明确指定`tokenizer.json`和`add_special_tokens`。
- **多语言训练不足。** 英语为主的语料库生成的词汇表会将非拉丁文字分成5-10倍多的词元。相同提示在GPT-3.5上日语/阿拉伯语的成本高出5-10倍。o200k_base部分修复了此问题。
- **表情符号分裂。** 单个表情符号可能占用5个词元。在上下文预算时检查表情符号处理。

## 使用它

2026年技术栈：

|  情况  |  选择  |
|-----------|------|
|  从头训练单语模型  |  HF Tokenizers (BPE)  |
|  训练多语言模型  |  SentencePiece (Unigram, `character_coverage=0.9995`)  |
|  服务OpenAI兼容API  |  tiktoken (`o200k_base` for GPT-4+)  |
|  领域特定词汇（代码、数学、蛋白质） | 在领域语料上训练自定义BPE，与基础词汇表合并  |
|  边缘推理，小模型 | Unigram（较小词汇表效果更好） |

词汇量大小是一个扩展决策，而不是常数。粗略启发式：<1B参数用32k，1-10B用50-100k，多语言/前沿用200k+。

## 发布

保存为 `outputs/skill-bpe-vs-wordpiece.md`：

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## 练习

1. **简单。** 在`code/main.py`的小语料上训练一个500合并的BPE（Byte-Pair Encoding）。对三个保留词进行编码。有多少词产生恰好1个令牌，多少个产生>1个令牌？
2. **中等。** 比较100个英文维基百科句子在`code/main.py`、`cl100k_base`以及你用vocab=32k训练的SentencePiece BPE上的令牌数。报告每种方法的压缩比。
3. **困难。** 用BPE、Unigram和WordPiece训练相同语料。在小情感分类器上使用每种方法时，测量下游准确率。选择是否会在F1上产生超过1个百分点的差异？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  BPE  |  字节对编码（Byte-Pair Encoding）  |  贪心合并最频繁字符对直到达到目标词汇量。 |
|  字节级BPE（Byte-level BPE）  |  永无未知令牌  |  在原始256字节上做BPE；GPT-2/Llama使用此方法。 |
|  Unigram  |  概率分词器（Probabilistic tokenizer）  |  使用对数似然从大型候选集中剪枝；T5、Gemma使用。 |
|  SentencePiece  |  处理空格的那个  |  在原始文本上训练BPE/Unigram的库；空格编码为`▁`。 |
|  tiktoken  |  快速的那个  |  OpenAI基于Rust的BPE编码器，用于预构建词汇表。不训练。 |
|  合并列表（Merge list）  |  魔法数字  |  有序的`(a, b) → ab`合并列表；推理时按顺序应用。 |
|  字符覆盖率（Character coverage）  |  多罕见算太罕见？  |  分词器必须覆盖的训练语料中字符的比例；典型值约0.9995。 |

## 延伸阅读

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)——BPE论文。
- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)——Unigram论文。
- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)——库。
- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)——简明参考。
- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)——指南+编码列表。
