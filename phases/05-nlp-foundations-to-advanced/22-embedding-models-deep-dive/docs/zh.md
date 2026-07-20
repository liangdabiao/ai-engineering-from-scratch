# 嵌入模型(Embedding Models)——2026深度解析

> Word2Vec为每个词提供了一个向量。现代嵌入模型(Embedding Models)则为每个段落提供一个向量，支持跨语言(Cross-lingual)，并提供稀疏(Sparse)、密集(Dense)和多向量(Multi-vector)表示，大小可根据你的索引调整。选错了，你的RAG就会检索到错误的内容。

**类型:** 学习
**语言:** Python
**先决条件:** 阶段5·03 (Word2Vec), 阶段5·14 (信息检索(Information Retrieval))
**时间:** 约60分钟

## 问题

你的RAG系统有40%的时间检索到错误的段落。罪魁祸首很少是向量数据库(Vector Database)或提示(Prompt)。而是嵌入模型(Embedding Model)。

在2026年选择嵌入(Embedding)意味着要在五个维度上权衡：

1. **密集(Dense) vs 稀疏(Sparse) vs 多向量(Multi-vector).** 每个段落一个向量，或每个词元(Token)一个向量，或一个稀疏加权词袋(Bag of Words)。
2. **语言覆盖(Language coverage).** 单语英语模型在纯英语任务上仍然胜出。多语言模型则在混合语料库中胜出。
3. **上下文长度(Context length).** 512词元(Tokens) vs 8,192 vs 32,768——实际有效容量通常为广告最大值的60-70%。
4. **维度预算(Dimension budget).** 全精度3,072个浮点数 = 每个向量12 KB。对于1亿个向量，存储费用为每月1300美元。Matryoshka截断(Matryoshka Truncation)可将此削减4倍。
5. **开源(Open) vs 托管(Hosted).** 开源权重意味着你控制完整的堆栈和数据。托管意味着你用控制权换取始终最新的版本。

本课程列出了这些权衡，以便你能基于证据选择，而不是基于上个季度流行的东西。

## 核心概念

![Dense, sparse, and multi-vector embeddings](../assets/embedding-modes.svg)

**密集嵌入(Dense Embeddings).** 每个段落一个向量（通常384-3,072维）。余弦相似度(Cosine Similarity)根据语义接近度对段落排序。OpenAI `text-embedding-3-large`, BGE-M3密集模式, Voyage-3。默认选择。

**稀疏嵌入(Sparse Embeddings).** SPLADE风格。一个变换器(Transformer)为每个词汇表中的词元预测一个权重，然后将大多数权重置零。结果是一个大小为|词汇表|的稀疏向量。捕获词汇匹配（如BM25），但带有学习到的术语权重。在关键词密集型查询中表现强劲。

**多向量(Multi-vector)（后期交互(Late Interaction)).** ColBERTv2, Jina-ColBERT。每个词元一个向量。使用MaxSim评分：对于每个查询词元，找到最相似的文档词元，求和得分。存储和评分成本更高，但在长查询和特定领域语料库上胜出。

**BGE-M3：三者合一。** 单个模型同时输出密集、稀疏和多向量表示。每个都可以独立查询；得分通过加权求和融合。当你想从一个检查点(Checkpoint)获得灵活性时，这是2026年的默认选择。

**Matryoshka表示学习(Matryoshka Representation Learning).** 训练方式使得向量的前N维形成一个有用的独立嵌入。将一个1,536维向量截断到256维，牺牲约1%的准确率换取6倍的存储节省。支持该技术的模型包括OpenAI text-3, Cohere v4, Voyage-4, Jina v5, Gemini Embedding 2, Nomic v1.5+。

### MTEB排行榜(Leaderboard)只反映了部分情况

大规模文本嵌入基准(Massive Text Embedding Benchmark)——发布时（2022年）包含8种任务类型的56个任务，在MTEB v2中扩展到100多个任务。2026年初，Gemini Embedding 2在检索任务(67.71 MTEB-R)上领先。Cohere embed-v4在通用任务(65.2 MTEB)上领先。BGE-M3在开源权重多语言任务(63.0)上领先。排行榜是必要的但不充分——始终在你的领域进行基准测试(Benchmark)。

### 三层模式

|  用例  |  模式  |
|----------|---------|
|  快速初筛  |  密集双编码器(Dense Bi-encoder) (BGE-M3, text-3-small)  |
|  召回率提升  |  稀疏(Sparse) (SPLADE, BGE-M3稀疏模式) + RRF融合  |
|  Top-50精确度  |  多向量(Multi-vector) (ColBERTv2) 或交叉编码器重排序器(Cross-encoder Reranker)  |

大多数生产堆栈使用所有三种模式。

## 动手构建

### 步骤1：基线——使用Sentence-BERT的密集嵌入

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` 使点积等于余弦相似度。始终设置它。

### 步骤2：Matryoshka截断

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

截断后重新归一化。Nomic v1.5, OpenAI text-3, 和 Voyage-4 经过训练，使得前几个级别的截断是无损的。非Matryoshka模型（原始Sentence-BERT）在截断时性能急剧下降。

### 步骤3：BGE-M3多功能性

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

三个索引，一次推理调用。得分融合：

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

根据你的领域调整权重。

### 步骤4：在自定义任务上进行MTEB评估

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

在*代表性*子集上运行你的候选模型。不要只相信排行榜排名——你的领域很重要。

### 步骤5：从头实现余弦相似度

参见 `code/main.py`。平均哈希技巧嵌入（仅限标准库）。虽然不如Transformer嵌入有竞争力，但展示了流程：分词 → 向量化 → 归一化 → 点积。

## 陷阱

- **查询和文档使用同一模型。** 有些模型（Voyage、Jina-ColBERT）使用非对称编码（Asymmetric Encoding）——查询和文档经过不同路径。务必查看模型卡片。
- **缺失前缀（Prefix）。** `bge-*` 模型需要在查询前添加 `"Represent this sentence for searching relevant passages: "`。忘记添加会导致召回率下降3-5个点。
- **过度修剪套娃向量（Matryoshka）。** 从1,536维降到256维通常是安全的，但降到64维则不安全。请在你的评估集上验证。
- **上下文截断。** 大多数模型会静默地截断超过其最大长度的输入。长文档需要分块（参见第23课）。
- **忽略延迟长尾（Latency Tail）。** MTEB分数隐藏了p99延迟。一个600M参数的模型可能比335M参数的模型高出2个点，但每次查询的成本高出3倍。

## 使用它

2026年技术栈：

|  情况  |  选择  |
|-----------|------|
|  仅英文、快速、API  |  `text-embedding-3-large` 或 `voyage-3-large`  |
|  开放权重、英文  |  `BAAI/bge-large-en-v1.5`  |
|  开放权重、多语言  |  `BAAI/bge-m3` 或 `Qwen3-Embedding-8B`  |
|  长上下文（32k以上）  |  Voyage-3-large、Cohere embed-v4、Qwen3-Embedding-8B  |
|  仅CPU部署  |  Nomic Embed v2（1.37亿参数，MoE）  |
|  存储受限  |  套娃向量截断（Matryoshka-truncated）+ int8量化  |
|  强关键词查询  |  添加SPLADE稀疏向量，与稠密向量进行RRF融合  |

2026年模式：从BGE-M3或text-3-large开始，使用MTEB在您的领域进行评估，如果领域专用模型胜出超过3个点则更换。

## 发布

保存为 `outputs/skill-embedding-picker.md`：

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## 练习

1. **简单。** 使用 `bge-small-en-v1.5` 对100个句子进行全维度（384）编码，然后使用套娃向量128维。在10个查询上测量MRR下降。
2. **中等。** 在您领域的500个段落上比较BGE-M3的稠密、稀疏和colbert模式。哪个在recall@10上胜出？RRF融合是否优于最佳单一模式？
3. **困难。** 在三个候选模型上运行MTEB，涵盖您的前两个领域任务。报告MTEB分数、100查询批次的p99延迟以及每百万查询的成本。选择帕累托最优（Pareto-optimal）模型。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  稠密嵌入（Dense Embedding）  |  向量  |  每个文本一个固定大小的向量。使用余弦相似度进行排序。  |
|  稀疏嵌入（Sparse Embedding）  |  学习型BM25  |  每个词汇标记一个权重；大部分为零；端到端训练。  |
|  多向量（Multi-vector）  |  ColBERT风格  |  每个标记一个向量；MaxSim评分；索引更大，召回率更高。  |
|  套娃向量（Matryoshka）  |  俄罗斯套娃技巧  |  前N维本身就是有效的较小嵌入。  |
|  MTEB  |  基准测试  |  大规模文本嵌入基准（Massive Text Embedding Benchmark）——发布时56个任务，v2中超过100个。  |
|  BEIR  |  检索基准  |  18个零样本检索任务；常被引用于跨领域鲁棒性。  |
|  非对称编码（Asymmetric Encoding）  |  查询≠文档路径  |  模型对查询和文档使用不同的投影。  |

## 延伸阅读

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084)——双编码器论文。
- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084)——排行榜论文。
- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084)——统一三模模型。
- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084)——维度阶梯训练目标。
- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084)——生产环境中的晚期交互。
- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084)——实时排名。
