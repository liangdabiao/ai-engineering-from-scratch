# 信息检索与搜索

> BM25精确但脆弱。稠密检索覆盖广泛但漏掉关键词。混合检索是2026年的默认方案。其他都是调优。

**类型：** 构建
**语言：** Python
**先决条件：** 阶段5·02（词袋+TF-IDF），阶段5·04（GloVe、FastText、子词）
**时间：** ~75分钟

## 问题

用户输入“如果有人撒谎骗钱怎么办”并期望找到实际覆盖该内容的法条：“IPC第420条”。关键词搜索完全遗漏（没有共享词汇）。语义搜索如果嵌入向量未在法律文本上训练也会遗漏。真实的搜索必须两者兼顾。

信息检索(IR)是每个RAG系统、每个搜索栏、每个文档站点模糊查找的底层管道。2026年可在生产中运行的架构并非单一方法，而是一系列互补方法的链条，每个方法捕捉前一个方法的失败。

本课构建每个部分并指出每个部分捕捉哪些失败。

## 核心概念

![Hybrid retrieval: BM25 + dense + RRF + cross-encoder rerank](../assets/retrieval.svg)

四层。选择你需要的。

1. **稀疏检索（BM25）。** 快速，精确匹配精确，语义上糟糕。在倒排索引上运行。百万级文档上每个查询低于10毫秒。准确获取法条引用、产品代码、错误消息、命名实体。
2. **稠密检索。** 将查询和文档编码为向量。最近邻搜索。捕获转述和语义相似性。遗漏相差一个字符的精确关键词匹配。使用FAISS或向量数据库每个查询50-200毫秒。
3. **融合。** 合并来自稀疏和稠密的排序列表。倒数排序融合(RRF)是简单的默认方法，因为它忽略原始分数（位于不同尺度）而仅使用排名位置。当你知道某个信号在你的领域中占主导时，加权融合是一个选项。
4. **交叉编码器重排序。** 取融合后的前30个结果。运行交叉编码器（查询和文档一起，对每对评分）。保留前5个。交叉编码器每对比双编码器慢，但更准确。通过仅对前30个运行来分摊成本。

三路检索（BM25 + 稠密 + 学习型稀疏如SPLADE）在2026年基准测试中优于两路，但需要学习型稀疏索引的基础设施。对于大多数团队，两路加交叉编码器重排序是最佳点。

## 动手构建

### 第一步：从零实现BM25

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

两个值得知道的参数。`k1=1.5`控制词频饱和；越高意味着词重复的权重越大。`b=0.75`控制长度归一化；0忽略文档长度，1完全归一化。默认值是Robertson在原始论文中的建议，很少需要调优。

### 第二步：使用双编码器的稠密检索

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

L2归一化嵌入，使得点积等于余弦。`all-MiniLM-L6-v2`是384维，快速且对大多数英文检索足够强。对于多语言工作，使用`paraphrase-multilingual-MiniLM-L12-v2`。对于最高精度，使用`bge-large-en-v1.5`或`e5-large-v2`。

### 第三步：倒数排序融合

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60`常数来自原始RRF论文。更高的`k`使排名差异的贡献变平；更低的`k`使顶级排名占主导。60是发布的默认值，很少需要调优。

### 第四步：混合搜索+重排序

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

三个阶段组合。BM25找到词汇匹配。稠密找到语义匹配。RRF合并两个排序而不需要分数校准。交叉编码器使用查询-文档对一起重新评分前30个，捕获双编码器遗漏的细粒度相关性。保留前5个。

### 第五步：评估

|  指标  |  含义  |
|--------|---------|
|  Recall@k  |  在存在正确文档的查询中，它在top-k中的频率。  |
|  MRR (平均倒数排名)  |  第一个相关文档的1/排名的平均值。  |
|  nDCG@k  |  考虑相关性分级，不仅仅是二元相关/不相关。  |

对于RAG特别重要，检索器的 **Recall@k** 是最重要的数字。如果你的阅读器获取的段落不在检索集合中，则无法回答。

调试提示：对于失败的查询，比较稀疏和稠密的排名。如果一个找到了正确的文档而另一个没有，则是词汇不匹配（解决：添加缺失的一半）或语义歧义（解决：更好的嵌入或重排序器）。

## 使用它

2026年技术栈：

|  规模  |  栈  |
|-------|-------|
|  1k-100k文档  |  内存BM25 + `all-MiniLM-L6-v2`嵌入 + RRF。无单独数据库。  |
|  100k-10M文档  |  FAISS或pgvector用于稠密 + Elasticsearch/OpenSearch用于BM25。并行运行。  |
|  10M+文档  |  Qdrant/Weaviate/Vespa/Milvus 支持混合查询。在前30个上运行交叉编码器重排序。  |
|  最佳质量前沿  |  三路（BM25 + 稠密 + SPLADE）+ ColBERT 延迟交互重排序  |

无论你选择什么，都要为评估预算。在基准测试端到端RAG准确性之前，先基准测试检索召回率。阅读器无法修复检索器遗漏的内容。

### 来自2026年生产级RAG的艰辛教训

- **80%的RAG失败根源在于数据摄取和分块，而非模型。**团队花费数周更换LLM并调整提示词，而检索系统却安静地每三次查询中返回一次错误上下文。请先修复分块问题。
- **分块策略比分块大小更重要。**固定大小的分块会破坏表格、代码和嵌套标题。句子感知是默认方案；语义或基于LLM的分块对于技术文档和产品手册效果显著。
- **父文档模式。**检索小的"子"分块以获得精确性。当同一父文档中的多个子分块出现时，替换为父块以保留上下文。这在不重新训练的情况下持续提升答案质量。
- **k_rerank=3通常是最优的。**每增加一个额外的分块会增加令牌成本和生成延迟，且不会提升答案质量。如果k=8对您来说仍然优于k=3，则重排序器性能不足。
- **HyDE/查询扩展。**从查询生成假设性答案，嵌入该答案，然后检索。弥合短问题和长文档之间的措辞差距。无需训练即可免费获得精确性提升。
- **上下文预算不超过8K令牌。**在该限制下持续命中意味着重排序器阈值过于宽松。
- **对所有内容进行版本控制。**提示词、分块规则、嵌入模型、重排序器。任何漂移都会悄悄破坏答案质量。CI门控基于忠实度、上下文精度和未回答问题率，在用户察觉到之前阻止回归。
- **三路检索（BM25 + 稠密 + 学习型稀疏如SPLADE）在2026年基准测试中优于两路检索**，特别是对于混合专有名词和语义的查询。当基础设施支持SPLADE索引时部署它。

根据2026年行业测量，适当的检索设计可将幻觉减少70-90%。大多数RAG性能提升来自更好的检索，而非模型微调。

## 发布

保存为 `outputs/skill-retrieval-picker.md`：

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## 练习

1. **简单。**在500个文档的语料库上实现`hybrid_search`。测试20个查询。比较BM25仅、稠密仅和混合检索在k=5时的召回率。
2. **中等。**添加MRR计算。对于每个已知正确文档的测试查询，找到正确文档在BM25、稠密和混合排名中的位置。报告每个的MRR。
3. **困难。**使用MultipleNegativesRankingLoss（Sentence Transformers）在您的领域微调稠密编码器。从500个查询-文档对构建训练集。比较微调前和微调后的召回率。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  BM25  |  关键词搜索  |  Okapi BM25。通过词频、逆文档频率和文档长度对文档评分。  |
|  稠密检索  |  向量搜索  |  将查询和文档编码为向量，寻找最近邻居。  |
|  双编码器  |  嵌入模型  |  独立编码查询和文档。查询时速度快。  |
|  交叉编码器  |  重排序模型  |  一起编码查询和文档。速度慢但准确。  |
|  RRF  |  排名融合  |  通过求和`1/(k + rank)`来合并两个排名。  |
|  Recall@k  |  检索指标  |  前k个结果中包含相关文档的查询比例。  |

## 延伸阅读

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — 权威的BM25论述。
- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — DPR，经典的双编码器。
- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — 缩小与稠密检索差距的学习型稀疏检索器。
- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — RRF论文。
- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — 晚期交互检索。
