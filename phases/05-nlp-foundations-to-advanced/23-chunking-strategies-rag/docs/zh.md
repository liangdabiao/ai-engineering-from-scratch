# RAG的分块策略

> 分块配置对检索质量的影响不亚于嵌入模型的选择（Vectara NAACL 2025）。分块做错了，再多的重排序也救不了你。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段5 · 14（信息检索），阶段5 · 22（嵌入模型）
**时间：** ~60分钟

## 问题

你把一份50页的合同放入RAG系统。用户问：“终止条款是什么？”检索器返回了封面页。为什么？因为模型是在512个token的分块上训练的，而终止条款在20页后，被分页符分割，且没有本地关键词与查询关联。

解决方法不是“买个更好的嵌入模型”，而是分块。多大？重叠？在哪里分割？是否包含上下文？

2026年2月的基准测试显示了令人惊讶的结果：

- Vectara 2026年的研究：递归512 token分块以69%对54%的准确率击败了语义分块。
- SPLADE + Mistral-8B在Natural Questions上的测试：重叠没有带来可衡量的收益。
- 上下文悬崖：在约2500个token的上下文处，响应质量急剧下降。

“显而易见”的答案（语义分块、20%重叠、1000个token）通常是错误的。本节课建立对六种策略的直觉，并告诉你何时使用哪种。

## 核心概念

![Six chunking strategies visualized on one passage](../assets/chunking.svg)

**固定分块。** 每隔N个字符或token进行分割。最简单的基线。会在句子中间断开。压缩好，连贯性差。

**递归分块。** LangChain的@@SK0000@@。首先尝试在`\n\n`处分割，然后是`\n`，然后是`.`，然后是空格。干净地回退。2026年的默认方案。

**语义分块。** 嵌入每个句子。计算相邻句子之间的余弦相似度。在相似度低于阈值的地方分割。保持主题连贯性。速度较慢；有时会产生40个token的小碎片，损害检索效果。

**句子分块。** 在句子边界分割。每个分块一个句子或N个句子的窗口。在最多~5000个token的范围内，以极低的成本匹配语义分块。

**父文档分块。** 存储小的子分块用于检索，以及大的父分块用于上下文。通过子分块检索，返回父分块。优雅降级：不好的子分块仍然返回合理的父分块。

**延迟分块（2024）。** 首先在token级别嵌入整个文档，然后将token嵌入汇聚成分块嵌入。保留跨分块上下文。适用于长上下文嵌入器（BGE-M3, Jina v3）。计算量更大。

**上下文检索（Anthropic，2024）。** 在每个分块前面加上LLM生成的关于其在文档中位置的摘要（“此分块是终止条款的第3.2节...”）。在Anthropic自己的基准测试中，检索性能提升35-50%。索引成本高。

### 击败所有默认值的规则

将分块大小与查询类型匹配：

|  查询类型  |  分块大小  |
|------------|-----------|
|  事实型（“CEO的名字是什么？”）  |  256-512 token  |
|  分析型/多跳  |  512-1024 token  |
|  整个章节的理解  |  1024-2048 token  |

NVIDIA 2026年的基准测试。分块应足够大以包含答案和局部上下文，同时足够小以便检索器的top-K结果聚焦于答案而非上下文噪声。

## 动手构建

### 步骤1：固定分块和递归分块

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### 步骤2：语义分块

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

根据你的领域调整@@SK0000@@。太高→碎片化。太低→一个巨大的分块。

### 步骤3：父文档分块

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

关键见解：对父分块去重。多个子分块可能映射到同一个父分块；返回所有会浪费上下文。

### 步骤4：上下文检索（Anthropic模式）

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

索引上下文化后的分块。在查询时，检索受益于额外的周围信号。

### 步骤5：评估

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

始终进行基准测试。最适合你语料库的策略可能与任何博客文章都不匹配。

## 陷阱

- **仅在事实性查询上评估分块。**多跳查询揭示出截然不同的优胜者。使用按查询类型分层的评估集。
- **无最小尺寸的语义分块。**产生40个token的片段，损害检索效果。始终强制执行`min_tokens`。
- **重叠作为盲从。**2026年的研究发现，重叠通常带来零收益，并使索引成本翻倍。测量，不要假设。
- **无最小/最大限制。**5个token或5000个token的分块都会破坏检索。要加以限制。
- **跨文档分块。**绝不允许一个分块跨越两个文档。始终按文档分块，然后合并。

## 使用它

2026年技术栈：

| 情景  |  策略 |
|-----------|----------|
| 首次构建，未知语料库  |  递归式，512 tokens，无重叠 |
| 事实性问答  |  递归式，256-512 tokens |
| 分析性/多跳  |  递归式，512-1024 tokens + 父文档 |
| 高交叉引用（合同、论文）  |  延迟分块或上下文检索 |
| 对话式/对话语料库  |  轮次级分块 + 说话人元数据 |
| 简短话语（推文、评论）  |  一个文档 = 一个分块 |

从递归式512开始。在50个查询的评估集上测量recall@5。然后进行调整。

## 发布

保存为 `outputs/skill-chunker.md`：

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## 练习

1. **简单。**对一个20页的文档分别使用fixed(512, 0)、recursive(512, 0)和recursive(512, 100)进行分块。比较分块数量和边界质量。
2. **中等。**基于5个文档构建30个查询的评估集。测量recursive、semantic和parent-document的recall@5。哪个胜出？是否与博客文章一致？
3. **困难。**实现上下文检索。测量相对于基线recursive的MRR提升。报告索引成本（LLM调用次数）与准确率收益。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
| 分块  |  文档的一部分  |  被嵌入、索引和检索的子文档单元。 |
| 重叠  |  安全余量  |  相邻分块之间共享的N个token；在2026年的基准测试中通常无用。 |
| 语义分块  |  智能分块  |  在相邻句子嵌入相似度下降处进行分割。 |
| 父文档  |  两级检索  |  检索小子块，返回更大的父文档。 |
| 延迟分块  |  嵌入后分块  |  在token级别嵌入整个文档，池化为分块向量。 |
| 上下文检索  |  Anthropic的技巧  |  在索引前，将LLM生成的摘要添加到每个分块之前。 |
| 上下文悬崖  |  2500 token壁垒  |  在RAG中观察到约2.5k上下文token时质量下降（2026年1月）。 |

## 延伸阅读

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 生产环境中的默认设置。
- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 分块与嵌入选择同样重要。
- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 延迟分块论文。
- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 使用LLM生成的上下文前缀可提升35-50%的检索效果。
- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 根据查询类型确定分块大小。
