# 问答系统

> 三种系统塑造了现代问答(Question Answering, QA)。抽取式(Extractive)查找片段。检索增强(Retrieval-Augmented)将其锚定在文档中。生成式(Generative)产生答案。每一个现代的AI助手都是这三者的混合。

**类型：** 构建
**语言：** Python
**先决条件：** 阶段5 · 11（机器翻译），阶段5 · 10（注意力机制）
**时间：** ~75分钟

## 问题

用户输入“第一代iPhone是什么时候发布的？”并期望得到“2007年6月29日”。而不是“苹果的历史悠久而多样”。也不是孤零零的“2007”没有句子。一个直接、有依据、正确的答案。

在过去十年中，三种架构主导了问答(QA)。

- **抽取式问答(Extractive QA)。** 给定一个问题和一个已知包含答案的段落，找出答案片段在段落中的起始和结束索引。SQuAD是经典的基准测试。
- **开放域问答(Open-domain QA)。** 段落不给定。先检索相关段落，然后抽取或生成答案。这是当今所有RAG管道的基石。
- **生成式/闭卷问答(Generative / Closed-book QA)。** 一个大语言模型从其参数化记忆中回答。无需检索。推理速度最快，事实可靠性最低。

2026年的趋势是混合式：检索最好的几个段落，然后提示一个生成式模型基于这些段落给出答案。这就是RAG，第14课将深入介绍检索部分。本课构建问答部分。

## 核心概念

![QA architectures: extractive, retrieval-augmented, generative](../assets/qa.svg)

**抽取式(Extractive)。** 使用Transformer（BERT系列）对问题和段落一起编码。训练两个头来预测答案起始和结束的token索引。损失函数是有效位置上的交叉熵。输出是段落中的一个片段。从不产生幻觉（由构造保证），从不处理段落无法回答的问题（由构造保证）。

**检索增强(RAG)。** 两个阶段。首先，检索器从语料库中找到前`k`个段落。其次，阅读器（抽取式或生成式）使用这些段落产生答案。检索器-阅读器的分离使得每个部分可以独立训练和评估。现代RAG通常会在它们之间添加一个重排序器。

**生成式(Generative)。** 一个仅解码器的LLM（GPT、Claude、Llama）从学习到的权重中回答。无需检索步骤。对常识知识表现出色，对罕见或近期事实灾难性的糟糕。幻觉率与预训练数据中事实出现频率成反比。

## 动手构建

### 步骤1：使用预训练模型进行抽取式问答

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2`在SQuAD 2.0上训练，其中包括不可回答的问题。默认情况下，`question-answering`管道会返回得分最高的片段，即使模型的空值得分更高——它*不会*自动返回空答案。要获得明确的“无答案”行为，请向管道调用传递`handle_impossible_answer=True`：当且仅当空值得分超过所有片段得分时，管道才会返回空答案。无论如何，始终检查`score`字段。

### 步骤2：检索增强管道（草图）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

两阶段管道。密集检索器（Sentence-BERT）通过语义相似度找到相关段落。抽取式阅读器（RoBERTa-SQuAD）从合并的顶部段落中提取答案片段。适用于小型语料库。对于百万文档语料库，请使用FAISS或向量数据库。

### 步骤3：使用RAG进行生成式问答

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

提示模式很重要。明确告诉模型以上下文为基础，并在上下文不足时返回“我不知道”，与简单提示相比，可将幻觉率降低40-60%。更复杂的模式会添加引用、置信度分数和结构化提取。

### 步骤4：反映真实世界的评估

SQuAD使用**精确匹配(Exact Match, EM)**和**token级别的F1**。EM是标准化后（小写、去除标点、移除冠词）的严格匹配——要么预测完全匹配，要么得0分。F1基于预测和参考答案的token重叠计算，并给予部分分。两者都欠考虑同义改写：“2007年6月29日” vs “2007年6月29号”通常EM得0（序数破坏了标准化），但由于token重叠，仍然能获得可观的F1分数。

对于生产环境的问答：

- **答案准确性**（由LLM或人类判断，因为指标不能捕捉语义等价性）。
- **引用准确性**。被引用的段落是否真的支持答案？通过生成引用与检索段落之间的字符串匹配，很容易自动检查。
- **拒答校准(Refusal calibration)**。当答案不在检索段落中时，系统是否正确地说“我不知道”？测量虚假置信度(false confidence rate)。
- **检索召回率(Retrieval recall)**。在评估阅读器之前，测量检索器是否将正确的段落纳入前`k`条。阅读器无法修复缺失的段落。

### RAGAS：2026年生产评估框架

`RAGAS`是为RAG系统专门构建的，是2026年默认的发布版本。它无需黄金参考即可在四个维度上评分：

- **忠实度(Faithfulness)**。答案中的每个声明是否都来自检索到的上下文？通过基于NLI的蕴含关系测量。这是你主要的幻觉指标。
- **答案相关性(Answer relevance)**。答案是否针对问题？通过从答案生成假设问题并与真实问题比较来测量。
- **上下文精确率(Context precision)**。在检索到的块中，实际相关的比例是多少？精确率低 = 提示中的噪声。
- **上下文召回率(Context recall)**。检索到的集合是否包含所有需要的信息？召回率低 = 阅读器无法成功。

无参考评分允许你在没有精心策划的黄金答案的情况下，在实时生产流量上进行评估。对于精确匹配指标无用的开放性问题，在上面叠加LLM作为裁判。

`pip install ragas`。接入你的检索器+阅读器。每次查询得到四个标量值。对回归进行告警。

## 使用它

2026年的技术栈。

|  用例  |  推荐  |
|---------|-------------|
|  给定段落，找出答案片段  |  `deepset/roberta-base-squad2`  |
|  在固定语料库上，闭卷不可接受  |  RAG：密集检索器 + LLM阅读器  |
|  在文档存储上的实时查询  |  带混合检索（BM25 + 密集）的RAG + 重排序器（第14课）  |
|  对话式问答（后续问题）  |  带对话历史的LLM + 每轮RAG  |
|  高度事实性、受监管的领域  |  在权威语料库上的抽取式问答；绝不单独使用生成式  |

提取式问答在2026年已不流行，因为带有LLM的RAG能处理更多情况。它仍然在需要逐字引用的场景中投入使用：法律研究、法规合规、审计工具。

## 发布

保存为 `outputs/skill-qa-architect.md`：

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## 练习

1. **简单：** 在10篇维基百科段落上建立上述SQuAD提取式流水线。人工编写10个问题。测量答案正确的次数。如果段落和问题清晰，你应该能看到7-9个正确。
2. **中等：** 添加一个拒绝分类器。当最高检索分数低于阈值（例如余弦0.3）时，返回“我不知道”而不是调用阅读器。在保留集上调整阈值。
3. **困难：** 在你选择的10,000文档语料库上构建一个RAG流水线。实现混合检索（BM25 + 稠密检索）和RRF融合（参见第14课）。测量有无混合步骤的答案准确率。记录哪些问题类型受益最大。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  提取式问答  |  寻找答案跨度  |  预测给定段落中答案的起始和结束索引。  |
|  开放域问答  |  在语料库上的问答  |  无给定段落；必须检索然后回答。  |
|  RAG  |  检索然后生成  |  检索增强生成。检索器+阅读器流水线。  |
|  SQuAD  |  标准基准  |  斯坦福问答数据集。EM + F1度量。  |
|  幻觉  |  编造的答案  |  阅读器输出不被检索到的上下文支持。  |
|  拒绝校准  |  知道何时保持沉默  |  系统在无法回答时正确地说“我不知道”。  |

## 延伸阅读

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — 基准论文。
- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — DPR，问答的标准稠密检索器。
- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — 命名RAG的论文。
- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — 全面的RAG综述。
