# LLM 评估 —— RAGAS、DeepEval、G-Eval

> 精确匹配(Exact-match)和 F1 分数无法捕捉语义等价性(Semantic Equivalence)。人工审核无法规模化。LLM-as-judge (以 LLM 为裁判)是生产环境中的答案——通过足够的校准(Calibration)来信任其数值。

**类型：** 构建(Build)
**语言：** Python
**前置要求：** 阶段 5 · 13 (问答(Question Answering))，阶段 5 · 14 (信息检索(Information Retrieval))
**时间：** 约 75 分钟

## 问题

你的 RAG 系统回答："June 29th, 2007."
黄金参考答案(Gold Reference)是："June 29, 2007."
精确匹配(Exact Match)得分为 0。F1 分数约为 75%。人类会给出 100% 的分数。

现在，将其乘以 10,000 个测试用例。再乘以对检索器(Retriever)、分块(Chunking)、提示(Prompt)或模型的每一次更改。你需要一个评估器(Evaluator)，它能理解语义、低成本大规模运行、不隐瞒回归(Regression)问题，并暴露出正确的失败模式(Failure Mode)。

2026 年有三个框架(Framework)主导了这个问题。

- **RAGAS.** Retrieval-Augmented Generation Assessment（检索增强生成评估）。四个 RAG 指标（忠实度(Faithfulness)、答案相关性(Answer Relevance)、上下文精确度(Context Precision)、上下文召回率(Context Recall)），后端使用 NLI + LLM-judge。研究支持、轻量级。
- **DeepEval.** Pytest for LLMs（面向大语言模型的 Pytest）。G-Eval、任务完成度(Task Completion)、幻觉(Hallucination)、偏见(Bias)指标。本地支持 CI/CD。
- **G-Eval.** 一种方法（也是 DeepEval 的一个指标）：LLM-as-judge 结合思维链(Chain-of-Thought)、自定义标准(Custom Criteria)，输出 0-1 分数。

这三个都依赖 LLM-as-judge。本节课将建立对该方法及其信任层(Trust Layer)的直觉(Intuition)。

## 核心概念

![Four evaluation dimensions, LLM-as-judge architecture](../assets/llm-evaluation.svg)

**LLM-as-judge.** 用一个 LLM 替换静态指标(Static Metric)，该 LLM 根据评分标准(Rubric)对输出进行评分。给定 `(query, context, answer)`，提示裁判 LLM(Judge LLM)："对忠实度(Faithfulness)评分 0-1。"返回分数。

为什么有效：LLM 以极低的成本近似人类判断。GPT-4o-mini 约为 ~$0.003 per scored case enables 1000-sample regression eval runs for under $5。

为什么它会静默失败：

1. **裁判偏见(Judge Bias).** 裁判更喜欢更长的答案、来自同一模型家族的答案、以及与提示风格匹配的答案。
2. **JSON 解析失败.** 错误的 JSON → NaN 分数 → 静默地从聚合中排除。RAGAS 用户深知此痛。用 try/except 加显式失败模式(Failure Mode)进行防护。
3. **模型版本漂移.** 升级裁判会改变所有指标。冻结裁判模型及其版本。

**RAG 四指标。**

|  指标(Metric)  |  问题(Question)  |  后端(Backend)  |
|--------|----------|---------|
|  忠实度(Faithfulness)  |  答案中的每个声明(Claim)是否来自检索到的上下文？  |  基于 NLI 的蕴涵(Entailment)  |
|  答案相关性(Answer Relevance)  |  答案是否针对问题？  |  从答案生成假设问题(Hypothetical Question)；与真实问题比较  |
|  上下文精确度(Context Precision)  |  检索到的分块中，相关比例是多少？  |  LLM-judge  |
|  上下文召回率(Context Recall)  |  检索是否返回了所有需要的信息？  |  LLM-judge 与黄金答案(Gold Answer)对比  |

**G-Eval.** 定义一个自定义标准(Custom Criterion)："答案是否引用了正确的来源？"框架自动扩展为思维链(Chain-of-Thought)评估步骤，然后评分 0-1。适用于 RAGAS 未覆盖的特定领域质量维度。

**校准(Calibration).** 在获得与人类标签的关联之前，永远不要相信原始裁判分数。运行 100 个手工标注示例。绘制裁判与人类的对比图。计算 Spearman 秩相关系数(Spearman rho)。如果 rho < 0.7，你的裁判评分标准需要改进。

## 动手构建

### 步骤 1：基于 NLI 的忠实度检查（RAGAS 风格）

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

将答案分解为原子声明(Atomic Claim)。对每个声明进行 NLI 检查，对比检索到的上下文。忠实度 = 被支持的分数。

### 步骤 2：答案相关性

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

如果答案暗示的问题与所问不同，则相关性降低。

### 步骤 3：G-Eval 自定义指标

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

评估步骤即为评分标准。显式步骤比隐式的"评分 0-1"提示更稳定。

### 步骤 4：CI 门控

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

以 pytest 文件形式发布。在每个 PR 上运行。在回归(Regression)时阻止合并。

### 步骤 5：从头开始的小型评估

参见 `code/main.py`。仅使用标准库近似忠实度（答案声明与上下文的交集）和相关性（答案词元与问题词元的交集）。非生产环境。展示大致形态。

## 陷阱

- **无需校准。** 与人工标注仅有0.3相关性的法官模型等同于噪声。在发布前要求进行校准运行。
- **自我评估。** 使用同一LLM进行生成和评估会使分数虚高10-20%。应使用不同模型族作为评估器。
- **成对评估中的位置偏差。** 评估器更倾向于第一个选项。始终随机化顺序并运行正反两次。
- **原始汇总掩盖失败。** 平均分0.85往往隐藏了5%的灾难性失败。始终检查最低分位数。
- **黄金数据集退化。** 未经版本管理的评估集随时间漂移，会破坏纵向对比。每次更改时对数据集打标签。
- **LLM成本。** 大规模下，评估调用占主导成本。使用满足校准阈值的最便宜模型。例如GPT-4o-mini、Claude Haiku、Mistral-small。

## 使用它

2026年技术栈：

|  用例  |  框架  |
|---------|-----------|
|  RAG质量监控  |  RAGAS（4个指标）  |
|  CI/CD回归门禁  |  DeepEval + pytest  |
|  自定义领域标准  |  DeepEval内的G-Eval  |
|  在线实时流量监控  |  RAGAS（无参考模式）  |
|  人工抽检  |  LangSmith或Phoenix（含标注UI）  |
|  红队/安全评估  |  Promptfoo + DeepEval  |

典型技术栈：RAGAS用于监控，DeepEval用于CI，G-Eval用于新维度。三者同时运行，它们会产生有意义的差异。

## 发布

保存为 `outputs/skill-eval-architect.md`：

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## 练习

1. **简单。** 在10个已知幻觉的RAG样本上使用RAGAS，验证忠实度指标能否捕捉到每个幻觉。
2. **中等。** 人工标注50个问答答案的正确性（0-1）。用G-Eval评分。测量评估器分数与人工分数的斯皮尔曼相关系数。
3. **困难。** 使用DeepEval构建pytest CI门禁。故意退化检索器。验证门禁生效。通过最低10%的阈值检查添加底部分位数告警。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  LLM作为评估器  |  用LLM评分  |  提示评估模型根据评分标准对输出进行0-1评分。  |
|  RAGAS  |  RAG指标库  |  开源评估框架，包含4个无参考RAG指标。  |
|  忠实度  |  答案是否有依据？  |  答案中能被检索上下文支持的断言比例。  |
|  上下文精确度  |  检索的块是否相关？  |  前K个块中实际有用的比例。  |
|  上下文召回率  |  检索是否找到所有内容？  |  黄金答案中被检索块支持的断言比例。  |
|  G-Eval  |  自定义LLM评估器  |  评分标准 + 思维链评估步骤 + 0-1分数。  |
|  校准  |  信任但验证  |  评估器分数与人工分数的斯皮尔曼相关系数。  |

## 延伸阅读

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — RAGAS论文。
- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — G-Eval论文。
- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — 开放生产栈。
- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — 偏差、校准、局限性。
- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — 整合RAGAS、DeepEval和Phoenix的统一框架。
