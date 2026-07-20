# 长上下文评估 — NIAH、RULER、LongBench、MRCR

> Gemini 3 Pro 宣称支持 1000 万 token 的上下文。在 100 万 token 时，8 针 MRCR 降至 26.3%。宣称的 ≠ 可用的。长上下文评估揭示了你实际部署的模型的实际能力。

**类型：** 学习
**语言：** Python
**先修知识：** 第 5 阶段 · 13（问答），第 5 阶段 · 23（分块策略）
**时间：** 约 60 分钟

## 问题

你有一份 200 页的合同。模型声称有 100 万 token 的上下文。你把合同粘贴进去并问：“终止条款是什么？”模型给出了答案——但答案来自封面页，因为终止条款位于 12 万 token 深处，超出了模型实际注意的范围。

这就是 2026 年的上下文容量差距。规格表上说 1000 万或 100 万。实际上，只有 60-70% 是可用的，而“可用”取决于任务。

- **检索（大海捞针）：** 在前沿模型上，直到宣称的最大长度时几乎完美。
- **多跳/聚合：** 在大多数模型上，超过约 12.8 万 token 后性能急剧下降。
- **分散事实的推理：** 第一个失败的任务。

长上下文评估衡量这些维度。本课介绍了这些基准测试的名称、每个测试实际衡量的内容，以及如何为你自己的领域构建自定义的“针”测试。

## 核心概念

![NIAH baseline, RULER multi-task, LongBench holistic](../assets/long-context-eval.svg)

**大海捞针（NIAH，2023 年）。** 在一个长上下文中，将一个事实（“魔法词是 pineapple”）放置在受控深度。让模型检索它。遍历深度×长度。这是最早的长上下文基准测试。前沿模型现在已饱和；这是一个必要但不充分的基线。

**RULER（英伟达，2024 年）。** 4 个类别共 13 种任务类型：检索（单键/多键/多值）、多跳追踪（变量追踪）、聚合（常见词频率）、问答。可配置上下文长度（4k 到 128k+）。揭示了那些在 NIAH 上饱和但在多跳任务上失败的模型。在 2024 年的版本中，17 个声称支持 32k+ 上下文的模型中只有一半在 32k 时保持了质量。

**LongBench v2（2024 年）。** 503 道选择题，上下文长度 8000 到 200 万词，六大任务类别：单文档问答、多文档问答、长上下文学习、长对话、代码仓库、长结构化数据。这是真实长上下文行为的生产级基准测试。

**MRCR（多轮共指消解）。** 大规模多轮共指消解。有 8 针、24 针、100 针变体。揭示了在注意力退化之前模型能同时处理多少个事实。

**NoLiMa。** “非字面针”。针和查询之间没有字面重叠；检索需要一步语义推理。比 NIAH 更难。

**HELMET。** 拼接多个文档，从其中一个文档提问。测试选择性注意。

**BABILong。** 将 bAbI 推理链嵌入无关的干草堆中。测试“草堆中推理”，而不仅仅是检索。

### 实际应报告的内容

- **宣称的上下文窗口。** 规格表上的数字。
- **有效检索长度。** 在某个阈值（例如 90%）下通过 NIAH 测试的长度。
- **有效推理长度。** 在该阈值下通过多跳或聚合测试的长度。
- **退化曲线。** 按任务类型绘制的准确度与上下文长度关系图。

规格表中两个数字：有效检索长度和有效推理长度。通常有效推理长度是宣称窗口的 25-50%。

## 动手构建

### 步骤 1：针对你的领域自定义 NIAH

参见 `code/main.py`。骨架如下：

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

遍历 `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}。绘制热力图。这就是目标模型的 NIAH 卡片。

### 步骤 2：多针变体

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

像“哪三个魔法词？”这样的问题需要检索所有三个。单针成功并不能预测多针成功。

### 步骤 3：多跳变量追踪（RULER 风格）

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

答案需要串联三个赋值。前沿模型在 128k 时常常降到 50-70% 的准确率。

### 步骤 4：在你的栈上运行 LongBench v2

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

报告每个类别的准确率。汇总分数掩盖了任务级别的大差异。

## 陷阱

- **仅 NIAH 评估。** 在 100 万 token 上通过 NIAH 对多跳任务没有任何说明。始终运行 RULER 或自定义的多跳测试。
- **均匀深度采样。** 许多实现只测试了深度 0.5。测试深度 0、0.25、0.5、0.75、1.0——“中间迷失”效应是真实存在的。
- **与填充词的字面重叠。** 如果针与填充词共享关键词，检索就变得微不足道。使用 NoLiMa 风格的非重叠针。
- **忽略延迟。** 100 万 token 的提示需要 30-120 秒的预填充。在测量准确率的同时也要测量首 token 时间。
- **厂商自报数字。** OpenAI、Google、Anthropic 都发布了自己的分数。始终根据你的用例独立重新运行。

## 使用它

2026年技术栈：

|  场景  |  基准测试  |
|-----------|-----------|
|  快速健康检查  |  自定义 NIAH，3 个深度 × 3 个长度  |
|  生产环境模型选择  |  RULER（13 个任务），在你目标长度上  |
| 真实世界问答质量 | LongBench v2 单文档问答子集 |
| 多跳推理 | BABILong 或自定义变量追踪 |
| 对话/交互 | 目标长度下的 MRCR 8针测试 |
| 模型升级回归 | 固定的内部 NIAH + RULER测试框架，每个新模型都运行 |

生产环境经验法则：在目标长度上完成 NIAH + 1个推理任务之前，永远不要信任上下文窗口。

## 发布

保存为 `outputs/skill-long-context-eval.md`：

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## 练习

1. **简单.** 构建一个包含3个深度(0.25, 0.5, 0.75)×3个长度(1k, 4k, 16k)的NIAH测试。在任何模型上运行。绘制通过率的3×3热力图。
2. **中等.** 增加一个三针变体。测量每个长度下对所有3个针的检索能力。与相同长度下的单针通过率进行比较。
3. **困难.** 构建一个嵌入在64k填充文本中的变量追踪任务（X1→X2→X3，3跳）。测量3个前沿模型的准确率。报告每个模型的有效推理长度。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
| NIAH | 大海捞针(Needle in haystack) | 在填充文本中植入一个事实，要求模型检索出来。 |
| RULER | 升级版NIAH(NIAH on steroids) | 涵盖检索/多跳/聚合/问答的13种任务类型。 |
| 有效上下文(Effective context) | 真实容量 | 准确率仍保持在阈值以上的长度。 |
| 中间迷失(Lost in the middle) | 深度偏差 | 模型对长输入中间内容的关注不足。 |
| 多针(Multi-needle) | 同时测试多个事实 | 多个植入点；测试注意力切换能力，而非仅检索。 |
| MRCR | 多轮共指(Multi-round coref) | 8、24或100针共指；揭示注意力饱和。 |
| NoLiMa | 非词汇掩码针(Non-lexical needle) | 针与查询无共同字面标记；需要推理。 |

## 延伸阅读

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 原始NIAH仓库。
- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 多任务基准。
- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 真实世界长上下文评估。
- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 更难的海捞针。
- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 干草堆中的推理。
- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 深度偏差论文。
