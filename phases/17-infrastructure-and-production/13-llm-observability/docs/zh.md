# LLM 可观测性技术栈选型

> 2026年可观测性(observability)市场分为两类。开发平台（LangSmith, Langfuse, Comet Opik）将监控与评估(evals)、提示管理(prompt management)、会话回放(session replays)捆绑在一起。网关/检测工具（Helicone, SigNoz, OpenLLMetry, Phoenix）专注于遥测(telemetry)。Langfuse 核心采用 MIT 许可，开源(OSS)平衡性好（免费云服务每月5万事件）。Phoenix 基于 OpenTelemetry，采用 Elastic License 2.0 — 擅长漂移(drift)/RAG 可视化，但不能作为持久化生产后端。Arize AX 使用零拷贝(zero-copy) Iceberg/Parquet 集成，声称比单体可观测性便宜100倍。LangSmith 在 LangChain/LangGraph 领域领先，每位用户每月39美元，仅企业版支持自托管(self-host)。Helicone 基于代理(proxy)，设置只需15-30分钟，免费每月10万请求，但在代理(agent)追踪方面深度不足。常见生产模式：网关（Helicone/Portkey）+ 评估平台（Phoenix/TruLens），通过 OpenTelemetry 粘合。

**类型：** 学习
**语言：** Python (标准库, 简易追踪采样模拟器)
**先决条件：** 阶段 17 · 08 (推理指标), 阶段 14 (代理工程)
**时间：** 约60分钟

## 学习目标

- 区分开发平台（捆绑：评估 + 提示 + 会话）与网关/遥测工具（仅追踪 + 指标）。
- 将六种主要工具（Langfuse, LangSmith, Phoenix, Arize AX, Helicone, Opik）映射到它们的许可、定价和最佳适用场景。
- 解释 OpenTelemetry 粘合模式，该模式允许你将网关工具与独立的评估平台结合。
- 指出2026年的成本差异化因素（Arize AX 的零拷贝方法与单体摄取(monolithic ingest)）并说明大约100倍的倍数。

## 问题

你发布了一个 LLM 功能。它工作正常。但你无法洞察提示失败、工具循环、延迟回归、成本激增或提示缓存命中率。你在谷歌搜索“LLM 可观测性”，得到八个工具，它们都声称以三种不同的价格点解决同一问题。

它们并不解决同一问题。LangSmith 回答“为什么这个 LangGraph 运行失败？” Phoenix 回答“我的 RAG 管道在漂移吗？” Helicone 回答“哪个应用在烧 tokens？” Langfuse 回答“我可以自托管整个系统吗？”不同的工具，不同的受众。

选择涉及四个维度：技术栈（LangChain？原始 SDK？多供应商？）、许可容忍度（仅MIT？Elastic 可以？商业版没问题？）、预算（免费层？$100/mo? $1000/月？）以及自托管（必须？有则更好？从不？）。

## 核心概念

### 两类

**开发平台**将可观测性与评估、提示管理、数据集版本控制、会话回放捆绑在一起。你运行实验，查看哪个提示有效，对新提示进行数据集回归(dataset-regression)以对比旧优胜者。LangSmith, Langfuse, Comet Opik.

**网关/遥测工具**对推理调用进行检测 — 提示、响应、tokens、延迟、模型、成本。Helicone, SigNoz, OpenLLMetry, Phoenix。极简主义。可以通过 OpenTelemetry 与单独的评估工具结合。

### Langfuse — 开源平衡

- 核心采用 Apache / MIT 许可；通过 Docker 自托管。
- 云免费层：每月5万事件。付费：团队版每月29美元。
- 评估、提示管理、追踪、数据集。全面覆盖四种开发平台功能。
- 最佳适用场景：你想要 LangSmith 级别的功能，但必须自托管或使用开源许可。

### Phoenix (Arize) — 遥测优先，原生 OpenTelemetry

- Elastic License 2.0；自托管简单。
- 擅长 RAG 和漂移可视化。嵌入空间散点图作为一等公民。
- 并非设计为持久化生产后端 — 主要用于开发阶段的可观测性。
- 最佳适用场景：RAG 管道开发、漂移调试，与独立的网关配合用于生产。

### Arize AX — 规模化玩法

- 商业版。通过 Iceberg/Parquet 实现零拷贝数据湖集成。
- 声称在规模化下比单体可观测性（如 Datadog 类）便宜约100倍。原理：你将追踪存储在自己的 S3 上的 Parquet 中；Arize 直接读取。
- 最佳适用场景：每天超过1000万条追踪，已有数据湖，想要 LLM 特定仪表板但不想承受 Datadog 的定价。

### LangSmith — LangChain/LangGraph 优先

- 商业版，每位用户每月39美元。仅企业版支持自托管。
- 在 LangChain 和 LangGraph 技术栈中最佳。如果你不使用两者之一，则吸引力较小。
- 最佳适用场景：团队致力于 LangChain，愿意付费。

### Helicone — 基于代理的最小可行产品

- 只需将你的 `OPENAI_API_BASE` 替换为 Helicone 代理，15-30分钟完成设置。
- MIT 许可；免费每月10万请求，付费每月20美元以上。
- 包括故障转移、缓存、速率限制 — 也充当网关。
- 在代理/多步追踪方面深度不足。
- 最佳适用场景：快速启动、单一技术栈应用、需要网关与可观测性合二为一。

### Opik (Comet) — 开源开发平台

- Apache 2.0，完全开源。
- 与 Langfuse 类似的功能集，带有 Comet 的传承。
- 最佳适用场景：已在 Comet 上的机器学习团队，希望在同一个面板中实现 LLM 可观测性。

### SigNoz — 优先 OpenTelemetry 的完整 APM

- Apache 2.0。通过 OpenTelemetry 处理通用 APM 及 LLM。
- 最佳适用场景：跨服务和 LLM 调用的统一可观测性。

### 粘合剂：OpenTelemetry + 生成式 AI 语义约定

OpenTelemetry 在2025年底发布了生成式 AI 语义约定（`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`）。消费 OTel 的工具可以互操作。正在出现的生产模式：

1. 从每个 LLM 调用发出带有 GenAI 约定的 OTel。
2. 路由到网关（Helicone / Portkey）用于日常。
3. 双发到评估平台（Phoenix / Langfuse）用于回归。
4. 存档到数据湖（Iceberg）以便通过 Arize AX 或 DuckDB 进行长期分析。

### 陷阱：在错误层面进行检测

在代理框架内部进行检测（例如，添加 LangSmith 追踪）会将你与该框架耦合。在 HTTP/OpenAI-SDK 层面（通过 OpenLLMetry 或你的网关）进行检测是可移植的。

### 采样 — 你不可能保留所有数据

当每天请求超过100万次时，全量追踪保留的成本超过 LLM 调用本身。按规则采样：100% 错误、100% 高成本、5% 成功。始终保留聚合数据；保留原始数据用于长尾(long tail)。

### 你应该记住的数字

- Langfuse 免费云服务：每月5万事件。
- LangSmith：每月39美元/用户。
- Helicone 免费：每月10万请求。
- Arize AX 声称：大规模下比单体架构便宜约100倍。
- OpenTelemetry GenAI 约定：2025年推出，2026年广泛采用。

## 使用它

`code/main.py` 模拟了跨保留策略（100%摄入、采样、采样+错误）的100万条追踪日。报告每种策略下的存储成本及丢失的内容。

## 发布

本课产出 `outputs/skill-observability-stack.md`。根据技术栈、规模、预算、许可模式，选择合适的工具。

## 练习

1. 你的团队在 LangChain 上需要开源自托管可观测性。选择 Langfuse 或 Opik 并说明理由。
2. 在每天500万条追踪、Datadog 报价每月15万美元的情况下，计算 Arize AX 的盈亏平衡点。
3. 设计一个 OpenTelemetry GenAI 属性集，作为你的组织应强制在每个LLM调用中包含的指南。
4. 论证 Phoenix 是否足以用于生产环境。何时不满足需求？
5. Helicone 有20ms的代理开销。在P99 TTFT为300ms时，这是否可接受？如果SLA为100ms呢？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  OpenLLMetry  |  "LLM的OTel"  |  用于LLM的开源OpenTelemetry仪表化  |
|  GenAI 约定  |  "OTel属性"  |  针对LLM调用的标准OTel属性名称  |
|  LangSmith  |  "LangChain可观测性"  |  与LangChain生态系统捆绑的商业平台  |
|  Langfuse  |  "开源LangSmith"  |  具有类似功能集的MIT开源项目  |
|  Phoenix  |  "Arize开发工具"  |  原生OpenTelemetry的开发/评估平台  |
|  Arize AX  |  "规模可观测性"  |  商业零拷贝Iceberg/Parquet可观测性平台  |
|  Helicone  |  "代理可观测性"  |  收集LLM遥测数据的HTTP代理，兼具网关功能  |
|  Opik  |  "Comet LLM"  |  来自Comet的Apache 2.0开源开发平台  |
|  会话重放  |  "追踪重放"  |  重放包含工具调用的完整代理会话  |
|  评估  |  "离线测试"  |  在标记数据集上运行候选模型/提示词  |

## 延伸阅读

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)
