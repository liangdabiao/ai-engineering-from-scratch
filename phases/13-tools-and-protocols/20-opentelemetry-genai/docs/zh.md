# OpenTelemetry GenAI — 端到端追踪工具调用

> 一个代理调用五个工具、三个MCP服务器和两个子代理。你需要一个跨越所有这些的追踪。OpenTelemetry GenAI 语义约定（v1.37及更高版本中的稳定属性）是2026年的标准，得到Datadog、Langfuse、Arize Phoenix、OpenLLMetry和AgentOps的原生支持。本课程列出了必需的属性，介绍了跨度层次结构（代理 → LLM → 工具），并附带了一个可以插入任何OTel导出器的标准库跨度发射器。

**类型：** 构建
**语言：** Python（标准库，OTel跨度发射器）
**先决条件：** 阶段13 · 07（MCP服务器），阶段13 · 08（MCP客户端）
**时间：** 约75分钟

## 学习目标

- 列出LLM跨度与工具执行跨度所需的OTel GenAI属性。
- 构建涵盖代理循环、LLM调用、工具调用和MCP客户端调度的追踪层次结构。
- 决定捕获哪些内容（选择性加入）与哪些内容编辑掉（默认）。
- 在不重写工具代码的情况下，将跨度发射到本地收集器（Jaeger，Langfuse）。

## 问题

2026年2月的一个调试：用户报告“我的代理有时需要30秒响应；其他时候3秒。”没有追踪。日志显示LLM调用，但没有显示工具调度、MCP服务器往返、子代理。你只能猜测。最终你发现：某个MCP服务器偶尔在冷启动时挂起。

没有端到端追踪，你无法发现这一点。OTel GenAI解决了这个问题。

这些约定于2025-2026年在OpenTelemetry语义约定组下确定。它们定义了稳定的属性名称，使得Datadog、Langfuse、Phoenix、OpenLLMetry和AgentOps都能解析相同的跨度。一次检测；发送到任何后端。

## 核心概念

### 跨度层次结构

```
agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

整个结构嵌套在一个追踪ID下。跨度ID链接父子关系。

### 必需属性

根据2025-2026年的语义约定：

- `gen_ai.operation.name` — `"chat"`, `"text_completion"`, `"embeddings"`, `"execute_tool"`, `"invoke_agent"`。
- `gen_ai.operation.name` — `"chat"`, `"text_completion"`, `"embeddings"`, `"execute_tool"`。
- `gen_ai.operation.name` — 请求的模型字符串（例如 `"chat"`）。
- `gen_ai.operation.name` — 实际服务的模型。
- `gen_ai.operation.name` / `"chat"`。
- `gen_ai.operation.name` — 用于关联的提供者响应ID。

对于工具跨度：

- `gen_ai.tool.name` — 工具标识符。
- `gen_ai.tool.name` — 特定调用ID。
- `gen_ai.tool.name` — 工具描述（可选）。

对于代理跨度：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### 跨度类型

- `SpanKind.CLIENT` 用于跨越进程边界的调用（LLM提供者，MCP服务器）。
- `SpanKind.CLIENT` 用于代理自身的循环步骤和工具执行。

### 选择性内容捕获

默认情况下，跨度携带度量和计时 — 不包含提示或完成。大型负载和PII默认关闭。设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 和特定的内容捕获环境变量以包含内容。在生产环境中启用前仔细审查。

### 跨度上的事件

可以添加令牌级事件作为跨度事件：

- `gen_ai.content.prompt` — 输入消息。
- `gen_ai.content.prompt` — 输出消息。
- `gen_ai.content.prompt` — 记录的工具调用。

事件在跨度内按时间顺序排列，以进行详细回放。

### 导出器

OTel跨度导出到：

- **Jaeger / Tempo.** 开源，本地部署。
- **Langfuse.** LLM可观测性专用；可视化令牌使用。
- **Arize Phoenix.** 评估与追踪结合。
- **Datadog.** 商业；原生解析 `gen_ai.*` 属性。
- **Honeycomb.** 列式存储；查询友好。

所有都使用OTLP（有线格式）。你的代码无需关心。

### 跨MCP传播

当MCP客户端调用服务器时，将W3C traceparent头注入到请求中。可流式HTTP支持标准头。Stdio本身不携带HTTP头；规范2026路线图讨论在JSON-RPC调用中添加一个 `_meta.traceparent` 字段。

在实现之前：手动在每个请求的`_meta`中包含 traceparent。服务器记录 trace id。

### 指标(Metrics)

除了跨度(Span)，GenAI semconv 还定义了指标：

- `gen_ai.client.token.usage` — 直方图(Histogram)。
- `gen_ai.client.token.usage` — 直方图。
- `gen_ai.client.token.usage` — 直方图。

使用这些指标构建不需要每个调用细节的仪表盘。

### AgentOps 层

AgentOps (成立于2024年) 专注于 GenAI 可观测性。它封装了流行的框架 (LangGraph、Pydantic AI、CrewAI)，自动发出 OTel 跨度。如果你的技术栈使用了支持的框架，这很有用；否则使用手动仪器化。

## 使用它

`code/main.py` 为一个调用 LLM、分派两个工具并进行一次 MCP 往返的代理发出 OTel 形状的跨度到 stdout (采用类似 OTLP-JSON 的格式)。没有真实的导出器——本课专注于跨度的形状和属性集。将输出粘贴到兼容 OTLP 的查看器中，或者直接阅读。

需要关注的内容：

- 所有跨度共享同一 trace id。
- 父子链接通过 `parentSpanId` 编码。
- 必需的 `parentSpanId` 属性已填充。
- 默认关闭内容捕获；一种场景通过环境变量启用。

## 发布

本课产生 `outputs/skill-otel-genai-instrumentation.md`。给定一个代理代码库，该技能产生一个仪器化计划：在何处添加跨度、填充哪些属性以及定位哪些导出器。

## 练习

1. 运行 `code/main.py`。统计跨度数量，并识别哪些是 CLIENT 与哪些是 INTERNAL。

2. 启用内容捕获 (环境变量)，确认出现 `gen_ai.content.prompt` 和 `gen_ai.content.completion` 事件。注意对个人身份信息 (PII) 的影响。

3. 添加工具执行指标 `gen_ai.tool.execution.duration`，并作为每次调用的直方图样本发出。

4. 将 traceparent 从父代理跨度传播到 MCP 请求的 `_meta.traceparent` 字段。验证 MCP 服务器将看到相同的 trace id。

5. 阅读 OTel GenAI semconv 规范。识别 semconv 中列出但本课代码未发出的一个属性。添加该属性。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  OTel  |  "OpenTelemetry"  |  用于跟踪(Traces)、指标(Metrics)、日志(Logs)的开放标准  |
|  GenAI semconv  |  "GenAI semantic conventions"  |  LLM / 工具 / 代理跨度的稳定属性名称  |
|  `gen_ai.*`  |  "The attribute namespace"  |  所有 GenAI 属性共享此前缀  |
|  Span  |  "Timed operation"  |  一个有开始、结束和属性的工作单元  |
|  Trace  |  "Cross-span ancestry"  |  共享同一 trace id 的跨度树  |
|  SpanKind  |  "CLIENT / SERVER / INTERNAL"  |  关于跨度方向的提示  |
|  OTLP  |  "OpenTelemetry Line Protocol"  |  导出器的传输格式  |
|  Opt-in content  |  "Prompt / completion capture"  |  默认关闭；通过环境变量启用  |
|  traceparent  |  "W3C header"  |  在服务间传播跟踪上下文  |
|  Exporter  |  "Backend-specific shipper"  |  将跨度发送到 Jaeger / Datadog 等的组件  |

## 延伸阅读

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 用于 GenAI 跨度、指标和事件的规范约定
- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — LLM 和工具执行跨度属性列表
- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 代理级 [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) 跨度
- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 托管在 GitHub 的真相源
- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 生产集成演练
