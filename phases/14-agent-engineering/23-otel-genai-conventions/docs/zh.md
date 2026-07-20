# OpenTelemetry GenAI 语义约定

> OpenTelemetry 的 GenAI 特别兴趣小组（SIG，于2024年4月启动）定义了代理遥测的标准架构。跨度名称、属性和内容捕获规则在各大供应商之间趋于一致，因此代理跟踪在 Datadog、Grafana、Jaeger 和 Honeycomb 中具有相同的含义。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前提条件：** 阶段14 · 13（LangGraph），阶段14 · 24（可观测性平台）
**时间：** 约60分钟

## 学习目标

- 命名 GenAI 跨度类别：模型/客户端、代理、工具。
- 区分 `invoke_agent` CLIENT 与 INTERNAL 跨度以及各自的应用场景。
- 列出顶层 GenAI 属性：提供商名称、请求模型、数据源 ID。
- 解释内容捕获约定：选择加入、`invoke_agent`、外部引用建议。

## 问题

每个供应商都发明了自己的跨度名称。运维团队最终为每个框架构建仪表板。OpenTelemetry 的 GenAI SIG 通过定义一个整个生态系统都遵循的标准来解决这个问题。

## 核心概念

### 跨度类别

1. **模型/客户端跨度。** 涵盖原始 LLM 调用。由提供商 SDK（Anthropic、OpenAI、Bedrock）和框架模型适配器发出。
2. **代理跨度。** `create_agent`（当代理被构造时）和 `invoke_agent`（当它运行时）。
3. **工具跨度。** 每个工具调用一个跨度；通过父子关系连接到代理跨度。

### 代理跨度命名

- 跨度名称：`invoke_agent {gen_ai.agent.name}`（如果已命名）；否则回退到 `invoke_agent`。
- 跨度种类：
  - **CLIENT** — 用于远程代理服务（OpenAI Assistants API、Bedrock Agents）。
  - **INTERNAL** — 用于进程内代理框架（LangChain、CrewAI、本地 ReAct）。

### 关键属性

- `gen_ai.provider.name` — `anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.provider.name` — 模型 ID。
- `gen_ai.provider.name` — 解析后的模型（可能因路由而与请求不同）。
- `gen_ai.provider.name` — 代理标识符。
- `gen_ai.provider.name` — `anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.provider.name` — 用于 RAG：查询的语料库或存储。

存在针对 Anthropic、Azure AI Inference、AWS Bedrock、OpenAI 的技术特定约定。

### 内容捕获

默认规则：检测工具默认不应捕获输入/输出。捕获通过以下方式选择加入：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

推荐的生产模式：将内容存储在外部（S3、您的日志存储），在跨度上记录引用（指针 ID，而非文本）。这是第27课中内置到可观测性中的内容污染防御。

### 稳定性

截至2026年3月，大多数约定仍处于实验阶段。通过以下方式选择加入稳定预览：

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 将 GenAI 属性原生映射到其 LLM 可观测性模式中。其他后端（Grafana、Honeycomb、Jaeger）支持原始属性。

### 这种模式出错的地方

- **在跨度中捕获完整提示。** 运维人员可读取跟踪中的个人身份信息（PII）、机密、客户数据。请将内容存储在外部。
- **缺少 `gen_ai.provider.name`。** 当缺少归属信息时，多提供商仪表板会崩溃。
- **没有父链接的跨度。** 孤立的工具跨度。始终传播上下文。
- **未设置稳定性选择加入。** 您的属性可能会在后端升级时被重命名。

## 动手构建

`code/main.py` 实现了一个符合 GenAI 约定的标准库跨度发射器：

- `Span` 带有 GenAI 属性架构。
- `Span` 带有 `Tracer`、嵌套上下文。
- 一个脚本化的代理运行，发出：`Span`、`Tracer`（INTERNAL）、每个工具的跨度、`start_span` 用于 LLM 调用的跨度。
- 一种内容捕获模式，将提示存储在外部并在跨度上记录 ID。

运行它：

```
python3 code/main.py
```

输出：一个包含所有必需 GenAI 属性的跨度树，以及一个显示选择加入内容引用的“外部存储”。

## 使用它

- **Datadog LLM 可观测性**（v1.37+）原生映射属性。
- **Langfuse / Phoenix / Opik**（第24课） — 自动检测生态系统。
- **Jaeger / Honeycomb / Grafana Tempo** — 原始 OTel 跟踪；从 GenAI 属性构建仪表板。
- **自托管** — 运行带有 GenAI 处理器的 OTel Collector。

## 发布

`outputs/skill-otel-genai.md` 将 OTel GenAI 跨度连接到现有代理，并具有内容捕获默认设置和外部引用存储。

## 练习

1. 为您的第01课 ReAct 循环添加 `invoke_agent`（INTERNAL）+ 每个工具的跨度检测。发送到 Jaeger 实例。
2. 添加“仅引用”模式的内容捕获：提示存储到 SQLite，跨度属性仅携带行 ID。
3. 阅读 `invoke_agent` 的规范。将其连接到您的第09课 Mem0 搜索中。
4. 设置 `invoke_agent` 并验证您的属性不会被收集器重命名。
5. 构建一个仪表板：仅从 GenAI 属性分析“哪些工具错误与哪些模型相关”。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  GenAI SIG  |  "OpenTelemetry GenAI 组"  |  OTel 工作组，定义架构  |
|  invoke_agent  |  "代理跨度"  |  表示代理运行的跨度名称  |
|  CLIENT 跨度  |  "远程调用"  |  对远程代理服务的调用跨度  |
|  INTERNAL 跨度  |  "进程内"  |  进程内代理运行的跨度  |
|  gen_ai.provider.name  |  "提供商"  |  anthropic / openai / aws.bedrock / google.vertex  |
|  gen_ai.data_source.id  |  "RAG源"  |  检索命中所属的语料库/数据存储  |
|  内容捕获  |  "提示日志记录"  |  选择性地捕获消息；在生产环境中外部存储  |
|  稳定性选择加入  |  "预览模式"  |  用于固定实验性约定的环境变量  |

## 延伸阅读

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 规范
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 默认的GenAI跨度
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 内置的OTel跨度
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — W3C追踪上下文传播
