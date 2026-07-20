# 顶点项目 11 —— 大语言模型可观测性与评估仪表板

> Langfuse 转向了开放核心。Arize Phoenix 发布了 2026 年 GenAI 语义约定映射。Helicone 和 Braintrust 都加倍投入了按用户成本归属。Traceloop 的 OpenLLMetry 成为了事实上的 SDK 仪表化标准。生产架构是 ClickHouse 用于追踪，Postgres 用于元数据，Next.js 用于 UI，以及一个由评估作业（DeepEval、RAGAS、LLM-judge）组成的小团队，它们对采样追踪运行。构建一个自托管系统，从至少四个 SDK 家族中摄取数据，并展示在五分钟内捕获注入的回归。

**类型：** 顶点项目
**语言：** TypeScript (UI), Python / TypeScript (摄取 + 评估), SQL (ClickHouse)
**先决条件：** 阶段 11 (LLM 工程), 阶段 13 (工具), 阶段 17 (基础设施), 阶段 18 (安全)
**所执行的阶段：** P11 · P13 · P17 · P18
**时间：** 25 小时

## 问题

每个在 2026 年运行生产流量的 AI 团队都会在模型之外保留一个可观测性平面。成本归属。幻觉检测。漂移监控。越狱信号。SLO 仪表盘。PII 泄露警报。开源参考方案——Langfuse、Phoenix、OpenLLMetry——汇聚于 OpenTelemetry GenAI 语义约定作为摄入模式。现在，您可以使用一个 SDK 对 OpenAI、Anthropic、Google、LangChain、LlamaIndex 和 vLLM 进行仪表化，并发送兼容的跨度。

您将构建一个自托管仪表盘，它从至少四个 SDK 家族中摄取数据，对采样追踪运行一小部分评估作业，检测漂移并发出警报。衡量标准：给定一个故意注入的回归（一个开始产生 PII 的提示），仪表盘在五分钟内捕获它并触发警报。

## 概念

摄取是 OTLP HTTP。SDK 生成 GenAI 语义约定跨度：`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.response.id`、`llm.prompts`、`llm.completions`。跨度落入 ClickHouse 进行列式分析；元数据（用户、会话、应用）落入 Postgres。

评估作业作为批处理作业在采样追踪上运行。DeepEval 对忠实度、毒性和答案相关性进行评分。RAGAS 在跟踪包含检索上下文时对检索指标进行评分。自定义 LLM 判断器运行领域特定检查（PII 泄露、违反策略的响应）。评估运行结果写回与父追踪相关联的评估跨度所在的同一 ClickHouse。

漂移检测监控嵌入空间分布随时间的变化（对提示嵌入的 PSI 或 KL 散度）以及评估分数趋势。警报馈入 Prometheus Alertmanager，然后进入 Slack / PagerDuty。UI 是 Next.js 15 与 Recharts。

## 架构

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## 技术栈

- 摄取: OpenTelemetry SDK + GenAI 语义约定; OTLP HTTP 传输
- 收集器: 带有尾部采样处理器（用于成本控制）的 OpenTelemetry 收集器
- 存储: ClickHouse 用于跨度, Postgres 用于元数据, S3 用于原始事件归档
- 评估: DeepEval, RAGAS 0.2, Arize Phoenix 评估器包, 自定义 LLM 判断器
- 漂移: 每周对汇总的提示嵌入（句子变换器）计算 PSI / KL
- 告警: Prometheus Alertmanager -> Slack / PagerDuty
- UI: Next.js 15 App Router + Recharts + 服务器操作
- 开箱即用支持的 SDK: OpenAI, Anthropic, Google GenAI, LangChain, LlamaIndex, vLLM

## 动手构建

1. **收集器配置。** 带有 OTLP HTTP 接收器的 OpenTelemetry 收集器，一个尾部采样器保留 100% 的出错追踪和 10% 的成功追踪，以及导出到 ClickHouse 和 S3 的导出器。

2. **ClickHouse 模式。** 表 `spans` 包含反映 GenAI 语义约定的列：`gen_ai_system`、`gen_ai_request_model`、`input_tokens`、`output_tokens`、`latency_ms`、`prompt_hash`、`trace_id`、`parent_span_id`，外加用于长负载的 JSON 包。添加按 user_id 和 app_id 的二级索引。

3. **SDK 覆盖测试。** 使用每个 SDK（OpenAI、Anthropic、Google、LangChain、LlamaIndex、vLLM）编写一个小型客户端应用程序，并启用 OpenLLMetry 自动仪表化。验证每个应用都能生成规范的 GenAI 跨度，并落入 ClickHouse。

4. **评估作业。** 一个定时作业读取最近 15 分钟的采样追踪，并运行 DeepEval 的忠实度、毒性和答案相关性。输出是与父追踪相关联的评估跨度。

5. **自定义 LLM 判断器。** 一个 PII 泄露判断器：给定一个响应，调用一个守卫 LLM 对 PII 泄露的可能性进行评分。高分响应进入一个分类队列。

6. **漂移检测。** 每周作业计算本周汇总的提示嵌入与过去 4 周基线之间的 PSI。如果 PSI 超过阈值，则发出警报。

7. **仪表盘。** Next.js 15 包含以下页面：概览（跨度/秒、成本/用户、p95 延迟）、追踪（搜索 + 瀑布图）、评估（忠实度趋势、毒性）、漂移（PSI 随时间变化）、告警。

8. **告警链。** Prometheus 导出器读取评估分数聚合和延迟百分位数；Alertmanager 将警告路由到 Slack，严重违规路由到 PagerDuty。

9. **回归探测。** 注入一个 Bug：被评估的聊天机器人开始有 1% 的时间泄露虚假的 SSN。测量 MTTR：从 Bug 部署到 Slack 告警的时间。

## 使用它

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## 发布

`outputs/skill-llm-observability.md` 是可交付成果。给定一个 LLM 应用程序，仪表盘摄取其追踪，运行评估，对漂移发出警报，并在 Next.js 中展示成本/用户分解。

|  权重  |  标准  |  衡量方式  |
|:-:|---|---|
|  25  |  追踪模式覆盖  |  生成规范 GenAI 跨度的 SDK 家族数量（目标：6+）  |
|  20  |  评估正确性  |  DeepEval / RAGAS 分数与手工标注集对比  |
|  20  |  仪表盘 UX  |  注入回归的 MTTR（目标低于 5 分钟）  |
|  20  |  成本/规模  |  持续以 1k 跨度/秒摄入，无积压  |
|  15  |  告警 + 漂移检测  |  Prometheus/Alertmanager 链端到端演练  |
|  **100**  |   |   |

## 练习

1. 为 Haystack 框架添加自定义仪表化。验证规范跨度落入 ClickHouse 并带有准确的 `gen_ai.*` 属性。

2. 在同一追踪上将 DeepEval 替换为 Phoenix 评估器。测量两个评估引擎之间的分数漂移。

3. 优化漂移检测器：按 app-id 而非全局计算 PSI。显示按应用的漂移轨迹。

4. 添加一个“用户影响”页面：每用户成本和每用户失败率，并附带迷你图。

5. 构建一个尾部采样策略，保留 100% 毒性大于 0.5 的追踪，以及其余部分的 10% 分层样本。测量引入的采样偏差。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
|  GenAI 语义约定  |  "OTel LLM 属性"  |  2025 年 OpenTelemetry 规范中关于 LLM 跨度属性（系统、模型、令牌）的定义  |
| 尾部采样  |  "追踪后采样"  |  收集器在追踪完成后决定保留或丢弃（可查看错误） |
| PSI  |  "群体稳定性指数"  |  比较两个分布的漂移指标；>0.2 通常表示有意义的漂移 |
| LLM裁判  |  "模型评估"  |  一个LLM根据评分标准（忠实度、毒性、PII）对另一个LLM的输出进行评分 |
| 尾部采样策略  |  "保留规则"  |  决定哪些追踪保留或丢弃的规则；错误 + 采样率 |
| 评估跨度  |  "关联评估跟踪"  |  携带评估分数的子跨度，与原始LLM调用跨度关联 |
| 每用户成本  |  "单位经济学"  |  在特定时间窗口内归属于某个用户ID的美元成本；关键产品指标 |

## 延伸阅读

- [Langfuse](https://github.com/langfuse/langfuse) — 参考的开源核心可观测性平台
- [Langfuse](https://github.com/langfuse/langfuse) — 具有强大漂移支持的替代参考
- [Langfuse](https://github.com/langfuse/langfuse) — 自动仪表化SDK系列
- [Langfuse](https://github.com/langfuse/langfuse) — 摄入模式
- [Langfuse](https://github.com/langfuse/langfuse) — 替代托管可观测性
- [Langfuse](https://github.com/langfuse/langfuse) — 替代评估优先平台
- [Langfuse](https://github.com/langfuse/langfuse) — 列式跨度存储
- [Langfuse](https://github.com/langfuse/langfuse) — 评估器库
