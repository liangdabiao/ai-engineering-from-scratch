# OpenAI Agents SDK：交接(Handoff)、护栏(Guardrail)、追踪(Tracing)

> OpenAI Agents SDK 是基于 Responses API 构建的轻量级多智能体框架。五个原语：Agent（智能体）、Handoff（交接）、Guardrail（护栏）、Session（会话）、Tracing（追踪）。交接是名为 `transfer_to_<agent>` 的工具。护栏在输入或输出时触发。追踪默认开启。

**类型：** 学习+构建
**语言：** Python (stdlib)
**前置条件：** 第14阶段 · 01 (智能体循环), 第14阶段 · 06 (工具使用)
**时间：** 约75分钟

## 学习目标

- 说出 OpenAI Agents SDK 的五个原语。
- 解释交接：为什么它们被建模为工具，模型看到的名称形状是什么，以及上下文如何传递。
- 区分输入护栏、输出护栏和工具护栏；解释 `run_in_parallel` 与阻塞模式。
- 实现一个带有交接、护栏和跨度样式追踪的标准库运行时。

## 问题

无法干净委派的智能体最终会将所有内容塞进一个提示中。没有护栏的智能体会泄露PII、输出违反策略的内容，或者永远循环。OpenAI 的 SDK 将使得多智能体工作变得易于处理的三个原语规范化。

## 核心概念

### 五个原语

1. **Agent（智能体）。** LLM + 指令 + 工具 + 交接。
2. **Handoff（交接）。** 委派给另一个智能体。在模型中表示为名为 `transfer_to_<agent_name>` 的工具。
3. **Guardrail（护栏）。** 对输入（仅第一个智能体）、输出（仅最后一个智能体）或工具调用（每个函数工具）进行验证。
4. **Session（会话）。** 跨轮次的自动对话历史。
5. **Tracing（追踪）。** 内置跨度，用于LLM生成、工具调用、交接、护栏。

### 交接作为工具

模型在其工具列表中看到 `transfer_to_billing_agent`。调用它指示运行时：

1. 复制对话上下文（或通过 `nest_handoff_history` beta 将其折叠）。
2. 用目标智能体的指令初始化它。
3. 继续使用目标智能体运行。

这是产品化的监督者模式（第13课/第28课）。

### 防护措施

三种风格：

- **输入护栏。** 在第一个智能体的输入上运行。在任何LLM调用之前拒绝不安全或超出范围的请求。
- **输出护栏。** 在最后一个智能体的输出上运行。捕获PII泄露、违反策略、格式错误的响应。
- **工具护栏。** 在每个函数工具上运行。验证参数、检查权限、审计执行。

模式：

- **并行**（默认）。护栏LLM与主LLM并行运行。降低尾延迟。如果触发，主LLM的工作将被丢弃（令牌浪费）。
- **阻塞**（`run_in_parallel=False`）。护栏LLM先运行。如果触发，主调用不会浪费令牌。

触发线会引发 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`。

### 追踪

默认开启。每次LLM生成、工具调用、交接和护栏都会发出一个跨度。`OPENAI_AGENTS_DISABLE_TRACING=1` 可以选择退出。`add_trace_processor(processor)` 将跨度分派到您自己的后端以及OpenAI的后端。

### 会话

`Session` 在后端（SQLite、Redis、自定义）存储对话历史。`Runner.run(agent, input, session=session)` 自动加载并追加。

### 这种模式出错的地方

- **交接漂移。** 智能体A交接给智能体B，智能体B又交接回智能体A。添加跳数计数器。
- **护栏绕过。** 工具护栏仅在函数工具上触发；内置工具（文件读取器、网络抓取）需要单独的策略。
- **过度追踪。** 跨度中包含敏感内容。与OTel GenAI内容捕获规则（第23课）配合使用——外部存储，通过ID引用。

## 动手构建

`code/main.py` 在标准库中实现了SDK形状：

- `Agent`, `FunctionTool`, `Handoff`（作为具有传递语义的函数工具）。
- `Agent` 带有输入/输出/工具护栏、交接分发和跳数计数器。
- 一个简单的跨度发射器来显示追踪形状。
- 一个分诊智能体，根据用户的查询交接给计费或支持；护栏在某个输入上触发。

运行它：

```
python3 code/main.py
```

追踪显示两次成功的交接、一次输入护栏触发，以及一个反映真实SDK发出的跨度树。

## 使用它

- **OpenAI Agents SDK** 用于以OpenAI为先的产品。
- **Claude Agent SDK**（第17课）用于以Claude为先的产品。
- **LangGraph**（第13课）当你需要显式状态和持久恢复时。
- **自定义**当你需要精确控制时（语音、多提供商、联邦部署）。

## 发布

`outputs/skill-agents-sdk-scaffold.md` 搭建了一个Agents SDK应用，包含分诊智能体、交接、输入/输出/工具护栏、会话存储和追踪处理器。

## 练习

1. 添加一个交接跳数计数器：N次传递后拒绝。追踪该行为。
2. 实现 `nest_handoff_history` 作为一个选项——在传递前将先前的消息折叠成一个摘要。
3. 编写一个阻塞输出护栏。比较会触发它的提示和通过的提示的延迟。
4. 将 `nest_handoff_history` 连接到JSON记录器。每个跨度发出什么形状？
5. 阅读SDK文档。将您的标准库玩具移植到 `nest_handoff_history`。您建模错了什么？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  Agent  |  "LLM + 指令"  |  SDK中的智能体类型；拥有工具和交接  |
|  Handoff  |  "传递"  |  模型调用的工具，用于委派给另一个智能体  |
|  Guardrail  |  "策略检查"  |  对输入/输出/工具调用的验证  |
|  Tripwire  |  "护栏触发"  |  当护栏拒绝时引发的异常  |
|  会话  |  "历史存储"  |  在运行之间持久化的对话记忆  |
|  追踪  |  "跨度"  |  内置的可观测性，覆盖LLM + 工具 + 移交 + 护栏  |
|  阻塞式护栏  |  "顺序检查"  |  护栏先运行；不会浪费旅程中的令牌  |
|  并行式护栏  |  "并发检查"  |  护栏同时运行；延迟更低，但会浪费旅程中的令牌  |

## 延伸阅读

- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — 原语、移交、护栏、追踪
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — Claude 风格的对应
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — 何时使用移交
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — 标准 Agents SDK 跨度映射到
