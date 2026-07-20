## §13 OpenAI Agents SDK 实战：交接与护栏

我用 Responses API 手写过交接。后来发现 SDK 把交接封装成"工具"，脑子一下清了。

### 13.1 五个原语

Agent、Handoff、Guardrail、Session、Tracing。

基于 Responses API，轻量。

> **标叔的经验**：交接即工具
>
> 我一开始以为 handoff 是特殊机制。后来发现：它就是个名叫 `transfer_to_x` 的工具。

### 13.2 Handoff：把"委派"当工具

模型看到 `transfer_to_billing_agent`，调用它，运行时复制上下文、加载目标 agent。

本质是把 supervisor 模式产品化。

### 13.3 Guardrail：三道防线

- 输入护栏：第一个 agent 的输入，先拦不安全/越界。
- 输出护栏：最后一个 agent 的输出，抓 PII 泄露。
- 工具护栏：每个函数工具的参数校验。

两种模式：

- 并行（默认）：护栏与主模型同时跑，延迟低，但跳闸会浪费主模型 token。
- 阻塞：护栏先跑，跳闸就不花主模型 token。

> **注意**：护栏能被绕过
>
> 工具护栏只对 function tool 生效。内置工具（读文件、抓网页）要另配策略。

### 13.4 Tracing 默认开

每次生成、工具调用、handoff、guardrail 都发 span。`OPENAI_AGENTS_DISABLE_TRACING=1` 可关。

### 13.5 Session

`Runner.run(agent, input, session=session)` 自动加载/追加历史。SQLite、Redis 都行。

[向前桥接] 框架两家讲完。下一章，最野的场景：让 Agent 操作电脑。
