# 函数调用深度解析——OpenAI、Anthropic、Gemini

> 2024年，三家前沿提供商在工具调用循环上达成一致，随后又在其他方面分道扬镳。OpenAI使用`tools`和`tool_calls`。Anthropic使用`tool_use`和`tool_result`块。Gemini使用`functionDeclarations`和唯一ID关联。本节将三家方案并排比较，确保代码在某一提供商上运行后，移植到其他平台时不会出错。

**类型：**构建
**语言：**Python（标准库、模式转换器）
**前置条件：**第13阶段·01（工具接口）
**时长：**约75分钟

## 学习目标

- 阐述OpenAI、Anthropic和Gemini在函数调用载荷（声明、调用、结果）上的三个形状差异。
- 将一条工具声明翻译成三家提供商的格式，并预测严格模式约束可能不同的地方。
- 在各提供商中使用`tool_choice`强制、禁止或自动选择工具调用。
- 了解各提供商的硬限制（工具数量、模式深度、参数长度），以及违反限制时返回的错误特征。

## 问题

函数调用请求的形状因提供商而异。以下是2026年生产环境中的三个具体示例：

**OpenAI Chat Completions / Responses API。** 你传入`tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型的响应包含`choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中`arguments`是一个必须解析的JSON字符串。严格模式（`strict: true`）通过约束解码来强制执行模式合规。

**Anthropic Messages API。** 你传入`tools: [{name, description, input_schema}]`。响应以`content: [{type: "text"}, {type: "tool_use", id, name, input}]`的形式返回。`input`已经被解析（一个对象，而非字符串）。你需要回复一条新的`user`消息，其中包含`{type: "tool_result", tool_use_id, content}`块。

**Google Gemini API。** 你传入`tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在`functionDeclarations`下）。响应以`candidates[0].content.parts: [{functionCall: {name, args, id}}]`的形式到达，其中`id`在Gemini 3及以上版本中是唯一的，用于并行调用关联。你需要回复`{functionResponse: {name, id, response}}`。

相同的循环。不同的字段名称、不同的嵌套方式、不同的字符串与对象约定、不同的关联机制。一个在OpenAI上编写天气代理的团队，在移植到Anthropic时需要两天，再到Gemini时又需要一天，仅仅是为了调整通信管道。

本节构建一个转换器，将三种格式统一为一种规范的工具声明，并在边缘路由。第13阶段·17将相同的模式泛化为一个LLM网关。

## 核心概念

### 通用结构

每个提供商都需要五样东西：

1. **工具列表。** 每个工具的名称、描述和输入模式。
2. **工具选择。** 强制使用特定工具、禁止工具或让模型决定。
3. **调用发出。** 结构化输出，指定工具和参数。
4. **调用ID。** 将响应关联到正确的调用（在并行场景中很重要）。
5. **结果注入。** 一条消息或块，将结果与调用关联起来。

### 形状差异，逐一字段

|  方面  |  OpenAI  |  Anthropic  |  Gemini  |
|--------|--------|-----------|--------|
|  声明外壳  |  `{type: "function", function: {...}}`  |  `{name, description, input_schema}`  |  `{functionDeclarations: [{...}]}`  |
|  模式字段  |  `parameters`  |  `input_schema`  |  `parameters`  |
|  响应容器  |  助手消息上的`tool_calls[]`  |  类型为`tool_use`的`content[]`  |  类型为`functionCall`的`parts[]`  |
|  参数类型  |  字符串化的JSON  |  已解析的对象  |  已解析的对象  |
|  ID格式  |  `call_...`（OpenAI生成）  |  `toolu_...`（Anthropic）  |  UUID（Gemini 3+）  |
|  结果块  |  角色`tool`，`tool_call_id`  |  包含`tool_result`、`tool_use_id`的`user`  |  包含匹配`id`的`functionResponse`  |
|  强制工具  |  `tool_choice: {type: "function", function: {name}}`  |  `tool_choice: {type: "tool", name}`  |  `tool_config: {function_calling_config: {mode: "ANY"}}`  |
|  禁止工具  |  `tool_choice: "none"`  |  `tool_choice: {type: "none"}`  |  `mode: "NONE"`  |
|  严格模式  |  `strict: true`  |  模式即模式（始终强制执行）  |  请求级别的`responseSchema`  |

### 你实际会遇到的上限

- **OpenAI。** 每个请求最多128个工具。模式深度为5。参数字符串不超过8192字节。严格模式要求不能有`$ref`、不能有重叠的`oneOf`/`anyOf`/`allOf`，且每个属性都必须在`required`中列出。
- **Anthropic。** 每个请求最多64个工具。模式深度理论上无限制，但实际限制为10。没有严格模式标志；模式是一种合约，模型倾向于遵守。
- **Gemini。** 每个请求最多64个函数。模式类型是OpenAPI 3.0子集（与JSON Schema 2020-12略有不同）。自Gemini 3起，并行调用使用唯一ID。

### `tool_choice`行为

所有提供商都支持三种模式，但命名不同。

- **自动。** 模型选择工具或文本。默认。
- **必需/任意。** 模型必须调用至少一个工具。
- **无。** 模型不得调用工具。

此外，每个提供商还有自己特有的一种模式：

- **OpenAI.** 按名称强制使用特定工具。
- **Anthropic.** 按名称强制使用特定工具；`disable_parallel_tool_use` 标志区分单次与多次。
- **Gemini.** `disable_parallel_tool_use` 使每次响应都通过架构验证器，无论模型意图如何。

### 并行调用

OpenAI 的 `parallel_tool_calls: true`（默认）在一条助手消息中发出多次调用。你需要执行所有调用，并回复一条包含每条 `tool_call_id` 对应条目的批处理工具角色消息。Anthropic 历史上是单次调用；`parallel_tool_calls: true`（自 Claude 3.5 起默认）启用多次调用。Gemini 2 允许并行调用但未提供稳定 ID；Gemini 3 添加了 UUID，以便乱序响应能正确关联。

### 流式传输

三者都支持流式工具调用。线路格式不同：

- **OpenAI.** `tool_calls[i].function.arguments` 的增量块逐步到达。你需要累积直到 `finish_reason: "tool_calls"`。
- **Anthropic.** 块开始/块增量/块停止事件。`tool_calls[i].function.arguments` 块携带部分参数。
- **Gemini.** `tool_calls[i].function.arguments`（Gemini 3 新增）发出带有 `finish_reason: "tool_calls"` 的块，以便多个并行调用可以交错。

阶段 13 · 03 深入研究并行+流式重组。本课重点在声明和单次调用形状。

### 错误与修复

无效参数错误的呈现方式也不同。

- **OpenAI（非严格模式）.** 模型返回 `arguments: "{bad json}"`，你的 JSON 解析失败，你注入错误消息并重新调用。
- **OpenAI（严格模式）.** 验证在解码期间进行；无效 JSON 不可能出现，但可能出现 `arguments: "{bad json}"`。
- **Anthropic.** `arguments: "{bad json}"` 可能包含意外字段；架构仅供参考。需要服务端验证。
- **Gemini.** OpenAPI 3.0 的异常：对象字段上的 `arguments: "{bad json}"` 被静默忽略；需要自行验证。

### 翻译器模式

代码中的规范工具声明如下所示（你可以选择形状）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小函数将其转换为三种提供者的形状。`code/main.py` 中的测试框架正是这样做的，然后通过每个提供者的响应形状往返模拟一次工具调用。无需网络——本课教的是形状，而非 HTTP。

生产团队将此翻译器封装在 `AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）中。阶段 13 · 17 提供了一个网关，在任意三者之前暴露 OpenAI 形状的 API。

## 使用它

`code/main.py` 定义了一个规范的 `Tool` 数据类和三个翻译器，它们生成 OpenAI、Anthropic 和 Gemini 的声明 JSON。然后，它将每个提供者形状的手工构建的响应解析为相同的规范调用对象，证明语义在底层是相同的。运行它并并排比较三个声明。

需要关注的内容：

- 三个声明块仅在封装和字段名称上有所不同。
- 三个响应块的不同之处在于调用的位置（顶层 `tool_calls`、`content[]` 块、`parts[]` 条目）。
- 一个 `tool_calls` 函数从所有三种响应形状中提取 `content[]`。

## 发布

本课产出 `outputs/skill-provider-portability-audit.md`。给定一个针对某个提供者的函数调用集成，该技能产生一份可移植性审计：依赖哪些提供者限制、哪些字段需要重命名、移植到其他提供者时会破坏什么。

## 练习

1. 运行 `code/main.py` 并验证三个提供者的声明 JSON 都序列化了相同的底层 `Tool` 对象。修改规范工具添加一个枚举参数，并确认只有 Gemini 翻译器需要处理 OpenAPI 异常。

2. 为每个提供者添加一个 `ListToolsResponse` 解析器，用于提取模型在 `list_tools` 或发现调用后返回的工具列表。OpenAI 本身没有这个功能；注意这种不对称性。

3. 实现 `tool_choice` 转换：将规范的 `ToolChoice(mode="force", tool_name="x")` 映射到所有三种提供者形状。然后映射 `mode="any"` 和 `mode="none"`。查看课程的差异表。

4. 选择三个提供者之一，完整阅读其函数调用指南。在其架构规范中找出一个其他两个提供者不支持的字段。候选：OpenAI `strict`、Anthropic `disable_parallel_tool_use`、Gemini `function_calling_config.allowed_function_names`。

5. 编写一个测试向量：一个参数违反声明架构的工具调用。通过每个提供者的验证器（课程 01 中的 stdlib 版本可作为代理）运行它，并记录哪些错误触发。文档说明在生产环境中你会选择哪个提供者以保证严格性。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  函数调用（Function calling）  |  "工具使用（Tool use）"  |  提供者级别的结构化工具调用发出 API  |
|  工具声明（Tool declaration）  |  "工具规范（Tool spec）"  |  名称 + 描述 + JSON Schema 输入载荷  |
|  `tool_choice`  |  "强制/禁止（Force / forbid）"  |  自动/必需/无/特定名称模式  |
|  严格模式（Strict mode）  |  "架构强制（Schema enforcement）"  |  OpenAI 标志，约束解码以匹配架构  |
|  `tool_use` 块  |  "Anthropic 的调用形状"  |  内联内容块，包含 id、name、input  |
|  `functionCall` 部分  |  "Gemini 的调用形状"  |  一个包含 name、args 和 id 的 `parts[]` 条目  |
|  参数作为字符串（Arguments-as-string）  |  "字符串化的 JSON"  |  OpenAI 将 args 作为 JSON 字符串返回，而非对象  |
|  并行工具调用（Parallel tool calls）  |  "单轮扇出（Fan-out in one turn）"  |  一条助手消息中的多次工具调用  |
| 拒绝  |  "模型拒绝"  |  仅严格模式的拒绝块而非调用 |
| OpenAPI 3.0 子集  |  "Gemini 模式怪癖"  |  Gemini 使用类似 JSON Schema 的方言，但略有不同 |

## 延伸阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — 包含严格模式和并行调用的规范参考
- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) 和 `tool_use` 块语义
- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — 并行调用、唯一 ID 和 OpenAPI 子集
- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — Gemini 的企业界面
- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — 严格模式模式强制实施细节
