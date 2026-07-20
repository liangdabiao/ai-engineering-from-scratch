# MCP基础——原语、生命周期、JSON-RPC基础

> 在MCP之前，每一次集成都是独特的。模型上下文协议(Model Context Protocol, MCP)于2024年11月由Anthropic首次发布，现在由Linux基金会旗下的Agentic AI Foundation管理，它标准化了发现和调用，使得任何客户端都能与任何服务器通信。2025-11-25规范定义了六个原语(三个服务器端，三个客户端)、一个三阶段生命周期以及JSON-RPC 2.0的线格式。掌握这些，本阶段MCP章节的其余部分就变成了阅读。

**类型：** 学习
**语言：** Python (标准库，JSON-RPC解析器)
**前置条件：** 阶段13 · 01到05 (工具接口和函数调用)
**时间：** ~45分钟

## 学习目标

- 列举所有六个MCP原语(服务器端的工具、资源、提示；客户端的根、采样、引导)，并各给出一个用例。
- 介绍三阶段生命周期(初始化、操作、关闭)，并说明每个阶段谁发送什么消息。
- 解析并生成JSON-RPC 2.0请求、响应和通知信封。
- 解释`initialize`处能力协商的作用，以及没有它会出什么问题。

## 问题

在MCP之前，每个使用工具的智能体都有自己的协议。Cursor有一个MCP形状但不相容的工具系统。Claude Desktop使用了另一个。VS Code的Copilot扩展有第三个。一个构建"Postgres查询"工具的团队将同一个工具写了三次，每次都针对不同主机的API。重用需要复制代码。

结果是大量专有集成的寒武纪大爆发，生态系统速度出现了天花板。

MCP通过标准化线格式解决了这个问题。一个MCP服务器可以在所有的MCP客户端中工作：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf，到2026年4月支持超过300个客户端。每月1.1亿次SDK下载。超过10,000个公共服务器。Linux基金会于2025年12月在新的Agentic AI Foundation下接管管理。

本阶段使用的规范版本是**2025-11-25**。它增加了异步任务(SEP-1686)、URL模式引导(SEP-1036)、带工具的采样(SEP-1577)、增量范围同意(SEP-835)以及OAuth 2.1资源指示器语义。阶段13 · 09到16涵盖了这些扩展。本课只讲基础。

## 核心概念

### 三个服务器端原语

1. **工具(Tools).** 可调用的动作。与阶段13 · 01相同的四步循环。
2. **资源(Resources).** 暴露的数据。只读内容，可通过URI访问：`file:///path`、`db://query/...`、自定义方案。
3. **提示(Prompts).** 可复用的模板。宿主UI中的斜杠命令；服务器提供模板，客户端填充参数。

### 三个客户端原语

4. **根(Roots).** 服务器允许接触的URI集合。客户端声明它们；服务器尊重它们。
5. **采样(Sampling).** 服务器请求客户端模型执行补全。使得服务器托管的智能体循环无需服务器端API密钥。
6. **引导(Elicitation).** 服务器在运行过程中向客户端用户请求结构化输入。表单或URL(SEP-1036)。

MCP中的每个能力都恰好属于这六个之一。阶段13 · 10到14将深入探讨每一个。

### 线格式：JSON-RPC 2.0

每条消息都是一个JSON对象，包含以下字段：

- 请求：`{jsonrpc: "2.0", id, method, params}`。
- 响应：`{jsonrpc: "2.0", id, method, params}`。
- 通知：`{jsonrpc: "2.0", id, method, params}` — 没有`{jsonrpc: "2.0", id, result | error}`，不期望响应。

基础规范约有15个方法，按原语分组。重要的有：

- `initialize` / `initialized` (握手)
- `initialize`, `initialized`
- `initialize`, `initialized`, `tools/list`
- `initialize`, `initialized`
- `initialize` (服务器到客户端)
- `initialize`, `initialized`, `tools/list`

### 三阶段生命周期

**阶段1：初始化(initialize).**

客户端发送`initialize`，包含其`capabilities`和`clientInfo`。服务器响应自己的`capabilities`、`serverInfo`以及它所支持的规范版本。客户端在消化响应后发送`notifications/initialized`。此后，任一方都可以根据协商的能力发送请求。

**阶段2：操作(operation).**

双向。客户端调用`tools/list`来发现，然后`tools/call`来调用。如果服务器声明了该能力，它可以发送`sampling/createMessage`。当工具集发生变化时，服务器可能发送`notifications/tools/list_changed`。当用户更改根范围时，客户端可能发送`notifications/roots/list_changed`。

**阶段3：关闭(shutdown).**

任一方关闭传输。MCP中没有结构化的关闭方法；传输(stdin/stdout或Streamable HTTP，阶段13 · 09)携带连接结束信号。

### 能力协商(Capability negotiation)

`capabilities` in the `initialize`握手是契约。服务器示例：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

服务器声明它可以发出`tools/list_changed`通知并支持`resources/subscribe`。客户端通过声明自己的来同意：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

如果客户端没有声明`sampling`，则服务器不得调用`sampling/createMessage`。对称：如果服务器没有声明`resources.subscribe`，则客户端不得尝试订阅。

这正是防止生态系统漂移的原因。不支持采样的客户端仍然是有效的MCP客户端；不调用`sampling`的服务器仍然是有效的MCP服务器。它们只是不同时使用那个功能。

### 结构化内容与错误形状

`tools/call` 返回一个 `content` 类型的类型化块数组：`text`、`image`、`resource`。阶段13·14将MCP应用程序（`ui://` 交互式UI）添加到该列表中。

错误使用JSON-RPC错误码。规范定义的补充：`-32002` "资源未找到"、`-32603` "内部错误"，以及MCP特定的错误数据作为`error.data`。

### 客户端能力与工具调用细节

一个常见的混淆：`capabilities.tools` 是客户端是否支持工具列表变更通知。客户端是否会调用特定工具是由其模型驱动的运行时选择，而不是能力标志。能力标志是规范级别的合约。模型的选择是正交的。

### 为什么是JSON-RPC而不是REST？

JSON-RPC 2.0（2010）是一个轻量级双向协议。REST是客户端发起的。MCP需要服务器发起的消息（采样、通知），因此具有对称请求/响应形状的JSON-RPC很自然。JSON-RPC还能干净地组合在stdio和WebSocket/可流式HTTP上，而无需重新发明HTTP的请求形状。

```figure
mcp-tool-call
```

## 使用它

`code/main.py` 附带一个最小的JSON-RPC 2.0解析器和发射器，然后手动遍历 `initialize` → `tools/list` → `tools/call` → `shutdown` 序列，打印每条消息。没有真实的传输；只是消息形状。与进一步阅读中链接的规范比较，以验证每个信封。

需要关注的内容：

- `initialize` 双向声明能力；响应包含 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `initialize` 返回一个 `serverInfo` 数组；每个条目包含 `protocolVersion: "2025-11-25"`、`tools/list`、`tools`。
- `initialize` 使用 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- 响应 `initialize` 是一个 `serverInfo` 块数组。

## 发布

本课生成 `outputs/skill-mcp-handshake-tracer.md`。给定一个类似pcap的MCP客户端-服务器交互记录，该技能用每个消息所属的原语、生命周期阶段和依赖的能力来注释。

## 练习

1. 运行 `code/main.py`。标识能力协商发生的行，并描述如果服务器未声明 `tools.listChanged` 会发生什么变化。

2. 扩展解析器以处理 `notifications/progress`。消息形状：`{method: "notifications/progress", params: {progressToken, progress, total}}`。在长时间运行的 `tools/call` 进行时发出它，并确认客户端处理器会显示进度条。

3. 从顶部到底部阅读MCP 2025-11-25规范——整个文档大约80页。找出大多数服务器不需要的一个能力标志。提示：它与资源订阅有关。

4. 在纸上勾勒出假设的“cron作业”功能所属的原语。（提示：服务器希望客户端在预定时间调用它。目前六个原语都不适合。）MCP的2026路线图有一个关于此的SEP草案。

5. 从GitHub上的一个开放MCP服务器解析一个会话日志。统计请求、响应和通知消息的数量。计算流量中生命周期与操作的比例。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  MCP  |  "模型上下文协议"  |  用于模型到工具发现和调用的开放协议  |
|  服务器原语  |  "服务器暴露的内容"  |  工具（动作）、资源（数据）、提示（模板）  |
|  客户端原语  |  "客户端允许服务器使用的内容"  |  根（范围）、采样（LLM回调）、启发（用户输入）  |
|  JSON-RPC 2.0  |  "线路格式"  |  对称的请求/响应/通知信封  |
|  `initialize` 握手  |  "能力协商"  |  第一对消息；服务器和客户端声明它们支持的功能  |
|  `tools/list`  |  "发现"  |  客户端向服务器询问其当前工具集  |
|  `tools/call`  |  "调用"  |  客户端请求服务器执行带有参数的工具  |
|  `notifications/*_changed`  |  "变更事件"  |  服务器告知客户端其原语列表已更改  |
|  内容块  |  "类型化结果"  |  工具结果中的 `{type: "text" \ |  "image" \ |  "resource" \ |  "ui_resource"}`  |
|  SEP  |  "规范演进提案"  |  命名草案提案（例如，用于异步任务的SEP-1686）  |

## 延伸阅读

- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 规范文档
- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 六原语心智模型
- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 2024年11月发布帖子
- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 一周年回顾与2025-11-25规范变更
- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — SEP-1686、1036、1577、835和1724的摘要
