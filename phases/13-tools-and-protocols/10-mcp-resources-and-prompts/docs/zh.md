# MCP 资源与提示——超越工具的上下文暴露

> 工具占据了MCP关注的90%。另外两个服务器原语解决了不同的问题。资源（Resources）暴露数据供读取；提示（Prompts）暴露可复用的模板作为斜杠命令。许多服务器应该使用资源而不是将读取包装在工具中，使用提示而不是在客户端提示中硬编码工作流。本节课命名了决策规则并讲解了`resources/*`和`prompts/*`消息。

**类型：** 构建
**语言：** Python（标准库，资源 + 提示处理程序）
**前置条件：** 阶段13 · 07（MCP服务器）
**时间：** 约45分钟

## 学习目标

- 对于给定领域，决定将能力暴露为工具、资源还是提示。
- 实现`resources/list`、`resources/read`、`resources/subscribe`并处理`notifications/resources/updated`。
- 使用参数模板实现`resources/list`和`resources/read`。
- 识别主机何时将提示暴露为斜杠命令与自动注入上下文。

## 问题

一个用于笔记应用的朴素MCP服务器将所有内容暴露为工具：`notes_read`、`notes_list`、`notes_search`。这将每次数据访问包装在模型驱动的工具调用中。后果：

- 模型必须为每个可能受益于上下文的查询决定是否调用`notes_read`。
- 只读内容无法订阅或流式传输到主机的侧面板。
- 客户端UI（Claude Desktop的资源附件面板、Cursor的“包含文件”选择器）无法显示数据。

正确的划分：将数据暴露为资源，将可变或计算操作暴露为工具，将可复用的多步骤工作流暴露为提示。每个原语都有其用户体验特性和访问模式。

## 核心概念

### 工具 vs 资源 vs 提示——决策规则

|  能力  |  原语  |
|------------|-----------|
|  用户想要搜索、筛选或转换数据  |  工具  |
|  用户希望主机将此数据作为上下文包含  |  资源  |
|  用户想要一个可重运行的可模板化工作流  |  提示  |

指导原则：如果模型会在每个相关查询中受益于调用它，则它是个工具。如果用户会受益于将其附加到对话中，则它是个资源。如果整个多步骤工作流是用户想要复用的单元，则它是个提示。

### 资源

`resources/list`返回`{resources: [{uri, name, mimeType, description?}]}`。`resources/read`接受`{uri}`并返回`{contents: [{uri, mimeType, text | blob}]}`。

URI可以是任何可寻址的东西：

- `file:///Users/alice/notes/mcp.md`
- `file:///Users/alice/notes/mcp.md`
- `file:///Users/alice/notes/mcp.md`（自定义方案）
- `file:///Users/alice/notes/mcp.md`（服务器特定）

`contents[]`支持文本和二进制。二进制使用`blob`作为base64编码字符串加`mimeType`。

### 资源订阅

在能力中声明`{resources: {subscribe: true}}`。客户端调用`resources/subscribe {uri}`。当资源改变时服务器发送`notifications/resources/updated {uri}`。客户端重新读取。

用例：一个笔记服务器，其资源是磁盘上的文件；文件监视器触发更新通知；当文件在主机外部被编辑时，Claude Desktop重新拉取文件到上下文中。

### 资源模板（2025-11-25添加）

`resourceTemplates`允许你暴露参数化的URI模式：`notes://{id}`，其中`id`作为补全目标。客户端可以在资源选择器中自动补全ID。

### 提示

`prompts/list`返回`{prompts: [{name, description, arguments?}]}`。`prompts/get`接受`{name, arguments}`并返回`{description, messages: [{role, content}]}`。

提示是一个模板，填充后会生成主机提供给其模型的消息列表。例如，一个`code_review`提示接受一个`file_path`参数，并返回一个三条消息的序列：一条系统消息、一条包含文件体的用户消息，以及一条带有推理模板的助手引导消息。

### 主机与提示

Claude Desktop、VS Code和Cursor将提示暴露为聊天UI中的斜杠命令。用户输入`/code_review`并从表单中选择参数。服务器的提示是“用户快捷方式”和“发送给模型的完整提示”之间的契约。

并非所有客户端都支持提示——检查能力协商。一个声明了提示能力但客户端不支持提示的服务器，根本不会显示斜杠命令。

### "列表已更改"通知

资源集和提示集发生变化时都会发出`notifications/list_changed`。一个刚刚导入20条新笔记的笔记服务器发出`notifications/resources/list_changed`；客户端重新调用`resources/list`以获取新增内容。

### 内容类型约定

对于文本：`mimeType: "text/plain"`, `text/markdown`, `application/json`。
对于二进制：`image/png`, `application/pdf`，加上`blob`字段。
对于MCP应用（第14课）：`text/html;profile=mcp-app`在一个`ui://` URI中。

### 动态资源

资源URI不一定对应静态文件。`notes://recent`可以在每次读取时返回最新的五条笔记。`db://query/users/active`可以执行参数化查询。服务器可以自由地动态计算内容。

规则：如果客户端可以按URI缓存，则URI必须稳定。如果计算是一次性的，URI应包含时间戳或临时值（nonce），以防止客户端缓存过期。

### 订阅与轮询

支持订阅的客户端通过`notifications/resources/updated`获取服务器推送。预订阅客户端或不支持它的主机通过重新读取进行轮询。两者都符合规范。服务器的能力声明告知客户端其支持哪种方式。

订阅的成本：服务器上的每个会话状态（谁订阅了什么）。保持订阅集合有界；断开的客户端应超时。

### 提示词与系统提示词

MCP中的提示词不是系统提示词。主机的系统提示词（其自身的操作指令）和MCP提示词（由用户调用的服务器提供的模板）并存。行为规范的客户端绝不允许服务器提示词覆盖自身的系统提示词；而是将它们分层。

## 使用它

`code/main.py`扩展了第07课的笔记服务器，新增了：

- 每条笔记的资源（`notes://note-1`等），支持`resources/subscribe`。
- 一个`notes://note-1`提示词，呈现为三条消息的模板。
- 一个文件监视器模拟，当笔记被修改时发出`notes://note-1`。
- 一个`notes://note-1`动态资源，始终返回最新的五条笔记。

运行演示以查看完整流程。

## 发布

本课生成`outputs/skill-primitive-splitter.md`。给定一个提议的MCP服务器，该技能将每个能力分类为工具/资源/提示词，并附上理由。

## 练习

1. 运行`code/main.py`。观察初始资源列表，然后触发笔记编辑并验证`notifications/resources/updated`事件触发。

2. 添加一个`resources/list_changed`发射器：当创建新笔记时，发送通知以便客户端重新发现。

3. 为GitHub MCP服务器设计三个提示词：`summarize_pr`, `triage_issue`, `release_notes`。每个都带有参数模式。提示词主体应可直接运行，无需进一步编辑。

4. 取第07课服务器中的一个现有工具，分类它是应保持为工具还是拆分为资源加工具对。用一句话说明理由。

5. 阅读规范的`server/resources`和`server/prompts`部分。找出`resources/read`中很少填充但规范支持的字段。提示：查看资源内容上的`_meta`。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
| 资源 | "暴露的数据" | 主机可读取的URI可寻址内容 |
| 资源URI | "数据指针" | 带方案前缀的标识符（`file://`, `notes://`等） |
| `resources/subscribe` | "监听变化" | 客户端选择加入的特定URI服务器推送更新 |
| `notifications/resources/updated` | "资源已更改" | 向客户端发出信号，表示订阅的资源有新内容 |
| 资源模板 | "参数化URI" | 带有完成提示的URI模式，供主机选择器使用 |
| 提示词 | "斜杠命令模板" | 具名多消息模板，带参数槽 |
| 提示词参数 | "模板输入" | 主机在渲染前收集的带类型参数 |
| `prompts/get` | "渲染模板" | 服务器返回填充后的消息列表 |
| 内容块 | "类型化块" | `{type: text \ |  image \ |  resource \ |  ui_resource}` |
| 斜杠命令用户体验 | "用户快捷方式" | 主机将提示词呈现为以`/`开头的命令 |

## 延伸阅读

- [MCP — Concepts: Resources](https://modelcontextprotocol.io/docs/concepts/resources) — 资源URI、订阅和模板
- [MCP — Concepts: Prompts](https://modelcontextprotocol.io/docs/concepts/prompts) — 提示词模板与斜杠命令集成
- [MCP — Server resources spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) — 完整的`resources/*`消息参考
- [MCP — Server prompts spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) — 完整的`prompts/*`消息参考
- [MCP — Protocol info site: resources](https://modelcontextprotocol.info/docs/concepts/resources/) — 官方文档的社区扩展指南
