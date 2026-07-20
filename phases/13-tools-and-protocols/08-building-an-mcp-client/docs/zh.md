# 构建MCP客户端 — 发现、调用、会话管理

> 大多数MCP内容提供服务器教程而对客户端一笔带过。客户端代码才是复杂编排所在：进程生成、能力协商、跨多个服务器的工具列表合并、采样回调、重连以及命名空间冲突解决。本节课构建一个多服务器客户端，将三个不同的MCP服务器提升到模型的一个扁平工具命名空间中。

**类型：** 构建
**语言：** Python（标准库，多服务器MCP客户端）
**前置条件：** 阶段13·07（构建MCP服务器）
**时长：** 约75分钟

## 学习目标

- 以子进程方式生成一个MCP服务器，完成`initialize`，并发送一个`notifications/initialized`。
- 维护每个服务器的会话状态（能力、工具列表、上次看到的通知ID）。
- 将多个服务器的工具列表合并到一个命名空间，并处理冲突。
- 将工具调用路由到拥有它的服务器，并重组响应。

## 问题

一个真实的代理主机（Claude Desktop、Cursor、Goose、Gemini CLI）会同时加载多个MCP服务器。用户可能同时运行文件系统服务器、Postgres服务器和GitHub服务器。客户端的工作是：

1. 生成每个服务器。
2. 独立握手每个服务器。
3. 在每个服务器上调用`tools/list`并展平结果。
4. 当模型发出`tools/list`时，在合并的命名空间中查找并路由到正确的服务器。
5. 处理来自任何服务器的通知（`tools/list`），不阻塞。
6. 在传输失败时重连。

亲手实现所有这些是将“玩具”与“可用产品”区分开来的关键。官方SDK封装了这些，但思维模型必须是你自己的。

## 核心概念

### 子进程生成

使用`stdin=PIPE, stdout=PIPE, stderr=PIPE`进行`subprocess.Popen`。设置`bufsize=1`并使用文本模式进行逐行读取。每个服务器是一个进程；客户端为每个服务器持有一个`Popen`句柄。

### 每个服务器的会话状态

每个服务器有一个`Session`对象，包含：

- `process` — Popen句柄。
- `process` — 服务器在`capabilities`处声明的内容。
- `process` — 上次`capabilities`的结果。
- `process` — 请求ID到等待响应的Promise/Future的映射。

请求本质上是异步的；在服务器B正在调用时发送给服务器A的`tools/call`不得阻塞。可以使用带队列的线程或asyncio。

### 合并的命名空间

当客户端看到聚合的工具列表时，名称可能冲突。两个服务器可能都公开`search`。客户端有三个选项：

1. **按服务器名称添加前缀。** `notes/search`, `files/search`。清晰但丑陋。
2. **静默先到先得。** 后到的服务器的`notes/search`覆盖先前的。有风险；隐藏冲突。
3. **冲突拒绝。** 拒绝加载第二个服务器；通知用户。对于安全敏感的主机最安全。

Claude Desktop使用按服务器添加前缀。Cursor使用冲突拒绝并给出明确错误。VS Code MCP也采用按服务器添加前缀。

### 路由

合并后，一个调度表映射`tool_name -> session`。模型按名称发出调用；客户端找到会话并向该服务器的stdin写入一条`tools/call`消息，然后等待响应。

### 采样回调

如果服务器在`initialize`处声明了`sampling`能力，它可能会发送`sampling/createMessage`请求客户端运行其LLM。客户端必须：

1. 阻塞对该服务器的进一步请求，直到采样解析完成，或者如果其实现支持并发则进行流水线处理。
2. 调用其LLM提供商。
3. 将响应发送回服务器。

第11课涵盖端到端的采样。本课为了完整性提供了一个存根。

### 通知处理

`notifications/tools/list_changed`意味着重新调用`tools/list`。`notifications/resources/updated`意味着如果资源正在使用则重新读取。通知不得产生响应——不要尝试确认它们。

一个常见的客户端错误：当通知在流中时，阻塞`tools/call`上的读取循环。使用后台读取器线程将每条消息推入队列；主线程出队并调度。

### 重连

传输可能失败：服务器崩溃、操作系统终止进程、stdio管道断开。客户端检测到stdout上的EOF并将该会话视为死亡。选项：

- 静默重启服务器并重新握手。适用于纯只读服务器。
- 将失败信息告知用户。适用于具有用户可见会话的有状态服务器。

阶段13·09涵盖可流式HTTP重连语义；stdio更简单。

### Keepalive 与会话 ID

Streamable HTTP 使用 `Mcp-Session-Id` 头。Stdio 没有会话 ID——进程身份本身就是会话。Keepalive 心跳是可选的；stdio 管道在不活动时不会中断。

## 使用它

`code/main.py` 启动三个模拟的 MCP 服务器作为子进程，与每个服务器握手，合并它们的工具列表，并将工具调用路由到正确的服务器。这些“服务器”实际上是运行玩具响应器的其他 Python 进程（没有真正的 LLM）。运行它可以看到：

- 三次初始化，每轮都有各自的能力集。
- 三个 `tools/list` 结果合并为一个包含 7 个工具的命名空间。
- 基于工具名称的路由决策。
- 通过命名空间前缀避免冲突。

需要关注的内容：

- `Session` 数据类清晰地保存了每个服务器的状态。
- 后台读取器线程在不阻塞主线程的情况下逐行读取 stdout。
- 调度表是一个简单的 `Session`。
- 冲突处理是显式的：当两个服务器声明相同名称时，后一个会被重命名并加上前缀。

## 发布

本课产生 `outputs/skill-mcp-client-harness.md`。给定一个声明式的 MCP 服务器列表（名称、命令、参数），该技能生成一个框架，用于启动这些服务器、合并工具列表，并提供带有冲突解决的路由函数。

## 练习

1. 运行 `code/main.py` 并观察服务器启动日志。用 SIGTERM 杀死其中一个模拟服务器进程，观察客户端如何检测 EOF 并将该会话标记为死亡。

2. 实现命名空间前缀。当两个服务器暴露 `search` 时，将第二个重命名为 `<server>/search`。更新调度表并验证工具调用是否正确路由。

3. 为服务器重启添加连接池风格的回退：连续失败时指数退避，上限为 30 秒，三次失败后向用户发送通知。

4. 设计一个支持 100 个并发 MCP 服务器的客户端。什么数据结构可以替代简单的调度字典？（提示：用于前缀命名空间的 trie，加上每个服务器的工具数量指标。）

5. 将客户端移植到官方的 MCP Python SDK。该 SDK 封装了 `stdio_client` 和 `ClientSession`。代码应从约 200 行缩减到约 40 行，同时保留多服务器路由。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
| MCP 客户端 | "代理主机" | 启动服务器并编排工具调用的进程 |
| 会话 | "每个服务器的状态" | 能力、工具列表和待处理请求的记账 |
| 合并命名空间 | "一个工具列表" | 所有活动服务器上工具名称的扁平集合 |
| 命名空间冲突 | "两个服务器相同工具" | 客户端必须添加前缀、拒绝或先到先得 |
| 路由 | "谁处理这个调用？" | 从工具名称到所属服务器的调度 |
| 后台读取器 | "非阻塞 stdout" | 将服务器 stdout 排入队列的线程或任务 |
| 采样回调 | "LLM 即服务" | 客户端处理 `sampling/createMessage` 来自服务器 |
| `notifications/*_changed` | "原始状态变化" | 信号客户端必须重新发现或重新读取 |
| 重连策略 | "服务器宕机时" | 传输失败时的重启语义 |
| Stdio 会话 | "进程 = 会话" | 无会话 ID；子进程生命周期即为会话 |

## 延伸阅读

- [Model Context Protocol — Client spec](https://modelcontextprotocol.io/specification/2025-11-25/client) — 标准客户端行为
- [Model Context Protocol — Client spec](https://modelcontextprotocol.io/specification/2025-11-25/client) — 使用 Python SDK 的 Hello World 客户端教程
- [Model Context Protocol — Client spec](https://modelcontextprotocol.io/specification/2025-11-25/client) — 参考 [MCP — Quickstart client guide](https://modelcontextprotocol.io/quickstart/client) 和 [MCP Python SDK — client module](https://github.com/modelcontextprotocol/python-sdk)
- [Model Context Protocol — Client spec](https://modelcontextprotocol.io/specification/2025-11-25/client) — TS 并行
- [Model Context Protocol — Client spec](https://modelcontextprotocol.io/specification/2025-11-25/client) — VS Code 如何在单个编辑器主机中复用多个 MCP 服务器
