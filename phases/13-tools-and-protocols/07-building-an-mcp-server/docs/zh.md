# 构建MCP服务器——Python + TypeScript SDK

> 大多数MCP教程只展示stdio的hello-world示例。一个真正的服务器会暴露工具、资源和提示词，处理能力协商，发出结构化错误，并在所有SDK中一致运行。这节课将构建一个端到端的笔记服务器：标准库stdio传输、JSON-RPC分发、三个服务器原语，以及一种纯函数风格，当你进阶时可以无缝接入Python SDK的FastMCP或TypeScript SDK。

**类型：** 构建
**语言：** Python (stdlib, stdio MCP服务器)
**前置条件：** 阶段 13 · 06 (MCP基础)
**时长：** 约75分钟

## 学习目标

- 实现`initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list`和`prompts/get`方法。
- 编写一个分发循环，从stdin读取JSON-RPC消息并将响应写入stdout。
- 根据JSON-RPC 2.0规范及MCP附加状态码发出结构化错误响应。
- 将标准库实现升级到FastMCP（Python SDK）或TypeScript SDK，无需重写工具逻辑。

## 问题

在你可以使用远程传输（阶段13·09）或认证层（阶段13·16）之前，你需要一个干净的本地服务器。本地意味着stdio：服务器由客户端作为子进程启动，消息通过stdin/stdout以换行符分隔方式传输。

2025-11-25规范规定stdio消息编码为JSON对象，带有明确的`\n`分隔符。此处不使用SSE；SSE是旧的远程模式，将在2026年中移除（Atlassian的Rovo MCP服务器于2026年6月30日弃用；Keboola于2026年4月1日弃用）。对于stdio，每行一个JSON对象就是整个线路格式。

笔记服务器是一个好的形态，因为它能练习所有三个服务器原语。工具执行变更（`notes_create`）。资源暴露数据（`notes://{id}`）。提示词提供模板（`review_note`）。这节课的形态可推广到任何领域。

## 核心概念

### 分发循环

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

三条规则：

- 不要向stdout打印任何非JSON-RPC信封的内容。调试日志输出到stderr。
- 每个请求必须与携带相同`id`的响应匹配。
- 通知不得被响应。

### 实现`initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

只声明你支持的内容。客户端依赖能力集来开关特性。

### 实现`tools/list`和`tools/call`

`tools/list`返回`{tools: [...]}`，其中每个条目包含`name`、`description`、`inputSchema`。`tools/call`接收`{name, arguments}`并返回`{content: [blocks], isError: bool}`。

内容块是带类型的。最常见的：

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

工具错误有两种形式。协议级错误（未知方法、错误参数）是JSON-RPC错误。工具级错误（调用有效但工具执行失败）以`{content: [...], isError: true}`返回。这使模型能在其上下文中看到失败。

### 实现资源

资源在设计上是只读的。`resources/list`返回清单；`resources/read`返回内容。URI可以是`file://...`、`http://...`或自定义方案如`notes://`。

当你将数据作为资源而不是工具暴露时：

- 模型不会“调用”它；客户端可以在用户请求时将其注入上下文。
- 订阅允许服务器在资源发生变化时推送更新（阶段13·10）。
- 阶段13·14通过`ui://`扩展了这一点，用于交互式资源。

### 实现提示词

提示词是带有命名参数的模板。宿主将它们作为斜杠命令展示。一个`review_note`提示词可能接收一个`note_id`参数，并生成一个多消息提示词模板，客户端将其提供给其模型。

### Stdio传输的微妙之处

- 以换行符分隔的JSON。没有长度前缀的帧。
- 不要缓冲。每次写入后`sys.stdout.flush()`。
- 客户端控制生命周期。当stdin关闭（EOF）时，干净退出。
- 不要静默处理SIGPIPE；记录并退出。

### 注解

每个工具可以携带`annotations`描述安全特性：

- `readOnlyHint: true` — 纯读取，安全重试。
- `readOnlyHint: true` — 不可逆副作用；客户端应确认。
- `readOnlyHint: true` — 相同输入产生相同输出。
- `readOnlyHint: true` — 与外部系统交互。

客户端利用这些来决定用户体验（确认对话框、状态指示器）和路由（阶段13·17）。

### 升级路径

在`code/main.py`中的标准库服务器大约180行。FastMCP (Python) 将相同的逻辑简化为装饰器风格：

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK 具有相同的结构。毕业路径是即插即用的，只要你准备好了；概念（能力、分发、内容块）是相同的。

## 使用它

`code/main.py` 是一个通过 stdio 运行的完整笔记 MCP 服务器，仅使用 stdlib。它处理三个工具（`notes_list`、`notes_search`、`notes_create`）的 `initialize`、`tools/list`、`tools/call`，每个笔记的 `resources/list` 和 `resources/read`，以及一个 `review_note` 提示。你可以通过管道传递 JSON-RPC 消息来驱动它：

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

需要关注的内容：

- 调度器是一个以方法名作为键的 `dict[str, Callable]`。
- 每个工具执行器返回一个内容块列表，而不是一个裸字符串。
- 当执行器抛出异常时，`dict[str, Callable]` 被设置。

## 发布

本课程产生 `outputs/skill-mcp-server-scaffolder.md`。给定一个领域（笔记、工单、文件、数据库），该技能搭建一个 MCP 服务器，具备正确的工具/资源/提示拆分和 SDK 毕业路径。

## 练习

1. 运行 `code/main.py` 并用手工构建的 JSON-RPC 消息驱动它。执行 `notes_create`，然后 `resources/read` 以检索新笔记。

2. 添加一个带有 `annotations: {destructiveHint: true}` 的 `notes_delete` 工具。验证客户端会弹出一个确认对话框（这需要一个真实的主机；Claude Desktop 可以）。

3. 实现 `resources/subscribe`，使得每当笔记被修改时，服务器推送 `notifications/resources/updated`。添加一个保活任务。

4. 将服务器移植到 FastMCP。Python 文件应缩减到 80 行以下。通信行为必须相同；使用相同的 JSON-RPC 测试工具验证。

5. 阅读规范的 `server/tools` 部分，找出本课程服务器中未实现的工具定义的一个字段。（提示：有多个；选择一个并添加它。）

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
| MCP 服务器  |  "暴露工具的东西"  |  通过 stdio 或 HTTP 进行 MCP JSON-RPC 通信的进程 |
| stdio 传输  |  "子进程模型"  |  服务器由客户端生成；通过 stdin/stdout 通信 |
| 调度器  |  "方法路由器"  |  将 JSON-RPC 方法名映射到处理函数的映射 |
| 内容块  |  "工具结果块"  |  工具响应的 `content` 数组中的类型化元素 |
| `isError`  |  "工具级失败"  |  表示工具失败；与 JSON-RPC 错误区分 |
| 注解  |  "安全提示"  |  readOnly / destructive / idempotent / openWorld 标志 |
| FastMCP  |  "Python SDK"  |  基于装饰器的高级框架，构建在 MCP 协议之上 |
| 资源 URI  |  "可寻址数据"  |  `file://`、`db://` 或自定义方案标识一个资源 |
| 提示模板  |  "斜杠命令简介"  |  服务器提供的模板，带有供主机 UI 使用的参数槽 |
| 能力声明  |  "特性开关"  |  在 `initialize` 中声明的每个原语的标志 |

## 延伸阅读

- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — 参考 Python 实现
- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — 并行 TS 实现
- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — MCP 服务器的装饰器风格 Python API
- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — 使用任一 SDK 的端到端教程
- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — tools/* 消息的完整参考
