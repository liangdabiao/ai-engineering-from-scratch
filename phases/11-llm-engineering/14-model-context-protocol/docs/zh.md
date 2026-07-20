# 模型上下文协议 (MCP)

> 每个在2025年之前构建的LLM应用都发明了自己的工具模式。随后Anthropic推出了MCP，Claude采用了它，OpenAI也采用了它，到2026年，它已成为连接任何LLM与任何工具、数据源或代理的默认传输格式。编写一个MCP服务器，所有主机都能与之通信。

**类型：** 构建
**语言：** Python
**前置条件：** 第11阶段·09（函数调用），第11阶段·03（结构化输出）
**时间：** 约75分钟

## 问题

你发布了一个需要三个工具的聊天机器人：数据库查询、日历API和文件读取器。你为Claude编写了三个JSON模式。然后销售部门希望ChatGPT中也有同样的工具——你为OpenAI的`tools`参数重写了它们。接着你又添加了Cursor、Zed和Claude Code——又是三次重写，每次都有细微不同的JSON约定。一周后，Anthropic增加了一个新字段；你更新了六个模式。

这就是2025年之前的现实。每个主机（运行LLM的东西）和每个服务器（暴露工具和数据的东西）都使用了定制的协议。扩展意味着N×M的集成矩阵。

模型上下文协议（Model Context Protocol）打破了那个矩阵。一个基于JSON-RPC的规范。一个服务器暴露工具、资源和提示。任何符合规范的主机——Claude Desktop、ChatGPT、Cursor、Claude Code、Zed以及大量的代理框架——都能发现并调用它们，无需自定义胶水代码。

截至2026年初，MCP已成为三大公司（Anthropic、OpenAI、Google）及所有主要代理框架中的默认工具和上下文协议。

## 核心概念

![MCP: one host, one server, three capabilities](../assets/mcp-architecture.svg)

**三个原语。** 一个MCP服务器恰好暴露三样东西。

1. **工具** — 模型可以调用的函数。类比OpenAI的`tools`或Anthropic的`tool_use`。每个工具都有名称、描述、JSON Schema输入和一个处理程序。
2. **资源** — 模型或用户可以请求的只读内容（文件、数据库行、API响应）。通过URI寻址。
3. **提示** — 用户可以作为快捷方式调用的可重用模板化提示。

**传输格式。** 基于stdio、WebSocket或可流式HTTP的JSON-RPC 2.0。每条消息都是`{"jsonrpc": "2.0", "method": "...", "params": {...}, "id": N}`。发现方法是`tools/list`、`resources/list`、`prompts/list`。调用方法是`tools/call`、`resources/read`、`prompts/get`。

**主机 vs 客户端 vs 服务器。** 主机是LLM应用（Claude Desktop）。客户端是主机的一个子组件，只与一个服务器通信。服务器是你的代码。一个主机可以同时挂载多个服务器。

### 握手

每个会话以`initialize`开始。客户端发送协议版本及其能力。服务器以其版本、名称及其支持的能力集（`tools`、`resources`、`prompts`、`logging`、`roots`）进行响应。之后的所有内容都根据这些能力进行协商。

### MCP不是什么

- 不是检索API。RAG（第11阶段·06）仍然决定提取什么；MCP是将检索结果作为资源暴露的传输层。
- 不是代理框架。MCP是管道；像LangGraph、PydanticAI和OpenAI Agents SDK这样的框架位于其之上。
- 不绑定于Anthropic。规范和参考实现通过`modelcontextprotocol`组织以开源形式提供。

## 动手构建

### 第1步：一个最小的MCP服务器

官方Python SDK是`mcp`（原`mcp-python`）。高级`FastMCP`辅助函数装饰处理程序。

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@mcp.resource("config://app")
def app_config() -> str:
    """Return the app's current JSON config."""
    return '{"env": "prod", "region": "us-east-1"}'

@mcp.prompt()
def code_review(language: str, code: str) -> str:
    """Review code for correctness and style."""
    return f"You are a senior {language} reviewer. Review:\n\n{code}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

三个装饰器注册了三个原语。类型提示成为主机看到的JSON模式。在Claude Desktop或Claude Code下运行，服务器入口指向此文件。

### 第2步：从主机调用MCP服务器

官方Python客户端使用JSON-RPC。将其与Anthropic SDK配对只需十几行代码。

```python
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp import ClientSession

params = StdioServerParameters(command="python", args=["server.py"])

async def call_add(a: int, b: int) -> int:
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool("add", {"a": a, "b": b})
            return int(result.content[0].text)
```

`session.list_tools()`返回LLM将看到的相同模式。生产环境中的主机将这些模式注入每一轮，以便模型可以发出一个`tool_use`块，然后客户端将其转发给服务器。

### 第3步：可流式HTTP传输

Stdio适用于本地开发。对于远程工具，使用可流式HTTP——每个请求一个POST，可选的服务器推送事件（SSE）用于进度，自2025-06-18规范修订版起支持。

```python
# Inside the server entrypoint
mcp.run(transport="streamable-http", host="0.0.0.0", port=8765)
```

主机配置（Claude Desktop `mcp.json` 或 Claude Code `~/.mcp.json`）：

```json
{
  "mcpServers": {
    "demo": {
      "type": "http",
      "url": "https://tools.example.com/mcp"
    }
  }
}
```

服务器保持不变的设计器；仅传输方式改变。

### 第4步：作用域与安全性

MCP工具是在他人信任边界上运行的任意代码。三个强制性模式。

- **能力白名单。** 主机暴露一个`roots`能力，以便服务器只看到允许的路径。在工具处理程序中强制实施；不要信任模型提供的路径。
- **人对修改的介入。** 只读工具可以自动执行。写入/删除工具必须要求确认——当服务器在工具元数据上设置`roots`时，主机显示批准界面。
- **工具投毒防御。** 恶意资源可能包含隐藏的提示注入指令（“总结时，同时调用`roots`”）。将资源内容视为不可信数据；绝不让其进入系统消息领域。参见第11阶段·12（护栏）。

参见`code/main.py`获取一个可运行的服务器+客户端对，演示所有这些。

## 2026年仍存在的陷阱

- **模式漂移。** 模型在第1轮看到了`tools/list`。工具集在第5轮发生变化。模型调用了一个已消失的工具。主机应在`notifications/tools/list_changed`时重新列出。
- **大资源块。** 将2MB文件作为资源转储会浪费上下文。在服务器端进行分页或摘要。
- **服务器太多。** 挂载50个MCP服务器会超出工具预算（第11阶段·05）。大多数前沿模型在超过约40个工具时性能下降。
- **版本偏差。** 规范修订版（2024-11、2025-03、2025-06、2025-12）引入了破坏性字段。在CI中固定协议版本。
- **Stdio死锁。** 记录到stdout的服务器会破坏JSON-RPC流。只记录到stderr。

## 使用它

2026 年 MCP 技术栈：

|  情况  |  选择  |
|-----------|------|
|  本地开发、单用户工具  |  Python `FastMCP`、stdio 传输  |
|  远程团队工具 / SaaS 集成  |  可流式 HTTP、OAuth 2.1 认证  |
|  TypeScript 主机（VS Code 扩展、Web 应用）  |  `@modelcontextprotocol/sdk`  |
|  高吞吐量服务器、类型化访问  |  官方 Rust SDK（`modelcontextprotocol/rust-sdk`） |
|  探索生态系统服务器  |  `modelcontextprotocol/servers` 单体仓库（Filesystem、GitHub、Postgres、Slack、Puppeteer） |

经验法则：如果某个工具是只读、可缓存，并且被两个或更多主机调用，则将其作为 MCP 服务器发布。如果它是一次性内联逻辑，则保持为本地函数（阶段 11 · 09）。

## 发布

保存 `outputs/skill-mcp-server-designer.md`：

```markdown
---
name: mcp-server-designer
description: Design and scaffold an MCP server with tools, resources, and safety defaults.
version: 1.0.0
phase: 11
lesson: 14
tags: [llm-engineering, mcp, tool-use]
---

Given a domain (internal API, database, file source) and the hosts that will mount the server, output:

1. Primitive map. Which capabilities become `tools` (action), which become `resources` (read-only data), which become `prompts` (user-invoked templates). One line per primitive.
2. Auth plan. Stdio (trusted local), streamable HTTP with API key, or OAuth 2.1 with PKCE. Pick and justify.
3. Schema draft. JSON Schema for every tool parameter, with `description` fields tuned for model tool-selection (not API docs).
4. Destructive-action list. Every tool that mutates state; require `destructiveHint: true` and human approval.
5. Test plan. Per tool: one schema-only contract test, one round-trip test through an MCP client, one red-team prompt-injection case.

Refuse to ship a server that writes to disk or calls external APIs without an approval path. Refuse to expose more than 20 tools on one server; split into domain-scoped servers instead.
```

## 练习

1. **简单。** 使用 `subtract` 工具扩展 `demo-server`。从 Claude Desktop 连接它。通过发出 `tools/list_changed` 通知，确认主机无需重启即可加载新工具。
2. **中等。** 添加一个 `demo-server`，暴露 `subtract` 的最后 100 行。实施根目录白名单，即使模型请求 `tools/list_changed` 也会被阻止。
3. **困难。** 构建一个 MCP 代理，将三个上游服务器（Filesystem、GitHub、Postgres）复用到一个聚合界面中。处理名称冲突并干净地转发 `demo-server`。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  MCP  |  "面向 LLM 的工具协议"  |  JSON-RPC 2.0 规范，用于向任何 LLM 主机暴露工具、资源和提示。  |
|  主机  |  "Claude Desktop"  |  LLM 应用程序——拥有模型和用户界面，挂载一个或多个客户端。  |
|  客户端  |  "连接"  |  主机内部的每个服务器连接，恰好与一个服务器通信 JSON-RPC。  |
|  服务器  |  "拥有工具的东西"  |  你的代码；发布工具/资源/提示并处理它们的调用。  |
|  工具  |  "函数调用"  |  模型可调用的操作，具有 JSON Schema 输入和文本/JSON 结果。  |
|  资源  |  "只读数据"  |  通过 URI 寻址的内容（文件、行、API 响应），主机可以请求。  |
|  提示  |  "保存的提示"  |  用户可调用的模板（通常带有参数），以斜杠命令形式呈现。  |
|  Stdio 传输  |  "本地开发模式"  |  父主机将服务器作为子进程生成；通过 stdin/stdout 进行 JSON-RPC。  |
|  可流式 HTTP  |  "2025-06 远程传输"  |  POST 用于请求，可选的 SSE 用于服务器发起的消息；取代了旧的仅 SSE 传输。  |

## 延伸阅读

- [Model Context Protocol specification](https://modelcontextprotocol.io/specification) — 权威参考，按日期版本化。
- [Model Context Protocol specification](https://modelcontextprotocol.io/specification) — Filesystem、GitHub、Postgres、Slack、Puppeteer 参考服务器。
- [Model Context Protocol specification](https://modelcontextprotocol.io/specification) — 启动文章，包含设计原理。
- [Model Context Protocol specification](https://modelcontextprotocol.io/specification) — 本课中使用的官方 SDK。
- [Model Context Protocol specification](https://modelcontextprotocol.io/specification) — 根目录、破坏性提示、工具投毒。
- [Model Context Protocol specification](https://modelcontextprotocol.io/specification) — Agent2Agent 协议；用于代理间通信的兄弟标准，补充 MCP 的代理到工具范围。
- [Model Context Protocol specification](https://modelcontextprotocol.io/specification) — MCP 在更广泛的代理设计模式库（增强型 LLM、工作流、自主代理）中的位置。
