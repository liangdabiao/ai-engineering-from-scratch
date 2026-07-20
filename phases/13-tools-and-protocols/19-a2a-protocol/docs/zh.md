# A2A — 代理到代理协议 (Agent-to-Agent Protocol)

> MCP 是代理到工具 (agent-to-tool)。A2A (Agent2Agent) 是代理到代理 (agent-to-agent) —— 一种开放协议，允许基于不同框架构建的不透明代理进行协作。由 Google 于 2025 年 4 月发布，2025 年 6 月捐赠给 Linux 基金会，2026 年 4 月达到 v1.0，拥有 150+ 支持者，包括 AWS、Cisco、Microsoft、Salesforce、SAP 和 ServiceNow。它吸收了 IBM 的 ACP 并增加了 AP2 支付扩展。本课程将介绍 Agent Card、Task 生命周期以及两种传输绑定。

**类型：** 构建
**语言：** Python (stdlib, Agent Card + Task harness)
**先决条件：** 阶段 13 · 06 (MCP 基础), 阶段 13 · 08 (MCP 客户端)
**时间：** ~75 分钟

## 学习目标

- 区分代理到工具 (MCP) 和代理到代理 (A2A) 用例。
- 在 `/.well-known/agent.json` 发布包含技能和端点元数据的 Agent Card。
- 走查 Task 生命周期 (submitted → working → input-required → completed / failed / canceled / rejected)。
- 使用带有 Parts (text, file, data) 的 Messages 和 Artifacts 作为输出。

## 问题

一个客户服务代理需要将报告撰写委托给一个专门的写作代理。A2A 之前的选项：

- 自定义 REST API。可行，但每对组合都是定制的一次性方案。
- 共享代码库。要求两个代理运行在同一个框架上。
- MCP。不合适：MCP 用于调用工具，而不是让两个代理在保持各自不透明内部推理的同时进行协作。

A2A 填补了这一空白。它将交互建模为一个代理向另一个代理发送 Task，包含生命周期、消息和工件 (Artifacts)。被调用代理的内部状态保持不透明 —— 调用者只能看到任务状态转换和最终输出。

A2A 是“让跨框架的代理相互通信”的协议。它不取代 MCP；两者是互补的。

## 核心概念

### Agent Card

每个符合 A2A 的代理在 `/.well-known/agent.json` 发布一个卡片：

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

发现机制基于 URL：获取卡片，了解 A2A 端点的 URL，列举技能。

### 签名 Agent Cards (AP2)

AP2 扩展（2025 年 9 月）为 Agent Cards 添加了加密签名。发布者使用 JWT 签署自己的卡片；消费者进行验证。防止冒充。

### Task 生命周期

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (loop via message)
```

客户端通过 `tasks/send` 发起。被调用代理在不同状态间转换；客户端通过 SSE 或轮询订阅状态更新。

### Messages 和 Parts

一条消息包含一个或多个 Parts：

- `text` — 纯文本内容。
- `text` — 带 mimeType 的 base64 blob。
- `text` — 类型化 JSON 负载（被调用代理的结构化输入）。

示例：

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### Artifacts

输出是 Artifacts，而不是原始字符串。Artifact 是命名、类型化的输出：

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Artifacts 可以分块流式传输。调用者进行累积。

### 两种传输绑定

1. **基于 HTTP 的 JSON-RPC。** `/a2a` 端点，请求使用 POST，流式传输可选 SSE。默认绑定。
2. **gRPC。** 适用于原生支持 gRPC 的企业环境。

两种绑定携带相同的逻辑消息形状。

### 不透明性保持

一个关键设计原则：被调用代理的内部状态是不透明的。调用者看到任务状态和工件。被调用代理的思维链、工具调用、子代理委派——都不可见。这与 MCP 不同，在 MCP 中工具调用是透明的。

理由：A2A 使竞争对手能够在不必透露内部细节的情况下进行协作。A2A 可以是“调用这个客户服务代理”，而调用者无需了解该代理如何实现服务。

### 时间线

- **2025-04-09.** Google 宣布 A2A。
- **2025-06-23.** 捐赠给 Linux 基金会。
- **2025-08.** 吸收 IBM 的 ACP。
- **2025-09.** AP2 扩展（代理支付）发布。
- **2026-04.** v1.0 发布，150+ 支持组织。

### 与 MCP 的关系

| 维度（Dimension）  |  MCP  |  A2A |
|-----------|-----|-----|
| 用例（Use case）  |  Agent-to-tool  |  Agent-to-agent |
| 不透明度（Opacity）  |  透明的工具调用（Transparent tool calls）  |  不透明的内部推理（Opaque inner reasoning） |
| 调用方（Typical caller）  |  代理运行时（Agent runtime）  |  另一个代理（Another agent） |
| 状态（State）  |  工具调用结果（Tool-call result）  |  带生命周期的任务（Task with lifecycle） |
| 授权（Authorization）  |  OAuth 2.1 (Phase 13 · 16)  |  JWT签名的代理卡（AP2） |
| 传输（Transport）  |  Stdio / Streamable HTTP  |  JSON-RPC over HTTP / gRPC |

当你要调用特定工具时，请使用MCP。当你要将整个任务委托给另一个代理时，请使用A2A。许多生产系统两者兼用：一个代理使用MCP作为其工具层，使用A2A作为其协作层。

## 使用它

`code/main.py`实现了一个最小的A2A封装：一个研究代理发布其卡片，一个写作代理接收一个`tasks/send`，其部分包括一个PDF和一个文本指令，经过working → input_required → working → completed的状态转换，并返回一个文本产物。全部使用标准库；使用内存传输以聚焦消息形状。

需要关注的内容：

- 代理卡JSON形状。
- 任务ID分配与状态转换。
- 包含混合类型部分的消息。
- 任务中的输入需求分支。
- 完成时产物的返回。

## 发布

本课生成`outputs/skill-a2a-agent-spec.md`。给定一个应能被其他代理调用的新代理，该技能将生成代理卡JSON、技能模式和端点蓝图。

## 练习

1. 运行`code/main.py`。跟踪完整的任务生命周期，包括被调用代理请求澄清的输入需求暂停。

2. 添加签名的代理卡。使用HMAC对卡片的规范JSON进行签名。编写验证器并确认它在变异的卡片上失败。

3. 实现任务流式传输：写作代理通过SSE发出三个增量产物块，调用方将其累积。

4. 设计一个包装MCP服务器的A2A代理。将每个MCP工具映射到一个A2A技能。注意权衡——失去了哪些不透明度？

5. 阅读A2A v1.0公告，并确定截至2026年4月任何框架都尚未实现的一个功能。（提示：与多跳任务委托相关。）

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
| A2A  |  "Agent-to-Agent protocol"（代理到代理协议）  |  用于不透明代理协作的开放协议 |
| Agent Card  |  "`.well-known/agent.json`"  |  描述代理技能和端点的已发布元数据 |
| Skill  |  "A callable unit"（可调用单元）  |  代理支持的命名操作（类似于MCP工具） |
| Task  |  "Unit of delegation"（委托单元）  |  具有生命周期和最终产物的工作项 |
| Message  |  "Task input"（任务输入）  |  携带部分（文本、文件、数据） |
| Part  |  "Typed chunk"（类型化块）  |  消息的`text` / `file` / `data`元素 |
| Artifact  |  "Task output"（任务输出）  |  完成时返回的命名、类型化输出 |
| AP2  |  "Agent Payments Protocol"（代理支付协议）  |  用于信任和支付的签名代理卡扩展 |
| Opacity  |  "Black-box collaboration"（黑盒协作）  |  被调用代理的内部细节对调用方隐藏 |
| Input-required  |  "Task pause"（任务暂停）  |  代理需要更多信息时的生命周期状态 |

## 延伸阅读

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — 规范的A2A规范
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — 参考实现和SDK
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — 2025年6月治理转让
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — 路线图和合作伙伴势头
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — v1.0发布说明和向后兼容指南
