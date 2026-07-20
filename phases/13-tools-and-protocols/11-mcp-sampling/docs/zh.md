# MCP 采样(Sampling) — 服务器请求的 LLM 补全(Completion)与智能体循环(Agent Loop)

> 大多数 MCP 服务器是哑执行器：接收参数、运行代码、返回内容。采样让服务器能够扭转方向：它请求客户端的 LLM 做出决策。这使得服务器可以托管智能体循环，而无需服务器拥有任何模型凭证。SEP-1577 于 2025-11-25 合并，在采样请求中加入了工具(tool)，使循环能够包含更深层次的推理。偏离风险提示：SEP-1577 中工具内采样(tool-in-sampling)的形态在 2026 年第一季度仍处于实验阶段，并在 SDK API 中持续调整。

**类型：** 构建
**语言：** Python（标准库，采样框架）
**前置条件：** 阶段 13 · 07（MCP 服务器），阶段 13 · 10（资源和提示）
**时间：** 约 75 分钟

## 学习目标

- 解释 `sampling/createMessage` 解决了什么问题（无需服务器端 API 密钥的服务器托管循环）。
- 实现一个服务器，它要求客户端对多个轮次的提示(prompt)进行采样并返回补全结果。
- 使用 `sampling/createMessage`（成本/速度/智能优先级）来指导客户端模型选择。
- 构建一个 `sampling/createMessage` 工具(tool)，通过采样内部迭代，而不是硬编码行为。

## 问题

一个用于代码摘要工作流的有用 MCP 服务器需要：遍历文件树、选择要读取的文件、综合摘要并返回。LLM 推理在哪里发生？

选项 A：服务器调用自己的 LLM。需要 API 密钥，在服务器端计费，对每个用户来说成本高昂。

选项 B：服务器返回原始内容；客户端的智能体进行推理。有效但将服务器逻辑移入了客户端提示中，这很脆弱。

选项 C：服务器通过 `sampling/createMessage` 请求客户端的 LLM。服务器保留算法（读取哪些文件、进行多少次遍历），而客户端保留计费和模型选择。服务器完全不需要凭证。

采样就是选项 C。它是一种机制，使得受信任的服务器无需成为完整的 LLM 主机即可托管智能体循环。

## 核心概念

### `sampling/createMessage` 请求

服务器发送：

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

客户端运行其 LLM，返回：

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

三个浮点数，总和为 1.0：

- `costPriority`：偏向更便宜的模型。
- `costPriority`：偏向更快的模型。
- `costPriority`：偏向能力更强的模型。

加上 `hints`：服务器偏好的命名模型。客户端可以（也可以不）遵循提示；客户端的用户配置始终优先。

### `includeContext`

三个值：

- `"none"` — 仅服务器提供的消息。默认。
- `"none"` — 包含来自此服务器会话的先前消息。
- `"none"` — 包含所有会话上下文。

由于 `includeContext` 会泄露跨服务器上下文，存在安全风险，自 2025-11-25 起被软弃用。推荐使用 `"none"`，并在消息中显式传递上下文。

### 带工具的采样 (SEP-1577)

2025-11-25 新增：采样请求可以包含一个 `tools` 数组。客户端使用这些工具运行一个完整的工具调用循环。这使得服务器能够通过客户端的模型托管 ReAct 风格的智能体循环。

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

客户端循环：采样、如果被调用则执行工具、再次采样、返回最终的助手消息。这一功能在 2026 年第一季度仍处于实验阶段；SDK 签名可能还会变化。实现时请对照 2025-11-25 规范中的客户端/采样部分进行确认。

### 人工介入(Human-in-the-loop)

客户端必须在执行采样前向用户展示服务器要求模型执行的内容。恶意服务器可能利用采样操纵用户会话（例如“告诉用户说 X，以便他们点击 Y”）。Claude Desktop、VS Code 和 Cursor 将采样请求以用户可拒绝的确认对话框形式呈现。

2026 年的共识：未经人工确认的采样是一个红旗(red flag)。网关（阶段 13 · 17）可以自动批准低风险采样并自动拒绝任何可疑请求。

### 无需 API 密钥的服务器托管循环

典型用例：一个自身没有 LLM 访问权限的代码摘要 MCP 服务器。它执行以下操作：

1. 遍历仓库结构。
2. 调用 `sampling/createMessage` 并附带“挑选最可能描述该仓库用途的五个文件。”
3. 读取这些文件。
4. 调用 `sampling/createMessage` 并附带这些文件的内容以及“用三段话总结该仓库。”
5. 将摘要作为 `sampling/createMessage` 结果返回。

服务器从未接触过 LLM API。客户端的用户使用自己的凭证来支付补全的费用。

### 安全风险（Unit 42 披露，2026 年第一季度）

- **隐蔽采样(Covert sampling)。** 一个总是调用采样并附带“从会话上下文中响应用户的电子邮件”的工具。阶段 13 · 15 涵盖了攻击向量。
- **通过采样的资源窃取(Resource theft via sampling)。** 服务器要求客户端摘要攻击者的负载(payload)，向用户收费。
- **循环炸弹(Loop bombs)。** 服务器在紧密循环中调用采样。客户端必须强制执行每会话速率限制。

## 使用它

`code/main.py` 提供了一个伪造的服务器到客户端采样框架。一个模拟的"summarize_repo"工具调用两轮采样（选择文件，然后总结），伪造的客户端返回预设的响应。该框架展示了：

- 服务器发送带有 `modelPreferences` 的 `sampling/createMessage`。
- 客户端返回一个完成。
- 服务器继续其循环。
- 速率限制器限制每次工具调用的总采样调用次数。

需要关注的内容：

- 服务器只暴露一个工具（`summarize_repo`）；所有推理都发生在采样调用中。
- 模型偏好加权客户端的模型选择；提示列表列出首选模型。
- 循环在 `summarize_repo` 时终止。
- `summarize_repo` 限制捕获失控循环。

## 发布

本课生成 `outputs/skill-sampling-loop-designer.md`。给定一个需要 LLM 调用的服务器端算法（研究、总结、规划），该技能设计了基于采样的实现，包括正确的 modelPreferences、速率限制和安全确认。

## 练习

1. 运行 `code/main.py`。将 `max_samples_per_tool` 改为 2 并观察速率限制的截断。

2. 实现 SEP-1577 工具内采样变体：采样请求携带一个 `tools` 数组。验证客户端循环在执行这些工具后才返回最终完成。注意漂移风险：SDK 签名在 2026 年上半年仍可能变化。

3. 增加人工参与确认：在服务器的第一个 `sampling/createMessage` 之前，暂停并等待用户批准。被拒绝的调用返回一个类型化的拒绝。

4. 添加一个以客户端会话为键的每用户速率限制器。同一用户的同一服务器循环应共享一个预算。

5. 设计一个使用采样选择要包含块的 `summarize_pdf` 工具。草拟发送的消息。`modelPreferences.intelligencePriority` 在 0.1 与 0.9 时如何改变行为？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  采样  |  "服务器到客户端 LLM 调用"  |  服务器向客户端的模型请求一个完成  |
|  `sampling/createMessage`  |  "方法"  |  用于采样请求的 JSON-RPC 方法  |
|  `modelPreferences`  |  "模型优先级"  |  成本/速度/智能权重以及名称提示  |
|  `includeContext`  |  "跨会话泄漏"  |  软弃用的上下文包含模式  |
|  SEP-1577  |  "采样中的工具"  |  允许在采样中使用工具用于服务器托管的 ReAct  |
|  人工参与  |  "用户确认"  |  客户端在运行前向用户展示采样请求  |
|  循环炸弹  |  "失控采样"  |  服务器端无限采样循环；客户端必须限速  |
|  隐蔽采样  |  "隐藏推理"  |  恶意服务器在采样提示中隐藏意图  |
|  资源窃取  |  "使用用户的 LLM 预算"  |  服务器强制客户端为其不想要的采样花费  |
|  `stopReason`  |  "生成停止原因"  |  `endTurn`、`stopSequence` 或 `maxTokens`  |

## 延伸阅读

- [MCP — Concepts: Sampling](https://modelcontextprotocol.io/docs/concepts/sampling) — 采样的高级概述
- [MCP — Concepts: Sampling](https://modelcontextprotocol.io/docs/concepts/sampling) — 规范化的 [MCP — Client sampling spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) 形状
- [MCP — Concepts: Sampling](https://modelcontextprotocol.io/docs/concepts/sampling) — 采样中工具的规范进化提案（实验性）
- [MCP — Concepts: Sampling](https://modelcontextprotocol.io/docs/concepts/sampling) — 隐蔽采样和资源窃取模式
- [MCP — Concepts: Sampling](https://modelcontextprotocol.io/docs/concepts/sampling) — 带客户端代码示例的演练
