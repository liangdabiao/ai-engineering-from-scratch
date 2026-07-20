# Claude Agent SDK：子代理与会话存储

> Claude Agent SDK 是 Claude Code 框架的库形式。内置工具、用于上下文隔离的子代理、钩子、W3C 追踪传播、会话存储对等。Claude Managed Agents 是面向长时间运行异步工作的托管替代方案。

**类型：** 学习+构建
**语言：** Python (标准库)
**前置条件：** 阶段14 · 01 (代理循环), 阶段14 · 10 (技能库)
**时长：** 约75分钟

## 学习目标

- 解释 Anthropic Client SDK（原始 API）与 Claude Agent SDK（框架形态）之间的区别。
- 描述子代理——并行化和上下文隔离——以及何时使用它们。
- 列举 Python SDK 的会话存储接口（`append`, `load`, `list_sessions`, `delete`, `list_subkeys`）以及 `--session-mirror` 的作用。
- 使用标准库实现一个包含内置工具、上下文隔离的子代理生成、生命周期钩子和会话存储的框架。

## 问题

原始的 LLM API 仅提供一次往返。生产级代理需要工具执行、MCP 服务器、生命周期钩子、子代理生成、会话持久化、追踪传播。Claude Agent SDK 以库的形式提供这一形态——即 Claude Code 使用的同一框架，供自定义代理使用。

## 核心概念

### 客户端 SDK 与代理 SDK

- **客户端 SDK（`anthropic`）。** 原始消息 API。您控制循环、工具和状态。
- **代理 SDK（`anthropic`）。** 内置工具执行、MCP 连接、钩子、子代理生成、会话存储。Claude Code 循环以库的形式提供。

### 内置工具

SDK 自带 10 多种工具：文件读写、shell、grep、glob、网络获取等。自定义工具通过标准工具模式接口注册。

### 子代理

Anthropic 文档中记载了两个用途：

1. **并行化。** 并发执行独立工作。"为这 20 个模块分别找到对应的测试文件" 就是一个包含 20 个并行子代理的任务。
2. **上下文隔离。** 子代理使用自己的上下文窗口；仅结果返回给编排器。编排器的预算得以保留。

Python SDK 近期新增：`list_subagents()`、`get_subagent_messages()` 用于读取子代理对话记录。

### 会话存储

协议与 TypeScript 对等：

- `append(session_id, message)` — 添加一轮对话。
- `append(session_id, message)` — 恢复对话。
- `append(session_id, message)` — 枚举。
- `append(session_id, message)` — 级联到子代理会话。
- `append(session_id, message)` — 列出子代理键。

`--session-mirror`（CLI 标志）在流式传输时将对话记录镜像到外部文件，用于调试。

### 钩子

您可以注册的生命周期钩子：

- `PreToolUse`、`PostToolUse` — 门控或审计工具调用。
- `PreToolUse`、`PostToolUse` — 设置和拆除。
- `PreToolUse` — 在模型看到用户输入之前对其进行操作。
- `PreToolUse` — 在上下文压缩之前运行。
- `PreToolUse` — 代理退出时清理。
- `PreToolUse` — 侧通道警报。

钩子是专业工作流（阶段14课程参考）等系统添加横切行为的方式。

### W3C 追踪上下文

调用者上活动的 OpenTelemetry 跨度通过 W3C 追踪上下文标头传播到 CLI 子进程。整个多进程追踪在您的后端显示为一个追踪。

### Claude Managed Agents

托管替代方案（beta 标头 `managed-agents-2026-04-01`）。长时间运行的异步工作、内置提示缓存、内置压缩。用控制权换取托管基础设施。

### 这种模式出错的地方

- **子代理过度生成。** 为100个小任务生成100个子代理。开销占主导。改为批处理。
- **钩子蔓延。** 每个团队添加钩子；启动时间膨胀。每季度审查钩子。
- **会话膨胀。** 会话累积，体积增长。使用 `list_sessions` 加过期策略。

## 动手构建

`code/main.py` 在标准库中实现了SDK形状：

- `Tool`、`ToolRegistry` 内置 `read_file`、`write_file`、`list_dir`。
- `Tool` — 私有上下文、隔离运行、返回结果。
- `Tool` — 追加、加载、列表、删除、list_subkeys。
- `Tool` — `ToolRegistry`、`read_file`、`write_file`、`list_dir`。
- 演示：主代理并行生成 3 个子代理（各自隔离），聚合结果，持久化会话。

运行它：

```
python3 code/main.py
```

追踪显示子代理上下文隔离（编排器上下文大小保持有界）、钩子执行和会话持久化。

## 使用它

- **Claude Agent SDK** 适用于希望使用 Claude Code 框架形态的以 Claude 为先的产品。
- **Claude Managed Agents** 适用于托管的长运行异步工作。
- **OpenAI Agents SDK**（第16课）适用于以 OpenAI 为先的对应产品。
- **LangGraph + 自定义工具** 如果您希望使用图状状态机。

## 发布

`outputs/skill-claude-agent-scaffold.md` 搭建一个包含子代理、钩子、会话存储、MCP 服务器连接和 W3C 追踪传播的 Claude Agent SDK 应用。

## 练习

1. 添加一个子代理生成器，将20个任务分批为5个并行子代理进行批处理。测量编排器上下文大小与每个任务一个子代理的对比。
2. 实现一个`PreToolUse`钩子，用于速率限制`write_file`调用（每个会话每分钟5次）。追踪该行为。
3. 将`PreToolUse`接入以渲染子代理树。深层嵌套看起来如何？
4. 将玩具示例移植到真实的`PreToolUse` Python包中。工具注册会发生什么变化？
5. 阅读Claude Managed Agents文档。何时应该从自托管切换到托管？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  Agent SDK  |  "Claude Code作为库"  |  框架形态：工具、MCP、钩子、子代理、会话存储  |
|  子代理  |  "子代理"  |  独立上下文，自有预算；结果向上冒泡  |
|  会话存储  |  "对话数据库"  |  持久化、加载、列出、删除包含子代理级联的轮次  |
|  钩子  |  "生命周期回调"  |  工具前后、会话、提示提交、压缩、停止  |
|  W3C追踪上下文  |  "跨进程追踪"  |  父跨度传播到CLI子进程  |
|  托管代理  |  "托管框架"  |  Anthropic托管的长时间运行异步工作  |
|  `--session-mirror`  |  "对话记录镜像"  |  将会话轮次流式写入外部文件  |
|  MCP服务器  |  "工具表面"  |  附加到代理的外部工具/资源源  |

## 延伸阅读

- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Code的库形式
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — 生产模式
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — 托管替代方案
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — 对应物
