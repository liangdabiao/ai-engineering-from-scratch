# Agent框架权衡 — LangGraph vs CrewAI vs AutoGen vs Agno

> 每个框架都推广相同的演示（研究代理生成报告），却隐藏了相同的缺陷（状态模式与编排层冲突）。选择其抽象层与问题形态匹配的框架；其余部分都是你需要重复编写的胶水代码。

**类型:** 学习
**语言:** Python
**前置知识:** 阶段11 · 09 (函数调用), 阶段11 · 16 (LangGraph)
**时间:** ~45分钟

## 问题

你有一个需要不止一次LLM调用的任务。可能是一个研究工作流（规划、搜索、总结、引用）。可能是一个代码审查流水线（解析差异、评审、补丁、验证）。可能是一个多轮助手，负责预订航班、撰写邮件和提交费用报告。你选择一个框架。

三天后，你发现框架的抽象层存在漏洞。CrewAI提供了角色，但当“研究员”需要将结构化计划交给“撰稿人”时却与你对抗。AutoGen在代理之间提供聊天，但没有一流的状态支持，因此你的检查点是一堆对话日志。LangGraph提供了状态图，但迫使你在知道代理将做什么之前命名每个转换。Agno提供了一个单代理抽象层，当你尝试扩展到三个并发工作代理时会出现问题。

解决方法不是“选择最好的框架”。而是将框架的核心抽象层与问题的形态匹配。本课绘制了这张图。

## 核心概念

![Agent framework matrix: core abstraction vs problem shape](../assets/framework-matrix.svg)

四个框架主导了2026年的格局。它们的核心抽象层并不相同。

|  框架  |  核心抽象  |  最佳适用  |  最差适用  |
|-----------|------------------|----------|-----------|
|  **LangGraph**  |  `StateGraph` — 类型化状态、节点、条件边、检查点器。  |  需要显式状态和人工参与的显式工作流；需要时间旅行调试的生产代理。  |  松散的、角色驱动的头脑风暴，拓扑未知。  |
|  **CrewAI**  |  `Crew` — 角色（目标、背景）、任务、流程（顺序或分层）。  |  角色扮演或个性驱动的工作流，具有较短的线性/分层计划。  |  任何有状态且超出Crew对话历史的东西；复杂的分支。  |
|  **AutoGen**  |  `ConversableAgent` 配对 — 两个或更多代理轮流发言直到退出条件。  |  多代理*对话*（师生、提议者-批评者、行动者-评审者），思维从聊天中涌现。  |  具有已知DAG的确定性工作流；任何需要在重启后持久化状态的东西。  |
|  **Agno**  |  `Agent` — 单个LLM + 工具 + 记忆，可组合成团队。  |  快速构建的单代理和轻量级团队；强大的多模态和内置存储驱动。  |  深度、显式分支的图，带有自定义reducer。  |

### “抽象层”的实际含义

框架的核心抽象层是你在架构设计时在白板上画的东西。

- **LangGraph** → 你画一个图。节点是步骤，边是转换，每个点的状态对象是类型化的。心智模型是状态机。
- **CrewAI** → 你画一个组织结构图。每个角色都有职位描述，管理者分配任务。心智模型是一个小型专家团队。
- **AutoGen** → 你画一个Slack私信。两个代理互相发消息；如果需要主持人，第三个加入。心智模型是聊天。
- **Agno** → 你画一个带有工具附件的单个方框。将方框并列放置形成团队。心智模型是“自带电池的智能体”。

### 状态问题

状态是大多数框架选择在生产中失败的地方。

- **LangGraph.** 类型化状态(`TypedDict` 或 Pydantic 模型)、逐字段的 reducer、一流的检查点器 (SQLite/Postgres/Redis)。恢复、中断和时间旅行都是免费的。*(参见阶段 11 · 16.)*
- **CrewAI.** 状态通过 `TypedDict` 字段在任务之间以字符串形式流动，或通过 `context` 结构化。开箱即用没有持久的每Crew存储；如果Crew必须承受重启，你需要自行添加。
- **AutoGen.** 状态是聊天历史和你定义的任何 `TypedDict`。对话记录持久化；任意工作流状态除非你编写适配器，否则不会持久化。
- **Agno.** 内置存储驱动（SQLite、Postgres、Mongo、Redis、DynamoDB）通过 `context` 附加到 `TypedDict` 上——对话会话和用户记忆自动持久化。不是完整的图检查点器，而是一个会话存储。

### 分支问题

每个非平凡代理都会分支。谁决定分支很重要。

- **LangGraph** — 由你决定，通过条件边。路由是一个Python函数，带有命名分支。分支在编译后的图中是一流的；检查点器记录了哪个分支被采用。
- **CrewAI** — 在分层模式下由管理者决定；在顺序模式下你在构建时决定。路由在任务列表中隐式体现；除了管理者的提示之外，没有一流的“if”。
- **AutoGen** — 代理通过聊天决定。分支从谁下一个发言中涌现。`GroupChatManager` 选择下一个发言者；你可以手动编写 `speaker_selection_method`，但默认是LLM驱动的。
- **Agno** — 代理通过下一步调用哪个工具来决定。团队有协调者/路由器/协作模式；超出该范围的分支是开发者的责任。

### 可观测性问题

- **LangGraph** — 通过LangSmith或任何OTel导出器的OpenTelemetry。每个节点转换都是一个追踪跨度；检查点同时作为可重放的追踪。LangSmith是第一方选项；Langfuse/Phoenix也有适配器。
- **CrewAI** — 自2025年末起支持一流的OpenTelemetry；与Langfuse、Phoenix、Opik、AgentOps集成。
- **AutoGen** — 通过 `autogen-core` 集成OpenTelemetry；AgentOps和Opik有连接器。追踪粒度是每代理消息，而不是每节点。
- **Agno** — 内置 `autogen-core` 标志加上OpenTelemetry导出器；与Langfuse紧密集成用于会话追踪。

### 成本和延迟

所有四个框架都会增加每次调用的开销（框架逻辑、验证、序列化）。粗略的开销递增顺序：Agno ≈ LangGraph < CrewAI ≈ AutoGen。差异主要由框架进行额外LLM路由的多少决定。CrewAI的分层管理者花费token决定谁下一个发言；AutoGen的 `GroupChatManager` 同样如此。LangGraph只在编写 `llm.invoke` 的地方花费token。Agno的单代理路径较薄。

当每次运行的成本重要时，优先选择显式路由（LangGraph边、AutoGen `speaker_selection_method`）而非LLM选择的路由。

### 互操作性

- **LangGraph** ↔ **LangChain** 工具、检索器、LLM。一流MCP适配器（作为MCP服务器导入的工具）。
- **CrewAI** ↔ 工具继承自 `BaseTool`；LangChain工具、LlamaIndex工具和MCP工具均可适配。通过 `allow_delegation=True` 实现Crew到Crew的委托。
- **AutoGen** → `BaseTool` 包装任何Python可调用对象；提供MCP适配器。与AG2生态系统紧密耦合，用于代理间模式。
- **Agno** → `BaseTool` 装饰器或BaseTool子类；MCP适配器；工具可以在代理和团队之间共享。

## 技能

> 你可以用一句话解释为什么给定框架适合给定的代理问题。

预构建清单：

1. **绘制图形。** 这是一个图（有类型状态、命名转换）吗？一个角色扮演（专家交接工作）？一个聊天（代理对话直到完成）？一个带有工具的独立代理？
2. **决定谁进行分支。** 开发者决定分支 → LangGraph。管理者代理决定 → CrewAI 层级。聊天涌现 → AutoGen。工具调用决定 → Agno。
3. **检查状态预算。** 是否需要从检查点恢复？时间旅行？运行中途的人工干预？如果需要，LangGraph 是默认选择；Agno 会话覆盖对话范围的状态。
4. **检查成本预算。** LLM 选择的路由每次交互消耗额外令牌。如果代理每天运行数千次，首选显式路由。
5. **评估框架开销。** 每个框架都是一个额外的依赖项。如果任务只有两个 LLM 调用和一个工具，写 30 行纯 Python 代码；没有框架比没有框架更便宜。

在你能够画出图形、组织结构图、聊天或代理框之前，拒绝选择框架。拒绝选择一个迫使你为其实际需求而对抗其状态模型的框架。

## 决策矩阵

|  问题形状  |  首选框架  |  原因  |
|---------------|---------------------|-----|
|  带类型状态、人工审批、长时间运行的工作流 DAG  |  LangGraph  |  一流的状态、检查点、中断、时间旅行。  |
|  具有不同角色的研究/写作流程  |  CrewAI（顺序）或 LangGraph 子图  |  角色-任务在 CrewAI 中表达简单；当分支复杂时，扩展到 LangGraph。  |
|  提议者-批评者或教师-学生对话  |  AutoGen  |  双代理聊天是其原生形态。  |
|  带工具、会话、记忆的独立代理  |  Agno  |  最薄的设置，内置存储和记忆。  |
|  带归约器的数千个并行扇出  |  LangGraph + `Send`  |  唯一具有一流并行分发 API 的框架。  |
|  快速原型，无框架承诺  |  纯 Python + 提供商 SDK  |  没有框架是最快的框架。  |

## 练习

1. **简单。** 完成同一任务——"研究 Anthropic 总部，写一份 200 字的简报，注明来源"——在 LangGraph（四个节点：规划、搜索、写作、引用）和 CrewAI（三个角色：研究员、写手、编辑）中实现。报告每次运行的令牌成本和代码行数。
2. **中等。** 在 AutoGen（研究员 ↔ 写手聊天，编辑通过 `GroupChat` 加入）和 Agno（一个带有 `search_tools` 和 `write_tools` 以及会话存储的单一代理）中构建相同任务。按 (a) 每次运行成本，(b) 崩溃后恢复能力，(c) 在写入步骤前注入人工审批能力对四种实现进行排名。
3. **困难。** 构建一个决策树脚本 `GroupChat`，它接收一个简短的问题描述（JSON: `search_tools`）并返回建议及一句话理由。在你自己设计的六个案例上验证它。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  编排  |  "代理如何协调"  |  决定下一个运行的节点/角色/代理的层。  |
|  持久状态  |  "重启后恢复"  |  在进程崩溃后仍然存在、附属于检查点或会话存储的状态。  |
|  LLM 选择的路由  |  "让模型决定"  |  规划 LLM 每轮选择下一步；灵活但每次决策消耗令牌。  |
|  显式路由  |  "开发者决定"  |  Python 函数或静态边选择下一步；廉价且可审计。  |
|  Crew  |  "CrewAI 团队"  |  角色 + 任务 + 流程（顺序或层级）绑定为一个可运行单元。  |
|  GroupChat  |  "AutoGen 的多代理聊天"  |  有发言人选择器的 N 个代理之间的受管理对话。  |
|  Team (Agno)  |  "Agno 的多代理"  |  在一组代理上的路由/协调/协作模式。  |
|  StateGraph  |  "LangGraph 的图"  |  类型化状态、节点、条件边、检查点抽象。  |

## 延伸阅读

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — StateGraph, checkpointers, interrupts, time-travel.
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — Crews, Flows, Agents, Tasks, Processes.
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — ConversableAgent, GroupChat, teams, tools.
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — Agent, Team, Workflow, storage, memory.
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — pattern library (prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer) framework-agnostic.
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — the loop every framework dresses up.
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — AutoGen's design paper.
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — role-play foundation that CrewAI-style persona stacks build on.
- Phase 11 · 16 (LangGraph) — the framework this lesson benchmarks against.
- Phase 11 · 19 (Reflexion) — a pattern that maps cleanly to LangGraph but awkwardly to CrewAI.
- Phase 11 · 22 (Production observability) — how to instrument whichever framework you pick.
