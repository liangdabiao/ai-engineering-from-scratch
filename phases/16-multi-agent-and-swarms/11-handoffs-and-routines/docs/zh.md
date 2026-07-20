# 交接与例程——无状态编排

> OpenAI 的 Swarm（2024年10月）将多智能体编排提炼为两个原语：**例程**（作为系统提示的指令和工具）和**交接**（返回另一个Agent的工具）。没有状态机，没有分支DSL——LLM通过调用正确的交接工具进行路由。OpenAI Agents SDK（2025年3月）是其生产级后继。Swarm本身仍是最清晰的概念参考——其全部源码仅几百行。这种模式之所以流行，是因为API表面大致为“agent = prompt + tools; handoff = function returning agent”。局限性：无状态，因此记忆是调用方的问题。

**类型：** 学习+构建
**语言：** Python（标准库）
**先决条件：** 阶段16·04（原始模型）
**时间：** 约60分钟

## 问题

每个多智能体框架都希望您学习其DSL：LangGraph的节点和边、CrewAI的团队和任务、AutoGen的GroupChat和管理器。这些DSL是真正的抽象，但它们让事情变得比实际更重。

Swarm则朝相反方向推进：直接使用模型已有的工具调用能力。交接变成工具调用。编排器是当前持有对话的Agent。状态机隐含在Agent的系统提示中。

## 概念

### 两个原语

**例程。** 定义Agent角色和可用工具的系统提示。可以将其视为一组作用域指令：“你是一个分诊Agent；如果用户询问退款，则交接给退款Agent。”

**交接。** Agent可以调用的一个工具，返回一个新的Agent对象。Swarm运行时检测到Agent返回值，并在下一轮切换活动的Agent。

这就是全部抽象。

```
def transfer_to_refunds():
    return refund_agent  # Swarm sees Agent return → switch active agent

triage_agent = Agent(
    name="triage",
    instructions="Route the user to the right specialist.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

分诊Agent的系统提示使其根据用户消息选择正确的交接。LLM的工具调用完成路由。

### 为什么它流行

- **小API。** 只需学习两个概念。
- **利用模型已有的能力。** 工具调用在各供应商中已是生产级。
- **无状态机负担。** 您不需要描述图；Agent的提示描述了它们交接给谁。

### 无状态的权衡

Swarm在运行之间显式地无状态。框架在运行期间保留消息历史，但不持久化任何东西。记忆、连续性、长时间运行的任务——全部是调用方的问题。

在生产中（OpenAI Agents SDK，2025年3月），这是主要变化之一：SDK添加了内置的会话管理、护栏和追踪，同时保留了交接原语。

### Swarm/交接适用的场景

- **分诊模式。** 前台Agent将用户路由给专家。
- **基于技能的交接。** “如果任务需要代码，调用编码员；如果需要研究，调用研究员。”
- **短且有边界的对话。** 客户支持、FAQ转工单、简单工作流。

### Swarm不适用的场景

- **需要共享记忆的长会话。** 交接将对话状态重置为新Agent的提示加上历史记录。没有调用方管理的记忆，Agent之间没有持久状态。
- **并行执行。** 交接是逐一进行的——活动的Agent切换。并行执行需要调用方编排多个Swarm运行。
- **审计和重放。** 无状态运行难以精确重放；LLM的交接选择不是确定性的。

### OpenAI Agents SDK（2025年3月）

生产级后继添加了：

- **会话状态。** 跨运行的持久线程。
- **护栏。** 输入/输出验证钩子。
- **追踪。** 每次工具调用和交接都被记录。
- **交接过滤器。** 控制交接时传递哪些上下文。

交接原语得以保留；生产级的人性化功能被添加在其周围。

### Swarm vs GroupChat

两者都使用LLM驱动的路由，但它们在**谁选择下一步**上不同：

- GroupChat：一个选择器（函数或LLM）从外部选择下一个发言者。
- Swarm：当前Agent通过调用交接工具选择其继任者。

Swarm是“Agent决定下一步”；GroupChat是“管理者决定下一步”。Swarm的决策存在于活动Agent的工具调用中；GroupChat的决策存在于`GroupChatManager`中。

## 动手构建

`code/main.py`从零实现Swarm：一个Agent数据类、一个交接机制（工具返回Agent）、以及一个检测Agent切换的运行循环。

演示：一个分诊Agent路由给退款、销售或支持专家。每个专家有自己的工具。运行循环打印每次交接。

运行：

```
python3 code/main.py
```

## 使用它

`outputs/skill-handoff-designer.md`为给定任务设计交接拓扑：哪些Agent存在、它们可以调用哪些交接、以及哪些上下文被传递。

## 发布

检查清单：

- **交接日志。** 每次交接写一个追踪事件，包含来源Agent、目标Agent、上下文快照。
- **上下文传递规则。** 决定交接时传递什么：完整历史（昂贵）、最后N条消息、或摘要。
- **交接护栏。** 交接给具有不同工具权限的专家时必须经过身份验证——否则提示注入可能强迫不必要的交接。
- **循环检测。** 两个Agent来回交接是常见故障；通过简单的最近K环检查检测。
- **后备Agent。** 如果交接目标不存在，则回退到安全默认值。

## 练习

1. 运行 `code/main.py`，分流至退款代理。确认第二轮的活动代理是退款。
2. 添加循环检测规则：如果相同的两个代理连续交接3次，强制退出。设计回退方案。
3. 阅读 OpenAI Agents SDK 文档中关于交接过滤器(Handoff Filter)的部分。实现一个“交接时总结”版本：在接管代理接管之前，移出代理将上下文压缩为要点总结。
4. 比较 Swarm 交接(Handoff)与 GroupChatManager 选择器(Selector)。哪种模式会使提示注入(Prompt Injection)更严重，为什么？
5. 阅读 Swarm 手册(`code/main.py`)。识别 Swarm 做出的一项明确设计决策，该决策被 OpenAI Agents SDK 更改或保留。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  Routine  |  "代理提示词"  |  系统提示词 + 工具列表。定义了角色和可用的交接。 |
|  Handoff  |  "转移给另一个代理"  |  活动代理可以调用的一个工具，它返回一个新的代理。运行时切换活动代理。 |
|  Stateless  |  "运行间无记忆"  |  Swarm 不持久化任何内容；记忆是调用方的责任。 |
|  Active agent  |  "当前谁在说话"  |  当前持有对话的代理。交接会改变这一点。 |
|  Context transfer  |  "交接时传递什么"  |  关于接管代理能看到什么历史记录的策略：全部、最后 N 条或总结。 |
|  Handoff loop  |  "代理乒乓"  |  两个代理不断互相交接的故障模式。 |
|  OpenAI Agents SDK  |  "生产级 Swarm"  |  2025 年 3 月的后继者；在交接原语之上添加了会话、护栏和追踪。 |
|  Handoff filter  |  "转移时的门控"  |  SDK 功能，可在交接边界检查和修改上下文。 |

## 延伸阅读

- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 参考描述
- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 原始实现，保留作为概念参考
- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 生产级后继者，具有会话和追踪
- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — Claude Code 子代理通过 [OpenAI Swarm repo](https://github.com/openai/swarm) 使用类似交接模式的方式
