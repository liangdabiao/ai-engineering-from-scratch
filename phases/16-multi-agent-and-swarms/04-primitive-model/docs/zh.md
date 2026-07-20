# 多智能体原语模型(Multi-Agent Primitive Model)

> 每个在2026年发布的多智能体框架——AutoGen、LangGraph、CrewAI、OpenAI Agents SDK、Microsoft Agent Framework——都是四维设计空间中的一个点。只有四个原语(Primitive)：智能体(Agent)、交接(Handoff)、共享状态(Shared State)和编排器(Orchestrator)。本课从零开始构建它们，运行一个涵盖全部四个原语的玩具系统，然后将每个主要框架映射到相同的坐标轴上，这样你只需一段话就能读懂任何新发布的框架。

**类型:** 学习
**语言:** Python (标准库)
**前置条件:** 第14阶段(智能体工程)，第16阶段·01(为何多智能体)
**时间:** 约60分钟

## 问题

每六个月就有一个新的多智能体框架发布。2023年AutoGen，2024年CrewAI，2024年LangGraph和OpenAI Swarm，2025年4月Google ADK，2026年2月Microsoft Agent Framework RC。每篇新闻稿都宣称自己是"正确的抽象"。

如果你试图逐个学习它们，你会精疲力竭。API看起来各不相同。文档对"智能体(Agent)"的定义也不一致。一个框架称其共享内存为"黑板(Blackboard)"，另一个称之为"消息池(Message Pool)"，第三个称之为"StateGraph"。你会开始怀疑这个领域只是在原地打转。

并非如此。在市场营销的表象之下，这四个原语是稳定的。只需学习一次，就能用一段话读懂每个新框架。

## 概念

### 四个原语

1. **智能体(Agent)** — 系统提示词加上工具列表。无状态；每次运行都从其系统提示词和当前消息历史开始。
2. **交接(Handoff)** — 从一个智能体到另一个智能体的结构化控制转移。机制上，是一个返回新智能体的工具调用，或是一个跟随条件的图边。
3. **共享状态(Shared State)** — 多个智能体可以读取（有时写入）的任何数据结构。消息池、黑板、键值存储、向量记忆。
4. **编排器(Orchestrator)** — 决定谁下一个发言的实体。选项：显式图（确定性）、LLM发言选择器（软性）、上一个发言者的交接调用（OpenAI Swarm），或队列上的调度器（群体架构）。

这就是整个设计空间。每个框架为每个轴选择默认值；剩下的只是表面语法。

### 每个2026年框架如何映射到它

|  框架  |  智能体  |  交接  |  共享状态  |  编排器  |
|-----------|-------|---------|--------------|--------------|
|  OpenAI Swarm / Agents SDK  |  `Agent(instructions, tools)`  |  工具返回智能体  |  调用方的问题  |  LLM的下一个交接调用  |
|  AutoGen v0.4 / AG2  |  `ConversableAgent`  |  群聊中的发言选择器  |  消息池  |  选择器函数（LLM或轮询）  |
|  CrewAI  |  `Agent(role, goal, backstory)`  |  `Process.Sequential / Hierarchical`  |  任务输出链式连接  |  管理者LLM或静态顺序  |
|  LangGraph  |  节点函数  |  图边+条件  |  `StateGraph` 归约器  |  图，确定性  |
|  Microsoft Agent Framework  |  智能体+编排模式  |  模式特定  |  线程/上下文  |  模式特定  |
|  Google ADK  |  智能体+A2A卡片  |  A2A任务  |  A2A产物  |  主机决定  |

表面差异看起来巨大。底下：相同的四个旋钮。

### 为何重要

一旦你理解了原语，框架对比就变成了一个简短的检查清单：

- 编排器是信任LLM进行路由（Swarm）还是将路由固定在代码中（LangGraph）？
- 共享状态是完整历史（GroupChat）还是投影（StateGraph归约器）？
- 智能体能否修改彼此的提示词（CrewAI管理者）还是只能交接（Swarm）？

这三个问题回答了80%的框架选择问题。你不再寻找"最佳多智能体框架"，而是开始为你真正关心的轴进行设计。

### 无状态洞察

除了共享状态，每个原语都是无状态的。智能体是（提示词，工具）的函数。交接是函数调用。编排器是调度器。**系统中唯一有状态的东西是共享状态。** 这正是所有有趣错误的来源：内存污染（第15课）、消息排序、版本控制、写竞争。

隐藏共享状态的框架（Swarm）将问题推给调用方。集中共享状态的框架（LangGraph检查点、AutoGen池）使其可检查，但将协调成本转移到了共享状态实现上。

### 单个原语的剖析

#### 智能体

```
Agent = (system_prompt, tools, model, optional_name)
```

无记忆。无状态。具有相同系统提示词和工具的两个智能体是可互换的。一切看似智能体状态的东西实际上都在共享状态或交接协议中。

#### 交接

```
Handoff = (from_agent, to_agent, reason, payload)
```

三种实现占主导地位：

- **函数返回** — 工具返回下一个智能体。这是OpenAI Swarm模式。智能体在其工具模式中携带路由。
- **图边** — LangGraph。边是声明式的。LLM产生一个值；条件选择下一个节点。
- **发言选择** — AutoGen GroupChat。一个选择器函数（有时本身是LLM调用）读取池并选择下一个发言者。

#### 共享状态(Shared state)

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

至少包括消息列表。通常还有更多：结构化工件（CrewAI Task 输出）、类型化上下文（LangGraph 缩减器）、外部记忆（MCP、向量数据库）。

两种拓扑：**全池**（每个智能体看到每条消息）和**投影**（智能体看到角色作用域视图）。全池简单但扩展性差。投影池可扩展但需要预先设计模式。

#### 协调器(Orchestrator)

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

四种类型：

- **静态**——图在构建时固定（LangGraph 确定性、CrewAI 顺序）。
- **LLM 选择**——LLM 读取池并选择下一个发言者（AutoGen、CrewAI 分层）。
- **交接驱动**——当前智能体通过调用交接工具决定（Swarm）。
- **队列驱动**——工作者从共享队列中拉取；没有明确的下一个发言者（群体架构、Matrix）。

### 框架之间变化的内容

一旦原语固定，其余设计决策为：

- **记忆策略**——临时与持久检查点（LangGraph 检查点器）。
- **安全边界**——谁可以批准交接（人在回路）。
- **成本核算**——每个智能体的令牌预算。
- **可观测性**——追踪交接、持久化状态以便重放。

所有这些都可以在原语之上实现。它们都不是新的原语。

## 动手构建

`code/main.py` 用约 150 行标准库 Python 实现了四个原语。没有真正的 LLM——每个智能体都是脚本化策略，因此重点放在协调结构上。

该文件导出：

- `Agent` ——名称、系统提示、工具、策略函数的数据类。
- `Agent` ——返回新智能体的函数。
- `Agent` ——线程安全的消息池。
- `Agent` ——三种变体：`Handoff`、`SharedState`、`Orchestrator`（模拟）。

演示通过所有三种协调器类型运行相同的三个智能体流水线（研究→撰写→审查），并在末尾打印消息池。你可以看到输出仅在*谁选择下一个*方面不同；智能体和共享状态在所有运行中相同。

运行它：

```
python3 code/main.py
```

预期输出：三次协调器运行，每种模式一次。每次打印最终消息池。如果研究者决定提前完成，交接驱动运行到达的智能体较少——这是 LLM 路由折衷的小例子。

## 使用它

`outputs/skill-primitive-mapper.md` 是一项技能，它读取任何多智能体代码库或框架文档并返回四个原语的映射。在新框架版本上运行它以获得一段话的理解，然后再深入阅读文档。

## 发布

在采用新框架之前，为其编写原语映射。如果做不到，说明文档不完整或框架发明了第五个原语（罕见——检查你是否见过共享状态变体）。

将映射固定在架构文档中。当新团队成员加入时，在 API 文档之前发送映射。当框架版本更改时，比较映射而不是更改日志。

## 练习

1. 使用不同的智能体策略运行 `code/main.py` 三次。观察协调器选择如何改变哪些智能体运行。
2. 实现第四种协调器类型：队列驱动型，智能体轮询共享状态以获取工作。可能发生什么死锁？如何检测？
3. 以 LangGraph 快速入门（`code/main.py`）为例，将其重写为四个原语。LangGraph 的哪些抽象是一一映射的，哪些是便利包装？
4. 阅读 OpenAI Swarm 食谱（`code/main.py`）。识别 Swarm 使四个原语中哪一个最符合人体工程学，哪一个推给调用者。
5. 在此表中找到一个完全隐藏共享状态的框架。解释当智能体需要在不重读历史的情况下跨交接协调时会出现什么问题。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  智能体(Agent)  |  "带工具的LLM"  |  A `(system_prompt, tools, model)` 三元组。无状态。  |
|  交接(Handoff)  |  "控制转移"  |  一个结构化调用，指定下一个智能体和可选载荷。三种实现：函数返回、图边、发言者选择。  |
|  共享状态(Shared state)  |  "记忆" / "上下文"  |  多智能体系统中唯一有状态的部分。消息池或黑板。  |
|  协调器(Orchestrator)  |  "协调者"  |  决定谁下一步运行的人。静态图、LLM选择器、交接驱动或队列驱动。  |
|  原语(Primitive)  |  "抽象"  |  每个框架参数化的四个轴之一。不是框架特性。  |
|  消息池(Message pool)  |  "共享聊天历史"  |  全历史共享状态。易于推理，扩展性差。  |
|  投影状态(Projected state)  |  "作用域视图"  |  共享状态中特定角色的视图。可扩展，需要模式设计。  |
|  发言者选择(Speaker selection)  |  "谁下一个发言"  |  协调器模式，其中函数（通常是LLM）从组中选择下一个智能体。  |

## 延伸阅读

- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) ——交接驱动式协调最清晰的表述
- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) ——GroupChat + 发言者选择是LLM选择式协调的参考
- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) ——图边协调和基于缩减器的共享状态
- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) ——角色-目标-背景智能体，顺序/分层流程
- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) ——微软将v0.4转入维护后的实时AutoGen v0.2线路
