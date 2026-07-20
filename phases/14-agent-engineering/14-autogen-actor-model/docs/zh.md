# AutoGen v0.4：Actor 模型与 Agent 框架

> AutoGen v0.4（微软研究院，2025年1月）基于Actor模型重新设计了Agent编排。异步消息交换、事件驱动Agent、故障隔离、自然并发。该框架现已进入维护模式，而微软Agent框架（2025年10月公开预览）将成为其继任者。

**类型：** 学习+构建
**语言：** Python (标准库)
**前提条件：** 阶段14·01 (智能体循环), 阶段14·12 (工作流模式)
**时间：** 约75分钟

## 学习目标

- 描述Actor模型：Agent作为Actor，消息作为唯一的进程间通信(IPC)，每个Actor的故障隔离。
- 列出AutoGen v0.4的三个API层——Core、AgentChat、Extensions——以及各自用途。
- 解释为什么将消息传递与处理解耦能实现故障隔离和自然并发。
- 用Python实现一个标准库Actor运行时，并将一个双Agent代码审查流程移植到其上。

## 问题

大多数Agent框架是同步的：一个Agent产生消息，一个Agent消费消息，形成调用栈。故障导致整个栈崩溃。并发是后加的。分布式需要重写。

AutoGen v0.4的答案：Actor模型。每个Agent是一个拥有私有收件箱的Actor。消息是唯一的交互方式。运行时将消息传递与处理解耦。故障隔离到单个Actor。并发是原生的。分布式只需更换传输层。

## 核心概念

### Actor

一个Actor包含：

- 私有状态（外部无法直接访问）。
- 收件箱（消息队列）。
- 处理器：`receive(message) -> effects`，其中效果可以是“回复”、“发送给其他Actor”、“生成新Actor”、“更新状态”、“停止自身”。

两个Actor不能共享内存。它们只能发送消息。

### AutoGen v0.4的三个API层

1. **Core.** 低级Actor框架。`AgentRuntime`、`Agent`、`Message`、`Topic`。异步消息交换，事件驱动。
2. **AgentChat.** 任务驱动的高级API（替代v0.2的ConversableAgent）。`AgentRuntime`、`Agent`、`Message`、`Topic`。
3. **Extensions.** 集成——OpenAI、Anthropic、Azure、工具、记忆。

### 为什么解耦很重要

在v0.2模型中，调用`agent_a.chat(agent_b)`会同步阻塞agent_a直到agent_b返回。在v0.4中，`send(agent_b, msg)`将消息放入agent_b的收件箱并立即返回。运行时稍后交付。三个后果：

- **故障隔离。** Agent B崩溃不会导致Agent A崩溃——运行时在B的处理器中捕获故障并决定如何处理（记录、重试、死信）。
- **自然并发。** 多条消息同时传输；Actor并发处理它们的收件箱。
- **分布式就绪。** 无论Actor在进程内还是另一台主机上，收件箱加传输层是相同的抽象。

### 拓扑

- **RoundRobinGroupChat.** Agent按固定顺序轮流发言。
- **SelectorGroupChat.** 一个选择器Agent根据对话上下文选择下一个发言者。
- **Magentic-One.** 参考多Agent团队，用于网页浏览、代码执行、文件处理。基于AgentChat构建。

### 可观测性

内置OpenTelemetry支持。每条消息生成一个span；工具调用根据2026年OTel GenAI语义约定（第23课）携带`gen_ai.*`属性。

### 状态：维护模式

2026年初：AutoGen v0.7.x稳定，适用于研究和原型开发。微软已将活跃开发转移到微软Agent框架（2025年10月1日公开预览；1.0正式版目标2026年第一季度末）。AutoGen模式可以干净地移植——Actor模型是持久的思想。

## 动手构建

`code/main.py`实现了一个标准库Actor运行时：

- `Message`——带有`sender`、`recipient`、`topic`、`body`的类型化载荷。
- `Message`——带有`sender`的抽象类。
- `Message`——事件循环，包含共享队列、交付、故障隔离。
- 一个双Actor演示：`Message`审查代码，`sender`运行检查清单；它们交换消息直到达成一致。

运行它：

```
python3 code/main.py
```

跟踪显示消息传递、一个Actor中模拟的故障不会导致另一个Actor崩溃，以及最终达成共同裁决。

## 使用它

- **AutoGen v0.4/v0.7**（维护）——稳定，适用于研究、原型开发、多Agent模式。
- **微软Agent框架**（公开预览）——前进路径；相同Actor模型思想，刷新API。
- **LangGraph swarm拓扑**（第13课）——通过共享工具交接的类似模式。
- **自定义Actor运行时**——当你需要特定传输层（NATS、RabbitMQ、gRPC）时。

## 发布

`outputs/skill-actor-runtime.md`为一个给定的多Agent任务生成最小的Actor运行时和团队模板（RoundRobin或Selector）。

## 练习

1. 添加死信队列：当处理器抛出异常时，将失败的消息暂存以供人工检查。在你的玩具中，死信队列被触发的频率如何？
2. 实现`SelectorGroupChat`：一个选择器Actor根据对话状态选择谁来处理下一条消息。
3. 添加分布式传输：将进程内队列替换为基于JSON的HTTP服务器，使Actor可以在不同进程中运行。
4. 为每条消息连接一个OTel span（或一个无操作的替代项）。根据第23课，发出`SelectorGroupChat`、`gen_ai.agent.name`。
5. 阅读AutoGen v0.4的架构文章。将你的玩具移植到真实的`SelectorGroupChat` API。在生产中你跳过了哪些重要环节？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  Actor  |  "Agent"  |  私有状态 + 收件箱 + 处理器；无共享内存  |
|  Message  |  "Event"  |  类型化载荷；Actor交互的唯一方式  |
|  Inbox  |  "Mailbox"  |  每个Actor的待处理消息队列  |
|  Runtime  |  "Agent host"  |  路由消息并隔离故障的事件循环  |
|  Topic  |  "Channel"  |  命名发布-订阅路由，连接Actor  |
| 故障隔离  |  "任其崩溃"  |  一个Actor失败不会导致其他崩溃 |
| RoundRobinGroupChat  |  "固定轮换团队"  |  代理按顺序轮流发言 |
| SelectorGroupChat  |  "上下文路由团队"  |  选择器决定下一个发言者 |
| Magentic-One  |  "参考团队"  |  用于网页+代码+文件的多智能体团队 |

## 延伸阅读

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 重新设计的文章
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 图状替代方案
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — AutoGen默认发出的跨度
