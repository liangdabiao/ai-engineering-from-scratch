# 生产环境扩展——队列、检查点与持久性

> 将多智能体系统扩展到数千个并发运行需要**持久执行**。LangGraph的运行时在每个超级步骤后写入由 `thread_id` 键控的检查点（默认为Postgres）；工作进程崩溃时会释放租约，另一个工作进程恢复运行。智能体可以无限期休眠等待人工输入。**MegaAgent**（arXiv:2408.09955）运行了一个每个智能体的生产者-消费者队列，具有三种状态（空闲/处理中/响应）和两层协调（组内聊天+组间管理聊天）。**纤程/异步**在LLM流式传输中优于每任务线程：线程99%的时间处于空闲等待令牌状态，纤程则在I/O上协作式让出。反方观点：Ashpreet Bedi的"Scaling Agentic Software"主张**FastAPI + Postgres + 其他什么也不用**，直到负载证明需要更多——简单架构往往超出预期。本课构建一个持久检查点日志、一个带状态转换的每智能体工作队列、一个异步与线程对比演示，并落脚于务实的“从简单开始”原则。

**类型：** 学习+构建
**编程语言：** Python（标准库，`asyncio`，`sqlite3`）
**前置要求：** 阶段16 · 09（并行群体网络），阶段16 · 13（共享内存）
**时间：** 约75分钟

## 问题

一个原型多智能体系统在单台笔记本电脑上以内存事件循环方式运行三个智能体。当迁移到生产环境时：

- 智能体有时会运行数小时（长时间研究、人在回路等待）。
- 工作进程崩溃。重启会丢失状态。
- 峰值负载是平均值的10倍；需要水平扩展。
- 用户按每次智能体运行付费；需要精确一次语义进行计费。

内存事件循环无法满足这些需求。你需要一个底层的持久执行层。2026年的标准选项包括：

1. 带检查点的工作流引擎（Temporal、LangGraph运行时）。
2. 带状态存储的消息队列（Postgres + SQS/RabbitMQ）。
3. 参与者模型框架（MegaAgent的每智能体生产者-消费者）。
4. 自主搭建的FastAPI + Postgres（Bedi的主张）。

本课将构建每种方案的一个微型实例。

## 概念

### 持久执行模式

持久执行引擎在每个“步骤”（在LangGraph术语中为超级步骤）后持久化完整的程序状态。当崩溃时：

```
worker crashes mid-step
  -> lease timeout
  -> another worker picks up the thread_id
  -> resumes from last checkpoint
  -> no duplicate side effects
```

实现此功能的要求：

- **可序列化的状态。** 所有智能体状态必须可持久化。持有活跃数据库连接的函数闭包无法存活。
- **确定性恢复。** 给定相同状态和相同输入，智能体产生相同动作（或将LLM调用委托给外部确定性预言机）。
- **幂等的副作用。** 外部调用（工具调用、支付）必须是幂等的，或使用去重键。

LangGraph在每个超级步骤后写入检查点；Temporal在每个活动后写入；Restate使用事件溯源日志。三者实现了相同的模式。

### LangGraph的运行时

每个智能体有一个 `thread_id`；状态是一个类型化的字典；每个超级步骤向检查点表写入一行。恢复时，运行时从最后一个检查点重放，而不是从头开始。智能体可以 `interrupt()` 等待人工输入；运行时持久化并释放工作进程。当输入到达时，任何工作进程都可以恢复。

这是2026年4月的参考生产设计。

### MegaAgent的每智能体队列

arXiv:2408.09955描述了一个规模实验：一个集群中数千个并发智能体。架构：

```
agent i:
  state ∈ {Idle, Processing, Response}
  in_queue   <- messages addressed to agent i
  out_queue  -> replies + side effects

coordinators:
  intra-group chat  (agents in the same group)
  inter-group admin chat  (high-level routing)
```

两层协调使得组内对话密集进行，而组间保持稀疏——这是用于在数千个智能体中保持成本线性的模式。

### 异步 vs 每任务线程

LLM调用是I/O密集型的。等待下一个令牌的线程99%的时间处于空闲状态。每个线程大约消耗1MB内存；在10,000个并发调用下，仅栈内存就需要10GB。

纤程（Python `asyncio`、Go协程、Rust `tokio`）在I/O上协作式让出。同样的10,000个调用可以轻松容纳在一个进程中。在LLM智能体规模下，异步不是优化——它就是架构。

例外：CPU密集型后处理（嵌入、分词技巧）仍然需要线程或进程。将I/O层与CPU层分离。

### Bedi的反方观点

"Scaling Agentic Software"（Ashpreet Bedi, 2026）认为，大多数团队在测量负载之前就过度设计了。务实的默认方案：

- FastAPI + Postgres。
- 每次智能体运行为一行；使用乐观并发原地更新状态。
- 通过 `pg_notify` 或简单的Celery工作进程执行后台任务。
- 在应用程序代码中实现重试策略。

对于可管理任务中低于约100个并发智能体运行的负载，这通常已足够。当测量到它失败时再升级。

规则：当遇到简单架构无法解决的具体问题时，再采用持久执行框架。过早采用会在毫无回报的流程上浪费时间。

### 精确一次语义

对于付费智能体运行，你需要“精确一次有效”（至少一次投递 + 幂等消费者）。工程措施：

- **每次运行的去重键。** 在每次副作用调用中包含它。
- **发件箱模式(Outbox Pattern)。** 副作用先写入一个表，然后一个单独的进程执行它们。两个步骤都是幂等的。
- **补偿事务(Compensating Transactions)。** 当副作用成功但其跟踪写入失败时，调度一个补偿操作。

这些是数据库工程模式，而非LLM特有。LLM带来的额外开销仅在于LLM调用速度慢；其他一切都是标准的分布式系统问题。

### 彩虹部署(Rainbow Deployment)

Anthropic的多智能体研究系统使用“彩虹部署”：多个版本的智能体运行时同时运行，这样长时间运行的智能体不必在每次代码部署时被杀死。在一部分流量上对新版本进行金丝雀测试；旧版本在其智能体完成后退役。

这对于长时间运行的有状态系统来说是标准的；2026年的改进在于智能体可能运行数小时，因此部署周期必须适应。

### 标准生产环境检查清单

- 持久化状态(检查点、快照或发件箱+可重放日志)。
- 幂等的副作用。
- 用于LLM调用的异步I/O层。
- 至少一次投递(At-Least-Once Delivery) + 去重。
- 针对有状态工作负载的彩虹/金丝雀部署。
- 可观测性：每个智能体的追踪、超级步(Super-Step)审计、重试计数。

## 动手构建

`code/main.py` 实现：

- `CheckpointStore` — 基于SQLite的检查点日志，使用线程ID作为键。每个超级步追加一行。
- `CheckpointStore` — 模拟运行中途崩溃；第二个工作者从最后一个检查点恢复。
- `CheckpointStore` — 每个智能体的空闲/处理/响应状态机，带一个小型工作队列。
- `CheckpointStore` — 通过asyncio和线程运行500个并发的模拟“LLM调用”；报告墙上时间和峰值内存(近似值)。

运行：

```
python3 code/main.py
```

预期输出：模拟崩溃后检查点恢复成功；异步版本在<1秒内处理500个并发调用；线程版本耗时数秒，且每个并发单元消耗的内存高几个数量级。

## 使用它

`outputs/skill-scaling-advisor.md` 就持久化执行选择提出建议：FastAPI + Postgres、LangGraph运行时、Temporal或自定义。根据负载、状态保留需求和部署频率进行校准。

## 发布

标准生产环境加固：

- **从简单开始(Bedi法则)。** 使用FastAPI + Postgres，直到你测量到它失败。
- **在优化前先测量一切。** 每次运行的延迟直方图、每步耗时、重试次数、失败分类。
- **针对副作用的发件箱模式。** 特别是支付和外部API调用。
- **彩虹部署。** 永远不要在部署期间杀死正在进行的智能体运行。
- **采用持久化执行引擎(Temporal / LangGraph / Restate)当**遇到特定问题时：长达一小时的人工介入等待、跨区域协调、复杂的重试/补偿策略。
- **为I/O层使用异步。** 线程仅用于CPU密集型后处理。

## 练习

1. 运行`code/main.py`。确认检查点恢复工作正常；测量异步与线程并发差异。
2. 实现一个**发件箱**表：每个工具调用先写入发件箱，然后一个单独的goroutine/任务执行。通过运行两次工具调用来验证幂等性。
3. 模拟一个**彩虹部署**：两个并发运行时版本；将一半的新线程ID路由到每个版本；确认旧版本上的正在进行中的线程不会被中断。
4. 阅读LangGraph运行时文档(如下链接)。确定运行时的哪些功能在手动实现的FastAPI + Postgres版本中最难复制。这是采用的理由，还是可以推迟？
5. 阅读MegaAgent (arXiv:2408.09955) 第3节。两层协调(组内 + 组间管理员聊天)是明确的。草拟如何将其映射到具有两个队列族的消息队列。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  持久化执行(Durable Execution)  |  "持久化程序状态"  |  引擎在每次超级步后写入状态；崩溃恢复是确定性的。  |
|  超级步(Super-Step)  |  "事务边界"  |  检查点之间的工作单元。LangGraph术语。  |
|  thread_id  |  "智能体运行标识符"  |  绑定检查点和恢复逻辑的键。  |
|  幂等性(Idempotency)  |  "可安全重试"  |  重复执行副作用产生与一次尝试相同的结果。  |
|  发件箱模式(Outbox Pattern)  |  "解耦副作用"  |  将意图写入表；一个单独的执行器执行并标记完成。  |
|  至少一次投递(At-Least-Once Delivery)  |  "可能的重复"  |  消息队列语义；去重键使消费者实现有效一次。  |
|  彩虹部署(Rainbow Deploy)  |  "重叠版本"  |  长时间运行的工作负载期间多个运行时版本同时存在。  |
|  异步纤程(Async Fiber)  |  "协作式让步"  |  用户态并发；对于I/O密集型负载，与线程相比开销小。  |
|  检查点(Checkpoint)  |  "状态快照"  |  超级步边界处的序列化状态；恢复的关键。  |

## 延伸阅读

- [LangChain — The runtime behind production deep agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — LangGraph运行时设计
- [LangChain — The runtime behind production deep agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — 每个智能体的生产者-消费者队列；数千个并发智能体的两层协调
- [LangChain — The runtime behind production deep agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — 以消息队列作为协调基础的去中心化框架
- [LangChain — The runtime behind production deep agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — 持久化执行的参考工作流引擎
- [LangChain — The runtime behind production deep agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — 包括彩虹部署在内的生产环境经验教训
