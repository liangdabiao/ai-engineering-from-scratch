# 生产运行时：队列、事件、定时任务

> 生产代理运行在六种运行时形态上：请求-响应、流式、持久化执行、基于队列的后台、事件驱动和定时任务。在选择框架之前先选择形态。可观测性是每个形态中的承重构件。

**类型：** 学习
**语言：** Python (标准库)
**前置条件：** 阶段14·13 (LangGraph), 阶段14·22 (语音)
**时间：** 约60分钟

## 学习目标

- 列出六种生产运行时形态，并将每种形态与框架/产品模式匹配。
- 解释为什么持久化执行 (LangGraph) 对长期任务至关重要。
- 描述事件驱动运行时以及何时适合使用 Claude Managed Agents。
- 解释对多步骤代理而言"可观测性即承重"的说法。

## 问题

生产代理的失败方式是 Jupyter notebook 无法暴露的：第37步的网络超时、用户中途挂断语音通话、cron 作业在机器重启时消亡、后台工作进程内存耗尽。运行时形态决定了哪些失败是可存活的。

## 核心概念

### 请求-响应

- 同步 HTTP。用户等待完成。
- 仅适用于短任务（<30秒）。
- 技术栈：Agno (Python + FastAPI)、Mastra (TypeScript + Express/Hono/Fastify/Koa)。
- 可观测性：标准 HTTP 访问日志 + OTel 跨度。

### 流式传输

- SSE 或 WebSocket 用于渐进式输出。
- LiveKit 将其扩展到 WebRTC 用于语音/视频（第22课）。
- 技术栈：任何支持流式的框架 + 处理 SSE/WS 的前端。
- 可观测性：每块时序、首令牌延迟、尾部延迟。

### 持久执行

- 每一步后状态检查点；失败时自动恢复。
- AutoGen v0.4 参与者模型将故障隔离到单个代理（第14课）。
- LangGraph 的核心区分点（第13课）。
- 当步数未知且恢复成本高时必不可少。

### 基于队列/后台

- 作业进入队列，工作进程拾取，通过 Webhook 或发布/订阅返回结果。
- 对长期代理至关重要（每个任务几十到几百步，根据 Anthropic 的计算机使用公告）。
- 技术栈：Celery (Python)、BullMQ (Node)、SQS + Lambda (AWS)、自定义。
- 可观测性：队列深度、每作业延迟分布、死信队列大小。

### 事件驱动

- 代理订阅触发器：新邮件、PR 打开、cron 触发。
- Claude Managed Agents 开箱即用（第17课）。
- CrewAI Flows（第15课）构建事件驱动的确定性工作流。
- 可观测性：触发器来源、事件到启动延迟、代理延迟。

### 定时任务

- 定期运行的 Cron 形态代理。
- 结合持久化执行，使失败的夜间运行能在下一个执行周期恢复。
- 技术栈：Kubernetes CronJob + 持久化框架；托管服务（Render cron、Vercel cron）。

### 2026年部署模式

- **CrewAI Flows** 用于事件驱动生产。
- **Agno** 无状态 FastAPI 用于 Python 微服务。
- **Mastra** 服务器适配器（Express、Hono、Fastify、Koa）用于嵌入。
- **Pipecat Cloud / LiveKit Cloud** 用于托管语音（第22课）。
- **Claude Managed Agents** 用于托管的长时间运行异步任务。

### 可观测性是承重的

没有 OpenTelemetry GenAI 跨度（第23课）加上 Langfuse/Phoenix/Opik 后端（第24课），你无法调试在第40步失败的多步骤代理。这对生产环境不是可选的。这是"我们快速调试"和"我们从零开始重新运行并添加更多日志"之间的区别。

### 生产运行时失败之处

- **错误的形态选择。** 为5分钟的任务选择请求-响应。用户挂断；工作进程堆积；重试叠加。
- **没有死信队列。** 没有死信队列的队列工作进程。失败的作业消失。
- **不透明的后台工作。** 后台代理运行没有追踪导出。故障在用户报告之前不可见。
- **跳过持久化状态。** 任何超过30秒且无法承受重启的运行都需要持久化执行。

## 动手构建

`code/main.py` 是一个标准库多形态演示：

- 请求-响应端点（普通函数）。
- 流式处理器（生成器）。
- 基于队列的工作进程（含死信队列）。
- 事件触发器注册表。
- Cron 形态调度器。

运行它：

```bash
python3 code/main.py
```

输出：五个追踪，展示每种形态在同一任务上的行为。相同的代理逻辑，不同的外部外壳。持久化执行（第六种形态）在第13课中特意通过 LangGraph 检查点覆盖。

## 使用它

- **请求-响应** 用于聊天式用户体验。
- **流式** 用于渐进式响应。
- **持久化** 用于长期任务。
- **队列** 用于批处理/异步/长时间运行。
- **事件** 用于代理响应性。
- **Cron** 用于维护性任务（内存整合、评估、成本报告）。

## 发布

`outputs/skill-runtime-shape.md` 为任务选择运行时形态并连接可观测性需求。

## 练习

1. 将你的第1课 ReAct 循环移植到你技术栈中的所有六种形态。哪种形态适合哪种产品表面？
2. 为基于队列的演示添加死信队列。模拟10%的作业失败；展示死信队列大小。
3. 编写一个 cron 触发的评估代理，每晚针对你当天的前20条追踪运行。
4. 实现带背压的流式：如果客户端慢，则暂停代理。这与轮次预算如何交互？
5. 阅读 Claude Managed Agents 文档。你何时会将自托管的长期代理迁移到托管服务？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  请求-响应  |  "同步"  |  用户等待；仅短任务  |
|  流式  |  "SSE/WS"  |  渐进式输出；更好的用户体验；每块延迟可观察  |
|  持久化执行  |  "从失败恢复"  |  检查点状态；从最后一步重启  |
|  基于队列(Queue-based)  |  "后台任务(Background jobs)"  |  生产者(Producer) / 工作池(Worker pool) / 死信队列(DLQ)  |
|  事件驱动(Event-driven)  |  "基于触发器(Trigger-based)"  |  代理(Agent)对外部事件(External events)做出反应  |
|  死信队列  |  "死信队列(Dead-letter queue)"  |  失败任务(Failed jobs)的停车场(Parking lot)  |
|  Claude 托管代理(Managed Agents)  |  "托管套件(Hosted harness)"  |  Anthropic托管的长时间运行异步任务(Long-running async)，带有缓存(Caching)和压缩(Compaction)  |

## 延伸阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 持久化执行(Durable execution)详情
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 托管长时间运行异步(Hosted long-running async)
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — "每个任务几十到几百步(dozens-to-hundreds of steps per task)"
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 参与者模型(Actor-model)故障隔离
