# Agno 和 Mastra：生产运行时

> Agno（Python）和Mastra（TypeScript）是2026年生产环境的配对运行时。Agno致力于微秒级的代理实例化和无状态FastAPI后端。Mastra在Vercel AI SDK基底上提供了代理、工具、工作流、统一模型路由和复合存储。

**类型：** 学习
**语言：** Python, TypeScript
**前置条件：** 第14阶段·01（代理循环），第14阶段·13（LangGraph）
**时间：** 约45分钟

## 学习目标

- 识别Agno的性能目标及其适用场景。
- 指出Mastra的三个原语——代理(Agent)、工具(Tool)、工作流(Workflow)——以及支持的服务器适配器。
- 解释为什么推荐将无状态会话级FastAPI后端作为Agno的生产路径。
- 针对给定技术栈（Python优先vs TypeScript优先）选择Agno或Mastra。

## 问题

LangGraph、AutoGen、CrewAI是框架偏重的。希望“仅代理循环，快速，在自己的运行时中”的团队会选择Agno（Python）或Mastra（TypeScript）。两者都牺牲了一些框架自有的原语以换取原始速度和与周围栈的更紧密贴合。

## 核心概念

### Agno

- Python运行时，前身为Phi-data。
- “没有图、链或复杂模式——只有纯Python。”
- 其文档中的性能目标：约2μs代理实例化，每个代理约3.75 KiB内存，约23个模型提供商。
- 生产路径：无状态会话级FastAPI后端。每个请求启动一个新代理；会话状态存储在数据库中。
- 原生多模态（文本、图像、音频、视频、文件）和代理型RAG。

当你每秒有数千个短生命周期代理（聊天扇入、评估管线）时，速度目标很重要。当一个代理运行10分钟时，它们就不那么重要了。

### Mastra

- TypeScript，基于Vercel AI SDK构建。
- 三个原语：**代理(Agent)**、**工具(Tool)**（Zod类型化）、**工作流(Workflow)**。
- 统一模型路由器——跨94个提供商的3300多个模型（2026年3月）。
- 复合存储：将内存、工作流、可观测性存储到不同后端；大规模可观测性推荐ClickHouse。
- Apache 2.0许可，但`ee/`目录采用源代码可用企业许可。
- 支持Express、Hono、Fastify、Koa的服务器适配器；一流的Next.js和Astro集成。
- 自带Mastra Studio（localhost:4111）用于调试。
- 22k+ GitHub星标，1.0版本（2026年1月）每周npm下载量超过30万。

### 定位

两者都不是要成为LangGraph。它们在以下方面竞争：

- **语言适配性。** Agno面向Python优先团队；Mastra面向TypeScript优先团队。
- **运行时人体工学。** Agno = 近乎零开销；Mastra = 与Vercel生态系统集成。
- **可观测性。** 两者都与Langfuse/Phoenix/Opik（第24课）集成，但Mastra Studio是第一方工具。

### 何时选择哪一个

- **Agno** — Python后端，大量短生命周期代理，强性能需求，FastAPI技术栈。
- **Mastra** — TypeScript后端，Next.js / Vercel部署，统一多提供商模型路由，Zod类型化工具。
- **LangGraph**（第13课）— 当持久化状态和显式图推理比原始速度更重要时。
- **OpenAI / Claude Agent SDK** — 当你想要提供商的成品化形态时（第16–17课）。

### 这种模式出错的地方

- **为性能而性能。** 当工作负载是每个请求一次慢速代理调用时，选择Agno只是因为“2μs”听起来不错。开销并非瓶颈。
- **生态系统锁定。** Mastra的Vercel风格集成在Vercel上是优势，在其他地方是劣势。
- **企业许可混淆。** Mastra的`ee/`目录是源代码可用的，并非Apache 2.0。如果你计划分叉，请阅读许可条款。

## 动手构建

本课主要是比较性的——单一的代码示例无法公正对待两个框架。参见`code/main.py`获取一个并排的玩具示例：一个最小的“运行代理、流式输出、持久化会话”流程，实现两次（一次Agno风格，一次Mastra风格）。

运行它：

```
python3 code/main.py
```

两个结构不同但功能等效的追踪。

## 使用它

- **Agno** — 需要速度和FastAPI形态的Python后端。
- **Mastra** — 拥有众多提供商和工作流原语的TypeScript后端。
- 两者都提供第一方可观测性钩子。两者都与Langfuse集成。

## 发布

`outputs/skill-runtime-picker.md`根据技术栈、延迟预算和运营形态选择Agno、Mastra、LangGraph或提供商SDK。

## 练习

1. 阅读Agno文档。将标准库ReAct循环（第01课）移植到Agno。什么消失了？什么保留了？
2. 阅读Mastra文档。将同一循环移植到Mastra。工具类型化（Zod vs 无）发生了什么变化？
3. 基准测试：在你的栈上测量代理实例化延迟。Agno的2μs对你的工作负载重要吗？
4. 设计迁移：如果你一直在Python中使用CrewAI，迁移到Agno会破坏什么？
5. 阅读Mastra的`ee/`许可条款。哪些限制会影响开源分叉？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  Agno  |  "快速Python代理"  |  无状态会话级代理运行时  |
|  Mastra  |  "Vercel AI SDK上的TypeScript代理"  |  代理 + 工具 + 工作流 + 模型路由器  |
|  统一模型路由器  |  "多提供商访问"  |  跨94个提供商的3300多个模型的单一客户端  |
|  复合存储  |  "多个后端"  |  内存/工作流/可观测性各自存储到不同存储  |
|  Mastra Studio  |  "本地调试器"  |  localhost:4111界面用于内省代理  |
|  源代码可用  |  "不是开源"  |  许可允许阅读源代码但限制商业使用  |

## 延伸阅读

- [Agno Agent Framework docs](https://www.agno.com/agent-framework) — 性能目标，FastAPI集成
- [Agno Agent Framework docs](https://www.agno.com/agent-framework) — 原语，服务器适配器，模型路由器
- [Agno Agent Framework docs](https://www.agno.com/agent-framework) — 有状态图替代方案
- [Agno Agent Framework docs](https://www.agno.com/agent-framework) — Mastra集成引用的可观测性比较
