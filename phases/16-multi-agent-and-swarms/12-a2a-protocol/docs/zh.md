# A2A — 代理到代理协议(Agent-to-Agent Protocol)

> Google 于 2025 年 4 月宣布了 A2A；到 2026 年 4 月，该规范已达到 https://a2a-protocol.org/latest/specification/ 版本，并获得 150 多家组织的支持。A2A 是 MCP（第 13 课）的水平补充：MCP 是垂直的（代理 ↔ 工具），而 A2A 是对等的（代理 ↔ 代理）。它定义了代理卡片(Agent Cards)（发现机制）、带有工件(Artifacts)的任务（文本、结构化数据、视频）、不透明任务生命周期(Opaque Task Lifecycles)和身份验证(Auth)。生产系统越来越多地将 MCP 与 A2A 结合使用。Google Cloud 在 2025-2026 年间将 A2A 支持集成到了 Vertex AI Agent Builder 中。

**类型：** 学习 + 构建
**语言：** Python (标准库, `http.server`, `json`)
**前置要求：** 阶段 16 · 04 (原始模型)
**时间：** 约 75 分钟

## 问题

你的代理需要调用另一个系统上的另一个代理。怎么做？你可以暴露一个 HTTP 端点，定义一个专用的 JSON 模式，并希望对方能理解。每对代理都变成了一个定制集成。

A2A 是这种调用的通用有线协议。标准发现机制、标准任务模型、标准传输层、标准工件。就像 HTTP+REST，但代理是头等公民。

## 概念

### 四个要素

**代理卡片(Agent Card)。** 位于 `/.well-known/agent.json` 的 JSON 文档，描述代理：名称、技能、端点、支持的模态、身份验证要求。发现机制通过读取卡片实现。

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**任务(Task)。** 工作单元。一个异步、有状态的对象，具有生命周期：`submitted → working → completed / failed / canceled`。客户端发送任务，轮询或订阅更新。

**工件(Artifact)。** 任务产生的结果类型。文本、结构化 JSON、图像、视频、音频。工件是有类型的，因此不同模态都是头等公民。

**不透明生命周期(Opaque Lifecycle)。** A2A 不规定远程代理如何解决任务。客户端看到状态转换和工件；实现可以自由使用任何框架。

### MCP/A2A 分工

- **MCP** (第 13 课)：代理 ↔ 工具。代理通过 JSON-RPC 对工具服务器进行读写。默认无状态。
- **A2A**：代理 ↔ 代理。对等协议；双方都是具有自身推理能力的代理。

生产环境中的多代理系统两者都使用。A2A 对等端在其一侧调用 MCP 工具。这种分工保持了两种关注点的清晰分离。

### 发现流程

```
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

或者使用流式传输：通过 SSE 订阅 `/tasks/{id}/events` 获取推送更新。

### 身份验证(Auth)

A2A 支持三种常见模式：

- **持有者令牌(Bearer Token)** — OAuth2 或不透明令牌。
- **mTLS** — 双向 TLS；组织之间相互证明身份。
- **签名请求(Signed Requests)** — 对负载的 HMAC。

身份验证在代理卡片(Agent Card)中声明；客户端发现并遵守。

### 截至 2026 年 4 月已有 150 多家组织

企业采用推动了 A2A 的规模化。关键点：A2A 成为企业代理系统跨越信任边界的方式。Google Cloud 发布了 Vertex AI Agent Builder 对 A2A 的支持；Microsoft Agent Framework 也支持它；大多数主流框架（LangGraph、CrewAI、AutoGen）都提供了 A2A 适配器。

### A2A 的优势领域

- **跨组织调用。** 公司 A 的代理调用公司 B 的代理。没有 A2A，每一对都需要定制契约。
- **异构框架。** LangGraph 代理调用 CrewAI 代理调用自定义 Python 代理。A2A 实现了标准化。
- **类型化工件(Typed Artifacts)。** 视频结果、结构化 JSON、音频——都是头等公民。
- **长时间运行的任务。** 不透明生命周期 + 轮询使耗时数小时的任务变得简单。

### A2A 的劣势领域

- **延迟敏感的微调用。** A2A 的生命周期是异步的。亚毫秒级的代理间通信不适合；应使用直接 RPC。
- **紧耦合进程内代理。** 如果两个代理运行在同一个 Python 进程中，A2A 的 HTTP 往返就是过度设计。
- **小团队。** 规范的开销是实际存在的；仅内部使用的代理可能不需要如此正式。

### A2A 与 ACP、ANP、NLIP 的比较

2024-2026 年间出现了几个相关规范：

- **ACP** (IBM/ Linux 基金会) — A2A 的前身，范围更窄。
- **ANP** (代理网络协议) — 侧重对等发现，去中心化优先。
- **NLIP** (Ecma 自然语言交互协议，2025 年 12 月标准化) — 自然语言内容类型。

截至 2026 年 4 月，A2A 是采纳最广泛的对等协议。参见 arXiv:2505.02279（Liu 等人，《代理互操作性协议综述》）以获取对比。

## 动手构建

`code/main.py` 使用 `http.server` 和 JSON 实现了一个 A2A 最小化服务器和客户端。服务器：

- 暴露`/.well-known/agent.json`，
- 接受`/.well-known/agent.json`，
- 管理任务状态，
- 在`/.well-known/agent.json`时返回产物。

客户端：

- 获取Agent Card，
- 提交任务，
- 轮询直到完成，
- 读取产物。

运行：

```
python3 code/main.py
```

脚本在后台线程中启动服务器，然后客户端运行。您会看到完整流程：发现、提交、轮询、产物。

## 使用它

`outputs/skill-a2a-integrator.md`设计A2A集成：Agent Card内容、任务架构、认证选择、流式 vs 轮询。

## 发布

检查清单：

- **固定规范版本。** A2A仍在演进中；Agent Card应声明协议版本。
- **幂等任务创建。** 重复提交（网络重试）应生成一个任务。
- **产物架构。** 声明代理返回的形状；消费者应进行验证。
- **速率限制+认证。** A2A面向公众；应用标准网络安全措施。
- **失败任务的死信队列。** 随时间检查模式以识别重复失败类型。

## 练习

1. 运行`code/main.py`。确认客户端发现服务器并收到正确的产物。
2. 为服务器添加第二个技能（例如，“summarize”）。更新Agent Card。编写一个根据任务类型选择技能的客户端。
3. 实现SSE流端点：`code/main.py`，用于发送状态变更。客户端需要做哪些不同？
4. 阅读A2A规范（`code/main.py`）。找出规范要求但此演示未实现的三件事。
5. 比较A2A（Agent Card发现）与MCP（通过`code/main.py`列出服务器端能力）。自描述代理与能力探测之间的权衡是什么？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  A2A  |  "代理到代理（Agent-to-agent）"  |  代理跨系统调用其他代理的对等协议。Google 2025。  |
|  Agent Card  |  "代理的名片"  |  位于`/.well-known/agent.json`的JSON，描述技能、端点、认证。  |
|  任务(Task)  |  "工作单元"  |  具有生命周期的异步有状态对象；完成后产生产物。  |
|  产物(Artifact)  |  "结果"  |  类型化输出：文本、结构化JSON、图像、视频、音频。一等公民媒体。  |
|  不透明生命周期  |  "如何解决是代理自己的事"  |  客户端看到状态转换；服务器自由选择框架/工具。  |
|  发现(Discovery)  |  "找到代理"  |  `GET /.well-known/agent.json`返回卡片。  |
|  MCP vs A2A  |  "工具 vs 对等"  |  MCP：垂直代理↔工具。A2A：水平代理↔代理。  |
|  ACP / ANP / NLIP  |  "同级协议"  |  相邻规范；A2A是2026年采用最广泛的。  |

## 延伸阅读

- [A2A specification](https://a2a-protocol.org/latest/specification/) — 规范原文
- [A2A specification](https://a2a-protocol.org/latest/specification/) — 2025年4月发布文章
- [A2A specification](https://a2a-protocol.org/latest/specification/) — 参考实现和SDK
- [A2A specification](https://a2a-protocol.org/latest/specification/) — MCP、ACP、A2A、ANP比较
