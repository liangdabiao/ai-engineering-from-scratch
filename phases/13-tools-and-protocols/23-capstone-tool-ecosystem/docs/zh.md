# 顶石项目 — 构建完整的工具生态系统

> 第13阶段教授了每一个部件。这个顶石项目将它们连接成一个生产级系统：一个MCP服务器（包含工具、资源、提示、任务和UI）、边缘的OAuth 2.1、RBAC网关、多服务器客户端、A2A子代理调用、OTel追踪到收集器、CI中的工具投毒检测，以及AGENTS.md和SKILL.md包。完成后，你可以捍卫每一个架构选择。

**类型：** 构建
**语言：** Python (stdlib, 端到端生态系统工具集)
**前提条件：** 第13阶段 第01至21课
**时间：** ~120分钟

## 学习目标

- 创建一个MCP服务器，通过`ui://`应用暴露工具、资源、提示和任务。
- 在服务器前部署一个OAuth 2.1网关，强制执行RBAC和固定哈希。
- 编写一个多服务器客户端，使用OTel GenAI属性进行端到端追踪。
- 将部分工作负载委托给A2A子代理；验证不透明性得到保留。
- 使用AGENTS.md和SKILL.md打包整个栈，以便其他代理可以驱动它。

## 问题

交付"研究与报告"系统：

- 用户询问："总结2026年arXiv上引用最多的三篇关于代理协议的论文。"
- 系统：通过MCP搜索arXiv；通过A2A将论文摘要任务委托给专门的作者代理；汇总结果；将交互式报告渲染为MCP Apps `ui://`资源；将每一步记录到OTel。

第13阶段的所有原语都出现了。这不是玩具——2026年由Anthropic（Claude Research产品）、OpenAI（带Apps SDK的GPTs）和第三方发布的生成式研究助手系统正是这种形态。

## 核心概念

### 架构

```
[user] -> [client] -> [gateway (OAuth 2.1 + RBAC)] -> [research MCP server]
                                                      |
                                                      +- MCP tool: arxiv_search (pure)
                                                      +- MCP resource: notes://recent
                                                      +- MCP prompt: /research_topic
                                                      +- MCP task: generate_report (long)
                                                      +- MCP Apps UI: ui://report/current
                                                      +- A2A call: writer-agent (tasks/send)
                                                      |
                                                      +- OTel GenAI spans
```

### 追踪层级

```
agent.invoke_agent
 ├── llm.chat (kick off)
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── task transitions (opaque internals)
 ├── mcp.call -> tools/call generate_report (task-augmented)
 │    └── tasks/status polling
 │    └── tasks/result (completed, returns ui:// resource)
 └── llm.chat (final synthesis)
```

一个追踪ID。每个跨度都有正确的`gen_ai.*`属性。

### 安全态势

- OAuth 2.1 + PKCE，资源指示器将受众固定到网关。
- 网关持有上游凭据；用户永远看不到它们。
- RBAC：`alice`拥有`research:read`和`research:write`权限，可以调用所有工具。`bob`拥有`research:read`权限，不能调用`generate_report`。
- 固定描述清单：丢弃任何工具哈希已更改的服务器。
- 双重规则审计：没有工具结合不受信任的输入、敏感数据和后果性操作。

### 渲染

最终的`generate_report`任务返回内容块以及一个`ui://report/current`资源。客户端主机（如Claude Desktop等）在沙箱iframe中渲染交互式仪表板。仪表板包含排序的论文列表、引用计数，以及一个按钮，点击任何论文时调用`host.callTool('summarize_paper', {arxiv_id})`。

### 打包

整个交付物包括：

```
research-system/
  AGENTS.md                     # project conventions
  skills/
    run-research/
      SKILL.md                  # the top-level workflow
  servers/
    research-mcp/               # the MCP server
      pyproject.toml
      src/
  agents/
    writer/                     # the A2A agent
  gateway/
    config.yaml                 # RBAC + pinned manifest
```

用户使用`docker compose up`部署。Claude Code、Cursor、Codex和opencode用户可以通过调用`run-research`技能来驱动系统。

### 第13阶段每课贡献的内容

|  课  |  顶石项目使用的部分  |
|--------|------------------------|
|  01-05  |  工具接口、提供者可移植性、并行调用、模式、代码检查  |
|  06-10  |  MCP原语、服务器、客户端、传输、资源和提示  |
|  11-14  |  采样、根+启发、异步任务、`ui://`应用  |
|  15-17  |  工具投毒、OAuth 2.1、网关+注册表  |
|  18  |  A2A子代理委托  |
|  19  |  OTel GenAI追踪  |
|  20  |  用于LLM层的路由网关  |
|  21  |  SKILL.md + AGENTS.md打包  |

## 使用它

`code/main.py`将先前课程的模式缝合到一个可运行的演示中。全部使用stdlib，全部在进程中，因此你可以端到端地阅读。它运行了研究与报告场景的完整流程：与网关握手、模拟OAuth 2.1、合并tools/list、将generate_report作为任务、向作者进行A2A调用、返回ui://资源、发出OTel跨度。

需要关注的内容：

- 所有跳转使用同一个追踪ID。
- 网关策略阻止第二个用户写入。
- 任务生命周期从执行中到完成，并返回文本和ui://内容。
- A2A调用的内部状态对编排器不透明。
- AGENTS.md和SKILL.md是其他代理重现工作流所需的唯一文件。

## 发布

本课产生`outputs/skill-ecosystem-blueprint.md`。给定一个产品需求（研究、摘要、自动化），该技能生成完整的架构：哪些MCP原语、哪些网关控制、哪些A2A调用、哪些遥测、哪些打包。

## 练习

1. 运行`code/main.py`。注意单一的追踪ID以及跨度如何嵌套。统计演示触及了第13阶段的多少个原语。

2. 扩展演示：添加第二个后端MCP服务器（例如`bibliography`），并确认网关将其工具合并到同一命名空间中。

3. 用运行在子进程中的真实A2A编写器代理替换假的。使用第19课的框架。

4. 在编排器和LLM之间的路由网关中添加PII（个人身份信息）编辑步骤。确认用户查询中的电子邮件被擦除。

5. 为将要维护此系统的队友编写一个AGENTS.md文件。阅读时间应不超过五分钟，并为他们提供在Cursor或Codex中驱动顶点项目所需的一切信息。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
| 顶点项目 | "Phase-13集成演示" | 使用每种原语的端到端系统 |
| 研究与报告 | "场景" | 搜索、总结、渲染模式 |
| 生态系统 | "所有组件整合" | 服务器+客户端+网关+子代理+遥测+包 |
| 追踪层次结构 | "单一追踪ID" | 每个跳转的跨度共享同一个追踪；通过跨度ID建立父子关系 |
| 网关发行令牌 | "传递式认证" | 客户端仅看到网关的令牌；网关持有上游凭据 |
| 合并命名空间 | "所有工具在一个扁平列表中" | 多服务器在网关处合并，冲突时加前缀 |
| 不透明边界 | "A2A调用隐藏内部细节" | 子代理的推理对编排器不可见 |
| 三层栈 | "AGENTS.md + SKILL.md + MCP" | 项目上下文+工作流+工具 |
| 纵深防御 | "多层安全" | 固定哈希、OAuth、RBAC、双重规则、审计日志 |
| 规范合规矩阵 | "我们交付的规范要求" | 将交付物映射到2025-11-25要求的检查表 |

## 延伸阅读

- [MCP — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 综合参考
- [MCP — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 协议的发展方向
- [MCP — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — A2A v1.0参考
- [MCP — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 标准追踪约定
- [MCP — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 生产级代理运行时模式
