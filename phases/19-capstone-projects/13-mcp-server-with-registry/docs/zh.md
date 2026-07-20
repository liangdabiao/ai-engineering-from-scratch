# 顶点项目 13 — 带注册表与治理的 MCP 服务器

> 模型上下文协议(MCP)不再只是未来，在2026年成为默认的工具使用规范。Anthropic、OpenAI、Google以及所有主流IDE都发布了MCP客户端。Pinterest发布了其内部MCP服务器生态系统。AAIF注册中心在`.well-known`处正式定义了能力元数据。AWS ECS发布了参考无状态部署。Block的goose-agent将同一协议置于托管助手内。2026年的生产形态是：StreamableHTTP传输、OAuth 2.1作用域、OPA策略门控，以及一个让平台团队能够发现、验证和启用服务器的注册中心。请端到端地构建它。

**类型：** 顶点项目
**语言：** Python（服务器，通过FastMCP）或TypeScript（@modelcontextprotocol/sdk），Go（注册中心服务）
**前置条件：** 阶段11（LLM工程）、阶段13（工具与MCP）、阶段14（智能体）、阶段17（基础设施）、阶段18（安全性）
**涉及阶段：** P11·P13·P14·P17·P18
**时间：** 25小时

## 问题

MCP成为工具使用的通用语言。Claude Code、Cursor 3、Amp、OpenCode、Gemini CLI以及所有托管智能体现在都使用MCP服务器。生产挑战不在于编写服务器（FastMCP让这变得简单），而在于以企业需求规模化部署：每租户OAuth作用域、对破坏性工具的OPA策略、StreamableHTTP无状态扩展、用于发现的注册中心、每次工具调用的审计日志。Pinterest的内部MCP生态系统和AAIF注册中心规范设定了2026年的标杆。

你将构建一个MCP服务器，暴露10个内部工具（Postgres只读、S3列表、Jira、Linear、Datadog等）、一个用于平台发现的注册中心UI，以及一个用于破坏性工具的人工批准门控。负载测试展示了StreamableHTTP的水平扩展。审计跟踪满足企业安全审查。

## 概念

MCP 2026修订版要求StreamableHTTP作为默认传输。与早期的stdio和SSE形态不同，StreamableHTTP默认是无状态的：单个HTTP端点接受JSON-RPC请求，流式传输响应，并支持用于通知的长连接。无状态意味着可以在负载均衡器后面水平扩展。

授权使用OAuth 2.1，并具有每工具作用域。令牌携带诸如`jira:read`、`s3:list`、`postgres:query:readonly`的作用域。MCP服务器在工具调用时（而不仅仅是会话启动时）检查作用域。对于高风险工具，服务器会拒绝任何作用域未在最近N分钟内提升为`approved:by:human`的调用——这种提升来自Slack审批卡片。

注册中心是一个单独的服务。每个MCP服务器暴露一个`.well-known/mcp-capabilities`文档，其中包含其工具清单、传输URL、认证要求。注册中心定期轮询、验证并建立索引。平台团队使用注册中心UI查看哪些工具可用、需要哪些作用域以及由哪个团队拥有。

## 架构

```
MCP client (Claude Code, Cursor 3, ...)
          |
          v
StreamableHTTP over HTTPS (JSON-RPC + streaming)
          |
          v
MCP server (FastMCP) behind load balancer
          |
   +------+------+---------+----------+------------+
   v             v         v          v            v
Postgres    S3 listing  Jira       Linear     Datadog
(read-only) (paged)     (read)     (read)     (query)
          |
   +------+-------------+
   v                    v
 OPA policy gate   destructive tool MCP (separate server)
                        |
                        v
                   human approval via Slack
                        |
                        v
                   audit log (append-only, per-tenant)

  registry service
     |
     v  GET /.well-known/mcp-capabilities from each server
     v
     UI: search / validate / enable-disable / ownership
```

## 技术栈

- 服务器框架：FastMCP（Python）或`@modelcontextprotocol/sdk`（TypeScript）
- 传输：通过HTTPS的StreamableHTTP（无状态）
- 认证：OAuth 2.1，通过SPIFFE/SPIRE的工作负载身份
- 策略：每个工具的OPA/Rego规则；每个请求的策略决策服务
- 注册中心：自托管，使用`@modelcontextprotocol/sdk`清单
- 人工批准：针对破坏性工具的Slack交互式消息
- 部署：AWS ECS Fargate或Fly.io，每租户一个服务器或共享租户作用域
- 审计：每租户结构化JSONL存储桶，包含每次调用的血统信息

## 动手构建

1. **工具表面。** 暴露10个内部工具：Postgres只读查询、S3列出对象、Jira搜索/获取、Linear搜索/获取、Datadog指标查询、PagerDuty值班查询、GitHub只读、Notion搜索、Slack搜索、Salesforce只读。每个工具都有类型化模式和作用域标签。

2. **FastMCP服务器。** 挂载工具。配置StreamableHTTP传输。添加用于OAuth令牌内省和作用域强制的中间件。

3. **OPA策略。** 每个工具的Rego策略：哪些作用域允许调用、应用哪些PII脱敏、应用哪些负载大小上限。每次工具调用时调用决策服务。

4. **注册中心服务。** 独立的Go或TS服务，从注册的服务器轮询`.well-known/mcp-capabilities`，用JSON Schema验证，并暴露列表/搜索/验证/启用-禁用UI。

5. **能力清单。** 每个服务器暴露`.well-known/mcp-capabilities`，包含：工具列表、认证要求、传输URL、所属团队、SLO。

6. **破坏性工具分离。** 修改状态的工具（Jira创建、Linear创建、Postgres写入）位于第二个MCP服务器上，具有更严格的认证流程：令牌必须具有在15分钟内通过Slack卡片提升的`approved:by:human`作用域。

7. **审计日志。** 每租户仅追加JSONL：`{timestamp, user, tool, args_redacted, response_redacted, outcome}`。写入前通过Presidio进行PII脱敏。

8. **负载测试。** StreamableHTTP上100个并发客户端。通过添加第二个副本演示水平扩展；显示负载均衡器无需会话粘性即可重新分配。

9. **一致性测试。** 对两个服务器运行官方MCP一致性套件。通过所有必填部分。

## 使用它

```
$ curl -H "Authorization: Bearer eyJhbGc..." \
       -X POST https://mcp.internal.example.com/ \
       -d '{"jsonrpc":"2.0","method":"tools/call",
            "params":{"name":"postgres.readonly","arguments":{"sql":"SELECT 1"}}}'
[registry]   capability validated: postgres.readonly v1.2
[policy]    scope postgres:query:readonly present; allowed
[audit]     logged: user=u42 tool=postgres.readonly outcome=ok
response:    { "result": { "rows": [[1]] } }
```

## 发布

`outputs/skill-mcp-server.md`描述了交付物。一个生产级MCP服务器+注册中心+内部工具审计层，具有OAuth 2.1作用域和OPA门控。

|  权重  |  标准  |  衡量方式  |
|:-:|---|---|
|  25  |  规范一致性  |  StreamableHTTP + 能力清单通过MCP一致性测试  |
|  20  |  安全性  |  作用域强制、每个工具的OPA覆盖、秘密卫生  |
|  20  |  可观测性  |  每次工具调用审计日志，带PII脱敏  |
|  20  |  可扩展性  |  100客户端负载测试水平扩展演示  |
|  15  |  注册中心用户体验  |  发现/验证/启用-禁用工作流程  |
|  **100**  |   |   |

## 练习

1. 添加一个新工具（Confluence搜索）。通过注册中心验证流程将其发布，而无需修改核心服务器。

2. 编写一条OPA策略，对包含名为`email`、`ssn`或`phone`的列的Postgres查询结果进行脱敏。使用探测查询进行测试。

3. 在本地延迟上对StreamableHTTP与stdio进行基准测试。报告每次调用的p50/p95。

4. 实现每租户配额：每个租户每个工具每分钟最多N次调用。通过第二条OPA规则强制执行。

5. 从[mcp-conformance-tests](https://github.com/modelcontextprotocol/conformance)运行MCP一致性套件并修复所有失败。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
|  StreamableHTTP  |  "2026 MCP传输"  |  无状态HTTP + 流式传输；替代网络服务器的SSE + stdio  |
| 能力清单 | "知名文档" | `.well-known/mcp-capabilities` 附带工具列表、认证、传输URL |
| OPA / Rego | "策略引擎" | 用于根据外部规则授权工具调用的开放策略代理(Open Policy Agent) |
| 权限提升 | "人工批准" | 通过Slack审批授予的短期权限，用于破坏性工具 |
| 注册表 | "工具发现" | 从能力清单中索引MCP服务器的服务 |
| 工作负载身份 | "SPIFFE / SPIRE" | 用于OAuth令牌发放的加密服务身份 |
| 一致性测试套件 | "规范测试" | 官方MCP测试套件，用于StreamableHTTP和工具清单的正确性 |

## 延伸阅读

- [Model Context Protocol 2026 Roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、能力元数据、注册表
- [Model Context Protocol 2026 Roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 2026年注册表规范
- [Model Context Protocol 2026 Roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 参考生产部署
- [Model Context Protocol 2026 Roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 参考内部部署
- [Model Context Protocol 2026 Roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 参考智能体消费模式
- [Model Context Protocol 2026 Roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — Python服务器框架
- [Model Context Protocol 2026 Roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 策略引擎参考
- [Model Context Protocol 2026 Roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 工作负载身份参考
