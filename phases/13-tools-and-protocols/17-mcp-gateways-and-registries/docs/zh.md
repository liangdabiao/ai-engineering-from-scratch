# MCP网关与注册中心——企业控制平面

> 企业不能让每个开发者随意安装随机的MCP服务器。网关集中管理身份验证、基于角色的访问控制(RBAC)、审计、速率限制、缓存和工具投毒检测，然后将合并后的工具面暴露为单个MCP端点。官方MCP注册中心（由Anthropic、GitHub、PulseMCP和Microsoft共同维护，命名空间已验证）是规范的上游来源。本课说明网关的适用位置，给出一个最小实现，并概述2026年的供应商格局。

**类型:** 学习
**语言:** Python（标准库，最小化网关）
**前置条件:** 阶段13·15（工具投毒），阶段13·16（OAuth 2.1）
**时间:** 约45分钟

## 学习目标

- 解释MCP网关的位置（位于MCP客户端和多个后端MCP服务器之间）。
- 实现网关的五项职责：身份验证、RBAC、审计、速率限制、策略。
- 在网关层强制执行固定工具哈希清单。
- 区分官方MCP注册中心与元注册中心（Glama、MCPMarket、MCP.so、Smithery、LobeHub）。

## 问题

一家财富500强公司拥有30个已批准的MCP服务器、5000名开发者，有合规和审计需求，安全团队希望集中管理策略。让每个开发者在他们的IDE中安装任意服务器是不可行的。

网关模式：

1. 网关作为一个单一的Streamable HTTP端点运行，开发者连接到此端点。
2. 网关持有每个后端MCP服务器的凭据。
3. 每个开发者请求都通过网关自身的OAuth进行身份验证和作用域限制。
4. 网关将调用路由到后端服务器，并应用策略。
5. 所有调用都被记录以供审计。

Cloudflare MCP Portals、Kong AI Gateway、IBM ContextForge、MintMCP、TrueFoundry、Envoy AI Gateway——所有这些都在2025-2026年推出了网关或网关功能。

同时，官方MCP注册中心作为规范的上游来源启动：经过精心策划、命名空间验证、采用反向DNS命名的服务器，网关可以从中拉取。元注册中心（Glama、MCPMarket、MCP.so、Smithery、LobeHub）聚合了来自多个来源的服务器。

## 核心概念

### 网关的五项职责

1. **身份验证.** 使用OAuth 2.1识别开发者，并映射到用户角色。
2. **RBAC.** 按用户策略：哪些服务器、哪些工具、哪些作用域。
3. **审计.** 每次调用都记录谁、什么、何时、结果。
4. **速率限制.** 按用户/工具/服务器设置上限以防止滥用。
5. **策略.** 拒绝被投毒的描述、强制执行双重规则、编辑个人身份信息(PII)。

### 网关作为单一端点

对开发者而言，网关看起来就像一个MCP服务器。内部它会路由到N个后端。会话ID（阶段13·09）在边界处被重写。

### 凭据保管

开发者永远不会看到后端令牌。网关持有它们（或委托给持有它们的身份提供者）。拥有网关``notes:read``的开发者可以传递性地访问笔记MCP服务器，使用网关自身的后端凭据——但仅在绑定传递性访问的策略下。

### 网关层的工具哈希固定

网关持有一份已批准的工具描述清单（SHA256哈希）。在发现时，它会获取每个后端的``tools/list``，将哈希与清单进行比较，并删除任何描述发生变异的工具。这是阶段13·15中的抽地毯防御措施的集中应用。

### 策略即代码

高级网关用OPA/Rego、Kyverno或Styra表达策略。规则如“用户``alice``只能对组织``acme``中的仓库调用``github.open_pr``”被声明式编码。简单网关使用手工编写的Python。两种形式都有效。

### 会话感知路由

当用户的会话包含多个服务器时，网关进行多路复用：开发者单一的MCP会话持有N个后端会话，每个服务器一个。来自任何后端的通知通过网关路由到开发者的会话。

### 命名空间合并

网关合并所有后端的工具命名空间，通常在冲突时添加前缀。``github.open_pr``、``notes.search``。这使得路由清晰无误。

### 注册中心

- **官方MCP注册中心（``registry.modelcontextprotocol.io``）。** 由Anthropic、GitHub、PulseMCP、Microsoft共同管理。命名空间已验证（反向DNS：``io.github.user/server``）。经过基本质量预过滤。
- **Glama.** 以搜索为中心的元注册中心，聚合多个来源。
- **MCPMarket.** 偏向商业的目录，包含供应商列表。
- **MCP.so.** 社区目录，开放提交。
- **Smithery.** 类似包管理器的安装流程。
- **LobeHub.** 在其LobeChat应用中集成了UI的注册中心。

企业网关默认从官方注册中心拉取，允许管理员从元注册中心添加经过策划的补充，并拒绝任何未固定的内容。

### 反向DNS命名

官方注册中心要求公共服务器使用反向DNS名称：``io.github.alice/notes``。命名空间防止抢注并使信任委托更清晰。

### 2026年4月的供应商调查

|  供应商  |  优势  |
|--------|----------|
| Cloudflare MCP Portals  |  边缘托管；集成OAuth；免费层级 |
| Kong AI Gateway  |  原生K8s；细粒度策略；日志输出至OpenTelemetry |
| IBM ContextForge  |  企业IAM；合规；审计导出 |
| TrueFoundry  |  偏向DevOps；指标优先 |
| MintMCP  |  面向开发者平台 |
| Envoy AI Gateway  |  开源；可定制过滤器 |

第17阶段（生产基础设施）深入探讨网关操作。

## 使用它

`code/main.py` 提供了一个约150行的最小化网关：通过伪造的Bearer令牌认证用户，维护基于用户的RBAC策略，将请求路由到两个后端MCP服务器，记录每次调用到审计日志，实施速率限制，并拒绝任何描述哈希值与固定清单不匹配的后端工具。

需要关注的内容：

- `RBAC` 是一个以 `user_id` 为键、包含允许的 `server_tool` 条目的字典。
- `RBAC` 是只允许追加的事件列表。
- 速率限制使用每个用户的令牌桶。
- 固定清单是一个 `RBAC` 字典。

## 发布

本节课生成 `outputs/skill-gateway-bootstrap.md`。给定一个企业MCP计划（用户、后端、合规），该技能生成一份网关配置规范。

## 练习

1. 运行 `code/main.py`。以允许用户身份发起调用；然后以被禁止用户身份；再发起超过速率限制的突发请求。验证所有三种流程。

2. 添加一个策略，在返回结果给客户端之前从结果中编辑掉PII。对SSN格式的字符串使用简单的正则表达式通过；注意缺口（电子邮件、电话号码）。

3. 扩展审计日志以发出 OpenTelemetry GenAI spans。第13·20阶段涵盖了确切的属性。

4. 为一个由50名开发者组成的团队设计RBAC策略，该团队有五个后端（notes、github、postgres、jira、slack）。每个后端谁拥有只读权限？谁拥有写入权限？

5. 从头到尾阅读Cloudflare企业MCP文章。识别出Cloudflare提供而此stdlib网关不具备的一项功能。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
| 网关  |  "MCP代理"  |  客户端和后端之间的集中式服务器 |
| 凭据保管  |  "后端令牌保留在服务端"  |  开发者永远看不到上游令牌 |
| 会话感知路由  |  "多后端会话"  |  网关为每个开发者会话复用N个后端会话 |
| 工具哈希固定  |  "已批准清单"  |  每个已批准工具描述的SHA256哈希；集中阻止拉地毯行为 |
| RBAC  |  "基于用户的策略"  |  基于角色的访问控制，用于工具和服务器 |
| 策略即代码  |  "声明式规则"  |  在网关处强制执行的OPA/Rego、Kyverno、Styra策略 |
| 审计日志  |  "谁、做了什么、何时"  |  仅追加的事件日志，用于合规 |
| 速率限制  |  "每用户令牌桶"  |  每分钟上限以防止滥用 |
| 官方MCP注册表  |  "权威上游"  |  `registry.modelcontextprotocol.io`，命名空间验证 |
| 反向DNS命名  |  "注册表命名空间"  |  `io.github.user/server` 约定 |

## 延伸阅读

- [Official MCP Registry](https://registry.modelcontextprotocol.io/) — 权威上游，命名空间验证
- [Official MCP Registry](https://registry.modelcontextprotocol.io/) — 带有OAuth和策略的网关模式
- [Official MCP Registry](https://registry.modelcontextprotocol.io/) — 开源参考网关
- [Official MCP Registry](https://registry.modelcontextprotocol.io/) — 特性比较文章
- [Official MCP Registry](https://registry.modelcontextprotocol.io/) — 来自IBM的企业网关
