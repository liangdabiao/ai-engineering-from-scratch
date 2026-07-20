# MCP 安全 II — OAuth 2.1、资源指示器、增量范围

> 远程 MCP 服务器不仅需要身份验证，还需要授权。2025-11-25 规范与 OAuth 2.1 + PKCE + 资源指示器（RFC 8707）+ 受保护资源元数据（RFC 9728）保持一致。SEP-835 通过在 403 WWW-Authenticate 上使用逐步授权来增加增量范围同意。本课将逐步流程实现为状态机，以便您可以看到每一步。

**类型：** 构建
**语言：** Python（标准库，OAuth 状态机模拟器）
**前提条件：** 阶段 13 · 09（传输），阶段 13 · 15（安全 I）
**时间：** ~75 分钟

## 学习目标

- 区分资源服务器与授权服务器的职责。
- 使用 PKCE 保护的 OAuth 2.1 授权码流程。
- 使用 `resource`（RFC 8707）和受保护资源元数据（RFC 9728）防止混淆代理攻击。
- 实现逐步授权：服务器响应 403 并带有 WWW-Authenticate，要求更高的范围；客户端重新提示用户同意并重试。

## 问题

早期的 MCP（2025 年之前）为远程服务器使用临时 API 密钥甚至无身份验证。2025-11-25 规范通过完整的 OAuth 2.1 配置文件填补了这一空白。

三个实际需求：

- **普通远程服务器。** 用户安装一个访问其 Notion / GitHub / Gmail 的远程 MCP 服务器。使用 PKCE 的 OAuth 2.1 是合适的形式。
- **范围升级。** 一个被授予 `notes:read` 的笔记服务器之后可能需要对某个特定操作使用 `notes:write`。不必重新执行整个流程，逐步授权（SEP-835）会请求额外的范围。
- **混淆代理防范。** 客户端持有一个受众限定为服务器 A 的令牌。服务器 A 是恶意的，并试图将令牌呈现给服务器 B。资源指示器（RFC 8707）将令牌绑定到其预期的受众。

OAuth 2.1 并不新鲜。新鲜的是 MCP 的配置文件：特定的必需流程（仅授权码 + PKCE；没有隐式流程，默认没有客户端凭据）、每个令牌请求必须包含资源指示器，以及发布受保护资源元数据以便客户端知道去哪里。

## 核心概念

### 角色

- **客户端。** MCP 客户端（Claude Desktop、Cursor 等）。
- **资源服务器。** MCP 服务器（笔记、GitHub、Postgres 等）。
- **授权服务器。** 颁发令牌。可能与资源服务器是同一个服务，也可能是独立的 IdP（Auth0、Keycloak、Cognito）。

在 MCP 的配置文件中，资源服务器和授权服务器可以是同一主机，但应通过 URL 区分。

### 授权码 + PKCE

流程如下：

1. 客户端生成 `code_verifier`（随机数）和 `code_challenge`（SHA256 哈希）。
2. 客户端将用户重定向到 `code_verifier`。
3. 用户同意。授权服务器重定向到 `code_verifier`。
4. 客户端 POST 到 `code_verifier`。
5. 授权服务器验证验证器的哈希值是否与存储的挑战值匹配，并颁发访问令牌。
6. 客户端使用令牌：在每个指向资源服务器的请求中携带 `code_verifier`。

PKCE 防止授权码拦截攻击。资源指示器防止令牌在其他地方有效。

### 受保护资源元数据（RFC 9728）

资源服务器发布一个 `.well-known/oauth-protected-resource` 文档：

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

客户端从资源服务器发现授权服务器。减少了配置——客户端只需要资源 URL。

### 资源指示器（RFC 8707）

令牌请求中的 `resource` 参数将令牌绑定到其预期的受众。颁发的令牌包含 `aud: "https://notes.example.com"`。接收到此令牌的其他 MCP 服务器会检查 `aud` 并拒绝它。

### 范围模型

范围是用空格分隔的字符串。常见的 MCP 约定：

- `notes:read`、`notes:write`、`notes:delete`
- `notes:read` 用于管理员功能（谨慎使用）
- `notes:read` 用于身份标识

范围选择应遵循最小权限原则：只请求当前需要的，需要更多时再逐步提升。

### 逐步授权（SEP-835）

用户授予了 `notes:read`。之后他们要求代理删除一条笔记。服务器响应：

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

客户端发现 insufficient_scope 错误，向用户显示一个同意对话框以获取额外的范围，执行一个微型 OAuth 流程获取它，然后使用新令牌重试请求。

### 令牌受众验证

每个请求：服务器检查 `token.aud == self.resource_url`。不匹配则返回 401。这阻止了跨服务器令牌重用。

### 短生命周期令牌和轮换

访问令牌应短时有效（默认1小时）。刷新令牌在每次刷新时轮换。客户端在后台处理静默刷新。

### 无令牌透传

采样服务器（阶段13·11）不得将客户端的令牌透传给其他服务。采样请求是边界。

### 混淆代理防范

令牌绑定到`aud`。客户端绑定到`client_id`。每个请求同时对两者进行验证。该规范明确禁止在MCP之前的远程工具生态系统中常见的旧式“令牌透传”模式。

### 客户端ID发现

每个MCP客户端在固定URL发布其元数据。授权服务器可获取客户端元数据文档以发现重定向URI和联系信息。这消除了手动客户端注册。

### 网关与OAuth

阶段13·17展示了企业网关如何处理OAuth：网关持有上游服务器的凭据，颁发给客户端的令牌由网关签发，上游令牌从不离开网关。这颠覆了信任模型——用户只需向网关认证一次；网关处理对N个服务器的授权。

## 使用它

`code/main.py`将完整的OAuth 2.1步进授权流程模拟为状态机。它实现了：

- PKCE码验证器/质询生成。
- 带资源指示器的授权码流程。
- 受保护资源元数据端点。
- 带受众检查的令牌验证。
- 在`insufficient_scope`上进行步进授权。

本课中没有HTTP服务器；状态机在内存中运行，因此您可以追踪每一步。阶段13·17的网关课程将其连接到实际传输。

## 发布

本课产出`outputs/skill-oauth-scope-planner.md`。给定一个带有工具的远程MCP服务器，这一技能设计作用域集、绑定规则和步进授权策略。

## 练习

1. 运行`code/main.py`。追踪双作用域步进授权流程。注意哪些步骤在步进时重复。

2. 添加刷新令牌轮换：每次刷新颁发新的刷新令牌并使旧令牌失效。模拟被盗的刷新令牌在轮换后使用，并确认其失败。

3. 使用标准库http.server将受保护资源元数据端点实现为真实的HTTP响应。模仿第09课的/mcp端点。

4. 为GitHub MCP服务器设计作用域层次结构：读取仓库、写入PR、批准PR、合并PR、管理员。每个级别之间使用步进授权。

5. 阅读RFC 8707和RFC 9728。找出9728中MCP使用方式与RFC示例不同的一个字段。（提示：涉及`scopes_supported`。）

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  OAuth 2.1  |  "现代OAuth"  |  整合的RFC，强制要求PKCE并禁止隐式流程  |
|  PKCE  |  "持有证明"  |  码验证器+质询，抵御授权码拦截  |
|  资源指示器  |  "令牌受众"  |  RFC 8707 `resource`参数，将令牌绑定到单个服务器  |
|  受保护资源元数据  |  "发现文档"  |  RFC 9728 `.well-known/oauth-protected-resource`  |
|  步进授权  |  "增量同意"  |  按需添加作用域的SEP-835流程  |
|  `insufficient_scope`  |  "带有WWW-Authenticate的403"  |  服务器要求针对更大作用域重新授权的信号  |
|  混淆代理  |  "跨服务令牌重用"  |  可信持有者不当地转发令牌的攻击  |
|  短时令牌  |  "访问令牌TTL"  |  快速过期的持有者令牌；由刷新令牌续期  |
|  作用域层次结构  |  "最小权限栈"  |  逐级步进的分层作用域集  |
|  客户端ID元数据  |  "客户端发现文档"  |  客户端发布自身OAuth元数据的URL  |

## 延伸阅读

- [MCP — Authorization spec](https://modelcontextprotocol.io/specification/draft/basic/authorization) — 标准MCP OAuth配置文件
- [MCP — Authorization spec](https://modelcontextprotocol.io/specification/draft/basic/authorization) — 2025-11-25变更的演练
- [MCP — Authorization spec](https://modelcontextprotocol.io/specification/draft/basic/authorization) — 受众绑定RFC
- [MCP — Authorization spec](https://modelcontextprotocol.io/specification/draft/basic/authorization) — 发现文档RFC
- [MCP — Authorization spec](https://modelcontextprotocol.io/specification/draft/basic/authorization) — 实际步进授权流程演练
