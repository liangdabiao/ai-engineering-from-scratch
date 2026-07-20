# MCP 传输方式 — stdio 对比 Streamable HTTP 对比 SSE 迁移

> stdio 在本地工作，其他地方不行。Streamable HTTP（2025-03-26）是远程标准。旧的 HTTP+SSE 传输方式已被弃用，将于 2026 年中期移除。选错传输方式会导致一次迁移；选对则能获得一个可远程部署的 MCP 服务器，具有会话持续性和 DNS 重绑定保护。

**类型：** 学习
**语言：** Python（stdlib，Streamable HTTP 端点骨架）
**前置条件：** 阶段 13 · 07、08（MCP 服务器和客户端）
**时间：** 约 45 分钟

## 学习目标

- 根据部署形态（本地 vs 远程，单进程 vs 集群）在 stdio 和 Streamable HTTP 之间选择。
- 实现 Streamable HTTP 单端点模式：POST 用于请求，GET 用于会话流。
- 强制实施 `Origin` 验证和会话 ID 语义，以抵御 DNS 重绑定。
- 在 2026 年中期移除截止日期前，将旧版 HTTP+SSE 服务器迁移到 Streamable HTTP。

## 问题

首个 MCP 远程传输方式（2024-11）是 HTTP+SSE：两个端点，一个用于客户端 POST，另一个用于服务器到客户端流的服务器发送事件通道。它有效，但也笨拙：每个会话两个端点，某些 CDN 前的缓存损坏，以及长期依赖某些 WAF 会积极终止的持久 SSE 连接。

2025-03-26 规范将其替换为 Streamable HTTP：一个端点，POST 用于客户端请求，GET 用于建立会话流，两者共享一个 `Mcp-Session-Id` 标头。此后构建或迁移的每个服务器都使用 Streamable HTTP。旧的 SSE 模式正在被弃用——Atlassian Rovo 于 2026 年 6 月 30 日移除；Keboola 于 2026 年 4 月 1 日；大多数剩余的企业服务器在 2026 年底前移除。

而 stdio 对本地服务器仍然重要。Claude Desktop、VS Code 以及每个类似 IDE 的客户端都通过 stdio 启动服务器。正确的思维模型：stdio 用于“本机”，Streamable HTTP 用于“网络”。无交叉。

## 核心概念

### stdio

- 子进程传输方式。客户端启动服务器，通过 stdin/stdout 通信。
- 每行一个 JSON 对象。以换行符分隔。
- 无会话 ID；进程身份即为会话。
- 无需认证（子进程继承父进程的信任边界）。
- 切勿用于远程服务器——你需要 SSH 或 socat 来隧道，此时应使用 Streamable HTTP。

### Streamable HTTP

单一端点 `/mcp`（或任何路径）。支持三种 HTTP 方法：

- **POST /mcp。** 客户端发送 JSON-RPC 消息。服务器回复单个 JSON 响应，或一个或多个响应的 SSE 流（用于批量响应和与该请求相关的通知）。
- **GET /mcp。** 客户端打开一个持久 SSE 通道。服务器用于服务器到客户端的请求（采样、通知、引导）。
- **DELETE /mcp。** 客户端显式终止会话。

会话由服务器在第一个响应中设置的 `Mcp-Session-Id` 标头标识，客户端在每个后续请求中回显该标头。会话 ID 必须是加密随机（128+ 位）；为了安全，拒绝客户端选择的 ID。

### 单端点 vs 双端点

旧规范中的双端点模式在 2026 年仍可调用——规范声明其为“遗留兼容”。但所有新服务器应使用单端点。官方 SDK 生成单端点；仅在与未迁移的远程通信时使用遗留模式。

### `Origin` 验证与 DNS 重绑定

浏览器不是 MCP 客户端（目前），但攻击者可以构造一个网页，诱使浏览器向 `localhost:1234/mcp` 发送 POST——用户的本地 MCP 服务器在该地址监听。如果服务器不检查 `Origin`，浏览器的同源策略将无法保护它，因为 `Origin: http://evil.com` 是跨源有效的。

2025-11-25 规范要求服务器拒绝其 `Origin` 不在允许列表中的请求。允许列表通常包含 MCP 客户端主机（`https://claude.ai`, `vscode-webview://*`）以及用于本地 UI 的 localhost 变体。

### 会话 ID 生命周期

1. 客户端发送第一个请求时不带 `Mcp-Session-Id`。
2. 服务器分配随机 ID，在响应标头中设置 `Mcp-Session-Id`。
3. 客户端在所有后续请求以及流的 `Mcp-Session-Id` 中回显该标头。
4. 会话可以被服务器撤销；客户端在后续请求中看到 404 并必须重新初始化。
5. 客户端可以显式 DELETE 会话以进行干净关闭。

### 保活与重连

SSE 连接会断开。客户端通过使用相同的 `Mcp-Session-Id` 重新 GET 来重新建立。服务器必须排队在中断期间错过的事件（在合理窗口内）并通过客户端回显的 `last-event-id` 标头重放。

阶段 13 · 13 涵盖任务，即使完全会话重连也能让长时间运行的工作持续。

### 向后兼容性探测

希望同时支持新旧服务器的客户端：

1. POST 到 `/mcp`。
2. 如果响应是带有 JSON 或 SSE 的 `/mcp`，则为 Streamable HTTP。
3. 如果响应是带有 `200 OK` 并且 `200 OK` 标头指向辅助端点的 `/mcp`，则为遗留 HTTP+SSE；遵循 `Content-Type: text/event-stream`。

### Cloudflare、ngrok 和托管

2026 年的生产远程 MCP 服务器运行在 Cloudflare Workers（使用其 MCP Agents SDK）、Vercel Functions 或容器化的 Node/Python 上。关键：你的托管必须支持用于 SSE GET 的持久 HTTP 连接。Vercel 的免费层级上限为 10 秒，不适用。Cloudflare Workers 支持无限流。

### 网关组合

当你通过网关（阶段 13 · 17）前置多个 MCP 服务器时，网关是一个单一的 Streamable HTTP 端点，它会重写会话 ID 并对上游进行多路复用。工具在网关层合并；客户端看到单个逻辑服务器。

### 传输故障模式

- **stdio SIGPIPE.** 子进程在写入中途死亡会引发SIGPIPE；服务器应干净地退出。客户端应检测EOF并将会话标记为失效。
- **HTTP 502 / 504.** Cloudflare、nginx和其他代理在上游故障时会发出这些状态码。Streamable HTTP客户端应在短暂回退后重试一次。
- **SSE连接断开.** TCP RST、代理超时或客户端网络变更会关闭流。客户端使用`Mcp-Session-Id`和可选的`last-event-id`重新连接以恢复。
- **会话撤销.** 服务器使会话id失效；客户端在下次请求时收到404。客户端必须重新握手。
- **时钟偏差.** 客户端上的资源TTL计算与服务器不一致。客户端应将服务器时间戳视为权威。

### 何时绕过Streamable HTTP

一些企业在其内部网络中部署基于gRPC或消息队列传输的MCP服务器。这并非标准做法——MCP的规范并未正式定义这些方式。网关可以向MCP客户端暴露Streamable HTTP接口，同时在内部使用gRPC。保持外部接口符合规范；网关负责转换。

## 使用它

`code/main.py` 使用 `http.server` (标准库) 实现了一个最小的Streamable HTTP端点。它处理 `/mcp` 上的POST、GET和DELETE，在首次响应中设置 `Mcp-Session-Id`，验证 `Origin`，并拒绝来自非许可来源的请求。该处理程序重用了第07课笔记服务器的分发逻辑。

需要关注的内容：

- POST处理程序读取JSON-RPC主体，进行分发，并写入JSON响应（单响应变体；SSE变体结构类似）。
- `Origin` 检查拒绝默认的 `http://evil.example` 探测，但接受 `http://localhost`。
- 会话id是随机的128位十六进制字符串；服务器在内存中维护每个会话的状态。

## 发布

本课程产出 `outputs/skill-mcp-transport-migrator.md`。给定一个HTTP+SSE（旧版）MCP服务器，该技能生成一个迁移计划，迁移到Streamable HTTP，包含会话id连续性、Origin检查和向后兼容的探测支持。

## 练习

1. 运行 `code/main.py`。从 `curl` 发送一个 `initialize` 的POST请求，并观察 `Mcp-Session-Id` 响应头。再发送一个回显该头的POST请求，验证会话连续性。

2. 添加一个打开SSE流的GET处理程序。每五秒发送一个 `notifications/progress` 事件。通过使用相同的会话id重新GET来重新连接，并确认服务器接受它。

3. 实现 `last-event-id` 重放逻辑。重新连接时，重放自该id以来生成的所有事件。

4. 扩展 `Origin` 验证以支持通配符模式 (`https://*.example.com`)，并确认它接受 `https://app.example.com` 但拒绝 `https://evil.example.com.attacker.net`。

5. 从官方注册表中选取一个旧版HTTP+SSE服务器（有多个），并勾勒迁移过程：在端点处理、会话id生成和头语义方面有哪些变化。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  stdio 传输  |  "本地子进程"  |  通过标准输入/输出传输JSON-RPC，换行分隔  |
|  Streamable HTTP  |  "远程传输"  |  单端点POST + GET + 可选SSE，2025-03-26规范  |
|  HTTP+SSE  |  "旧版"  |  双端点模型，将于2026年中移除  |
|  `Mcp-Session-Id`  |  "会话头"  |  服务器分配的随机id，每次后续请求中回显  |
|  `Origin` 白名单  |  "DNS重绑定防御"  |  拒绝Origin未经批准的请求  |
|  单端点  |  "一个URL"  |  `/mcp` 处理所有会话操作的POST/GET/DELETE  |
|  `last-event-id`  |  "SSE重放"  |  用于恢复断开流而不丢失事件的头  |
|  向后兼容探测  |  "新旧检测"  |  客户端响应形状检查，自动选择传输方式  |
|  长寿命HTTP  |  "SSE流式传输"  |  服务器在一个TCP连接上推送数分钟或数小时的事件  |
|  会话撤销  |  "强制重新初始化"  |  服务器使会话id失效；客户端必须重新握手  |

## 延伸阅读

- [MCP — Basic transports spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — stdio和Streamable HTTP的规范参考
- [MCP — Basic transports spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — 引入Streamable HTTP的修订版
- [MCP — Basic transports spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — Workers托管的Streamable HTTP模式
- [MCP — Basic transports spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — 跨部署形态的比较
- [MCP — Basic transports spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — 具体迁移截止日期示例
