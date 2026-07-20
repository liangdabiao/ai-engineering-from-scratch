# MCP App — 通过 `ui://` 提供的交互式 UI 资源

> 仅文本的工具输出限制了Agent能展示的内容。MCP App（SEP-1724，于2026年1月26日正式发布）允许工具返回在Claude Desktop、ChatGPT、Cursor、Goose和VS Code中内联渲染的沙盒交互式HTML。仪表盘、表单、地图、3D场景，全部通过一个扩展实现。本课将讲解`ui://`资源方案、`text/html;profile=mcp-app` MIME类型、iframe沙盒postMessage协议，以及服务器渲染HTML带来的安全风险。

**类型：** 构建
**语言：** Python（标准库，UI资源发射器），HTML（示例应用）
**前置条件：** 阶段13·07（MCP服务器），阶段13·10（资源）
**时间：** 约75分钟

## 学习目标

- 从工具调用返回一个`ui://`资源，并设置正确的MIME类型和元数据。
- 使用`ui://`、`_meta.ui.resourceUri`和`_meta.ui.csp`声明工具关联的UI。
- 实现iframe沙盒postMessage JSON-RPC以支持UI与主机的通信。
- 应用CSP和权限策略默认值，防御源自UI的攻击。

## 问题

2025时代的`visualize_timeline`工具可以返回“以下是按时间顺序排列的14条笔记：……”。这只是一段文字。用户实际想要的是交互式时间线。在MCP App之前，可选方案是：客户端特定的Widget API（Claude工件，OpenAI自定义GPT HTML），或者根本没有UI。

MCP App（SEP-1724，于2026年1月26日发布）标准化了该约定。工具结果包含一个URI为`ui://...`、MIME类型为`text/html;profile=mcp-app`的`resource`资源。主机在一个沙盒iframe中渲染它，应用受限的CSP，除非明确授权，否则无网络访问。iframe内部的UI通过一个小型postMessage JSON-RPC协议向主机发送消息。

每个兼容客户端（Claude Desktop、ChatGPT、Goose、VS Code）都以相同的方式渲染相同的`ui://`资源。一个服务器，一个HTML包，通用UI。

## 核心概念

### `ui://`资源方案

一个工具返回：

```json
{
  "content": [
    {"type": "text", "text": "Here is your notes timeline:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

然后主机对`ui://notes/timeline` URI调用`resources/read`，并得到：

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### iframe沙盒

主机在沙盒化的`<iframe>`内渲染HTML，并应用：

- `sandbox="allow-scripts allow-same-origin"`（或根据服务器声明更严格）
- 通过响应头应用的服务器声明的CSP。
- 无来自主机源（origin）的Cookie或localStorage。
- 网络访问限制为CSP中的`sandbox="allow-scripts allow-same-origin"`。

### postMessage协议

iframe通过`window.postMessage`与主机通信。一个小型JSON-RPC 2.0方言：

始终将`targetOrigin`固定对等方的确切源（origin），并在接收端根据白名单验证`event.origin`，然后再处理任何载荷。永远不要在此通道的任何一侧使用`"*"`——消息体携带工具调用和资源读取。

```js
// iframe to host  (pin to host origin)
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// host to iframe  (pin to iframe origin)
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// receiver on both sides
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // safe to process event.data
});
```

UI可调用的主机端可用方法：

- `host.callTool(name, arguments)` — 调用服务器工具。
- `host.callTool(name, arguments)` — 读取MCP资源。
- `host.callTool(name, arguments)` — 获取提示词模板。
- `host.callTool(name, arguments)` — 关闭UI。

每次调用仍然通过MCP协议，并继承服务器的权限。

### 权限

`_meta.ui.permissions`列表请求额外能力：

- `camera` — 访问用户的摄像头（用于文档扫描UI）。
- `camera` — 语音输入。
- `camera` — 位置信息。
- `camera` — 比`microphone`允许的更广泛的网络访问。

每个权限都是用户在UI渲染前看到的提示。

### 安全风险

iframe中的HTML仍然是HTML。新的攻击面：

- **通过UI的提示注入。** 恶意服务器UI可以显示看起来像系统消息的文本，欺骗用户。主机渲染应明显区分服务器UI和主机UI。
- **通过`connectSrc`的数据窃取。** 如果CSP允许`connect-src: *`，UI可以将数据发送到任何地方。默认应为严格。
- **点击劫持。** UI覆盖主机chrome。主机必须防止z-index操作并强制不透明度规则。
- **窃取焦点。** UI获取键盘焦点并捕获下一条消息。主机必须拦截。

阶段13·15作为MCP安全的一部分深入介绍这些内容；本课只做介绍。

### `ui/initialize`握手

iframe加载后，通过postMessage发送`ui/initialize`：

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

主机响应能力集（capabilities）和会话令牌。UI在每次后续主机调用时使用会话令牌。

### AppRenderer / AppFrame SDK原语

ext-apps SDK 提供了两个便捷原语：

- `AppRenderer`（服务端）— 包装 React/Vue/Solid 组件，并发出一个带有正确 MIME 和元数据的 `ui://` 资源。
- `AppRenderer`（客户端）— 接收资源，挂载 iframe，并中介 postMessage。

你可以使用这些原语，或者手动编写 HTML 和 JSON-RPC。

### 生态状态

MCP Apps 于 2026 年 1 月 26 日发布。截至 2026 年 4 月的客户端支持情况：

- **Claude Desktop。** 自 2026 年 1 月起全面支持。
- **ChatGPT。** 通过 Apps SDK（相同的底层 MCP Apps 协议）全面支持。
- **Cursor。** Beta 版；通过设置启用。
- **VS Code。** 仅限 Insider 版本。
- **Goose。** 全面支持。
- **Zed、Windsurf。** 已列入路线图。

生产环境中的服务器：仪表盘、地图可视化、数据表格、图表构建器、沙箱 IDE 预览。

## 使用它

`code/main.py` 为笔记服务器扩展了一个 `visualize_timeline` 工具，该工具返回一个 `ui://notes/timeline` 资源，并添加了一个针对该 URI 上 `resources/read` 的处理程序，返回一个包含 SVG 时间轴的简洁但完整的 HTML 包。HTML 使用 stdlib 模板化 — 无需构建系统。由于 stdlib 无法驱动浏览器，postMessage 在 JS 注释中进行了勾勒。

需要关注的内容：

- `_meta.ui` 在工具响应中携带 resourceUri、CSP、权限。
- HTML 在无网络访问的情况下渲染；所有数据都内联。
- JS 通过 `host.callTool` 调用 `_meta.ui`（在 stdlib 演示中有文档但处于非激活状态）。

## 发布

本课程产生 `outputs/skill-mcp-apps-spec.md`。给定一个可从交互式 UI 受益的工具，该技能生成完整的 MCP Apps 合约：`ui://` URI、CSP、权限、postMessage 入口点以及安全检查清单。

## 练习

1. 运行 `code/main.py` 并检查发出的 HTML。直接在浏览器中打开 HTML；验证 SVG 渲染。然后勾勒 UI 用于调用 `host.callTool("notes_update", ...)` 的 postMessage 合约。

2. 收紧 CSP：移除 `'unsafe-inline'` 并使用基于 nonce 的脚本策略。HTML 生成代码中有哪些变化？

3. 添加第二个 UI 资源 `ui://notes/editor`，其中包含一个用于原地编辑笔记的表单。当用户提交时，iframe 调用 `host.callTool("notes_update", ...)`。

4. 审计 UI 的攻击面。恶意服务器可能在何处注入内容？iframe 沙箱防御了什么，没有防御什么？

5. 阅读 SEP-1724 规范，并指出这个玩具实现未使用的 MCP Apps SDK 中的一个功能。（提示：组件级状态同步。）

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  MCP Apps  |  "交互式 UI 资源"  |  SEP-1724 扩展于 2026-01-26 发布  |
|  `ui://`  |  "App URI 方案"  |  UI 包的资源方案  |
|  `text/html;profile=mcp-app`  |  "MIME 类型"  |  MCP App HTML 的内容类型  |
|  Iframe 沙箱  |  "渲染容器"  |  通过 CSP 和权限对 UI 进行浏览器沙箱隔离  |
|  postMessage JSON-RPC  |  "UI 到宿主连线"  |  用于宿主调用的微小 JSON-RPC 通过 postMessage 协议  |
|  `_meta.ui`  |  "工具-UI 绑定"  |  将工具结果链接到 UI 资源的元数据  |
|  CSP  |  "Content-Security-Policy"  |  声明允许的脚本、网络、样式来源  |
|  AppRenderer  |  "服务端 SDK 原语"  |  将框架组件转换为 `ui://` 资源  |
|  AppFrame  |  "客户端 SDK 原语"  |  中介 postMessage 的 iframe 挂载辅助程序  |
|  `ui/initialize`  |  "握手"  |  从 UI 到宿主的首次 postMessage  |

## 延伸阅读

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — 参考实现和 SDK
- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — 正式规范文档
- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — 高级文档
- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — 2026 年 1 月发布文章
- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — JSDoc 风格的 SDK 参考
