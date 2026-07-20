# 根与引出 —— 作用域与飞行中用户输入

> 硬编码路径在用户打开不同项目时立即失效。预填充的工具参数在用户未指定时也会失效。根将服务器作用域限定为用户控制的 URI 集合；引出则在工具调用中途暂停，通过表单或 URL 向用户询问结构化输入。两个客户端原语，对常见 MCP 故障模式的两种修复。SEP-1036（URL 模式引出，2025-11-25）在 2026 上半年之前为实验性功能——依赖它之前请检查 SDK 版本。

**类型：** 构建
**语言：** Python（stdlib，根和引出演示）
**前置条件：** 第13阶段·07（MCP 服务器）
**时间：** ~45 分钟

## 学习目标

- 声明 `roots` 并响应 `notifications/roots/list_changed`。
- 将服务器文件操作限制在已声明根集内的 URI。
- 使用 `roots` 在工具调用中途向用户请求确认或结构化输入。
- 在表单模式和 URL 模式引出之间选择（后者为实验性；注意漂移风险）。

## 问题

笔记 MCP 服务器在生产环境中遇到的两种具体故障。

**路径假设错误。** 服务器针对 `~/notes` 编写。另一位用户在不同机器上使用 `~/Documents/Notes` 中的笔记时，工具调用静默失败（未找到文件），更糟的是可能写入错误位置。

**用户本应知道的参数缺失。** 用户要求“删除旧的 TPS 报告笔记”。模型调用 `notes_delete(title: "TPS report")`，但有来自 2023、2024 和 2025 的三个匹配笔记。工具无法猜测。返回“不明确”令人烦恼；对三者全部操作则后果严重。

根解决第一个问题：客户端在 `initialize` 声明服务器可访问的 URI 集合。引出解决第二个问题：服务器暂停工具调用并发送 `elicitation/create` 让用户选择。

## 核心概念

### 根

客户端在 `initialize` 声明一个根列表：

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

服务器随后可以调用 `roots/list`：

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

服务器必须将根视为边界：任何在根集之外的文件读写操作都将被拒绝。这一点不由客户端强制执行（服务器仍然是用户信任的代码），但符合规范的服务器会遵守它。

当用户添加或移除根时，客户端发送 `notifications/roots/list_changed`。服务器重新调用 `roots/list` 并更新其边界。

### 为什么根是客户端原语

根由客户端声明，因为它们代表了用户的同意模型。用户告诉 Claude Desktop“赋予这个笔记服务器访问这两个目录的权限”。服务器无法扩展该范围。

### 引出：表单模式默认

`elicitation/create` 接受一个表单模式加一个自然语言提示：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Delete 'TPS report'? Multiple notes match; pick one.",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

客户端渲染表单，收集用户答案，返回：

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

三种可能动作：`accept`（用户已填写）、`decline`（用户已关闭）、`cancel`（用户中止了整个工具调用）。

表单模式是扁平的——v1 不支持嵌套对象。SDK 通常会拒绝任何比单层更复杂的东西。

### 引出：URL 模式（SEP-1036，实验性）

2025-11-25 新增。服务器不再发送模式，而是发送一个 URL：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Sign in to GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

客户端在浏览器中打开 URL，等待完成，在用户返回时返回。适用于 OAuth 流程、支付授权和文档签名等表单不足的情况。

漂移风险提示：SEP-1036 响应形状仍在确定中；一些 SDK 返回回调 URL，另一些返回完成令牌。在生产环境中使用 URL 模式之前，请阅读你的 SDK 的发布说明。

### 何时引出是合适的工具

- 破坏性操作前的用户确认（破坏性提示 + 引出）。消除歧义（从 N 个匹配项中选择一个）。首次运行设置（API 密钥、目录、偏好设置）。OAuth 风格流程（URL 模式）。
- 
- 
- 

### 何时引出不合适

- 填写模型本可通过文字询问的工具必需参数。应使用正常的重新提示，而非引出对话框。高频调用。引出会中断对话；不要在循环内触发它。任何服务器可以在事后验证的内容。验证并返回错误，让模型以文本形式询问用户。
- 
- 

### 人在环中桥接

引出加上采样共同实现了 MCP 的“人在环中”模型。服务器的代理循环可以暂停以等待用户输入（引出）或模型推理（采样）。第13阶段·11 介绍了采样；本课介绍引出。将两者结合即可实现完整的环中控制。

## 使用它

`code/main.py` 扩展了笔记服务器，增加了以下功能：

- `roots/list` 服务器在收到根列表更改通知后重新查询的响应。
- 使用 `notes_delete` 来消除多个笔记匹配时的歧义的 `roots/list` 工具。
- 使用 URL 模式引导来打开首次运行配置页面（模拟）的 `roots/list` 工具。
- 一个边界检查，拒绝在声明的根目录之外的 URI 上执行操作。

演示运行三个场景：愉快路径（一个匹配）、消歧（三个匹配，触发引导）、根外写入（被拒绝）。

## 发布

本课产生 `outputs/skill-elicitation-form-designer.md`。给定一个可能需要用户确认或消歧的工具，该技术设计了引导表单架构和消息模板。

## 练习

1. 运行 `code/main.py`。触发消歧路径；确认模拟用户答案被路由回工具。

2. 添加一个新工具 `notes_archive`，每次都需要引导确认（破坏性提示）。检查用户体验：这与模型在文本中重新提问相比如何？

3. 为首次运行的 OAuth 流实现 URL 模式引导。注意漂移风险并添加 SDK 版本保护。

4. 扩展 `roots/list` 处理：当通知到达时，服务器应原子性地重新读取并重新扫描可能已超出范围的文件句柄。

5. 阅读 GitHub 上的 SEP-1036 问题讨论线程。找出一个影响服务器应如何处理 URL 模式回调的未解决问题。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  根目录  |  "同意边界"  |  客户端允许服务器触碰的 URI  |
|  `roots/list`  |  "服务器请求范围"  |  客户端返回当前根集合  |
|  `notifications/roots/list_changed`  |  "用户更改了范围"  |  客户端信号表明根集合已变更  |
|  引导  |  "在调用中询问用户"  |  服务器发起的有结构用户输入请求  |
|  `elicitation/create`  |  "方法"  |  用于引导请求的 JSON-RPC 方法  |
|  表单模式  |  "模式驱动的表单"  |  在客户端 UI 中呈现为表单的扁平 JSON 架构  |
|  URL 模式  |  "浏览器重定向"  |  SEP-1036 实验性；打开一个 URL 并等待  |
|  `accept` / `decline` / `cancel`  |  "用户响应结果"  |  服务器处理的三个分支  |
|  消歧  |  "选择一个"  |  当一个工具有 N 个候选时常见的引导用例  |
|  扁平表单  |  "仅顶层属性"  |  引导架构不能嵌套  |

## 延伸阅读

- [MCP — Client roots spec](https://modelcontextprotocol.io/specification/draft/client/roots) — 规范根引用
- [MCP — Client roots spec](https://modelcontextprotocol.io/specification/draft/client/roots) — 规范引导引用
- [MCP — Client roots spec](https://modelcontextprotocol.io/specification/draft/client/roots) — 2025-11-25 新增内容导览
- [MCP — Client roots spec](https://modelcontextprotocol.io/specification/draft/client/roots) — URL 模式引导提案（实验性，存在漂移风险）
- [MCP — Client roots spec](https://modelcontextprotocol.io/specification/draft/client/roots) — 用户体验导览
