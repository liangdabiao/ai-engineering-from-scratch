# 异步任务(SEP-1686) — 即调即取，适用于长时间运行的工作

> 实际代理工作需要几分钟到几小时：CI运行、深度研究综合、批量导出。同步工具调用会断开连接、超时或阻塞UI。SEP-1686（于2025-11-25合并）添加了任务(Tasks)原语：任何请求都可以增强为任务，结果可以在以后获取或通过状态通知流式传输。漂移风险说明：任务在2026年上半年之前是实验性的；SDK接口仍在围绕规范设计。

**类型:** 构建
**语言:** Python（标准库，异步任务状态机）
**先决条件:** 阶段13·07（MCP服务器），阶段13·09（传输）
**时间:** 约75分钟

## 学习目标

- 识别何时将工具从同步提升为任务增强（服务器端工作超过30秒）。
- 遍历任务生命周期：`working` → `input_required` → `completed` / `failed` / `cancelled`。
- 持久化任务状态，使崩溃不会丢失进行中的工作。
- 正确轮询`working`并获取`input_required`。

## 问题

一个`generate_report`工具运行一个持续数分钟的提取管道。同步模型下的选项：

1. 保持连接打开三分钟。远程传输会断开；客户端超时；UI冻结。
2. 立即返回一个占位符；要求客户端轮询自定义端点。破坏了MCP的一致性。
3. 发射后不管；无结果。

都不好。SEP-1686增加了第四种：任务增强。任何请求（通常是`tools/call`）都可以标记为任务。服务器立即返回任务ID。客户端轮询`tasks/status`，完成后获取`tasks/result`。服务器端状态在重启后保留。

## 核心概念

### 任务增强

通过设置`params._meta.task.required: true`（或`optional: true`，由服务器决定），请求变为任务。服务器立即响应：

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl`是服务器保留状态的承诺；TTL之后任务结果被丢弃。

### 每工具选择加入

工具注解可以声明任务支持：

- `taskSupport: "forbidden"` — 此工具始终同步运行。适合快速工具。
- `taskSupport: "forbidden"` — 客户端可以请求任务增强。
- `taskSupport: "forbidden"` — 客户端必须使用任务增强。

一个`generate_report`工具将是`required`。一个`notes_search`工具将是`forbidden`。

### 状态

```
working  -> input_required -> working  (loop via elicitation)
working  -> completed
working  -> failed
working  -> cancelled
```

状态机是仅追加的：一旦进入`completed`、`failed`或`cancelled`，任务即终止。

### 方法

- `tasks/status {taskId}` — 返回当前状态和进度提示。
- `tasks/status {taskId}` — 阻塞或返回404（如果尚未完成）。
- `tasks/status {taskId}` — 幂等；终止状态忽略。
- `tasks/status {taskId}` — 可选；枚举活动任务和最近完成的任务。

### 流式状态变更

当服务器支持时，客户端可以订阅状态通知：

```
server -> notifications/tasks/updated {taskId, state, progress?}
```

流式传输而非轮询的客户端获得更好的用户体验。轮询始终作为最小接口得到支持。

### 持久状态

规范要求声明任务支持的服务器持久化状态。崩溃不应丢失TTL内的已完成结果。存储方式从SQLite到Redis到文件系统。第13课测试工具使用文件系统。

### 取消语义

`tasks/cancel`是幂等的。如果任务正在执行，服务器尝试停止（检查执行器协作取消）。如果已经终止，则请求无操作。

### 崩溃恢复

当服务器进程重启时：

1. 加载所有持久化的任务状态。
2. 将任何进程死亡的`working`任务标记为`failed`，错误`CRASH_RECOVERY`。
3. 保留`working` / `failed` / `CRASH_RECOVERY`直到其TTL。

### 异步任务加采样

任务本身可以调用`sampling/createMessage`。这就是长时间运行的研究任务的工作方式：服务器的任务线程根据需要采样客户端的模型，而客户端的UI以`working`显示任务并定期更新进度。

### 为什么这是实验性的

SEP-1686 于 2025-11-25 发布，但更广泛的路线图指出了三个未解决的问题：持久订阅原语、子任务（父子任务关系）和结果-TTL 标准化。预计该规范将在 2026 年持续演进。生产代码应仅将 Tasks 视为常见情况下的稳定功能，并防范未来针对子任务的 SDK 更改。

## 使用它

`code/main.py` 实现了一个持久任务存储（基于文件系统）和一个在后台线程中运行的 `generate_report` 工具。客户端调用该工具，立即获取任务 ID，在工作线程更新进度时轮询 `tasks/status`，并在完成后获取 `tasks/result`。取消功能有效；通过终止工作线程并重新加载状态来模拟崩溃恢复。

需要关注的内容：

- 任务状态 JSON 持久化到 `/tmp/lesson-13-tasks/<id>.json`。
- 工作线程更新 `/tmp/lesson-13-tasks/<id>.json` 字段；轮询显示其进展。
- 客户端取消设置一个事件；工作线程检查并提前退出。
- “崩溃”时重新加载状态，将正在进行的任务标记为 `/tmp/lesson-13-tasks/<id>.json` 并带有 `progress`。

## 发布

本课程产出 `outputs/skill-task-store-designer.md`。给定一个长时间运行的工具（研究、构建、导出），该技能设计任务存储（状态形状、ttl、持久性），选择正确的 taskSupport 标志，并勾勒进度通知。

## 练习

1. 运行 `code/main.py`。启动一个 `generate_report` 任务，轮询状态，然后获取结果。

2. 在运行途中添加一个 `tasks/cancel` 调用。验证工作线程是否遵从该调用，并且状态变为 `cancelled`。

3. 模拟崩溃恢复：终止工作线程，重新启动加载器，并观察 `CRASH_RECOVERY` 故障模式。

4. 将存储扩展为 SQLite。持久性优势相同；查询选项增多（列出来自会话 X 的所有任务）。

5. 阅读 MCP 2026 年路线图文章。确定最有可能在明年影响 SDK API 设计的一个与 Tasks 相关的未解决问题。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  Task  |  "长时间运行的工具调用"  |  为异步执行增加了 `_meta.task` 的请求  |
|  SEP-1686  |  "Tasks spec"  |  于 2025-11-25 添加 Tasks 的规范演进提案  |
|  `_meta.task`  |  "Task envelope"  |  包含 id、state、ttl 的每请求元数据  |
|  taskSupport  |  "Tool flag"  |  每个工具的 `forbidden` / `optional` / `required`  |
|  `tasks/status`  |  "Poll method"  |  获取当前状态和可选的进度提示  |
|  `tasks/result`  |  "Fetch result"  |  返回已完成的有效载荷，如果尚未完成则返回 404  |
|  `tasks/cancel`  |  "Stop it"  |  幂等取消请求  |
|  ttl  |  "Retention budget"  |  服务器承诺保留任务状态的毫秒数  |
|  `notifications/tasks/updated`  |  "State push"  |  服务器发起的状态变更事件  |
|  Durable store  |  "Crash-safe state"  |  文件系统 / SQLite / Redis 持久化层  |

## 延伸阅读

- [MCP — GitHub SEP-1686 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) — 原始提案及完整讨论
- [MCP — GitHub SEP-1686 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) — 含原理的设计详解
- [MCP — GitHub SEP-1686 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) — 机制与状态机
- [MCP — GitHub SEP-1686 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) — SDK 级别的任务实现模式
- [MCP — GitHub SEP-1686 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) — 未解决问题及 2026 年优先级（包括子任务）
