## §12 Claude Agent SDK 实战：子代理与生命周期

2024 年底我第一次跑子代理。几百步的任务，主代理不用自己扛。那种"分身"感很上头。

### 12.1 Client SDK 还是 Agent SDK

- `anthropic`：裸 Messages API。循环你写。
- `claude-agent-sdk`：Claude Code 的 harness 变库。工具、MCP、hooks、子代理、session store 都带。

> **标叔的经验**：要省事就上 Agent SDK
>
> 我早期手搓循环。后来接 SDK，生命周期和持久化一行搞定。

### 12.2 子代理：隔离而非堆叠

两种用途：

- 并行：20 个模块找测试文件 = 20 个并行子代理。
- 隔离：子代理用自己的上下文，只回结果。主代理预算不被撑爆。

### 12.3 生命周期 Hooks

- PreToolUse / PostToolUse：给工具调用加闸、审计。
- SessionStart / SessionEnd：起停清理。
- UserPromptSubmit：用户的话进模型前先处理。

> **核心建议**：hooks 别乱加
>
> 每队都加 hook，启动越来越慢。按季度审查。

### 12.4 Session Store

`append / load / list_sessions / delete / list_subkeys`。`--session-mirror` 把 transcript 实时镜像到文件，方便调试。

### 12.5 W3C trace

OTel span 通过 W3C 头传进 CLI 子进程。多进程的 trace 在后端合成一条。

手写 harness 骨架：

```python
class SessionStore:                  # 标叔的极简版
    def append(self, sid, msg): ...
    def load(self, sid): ...
hooks = {"PreToolUse": rate_limit, "Stop": cleanup}
```

[向前桥接] Claude 这边讲完。下一章，OpenAI 那套。
