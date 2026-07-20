# 顶点项目 01 — 终端原生编码智能体

> 到2026年，编码代理的形状已经确定。一个TUI框架、一个状态化计划、一个沙盒化工具表面、一个计划-行动-观察-恢复的循环。Claude Code、Cursor 3和OpenCode从50英尺外看都一样。这个顶点项目要求你端到端地构建一个——从CLI输入到Pull Request输出——并在SWE-bench Pro上与mini-swe-agent和Live-SWE-agent进行对比评估。你将了解到难点不在于模型调用，而在于工具循环、沙盒以及50回合运行的成本天花板。

**类型：** 顶点项目
**语言：** TypeScript / Bun（框架），Python（评估脚本）
**先决条件：** 第11阶段（LLM工程），第13阶段（工具与协议），第14阶段（代理），第15阶段（自主系统），第17阶段（基础设施）
**涉及阶段：** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**时间：** 35小时

## 问题

编码代理在2026年成为主导的AI应用类别。Claude Code (Anthropic)、Cursor 3 with Composer 2 and Agent Tabs (Cursor)、Amp (Sourcegraph)、OpenCode (112k星)、Factory Droids和Google Jules都发布了同一架构的变体：一个终端框架、一个授权工具表面、一个沙盒，以及一个围绕前沿模型构建的计划-行动-观察循环。前沿很窄——Live-SWE-agent在SWE-bench Verified上使用Opus 4.5达到了79.2%——但工程工艺很广。大多数失败模式并非模型错误，而是工具循环不稳定、上下文污染、失控的Token成本以及破坏性的文件系统操作。

你无法从外部理解这些代理。你必须构建一个，观察循环在第47回合因ripgrep返回8MB匹配而崩溃，然后重建截断层。这就是这个顶点项目的意义所在。

## 概念

框架有四个表面。**计划**维护一个类似TodoWrite的状态对象，模型每回合重写它。**行动**分派工具调用（读取、编辑、运行、搜索、git）。**观察**捕获stdout/stderr/退出码，截断，并反馈摘要。**恢复**处理工具错误，而不炸开上下文窗口或无限循环。2026年的形状增加了一件事：**钩子**。`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop`和`PreCompact`——可配置的扩展点，操作员在此注入策略、遥测和防护栏。

沙盒是E2B或Daytona。每个任务在一个新的开发容器中运行，挂载一个可读写的工作树。框架从不接触主机文件系统。工作树在成功或失败时被拆除。成本控制在三层实施：每回合Token上限、每次会话美元预算以及硬回合限制（通常为50）。可观测性层是带有GenAI语义约定的OpenTelemetry跨度，发送到自托管的Langfuse。

## 架构

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## 技术栈

- 框架运行时：Bun 1.2 + Ink 5 (React-in-terminal)
- 模型访问：OpenRouter统一API，支持Claude Sonnet 4.7、GPT-5.4-Codex、Gemini 3 Pro、Opus 4.5（用于最困难的任务）
- 工具传输：Model Context Protocol StreamableHTTP (MCP 2026修订版)
- 沙盒：E2B沙盒（JS SDK）或Daytona开发容器
- 代码搜索：ripgrep子进程，支持17种语言的tree-sitter解析器（预编译）
- 隔离：`git worktree add`每个任务，成功/失败时清理
- 评估框架：SWE-bench Pro（已验证子集）+ Terminal-Bench 2.0 + 你自己的30任务保留集
- 可观测性：OpenTelemetry SDK with `git worktree add` semconv → 自托管Langfuse
- PR发布：带有细粒度Token的GitHub App，作用域限制在目标仓库

## 动手构建

1. **TUI和命令循环。** 用Ink搭建一个Bun项目。接受`agent run <repo> "<task>"`。打印一个分屏视图：计划窗格（顶部）、工具调用流（中间）、Token预算（底部）。添加Ctrl-C取消功能，在退出前触发`SessionEnd`钩子。

2. **计划状态。** 定义一个类型化的TodoWrite模式（待处理/进行中/已完成项，带备注）。模型每回合通过工具调用重写整个状态——不允许增量修改。将计划持久化到`.agent/state.json`，以便崩溃后可以恢复。

3. **工具表面。** 定义六个工具：`read_file`、`edit_file`（带差异预览）、`ripgrep`、`tree_sitter_symbols`、`run_shell`（带超时）、`git`（状态/差异/提交/推送）。通过MCP StreamableHTTP暴露，以便框架与传输无关。每个工具返回截断后的输出（每次调用上限4k Token）。

4. **沙盒包装。** 每个任务生成一个E2B沙盒。`git worktree add -b agent/$TASK_ID`一个新分支。所有工具调用在沙盒内执行。主机文件系统不可达。

5. **钩子。** 实现所有八种2026钩子类型。至少连接四个用户编写的钩子：(a) `PreToolUse`破坏性命令防护，阻止`rm -rf`在工作树之外执行；(b) `PostToolUse` Token核算；(c) `SessionStart`预算初始化；(d) `Stop`写入最终追踪包。

6. **评估循环。** 克隆SWE-bench Pro Python的30个问题子集。用你的框架运行每个问题。与mini-swe-agent（最小基线）在pass@1、每任务回合数和每任务花费上进行比较。将结果写入`eval/results.jsonl`。

7. **成本控制。** 硬性截断：50回合、200k上下文、每任务5美元。`PreCompact`钩子将较早的回合总结成一个先前状态块，在150k标记处释放空间，以便在不丢失计划的情况下容纳新观察。

8. **PR发布。** 成功后，最后一步是`git push` + 一个GitHub API调用，打开一个PR，正文中包含计划和差异摘要。

## 使用它

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## 发布

可交付技能存在于`outputs/skill-terminal-coding-agent.md`中。给定一个仓库路径和任务描述，它在沙盒中运行完整的计划-行动-观察循环，并返回一个PR URL和一个追踪包。该顶点项目的评分标准：

|  权重  |  标准  |  衡量方式  |
|:-:|---|---|
|  25  |  SWE-bench Pro pass@1 vs 基线  |  你的框架 vs mini-swe-agent 在30个匹配的Python任务上  |
|  20  |  架构清晰度  |  计划/行动/观察分离、钩子表面、工具模式——与Live-SWE-agent布局对比评审  |
|  20  |  安全性  |  沙盒逃逸测试、权限提示、破坏性命令防护通过红队测试  |
|  20  |  可观测性  |  追踪完整性（100%工具调用有跨度）、每回合Token核算  |
|  15  |  开发者体验  |  冷启动 < 2秒、崩溃恢复恢复计划、Ctrl-C干净地取消中间工具  |
|  **100**  |   |   |

## 练习

1. 将底层模型从Claude Sonnet 4.7切换到在vLLM上服务的Qwen3-Coder-30B。比较pass@1和每任务花费。报告开源模型表现不佳的地方。

2. 添加一个`reviewer`子代理，在PR发布前读取差异，并可以请求修订循环。测量误报审查是否将SWE-bench通过率降低到单代理基线以下（提示：通常是）。

3. 压力测试沙盒：编写一个试图`curl`外部URL的任务，以及一个试图在工作树之外写入的任务。确认两者都被PreToolUse钩子阻止。记录尝试。

4. 使用较小的模型（Haiku 4.5）实现`PreCompact`摘要。测量在3倍压缩下丢失了多少计划保真度。

5. 将MCP StreamableHTTP传输替换为stdio。基准测试冷启动和每次调用延迟。为仅本地使用选择一个胜者。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
|  框架  |  "代理循环"  |  围绕模型分派工具、维护计划状态和执行预算的代码  |
|  Hook  |  "Agent事件监听器"  |  由Harness在八个生命周期事件之一上运行的用户编写的脚本  |
|  Worktree  |  "Git沙箱"  |  一个在独立路径上的链接Git检出；可随时丢弃，不影响主克隆  |
|  TodoWrite  |  "计划状态"  |  一个类型化的待办/进行中/已完成项列表，模型在每轮中重写  |
|  StreamableHTTP  |  "MCP传输"  |  2026年MCP修订版：具有双向流的长寿命HTTP连接；替代SSE  |
|  Token ceiling  |  "上下文预算"  |  每轮或每次会话的输入+输出令牌上限；触发压缩或终止  |
|  pass@1  |  "单次尝试通过率"  |  在第一次运行中无需重试或窥视测试集即可解决的SWE-bench任务比例  |

## 延伸阅读

- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code) — Anthropic的参考Harness
- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code) — Agent Tabs和Composer 2产品说明
- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code) — SWE-bench Harness比较的最小基线
- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code) — 使用Opus 4.5达到79.2%的SWE-bench已验证
- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code) — 开源Harness，112k星
- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code) — 此总结性项目所针对的评估
- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code) — StreamableHTTP，能力元数据
- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code) — 工具调用和令牌使用的跨度模式
