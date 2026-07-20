# 技能与Agent SDK — Anthropic Skills、AGENTS.md、OpenAI Apps SDK

> MCP 说“有什么工具”，技能说“如何完成任务”。2026年的技术栈将两者叠加。Anthropic 的 Agent Skills（开放标准，2025年12月）以 SKILL.md 形式发布，采用渐进式披露。OpenAI 的 Apps SDK 是 MCP 加上小部件元数据。AGENTS.md（现已存在于60,000多个仓库中）位于仓库根目录，作为项目级的 Agent 上下文。本课说明每层涵盖的内容，并构建一个可在多个 Agent 之间移植的最小 SKILL.md + AGENTS.md 组合。

**类型：** 学习
**语言：** Python（标准库，SKILL.md 解析器和加载器）
**前置条件：** 阶段13 · 07（MCP 服务器）
**时间：** 约45分钟

## 学习目标

- 区分三个层：AGENTS.md（项目上下文）、SKILL.md（可复用的操作知识）、MCP（工具）。
- 编写一个包含 YAML 前置元数据和渐进式披露的 SKILL.md。
- 以文件系统方式将技能文件加载到 Agent 运行时中。
- 将一个技能与 MCP 服务器和 AGENTS.md 组合，使单个包可在 Claude Code、Cursor 和 Codex 中使用。

## 问题

一位工程师将发布说明编写工作流程提炼成一个多步骤提示：“阅读最新的已合并 PR。按区域分组。总结每个区域。按照团队风格编写变更日志条目。发布到 Slack 草稿。”他们将其放在团队的 Notion 文档中。

现在他们想从 Claude Code、Cursor 和 Codex CLI 使用此工作流程。每个 Agent 加载指令的方式不同：Claude Code 斜杠命令、Cursor 规则、Codex `.codex.md`。该工程师将工作流复制了三次，并维护三个副本。

AGENTS.md 和 SKILL.md 共同解决了这个问题：

- **AGENTS.md** 位于仓库根目录。每个兼容的 Agent 在会话启动时读取它。“这个项目如何工作？有什么约定？哪些命令运行测试？”
- **SKILL.md** 是一个可移植的包：YAML 前置元数据（名称、描述）+ Markdown 正文 + 可选资源。支持技能的 Agent 可按需通过名称加载它们。
- **MCP**（阶段13 · 06-14）处理技能需要调用的工具。

三层，一个可移植的工件。

## 核心概念

### AGENTS.md (agents.md)

2025年底发布，截至2026年4月已被60,000多个仓库采用。一个文件位于仓库根目录。格式：

```markdown
# Project: my-service

## Conventions
- TypeScript with strict mode.
- Use Pydantic for models on the Python side.
- Tests run with `pnpm test`.

## Build and run
- `pnpm dev` for local dev server.
- `pnpm build` for production bundle.
```

Agent 在会话启动时读取此文件，并用于校准其在该项目中的行为。2026年每个编码 Agent 都支持 AGENTS.md：Claude Code、Cursor、Codex、Copilot Workspace、opencode、Windsurf、Zed。

### SKILL.md 格式

Anthropic 的 Agent Skills（于2025年12月作为开放标准发布）：

```markdown
---
name: release-notes-writer
description: Write a changelog entry for the latest merged PRs following this project's style.
---

# Release notes writer

When invoked, run these steps:

1. List PRs merged since the last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR in each group, write one line: `- <title> (#<num>)`.
4. Draft the release notes and stage them in CHANGELOG.md.

If the user says "ship", run `git tag vX.Y.Z` and `gh release create`.

## Notes

- Never include commits without a PR.
- Skip "chore" entries from the public changelog.
```

前置元数据声明技能的标识。正文是技能加载时向模型显示的提示。

### 渐进式披露

技能可以引用子资源，Agent 仅在需要时获取它们。示例：

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md 说“参见 style-guide.md 获取样式规则”。Agent 仅在技能实际运行时拉取 style-guide.md。这避免了用模型可能不需要的细节臃肿提示。

### 文件系统发现

Agent 运行时扫描已知目录以查找 SKILL.md 文件：

- `~/.anthropic/skills/*/SKILL.md`
- 项目 `~/.anthropic/skills/*/SKILL.md`
- `~/.anthropic/skills/*/SKILL.md`

通过文件夹名称和前置元数据 `name` 加载。Claude Code、Anthropic Claude Agent SDK 和 SkillKit（跨Agent）都遵循此模式。

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk` (TypeScript) 和 `claude-agent-sdk` (Python) 在会话启动时加载技能，并将其公开为运行时内可调用的“Agent”。当用户调用技能时，Agent 循环会调度到该技能。

### OpenAI Apps SDK

于2025年10月发布；直接构建在 MCP 之上。将 OpenAI 之前的 Connectors 和 Custom GPT Actions 统一到单个开发者界面下。Apps SDK 应用是：

- 一个 MCP 服务器（工具、资源、提示）。
- 加上用于 ChatGPT 界面的小部件元数据。
- 加上一个可选的 MCP Apps `ui://` 资源，用于交互式界面。

同一协议，更丰富的用户体验。

### 通过 SkillKit 实现跨Agent可移植性

像 SkillKit 及其类似的跨Agent分发层，将单个 SKILL.md 转换为32+个 AI Agent（Claude Code、Cursor、Codex、Gemini CLI、OpenCode 等）的原生格式。一个事实来源，众多消费者。

### 三层栈

|  层  |  文件  |  加载时机  |  目的  |
|-------|------|-------------|---------|
|  AGENTS.md  |  仓库根目录  |  会话启动  |  项目级约定  |
|  SKILL.md  |  技能目录  |  技能被调用  |  可复用工作流  |
|  MCP server  |  外部进程  |  所需工具  |  可调用动作  |

三者组合：代理在会话启动时读取AGENTS.md，用户调用技能，技能指令包含MCP工具调用，代理通过MCP客户端分发。

## 使用它

`code/main.py` 提供了一个标准库(Stdlib)的SKILL.md解析器和加载器。它在`./skills/`目录下发现技能，解析YAML前置元数据和Markdown正文，生成按技能名索引的字典。然后模拟代理循环，按名称调用`release-notes-writer`。

需要关注的内容：

- YAML前置元数据使用最小化的标准库解析器解析（无`pyyaml`依赖）。技能正文原样存储；代理在调用时将其附加到系统提示前。通过一个`pyyaml`函数演示渐进式披露，该函数按需拉取引用文件。
- 
- 

## 发布

本节课生成`outputs/skill-agent-bundle.md`。给定一个工作流，该技能生成组合的SKILL.md + AGENTS.md + MCP-server-blueprint包，可在代理间移植。

## 练习

1. 运行`code/main.py`。在`skills/`下添加第二个技能，确认加载器能识别它。

2. 为本课程仓库编写AGENTS.md。包含测试命令、样式约定和Phase 13心智模型。

3. 将团队内部文档中的多步工作流移植到SKILL.md中。验证它能在Claude Code中加载。

4. 手动将技能转换为Cursor和Codex的原生规则格式。计算格式之间的差异——这是SkillKit自动化的翻译面。

5. 阅读Anthropic Agent Skills博客文章。找出Claude Agent SDK中本节课加载器未覆盖的一项功能。（提示：代理子调用。）

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  SKILL.md  |  "技能文件"  |  YAML前置元数据加Markdown正文，由代理运行时加载  |
|  AGENTS.md  |  "仓库根代理上下文"  |  项目级约定文件，在会话启动时读取  |
|  Progressive disclosure  |  "延迟加载子资源"  |  技能正文引用的文件仅在需要时获取  |
|  Frontmatter  |  "顶部YAML块"  |  `---`分隔符中的元数据（名称、描述）  |
|  Claude Agent SDK  |  "Anthropic的技能运行时"  |  `@anthropic-ai/claude-agent-sdk`，加载技能并进行路由  |
|  OpenAI Apps SDK  |  "MCP + widget元数据"  |  OpenAI的开发界面，基于MCP并加上ChatGPT UI钩子  |
|  Skill discovery  |  "文件系统扫描"  |  遍历已知目录寻找SKILL.md，按名称索引  |
|  Cross-agent portability  |  "一个技能多个代理"  |  通过SkillKit风格的工具将一个SKILL.md转换为32个以上代理的格式  |
|  Agent Skill  |  "可移植知识"  |  MCP工具概念之外的可复用任务模板  |
|  Apps SDK  |  "MCP加ChatGPT UI"  |  连接器和自定义GPT在MCP上统一  |

## 延伸阅读

- [Anthropic — Agent Skills announcement](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — 2025年12月发布
- [Anthropic — Agent Skills announcement](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — SKILL.md格式参考
- [Anthropic — Agent Skills announcement](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — 基于MCP的ChatGPT开发者平台
- [Anthropic — Agent Skills announcement](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — AGENTS.md格式与采纳列表
- [Anthropic — Agent Skills announcement](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — 官方技能示例
