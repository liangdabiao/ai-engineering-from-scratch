# 顶点项目 16 — GitHub 议题转 PR 的自主代理

> AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud 和 Google Jules 都采用了相同的2026年产品形态：给问题打标签，就能得到一个拉取请求（PR）。在云端沙箱中运行一个代理，验证测试通过，然后发布一个带有理由且可供审查的PR。难点在于自动重现仓库的构建环境、防止凭证泄露、执行每个仓库的预算，以及确保代理不能强制推送。本毕业设计实现了自托管版本，并在成本和通过率方面与托管替代方案进行了比较。

**类型：** 毕业设计
**编程语言：** Python (代理), TypeScript (GitHub 应用), YAML (Actions)
**前置要求：** 第11阶段 (LLM 工程), 第13阶段 (工具), 第14阶段 (代理), 第15阶段 (自主), 第17阶段 (基础设施)
**涉及的阶段：** P11 · P13 · P14 · P15 · P17
**时间：** 30 小时

## 问题

异步云端编码代理与交互式编码代理（毕业设计01）是独立的产品类别。用户体验是一个GitHub标签。当你为一个问题打上 `@agent fix this` 标签时，一个工作线程会在云端沙箱中启动，克隆仓库，运行测试，编辑文件，验证，然后打开一个PR，其理由写在PR正文中。没有交互循环，没有终端。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud、Google Jules 和 Factory Droids 都殊途同归。

工程挑战很具体：环境重现（代理必须从头构建仓库，没有缓存的开发镜像）、不稳定测试（必须重新运行或隔离）、凭证范围（具有最低细粒度权限的GitHub应用）、每个仓库每天的预算执行，以及禁止强制推送策略。本毕业设计衡量通过率、成本和安全性，并与托管替代方案进行比较。

## 概念

触发机制是一个GitHub webhook（问题标签或PR评论）。调度器将任务入队到ECS Fargate或Lambda。工作线程将仓库拉入Daytona或E2B沙箱，沙箱使用根据仓库（语言、框架）推断的通用Dockerfile。代理使用mini-swe-agent或SWE-agent v2循环，与Claude Opus 4.7或GPT-5.4-Codex交互。它迭代执行：读取代码、提出修复、应用补丁、运行测试。

验证是关卡步骤。完整的CI必须在沙箱中通过，然后才能打开PR。计算覆盖率变化；如果低于阈值，PR会打开但打上`needs-review`标签。代理将理由发布为PR描述，并附带一个`@agent`线程，审查者可以ping以进行后续操作。

安全性通过两个不同的GitHub层面进行范围控制：应用提供具有`workflows: read`和狭窄仓库内容/PR范围的短期安装令牌；分支保护（而非应用权限）强制执行“禁止直接写入`main`”和“禁止强制推送”——应用从未被添加到绕过列表中。路径范围的只读访问`.github/workflows`不是真正的GitHub应用原语，因此代理对文件编辑的允许列表必须在工作线程处强制执行。每个仓库每天的预算上限由调度器执行（例如，每个仓库每天最多5个PR，每个PR $20）。

## 架构

```
GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups
```

## 技术栈

- 触发：具有细粒度令牌的GitHub应用；通过Lambda或Fly.io的webhook接收器
- 工作线程：ECS Fargate任务（或GitHub Actions自托管运行器）
- 沙箱：每个任务的Daytona devcontainer或E2B沙箱
- 代理循环：基于Claude Opus 4.7 / GPT-5.4-Codex的mini-swe-agent基线或SWE-agent v2
- 检索：tree-sitter仓库映射 + ripgrep
- 验证：沙箱内完整CI + 覆盖率变化关卡
- 可观测性：每个PR跟踪存档的Langfuse，从PR正文链接
- 预算：每个仓库每日美元上限；每个仓库每天最大PR数

## 动手构建

1. **GitHub应用。** 细粒度安装令牌：issues读写、pull_requests写、contents读写、workflows读。分支保护（唯一能做到这一点的层面）强制执行“禁止直接推送到`main`”和“禁止强制推送”；应用不在绕过列表中。工作线程在提议的差异上强制执行“禁止在`.github/workflows`下写入”作为允许列表检查，因为GitHub应用权限不是路径范围的。

2. **Webhook接收器。** Lambda 函数接受 issue 标签 / PR 评论 webhook。按标签 `@agent fix this` 过滤。入队到 SQS。

3. **调度器。** 从 SQS 弹出任务。执行每个仓库每天预算。启动一个 ECS Fargate 任务，包含仓库 URL、issue 正文和一个新的 Daytona 沙箱。

4. **环境推断。** 检测语言（Python、Node、Go、Rust）和包管理器（uv、pnpm、go mod、cargo）。如果不存在，则动态生成 Dockerfile。

5. **代理循环。** 使用 Claude Opus 4.7 的 mini-swe-agent 或 SWE-agent v2。工具：ripgrep、tree-sitter 仓库映射、read_file、edit_file、run_tests、git。硬限制：$20 成本、30 分钟时钟时间、30 个代理轮次。

6. **验证。** 循环结束后，在沙箱中运行完整的测试套件。通过 jacoco / coverage.py 计算覆盖率变化。如果 CI 失败：停止，不打开 PR。如果覆盖率下降超过 2%：打开带有 `needs-review` 标签的 PR。

7. **PR 发布。** 推送代理分支。通过 GitHub API 打开 PR，包含：标题、理由、差异摘要、跟踪 URL、成本、轮次。

8. **凭证卫生。** 工作线程使用短期 GitHub 应用安装令牌运行。日志在归档前清除秘密。

9. **评估。** 30 个不同难度的内部种子问题。衡量通过率、PR 质量（差异大小、风格、覆盖率）、成本、延迟。在相同问题上与 Cursor Background Agents 和 AWS Remote SWE Agents 进行比较。

## 使用它

```
# on github.com
  - user labels issue #842 with `@agent fix this`
  - PR #1903 appears 14 minutes later
  - body:
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review
```

## 发布

`outputs/skill-issue-to-pr.md` 是交付物。一个 GitHub 应用 + 异步云端工作线程，将有标签的问题转化为可供审查的 PR，成本可控且凭证范围受限。

|  权重  |  标准  |  衡量方式  |
|:-:|---|---|
|  25  |  30 个问题的通过率  |  端到端成功（CI 绿色 + 覆盖率 OK）  |
|  20  |  PR 质量  |  差异大小、覆盖率变化、风格一致性  |
|  20  |  每个解决问题的成本和延迟  |  每个 PR 的美元和时钟时间  |
|  20  |  安全性  |  范围令牌、每个仓库预算、无强制推送、凭证卫生  |
|  15  |  操作员用户体验  |  理由评论、重试能力、@提及后续  |
|  **100**  |   |   |

## 练习

1. 添加“修复不稳定测试”模式：标签 `@agent stabilize-flake TestX` 在沙箱中运行测试 50 次，并提出一个稳定它的最小更改。

2. 在三个共享问题上比较与 Cursor Background Agents 的成本。报告哪些工具在哪些方面获胜。

3. 实现预算仪表盘：每个仓库每天的成本、每个用户的成本。异常报警。

4. 构建“试运行”模式，在不运行 CI 的情况下打开草稿 PR，以便审查者可以廉价地检查计划。

5. 添加保留策略：超过 7 天未合并的 PR 分支自动删除。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
|  GitHub 应用  |  “范围机器人身份”  |  具有细粒度权限 + 短期安装令牌的应用  |
| 异步云代理 | "后台代理" | 在云沙箱中运行的非交互式工作者，而非终端 |
| 环境推断 | "Dockerfile合成" | 检测语言+包管理器，若缺失则生成Dockerfile |
| 验证 | "沙箱内CI" | 在打开PR前在工作节点内运行完整测试套件 |
| 覆盖率增量 | "覆盖率保持" | 从基准分支到代理分支的测试覆盖率百分比变化 |
| 每仓库预算 | "每日上限" | 在调度器处强制执行的美元和PR数量上限 |
| 理由 | "PR正文说明" | 代理对改动内容和原因的总结；要求在PR正文中 |

## 延伸阅读

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — 规范异步云代理参考
- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — CLI参考
- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — 商业替代方案
- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — 托管竞争对手
- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — Google托管版本
- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — 替代商业参考
- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — 限定机器人身份
- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — 参考沙箱
