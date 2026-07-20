# 顶点项目10 — 多智能体软件工程团队

> SWE-AF的工厂架构、MetaGPT的基于角色的提示(Prompting)、AutoGen 0.4的类型化参与者图(Typed Actor Graph)、Cognition的Devin以及Factory的Droids都收敛于相同的2026年形态：一个架构师(Architect)制定计划，N个程序员(Coder)在并行工作树(Parallel Worktree)中工作，一个审查员(Reviewer)把关，一个测试员(Tester)验证。并行工作树将挂钟时间(Wall-clock)转化为吞吐量(Throughput)。共享状态(Shared State)和交接协议(Handoff Protocol)成为失败面(Failure Surface)。顶点项目是构建团队，在SWE-bench Pro上进行评估，并报告哪些交接失败以及失败频率。

**类型：** 顶点项目
**语言：** Python / TypeScript（智能体），Shell（工作树脚本）
**先决条件：** 阶段11（LLM工程），阶段13（工具），阶段14（智能体），阶段15（自主），阶段16（多智能体），阶段17（基础设施）
**涉及的阶段：** P11 · P13 · P14 · P15 · P16 · P17
**时间：** 40小时

## 问题

单智能体编码框架(Single-agent coding harness)在大型任务上遭遇上限。不是因为单个智能体弱，而是因为一个200k令牌的上下文无法同时容纳架构计划、四个并行代码库切片、审查员评论和测试输出。多智能体工厂(Multi-agent factory)将问题拆分：架构师拥有计划，程序员在并行工作树中实现，审查员把关，测试员验证。SWE-AF的“工厂”架构、MetaGPT的角色、AutoGen的类型化参与者图——这三种描述都指向相同的形态。

失败面在于交接。架构师计划了程序员无法实现的内容。程序员产生冲突的差异(Conflict diff)。审查员批准了一个幻觉修复(Hallucinated fix)。测试员与正在编写的程序员竞争。你将构建其中一个团队，在50个SWE-bench Pro问题上运行，跟踪每一次交接，并发布事后总结(Post-mortem)。

## 概念

角色是类型化智能体。**架构师**（Claude Opus 4.7）阅读问题，编写计划，并将其分解为具有显式接口的子任务。**程序员**（Claude Sonnet 4.7，N个并行实例，每个在`git worktree` + Daytona沙箱中）独立实现子任务。**审查员**（GPT-5.4）读取合并后的差异，并批准或请求特定更改。**测试员**（Gemini 2.5 Pro）在隔离环境中运行测试套件，并报告通过/失败及工件(Artifact)。

通信通过共享任务板（文件后端或Redis）进行。每个角色消费其允许处理的任务。交接是A2A协议类型化消息。协调关注点：合并冲突解决（协调者角色或自动三方合并），共享状态同步（一旦程序员开始，计划即冻结；重新计划是独立事件），以及审查员把关（审查员不能批准自己所做的更改或自己提议的更改）。

令牌放大(Token amplification)是隐藏成本。每个角色边界增加摘要提示和交接上下文。一个40轮次的单智能体运行变成跨四个角色的160总轮次。评分标准特别权衡令牌效率与单智能体基线，因为问题不是“多智能体是否有效”，而是“每美元它是否胜出”。

## 架构

```
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## 技术栈

- 编排(Orchestration)：LangGraph带共享状态 + 每智能体子图
- 消息传递：A2A协议（Google 2025）用于类型化智能体间消息
- 模型：Opus 4.7（架构师），Sonnet 4.7（程序员），GPT-5.4（审查员），Gemini 2.5 Pro（测试员）
- 工作树隔离：每个程序员`git worktree add` + Daytona沙箱
- 合并协调者：自定义三方合并 + LLM介导的冲突解决
- 评估：SWE-bench Pro（50个问题），SWE-AF场景，HumanEval++用于单元测试
- 可观测性：Langfuse带有角色标记跨度，每智能体令牌记录
- 部署：K8s，每个角色作为独立Deployment + HPA

## 动手构建

1. **任务板。** 文件后端的JSONL，带有类型化消息：`plan_request`, `subtask`, `diff_ready`, `review_needed`, `test_needed`, `approved`, `rejected`, `replan_needed`。智能体订阅标签。

2. **架构师。** 读取GitHub问题，使用Opus 4.7运行计划模板，要求显式子任务接口（涉及的文件、公共函数、测试影响）。发出一个`plan_request`，附带子任务的DAG。

3. **程序员。** N个并行工作者，每个从板上认领一个子任务。每个生成一个新的`git worktree add`分支加上Daytona沙箱。实现子任务。发出`diff_ready`，附带补丁+测试增量。

4. **合并协调者。** 在所有程序员完成时，将N个分支三方合并到一个暂存分支。仅在存在文件级重叠时进行LLM介导的冲突解决。

5. **审查员。** GPT-5.4读取合并后的差异。不能批准自己编写的差异。发出`approved`（无操作）或`review_feedback`附带路由回相关程序员的特定更改请求。

6. **测试员。** Gemini 2.5 Pro在干净沙箱中运行测试套件。捕获工件。发出`test_passed`或`test_failed`附带堆栈跟踪。失败的测试循环回拥有失败子任务的程序员。

7. **交接记账。** 跨越角色边界的每条消息在Langfuse中获得跨度，包含有效负载大小和使用的模型。计算每子任务令牌放大（程序员令牌 + 审查员令牌 + 测试员令牌 + 架构师份额 / 程序员令牌）。

8. **评估。** 在50个SWE-bench Pro问题上运行。对比单智能体基线（一个Sonnet 4.7在单个工作树中）的pass@1和每解决问题美元成本。

9. **事后总结。** 对于每个失败的问题，识别失败的交接（计划太模糊、合并冲突、审查员误批准、测试员不稳定）。生成交接失败直方图。

## 使用它

```
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## 发布

`outputs/skill-multi-agent-team.md`是交付物。给定问题URL和并行度，团队生成一个合并就绪的PR，包含每角色令牌记录。

|  权重  |  标准  |  衡量方式  |
|:-:|---|---|
|  25  |  SWE-bench Pro pass@1  |  匹配50问题子集，pass@1  |
|  20  |  并行加速  |  挂钟时间 vs 单智能体基线  |
|  20  |  审查质量  |  注入错误探测上的误批准率  |
|  20  |  令牌效率  |  每解决问题总令牌 vs 单智能体  |
|  15  |  协调工程  |  合并冲突解决、交接失败直方图  |
|  **100**  |   |   |

## 练习

1. 在运行中途向差异注入一个明显错误（在主正文前额外加`return None`）。测量审查员的误批准率。调整审查员提示，直到误批准率低于5%。

2. 减少为两个程序员（架构师+程序员+审查员+测试员，程序员顺序运行两个子任务）。比较挂钟时间和通过率。

3. 用单写入者约束替换合并协调者（子任务接触不相交的文件集）。衡量架构师的计划负担。

4. 将审查员从GPT-5.4替换为Claude Opus 4.7。测量误批准率和令牌成本差异。

5. 添加第五个角色：文档员（Haiku 4.5）。审查后，它生成一个变更日志条目。衡量文档质量是否证明额外的令牌花费是合理的。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
| 并行工作区(Parallel worktree) | "隔离分支(Isolated branch)" | `git worktree add` 为每个程序员生成一个全新的工作区 |
| 任务板(Task board) | "共享消息总线(Shared message bus)" | 文件或Redis存储的类型化消息，代理订阅 |
| 交接(Handoff) | "角色边界(Role boundary)" | 任何从一个角色上下文跨越到另一个角色的消息 |
| 令牌放大(Token amplification) | "多代理开销(Multi-agent overhead)" | 所有角色的总令牌数 / 同一任务的单代理令牌数 |
| A2A协议(A2A protocol) | "代理间(Agent-to-agent)" | 谷歌2025年关于类型化代理间消息的规范 |
| 合并协调器(Merge coordinator) | "整合器(Integrator)" | 运行三方合并并调解冲突的组件 |
| 虚假审批(False approval) | "审阅者幻觉(Reviewer hallucination)" | 审阅者批准了已知有缺陷的差异 |

## 延伸阅读

- [SWE-AF factory architecture](https://github.com/Agent-Field/SWE-AF) — 参考2026年多代理工厂
- [SWE-AF factory architecture](https://github.com/Agent-Field/SWE-AF) — 基于角色的多代理框架
- [SWE-AF factory architecture](https://github.com/Agent-Field/SWE-AF) — 微软的类型化参与者框架
- [SWE-AF factory architecture](https://github.com/Agent-Field/SWE-AF) — 参考产品
- [SWE-AF factory architecture](https://github.com/Agent-Field/SWE-AF) — 替代参考产品
- [SWE-AF factory architecture](https://github.com/Agent-Field/SWE-AF) — 代理间消息传递规范
- [SWE-AF factory architecture](https://github.com/Agent-Field/SWE-AF) — 隔离基础
- [SWE-AF factory architecture](https://github.com/Agent-Field/SWE-AF) — 评估目标
