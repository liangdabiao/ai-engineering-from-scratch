# 将智能体指令作为可执行约束

> 写成散文的指令只是愿望。写成约束的指令才是测试。工作台将每条规则转化为智能体可在运行时检查、审阅者可在事后验证的对象。

**类型：** 构建
**语言：** Python (stdlib)
**前置条件：** 阶段 14 · 32 (最小工作台)
**时间：** ~50分钟

## 学习目标

- 将路由散文与操作规则分离。
- 将启动规则、禁止操作、完成定义、不确定性处理以及审批边界表示为机器可检查的约束。
- 实现一个规则检查器，根据规则集对运行进行评分。
- 使规则集易于差异比较，以便审阅者能看到改动。

## 问题

典型的`AGENTS.md`读起来像入职文档。它告诉智能体“要小心”、“彻底测试”、“如果不确定就问”。三天后，智能体在没有测试的情况下推送了更改，写入了禁止目录，并且从未询问，因为它根本不知道界限在哪里。

指令在可操作时强大，在鼓舞人心时软弱。解决方法是编写工作台能解释、审阅者能评分的规则。

## 核心概念

规则应放在`docs/agent-rules.md`中，远离简短根路由器。每条规则都有一个名称、一个类别和一个检查。

```mermaid
flowchart LR
  Router[AGENTS.md] --> Rules[docs/agent-rules.md]
  Rules --> Checker[rule_checker.py]
  Checker --> Report[rule_report.json]
  Report --> Reviewer[Reviewer]
```

### 覆盖大多数规则的五个类别

|  类别  |  规则回答的问题  |  示例  |
|----------|---------------------------|---------|
|  启动  |  开始工作前必须满足什么？  |  "状态文件存在且为最新"  |
|  禁止  |  什么绝不能发生？  |  "不要编辑`scripts/release.sh`"  |
|  完成定义  |  什么证明任务完成？  |  "pytest退出码为0且验收测试通过"  |
|  不确定性  |  智能体不确定时该做什么？  |  "创建一个问题笔记而不是猜测"  |
|  审批  |  什么需要人类审批？  |  "任何新依赖，任何生产环境写入"  |

不适合这五个类别之一的规则通常应该拆分成两条。强制拆分。

### 规则是机器可读的

每条规则都有一个短标识、一个类别、一行描述和一个`check`字段，该字段命名`rule_checker.py`中的一个函数。添加规则意味着添加一个检查；检查器随工作台一起增长。

### 规则是易于差异比较的

规则存放在单个Markdown文件中，每条规则占一个标题。重命名在差异中可见。新规则位于其类别的顶部。过时的规则被删除而不是注释掉，因为工作台是事实来源，而不是团队上个季度感受的聊天记录。

### 规则与框架护栏

框架护栏（OpenAI Agents SDK护栏，LangGraph中断）在运行时级别执行规则。本课中的规则集是人类可读、可审阅的契约，这些护栏实现了该契约。两者都需要：运行时在回合中捕获违规，规则集证明运行时正在做正确的事情。

### 渐进式呈现：一份地图，而非百科全书

`AGENTS.md`不断增长的原因是每次事故都会添加一条规则，而没有任何事故会移除一条。一年后，文件长达两千行，智能体只读取第一屏，注意力预算耗尽，仅根据它被告知的一小部分行动。庞大的指令文件失败的原因与四十页的入职文档相同：读者只需略读一次，就再也不会回去看重要的部分。

解决方法不是更短的文件，而是分层文件。根路由器保持足够短，每次会议都能阅读，只包含指针。深度存在于主题文件中，智能体仅在任务涉及这些主题时才加载它们。给智能体一份地图，而不是整本百科全书，让它自己走到需要的页面。

```
AGENTS.md                  # router, < 50 lines: what this repo is, where to look, the 5 hard rules
docs/
  agent-rules.md           # the full rule set (this lesson)
  architecture.md          # loaded when the task touches module boundaries
  testing.md               # loaded when the task writes or runs tests
  deploy.md                # loaded only for release work, gated behind an approval rule
feature_list.json          # the backlog (Phase 14 · 36)
```

|  层级  |  存放位置  |  何时读取  |  大小预算  |
|------|----------|-----------|-------------|
|  路由器  |  `AGENTS.md`  |  每次会议，始终  |  约50行以内  |
|  规则  |  `docs/agent-rules.md`  |  每次会议，启动时  |  每个类别一屏  |
|  主题文档  |  `docs/<topic>.md`  |  仅当任务涉及该主题时  |  按需深入  |

两个测试保持分层诚实。可达性测试：智能体最多从路由器跳两次就能到达任何规则，因此路由器必须通过路径链接每个主题文档，而不是用散文描述。新鲜性测试：路由器足够短，审阅者在每个PR上都会重新阅读它，这是唯一能阻止它悄悄长回它替代的百科全书的东西。一个无法解析的指针比缺少规则更严重的失败，因此路由器中的断链本身就是启动检查违规。

## 动手构建

`code/main.py`附带：

- `agent-rules.md`解析器，将规则加载到数据类中。
- `agent-rules.md`风格的检查器函数，每个`rule_checker.py`引用对应一个。
- 一个违反两条规则的演示智能体运行，以及捕获它们的检查通过。

运行它：

```
python3 code/main.py
```

输出：解析后的规则集、运行跟踪、每条规则的通过/失败状态，以及保存在脚本旁边的`rule_report.json`。

## 实际中的生产模式

三个模式区分了持续一个季度的规则集和一周内失效的规则集。

**写入时标注严重性。** 每条规则携带`severity`: `block`、`warn`或`info`。检查器会报告所有三种；运行时仅在`block`上拒绝。大多数团队早期夸大严重性，然后在截止日期压力下悄悄削弱它；写入时标注迫使校准前置。与验证门（阶段14·38）配对，该门将任何覆盖`block`规则的操作签署到`overrides.jsonl`审计日志中。

**规则过期作为强制函数。** 每条规则携带`expires_at`日期（默认为创建后90天）。当一条未过期的规则连续60天没有违规时，检查器会发出警告；下一个季度评审要么证明保留它，要么将其削弱为`info`，要么删除它。Cloudflare的生产AI代码审查数据（2026年4月，30天内针对5,169个仓库的131,246次审查运行）显示，具有明确过期时间的规则集每个仓库保持在30条以下；没有过期时间的规则集增长到80条以上，并且大多数从未触发。

**Markdown作为源，JSON作为缓存。** `agent-rules.md`是编写的文件；`agent-rules.lock.json`是检查器在热路径中读取的缓存。锁由预提交钩子重新生成。Markdown差异是可审查的；JSON解析不参与每个环节。与`package.json`/`package-lock.json`和`Cargo.toml`/`Cargo.lock`形状相同。

## 使用它

在生产中：

- Claude Code、Codex、Cursor在会话开始时读取规则，并在拒绝操作时引用它们。检查器在CI中重新运行它们以捕获静默漂移。
- OpenAI Agents SDK护栏将相同的检查注册为输入和输出护栏。Markdown是文档表面；SDK是运行表面。
- 当飞行中的节点违反规则时，LangGraph中断触发。中断处理程序读取规则，询问人类，然后恢复。

规则集在所有三个平台间可移植，因为它只是markdown加函数名。

## 发布

`outputs/skill-rule-set-builder.md`采访项目所有者，将现有的散文指令分类为五个类别，并输出一个版本化的`agent-rules.md`以及一个检查器存根。

## 练习

1. 如果你的产品确实需要，添加第六个类别。论证它为什么不会坍缩到五个类别之一。
2. 扩展检查器，使规则可以携带严重性（`block`、`warn`、`info`），并且报告相应地进行聚合。
3. 将检查器连接到CI：如果阻塞级别的规则在最新的代理运行中失败，则构建失败。
4. 为每条规则添加“过期”字段。在90天内没有检查失败后，该规则将接受审查。
5. 找一个真实的`block`并将其重写为五类规则。其中有多少行是可操作的？有多少是愿望性的？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  可操作规则 | "真实指令" | 工作台可以在运行时检查的规则  |
|  愿望性规则 | "小心" | 没有检查的规则；要么删除要么升级  |
|  完成定义 | "验收" | 任务完成的客观、基于文件的证明  |
|  阻塞严重性 | "硬规则" | 违规会中止运行；没有操作员无法静音  |
|  规则过期 | "过时规则清理" | N天内没有失败的规则将被淘汰  |

## 延伸阅读

- [OpenAI Agents SDK guardrails](https://platform.openai.com/docs/guides/agents-sdk/guardrails)
- [OpenAI Agents SDK guardrails](https://platform.openai.com/docs/guides/agents-sdk/guardrails)
- [OpenAI Agents SDK guardrails](https://platform.openai.com/docs/guides/agents-sdk/guardrails)
- [OpenAI Agents SDK guardrails](https://platform.openai.com/docs/guides/agents-sdk/guardrails) — 生产环境中的阻塞/警告/信息严重性
- [OpenAI Agents SDK guardrails](https://platform.openai.com/docs/guides/agents-sdk/guardrails) — 131k次审查运行，规则组成经验
- [OpenAI Agents SDK guardrails](https://platform.openai.com/docs/guides/agents-sdk/guardrails) — 规则与CI之间的深度防御
- [OpenAI Agents SDK guardrails](https://platform.openai.com/docs/guides/agents-sdk/guardrails) — Lean 4作为规则即检查的上限
- [OpenAI Agents SDK guardrails](https://platform.openai.com/docs/guides/agents-sdk/guardrails) — 合并门实现：范围、变异测试、违规预算
- 阶段14·32 — 此规则集放入的最简工作台
- 阶段14·38 — 消费规则报告的验证门
- 阶段14·39 — 对规则合规性评分的审查代理
