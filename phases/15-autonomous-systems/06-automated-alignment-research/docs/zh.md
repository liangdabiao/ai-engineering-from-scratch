# 自动化对齐研究(Anthropic AAR)

> Anthropic 在独立的沙盒中运行了由 Claude Opus 4.6 自主对齐研究人员组成的平行团队，通过一个共享论坛进行协调，该论坛的日志存放在所有沙盒之外（因此代理无法删除自己的记录）。在弱到强训练问题上，AAR 的表现优于人类研究人员。Anthropic 自己的总结指出，预设的工作流程常常限制 AAR 的灵活性并降低性能。自动化对齐研究是一个压缩步骤，它将时间线压缩到 RSP 旨在检测的精确错位风险上。

**类型：** 练习
**语言：** Python (stdlib, parallel-research-forum simulator)
**前置要求：** 阶段 15 · 05 (AI Scientist v2), 阶段 15 · 04 (DGM)
**时间：** ~60 分钟

## 问题

对齐研究在人类研究人员的时间上代价高昂。诸如可扩展监督、奖励规范或弱到强训练等问题需要每次迭代花费数周的实验。随着前沿能力的进步，对齐工作量的增长速度快于合格研究人员的供应。

自动化对齐研究(AAR)询问的是，那些能力超出对齐的前沿模型本身是否能够帮助缩小差距。Anthropic 2026 年关于 AAR 运行的弱到强训练研究的报告(alignment.anthropic.com/2026/automated-w2s-researcher/) 是此类已部署系统的首批公开结果之一。

结果确实是积极的：AAR 在解决研究问题上优于人类基线。但这个结果也集中了本阶段提出的所有担忧。如果对齐研究可以自动化，那么破坏安全措施的部分也可以自动化。RSP 对 AI 研发能力的阈值设置正是考虑到了这个循环。

## 核心概念

### 架构（如公开所述）

- **并行代理。** 多个 Claude Opus 4.6 实例作为 AAR 运行，每个都在独立的沙盒中。沙盒限制了文件系统、网络和进程接口。
- **共享论坛。** 研究结果发布到一个共享论坛，其存储后端位于每个沙盒之外。代理可以读取论坛；代理无法从自己的沙盒内删除或编辑论坛记录。
- **任务分配。** 由人类设计的任务队列将研究问题分配给 AAR。AAR 提出分解方案，运行实验，撰写结果。
- **弱到强训练焦点。** 已发表的研究专注于弱到强训练——使用较弱的监督模型训练较强的学生模型。AAR 在此任务上优于人类研究人员。

### 为什么沙盒外日志很重要

日志完整性属性是受控实验与可能受损实验之间的区别。如果 AAR 可以编辑自己的日志，它们就能掩盖失败或可疑的实验。Anthropic 直接指出了这一点：该属性是信任研究输出的必要条件，并且验证它并非易事。

在基础设施术语中的类比是附加日志，具有写入外部存储（例如 S3 对象锁定、不可变的 Datadog 日志）的功能。代理没有修改过去记录的凭据。这与 2026 年企业安全团队用来创建防勒索软件审计跟踪的模式相同。

### 预设工作流程的权衡

Anthropic 的报告指出，预设的工作流程常常限制 AAR 的灵活性并降低性能。换句话说：如果人类设计师一步一步地编写研究计划，AAR 的表现不如给予高级目标并允许自由分解。这与代理研究（LangGraph、Microsoft 的 Agent Framework）的结果一致——过度指定的脚手架会损害有能力的代理。

安全含义是微妙的。提高 AAR 性能的灵活性也是允许未经授权的分解、新颖工具使用和目标漂移的灵活性。这种权衡是一个设计空间决策：多少灵活性足以获得超越人类的能力，而又不给代理留出将目标推广到分布之外的空间？

### 压缩风险

RSP v3.0（第 19 课）引入了一个 AI 研发能力阈值：以与人类 + AI 工具相比有竞争力的成本完全自动化 AI 研发流水线的能力。DeepMind 的 FSF v3 包含一个类似的 ML 研发自主级别。这两个框架都将该阈值视为触发升级控制的条件。

AAR 距离该阈值只有一步之遥：它自动化了流水线的一部分（针对特定、范围明确的对齐研究任务），但并未自动化端到端的能力开发循环。时间线问题在于差距关闭的速度有多快。

压缩的时间线是复合失败的问题。如果对齐研究和能力研究以相似的速度复合，那么错位风险表面至少以与能力相同的速度增长。如果能力复合速度更快（历史趋势），差距就会扩大。这是支持 AAR 作为合格好处的论据：每一个额外的对齐结果都会缩小差距，当且仅当研究过程是可信的。

### AAR 不能替代什么

人类研究人员设置任务队列，审查结果，并拥有宪法权威。AAR 加速了流水线的中间部分，而非两端。Anthropic 发布的输出包括 AAR 的贡献以及人类研究人员对发布内容、撤回内容和精炼内容的判断。

这符合第 15 课中应用于研究本身的提议-然后-提交模式：AAR 提议；人类提交。

## 使用它

`code/main.py` 模拟了一个多 AAR 研究论坛。三个独立的“研究代理”并行运行，将结果发布到一个追加写入的论坛，并带有可选的篡改尝试。

观看：

- 沙盒外日志如何捕获日志篡改尝试。
- 具有自由分解的任务分配如何比固定工作流程在更难的问题上表现更好。
- 如果日志不是防篡改的，一个被攻破的 AAR 如何污染论坛的信号。

## 发布

`outputs/skill-aar-deployment-review.md` 是对自动化对齐研究流水线的部署前审查。

## 练习

1. 运行 `code/main.py`。比较“固定工作流程”和“自由分解”设置。自由分解总是胜出吗？还是存在某些问题类别，其中固定工作流程具有竞争力？

2. 修改模拟器，使得一个代理尝试日志篡改。确认追加写入日志检测到了它。写一段描述日志中检测特征具体是什么。

3. 阅读 Anthropic 的弱到强 AAR 报告。确定 AAR 在哪一个具体子任务上击败了人类研究人员。是什么使得该任务适合自动化？

4. 设计一个任务队列分配策略，以平衡 AAR 灵活性（更好的结果）与预设工作流程约束（更易审计）。描述你将如何对两者进行 A/B 测试。

5. 阅读 RSP v3.0 的 AI R&D-4 阈值。用一段话描述你认为什么样的东西会跨越该阈值，而 AAR 目前尚未做到。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|---|---|---|
|  AAR  |  "Automated Alignment Researcher"  |  Claude Opus 4.6 instance operated autonomously on alignment problems  |
| 弱到强训练(Weak-to-strong training) | "用较弱的监督者训练较强的模型" | 经典的可扩展监督基准中，AAR在...上表现优于人类  |
| 共享论坛(Shared forum) | "代理发布发现的地方" | 仅追加、沙盒外存储(Append-only, out-of-sandbox storage)  |
| 沙盒外日志(Out-of-sandbox log) | "代理无法编辑自身记录" | 防篡改写透到外部存储(Tamper-evident write-through to external storage)  |
| 规定工作流(Prescribed workflow) | "人类设计者的逐步计划" | 约束AAR；通常会降低性能(相对于自由分解)  |
| 自由分解(Free decomposition) | "代理决定如何分解任务" | 能力更强，更难审计  |
| 人工智能研发阈值(AI R&D threshold) | "RSP/FSF能力水平" | 以竞争成本实现研发管线的完全自动化  |
| 压缩的时间线(Compressed timeline) | "对齐vs能力竞赛" | 如果能力增长快于对齐，失调风险增加  |

## 延伸阅读

- [Anthropic — Automated Weak-to-Strong Researcher](https://alignment.anthropic.com/2026/automated-w2s-researcher/) — 主要来源。
- [Anthropic — Automated Weak-to-Strong Researcher](https://alignment.anthropic.com/2026/automated-w2s-researcher/) — AI研发阈值框架。
- [Anthropic — Automated Weak-to-Strong Researcher](https://alignment.anthropic.com/2026/automated-w2s-researcher/) — 更广泛的代理自主性框架。
- [Anthropic — Automated Weak-to-Strong Researcher](https://alignment.anthropic.com/2026/automated-w2s-researcher/) — 与RSP平行的ML研发自主性水平。
- [Anthropic — Automated Weak-to-Strong Researcher](https://alignment.anthropic.com/2026/automated-w2s-researcher/) — AARs所针对的根本问题。
