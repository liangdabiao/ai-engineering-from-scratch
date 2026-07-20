# 自主编码智能体格局（2026）

> SWE-bench Verified 在不到三年内从 4% 提升至 80.9%。同样的 Claude Sonnet 4.5 在 SWE-agent v1 上得分为 43.2%，而在 Cline 自主模式下为 59.8% —— 模型周围的脚手架（Scaffolding）如今与模型本身同等重要。OpenHands（原名 OpenDevin）是 MIT 许可下最活跃的平台，其 CodeAct 循环直接在沙箱中执行 Python 动作，而非 JSON 工具调用。这些头条数字掩盖了一个方法论问题：500 个 SWE-bench Verified 任务中有 161 个仅需 1–2 行修改，而 SWE-bench Pro（10 行以上任务）同类前沿模型的得分仅为 23–59%。

**类型：** 学习
**语言：** Python（标准库，CodeAct 与 JSON 工具调用对比）
**先修知识：** 第 14 阶段 · 07（工具使用），第 15 阶段 · 01（长周期智能体）
**时间：** ~45 分钟

## 问题

“哪个编码智能体最好”是个错误的问题。正确的问题是：在与我的工作相匹配的任务分布上，使用我将投入生产的脚手架，我能获得怎样的端到端可靠性？

从 2022 年到 2026 年，该领域认识到脚手架——检索层、规划器、沙箱、编辑-验证循环、反馈格式——是承重结构。Claude Sonnet 4.5 在 SWE-agent v1 上得分为 43.2%；同一模型在 Cline 自主脚手架内得分为 59.8%。相差 16.6 个绝对百分点，权重相同。基础模型是组件，循环才是产品。

伴随的问题是基准测试饱和掩盖了性能倒退。SWE-bench Verified 已接近饱和，容易任务的尾部（500 个任务中有 161 个需要 ≤2 行修改）拉高了顶尖分数。实际质量更适合在 SWE-bench Pro（10 行以上修改）等分布上衡量，而同类领先系统在此仍只有 23–59%。

## 核心概念

### SWE-bench，一段简介

SWE-bench（Jimenez 等人）选取带有真实补丁的 GitHub 问题，要求智能体生成一个补丁使测试套件通过。SWE-bench Verified（OpenAI，2024）是人工精选的 500 任务子集，剔除了模糊和有缺陷的任务。SWE-bench Pro 是更难的后续版本——任务需要 10 行以上修改，当前前沿智能体得分为 23–59%。

### 2022 → 2026 曲线实际展示的内容

- **2022**：研究模型在原始 SWE-bench 上约 4%。
- **2024**：GPT-4 + Devin 式脚手架约 14%；SWE-agent 约 12%。
- **2025**：Claude 3.5/3.7 Sonnet 结合 Aider 和 SWE-agent 进入 40–55% 范围。
- **2026**：Claude Sonnet 4.5 及前沿竞品在 SWE-bench Verified 上达到 70–80%+。Epoch AI 的排行榜实时追踪这一进展。

增长曲线来自三个复合来源：更好的基础模型、更好的脚手架（CodeAct、反思、验证器循环）、以及更好的基准测试（Verified 消除了噪声）。

### CodeAct 与 JSON 工具调用对比

OpenHands（All-Hands-AI，arXiv:2407.16741，原名 OpenDevin）采取了一种特定的架构决策：模型不发出由主机解码并执行的 JSON 工具调用，而是发出 Python 代码，并由类似 Jupyter 的内核在沙箱中执行。智能体可以遍历文件、串联工具、并在单个动作内捕获自身异常。

权衡：

- **JSON 工具调用**：每个动作是一个回合；易于审计；组合性有限；默认安全，因为每次调用都经过显式验证器。
- **CodeAct**：一个动作可以是整个程序；组合性强；需要加固的沙箱（OpenHands 使用 Docker 隔离）；故障模式包括沙箱运行时允许的任何情况。

两种架构均已投入生产。CodeAct 在开放平台（OpenHands、smolagents）中占主导地位。JSON 工具调用在托管服务（Anthropic Managed Agents、OpenAI Assistants）中仍占主导，因为提供者控制执行器。

### 2026 年格局中的脚手架

|  脚手架  |  许可证  |  执行模型  |  显著特性  |
|---|---|---|---|
|  OpenHands (OpenDevin)  |  MIT  |  CodeAct 在 Docker 中  |  最活跃的开放平台；事件流可重放  |
|  SWE-agent  |  MIT  |  智能体-计算机接口 (ACI)  |  首个端到端 SWE-bench 脚手架  |
|  Aider  |  Apache-2  |  通过差异编辑本地仓库  |  最小化脚手架，回归稳定性强  |
|  Cline  |  Apache-2  |  带工具策略的 VS Code 智能体  |  Sonnet 4.5 上得分最高的开放脚手架  |
|  Devin (Cognition)  |  专有  |  托管虚拟机 + 规划器  |  首个“AI 软件工程师”产品类别  |
|  Claude Code  |  专有  |  权限模式 + 例程  |  第 10 课详细介绍了智能体循环  |

### 为什么脚手架占据主导地位

编码运行是一个长周期轨迹（第 1 课）。可靠性在步骤间累积。脚手架在三个方面带来分数提升：

1. **检索**：找到正确的文件读取是隐藏的瓶颈。SWE-agent 的 ACI、OpenHands 的文件索引、Aider 的仓库地图都在解决这个问题。
2. **验证器循环**：运行测试、读取堆栈跟踪、重试，在 SWE-bench 上带来 10 多个百分点的提升。
3. **故障隔离**：遇到错误时回滚的沙箱可防止复合损害。同一模型有无验证器循环看起来就像两种不同产品。

### 基准测试饱和与实际分布

OpenHands 作者和 Epoch AI 都指出 SWE-bench Verified 存在容易的尾部：500 个任务中有 161 个仅需 1–2 行修改。高分部分受此尾部驱动。SWE-bench Pro 限制为 10 行以上修改，即使前沿系统得分也仅在 23–59% 范围内。你的生产分布几乎肯定更接近 Pro 而非 Verified。

选择智能体的启示：从自己的 bug 积压中运行一个类似 Pro 的子集。真正重要的分数是与你所交付任务具有代表性的任务上的得分。

## 使用它

`code/main.py` 在两个固定的小型任务分布上比较了两个玩具智能体框架：

1. 一个每轮执行一个动作的 **JSON 工具调用** 框架。
2. 一个每动作可发出一个小型 Python 片段的 **CodeAct** 框架。

两者都使用一个存根“模型”（确定性规则），因此比较将框架与模型质量隔离开来。结果显示，CodeAct 框架以更大的每动作影响范围为代价，在更少的轮数内解决了更多任务。

## 发布

`outputs/skill-scaffold-audit.md` 帮助你在采用前审核提议的编码智能体框架：检索质量、验证器存在性、沙箱隔离以及基准到分布的拟合度。

## 练习

1. 运行 `code/main.py`。每个框架在同一任务集上需要多少轮？每个框架的每动作影响范围是多少？

2. 阅读 OpenHands 论文 (arXiv:2407.16741)。该论文认为 CodeAct 在复杂任务上胜过 JSON 工具调用。识别论文承认的一个失败模式，并写一句话说明该模式何时会在生产中占主导地位。

3. 从你的 bug 积压中挑选一个需要跨两个文件更改 10 行以上的任务。估计前沿模型在 (a) JSON 工具调用和 (b) CodeAct 下的端到端成功概率。论证差距。

4. SWE-bench Verified 有 161 个单文件、1-2 行任务。构建一个排除这些任务的分数。排行榜如何重新排列？

5. 阅读“Introducing SWE-bench Verified”（OpenAI）。解释用于移除模糊任务的具体方法，并指出筛选可能遗漏的一个类别。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|---|---|---|
|  SWE-bench  |  “编码基准”  |  带有真实补丁和测试套件的实际 GitHub 问题  |
|  SWE-bench Verified  |  “清理后的子集”  |  500 个人工筛选的任务，存在更简单的尾部  |
|  SWE-bench Pro  |  “更难子集”  |  10 行以上改动；前沿模型处于 23-59%  |
|  CodeAct  |  “代码即动作”  |  智能体发出 Python；Jupyter 风格内核在沙箱中执行  |
|  JSON 工具调用  |  “函数调用”  |  每个动作都是执行前验证的结构化 JSON 载荷  |
|  框架  |  “智能体框架”  |  围绕基础模型的检索 + 规划器 + 执行器 + 验证器循环  |
|  ACI（智能体-计算机接口）  |  “SWE-agent 的格式”  |  为 LLM 人机工程学设计的命令集，非人类 shell  |
|  验证器循环  |  “测试并重试”  |  运行测试、读取输出、修改补丁；最大的非模型可靠性提升  |

## 延伸阅读

- [Jimenez et al. — SWE-bench](https://www.swebench.com/) — 原始基准和方法论。
- [Jimenez et al. — SWE-bench](https://www.swebench.com/) — 如何构建精选子集。
- [Jimenez et al. — SWE-bench](https://www.swebench.com/) — CodeAct 架构和事件流设计。
- [Jimenez et al. — SWE-bench](https://www.swebench.com/) — 实时追踪的分数。
- [Jimenez et al. — SWE-bench](https://www.swebench.com/) — 长周期编码智能体可靠性框架。
