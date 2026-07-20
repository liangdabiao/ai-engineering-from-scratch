# Reflexion: 言语强化学习

> 基于梯度的强化学习需要数千次试验和一个GPU集群来修复一个失败模式。Reflexion（Shinn等人，NeurIPS 2023）用自然语言完成：每次失败尝试后，智能体撰写一条反思，将其存入情景记忆，并在下一次试验中基于该记忆进行条件化。这是Letta的睡眠时间计算、Claude Code的CLAUDE.md学习记录和pro-workflow的learn-rule背后的模式。

**类型：** 构建
**语言：** Python (标准库)
**先修条件：** 第14阶段·01（智能体循环），第14阶段·02（ReWOO）
**时间：** 约60分钟

## 学习目标

- 命名Reflexion的三个组件（Actor, Evaluator, Self-Reflector）以及情景记忆的角色。
- 使用二元评估器、反思缓冲区和全新重试实现一个标准库Reflexion循环。
- 为给定任务在标量反馈、启发式反馈和自我评估反馈源之间做出选择。
- 解释为什么言语强化能够捕捉到基于梯度的强化学习需要数千次试验才能修复的错误。

## 问题

一个智能体未能完成任务。在标准强化学习中，你会再运行数千次试验，计算梯度，更新权重。昂贵、缓慢，而且大多数生产级智能体并没有为每次失败准备训练预算。

Reflexion（Shinn等人，arXiv:2303.11366）提出了一个不同的问题：如果智能体只是思考自己为何失败，然后带着这个想法再次尝试呢？没有权重更新。没有梯度。只是在试验之间存储自然语言。

结果：在ALFWorld上，它超越了ReAct和其他非微调基线。在HotpotQA上，它优于ReAct。在代码生成（HumanEval/MBPP）上，它当时达到了最先进水平。所有这一切都没有一个梯度步骤。

## 核心概念

### 三个组件

```
Actor         : generates a trajectory (ReAct-style loop)
Evaluator     : scores the trajectory — binary, heuristic, or self-eval
Self-Reflector: writes a natural-language reflection on the failure
```

外加一个数据结构：

```
Episodic memory: list of prior reflections, prepended to the next trial's prompt
```

一次试验中，Actor运行，Evaluator对其进行评分。如果分数低，Self-Reflector生成一条反思（“我选错了工具，因为我误以为问题在问X，而实际上它是在问Y”）。反思被存入情景记忆。下一次试验重新开始，但会看到这条反思。

### 三种评估器类型

1. **标量** — 外部二元信号。ALFWorld成功或失败。HumanEval测试通过或失败。最简单，信号最强。
2. **启发式** — 预定义的失败特征。“如果智能体连续两次产生相同动作，标记为卡住。”“如果轨迹超过50步，标记为低效。”
3. **自我评估** — LLM对自己的轨迹进行评分。当没有真实答案时使用。信号较弱；与基于工具的验证（第05课 — CRITIC）搭配效果良好。

2026年的默认做法是混合使用：有标量时用标量，没有时用自我评估，启发式作为安全护栏。

### 为什么这具有通用性

Reflexion与其说是一种新算法，不如说是一种命名模式。几乎所有生产级“自愈”智能体都运行某种变体：

- Letta的睡眠时间计算（第08课）：一个独立的智能体反思过去的对话并写入内存块。
- Claude Code的`CLAUDE.md` / “save memory”模式：将反思作为学习记录捕获，并前置到未来的会话中。
- pro-workflow的`CLAUDE.md`命令：将纠正作为显式规则捕获。
- LangGraph的反思节点：一个对输出进行评分并在需要时路由到精炼的节点。

所有这些都源于同一个洞察：自然语言是一种足够丰富的媒介，可以在运行之间携带“我从失败中学到了什么”。

### 何时有效，何时无效

Reflexion在以下情况下有效：

- 存在明确的失败信号（测试失败、工具错误、错误答案）。
- 任务类别是可复现的（同类型的问题可以再次提出）。
- 反思有改进轨迹的空间（足够的动作预算）。

Reflexion在以下情况下没有帮助：

- 智能体已经第一次尝试就成功了。
- 失败是外部原因（网络中断、工具故障）——对“网络中断”的反思无助于未来的运行。
- 反思变成了迷信——存储关于一次偶发不稳定运行的叙述。

2026年的陷阱：记忆腐烂。反思不断积累；有些已过时或错误；随着情景缓冲区的增长，重新运行变慢。缓解措施：定期压缩（第06课），为反思设置TTL，或使用独立的睡眠时间清理智能体（Letta）。

```figure
react-trace
```

## 动手构建

`code/main.py` 在一个玩具谜题上实现了Reflexion：生成一个总和为目标值的3元素列表。Actor生成候选列表；Evaluator检查总和；Self-Reflector写一行关于出错原因的分析。反思被存入情景记忆以供下一次试验使用。

组件：

- `Actor` — 一种在看到反思时改进的脚本化策略。
- `Actor` — 对目标总和进行通过/失败判断。
- `Actor` — 生成一行失败诊断。
- `Actor` — 一个具有TTL语义的有界列表。

运行它：

```
python3 code/main.py
```

跟踪显示三次试验。试验1失败，存入一条反思；试验2看到反思后有所改进但仍失败；试验3成功。与基线运行（无反思）相比——基线一直卡在试验1的答案上。

## 使用它

LangGraph将反思作为节点模式提供。Claude Code的`/memory`命令和pro-workflow的`/learn-rule`将情景缓冲区外部化为一个markdown文件。Letta的睡眠时间计算在空闲时间运行Self-Reflector，以便主智能体保持延迟受限。OpenAI Agents SDK没有直接提供Reflexion；你需要通过自定义的Guardrail（根据分数拒绝轨迹）和跨运行持久化的内存`Session`来构建它。

## 发布

`outputs/skill-reflexion-buffer.md` 创建并维护一个包含反思捕获、TTL和去重功能的情景缓冲区。给定一个任务类别和一次失败，它会输出一条对下一次试验真正有帮助的反思（而不是泛泛的“更加小心”）。

## 练习

1. 从二元评估器切换到返回距离度量（距离目标有多远）的标量评估器。收敛速度会更快吗？
2. 为反思添加10次试验的TTL。此后，旧的反思是有害还是有益？
3. 实现启发式评估器：如果相同动作重复出现，则标记试验为卡住。这与Self-Reflector如何交互？
4. 使用一个忽略反思的对抗性Actor运行Reflexion。最小需要怎样的反思提示工程才能迫使Actor注意到它们？
5. 阅读AlfWorld上Reflexion论文的第4节。概念上复现130%的成功率提升：与普通ReAct相比，关键差异是什么？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  Reflexion  |  "自我纠正(Self-correction)"  |  Shinn et al. 2023 — 行动者(Actor)、评估者(Evaluator)、自我反思器(Self-Reflector)加情景记忆(Episodic Memory)  |
|  言语强化(Verbal Reinforcement)  |  "无梯度学习(Learning without gradients)"  |  自然语言反思被附加到下一轮试验的提示之前  |
|  情景记忆(Episodic Memory)  |  "每任务反思(Per-task reflections)"  |  针对一个任务类别的先前反思的有界缓冲区  |
|  标量评估者(Scalar Evaluator)  |  "二元成功信号(Binary success signal)"  |  来自真实数据(ground truth)的通过/失败或数值评分  |
|  启发式评估者(Heuristic Evaluator)  |  "基于模式的检测器(Pattern-based detector)"  |  预定义的失败特征（例如，陷入循环、步骤过多）  |
|  自我评估者(Self-evaluator)  |  "LLM作为自我的追踪裁判(LLM-as-judge on own trace)"  |  在没有真实数据时的低信号后备方案——与基于工具的验证配对使用  |
|  记忆腐烂(Memory Rot)  |  "过期反思(Stale reflections)"  |  情景缓冲区充满过时条目；通过压缩/生存时间(TTL)修复  |
|  睡眠时间反思(Sleep-time Reflection)  |  "异步自我反思(Async self-reflection)"  |  在非关键路径上运行自我反思器，使主智能体保持快速  |

## 延伸阅读

- [Shinn et al., Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) — 经典论文(canonical paper)
- [Shinn et al., Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) — 生产中的异步反思(async reflection in production)
- [Shinn et al., Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) — 将情景缓冲区作为上下文的一部分进行管理(managing the episodic buffer as part of context)
- [Shinn et al., Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) — 反思节点模式(reflection node pattern)
