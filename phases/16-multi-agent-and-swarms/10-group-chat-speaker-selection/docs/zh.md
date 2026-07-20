# 群聊与发言者选择

> AutoGen GroupChat 和 AG2 GroupChat 在 N 个智能体之间共享一个会话；一个选择器函数（LLM、轮询或自定义）选择下一个发言者。这是涌现式多智能体对话的原型——智能体不知道自己在静态图中的角色，它们只是对共享池做出反应。AutoGen v0.2 的 GroupChat 语义在 AG2 分支中得以保留；AutoGen v0.4 将其重写为事件驱动的参与者模型。微软于 2026 年 2 月将 AutoGen 置于维护模式，并将其与 Semantic Kernel 合并为 Microsoft Agent Framework（2026 年 2 月 RC）。GroupChat 原语在 AG2 和 Microsoft Agent Framework 中均得以保留——学习一次，随处使用。

**类型：** 学习+构建
**语言：** Python（标准库）
**先决条件：** 阶段16·04（原始模型）
**时间：** 约60分钟

## 问题

当工作流已知时，静态图（LangGraph）表现出色。但真实对话并非静态：有时程序员询问审查者，有时询问研究员，有时询问写作者。硬编码每一种可能的交接会导致边爆炸。你需要的是*智能体对共享池做出反应*，并由某个函数决定下一个发言者。

这正是 AutoGen GroupChat 所做的。

## 概念

### 形状

```
              ┌─── shared pool ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │ (everyone reads all)
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  "next speaker = C"
```

每个智能体看到每条消息。每一轮调用一个选择器函数来选择下一个发言者。

### 三种选择器类型

**轮询。** 固定循环。确定性。线性扩展至 N，但忽略上下文——即使主题是法律审查，程序员也会获得发言机会。

**LLM 选择。** 调用 LLM 读取最近的消息池并返回最佳下一个发言者。上下文感知但速度慢：每一轮增加一次 LLM 调用。AutoGen 默认。

**自定义。** 一个 Python 函数，包含任意逻辑。典型做法：LLM 选择并带有回退规则（例如，“总是在程序员之后给验证者发言机会”）。

### ConversableAgent API

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` 持有选择器。当一个智能体完成一轮发言时，管理器调用选择器，选择器返回下一个智能体。循环继续，直到满足终止条件。

### 终止条件

三种常见模式：

- **最大轮数。** 总发言轮数的硬性上限。
- **"TERMINATE" 令牌。** 智能体可以发出一个哨兵消息；当该消息出现时，管理器停止。
- **目标达成检查。** 一个轻量级验证器每轮运行，在任务完成时停止对话。

### AutoGen → AG2 分裂与 Microsoft Agent Framework 合并

2025 年初，微软开始围绕事件驱动的参与者模型对 AutoGen 进行重大重写（v0.4）。社区将 AutoGen v0.2 的 GroupChat 语义分支为 AG2，保留了早期采用者已集成的 API。

2026 年 2 月，微软宣布 AutoGen 将进入维护模式，事件驱动的参与者模型将合并到 **Microsoft Agent Framework** 中（2026 年 2 月 RC，现已与 Semantic Kernel 合并）。GroupChat 概念在两条路径上均得以保留；实现细节有所不同。AG2 是 v0.2 兼容代码的首选上游。

### GroupChat 的适用场景

- **涌现式对话。** 你不希望预先连接每一种可能的下一个发言者。
- **角色混合任务。** 程序员询问研究员，研究员询问档案管理员，档案管理员再询问程序员。流程不是有向无环图。
- **探索性问题求解。** 可以理解为“头脑风暴会议”，而非“流水线”。

### 不适用的场景

- **严格确定性。** LLM 选择器可能不一致。相同提示，不同运行，不同下一个发言者。
- **谄媚级联。** 智能体倾向于赞同发言最自信的一方。需显式地进行反向提示。
- **上下文膨胀。** 每个智能体读取每条消息；10 轮后上下文变得巨大。使用投影（第 15 课）来限定视野。
- **热门发言者。** 某个智能体主导对话，因为选择器偏爱其专长。引入发言平衡作为选择器的一个特性。

### 群聊 vs 监督者

相同原语，不同默认设置：

- 监督者：一个智能体规划，其他智能体执行。选择器为“询问规划者做什么”。
- 群聊：所有智能体对等；选择器是一个作用于共享池的函数。

两者都使用第 04 课的四个原语。群聊默认使用 LLM 选择的编排方式和全池共享状态。

## 动手构建

`code/main.py` 使用标准库从头实现了一个 GroupChat。三个智能体（程序员、审查者、管理器），轮询和 LLM 选择两种变体，以及基于 `TERMINATE` 令牌的终止。

演示打印对话记录以及两种变体的选择器决策轨迹。

运行：

```
python3 code/main.py
```

## 使用它

`outputs/skill-groupchat-selector.md` 为给定任务配置一个 GroupChat 选择器——轮询 vs LLM 选择 vs 自定义，以及选择器输入（最近消息、智能体专长、发言次数）的使用。

## 发布

检查清单：

- **最大轮数上限。** 始终设置。典型任务为 10-20 轮。
- **发言平衡指标。** 跟踪每个智能体的发言次数；当不平衡超过阈值时发出警报。
- **终止令牌。** `TERMINATE` 或一个专用的验证器智能体。
- **投影或限定记忆。** 大约 10 条消息后，考虑只给每个智能体一个限定视野以防止上下文膨胀。
- **选择器日志记录。** 对于 LLM 选择的变体，记录选择器的输入和选择。否则调试不可能。

## 练习

1. 运行 `code/main.py`。比较轮询和 LLM 选择下的对话。每种方式下哪个智能体占主导？
2. 在选择器中添加一个“每个智能体最大发言次数”规则。它如何影响对话记录？
3. 实现一个目标达成终止：当审查者返回“批准”时停止。在达到轮数上限之前它多久触发一次？
4. 阅读 AutoGen 稳定版文档中关于 GroupChat 的内容（`code/main.py`）。识别 https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html 使用的默认选择器。
5. 阅读 AG2 仓库（`code/main.py`），比较其 v0.2 GroupChat 与 v0.4 事件驱动版本。v0.4 添加了什么具体特性（吞吐量、容错性、可组合性）？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  GroupChat  |  "同一聊天室中的智能体"  |  共享消息池 + 选择器函数。AutoGen / AG2 原语。  |
|  Speaker selection  |  "谁下一个发言"  |  选择下一个智能体的函数。轮询、LLM选择或自定义。  |
|  GroupChatManager  |  "会议主持人"  |  拥有选择器并循环轮次的 AutoGen 组件。  |
|  ConversableAgent  |  "基础智能体"  |  AutoGen 基类；一个可以发送和接收消息的智能体。  |
|  Termination token  |  "停止词"  |  结束聊天的哨兵字符串（通常是 `TERMINATE`）。  |
|  Hot speaker  |  "一个智能体主导"  |  选择器不断选择同一个智能体的失败模式。  |
|  Context bloat  |  "池无限增长"  |  每个智能体读取所有历史消息；上下文随轮次增长。  |
|  Projection  |  "限定视图"  |  针对角色的共享池视图，以防止上下文膨胀。  |

## 延伸阅读

- [AutoGen group chat docs](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — 参考实现
- [AutoGen group chat docs](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — 社区 AutoGen v0.2 延续
- [AutoGen group chat docs](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — 合并后的继任者，2026 年 2 月 RC
- [AutoGen group chat docs](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — 事件驱动 actor 模型重写细节
