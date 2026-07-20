# Self-Refine与CRITIC：迭代输出改进

> Self-Refine (Madaan 等人, 2023) 使用一个LLM扮演三个角色——生成(Generate)、反馈(Feedback)、精炼(Refine)——循环进行。在7个任务上平均提升+20个绝对值。CRITIC (Gou 等人, 2023) 通过将验证路由到外部工具来强化反馈步骤。到2026年，这种模式将作为"评估器-优化器(Evaluator-Optimizer)"（Anthropic）或护栏循环(Guardrail Loop)（OpenAI Agents SDK）出现在每个框架中。

**类型：** 构建
**语言：** Python (标准库)
**先决条件：** 阶段14·01（Agent循环），阶段14·03（反思）
**时间：** 约60分钟

## 学习目标

- 陈述Self-Refine的三个提示（生成、反馈、精炼），并解释为什么历史对精炼提示很重要。
- 解释CRITIC的关键见解：LLM在没有外部基础的情况下进行自我验证是不可靠的。
- 实现一个带有历史和可选外部验证器的stdlib Self-Refine循环。
- 将这种模式映射到Anthropic的"评估器-优化器"工作流和OpenAI Agents SDK的输出护栏。

## 问题

智能体(Agent)产生的答案几乎正确。可能一行代码有语法错误。可能摘要太长。可能计划遗漏了一个边缘情况。你想要的是：智能体批判自己的输出，然后修复它。

Self-Refine证明了这一点可以通过单个模型实现，无需训练数据，无需强化学习。但有一个问题：LLM在硬事实的自我验证上表现不佳。CRITIC指出了解决方案——将验证步骤通过外部工具（搜索、代码解释器、计算器、测试运行器）进行路由。

这两篇论文共同定义了2026年迭代改进的默认模式：生成、验证（可能时外部验证）、精炼、当验证通过时停止。

## 核心概念

### Self-Refine (Madaan 等人, NeurIPS 2023)

一个LLM，三个角色：

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

关键细节：`refine` 看到完整的历史——所有先前的输出和批判——因此不会重复错误。论文通过消融实验证明：去掉历史，质量急剧下降。

标题：在包括GPT-4在内的7个任务（数学、代码、缩写、对话）上平均提升+20绝对值。无需训练，无需外部工具，单个模型。

### CRITIC (Gou 等人, arXiv:2305.11738, v4 Feb 2024)

Self-Refine的弱点：反馈步骤是LLM对自己进行评分。对于事实性陈述，这是不可靠的（产生的幻觉往往对模型本身来说看起来很有说服力）。CRITIC用`verify(task, output, tools)`替换`feedback(task, output)`，其中`tools`包括：

- 一个用于事实性陈述的搜索引擎。
- 一个用于代码正确性的代码解释器。
- 一个用于算术的计算器。
- 领域特定的验证器（单元测试、类型检查器、代码检查器）。

验证器产生基于工具结果的结构化批判。然后精炼器以此批判为条件。

标题：CRITIC在事实性任务上优于Self-Refine，因为批判是有基础的。在无外部验证器的任务（创意写作、格式化）上，CRITIC退化为Self-Refine。

### 停止条件

两种常见形式：

1. **验证通过。** 外部测试返回成功。可用时优先（单元测试、类型检查器、护栏断言）。
2. **未发出反馈。** 模型说"输出没问题。" 更便宜但不可靠；配合最大迭代次数上限使用。

2026年默认：组合它们。"如果验证通过或模型认为正常且迭代次数>=2或迭代次数>=最大迭代次数，则停止。"

### 评估器-优化器(Evaluator-Optimizer) (Anthropic, 2024)

Anthropic在2024年12月的文章中将此列为五种工作流模式之一。两个角色：

- 评估器(Evaluator)：对输出评分并产生批判。
- 优化器(Optimizer)：根据批判修改输出。

循环直到评估器通过。这是Anthropic框架中的Self-Refine/CRITIC。Anthropic添加的关键工程细节：评估器和优化器提示应该显著不同，这样模型就不会只是橡皮图章。

### OpenAI Agents SDK输出护栏(Output Guardrails)

OpenAI Agents SDK以"输出护栏"的形式提供这种模式。护栏是一个在智能体最终输出上运行的验证器。如果护栏触发（引发`OutputGuardrailTripwireTriggered`），则输出被拒绝，智能体可以重试。护栏可以调用工具（CRITIC风格）或作为纯函数（Self-Refine风格）。

### 2026年的陷阱

- **橡皮图章循环。** 同一模型以相同提示风格进行生成和批判会收敛到"看起来不错。" 使用结构不同的提示，或使用更小更便宜的模型进行批判。
- **过度精炼。** 每次精炼过程增加延迟和令牌。预算1-3次；之后升级到人工审查。
- **对琐碎任务使用CRITIC。** 如果没有外部验证器，CRITIC退化为Self-Refine；不要为存根验证器付出延迟代价。

## 动手构建

`code/main.py` 在一个玩具任务上实现Self-Refine和CRITIC：给定主题生成一个简短的要点列表。验证器检查格式（3个要点，每个不超过60个字符）。CRITIC添加一个外部"事实验证器"，惩罚已知的幻觉。

组件：

- `generate` — 脚本化生产者。
- `generate` — LLM式自我批判。
- `generate` — CRITIC式有基础验证器。
- `generate` — 根据历史重写输出。
- 停止条件 — 验证通过或最多4次迭代。

运行它：

```
python3 code/main.py
```

比较Self-Refine与CRITIC的运行。CRITIC捕获了Self-Refine漏掉的一个事实错误，因为外部验证器具有自我批判所没有的基础。

## 使用它

Anthropic的评估器-优化器（evaluator-optimizer）是这种模式在Claude友好语言中的体现。OpenAI Agents SDK的输出护栏（output guardrails）是CRITIC形状的（护栏可以调用工具）。LangGraph提供了一个类似于Self-Refine的反思节点（reflection node）。Google的Gemini 2.5 Computer Use添加了一个逐步骤安全评估器，它是CRITIC的变体：每个动作在提交前都经过验证。

## 发布

`outputs/skill-refine-loop.md` 根据任务形状、验证器可用性和迭代预算配置评估器-优化器循环。为生成器、评估器/验证器和优化器生成提示，以及一个停止策略。

## 练习

1. 使用max_iterations=1运行这个玩具程序。CRITIC仍然有用吗？把外部验证器替换为有噪声的验证器（随机30%误报）。循环会做什么？这是2026年大多数护栏栈的现实。实现一个“生成器-评论家在不同模型上”的变体：大模型生成，小模型评论。它能否超过同模型？阅读CRITIC第3节（arXiv:2305.11738 v4）。命名三个验证工具类别，并为每个类别给出一个示例。将OpenAI Agents SDK的`output_guardrails`映射到CRITIC的验证器角色。SDK哪里错了，哪里对了？
2. 
3. 
4. 
5. 

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  Self-Refine  |  "自我修复的大模型"  |  在单个模型中生成 -> 反馈 -> 改进循环，带有历史记录  |
|  CRITIC  |  "基于工具的验证"  |  用外部验证器（搜索、代码、计算、测试）替换反馈  |
|  评估器-优化器  |  "Anthropic工作流模式"  |  两个角色——评估器打分，优化器修订——循环直至收敛  |
|  输出护栏  |  "事后检查"  |  OpenAI Agents SDK验证器，在agent产生输出后运行  |
|  验证步骤  |  "评论阶段"  |  承载决策：基于工具还是自我评分  |
|  改进历史  |  "模型已尝试过的内容"  |  之前的输出+评论前置到改进提示中；删除后质量崩溃  |
|  橡皮图章循环  |  "自我一致性失败"  |  相同提示的评论返回"看起来不错"；通过结构不同的提示修复  |
|  停止条件  |  "收敛测试"  |  验证器通过 或 无反馈且达到迭代上限；从不单条件  |

## 延伸阅读

- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — 经典论文
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — 基于工具的验证
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — 评估器-优化器工作流模式
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — 作为CRITIC形状验证器的输出护栏
