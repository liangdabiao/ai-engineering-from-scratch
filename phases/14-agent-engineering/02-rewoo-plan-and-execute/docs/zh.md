# ReWOO 与计划-执行(Plan-and-Execute)：分离式规划

> ReAct 将思考和行动交错在同一个流中。ReWOO 将它们分离：先制定一个总体计划，然后执行。在 HotpotQA 上减少 5 倍的 token 数量，准确率提升 4%，并且可以将规划器(Planner)蒸馏到 7B 模型中。计划-执行(Plan-and-Execute)对其进行了泛化；计划-行动(Plan-and-Act)将其扩展到网页导航。

**类型：** 构建
**语言：** Python (标准库)
**前置条件：** 阶段 14 · 01 (Agent 循环)
**时间：** 约 60 分钟

## 学习目标

- 解释为什么 ReWOO 的规划器(Planner)/工作器(Worker)/求解器(Solver)分离比 ReAct 的交错循环节省 token 并提高鲁棒性。
- 实现一个计划有向无环图(DAG)、一个依赖顺序执行器(Executor)和一个组合工作器输出的求解器——全部使用标准库。
- 使用 2026 年的"五种工作流模式"框架（Anthropic）判断任务应运行计划-执行模式还是交错 ReAct 模式。
- 识别何时需要 Plan-and-Act 的合成计划数据来处理长周期网页或移动任务。

## 问题

ReAct 的交错思考-行动-观察循环简单灵活，但每次工具调用都必须携带完整的先前上下文——包括每一个之前的思考。Token 使用量随深度呈二次方增长。更糟的是：当工具在循环中间失败时，模型必须从错误观察中重新推导整个计划。

ReWOO (Xu 等人, arXiv:2305.18323, 2023年5月) 注意到了这一点并下了一个赌注：预先规划整个事情，并行获取证据，最后组合答案。一次 LLM 调用用于规划，N 次工具调用用于证据（可以并行），一次 LLM 调用用于求解。这种权衡是牺牲了灵活性（计划是静态的）以换取更好的 token 效率和更清晰的故障模式。

## 核心概念

### 三种角色

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (tool calls, possibly parallel)
Solver:   user_question, plan_dag, evidence -> final_answer
```

规划器(Planner)生成一个有向无环图(DAG)。每个节点命名一个工具、其参数以及它所依赖的早期节点（如 `#E1`, `#E2` 等引用）。工作器(Worker)按照拓扑顺序执行节点。求解器(Solver)将所有内容拼接起来。

### 为什么 token 减少 5 倍

ReAct 的提示长度随步数线性增长。在第 10 步时，提示包含思考 1 加上行动 1 加上观察 1 加上思考 2 加上行动 2 加上观察 2，依此类推。每个中间步骤还冗余包含原始提示。

ReWOO 支付一次规划器提示（大），N 次小的工作器提示（每个只是工具调用，没有链），以及一次求解器提示。在 HotpotQA 上，论文测量到约 5 倍的 token 减少，同时准确率提升了 4 个绝对百分点。

### 为什么更鲁棒

如果工作器 3 在 ReAct 中失败，循环必须在中间流中从错误中推理。而在 ReWOO 中，工作器 3 返回一个错误字符串；求解器在上下文中看到它与原始计划，并且可以优雅降级。故障定位是按节点而非按步骤。

### 规划器蒸馏

论文的第二个结果是：因为规划器看不到观察结果，你可以用一个 175B 教师模型的规划器输出微调一个 7B 模型。小模型处理规划；推理时不需要大模型。这现在已成为标准——许多 2026 年的生产级智能体使用小规划器和大执行器，反之亦然。

### 计划-执行 (LangChain, 2023)

LangChain 团队 2023 年 8 月的帖子将 ReWOO 泛化为一个模式名称：计划-执行(Plan-and-Execute)。前置规划器发出步骤列表，执行器运行每个步骤，一个可选的重规划器可以在观察结果后进行修订。这比 ReWOO 更接近 ReAct（重规划器将观察结果带回规划中），但保留了 token 节省。

### 计划-行动 (Erdogan 等人, arXiv:2503.09572, ICML 2025)

计划-行动(Plan-and-Act)将该模式扩展到长周期网页和移动智能体。关键贡献是合成计划数据：一个带标签的轨迹生成器产生计划显式的训练数据。用于微调规划器模型，使其在 WebArena 类任务中保持超过 30-50 步的连贯性，而单一的 ReAct 轨迹会失去连贯性。

### 如何选择

|  模式  |  何时使用  |
|---------|------|
|  ReAct  |  短任务、未知环境、需要响应式异常处理  |
|  ReWOO  |  具有已知工具的结构化任务、对 token 敏感、可并行化证据  |
|  计划-执行  |  类似于 ReWOO 但可以在部分执行后重新规划  |
|  计划-行动  |  长周期（>30 步）、网页/移动/电脑使用  |
|  思维树  |  搜索值得付出代价（第 04 课）  |

Anthropic 2024 年 12 月的指导：从最简单的开始。如果任务是一个工具调用加一个总结，不要构建 ReWOO。如果任务是 40 步的研究任务，不要单独使用 ReAct。

## 动手构建

`code/main.py` 实现了一个玩具版 ReWOO：

- `Planner` — 一个脚本化策略，从提示中生成计划 DAG。
- `Planner` — 通过注册表分发每个节点的工具调用。
- `Planner` — 脚本化组合，读取证据并产生最终答案。
- 依赖解析 — 类似 `Planner` 的引用被替换为早期工作器的输出。

该演示使用两步计划回答"法国首府的人口是多少，四舍五入到百万？"：(1) 查找首府，(2) 查找人口，然后求解。

运行它：

```
python3 code/main.py
```

追踪首先显示完整计划，然后是工作器结果，最后是求解器组合。比较 token 数量（我们打印粗略字符数）与 ReAct 风格的交错运行——在这个结构化任务上 ReWOO 获胜。

## 使用它

LangGraph 将 Plan-and-Execute 作为配方提供（针对 ReAct 的 `create_react_agent`，以及用于 plan-execute 的自定义图）。CrewAI 的 Flows 直接编码该模式：你事先定义任务，Flow DAG 执行它们。Plan-and-Act 的合成数据方法仍主要处于研究阶段；运行时模式（显式计划 DAG）通过 LangGraph 和 CrewAI Flows 投入生产。

## 发布

`outputs/skill-rewoo-planner.md` 根据用户请求生成 ReWOO 计划 DAG，给出工具目录。它在将计划交给执行器之前验证计划（无环、所有引用已解析、所有工具存在）。

## 练习

1. 并行执行独立计划节点的工作节点。对于一个有 2 个并行组的 6 节点 DAG，它能给你带来什么？
2. 添加一个重规划器节点，如果任何工作节点返回错误则触发。对 ReWOO 进行最小的改动使其成为 Plan-and-Execute 是什么？
3. 将 `Planner` 替换为小模型（7B 类），并将 `Solver` 保留在前沿模型上。比较端到端质量——拆分在哪里失败？
4. 阅读 ReWOO 论文的第 4 节关于规划器蒸馏的内容。从概念上重现 175B -> 7B 的结果：你需要什么训练数据，以及如何评分计划质量？
5. 将玩具移植到 Plan-and-Act 的轨迹形状：计划是一个序列，而不是 DAG。哪些权衡发生了变化？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  ReWOO  |  "无观察推理"  |  先计划，然后并行获取证据，再求解——规划提示中无观察  |
|  Plan-and-Execute  |  "LangChain 的 plan-execute 模式"  |  带有可选重规划器节点的 ReWOO，在执行之后  |
|  Plan-and-Act  |  "规模化 plan-execute"  |  显式的规划器/执行器拆分，使用合成计划训练数据用于长周期任务  |
|  Evidence reference  |  "#E1, #E2, ..."  |  计划节点占位符，在调度时替换为先前工作节点的输出  |
|  Planner distillation  |  "小规划器，大执行器"  |  在大教师模型的规划器轨迹上微调小模型  |
|  Token efficiency  |  "更少的往返"  |  在 HotpotQA 上比 ReAct 减少 5 倍 token（根据论文）  |
|  DAG executor  |  "拓扑调度器"  |  按依赖顺序运行计划节点；每层并行  |

## 延伸阅读

- [Xu et al., ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) —— 经典论文
- [Xu et al., ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) —— 带合成计划的规模化规划器-执行器
- [Xu et al., ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) —— 框架配方
- [Xu et al., ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) —— 选择最简单的有效模式
