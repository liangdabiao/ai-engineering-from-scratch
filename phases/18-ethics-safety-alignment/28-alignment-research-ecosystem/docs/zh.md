# Alignment Research Ecosystem — MATS, Redwood, Apollo, METR

> 五个组织定义了2026年的非实验室对齐研究层。MATS（机器学习对齐与理论学者）：自2021年末以来已有527+名研究者，180+篇论文，10K+引用，h指数47；2024年夏季 cohort 注册为501(c)(3)组织，约有90名学者和40名导师；2025年前校友中80%从事安全/安保工作，其中200+人在Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo。Redwood Research：由Buck Shlegeris创立的应用对齐实验室；提出了AI控制（第10课）；与UK AISI合作开展控制安全案例研究。Apollo Research：为前沿实验室提供部署前的诡计评估；撰写了《上下文诡计》（第8课）和《迈向AI诡计的安全案例》。METR（模型评估与威胁研究）：基于任务的能力评估、自主任务时间跨度研究；《前沿AI安全政策的共同要素》比较了各实验室框架。Eleos AI Research：部署前的模型福利评估（第19课）；进行了Claude Opus 4福利评估。

**类型：** 学习
**语言：** 无
**前置条件：** 阶段18 · 01-27（阶段18之前的课程）
**时间：** ~45分钟

## 学习目标

- 识别非实验室对齐研究生态系统的五个组织及其核心输出。
- 描述MATS的规模（学者、论文、h指数）及其作为人才输送渠道的作用。
- 描述Redwood的AI控制议程及其与UK AISI的合作。
- 描述METR基于任务的评估方法论。

## 问题

前沿实验室（第18课）在内部进行安全评估并发布选定的结果。实验室之外的生态系统是评估得到验证、新的失败模式首次被发现以及人才得到训练的地方。理解生态系统有助于解释哪些研究发现被谁信任。

## 核心概念

### MATS（机器学习对齐与理论学者）

始于2021年底。研究指导项目；学者与资深研究员花10-12周时间解决特定对齐问题。

规模（2026年）：
- 自成立以来527+名研究者。
- 发表180+篇论文。
- 10K+次引用。
- h指数47。
- 2024年夏季：90名学者 + 40名导师；注册为501(c)(3)组织。

职业成果：2025年前校友中约80%从事安全/安保工作。200+人在Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo。

### Redwood Research

应用对齐实验室。由Buck Shlegeris创立。提出了AI控制议程（第10课）。与UK AISI合作开展控制安全案例研究。为DeepMind和Anthropic提供评估设计建议。

经典论文：Greenblatt, Shlegeris等，《AI控制》(arXiv:2312.06942, ICML 2024)；《对齐伪装》(Greenblatt, Denison, Wright等，arXiv:2412.14093，与Anthropic合作)。

风格：具体的威胁模型、最坏情况对手、可压力测试的具体协议。

### Apollo Research

为前沿实验室提供部署前的诡计评估。撰写了《上下文诡计》（第8课，arXiv:2412.04984）。参与了2025年OpenAI反诡计训练合作。撰写了《迈向AI诡计的安全案例》（2024年）。

风格：可能产生欺骗的智能体环境评估；三支柱分解（错位、目标导向性、情境感知）。

### METR（模型评估与威胁研究）

基于任务的能力评估。自主任务完成时间跨度研究。《前沿AI安全政策的共同要素》(metr.org/common-elements, 2025)比较了各实验室框架。

与Apollo共同撰写了AI诡计安全案例草图。

风格：长周期任务评估、经验能力测量、框架综合。

### Eleos AI Research

部署前的模型福利评估。对Claude Opus 4进行了福利评估，记录在系统卡的第5.3节。为第19课的福利相关主张提供外部方法论检查。

### 流程

MATS训练研究者。毕业生前往Anthropic、DeepMind、OpenAI（实验室安全团队）或Redwood、Apollo、METR、Eleos（外部评估）。外部评估者与实验室以及UK AISI / CAISI合作。出版物反馈到MATS生态系统，供下一个 cohort 使用。

### 为什么这一层很重要

单一来源的评估不可靠：实验室评估自己的模型存在结构性利益冲突。外部评估者可以提出并验证实验室可能少报的失败模式。2024年的《潜伏特工》论文（第7课）是Anthropic + Redwood；《对齐伪装》是Anthropic + Redwood；《上下文诡计》是Apollo；《反诡计》是Apollo + OpenAI。多组织结构是质量控制。

### 这在阶段18中的位置

第7-11课引用了Redwood和Apollo的工作；第18课引用了METR的框架比较；第19课引用了Eleos。第28课是生态系统其余阶段所依赖的明确组织图。

## 使用它

无代码。阅读METR的《前沿AI安全政策的共同要素》，作为外部综合如何为实验室内部政策工作增加价值的例子。

## 发布

本课程产生`outputs/skill-ecosystem-map.md`。给定一个对齐主张或评估，它可以识别组织、发表场所和方法论风格，并与已知的对等组织进行交叉核对。

## 练习

1. 从第7课到第15课中选一篇论文，识别其中涉及的组织。将作者与MATS校友及当前生态系统所属机构进行交叉核对。

2. 阅读METR的《前沿人工智能安全政策的共同要素》。识别他们强调的三个跨实验室趋同点以及两个最大的分歧点。

3. MATS的职业成果约80%是安全/安保。论证这种选择压力是适应性的（培养该领域）还是有偏见的（过滤掉异端立场）。

4. Redwood和Apollo都从事控制/谋划工作，但风格不同。选择一个失败模式，描述每家公司将如何调查它。

5. Eleos AI是唯一纯粹关注模型福利的组织。设计一个假设的第二个组织，专注于另一个福利相关的问题（认知自由、机器人具身等），并阐明其方法。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
|  MATS  |  "导师计划"  |  ML Alignment & Theory Scholars项目；自2021年以来已有527+名研究人员  |
|  Redwood Research  |  "控制实验室"  |  应用对齐；AI控制论文作者；英国AISI合作伙伴  |
|  Apollo Research  |  "谋划评估"  |  面向前沿实验室的部署前谋划评估  |
|  METR  |  "任务范围评估"  |  基于任务的能力评估；框架综合  |
|  Eleos AI  |  "福利实验室"  |  模型福利部署前评估  |
|  人才管道  |  "MATS -> 实验室"  |  MATS毕业生流向Anthropic、DeepMind、OpenAI、Redwood、Apollo、METR  |
|  外部评估  |  "非实验室检查"  |  不由模型生产者进行的评估；增加可信度  |

## 延伸阅读

- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/) — 导师计划
- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/) — AI Control论文
- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/) — 谋划评估
- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/) — 框架比较
- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/) — 模型福利方法论
