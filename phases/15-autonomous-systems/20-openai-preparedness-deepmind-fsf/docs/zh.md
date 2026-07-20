# OpenAI Preparedness Framework 和 DeepMind Frontier Safety Framework

> OpenAI Preparedness Framework v2（2025年4月）引入了研究类别(Research Categories) —— 长程自主性(Long-range Autonomy)、沙袋效应(Sandbagging)、自主复制与适应(Autonomous Replication and Adaptation)、削弱安全措施(Undermining Safeguards) —— 与追踪类别(Tracked Categories)区分开来。追踪类别会触发能力报告(Capabilities Reports)和安全措施报告(Safeguards Reports)，由安全咨询小组(Safety Advisory Group)审查。DeepMind 的 FSF v3（2025年9月；追踪能力级别(Tracked Capability Levels)于2026年4月17日添加）将自主性纳入机器学习研发(ML R&D)和网络(Cyber)领域（ML R&D 自主性级别 1 = 以与人类+AI工具具有竞争力的成本完全自动化 AI 研发管道）。FSF v3 明确通过自动化监控应对欺骗性对齐(deceptive alignment)，以防止工具推理滥用(instrumental-reasoning misuse)。需要诚实指出：PF v2 中的研究类别（包括长程自主性）不会自动触发缓解措施；政策语言是“潜在的”。DeepMind 自身也表示，如果工具推理增强，自动化监控“长期来看将不足以维持效果”。

**类型：** 学习
**语言：** Python (stdlib, three-framework decision-table diff tool)
**前提条件：** 阶段 15 · 19 (Anthropic RSP)
**时间：** 约45分钟

## 问题

第19课仔细阅读了Anthropic的扩展策略。本课通过阅读OpenAI和DeepMind的策略来完善全貌。这三份文件是同类产物，回答同一个问题——前沿实验室何时应暂停或把关模型——它们在一小组类别上趋于一致，并在某些具体重要方面存在分歧。

共同点：三者都将长程自主性(long-range autonomy)标记为值得追踪的能力类别。三者都承认欺骗行为(deceptive behavior)（对齐伪装(alignment faking)、沙袋效应(sandbagging)）是一类特定的风险。三者都有内部审查机构。分歧：OpenAI将类别分为“追踪类别(Tracked)”（强制性缓解）和“研究类别(Research)”（无自动触发）。DeepMind将自主性纳入两个领域而非单独命名。各实验室分别命名追踪与研究、关键与中等、一级与二级；能力所在的“桶”(bucket)的操作后果在不同实验室有所不同。

将它们一起阅读是有益的练习。同一种能力在Anthropic可能是“强制性缓解”，在OpenAI是“监控但不触发”，在DeepMind是“在特定领域追踪”。政策立场很重要。

## 核心概念

### OpenAI Preparedness Framework v2（2025年4月）

结构：

- **追踪类别(Tracked Categories)**：触发能力报告(Capabilities Reports)（模型能做什么）以及安全措施报告(Safeguards Reports)（实施了哪些缓解措施）。在部署前由安全咨询小组(Safety Advisory Group)审核。
- **研究类别(Research Categories)**：实验室正在追踪但尚未承诺具体缓解措施的模型能力。包括长程自主性(Long-range Autonomy)、沙袋效应(Sandbagging)、自主复制与适应(Autonomous Replication and Adaptation)、削弱安全措施(Undermining Safeguards)。

v2的关键框架：研究类别不会自动触发缓解措施。政策语言是“潜在的”缓解措施。这是需要关注的具体差异：一个类别位于追踪还是研究，是“把关”与“观察”之间的关键。

### DeepMind Frontier Safety Framework v3（2025年9月；追踪能力级别(Tracked Capability Levels)于2026年4月17日添加）

结构：

- **关键能力级别(Critical Capability Levels, CCLs)**：跨越五个领域的能力阈值：网络(Cyber)、生物(Bio)、机器学习研发(ML R&D)、化学-生物-放射-核(CBRN)、自主性(Autonomy)（归并到ML R&D和网络）。
- **追踪能力级别(Tracked Capability Levels)**：2026年4月增加的额外粒度。具体示例：ML R&D自主性级别1 = 以与人类+AI工具具有竞争力的成本完全自动化AI研发管道。
- **欺骗性对齐监控(Deceptive alignment monitoring)**：明确承诺对工具推理滥用(instrumental-reasoning misuse)进行自动化监控。

自主性的框架与OpenAI不同。DeepMind没有将“自主性”作为顶层领域；而是将其归入自主性可能造成危害的领域（ML R&D和网络）。其观点是，没有领域的自主性是无风险的能力；相反的观点是，跨领域的自主性是框架应命名的元风险(meta-risk)。

### 三者共同点

- 内部安全咨询小组（分别称为Anthropic SAG、OpenAI SAG、DeepMind内部委员会）。对高能力模型在部署前进行审查。
- 明确提及欺骗性对齐/对齐伪装作为一类风险。
- 按声明节奏发布常设文件（Anthropic：前沿安全路线图(Frontier Safety Roadmap)、风险报告(Risk Report)；OpenAI：能力报告和安全措施报告；DeepMind：FSF更新周期）。
- 承认仅依赖监控的防御存在天花板。DeepMind明确表示：“自动化监控长期来看将不足以维持效果。”

### 分歧点

- **Anthropic**：v3.0移除了暂停承诺；AI R&D-4阈值被命名为下一道门。
- **OpenAI**：追踪与研究的分裂；研究类别（包括长程自主性）不会自动把关。
- **DeepMind**：自主性归入其他领域；追踪能力级别在2026年4月增加粒度。

### 沙袋效应：一种使三者复杂化的特定能力

沙袋效应(Sandbagging)（模型在评估中策略性地表现不佳）位于OpenAI的研究类别中。Anthropic的RSP v3.0通过评估上下文差距（第1课）来应对。DeepMind通过FSF v3中的欺骗性对齐监控来应对。

如果模型在评估中沙袋效应，每个框架的能力阈值都会被低估。框架只有在测量有效时才有效。这就是为什么除了实验室自我评估外，还需要外部测量（第21课，METR）和对抗性评估(adversarial evaluation)。

### 政策阅读技巧

- 定位：你关心的每一项能力都应该能在政策中找到。如果找不到，则政策未覆盖。
- 分类：它是追踪类别（触发缓解）还是研究类别（追踪但不触发）？OpenAI明确命名；Anthropic和DeepMind有各自的等效分类。
- 节奏：政策是按声明的时间表更新，还是仅在特定事件后更新？声明节奏更强。
- 独立性：外部审查是强制性的还是可选的？Anthropic与Apollo和美国AI安全研究所合作；OpenAI与METR合作；DeepMind主要依赖内部SAG。

## 使用它

`code/main.py`实现了一个小型的决策表差异工具。给定一个能力（自主性、欺骗性对齐、研发自动化、网络提升等），它输出三项政策各自如何分类该能力，以及触发哪些缓解措施。它是一个阅读辅助工具，而非政策工具。

## 发布

`outputs/skill-cross-policy-diff.md`针对特定能力生成跨政策比较，使用这三个框架作为参考。

## 练习

1. 运行`code/main.py`。确认差异工具的输出与你至少能从源文档验证的两个能力的政策相匹配。

2. 完整阅读OpenAI Preparedness Framework v2。识别每个研究类别。对于每个类别，写一句话说明为什么它属于研究类别而不是追踪类别。

3. 完整阅读DeepMind FSF v3，加上2026年4月追踪能力级别更新。确定ML R&D自主性级别1的具体评估标准。你将如何从外部测量它？

4. 沙袋效应位于OpenAI的研究类别中。设计一个能够迫使沙袋效应模型揭示其实际能力的评估。参考第1课关于评估上下文博弈的讨论。

5. 比较三项政策在某一特定能力（自选）上的处理。指出你认为哪项政策的分类最严格，哪项最不严格，并用源文本证明。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|---|---|---|
|  准备框架  |  "OpenAI的扩展策略"  |  PF v2（2025年4月）；追踪与研究类别  |
|  跟踪类别  |  "强制缓解"  |  触发能力 + 保障报告；SAG 审查  |
|  研究类别  |  "仅监控"  |  被跟踪但无自动缓解；包括远程自主  |
|  前沿安全框架  |  "DeepMind 的扩展策略"  |  FSF v3（2025 年 9 月）+ 跟踪能力级别（2026 年 4 月）  |
|  CCL  |  "关键能力级别"  |  DeepMind 每个域的阈值（网络、生物、机器学习研发、CBRN）  |
|  ML 研发自主级别 1  |  "研发自动化"  |  以竞争性成本完全自动化 AI 研发流程  |
|  沙袋效应  |  "策略性表现不佳"  |  模型在评估中表现不佳；在 OpenAI 研究类别中  |
|  工具性推理  |  "手段-目的推理"  |  关于如何实现目标的推理；DeepMind 监控的目标  |

## 延伸阅读

- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — v2 公告。
- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — 完整文档。
- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — FSF v3 公告。
- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — 跟踪能力级别添加。
- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — FSF 格式风险报告示例。
