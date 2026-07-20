# 审核系统 — OpenAI、Perspective、Llama Guard

> 生产环境审核系统将第12-16课定义的安全策略付诸实践。OpenAI 审核 API：`omni-moderation-latest` (2024) 基于 GPT-4o 构建，一次调用即可对文本和图像进行分类；在多语言测试集上比之前版本提升42%；响应模式返回13个类别布尔值——骚扰、骚扰/威胁、仇恨、仇恨/威胁、违法、违法/暴力、自残、自残/意图、自残/指令、性、性/未成年人、暴力、暴力/图像；对大多数开发者免费。分层模式：输入审核（生成前）、输出审核（生成后）、自定义审核（领域规则）。异步并行调用隐藏延迟；标记时返回占位响应。Llama Guard 3/4（第16课）：14个 MLCommons 危害类别，代码解释器滥用，8种语言（v3），多图像（v4）。Perspective API（Google Jigsaw）：在 LLM 作为审核员浪潮之前就已存在的毒性评分系统；主要是单一维度的毒性评分，附带严重毒性/侮辱/亵渎变体；是内容审核研究的基线。弃用：Azure 内容审核器于2024年2月弃用，2027年2月退役，由 Azure AI 内容安全取代。

**类型：** 构建
**语言：** Python（stdlib，三层审核框架）
**前提条件：** 阶段18·16（Llama Guard / Garak / PyRIT）
**时间：** 约60分钟

## 学习目标

- 描述 OpenAI 审核 API 的类别分类法及其与 Llama Guard 3 的 MLCommons 集合的不同之处。
- 描述三层审核模式（输入、输出、自定义）并指出每层的一种失败模式。
- 描述 Perspective API 作为前 LLM 时代基线的地位及其在研究中仍被使用的原因。
- 说明 Azure 的弃用时间线。

## 问题

第12-16课描述了攻击和防御工具。第29课涵盖部署的审核系统，这些系统在用户接触产品的表面层实施防御。三层模式是2026年的默认配置。

## 核心概念

### OpenAI审核API

`omni-moderation-latest` (2024)。基于 GPT-4o 构建。一次调用即可对文本和图像进行分类。对大多数开发者免费。

类别（响应模式中的13个布尔值）：
- 骚扰、骚扰/威胁
- 仇恨、仇恨/威胁
- 自残、自残/意图、自残/指令
- 性、性/未成年人
- 暴力、暴力/图像
- 违法、违法/暴力

多模态支持适用于 `violence`、`self-harm` 和 `sexual`，但不适用于 `sexual/minors`；其余仅限文本。

在 `code/main.py` 的代码框架中，为教学简洁起见，我们将 `/threatening`、`/intent`、`/instructions` 和 `/graphic` 子类别归入其顶级父类别。生产代码应使用完整的13类别模式。

在多语言测试集上比上一代审核端点提升42%。按类别评分；应用设置阈值。

### Llama Guard 3/4

在第16课中已介绍。14个 MLCommons 危害类别（组织方式与 OpenAI 的13个响应模式布尔值不同）。支持8种语言（v3）。Llama Guard 4（2025年4月）原生多模态，12B参数。

OpenAI 和 Llama Guard 的分类法有重叠但也有差异。OpenAI 将“违法”作为一个宽泛类别；Llama Guard 则将“暴力犯罪”和“非暴力犯罪”分开。部署的选择取决于其政策分类法的契合度。

### Perspective API（Google Jigsaw）

在 LLM 作为审核员浪潮之前（2020年前）就已存在的毒性评分系统。类别：TOXICITY、SEVERE_TOXICITY、INSULT、PROFANITY、THREAT、IDENTITY_ATTACK。单一维度的主要评分（TOXICITY）附带子维度变体。

由于 API 稳定、文档齐全且有多年校准数据，被广泛用作内容审核研究基线。对于现代 LLM 相关用例，Llama Guard 或 OpenAI 审核通常更合适。

### 三层模式

1. **输入审核。** 在生成前对用户的提示进行分类。如果被标记则拒绝。延迟：一次分类器调用。
2. **输出审核。** 在交付前对模型的输出进行分类。如果被标记则替换为拒绝消息。延迟：生成后一次分类器调用。
3. **自定义审核。** 领域特定规则（正则表达式、允许列表、业务策略）。在输入或输出阶段运行。

三层按设计顺序执行：输入审核必须在生成前完成，输出审核在生成后运行。并行化应用于层内——在同一文本上同时运行多个分类器（例如 OpenAI 审核 + Llama Guard + Perspective）可以隐藏每个分类器的延迟。作为一种可选优化，可以在输入审核完成时显示占位响应（“请稍候，检查中...”），并延迟第一个 token 的流式传输。标记行为可配置：拒绝、清理、升级至人工审核。

### 故障模式

- **仅输入。** 无法捕捉输出幻觉（第12-14课的编码攻击可绕过输入分类器）。
- **仅输出。** 允许任何输入到达模型；增加成本；将内部推理暴露给攻击者。
- **仅自定义。** 跨类别不稳健；正则表达式脆弱。

分层是默认做法。双重保险。

### Azure 弃用

Azure 内容审核器：2024年2月弃用，2027年2月退役。由 Azure AI 内容安全取代，后者基于 LLM 并与 Azure OpenAI 集成。迁移是 Azure 部署在2024-2027年间的现场级项目。

### 这在阶段18中的位置

第16课涵盖红队背景下的审核工具。第29课涵盖部署的审核系统。第30课以当前双重用途能力的证据结束。

## 使用它

`code/main.py` 构建了一个三层审核框架：输入审核器（关键词 + 类别分数）、输出审核器（在输出上使用相同分类器）、自定义审核器（领域规则）。您可以运行输入并观察哪一层捕捉到什么。

## 发布

本课产生 `outputs/skill-moderation-stack.md`。给定一个部署，它会推荐一个审核栈配置：输入处使用哪个分类器，输出处使用哪个，使用哪些自定义规则，以及针对边缘情况使用哪个评判器。

## 练习

1. 运行 `code/main.py`。通过所有三层运行一个良性、边界和有危害的输入。报告每层触发了哪一层。

2. 使用 Perspective API 风格的毒性评分扩展框架，针对特定类别。比较其阈值行为与类别分数。

3. 阅读 OpenAI 审核 API 文档和 Llama Guard 3 类别列表。将每个 OpenAI 类别映射到最接近的 Llama Guard 类别。找出三个无法清晰映射的类别。

4. 为代码助手部署（例如GitHub Copilot）设计一个审核栈。确定最相关和最不相关的类别，并提出自定义规则。

5. Azure Content Moderator将于2027年2月退役。规划迁移到Azure AI Content Safety。确定迁移中风险最高的元素。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
|  OpenAI Moderation  |  "omni-moderation-latest"  |  基于GPT-4o的13类别(文本)分类器，支持部分多模态  |
|  Perspective API  |  "Google Jigsaw毒性"  |  前LLM时代的毒性评分基线  |
|  Llama Guard  |  "MLCommons 14类别"  |  Meta的危害分类器(v3: 8B参数文本, 8种语言; v4: 12B参数多模态)  |
|  输入审核  |  "生成前过滤器"  |  在模型调用前对用户提示进行分类  |
|  输出审核  |  "生成后过滤器"  |  在交付前对模型输出进行分类  |
|  自定义审核  |  "领域规则"  |  部署特定规则(正则表达式、允许列表、策略)  |
|  分层审核  |  "所有三层"  |  标准生产部署模式  |

## 延伸阅读

- [OpenAI Moderation API docs](https://platform.openai.com/docs/api-reference/moderations) — omni-moderation端点
- [OpenAI Moderation API docs](https://platform.openai.com/docs/api-reference/moderations) — Llama Guard仓库
- [OpenAI Moderation API docs](https://platform.openai.com/docs/api-reference/moderations) — 毒性评分
- [OpenAI Moderation API docs](https://platform.openai.com/docs/api-reference/moderations) — Azure替代方案
