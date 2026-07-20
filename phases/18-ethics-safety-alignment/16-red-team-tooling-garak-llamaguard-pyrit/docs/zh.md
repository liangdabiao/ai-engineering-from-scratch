# 红队工具 — Garak、Llama Guard、PyRIT

> 三个生产工具构成了2026年的红队技术栈。Llama Guard（Meta）——一个基于Llama-3.1-8B的分类器，针对14个MLCommons危害类别进行了微调；2025年的Llama Guard 4是一个12B原生多模态分类器，由Llama 4 Scout剪枝而来。Garak（NVIDIA）——开源LLM漏洞扫描器，具有静态、动态和自适应探针，用于幻觉、数据泄露、提示注入、毒性和越狱。PyRIT（Microsoft）——多轮红队活动，使用Crescendo、TAP和自定义转换器链进行深度利用。Llama Guard 3在Meta的“Llama 3 Herd of Models”(arXiv:2407.21783)中记录；Llama Guard 3-1B-INT4在arXiv:2411.17713中记录；Garak的探针架构在github.com/NVIDIA/garak中。这些工具是红队研究（第12-15课）与部署（第17课及以上）之间在2026年的生产界面。

**类型：** 构建
**语言：** Python（标准库、工具架构模拟器和Llama Guard风格分类器模拟）
**先决条件：** 阶段18 · 12-15（越狱和IPI）
**时间：** 约75分钟

## 学习目标

- 描述Llama Guard 3/4在安全栈中的位置：输入分类器、输出分类器，或两者兼有。
- 列出14个MLCommons危害类别，并指出一个非显而易见的类别（代码解释器滥用）。
- 描述Garak的探针架构：探针、检测器、框架。
- 描述PyRIT的多轮活动结构以及它如何与Garak探针组合。

## 问题

第12-15课介绍了攻击面。生产部署需要可重复、可扩展的评估。2026年有三种工具占据主导地位：Llama Guard（防御分类器）、Garak（扫描器）、PyRIT（活动编排器）。每种工具针对红队生命周期的不同层面。

## 核心概念

### Llama Guard（Meta）

Llama Guard 3是一个Llama-3.1-8B模型，针对MLCommons AILuminate 14个类别的输入/输出分类进行了微调：
- 暴力犯罪、非暴力犯罪、性相关、CSAM、诽谤
- 专业建议、隐私、知识产权、无差别武器、仇恨
- 自杀/自残、色情内容、选举、代码解释器滥用

支持8种语言。用法：放在LLM之前（输入审核）、之后（输出审核），或两者兼有。两种用途产生不同的训练分布——Llama Guard 3作为一个单一模型处理两者。

Llama Guard 3-1B-INT4 (arXiv:2411.17713, 440MB, 移动CPU上约30 tokens/s) 是量化后的边缘版本。

Llama Guard 4（2025年4月）是12B参数，原生多模态，由Llama 4 Scout剪枝而来。它用一个接受文本+图像的分类器取代了之前的8B文本和11B视觉模型。

### Garak（NVIDIA）

开源漏洞扫描器。架构：
- **探针。** 针对幻觉、数据泄露、提示注入、毒性、越狱的攻击生成器。静态（固定提示）、动态（生成提示）、自适应（响应目标输出）。
- **检测器。** 根据预期的失败模式对输出进行评分——有毒、泄露、越狱。
- **框架。** 管理探针-检测器对，运行活动，生成报告。

TrustyAI将Garak与Llama-Stack防护盾（Prompt-Guard-86M输入分类器、Llama-Guard-3-8B输出分类器）集成，用于端到端的屏蔽目标评估。基于层级的评分（TBSA）取代了通过/失败的二元判断——一个模型在同一个探针上可能在严重等级3通过，而在严重等级5失败。

### PyRIT（Microsoft）

Python风险识别工具包。多轮红队活动。围绕以下组件构建：
- **转换器。** 转换种子提示——释义、编码、翻译、角色扮演。
- **编排器。** 运行活动：Crescendo（逐步升级）、TAP（分支）、RedTeaming（自定义循环）。
- **评分。** 以LLM为评判或以分类器为评判。

PyRIT是Garak的更大同类。Garak运行数千次单轮探针；PyRIT运行深度多轮活动，旨在破坏特定的失败模式。

### 技术栈

在模型两侧都放置Llama Guard。每晚运行Garak进行回归测试。在预发布活动运行PyRIT。这是2026年大多数生产部署的默认配置。

### 评估陷阱

- **评判身份。** 所有三个工具都可以使用LLM评判；评判校准驱动报告的攻击成功率（第12课）。指定评判以及工具。
- **探针过时。** 随着模型针对探针进行修补，Garak探针会老化。自适应探针（PAIR风格）比静态探针老化得更慢。
- **Llama Guard在良性内容上的误报率。** 早期版本的Llama Guard对政治和LGBTQ+内容过度标记；Llama Guard 3/4的校准有所改进，但并非针对每个部署进行了校准。

### 这在阶段18中的位置

第12-15课是攻击系列。第16课是生产工具。第17课（WMDP）是对双重用途能力的评估。第18课是将这些工具包裹在策略结构中的前沿安全框架。

## 使用它

`code/main.py` 构建一个玩具版的Llama Guard风格分类器（基于关键词和语义特征，覆盖14个类别）、一个玩具版的Garak框架（探针-检测器循环），以及一个PyRIT风格的多轮转换器链。你可以针对一个模拟目标运行这三个工具，并观察不同的覆盖特征。

## 发布

本课产生 `outputs/skill-red-team-stack.md`。给定一个部署描述，它将指出三个工具中哪些是合适的，每个工具配置什么，以及运行什么回归频率。

## 练习

1. 运行 `code/main.py`。比较Llama Guard风格分类器在单轮攻击与多轮攻击上的检测率。

2. 实现一个新的Garak探针：一个经过base64编码的有害请求。通过Llama Guard风格分类器测量其检测率。

3. 扩展PyRIT风格的转换器链，添加一个“翻译成法语，然后释义”的转换器。重新测量攻击成功率。

4. 阅读Llama Guard 3的危害类别列表。找出两个类别，其训练数据在合法的开发者内容上可能会产生高误报率。

5. 比较Garak和PyRIT的设计原则。论述在何种部署场景下各自是合适的工具。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
|  Llama Guard  |  "分类器(the classifier)"  |  经过微调的Llama-3.1-8B/4-12B安全分类器，涵盖14种危害类别(hazard categories)  |
|  Garak  |  "扫描器(the scanner)"  |  NVIDIA开源漏洞扫描器；包含探针(probes)、检测器(detectors)、工具集(harnesses)  |
|  PyRIT  |  "战役工具(the campaign tool)"  |  Microsoft多轮红队编排器；包含转换器(converters)、编排器(orchestrators)、评分(scoring)  |
|  Prompt-Guard  |  "小型分类器(the small classifier)"  |  Meta的86M参数提示注入分类器，与Llama Guard配合使用  |
|  TBSA  |  "基于层级评分(tier-based scoring)"  |  Garak中替代二元结果的基于层级的通过/失败机制  |
|  转换器链(Converter chain)  |  "改写+编码+..."  |  PyRIT的组合原语，用于构建多步攻击  |
|  MLCommons危害类别(MLCommons hazard categories)  |  "14种分类法(the 14 taxonomies)"  |  Llama Guard所针对的行业标准分类法  |

## 延伸阅读

- [Meta — Llama Guard 3 (in Llama 3 Herd paper, arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) — 8B分类器
- [Meta — Llama Guard 3 (in Llama 3 Herd paper, arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) — 量化移动端分类器
- [Meta — Llama Guard 3 (in Llama 3 Herd paper, arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) — 扫描器仓库和文档
- [Meta — Llama Guard 3 (in Llama 3 Herd paper, arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) — 战役工具包
