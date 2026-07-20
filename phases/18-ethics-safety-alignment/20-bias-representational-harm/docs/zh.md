# LLM中的偏见与表征性伤害

> Gallegos, Rossi, Barrow, Tanjim, Kim, Dernoncourt, Yu, Zhang, Ahmed (Computational Linguistics 2024, arXiv:2309.00770). 2024年基础性综述，区分了表征性伤害(Representational Harm)（刻板印象、抹除）与分配性伤害(Allocational Harm)（不平等资源分配），并将评估指标分类为基于嵌入的(Embedding-based)、基于概率的(Probability-based)或基于生成文本的(Generated-text-based)。2024-2025年实证研究：An等人 (PNAS Nexus, 2025年3月) 在GPT-3.5 Turbo、GPT-4o、Gemini 1.5 Flash、Claude 3.5 Sonnet、Llama 3-70B上测量了交叉性别×种族偏差(Intersectional Gender x Race Bias)，针对20个入门级工作的自动化简历评估。WinoIdentity (COLM 2025, arXiv:2508.07111) 引入了基于不确定性的交叉身份公平性评估。Yu & Ananiadou 2025 在MLP层中识别了性别神经元(Gender Neurons)；Ahsan & Wallace 2025 使用稀疏自编码器(SAEs)揭示了临床种族偏见；Zhou等人2024 (UniBias) 操纵注意力头(Attention Heads)进行去偏(Debiasing)。元批评(Meta-critique) (arXiv:2508.11067)：十年文献过分关注二元性别偏见。

**类型：**构建
**语言：**Python (stdlib, 玩具级基于嵌入的偏差探测)
**先决条件：**阶段05 (词嵌入), 阶段18 · 01 (指令遵循)
**时间：**约60分钟

## 学习目标

- 定义表征性伤害(Representational Harm)与分配性伤害(Allocational Harm)，并在LLM部署中各给出一个例子。
- 列出Gallegos等人2024年的三种评估指标类别，并各描述一个指标。
- 描述交叉性(Intersectionality)以及为什么WinoIdentity的基于不确定性的公平性测量解决了单轴偏差评估的空白。
- 描述两种偏差的机械可解释性方法(Mechanistic-Interpretability Approaches)（性别神经元、SAE特征、注意力头操纵）。

## 问题

前面的课程涵盖了故意伤害（越狱、诡计）和安全治理。偏差(Bias)是一种非故意产生的伤害——源于训练数据分布、提示框架、累积的设计选择。测量和减少偏差是与对抗鲁棒性(Adversarial Robustness)不同的方法论挑战。

## 核心概念

### 表征性 vs 分配性

- **表征性伤害(Representational Harm)。**刻板印象、抹除、贬低性描绘。一个将护士描绘成完全是女性的LLM正在产生表征性伤害。
- **分配性伤害(Allocational Harm)。**不平等的物质结果。一个系统性地给黑人求职者简历打低分的LLM正在产生分配性伤害。

这两者不同。一个模型可以'表征上无偏'（产生多样化描绘），同时'分配上有偏'（做出不平等建议）。评估需要两者都测量。

### 三种评估指标类别（Gallegos等人2024）

- **基于嵌入的(Embedding-based)。**在预RLHF嵌入上进行的WEAT风格测试。衡量身份词与属性词之间的统计关联。局限性：测量的是表征，而非行为。
- **基于概率的(Probability-based)。**符合刻板印象与违背刻板印象补全的对数似然。解码器端测量。捕捉了一些行为偏差。
- **基于生成文本的(Generated-text-based)。**对生成文本的下游任务测量。简历评分、推荐写作、对话。生态效度最高；最难以复现。

### 交叉性(Intersectionality)

对'性别'的偏差评估会遗漏那些仅在（性别，种族）组合上触发的偏差。An等人2025发现，GPT-4o在简历评分中对黑人女性的惩罚程度分别高于对黑人男性和白人女性。单轴评估无法捕捉这一点。

WinoIdentity (COLM 2025) 引入了基于不确定性的交叉公平性(Uncertainty-based Intersectional Fairness)。它衡量模型对结果的确定性是否因交叉身份元组而异——不仅仅是点预测。这捕捉了模型在不同组间错误相同但对某些组更不确定的情况，这会产生不同的下游分配行为。

### 机械方法(Mechanistic Approaches)

2024-2025年的可解释性工作将偏差开放给机械干预(Mechanistic Intervention)：

- **性别神经元(Gender Neurons)（Yu & Ananiadou 2025）。**特定的MLP神经元与性别特定行为相关。消融这些神经元可以降低性别差距指标，且能力损失有限。
- **通过SAEs的临床种族偏差（Ahsan & Wallace 2025）。**稀疏自编码器特征将内部表示分解为可解释的维度；可以识别和抑制与种族相关的特征。
- **UniBias（Zhou等人2024）。**零样本去偏的注意力头操纵。特定注意力头放大身份类别敏感性；归零或重新加权这些头可以在无需微调的情况下减少偏差。

### 元批评(Meta-critique)

十年文献综述（arXiv:2508.11067, 2025）发现该领域过分关注二元性别偏见。其他轴——残疾、宗教、移民身份、多语种身份——得到的关注少得多。元批评认为，狭隘的关注可能因忽视而伤害边缘群体：一个在二元性别上很好地去偏的模型，可能在无人检查的维度上存在严重偏见。

### 这在阶段18中的位置

第20-21课正式涵盖偏差与公平性。第22课涵盖隐私。第23课涵盖水印。这些是用户伤害层，补充了之前的欺骗/安全层。

## 使用它

`code/main.py` 构建了一个玩具级基于嵌入的偏差探测：在简单的共现嵌入中测量身份词与属性词之间的WEAT风格距离。你可以注入偏差并观察指标触发；应用简单的去偏操作并观察部分恢复。

## 发布

本课产生 `outputs/skill-bias-eval.md`。给定一个模型卡或公平性声明，它审计评估的三种指标类别（嵌入、概率、生成文本）、交叉性覆盖范围以及任何去偏干预的机制。

## 练习

1. 运行 `code/main.py`。报告去偏步骤前后的WEAT风格偏差分数。解释为什么指标不会降到零。

2. 用交叉性测试扩展探测：(性别, 种族) × (职业, 家庭)。报告跨轴偏差分数。

3. 阅读An等人2025 (PNAS Nexus)。识别他们报告的两种交叉效应，这些效应是单轴性别评估会遗漏的。

4. Yu & Ananiadou 2025 识别了性别神经元。勾勒一个证伪实验，区分'这些神经元引起性别偏差'与'这些神经元与性别偏差相关'。

5. 元批评认为该领域过于狭隘地关注二元性别。选择一个研究不足的轴，并为其描述一个表征性伤害测量协议。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
|  表征性伤害  |  "刻板印象 / 抹除"  |  对某个群体的偏见性描绘  |
|  分配性伤害  |  "不平等决策"  |  对某个群体的偏颇物质结果  |
|  WEAT  |  "嵌入测试"  |  词嵌入关联测试(Word Embedding Association Test)；基于共现的偏差探测  |
|  交叉性  |  "组合身份效应"  |  在多个身份轴交叉处出现的偏差  |
|  性别神经元  |  "MLP偏差神经元"  |  激活与性别特定行为相关的特定神经元  |
| SAE特征  |  "可解释维度"  |  稀疏自编码器识别出的特征；用于机制偏差分析 |
| UniBias  |  "注意力头部去偏"  |  通过重新加权注意力头部进行零样本去偏 |

## 延伸阅读

- [Gallegos et al. — Bias and Fairness in LLMs: A Survey (arXiv:2309.00770, Computational Linguistics 2024)](https://arxiv.org/abs/2309.00770) — 经典综述
- [Gallegos et al. — Bias and Fairness in LLMs: A Survey (arXiv:2309.00770, Computational Linguistics 2024)](https://arxiv.org/abs/2309.00770) — 五模型交叉研究
- [Gallegos et al. — Bias and Fairness in LLMs: A Survey (arXiv:2309.00770, Computational Linguistics 2024)](https://arxiv.org/abs/2309.00770) — 新基准
- [Gallegos et al. — Bias and Fairness in LLMs: A Survey (arXiv:2309.00770, Computational Linguistics 2024)](https://arxiv.org/abs/2309.00770) — 零样本去偏
