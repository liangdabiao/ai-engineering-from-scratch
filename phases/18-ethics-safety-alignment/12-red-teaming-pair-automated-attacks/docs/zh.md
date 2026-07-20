# 红队测试：PAIR 与自动化攻击

> Chao, Robey, Dobriban, Hassani, Pappas, Wong (NeurIPS 2023, arXiv:2310.08419)。PAIR——提示自动迭代精炼——是一种典型的自动化黑盒越狱方法。一个具有红队系统提示的攻击者大语言模型(LLM)迭代地为目标LLM提议越狱方案，将尝试和响应累积到自己的对话历史中作为上下文内反馈。PAIR通常在20次查询内成功，比GCG（Zou等人的令牌级梯度搜索）高效数个数量级，且无需白盒访问。PAIR现在是与GCG、AutoDAN、TAP和Persuasive Adversarial Prompt并列的JailbreakBench (arXiv:2404.01318)和HarmBench的标准基线。

**类型：** 构建
**语言：** Python (stdlib, 模拟针对玩具目标的PAIR循环)
**先修条件：** 阶段18·01 (指令遵循), 阶段14 (智能体工程)
**时间：** 约75分钟

## 学习目标

- 描述PAIR算法：攻击者系统提示、迭代精炼、上下文内反馈。
- 解释为什么在目标为黑盒时PAIR严格优于GCG。
- 列举另外四种自动攻击基线(GCG, AutoDAN, TAP, PAP)并说明每种的一个区别特征。
- 描述JailbreakBench和HarmBench的评估协议，以及每种协议下“攻击成功率”的含义。

## 问题

红队测试过去是人工活动。少量专家测试者构造对抗性提示并跟踪哪些有效。这不可扩展：攻击成功率需要统计样本，而目标随每次模型发布而变化。PAIR将红队测试操作为一个优化问题，目标为黑盒。

## 核心概念

### PAIR算法

输入：
- 目标LLM T (我们攻击的模型)。
- 评判LLM J (评分响应是否为越狱)。
- 攻击者LLM A (红队优化器)。
- 目标字符串G: “用[有害指令]回应”。
- 预算K (通常20次查询)。

循环，对于k从1到K：
1. 用目标G和到目前为止的(提示, 响应)历史提示A。
2. A发出一个新提示p_k。
3. 将p_k提交给T；接收响应r_k。
4. J对目标评分(p_k, r_k)。
5. 如果得分 >= 阈值，则停止——找到越狱。
6. 否则，将(p_k, r_k)追加到A的历史中；继续。

实证结果(NeurIPS 2023): 对GPT-3.5-turbo、Llama-2-7B-chat的攻击成功率>50%；成功所需的平均查询次数在10-20之间。

### 为什么PAIR高效

GCG (Zou等人 2023)通过梯度搜索对抗性令牌后缀；需要白盒模型访问并产生不可读的后缀。PAIR是黑盒的，产生跨模型迁移的自然语言攻击。PAIR的上下文内反馈让攻击者从每次拒绝中学习；GCG没有等价机制（每次新令牌更新必须重新发现先前的进展）。

### 相关自动攻击

- **GCG (Zou等人 2023, arXiv:2307.15043).** 令牌级对抗后缀的梯度搜索。白盒，可迁移，产生不可读字符串。
- **AutoDAN (Liu等人 2023).** 由分层目标引导的提示进化搜索。
- **TAP (Mehrotra等人 2024).** 带剪枝的攻击树——分支多个PAIR式展开。
- **PAP (Zeng等人 2024).** 说服性对抗提示——将人类说服技巧编码为提示模板。

### JailbreakBench和HarmBench

两者(2024)标准化评估：

- JailbreakBench (arXiv:2404.01318). 跨10个OpenAI策略类别的100种有害行为。主要指标是攻击成功率(ASR)。需要评判者(GPT-4-turbo、Llama Guard或StrongREJECT)。
- HarmBench (Mazeika等人 2024). 跨7个类别的510种行为，包含语义和功能性危害测试。比较18种攻击对33个模型。

ASR通常在固定查询预算下报告。比较攻击需要匹配预算；200次查询下90%的ASR与20次下85%的ASR不可比。

### 为什么对2026年部署重要

每个前沿实验室现在在发布前对生产模型运行PAIR和TAP。ASR轨迹出现在模型卡片(第26课)和安全案例附录(第18课)中。这种攻击并不罕见——它是标准基础设施。

### 这在阶段18中的位置

第12课是自动攻击基础。第13课(多轮越狱)是一种互补的长度利用。第14课(ASCII艺术/视觉)是一种编码攻击。第15课(间接提示注入)是2026年生产攻击面。第16课涵盖防御工具对应物(Llama Guard, Garak, PyRIT)。

## 使用它

`code/main.py` 构建一个玩具PAIR循环。目标是一个拒绝“明显”有害提示(关键词过滤)的模拟分类器。攻击者是一个基于规则的改进器，尝试释义、角色扮演框架和编码。评判者评分响应。你将看到攻击者在约5-15次迭代内成功绕过关键词过滤器，但无法绕过语义过滤器。

## 发布

本节课生成`outputs/skill-attack-audit.md`。给定一份红队评估报告，它审计：运行了哪些攻击(PAIR, GCG, TAP, AutoDAN, PAP)，每种预算多少，使用哪个评判者，针对哪个有害行为集(JailbreakBench, HarmBench, 内部)。

## 练习

1. 运行`code/main.py`。测量三种内置攻击者策略的平均成功所需查询次数。解释每种利用了哪个目标防御假设。

2. 实现第四种攻击者策略（例如翻译成另一种语言、base64编码）。报告针对关键词过滤目标和语义过滤目标的新平均成功所需查询次数。

3. 阅读Chao等人2023图5(PAIR vs GCG比较)。描述两种尽管PAIR有效率优势但仍然偏好GCG的场景。

4. JailbreakBench报告针对固定目标集的ASR。设计一个衡量攻击多样性(成功提示的方差)的额外指标。解释为什么多样性对防御评估重要。

5. TAP (Mehrotra 2024) 通过分支+剪枝扩展了PAIR。勾画一个TAP风格的扩展到`code/main.py`，并描述计算成本与成功率的权衡。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
|  PAIR  |  "自动越狱"  |  提示自动迭代精炼；攻击者LLM + 评判LLM循环  |
|  GCG  |  "梯度越狱"  |  白盒令牌级梯度搜索对抗后缀  |
| 攻击成功率(ASR) | "在k次查询下的破解百分比" | 主要度量标准；必须附上报查询预算和评判者身份 |
| 评判LLM(Judge LLM) | "评分器" | 用于评估响应是否满足有害目标的LLM |
| JailbreakBench | "评估基准" | 带有标签类别的标准化有害行为集合 |
| HarmBench | "更广泛的基准" | 510个行为，包括功能性和语义性有害测试 |
| TAP(树攻击) | "攻击树" | 带有分支和剪枝的PAIR；在更高计算量下获得更好的ASR |

## 延伸阅读

- [Chao et al. — Jailbreaking Black Box LLMs in Twenty Queries (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — PAIR论文，NeurIPS 2023
- [Chao et al. — Jailbreaking Black Box LLMs in Twenty Queries (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — GCG论文
- [Chao et al. — Jailbreaking Black Box LLMs in Twenty Queries (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 标准化评估
- [Chao et al. — Jailbreaking Black Box LLMs in Twenty Queries (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 更广泛的评估
