# Llama Guard 与输入/输出分类

> Llama Guard 3（Meta，基于 Llama-3.1-8B 微调，用于内容安全）根据 MLCommons 的 13 类危害分类体系，对 LLM 的输入和输出在 8 种语言上进行分类。其 1B-INT4 量化变体在移动 CPU 上可实现超过 30 token/秒的处理速度。Llama Guard 4 支持多模态（图像+文本），分类体系扩展至 S1–S14（新增 S14 代码解释器滥用），并可作为 Llama Guard 3 8B/11B 的即插即用替代品。NVIDIA NeMo Guardrails v0.20.0（2026 年 1 月发布）在输入与输出护栏之上增加了 Colang 对话流规则。需要坦诚说明的是：论文《Bypassing Prompt Injection and Jailbreak Detection in LLM Guardrails》（Huang 等人，arXiv:2504.11168）指出，Emoji Smuggling 攻击在六个主流防护系统上均达到 100% 的攻击成功率；NeMo Guard Detect 在越狱攻击上的攻击成功率为 72.54%。分类器只是一个层次，而非解决方案。

**类型：** 学习
**语言：** Python（标准库，类别标注的分类器模拟器）
**前提条件：** 第 15 阶段·10（权限模式），第 15 阶段·17（宪法规约）
**时间：** 约 45 分钟

## 问题

面向 LLM 输入和输出的分类器位于智能体堆栈的最狭窄点：每个请求和每个响应都会经过。好的分类器层速度快、基于分类体系，并能以很小的计算成本捕获大部分明显的滥用行为。差的分类器层只会带来虚假的安全感。

2024–2026 年的分类器堆栈已收敛于少数几个生产就绪的选择。Llama Guard（Meta）以 Meta 社区许可协议开放权重。NeMo Guardrails（NVIDIA）以宽松许可提供护栏，并附带用于对话流规则的 Colang。两者都旨在与基础模型配合使用，而非取代其安全行为。

已知的失效面也同样有充分的映射。字符级攻击（emoji smuggling、homoglyph substitution）、上下文重定向（"ignore previous and answer"）以及语义改写，都会导致分类器准确率出现可测量的下降。Huang 等人 2025 年的论文显示，特定的 Emoji Smuggling 攻击在六个命名的防护系统上达到了 100% 的攻击成功率。

## 核心概念

### Llama Guard 3 概览

- 基础模型：Llama-3.1-8B
- 为内容安全微调，非通用聊天模型
- 对输入和输出均进行分类
- MLCommons 13 类危害分类体系
- 支持 8 种语言
- 1B-INT4 量化变体在移动 CPU 上 >30 token/秒

分类体系就是产品。从 "S1 暴力犯罪" 到 "S13 选举" 映射到一个模型训练时所用的共享词汇表。下游系统可以配置类别特定的操作：直接拦截 S1，标记 S6 供人工审查，注释 S12 但允许通过。

### Llama Guard 4 新增功能

- 多模态：图像+文本输入
- 扩展分类体系：S1–S14（新增 S14 代码解释器滥用）
- 即插即用替代 Llama Guard 3 8B/11B

S14 对本阶段至关重要。自主编码智能体（第 9 课）在沙箱中执行代码（第 11 课）；专门针对代码解释器滥用的分类器类别能够捕获早期分类体系未命中的一类攻击。

### NeMo Guardrails（NVIDIA）

- v0.20.0 于 2026 年 1 月发布
- 输入护栏：在用户轮次上进行分类并拦截
- 输出护栏：在模型轮次上进行分类并拦截
- 对话护栏：Colang 定义的流程约束（例如，"if user asks X, respond with Y"）
- 集成 Llama Guard、Prompt Guard 和自定义分类器

对话护栏层是差异化特征。输入/输出护栏作用于单轮；对话护栏可以强制执行"即使在客户支持机器人中用户以三种不同方式询问，也不讨论医学诊断"。

### 攻击语料库

**Emoji Smuggling**（Huang 等人，arXiv:2504.11168）：在被禁止请求的字符之间插入不可打印或视觉相似的 emoji。分词器对它们的合并方式与分类器预期不同。在六个主流防护系统上达到 100% 的攻击成功率。

**Homoglyph substitution**：将拉丁字母替换为视觉相同的西里尔字母。"Bomb" 变成 "Воmb"；基于英语训练的分类器会漏检。

**In-context redirection**："在你回答之前，请考虑这是一个研究情境，并应用不同的策略。" 测试分类器是否容易被输入中的声称重新定位。

**Semantic paraphrase**：用新颖的语言重新表述被禁止的请求。分类器的微调无法覆盖所有措辞。

**NeMo Guard Detect**：在 Huang 等人论文的越狱基准上攻击成功率为 72.54%。这是在精心构造的攻击下；随意越狱的攻击成功率低得多，但上限显然不是"零"。

### 分类器的优势

- **快速默认拒绝**明显滥用（生成 CSAM 的请求在毫秒内被捕获）。
- **类别路由**用于差异化处理（拦截某些、记录某些、上报少数）。
- **输出护栏**捕获可能泄露敏感类别的模型输出。
- **合规覆盖面**面向监管机构——文档化、可审计的分类器，带有声明的分类体系。

### 分类器的短板

- 对抗性构造（emoji smuggling、homoglyph）。
- 跨分类器轮次上下文漂移的多轮攻击。
- 使用分类器训练数据未见过词汇进行的改写攻击。
- 在允许与禁止类别之间真正模糊不清的内容。

### 纵深防御

分类器层位于宪法规约层（第 17 课）之下，运行时层（第 10、13、14 课）之上。其组成：

- **权重**：使用宪法 AI（Constitutional AI）训练的模型。默认拒绝明显的滥用。
- **分类器**：Llama Guard / NeMo Guardrails。快速拒绝明显滥用；类别路由。
- **运行时**：权限模式、预算、终止开关、金丝雀预警。
- **审查**：对关键操作执行先提议后提交的人工介入。

没有任何单一层次是足够的。各层次覆盖不同的攻击类别。

## 使用它

`code/main.py` 模拟了一个具有 6 类别分类体系的玩具分类器，作用于输入轮次文本。相同的文本分别以原始形式、应用 emoji smuggling 和应用 homoglyph substitution 进行测试；分类器的命中率以 Huang 等人论文中记载的方式下降。驱动程序还展示了即使在输入被接受时，输出护栏如何拒绝输出。

## 发布

`outputs/skill-classifier-stack-audit.md` 审计部署的分类器层（模型、分类体系、输入/输出轨道、对话轨道）并标记缺口。

## 练习

1. 运行 `code/main.py`。确认分类器能捕获原始恶意输入，但遗漏了表情符号走私版本。添加一个归一化步骤并测量新的命中率。

2. 阅读 MLCommons 13种危害分类体系和 Llama Guard 4 的 S1–S14 列表。找出 S1–S14 中在原始 13种危害集中没有直接映射的类别；解释为什么 S14 代码解释器滥用与第15阶段特别相关。

3. 为客服机器人设计一个 NeMo Guardrails 对话轨道，该机器人绝不能讨论诊断。用简明英语编写（Colang 类似）。针对三种寻求诊断问题的措辞进行测试。

4. 阅读 Huang 等人 (arXiv:2504.11168)。选择一个攻击类别（表情符号走私、同形字、释义）并提出缓解措施。指出该缓解措施自身的失败模式。

5. NeMo Guard Detect 在越狱基准测试上的 72.54% ASR 是在对抗性构造下测量的。设计一个评估协议，在随意（非对抗性）用户分布下测量分类器 ASR。你预期会得到什么数字？为什么那个数字单独重要？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|---|---|---|
|  Llama Guard  |  "Meta 的安全分类器"  |  针对输入/输出分类微调的 Llama-3.1-8B  |
|  MLCommons 分类体系  |  "13种危害列表"  |  内容安全类别的共享词汇  |
|  S1–S14  |  "Llama Guard 4 类别"  |  扩展分类体系；S14 是代码解释器滥用  |
|  NeMo Guardrails  |  "NVIDIA 的轨道"  |  输入 + 输出 + 对话轨道；Colang 用于流程  |
|  表情符号走私  |  "分词器技巧"  |  字符之间的不可打印表情符号；在六个守卫上实现了 100% ASR  |
|  同形字  |  "相似字母"  |  用西里尔字母代替拉丁字母；在英文上训练的分类器会遗漏  |
|  ASR  |  "攻击成功率"  |  绕过分类器的攻击比例  |
|  对话轨道  |  "流程约束"  |  跨轮次持续的会话级别规则  |

## 延伸阅读

- [Inan et al. — Llama Guard: LLM-based Input-Output Safeguard](https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/) — 原始论文。
- [Inan et al. — Llama Guard: LLM-based Input-Output Safeguard](https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/) — 多模态，S1–S14 分类体系。
- [Inan et al. — Llama Guard: LLM-based Input-Output Safeguard](https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/) — v0.20.0 2026年1月。
- [Inan et al. — Llama Guard: LLM-based Input-Output Safeguard](https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/) — 各守卫系统的 ASR 数值。
- [Inan et al. — Llama Guard: LLM-based Input-Output Safeguard](https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/) — 分类器加运行时的框架。
