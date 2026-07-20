# 顶点项目 17 — 个人 AI 导师（自适应、多模态、带记忆）

> Khanmigo（可汗学院）、Duolingo Max、Google LearnLM / Gemini for Education、Quizlet Q-Chat 和 Synthesis Tutor 都在2026年大规模推出了自适应多模态辅导。其共同形态是苏格拉底式策略（永远不只是给出答案）、每次交互后更新的学习者模型（贝叶斯知识追踪风格）、语音+文本+拍照数学输入、课程图谱检索、间隔重复调度，以及针对适龄内容的严格安全过滤器。收官之作是推出特定学科的辅导程序（K-12代数或Python入门），进行为期两周、10名学习者的效果研究，并通过内容安全审核。

**类型：** 收官项目
**语言：** Python（后端，学习者模型），TypeScript（网页应用），SQL（通过Postgres + Neo4j的课程图谱）
**前置条件：** 阶段5（NLP）、阶段6（语音）、阶段11（LLM工程）、阶段12（多模态）、阶段14（智能体）、阶段17（基础设施）、阶段18（安全）
**涉及的阶段：** P5 · P6 · P11 · P12 · P14 · P17 · P18
**时间：** 30小时

## 问题

自适应辅导曾经是教育科技研究的一个小众领域。到2026年，它已成为消费品。Khanmigo 已部署在美国大多数学区。Duolingo Max 达到了数千万的月活跃用户。Google 的 LearnLM/Gemini for Education 为 Google Classroom 中的辅导提供支持。Quizlet Q-Chat 与抽认卡并列存在。Synthesis Tutor 因其面向好奇孩子的辅导而迅速走红。共同要素：多模态输入（打字、说话、拍摄方程），苏格拉底式教学法（先提问，后解释），每次交互后更新的学习者模型，以及严格的适龄安全。

你将为一个特定群体构建其中一个。衡量标准是一项实际的效果研究：为期两周、10名学习者的前测和后测分数。语音回路必须感觉自然（收官项目03子栈）。记忆必须尊重隐私。安全过滤器必须通过针对K-12的COPPA意识红队测试。

## 概念

四个组件。**辅导策略**是一个苏格拉底式循环：当学习者询问答案时，策略会问一个引导性问题；当他们答对时，进入下一个概念；当他们卡住时，提供有支架的提示。**学习者模型**是贝叶斯知识追踪（或简单变体），每次交互后更新每个课程节点的掌握概率。**课程图谱**是一个带有先决条件边的Neo4j概念图；策略遍历该图以选择下一个概念。**记忆**是一个情景+语义存储（agentmemory风格），保存过去的交互、错误和偏好。

用户体验是多模态的。文本输入用于键入答案。通过LiveKit + Whisper进行语音输入（复用收官项目03）。通过dots.ocr或PaliGemma 2进行数学问题的照片输入。通过Cartesia Sonic-2进行语音输出。安全使用Llama Guard 4加一个适龄过滤器（阻止成人内容、暴力、自残）以及一个COPPA意识的记忆保留策略。

效果研究是最终成果。10名学习者，前测和后测，两周。报告学习增益差值和置信区间。与非自适应基线（相同内容以线性方式提供，无辅导策略）进行比较。

## 架构

```
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## 技术栈

- 科目选择：K-12代数或Python入门（选择一个深入）
- 辅导策略：基于Claude Sonnet 4.7的LangGraph（带提示缓存）
- 学习者模型：贝叶斯知识追踪（经典）或用于间隔重复的FSRS
- 课程图谱：Neo4j概念图 + 先决条件边 + OER内容
- 记忆：agentmemory风格的持久化向量 + 情景 + 语义存储
- 语音：LiveKit Agents 1.0 + Cartesia Sonic-2（复用收官项目03子栈）
- 拍照数学：dots.ocr或PaliGemma 2用于方程识别
- 安全：Llama Guard 4 + 自定义适龄过滤器
- 评估：布鲁姆层级问题生成、前/后测工具、效果研究工具

## 动手构建

1. **课程图谱。** 构建一个包含50-150个概念节点（例如，从“数轴”到“二次公式”的K-12代数）的Neo4j图，带有先决条件边。为每个节点附加OER内容（Open Textbook, OpenStax）。

2. **学习者模型。** 使用先验参数初始化贝叶斯知识追踪：猜测、失误、学习率。每次交互后更新每个概念的掌握程度。对每个学习者进行持久化。

3. **辅导策略。** 带节点的LangGraph：`read_signal`（学习者的答案是否正确/部分正确/卡住？），`select_concept`（遍历课程图谱选择最高优先级的概念），`scaffold`（苏格拉底式提示），`update_mastery`。

4. **记忆。** 每次交互写入情景存储。错误和偏好提升到语义记忆。COPPA意识的保留策略：1年后自动删除，家长可访问。

5. **语音路径。** 连接到辅导策略的LiveKit Agents工作器。通过Whisper-v3-turbo进行ASR。通过Cartesia Sonic-2进行TTS。支持插话（复用收官项目03机制）。

6. **拍照数学路径。** 上传或拍摄图像；运行dots.ocr或PaliGemma 2识别方程；作为结构化输入提供给辅导程序。

7. **安全。** 每个模型输出都经过Llama Guard 4 + 适龄过滤器（阻止自残、成人内容、暴力）。记忆访问按学习者ID范围限定；提供家长访问界面以供删除。

8. **效果研究。** 10名学习者，前测（标准化30题基线），两周的辅导交互（每周3次），后测。与相同内容上10名学习者的非自适应基线队列进行比较。

9. **每周进度报告。** 每位学习者，自动生成PDF摘要，包含探索的主题、掌握轨迹和推荐的下一步步骤。

## 使用它

```
learner: "I don't understand why 3x + 6 = 12 means x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "what number would you subtract from both sides to start?"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "great. now what is 3x / 3 equal to?"
```

## 发布

`outputs/skill-ai-tutor.md` 是最终成果。一个具有多模态输入、学习者模型、记忆、安全和可衡量效果的学科特定自适应辅导程序。

|  权重  |  标准  |  衡量方式  |
|:-:|---|---|
|  25  |  学习增益差值  |  10名学习者两周研究中的前/后测差值  |
|  20  |  苏格拉底忠实度  |  对话样本的评分标准得分  |
|  20  |  多模态用户体验  |  语音+照片+文本端到端一致性  |
|  20  |  安全与隐私姿态  |  Llama Guard 4通过率 + COPPA意识保留  |
|  15  |  课程广度与图谱质量  |  概念覆盖率 + 先决条件图一致性  |
|  **100**  |   |   |

## 练习

1. 分别运行带有和不带有自适应学习者模型的效果研究（随机概念顺序）。报告差值。预期自适应模型获胜，但大小是值得关注的数据。

2. 添加一个多模态探测：同一概念问题以文本、语音和照片形式呈现。衡量学习者是否在其偏爱的模态下更快掌握。

3. 构建一个家长仪表板：练习的主题、掌握轨迹、即将学习的概念、安全事件（任何防护栏命中）。符合COPPA。

4. 添加语言切换模式：辅导程序接受西班牙语输入并用西班牙语教学。衡量X-Guard覆盖率。

5. 强调记忆隐私：验证学习者A即使通过语音片段重新注入攻击也无法看到学习者B的数据。记录尝试的访问并发出警报。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
| 苏格拉底式策略  |  "提问，而非倾倒"  |  辅导程序提出引导性问题而不是给出答案  |
| 贝叶斯知识追踪(Bayesian knowledge tracing)  |  "BKT"  |  经典学习者模型方程，用于掌握每个概念的概率 |
| FSRS  |  "自由间隔重复调度器(Free Spaced Repetition Scheduler)"  |  2024年的间隔重复调度器，优于SM-2 |
| 课程图(Curriculum graph)  |  "概念有向无环图(Concept DAG)"  |  使用Neo4j存储具有先修关系边的概念 |
| 情景记忆(Episodic memory)  |  "每次交互日志(Per-interaction log)"  |  每次交互存储以供后续检索 |
| 语义记忆(Semantic memory)  |  "学习模式存储(Learned pattern store)"  |  从情景记忆中归纳压缩的错误和偏好 |
| COPPA  |  "儿童隐私法(Kids privacy law)"  |  美国限制向13岁以下儿童收集数据的法律 |

## 延伸阅读

- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) — 参考消费者K-12辅导
- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) — 参考语言学习辅导
- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) — 托管参考模型
- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) — 备用参考
- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) — 初创公司参考
- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) — 间隔重复调度器
- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) — 经典学习者模型
- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) — 语音栈
