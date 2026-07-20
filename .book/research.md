# 研究纪要：本文件夹能写成一本什么样的书

> 研究阶段产出（huashu-bookwriter · Step 2）。用于支撑 `plan.md` 的方案决策。

## 1. 这个文件夹到底是什么

这是一个**开源 AI 工程课程仓库**（`ai-engineering-from-scratch`），不是某个产品的文档。

| 维度 | 数据 |
|------|------|
| 课程阶段（Phase） | 20 个 |
| 课程（Lesson） | 503 课 |
| 可运行代码文件（`main.*`） | 483 个 |
| 产物（outputs: prompt/skill/agent/MCP） | 536 个 |
| 测验（quiz.json） | 338 个 |
| 覆盖语言 | Python / TypeScript / Rust / Julia |
| 总时长 | ~314–320 小时 |
| 许可证 | MIT，免费开源 |
| 读者规模 | README 标注 15 万+ 读者、近 30 天 24 万+ 页面浏览 |

阶段堆叠关系：数学地基（P1）→ 机器学习（P2）→ 深度学习（P3）→ 视觉/ NLP / 语音 / 强化学习（P4-9）→ Transformer（P7）→ 生成式（P8）→ 大模型从零（P10）→ 大模型工程（P11）→ 多模态（P12）→ 工具与协议（P13）→ **智能体工程（P14）** → 自治系统（P15）→ 多智能体（P16）→ 生产基础设施（P17）→ 伦理对齐（P18）→ 毕业项目（P19）。

## 2. 内容质量评估（抽样 6+ 课）

抽看了 `03-backpropagation`、`14-agent-engineering/01-the-agent-loop`、`14-26-failure-modes`、`glossary/terms.md` 等。

**结论：质量极高，且天然适配"通俗易懂 + 有深度 + 有案例"。**

- **主线清晰**：每课都走 6 拍，核心是 **Build It / Use It**——先从原始数学/stdlib 手写算法，再跑一遍生产库（如 PyTorch）。这条主线本身就是书的叙事骨架。
- **代码可运行**：抽看的 `backpropagation` 用 200 行 stdlib 实现自动求导引擎，能训练 XOR 和圆分类，注释到位。不是伪代码。
- **引用真实**：ReAct 标了 arXiv:2210.03629（ICLR 2023）；失败模式引用了 Berkeley 2025 的 MASFT（14 种多智能体失败模式）；还有 Anthropic《Building Effective Agents》等。可验证，不是编的。
- **"人话 vs 真相"表**：每课末尾有 "What people say / What it actually means" 对照表。这正是标叔风格的核心装置，可直接复用。
- **案例密集**：Agent 阶段有 LangGraph / AutoGen / CrewAI / Claude Agent SDK / OpenAI Agents SDK 的真实框架课，还有 computer-use、语音 Agent、可观测性、提示注入防御等生产级案例。

## 3. 为什么不能写"全课程百科"

用户要求"通俗易懂、有深度、有案例"。三者对篇幅的要求互相打架：

- 503 课若摊薄成一本，每课只能 1–2 页 → 只剩定义，**没有深度也没有案例**。
- 若要深度+案例，必须**聚焦一条主线**，把这条线讲透。

所以结论：**不写全集，选一条矿脉深挖。** 这既符合标叔"先让人看懂，再追求完整"的信念，也符合课程"Skip ahead if you already know the lower layers"的设计哲学。

## 4. 候选矿脉对比

| 矿脉 | 阶段 | 课数 | 通俗度 | 深度 | 案例丰富度 | 取舍 |
|------|------|------|--------|------|-----------|------|
| **智能体工程** | P14（+P11/P13 支撑） | 42 | 高（一个循环就能讲清） | 极高 | 极高（框架横评+computer-use+语音+可观测） | **首选** |
| 大模型工程 / RAG | P11 | 17 | 高 | 高 | 中高 | 次选 |
| 深度学习从零手写 | P1-3 | ~55 | 中 | 高 | 低（偏原理） | 偏教材 |
| Transformer 深潜 | P7 | 16 | 中 | 高 | 中 | 偏架构 |
| 全课程精简概览 | 全 20 阶段 | 503 | 高 | 低 | 低 | 违背"深度"要求 |

**智能体工程胜出**：它同时满足"通俗"（ReAct 循环一句话能懂）、"深度"（42 课，含原生推理、KV cache、失败模式、可观测性）、"案例"（5+ 主流框架实战 + 计算机使用 + 语音 + 生产运行时）。

## 5. 可复用资产清单（来自本文件夹）

- `phases/14-agent-engineering/**/docs/en.md`：42 课正文，含 Build It / Use It 双段、Key Terms 表、Further Reading。
- `phases/14-agent-engineering/**/code/main.py`：可运行智能体实现（stdlib 优先）。
- `phases/14-agent-engineering/**/outputs/*`：每课产出的 skill / prompt / agent（如 `skill-agent-loop.md`），可作书附赠"可复制工具"。
- `phases/11-llm-engineering/**`：RAG、function calling、eval、guardrails、MCP、prompt caching 等支撑章。
- `phases/13-tools-and-protocols/**`：工具接口、结构化输出、并行/流式工具调用。
- `glossary/terms.md`：现成的"人话 vs 真相"术语表，直接作为附录 A。
- `ROADMAP.md` / `README.md`：阶段依赖图、读者数据，作为前言与趋势附录素材。

## 6. 风险与对策

- **时效风险**：Agent 领域 2025–2026 变化快（原生推理、Responses API）。对策：所有版本相关断言都锚定课程中已出现的论文/文档日期，不凭空预测。
- **代码验证风险**：书里代码必须能跑。对策：优先直接复用文件夹内已通过测验的 `main.*`，不另起炉灶。
- **风格风险**：标叔风格要求短句≤25字、第一人称、禁用词。对策：每章走 QC 12 项 + 全书 QC 10 项（见 skill references/quality-checkpoints.md）。
