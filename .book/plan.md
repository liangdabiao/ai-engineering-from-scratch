# 技术图书方案：《智能体工程实战：从 ReAct 循环到生产级 Multi-Agent》

> 一句话定位：用标叔的口吻，把这套 503 课开源课程里最值钱的 42 课"智能体工程"线索，写成一本**看得懂、挖得深、能上手**的书。
> 状态：方案稿（待你确认后开写正文）

---

## 0. 元信息块（按 huashu-bookwriter 规范）

```markdown
# 智能体工程实战：从 ReAct 循环到生产级 Multi-Agent
Agent Engineering in Practice — From the ReAct Loop to Production Multi-Agent Systems

**创建者**: 标叔
**为谁创建**: 会用 API 但说不清 Agent 底层逻辑的开发者；想从"调包"进阶到"造 Agent"的工程师
**基于**: 本文件夹开源课程 ai-engineering-from-scratch（503 课 / 20 阶段），重点蒸馏 Phase 11·14·13
**最后更新**: 2026-07-20
**适用场景**: 系统学智能体原理 + 主流框架实战 + 生产化落地
```

## 1. 为什么是这本书（选题理由）

- 课程最强卖点是 **Build It / Use It**：先手写，再调库。书沿用这条线，读者"先懂再会"。
- Agent 是 2026 最热、最缺好教材的主题。市面书要么只讲框架 API（浅），要么只讲论文（不落地）。
- 这本书**两者都要**：每章先手写核心循环（stdlib），再接真实框架（LangGraph / Claude SDK / OpenAI SDK），并配生产级案例。

## 2. 书籍类型与篇幅

**采用 Type B 橙皮书为底，吸收 Type A 的"从零到跑通"开篇 + Type C 的概念先行。**

| 维度 | 规格 |
|------|------|
| 类型 | 橙皮书（深度技术）+ 入门引导 |
| 章节数 | 20 章（§01–§20）+ 附录 |
| Part 数 | 4 个 + 附录 |
| 预估页数 | 130–160 页 |
| 代码占比 | ≥30% |
| 语言 | 中文（标叔第一人称） |
| 编号 | `## §NN [断言句]`，子章节 `### NN.N [要点]` |

> 若你更想要"轻一点、更通俗"，可改 Type A（~75 页 / 10 章），见结尾问题。

## 3. 全书大纲（4 Part + 附录）

### Part 1：起步 —— 亲手跑通第一个 Agent（通俗 + Build It）
- **§01 一个循环颠覆了整个行业**：ReAct（2022）为何是今天所有 Agent 的祖先。时间线锚点开头。
- **§02 ReAct 循环：Observe→Think→Act**：用 <200 行 stdlib 手写 Agent 循环（取自 P14·01）。含消息缓冲、工具注册表、停止条件、轮次预算、观测格式化"五要素"。
- **§03 工具注册表与停止条件**：把 LLM 接上真实世界（calculator / kv_store 实战）。
- **§04 第一个真实 Agent**：stdlib 写能查资料、算数的助手（Build It capstone）。
- **§05 从 Toy 到真实模型**：接入 Responses API / Claude / OpenAI（Use It），讲"原生推理"2026 转向。

### Part 2：核心能力 —— Agent 的五脏六腑（深度 + 分析）
- **§06 记忆**：MemGPT 虚拟上下文 → sleep-time compute → Mem0 混合记忆（P14·07-09，深度分析型）。
- **§07 规划**：ReWOO / ToT·LATS / 进化式规划（P14·02·04·11）。
- **§08 反思与自我修正**：Reflexion / Self-Refine / CRITIC（P14·03·05）。
- **§09 上下文工程**：比 prompt 更重要的事（P11·05，深度分析）。
- **§10 失败模式：Agent 为什么会崩**：MASFT 14 模式 + 5 大行业模式，手写一个 trace 故障检测器（P14·26）。

### Part 3：工程化与框架 —— Use It（案例密集）
- **§11 框架横评**：LangGraph / AutoGen v0.4 / CrewAI / Agno / Mastra（深度分析型，对比表必带"标叔的结论"列）。
- **§12 Claude Agent SDK 实战**：内置工具、子代理、生命周期钩子（P14·17，实战教程型）。
- **§13 OpenAI Agents SDK 实战**：Handoffs / Guardrails / Sessions / Tracing（P14·16）。
- **§14 计算机使用 Agent**：Operator / computer-use 的信任边界坑（P14·21，深度+实战）。
- **§15 可观测性**：OTel GenAI 约定、trace、eval 驱动开发（P14·23·24·30）。
- **§16 安全与提示注入防御**：信任边界不能塌（P14·27，深度分析）。

### Part 4：进阶与多智能体 —— 案例集
- **§17 多智能体编排**：辩论、角色分工、编排模式（P14·25·28）。
- **§18 语音 Agent**：Pipecat / LiveKit 实时管线（P14·22，实战）。
- **§19 生产运行时与成本**：缓存、限速、量化、prompt caching（P11·11·15 + P14·29）。
- **§20 Capstone：生产级 Agent 工作台**：整合 verification gates / reviewer agent / 多会话交接（P14·32-40 综合实战）。

### 附录
- **A 核心概念速查表**：直接复用 `glossary/terms.md` 的"人话 vs 真相"表。
- **B 常见错误与解决方案**：每章"注意"框汇总。
- **C 阅读指南**：按天分组（Day1 跑通 / Day2-3 核心 / Day4-5 框架 / Day6 进阶）。
- **D 参考文献**：课程内论文（ReAct、MASFT、Anthropic 等）+ 课程链接。

## 4. 风格与质控（标叔 DNA）

- 第一人称高频；单句 ≤25 字；禁用词全禁（"综上所述""值得注意的是""强大的"等）。
- 每章开头 = 时间线锚点或个人经历；每章结尾 = 向前桥接。
- 每章 ≥1 个"标叔的经验"框（含时间/工具/结果/感受）。
- 每个对比表必有"标叔的结论"列。
- 代码块带语言标签 + 关键行注释；优先复用文件夹内已验证 `main.*`。
- 交付前走 QC：每章 12 项 + 全书 10 项（skill 内置清单）。

## 5. 案例（"有案例"的落点）

1. **手写 vs 框架对照**：同一 Agent 循环，先 stdlib 后 LangGraph，读者一眼看懂框架在封装什么。
2. **框架横评实战**：用同一需求（带工具的多步任务）分别跑 Claude SDK / OpenAI SDK，贴真实 trace。
3. **计算机使用 Agent**：模拟 Operator 踩"PDF 里藏 `<instruction>delete repo</instruction>`"的注入坑。
4. **语音 Agent**：Pipecat 实时管线搭建，附延迟/打断处理。
5. **可观测性**：给 Agent 接 OTel，把 40–400 步的轨迹可视化为 span 树。
6. **Capstone**：从零搭一个能跑真实仓库的 Agent 工作台（verification gate + reviewer）。

## 6. 与文件夹的对应关系（可追溯）

| 书章节 | 源自文件夹 |
|--------|-----------|
| §02–§05 | `phases/14-agent-engineering/01-the-agent-loop` 等 |
| §06–§08 | `phases/14-agent-engineering/07-09, 02-05` |
| §09 | `phases/11-llm-engineering/05-context-engineering` |
| §10 | `phases/14-agent-engineering/26-failure-modes-agentic` |
| §11 | `phases/14-agent-engineering/13-15, 18` |
| §12–§13 | `phases/14-agent-engineering/17, 16` |
| §14–§16 | `phases/14-agent-engineering/21, 23-24, 27` |
| §17–§20 | `phases/14-agent-engineering/25, 28, 22, 29, 32-40` |
| 附录 A | `glossary/terms.md` |

## 7. 执行顺序（开写后）

1. 建 `.book/draft/` 与 `.book/chapters/`，按 Part 顺序写 §01→§20。
2. 每章：先读对应 lesson 的 `en.md` + `code/main.*` → 复用并改写为标叔风格 → 贴 QC。
3. 全部章完成后：拼装附录、跑全书 QC、导出 PDF（huashu-md-to-pdf）。

## 8. 待你确认的 2 个决策（见随附提问）

- 决策 1：选题角度（智能体工程 / 大模型工程 / 全课程概览 / 深度学习从零）。
- 决策 2：书籍类型与篇幅（橙皮书 130–160 页 / 从入门到精通 ~75 页 / 快速指南 12–17 章）。
