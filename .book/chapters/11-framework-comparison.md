## §11 框架横评：五个主流怎么选

2025 年我连换三个框架。最后发现：框架差在循环外的那一圈，不在循环本身。

### 11.1 先说结论

循环都一样（§02）。框架差在"循环外面那圈"。

标叔的选法：看你要状态、并发，还是角色。

> **标叔的经验**：别追新框架
>
> 我 2024 踩过 LangChain 的坑。后来明白：框架是循环的外壳，不是魔法。

### 11.2 五家对照

| 框架 | 内核模型 | 强项 | 标叔的结论 |
|------|--------|------|-----------|
| LangGraph | 状态图 + 检查点 | 可恢复、确定性 | 要断点续跑选它 |
| AutoGen v0.4 | Actor 模型 | 高并发消息传递 | 海量并发选它 |
| CrewAI | 角色 + 任务 | 协作草稿快 | 探索性多选它 |
| Agno | 轻量运行时 | 启动快、简单 | 小项目首选 |
| Mastra | TS 原生 | TypeScript 栈 | 前端栈选它 |

### 11.3 LangGraph：状态即一等公民

Agent 是状态机。节点是函数，边是转移，状态每步落盘（checkpoint）。

第 38 步挂了？`resume(session_id)` 从第 39 步起。Klarna、Uber、摩根大通都在用。

三种拓扑：supervisor（中央路由）、swarm（点对点交接）、hierarchical（嵌套子图）。

### 11.4 CrewAI：角色驱动

四个原语：Agent（role+goal+backstory）、Task、Crew、Process。

两种形态，文档写得很直白："生产环境，从 Flow 开始。"

- Crew：LLM 自主协作，适合调研、草稿。难回放。
- Flow：事件驱动、确定性、可测。生产用这个。

> **注意**：Hierarchical 多一个 manager LLM 调用
>
> 五步任务变六次调用，token 能翻三倍。顺序固定就别用。

### 11.5 SDK 也算框架

Claude / OpenAI Agents SDK 不是图，是"harness 即库"（§12、§13）。横评时别漏。

### 11.6 按场景推荐

- 要断点续跑、严格顺序 → LangGraph
- 要高并发、容错隔离 → AutoGen
- 要角色协作、快速草稿 → CrewAI（包 Flow）
- 要轻量、TS 栈 → Agno / Mastra

[向前桥接] 横评完。下一章，用 Claude Agent SDK 真写一个。
