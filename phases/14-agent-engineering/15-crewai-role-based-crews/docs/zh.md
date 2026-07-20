# CrewAI：基于角色的团队与流程（Crews and Flows）

> CrewAI 是2026年基于角色的多智能体框架。四个基本概念（Primitives）：智能体（Agent）、任务（Task）、团队（Crew）、流程（Process）。两种顶层形态：团队（Crews，自主的、基于角色的协作）和流程（Flows，事件驱动的、确定性的）。文档直言：“对于任何生产就绪的应用，请从流程（Flow）开始。”

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置知识：** 阶段14·12（工作流模式），阶段14·14（Actor模型）
**时间：** 约75分钟

## 学习目标

- 说出CrewAI的四个基本概念（智能体、任务、团队、流程）及其各自拥有的内容。
- 区分顺序（Sequential）、层级（Hierarchical）和计划中的共识（Consensus）流程；为每种工作负载选择一种。
- 区分团队（自主的、基于角色的）与流程（事件驱动的、确定性的），并解释文档中的生产环境推荐。
- 使用`@tool`装饰器和`BaseTool`子类集成工具；推理结构化输出与自由文本的区别。
- 说出CrewAI的四种记忆类型及其各自适用的场景。
- 实现一个由三个智能体（研究员、写手、编辑）组成的标准库团队，生成一份简报。
- 识别CrewAI的三种失败模式：提示膨胀（prompt-bloat）、管理者LLM税（manager-LLM tax）、脆弱的交接（brittle handoffs）。

## 问题

采用多智能体框架的团队常会遇到相同的困境。“自主协作”在演示中听起来很棒。但客户提交了一个bug，你需要确定性重放。或者财务部门想知道一个由LLM路由的团队每次运行的成本。或者值班人员需要知道凌晨3点哪个智能体卡住了。

自由形式的LLM路由团队无法清晰回答以上任何问题。纯DAG可以回答所有问题，但失去了头脑风暴智能体所需的探索形态。

CrewAI的拆分诚实地反映了这种权衡。团队（Crews）用于协作的、基于角色的、探索性的工作。流程（Flows）用于事件驱动的、代码拥有的、可审计的生产环境。同一个框架，两种形态，根据场景选择。

## 核心概念

### 四个基本概念

CrewAI的表面很小。记住这些，其余都是配置。

- **智能体（Agent）。** `role + goal + backstory + tools + (optional) llm`。背景故事（backstory）是承载性的。它塑造了语气、判断力以及智能体何时停止。工具（Tools）是智能体可以调用的函数（见下文）。
- **任务（Task）。** `role + goal + backstory + tools + (optional) llm`。一个可重用的工作单元。`description + expected_output + agent + (optional) context + (optional) output_pydantic`是合约。`expected_output`列出了上游任务，其输出会被传入。`context`强制结构化形状。
- **团队（Crew）。** 容器。拥有`role + goal + backstory + tools + (optional) llm`列表、`description + expected_output + agent + (optional) context + (optional) output_pydantic`列表、`expected_output`以及可选的`context` + `output_pydantic` + `agents`设置。
- **流程（Process）。** 执行策略。顺序（Sequential）、层级（Hierarchical）、共识（Consensus，计划中）。选择运行的形态。

智能体不直接相互查看。任务引用智能体。团队（Crew）对任务进行排序。流程（Process）决定谁选择下一个任务。这就是整个心智模型。

> **验证于** CrewAI 0.86（2026年5月）。较新版本可能重命名或合并流程类型；依赖特定形态前请检查[CrewAI Processes docs](https://docs.crewai.com/concepts/processes)。

### 顺序（Sequential）vs 层级（Hierarchical）vs 共识（Consensus）

- **顺序（Sequential）。** 任务按声明顺序执行。任务N的输出作为`context`传递给任务N+1。成本最低。最可预测。当顺序固定时使用。
- **层级（Hierarchical）。** 一个管理者智能体（Agent，独立的LLM调用）在专家之间进行路由。CrewAI从你的`context`配置或默认值中生成管理者。管理者每轮选择下一个任务，可以拒绝或重新路由。当你有四个或更多专家，且顺序确实依赖于先前输出时使用。
- **共识（Consensus）。** 计划中，目前在公共API中未实现。文档保留此名称用于未来基于投票的流程。今天请勿依赖它。

层级（Hierarchical）在每个专家调用的基础上增加了一轮LLM调用（管理者）。对于五步运行，令牌成本可能增加两倍。仅在需要路由时才为此付费。

### 团队（Crews）vs 流程（Flows）

这是文档在2026年首推的框架。

- **团队（Crew）。** LLM驱动的自主性。框架在运行时选择形态。适用于：研究、头脑风暴、初稿，任何路径本身是答案一部分的场景。难以重放。难以测试。原型成本低。
- **流程（Flow）。** 由你拥有的事件驱动图。`@start`标记入口。`@listen(topic)`标记一个步骤，当另一个步骤发出该主题时触发。每个步骤是纯Python（可以在内部调用团队（Crew））。适用于：生产环境。可观察。可测试。确定性。

文档的2026年生产环境推荐：从流程（Flow）开始。在自主性值得其成本时，将团队（Crews）作为流程步骤内部的`Crew.kickoff()`调用嵌入。流程给你审计追踪，团队给你探索。组合，而非选择。

### 工具集成

三种方式为智能体（Agent）提供工具。选择最适合的简单方案。

1. **`@tool`装饰器。** 纯函数变为工具。签名是模式；文档字符串是LLM看到的描述。最适合一次性辅助工具。

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool`子类。** 基于类的工具，具有显式参数模式、异步支持、重试。当工具有状态（客户端、缓存）或需要结构化参数时使用。

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **内置工具包。** CrewAI提供第一方适配器：`SerperDevTool`、`FileReadTool`、`DirectoryReadTool`、`CodeInterpreterTool`、`RagTool`、`WebsiteSearchTool`。通过一次导入即可连接。

结构化输出使用Pydantic。在任务上传递`output_pydantic=MyModel`。CrewAI根据模型验证LLM响应，并强制或重试。将此与紧凑的`expected_output`字符串配对。自由文本输出适用于草稿；结构化输出是下游流程（Flows）可以消费的。

### 记忆钩子（Memory hooks）

CrewAI开箱即用提供四种记忆类型。它们可以组合：一个团队（Crew）可以同时启用所有四种。

> **验证于** CrewAI 0.86（2026年5月）。最近的版本将所有内容通过统一的`Memory`系统路由，该系统封装了这四种存储。以下概念模型仍然成立，但公共类表面可能在新版本中坍缩为单个`Memory`入口点；请检查[CrewAI memory docs](https://docs.crewai.com/concepts/memory)以获取当前API。

- **短期记忆（Short-term）。** 单次运行中的对话缓冲区。运行结束时清除。
- **长期记忆（Long-term）。** 跨运行持久化。存储在向量数据库（默认Chroma，可替换）中。通过与当前任务的相似性检索。
- **实体记忆（Entity）。** 每个实体的事实。“客户X是企业版。”按实体键检索，而非相似性。跨运行持久化。
- **上下文记忆（Contextual）。** 组装时检索。在智能体（Agent）需要时拉取相关记忆，而非预加载。

在团队（Crew）上通过`memory=True`或按类型配置启用。由你配置的嵌入提供者（默认OpenAI，可替换为本地）支持。记忆（Memory）是CrewAI相对于更轻量框架的优势之一；纯LangGraph需要你自己连接每一个。

### CrewAI 适用场景

- 三到六个具有命名角色和协作工作流的智能体。起草、审查、规划、头脑风暴。
- 路由中，LLM关于下一步的判断本身就是价值的一部分（层级式）。
- 任何团队更愿意阅读 `role + goal + backstory` 而非图形定义的场景。

### CrewAI 不适用场景

- 具有严格顺序的确定性 DAG。请使用 LangGraph（第13课）。图形形状是合适的抽象；CrewAI 的角色框架反而造成摩擦。
- 亚秒级延迟预算。层级式增加了往返次数。即使是顺序式也会序列化包含背景故事和先前输出的提示。
- 单智能体循环。跳过框架；智能体循环（第1课）加上工具注册表更简洁。

第17课（智能体框架权衡）以矩阵形式阐述了这一点。简言之：CrewAI 位于“协作角色型”角落。

### 依赖形态

独立于 LangChain。Python 3.10 至 3.13。使用 `uv`。星标数量：参见 [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)（截至2026年5月的快照）。AWS Bedrock 集成有文档记载；供应商基准测试报告称，在问答工作负载上比 LangGraph 有显著加速，但方法（数据集、硬件、评估指标）未公开，因此框架供应商的数据仅作方向性参考。

### 这种模式出错的地方

- **背景故事导致的提示膨胀。** 每个智能体2000词的背景故事，五个智能体的团队在第一次工具调用前就消耗了上下文预算。将背景故事控制在200词以内。在不同智能体间复用短语；不要重复五次相同的写作风格。
- **管理层LLM令牌税。** 层级式流程在每次专业调用前增加一次管理层LLM调用。在五个任务的团队中，就是六次LLM调用而非五次，且管理层调用携带完整任务列表和先前输出。除非路由依赖于输出，否则切换为顺序式。
- **脆弱的交接。** 任务N的 `expected_output` 是“一个大纲”。任务N+1将其读作 `context` 并尝试解析三个部分。LLM生成了四个部分。下游智能体即兴发挥。解决方案：在任务N上使用 `output_pydantic`，以便任务N+1读取类型化对象，而非自由文本。
- **将团队视为生产环境。** 未经 Flow 包装就交付自由形式的团队至生产环境。输出变异性高；无法重放；值班人员无法区分不良运行与良好运行。使用 Flow 包装。

## 动手构建

`code/main.py` 实现了两种形态的标准库版本以及一个三智能体团队。

形态：

- `Agent`、`Task` 数据类，匹配 CrewAI 的表面。
- `Agent` 按声明顺序运行任务，将输出作为 `Task` 传递。
- `Agent` 增加一个管理层智能体，每轮选择下一个专业智能体，直至“完成”时停止。
- `Agent` 搭配 `Task` 和 `SequentialCrew.kickoff(inputs)` 装饰器、一个小型事件循环和一个跟踪。
- `Agent` 装饰器，镜像 CrewAI 的 `Task` 形态。
- `Agent` 搭配 `Task`、`SequentialCrew.kickoff(inputs)`、`context` 存储；模拟相似度使用 numpy。
- 模拟 LLM 响应是基于角色加输入前缀的硬编码字符串。无网络。确定性。

具体演示：研究员、写手、编辑团队制作关于“2026年智能体工程”的简报。研究员拉取（模拟的）资料。写手起草。编辑精炼。同一团队通过 Flow 运行，展示确定性形态。

运行它：

```bash
python3 code/main.py
```

跟踪涵盖：顺序式团队通过 `context` 传递输出；层级式团队由管理层选择（研究员、写手、编辑，然后“完成”）；Flow 运行相同三步，明确主题（`researched`、`drafted`、`edited`）；工具调用通过 `@tool` 路由；长期记忆跨越两次启动持续存在。

团队跟踪是流动的；管理层原则上可以重新排序。Flow 跟踪是固定的。这个选择就是本课的要点。

## 使用它

- **CrewAI Flow** 用于生产环境。即使 Flow 仅包含一个调用 `Crew.kickoff()` 的步骤。Flow 提供了审计边界。
- **CrewAI Crew（顺序式）** 用于清晰排序的协作工作，尤其是初稿和审阅循环。
- **CrewAI Crew（层级式）** 当路由依赖于输出且你有四个或更多专业智能体时。
- **LangGraph**（第13课）用于显式状态机、持久恢复、严格排序。
- **AutoGen v0.4**（第14课）用于参与者模型并发和故障隔离。
- **OpenAI Agents SDK**（第16课）用于以OpenAI为先的产品，支持交接和护栏。
- **Claude Agent SDK**（第17课）用于以Claude为先的产品，支持子智能体和会话存储。

## 发布

`outputs/skill-crew-or-flow.md` 为任务选择 Crew 还是 Flow，并搭建最小实现。硬性拒绝：无背景故事的 Crew、无明确主题的 Flow、专业智能体少于三个的层级式。

## 陷阱

- **背景故事作为调味料。** 它塑造输出。每个智能体测试三个变体；差异是真实的。选择一个并固定下来。
- **跳过 `expected_output`。** 每个任务没有契约，下游任务会接收 LLM 产生的任何内容。团队运行了；审计失败。
- **始终开启记忆。** 长期记忆每次运行都写入。向量数据库增长。检索变得嘈杂。将写入范围限定在事实持久存在的任务。
- **管理层提示漂移。** 层级式的管理层提示是隐式的。如果路由变得奇怪，以详细模式转储并阅读。
- **团队中的工具副作用。** 团队可能比预期更频繁地调用工具。POST、DELETE、支付操作属于 Flow 步骤，绝不属于团队工具。

## 练习

1. 将顺序式团队转换为 Flow。统计变异性下降的接触点。注意可读性下降的地方。
2. 为团队添加实体记忆：关于客户的事实跨越多次启动持续存在。验证检索拉取正确的实体。
3. 实现层级式流程，其中管理层拒绝将任务路由给编辑，直到写手的输出至少包含三个段落。追踪重试。
4. 为（模拟的）网络搜索接入一个 `BaseTool` 子类。比较跟踪形态与 `@tool` 装饰器版本。
5. 向编辑任务添加 `BaseTool`，其中 `@tool` 包含 `output_pydantic=Brief`、`Brief`、`title`。让写手任务输出一次畸形的 JSON；在跟踪中验证 CrewAI 的重试行为。
6. 阅读 CrewAI 的文档介绍。将玩具代码移植到真实的 `BaseTool` API。标准库版本跳过了哪些保证？
7. 接入 AgentOps 或 Langfuse（第24课）进行一次实际运行。在标准库版本中你错过了哪些跟踪？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  智能体  |  "角色"  |  角色 + 目标 + 背景故事 + 工具  |
|  任务  |  "工作单元"  |  描述 + 预期输出 + 分配对象 + 可选的输出结构  |
|  团队  |  "智能体团队"  |  智能体 + 任务 + 流程的容器  |
|  流程  |  "执行策略"  |  顺序式 / 层级式 / 共识（计划中）  |
|  Flow  |  "确定性工作流"  |  事件驱动、代码拥有、可测试  |
|  背景故事  |  "角色提示"  |  智能体的语气和判断塑造器  |
|  `@tool`  |  "函数工具"  |  将函数转化为智能体可调用工具的装饰器  |
|  `BaseTool`  |  "类工具"  |  基于类的工具，包含参数模式、重试、异步支持  |
|  实体记忆  |  "每个实体的事实"  |  限定于客户/账户/问题的记忆  |
|  长期记忆  |  "跨运行记忆"  |  向量支持的记忆，在启动之间持续存在  |
|  上下文记忆  |  "即时检索"  |  在智能体需要时拉取的记忆  |
|  管理层 LLM  |  "路由智能体"  |  层级式流程中额外的 LLM，负责选择下一个任务  |
|  `expected_output`  |  "任务契约"  |  告知Agent（以及审计）应返回何种形状的字符串 |

## 延伸阅读

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction)：概念与推荐的生产路径
- [CrewAI docs introduction](https://docs.crewai.com/en/introduction)：事件驱动形状，[CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows)，`@start`
- [CrewAI docs introduction](https://docs.crewai.com/en/introduction)：[CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows)，`@start`，内置工具包
- [CrewAI docs introduction](https://docs.crewai.com/en/introduction)：短期、长期、实体、上下文
- [CrewAI docs introduction](https://docs.crewai.com/en/introduction)：多代理何时有帮助以及何时无帮助
- [CrewAI docs introduction](https://docs.crewai.com/en/introduction)：状态机替代方案
