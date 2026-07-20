# 浏览器代理与长时域Web任务

> ChatGPT Agent（2025年7月发布）将Operator和深度研究合并为一个浏览器/终端代理，并将BrowseComp的SOTA提升至68.9%。OpenAI于2025年8月31日关闭了Operator——产品层整合。Anthropic收购Vercept使Claude Sonnet在OSWorld上的成绩从不到15%提升至72.5%。WebArena-Verified（ServiceNow，ICLR 2026）修复了原始WebArena中11.3个百分点的假阴性率，并发布了258个任务的Hard子集。这些数字是真实的。攻击面也是真实的：OpenAI的预研主管公开表示，对浏览器代理的间接提示注入"不是一个可以完全修补的漏洞"。记录的2025-2026年攻击包括：Tainted Memories（Atlas CSRF）、HashJack（Cato Networks）以及在Perplexity Comet中的一键劫持。

**类型：** 学习
**编程语言：** Python（标准库，间接提示注入攻击面模型）
**前置条件：** 第15章第10节（权限模式），第15章第1节（长时域代理）
**时间：** ~45分钟

## 问题

浏览器代理是一种长时域代理，它读取不受信任的内容并执行重要的操作。代理访问的每个页面都是用户未编写的输入。每个页面上的每个表单都是潜在的命令通道。2025-2026年的攻击实例表明这并非假设：Tainted Memories允许攻击者通过精心构造的页面将恶意指令绑定到代理的内存中；HashJack将命令隐藏在代理访问的URL片段中；Perplexity Comet劫持一击即中。

防御形势令人不安。OpenAI的预研主管说出了那个显而易见的事实：间接提示注入"不是一个可以完全修补的漏洞"。这是因为攻击存在于代理的读取与执行边界之间，这在架构上是模糊的——模型读取的每个标记原则上都可能被解读为指令。

本课命名了攻击面，命名了基准测试格局（BrowseComp、OSWorld、WebArena-Verified），并模拟了一个最小的间接提示注入场景，以便你能够在第14课和第18课中推理实际的防御措施。

## 核心概念

### 2026年的格局，每个系统用一段话说明

**ChatGPT Agent（OpenAI）。** 2025年7月发布。统一了Operator（浏览）和深度研究（多小时研究）。2025年8月31日关闭了独立Operator。BrowseComp的SOTA为68.9%；在OSWorld和WebArena-Verified上表现强劲。

**Claude Sonnet + Vercept（Anthropic）。** Anthropic收购Vercept专注于计算机使用能力。使Claude Sonnet在OSWorld上从不到15%提升至72.5%。Claude Computer Use作为工具API提供。

**Gemini 3 Pro with Browser Use（DeepMind）。** Browser Use集成提供了计算机使用控制；FSF v3（2026年4月，第20课）专门跟踪ML研发领域的自主性。

**WebArena-Verified（ServiceNow，ICLR 2026）。** 修复了一个记录在案的问题：原始WebArena约有11.3%的假阴性率（标记为失败但实际上已解决的任务）。Verified版本通过人工筛选的成功标准重新评分，并添加了258个任务的Hard子集（ICLR 2026论文，openreview.net/forum?id=94tlGxmqkN）。

### BrowseComp vs OSWorld vs WebArena

|  基准测试  |  测量内容  |  时间跨度  |
|---|---|---|
|  BrowseComp  |  在时间压力下在开放网络中查找特定事实  |  分钟  |
|  OSWorld  |  代理操作完整桌面（鼠标、键盘、Shell）  |  几十分钟  |
|  WebArena-Verified  |  在模拟网站中执行事务性Web任务  |  分钟  |
|  Hard子集  |  具有多页面状态转换的WebArena-Verified任务  |  几十分钟  |

不同的维度。高BrowseComp分数表示代理能查找事实，但并不意味着它能预订航班。OSWorld分数更接近"能否在桌面工作"。WebArena-Verified更接近"能否完成流程"。任何生产决策都需要匹配任务分布的基准测试。

### 攻击面，命名

1. **间接提示注入。** 不受信任的页面内容包含指令。代理读取它们。代理执行它们。公开示例：2024年Kai Greshake等人，2025年Tainted Memories论文，2026年HashJack（Cato Networks）。
2. **URL片段/查询注入。** 爬取URL的`#fragment`或查询字符串包含命令。从未可见渲染，但仍在代理的上下文内。
3. **内存绑定攻击。** 页面指示代理写入持久内存（第12课涵盖持久状态）。下一会话中，内存在无可见触发的情况下发起载荷。
4. **对认证会话的CSRF样式攻击。** Tainted Memories类：代理在某处登录；攻击者的页面发出状态更改请求，代理使用用户cookie执行。
5. **一键劫持。** 视觉上无害的按钮携带代理跟随的载荷。Comet类。
6. **代理主机表面上的内容安全策略漏洞。** 渲染层和工具层本身可能成为攻击向量；浏览器中的浏览器代理栈非常广泛。

### 为什么"无法完全修补"

该攻击与代理的能力同构。代理必须读取不受信任的内容才能完成工作。代理读取的任何内容都可能包含指令。代理遵循的任何指令都可能与用户的实际请求不一致。防御措施（信任边界、分类器、工具白名单、对重要操作的人机交互）提高了攻击成本并减少了爆炸半径。它们并未关闭该类攻击。

这与Lob定理（第8课）的推理模式相同：代理无法证明下一个标记是安全的；它只能建立一个系统，使不安全的标记更可检测。

### 实际部署的防御策略

- **读/写边界。** 读取从不是重要的。写入（提交表单、发布内容、调用具有副作用的工具）如果发起内容来自信任边界之外，则需要重新获得人工批准。
- **每任务工具白名单。** 代理可以浏览；除非为该任务显式启用该工具，否则不能发起电汇。第13课涵盖预算。
- **会话隔离。** 浏览器代理会话仅使用限定范围的凭证运行。无生产认证，无个人电子邮件。保留每次HTTP请求的日志以供审计。
- **内容消毒器。** 获取的HTML在合并到模型上下文之前会剥离已知的不良模式。（减少简单攻击；不会阻止复杂载荷。）
- **对重要操作的人机交互。** 提议-提交模式（第15课）。
- **内存上的金丝雀令牌。** 如果内存条目触发，用户会看到它（第14课）。

## 使用它

`code/main.py`对三个合成页面的一个小型浏览器代理运行进行建模。一个页面是良性的，一个在可见文本中有直接的提示注入块，一个具有URL片段注入（不可见但在代理上下文内）。脚本显示了（a）幼稚代理会做什么，（b）读/写边界能捕获什么，（c）消毒器能捕获什么，（d）两者都无法捕获什么。

## 发布

`outputs/skill-browser-agent-trust-boundary.md`界定了一个提议的浏览器代理部署范围：它接触哪些信任区域，被授权写入什么，以及首次运行前必须部署哪些防御措施。

## 练习

1. 运行`code/main.py`。识别消毒器能捕获但读/写边界不能捕获的攻击，以及只有读/写边界能捕获的攻击。

2. 扩展消毒器以检测一类HashJack样式的URL片段注入。测量对具有合法片段的良性URL的假阳性率。

3. 选择一个你知道的真实浏览器代理工作流程（例如"预订航班"）。列出每次读取和每次写入。标记哪些写入需要人机交互并说明原因。

4. 阅读 WebArena-Verified ICLR 2026 论文。确定原始 WebArena 评分不可靠的一类任务，并解释 Verified 子集如何解决该问题。

5. 为浏览器智能体设置设计一个内存金丝雀(Canary)。你会存储什么内容、存储在何处，以及什么会触发警报？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|---|---|---|
|  间接提示注入(Indirect Prompt Injection)  |  "不良页面文本"  |  智能体读取的页面中的不可信内容包含其执行的指令  |
|  受污染记忆(Tainted Memories)  |  "内存攻击"  |  智能体将攻击者提供的指令写入持久化内存；下次会话触发  |
|  HashJack  |  "URL片段攻击"  |  隐藏在URL片段/查询字符串中的载荷存在于智能体的上下文中但未可见渲染  |
|  一键劫持(One-click Hijack)  |  "不良按钮"  |  可见的交互元素加载了智能体执行的后继载荷  |
|  BrowseComp  |  "网页搜索基准"  |  在开放网络上查找特定事实；分钟级时间跨度  |
|  OSWorld  |  "桌面基准"  |  完全操作系统控制；多步骤GUI任务  |
|  WebArena-Verified  |  "修正的网页任务基准"  |  ServiceNow 的重新评分版 WebArena 包含 Hard 子集  |
|  读/写边界(Read/Write Boundary)  |  "副作用门控"  |  读取从不产生后果；写入时若内容超出信任范围需重新审批  |

## 延伸阅读

- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — Operator 与深度研究(Deep Research)的融合；BrowseComp 当前最优水平。
- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — Operator 的血统及演变为 ChatGPT 智能体的架构。
- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — 原始基准。
- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — ICLR 2026 修正子集的论文。
- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — 包含针对计算机使用智能体的攻击面讨论。
