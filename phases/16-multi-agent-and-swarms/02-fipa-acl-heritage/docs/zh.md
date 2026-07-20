# FIPA-ACL与言语行为之传承

> 在MCP和A2A之前，已有FIPA-ACL。2000年，IEEE智能物理代理基金会(Foundation for Intelligent Physical Agents)批准了一种包含二十个执行语(performatives)、两种内容语言以及一组交互协议（合同网、订阅/通知、请求-何时）的代理通信语言。由于本体(ontology)开销对网络过于沉重，它逐渐从工业界淡出；但LLM对多代理系统的复兴正在悄悄地重新实现同样的思想，而无需形式语义：JSON合同代替执行语，自然语言代替本体。本课将认真研读FIPA-ACL，以便你能看清2026年的哪些协议决策是重新发明，哪些是新事物，以及当前浪潮将在何处重新发现2000年代已解决的问题。

**类型：** 学习
**语言：** Python (stdlib)
**先决条件：** 阶段16 · 01 (为何多代理)
**时长：** 约60分钟

## 问题

2026年的代理协议领域热闹非凡：MCP用于工具，A2A用于代理，ACP用于企业审计，ANP用于去中心化信任，NLIP用于自然语言内容，再加上CA-MCP和二十多个研究提案。每个规范都宣称自己是基础性的。

坦诚地说，它们中的大多数都在重新发现一个非常具体的二十年老决策树。Austin（1962）和Searle（1969）的言语行为理论(Speech-act theory)给了我们'话语即行动'。KQML（1993）将其转化为有线协议。FIPA-ACL（2000年批准）产生了参考标准化：二十个执行语、内容语言SL0/SL1、用于合同网和订阅-通知的交互协议。JADE和JACK是Java参考平台。这一努力大约在2010年消退，因为本体开销太重，而网络正在获胜。

当你审视MCP的`tools/call`、A2A的任务生命周期或CA-MCP的共享上下文存储时，你看到的其实是FIPA决策的一个更柔和、原生JSON的翻版。了解传承告诉你两件事：哪些新的'创新'实际上是重新发明，以及新规范将重新发现哪些旧的失败模式。

## 概念

### 言语行为，概述

Austin注意到有些句子不是描述世界，而是改变世界。'我承诺。''我请求。''我宣布。'他称这些为执行话语(performative utterances)。Searle将其形式化为五类：断言(assertive)、指令(directive)、承诺(commissive)、表达(expressive)、宣告(declarative)。KQML（Finin等，1993）将其操作化用于软件代理：一条消息是一个执行语（行动）加上内容（行动的内容）。FIPA-ACL填补了KQML的空白，并围绕二十个执行语实现了标准化。

### 二十个FIPA执行语（部分列表）

|  执行语  |  意图  |
|---|---|
|  `inform`  |  "我告诉你P为真"  |
|  `request`  |  "我请求你做X"  |
|  `query-if`  |  "P为真吗？"  |
|  `query-ref`  |  "X的值是多少？"  |
|  `propose`  |  "我提议我们做X"  |
|  `accept-proposal`  |  "我接受提议"  |
|  `reject-proposal`  |  "我拒绝提议"  |
|  `agree`  |  "我同意做X"  |
|  `refuse`  |  "我拒绝做X"  |
|  `confirm`  |  "我确认P为真"  |
|  `disconfirm`  |  "我否认P"  |
|  `not-understood`  |  "你的消息无法解析"  |
|  `cfp`  |  "就X征求提案"  |
|  `subscribe`  |  "X变化时通知我"  |
|  `cancel`  |  "取消正在进行的X"  |
|  `failure`  |  "我尝试X但失败了"  |

完整列表在`fipa00037.pdf`（FIPA ACL消息结构）中。重点不是记住它——重点是每一个执行语都对应着LLM协议最终会重新添加的一个原语(primitive)。

### 规范FIPA-ACL消息

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

七个字段承载协议信封；一个字段（`content`）承载有效载荷。其余字段正是你每次将重试、线程和本体附加到JSON协议时重新发明的东西。

### The two legacy platforms

**JADE** (Java Agent DEvelopment framework, 1999–2020s) was the most-used FIPA-compliant runtime. Agents extended a base class, exchanged ACL messages, ran inside containers, and coordinated using "behaviors." The interaction-protocol library shipped with contract-net, subscribe-notify, request-when, and propose-accept.

**JACK** (Agent Oriented Software, commercial) emphasized BDI (Belief-Desire-Intention) reasoning on top of FIPA messages. More formal, less adopted.

Both declined once the web stack ate multi-agent use cases. MCP and A2A are the runtime "containers" of 2026.

### Why FIPA faded

- **Ontology overhead.** FIPA required a shared ontology to parse `content`. Agreeing on ontologies is a years-long standards process. The web just used HTTP + JSON.
- **Formal semantics nobody used.** SL (Semantic Language) gave rigorous truth conditions, but most production systems used free-form content and ignored the formalism.
- **Tooling lock-in.** JADE was Java-only; JACK was commercial. Polyglot teams routed around both.
- **The internet won the stack.** REST, then JSON-RPC, then gRPC replaced ACL's transport.

### The LLM revival is FIPA-lite

Compare a FIPA `request` to an MCP `tools/call`:

```
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

Same envelope, different syntax. Both carry: who, whom, intent, payload, correlation id. Neither is a revolution over the other — they are different trade-offs on the same design.

The 2025 survey by Liu et al. ("A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP", arXiv:2505.02279) makes this lineage explicit: MCP corresponds to tool-use speech acts, A2A to agent-peer speech acts, ACP to audit-trail speech acts, ANP to decentralized-identity extensions. The new specs are ACL descendants with JSON syntax and looser semantics.

### The trade-off, stated plainly

**What FIPA gave you and modern specs drop:**

- Formal semantics — you can prove `inform` implies the sender believes the content.
- A canonical catalog of performatives — you do not have to re-argue "should we have a `inform`?".
- Decades of interaction-protocol patterns — contract-net, subscribe-notify, propose-accept — with known correctness properties.

**What modern specs give you and FIPA did not:**

- JSON-native payloads compatible with every modern tool.
- Natural-language content that LLMs can interpret without a hand-coded ontology.
- Web-stack transport (HTTP, SSE, WebSocket).
- Capability discovery via self-describing documents (MCP `listTools`, A2A Agent Card).

Looser intent semantics for easier implementation. That is the exact trade.

### Interaction protocols worth porting

FIPA shipped ~15 interaction protocols. Three are worth carrying forward into LLM multi-agent systems:

1. **Contract Net Protocol (CNP).** Manager issues `cfp` (call for proposals); bidders respond with `propose`; manager accepts/rejects. This is the canonical task-market pattern (Phase 16 · 16 Negotiation).
2. **Subscribe/Notify.** Subscriber sends `cfp`; publisher sends `propose` whenever the topic changes. This is every event-bus in 2026.
3. **Request-When.** "Do X when condition Y holds." Delayed-action with pre-conditions. The 2026 analog is deferred tasks in durable workflow engines (Phase 16 · 22 Production Scaling).

Each maps cleanly onto modern message queues, HTTP + polling, or SSE streaming.

### What breaks when you drop the ontology

Without a shared ontology, agents infer meaning from natural-language content. The documented 2026 failure mode is **semantic drift**: two agents use the same word (`"customer"`) for subtly different concepts, the receiver's agent acts on the wrong interpretation, no schema validator catches it. FIPA's ontology requirement would have rejected the message at parse time.

Mitigations without going full ontology:

- JSON Schema on `content` — rejects structural errors at the wire.
- Typed artifacts (A2A) — rejects wrong modality.
- Explicit performative in the envelope — makes intent unambiguous even when content is natural language.

### The 2026 specs, mapped to speech-act heritage

|  Modern spec  |  FIPA analog  |  What it keeps  |  What it drops  |
|---|---|---|---|
|  MCP `tools/call`  |  `request`  |  explicit intent, correlation id  |  formal semantics, ontology  |
|  MCP `resources/read`  |  `query-ref`  |  explicit intent, correlation id  |  formal semantics  |
|  A2A Task lifecycle  |  contract-net + request-when  |  async lifecycle, state transitions  |  formal completeness guarantees  |
|  A2A streaming events  |  subscribe/notify  |  async push  |  typed-predicate subscription  |
| CA-MCP 共享上下文 | 黑板（Hayes-Roth 1985） | 多写入者共享内存 | 逻辑一致性模型 |
| NLIP | 自然语言内容 | LLM 原生 | 模式 |

从上到下阅读表格，模式是：保留结构原语，丢弃形式主义，让 LLM 掩盖歧义。

## 动手构建

`code/main.py` 实现了一个纯标准库的 FIPA-ACL 翻译器。它编码和解码规范的 ACL 信封，并展示每个 MCP / A2A 消息形状如何归结为相同的七个字段。演示如下：

- 将五个 MCP 风格和 A2A 风格的消息编码为 FIPA-ACL。
- 将 FIPA-ACL 解码回现代等价形式。
- 使用 `cfp`、`propose`、`accept-proposal`、`reject-proposal` 运行一个管理者和三个投标者之间的简单合同网协商。

运行：

```
python3 code/main.py
```

输出是一个并排的追踪，显示每个现代消息的 2026 JSON 形式及其 FIPA-ACL 形式，然后是一个合同网投标的往返。相同的协议原语在往返中存活；只有语法不同。

## 使用它

`outputs/skill-fipa-mapper.md` 是一种技能，它读取任何代理协议规范并产生 FIPA-ACL 映射。在采用新协议之前使用它来回答：“这是真正新颖的，还是带有 JSON 语法的 `inform`？”

## 发布

不要带回 FIPA-ACL。带回它的检查清单：

- 每条消息的意图原语（performative）是什么？
- 是否有用于请求-响应和取消的关联 ID？
- 是否有显式的内容语言（JSON-RPC、纯文本、结构化类型工件）？
- 交互协议是否是第一类公民，还是你在从头重新实现合同网？
- 当两个代理对内容含义（语义漂移）有分歧时会发生什么？

在将任何新协议投入生产之前，记录这五个问题。

## 练习

1. 运行 `code/main.py`。观察往返编码。识别哪个 FIPA 执行语对应于 `tools/call`、`resources/read` 和 A2A 任务创建。
2. 用 `code/main.py` 执行语扩展合同网演示，让管理者在投标过程中撤回任务。`tools/call` 解决了哪些重试单独无法解决的失败案例？
3. 阅读 FIPA ACL 消息结构（`code/main.py`）第 4.1–4.3 节。选择本课未涉及的一个执行语，描述其现代 JSON-RPC 模拟。
4. 阅读 Liu 等人，arXiv:2505.02279。对于 MCP、A2A、ACP、ANP 中的每一个，列出它们保留和丢弃的 FIPA 执行语家族。
5. 在自己的系统中为 `tools/call` 执行语的 `code/main.py` 字段设计一个最小的 JSON-Schema。该模式提供了纯自然语言所没有的什么，代价是什么？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
| 言语行为 | “说出的话做了某事” | Austin/Searle：话语即行动。ACL 的理论基础。 |
| FIPA | “那个过时的 XML 东西” | IEEE 智能物理代理基金会。2000年标准化了 ACL。 |
| ACL | “代理通信语言” | FIPA 的信封格式：执行语 + 内容 + 元数据。 |
| 执行语 | “动词” | 消息的意图类别：`inform`、`request`、`propose`、`cfp` 等。 |
| KQML | “FIPA 的前身” | 知识查询与操作语言（1993）。更简单，更狭隘。 |
| 本体 | “共享词汇” | 对内容语言所讨论的概念的正式定义。 |
| SL0 / SL1 | “FIPA 内容语言” | 语义语言级别 0 和 1——正式的内容语言家族。 |
| 合同网 | “任务市场” | 管理者发布 cfp；投标者提议；管理者接受。标准的交互协议。 |
| 交互协议 | “消息模式” | 具有已知正确性的执行语序列：request-when、subscribe-notify 等。 |

## 延伸阅读

- [Liu et al. — A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) — 连接现代规范与 FIPA 遗产的 2025 年标准调查
- [Liu et al. — A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) — 2000 年批准的信封格式
- [Liu et al. — A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) — 完整的执行语目录
- [Liu et al. — A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) — 对应 [FIPA ACL Message Structure Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/)/[FIPA Communicative Act Library Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) 的现代工具使用等价物
- [Liu et al. — A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) — 对应合同网和 subscribe-notify 的现代代理对等等价物
