# Agent Economies, Token Incentives, Reputation

> Long-horizon autonomous agents (METR's 1-hour to 8-hour work-curve) need economic agency. The emerging **5-layer stack** is: **DePIN** (physical compute) → **Identity** (W3C DIDs + reputation capital) → **Cognition** (RAG + MCP) → **Settlement** (account abstraction) → **Governance** (Agentic DAOs). Production agent-incentive networks include **Bittensor** (TAO subnets reward task-specific models), **Fetch.ai / ASI Alliance** (ASI-1 Mini LLM + FET token), and **Gonka** (transformer-based PoW that reallocates compute to productive AI tasks). Academic work: AAMAS 2025's decentralized LaMAS uses **Shapley-value credit attribution** to fairly reward contributing agents; Google Research "Mechanism design for large language models" proposes **token auctions** with second-price payment under monotone aggregation. This lesson builds a minimal agent marketplace, applies Shapley-value credit attribution to a multi-agent pipeline, and runs a second-price token auction so the game-theory machinery lands concretely.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 16 (Negotiation and Bargaining), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 minutes

## 问题

Multi-agent systems get complicated when agents produce value jointly but need to be rewarded individually. Classical mechanisms — equal split, last-contributor-takes-all — are unfair or gameable. Coalition-based rewarding via Shapley values is fair by construction but expensive to compute. The 2025-2026 literature pushes useful approximations: Shapley sampling, monotone aggregation auctions, and on-chain reputation that accrues from confirmed contributions.

Beyond credit attribution, the field has turned to actual economic agents: Bittensor TAO rewards mining compute to fine-tune subnet-specific models, Fetch.ai/ASI rewards ASI-1 Mini LLM usage with FET tokens, Gonka reallocates transformer proof-of-work toward productive AI tasks. Agents that transact autonomously exist today; the question is how to align incentives.

This lesson treats agent economies as a specific problem family — credit attribution, mechanism design, and reputation — and builds each with the minimal math so the ideas stick.

## 概念

### The 5-layer agent-economy stack

1. **DePIN (physical compute).** Decentralized infrastructure that rents GPU, storage, bandwidth. Bittensor subnets, Render Network, Akash. Not agent-specific; agents use it.
2. **Identity.** W3C Decentralized Identifiers (DIDs) give each agent a durable ID independent of any platform. Reputation accrues to the DID. The Agent Network Protocol (ANP) uses DID as the discovery layer.
3. **Cognition.** The agent's reasoning loop: LLM + RAG + MCP. This is what the other phases build.
4. **Settlement.** Account abstraction (ERC-4337) lets agents pay gas from their own balances without holding ETH. Agents can pay for services, each other, or compute.
5. **Governance.** Agentic DAOs: governance structures where humans *and* agents vote on protocol changes, with voting power tied to reputation.

Not every production system uses all five. Bittensor uses 1, 2, partially 3, partially 4, none of 5. OpenAI agents use none except 3. The stack is a reference map, not a requirement.

### Bittensor, Fetch.ai, Gonka — what runs

**Bittensor (TAO).** Subnets are specialized tasks (language modeling, image generation, forecasting). Miners submit model outputs. Validators rank them; stake-weighted scoring distributes the TAO rewards. Each subnet has its own evaluation. The economic lesson: pay for task-specific output quality, not compute used.

**Fetch.ai / ASI Alliance.** ASI-1 Mini LLM runs on Fetch.ai's network; users pay FET tokens for inference. The agents-as-peers narrative is stronger here: an agent on Fetch can call another for a task and pay in FET.

**Gonka.** Transformer proof-of-work: the "work" is forward passes of a transformer. Miners earn by running inference tasks that have known correct outputs (from training data). Resource-productive PoW instead of hash-based PoW.

All three are production-grade as of April 2026. Payoff distribution differs. Bittensor rewards quality relative to subnet validators; Fetch rewards utility measured by paying users; Gonka rewards verifiable inference work.

### Shapley-value credit attribution

Three agents collaborate on a task. The output scores 0.8. Who contributed what?

Shapley value: the unique credit allocation satisfying four axioms (efficiency, symmetry, linearity, null). For agent `i`:

```
shapley(i) = (1/N!) * sum over all orderings O of (v(S_i_O ∪ {i}) - v(S_i_O))
```

where `S_i_O` is the set of agents before `i` in ordering `O`. In practice: enumerate all permutations, record marginal contribution of each agent in each permutation, average.

For N=3 agents, there are 6 permutations. For N=10, 3.6M — so in practice you sample orderings rather than enumerate.

### Second-price auction for aggregation

Google Research ("Mechanism design for large language models") proposes second-price token auctions for aggregating LLM outputs. Setup: N agents each propose a completion; each has a private value for being selected. The auctioneer picks the highest-value proposal and pays the *second-highest* value. Under monotone aggregation (value depends on which proposal is chosen, not how many were bid), this is truthful — agents bid their true value.

Why this matters for LLM systems: you can outsource completion tasks to multiple agents with different pricing; the auction picks the best + pays fairly, and agents have no incentive to misreport.

### Reputation capital

A DID-bound reputation score accumulates from confirmed contributions. A simple update rule:

```
rep(i, t+1) = alpha * rep(i, t) + (1 - alpha) * contribution_quality(i, t)
```

With decay factor `alpha` close to 1. Reputation:

- Is cheap to read for routing decisions ("send hard tasks to high-rep agents").
- Is expensive to forge (accumulates over time, bound to DID).
- Can be slashed: contributions that fail verification subtract.

### AAMAS 2025 decentralized LaMAS

The LaMAS proposal (AAMAS 2025) combines: DID identity, Shapley-value credit attribution, and a simple auction mechanism. The key claim: decentralizing the credit attribution step makes the system auditable and immune to single-point manipulation.

### Where the economics falls apart

- **Price oracle manipulation.** If the credit function can be gamed, agents will game it. Every mechanism needs an adversarial test.
- **Sybil attacks.** One operator spins up N fake agents to inflate their own contribution. DIDs slow but do not stop this; reputation cost-to-forge is the mitigation.
- **Verification cost.** Credit attribution is only as fair as the verifier. If verification is cheap (small LLM), it can be gamed; if expensive (human panel), the system does not scale.
- **Regulatory overhang.** Agent economies intersect with financial regulation. Bittensor, Fetch, and Gonka all operate in legal gray areas in some jurisdictions as of 2026.

### 代理经济何时有意义

- **开放网络与异构运营者。** 没有单一团队控制所有代理。
- **可验证的输出。** 没有验证，信用归属只能靠猜测。
- **长周期工作流。** 一次性任务无法从声誉积累中受益。
- **代币化支付在法律上可行** 在您的司法管辖区。

在封闭的企业系统中，经济学让位于更简单的分配（管理者分配工作，指标是内部的）。经济学文献主要适用于开放网络。

## 动手构建

`code/main.py` 实现：

- `shapley(value_fn, agents)` — 通过枚举对小型N进行精确Shapley值(Shapley value)计算。
- `shapley(value_fn, agents)` — 诚实机制；赢家支付第二高价。
- `shapley(value_fn, agents)` — 绑定DID(DID)的声誉，具有指数衰减和罚没。
- 演示1：三个代理协作，精确Shapley归属信用。
- 演示2：五个代理竞标一个任务槽；二级价格拍卖(Second-price auction)选出赢家+支付。
- 演示3：100轮任务分配给具有异质声誉的代理；声誉加权路由优于随机。

运行：

```
python3 code/main.py
```

预期输出：每个代理的Shapley值；显示诚实投标均衡的拍卖结果；声誉加权路由在预热后比随机高出10-20%的质量增益。

## 使用它

`outputs/skill-economy-designer.md`设计了一个最小代理经济：身份层、信用归属机制、支付机制、声誉规则的选择。

## 发布

在2026年运行代理经济：

- **从声誉开始，而不是代币。** 声誉实施成本低且单独有价值；代币增加了法律和经济复杂性。
- **在奖励之前先验证。** 没有独立验证步骤就不要分配信用。自我报告的质量会招致女巫攻击。
- **Shapley采样，而非精确Shapley。** 采样100-1000个排序；精确枚举无法扩展。
- **限制衰减因子和最低声誉。** 无界衰减会消灭合法贡献者；衰减过慢则奖励过时的高声誉代理。
- **对抗性审计机制。** 在开放网络之前运行红队场景。每个机制都有博弈论；你要找到漏洞，而不是攻击者。

## 练习

1. 运行`code/main.py`。确认Shapley值之和等于总价值（效率公理）。改变价值函数；Shapley分配是否按预期方向变化？
2. 实现Shapley*采样*（对K个排序的蒙特卡洛）。K如何影响近似精度？与N=4的精确结果比较。
3. 在拍卖前实现一个联盟形成步骤：代理可以合并成团队并作为一个单位投标。哪些联盟形成？结果是否帕累托优于个人投标？
4. 阅读Google Research机制设计文章。找出一个假设，如果违反，会破坏诚实性。这种失败模式在LLM环境中是什么样的？
5. 阅读AAMAS 2025去中心化LaMAS论文。在一个合成任务上对10个代理实现他们的Shapley步骤。精确计算需要多长时间？100次抽样的采样有多接近？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  DePIN  |  "去中心化物理基础设施(Decentralized physical infrastructure)"  |  代币激励的计算/存储/带宽。Bittensor、Akash、Render。  |
|  DID  |  "去中心化标识符(Decentralized identifier)"  |  可移植ID的W3C规范。代理声誉绑定到DID，而非平台。  |
|  ERC-4337  |  "账户抽象(Account abstraction)"  |  可以赞助Gas的合约账户，实现代理支付。  |
|  Shapley值(Shapley value)  |  "公平信用归属(Fair credit attribution)"  |  满足效率、对称性、线性和零元素的唯一分配。  |
|  二级价格拍卖(Second-price auction)  |  "维克里拍卖(Vickrey auction)"  |  诚实机制：赢家支付第二高价。与单调聚合兼容。  |
|  声誉资本(Reputation capital)  |  "累积质量分数(Accumulated quality score)"  |  来自已确认贡献的绑定DID的分数；随时间衰减。  |
|  代理型DAO(Agentic DAO)  |  "代理+人类治理(Agents + humans govern)"  |  以代理投票者为第一类公民的DAO，投票权与声誉挂钩。  |
|  TAO / FET / GPU credits  |  "代币面额(Token denominations)"  |  Bittensor的TAO、Fetch.ai的FET、各种DePIN代币。  |

## 延伸阅读

- [The Agent Economy](https://arxiv.org/abs/2602.14219) — 2026年五层代理经济栈调查
- [The Agent Economy](https://arxiv.org/abs/2602.14219) — 具有单调聚合的代币拍卖
- [The Agent Economy](https://arxiv.org/abs/2602.14219) — Shapley值信用归属
- [The Agent Economy](https://arxiv.org/abs/2602.14219) — 子网结构与奖励分配
- [The Agent Economy](https://arxiv.org/abs/2602.14219) — ASI-1 Mini LLM与FET代币
- [The Agent Economy](https://arxiv.org/abs/2602.14219) — 身份基础
