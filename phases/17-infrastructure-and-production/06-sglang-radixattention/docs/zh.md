# SGLang 与 RadixAttention：面向前缀密集型工作负载

> SGLang 将 KV 缓存视为存储于基数树(Radix Tree)中的一等可复用资源。当 vLLM 按先到先服务(FCFS)调度请求时，SGLang 的缓存感知调度器优先处理具有更长共享前缀的请求——实质上是一种深度优先的基数遍历，使得热点分支常驻 HBM。在 Llama 3.1 8B 及类 ShareGPT 的 1K 提示词测试中，SGLang 达到约 16,200 tok/s，vLLM 约 12,500 tok/s，性能优势约 29%。对于前缀密集的 RAG 工作负载，优势可达 6.4 倍。在语音克隆类工作负载中，缓存命中率超过 86%。2026 年已部署于 xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS 等平台的 40 万+ GPU 上。需要注意的是：当前缀顺序不一致时，6.4 倍的优势会消失——顺序是工程师的杠杆。

**类型：** 学习
**语言：** Python (标准库，玩具级基数树缓存 + 缓存感知调度器)
**前置条件：** 第 17 阶段·04 (vLLM 服务内部原理)，第 14 阶段 (智能体 RAG)
**时间：** 约 75 分钟

## 学习目标

- 图解 RadixAttention：前缀如何存储在基数树中，以及 KV 块如何在同一分支的序列间共享。
- 解释缓存感知调度，以及为何 FCFS 不适用于前缀密集流量。
- 根据前缀缓存命中率和提示词长度分布，计算工作负载的预期加速比。
- 指明使 6.4 倍数字真实存在的提示词排序纪律，否则优势将丧失。

## 问题

传统服务将每个请求的提示词视为不透明。即使 5,000 个 RAG 请求都以相同的 2,000 token 系统提示词加上相同的检索前导开头，vLLM 仍会填充该 2,000 token 前缀 5,000 次。GPU 反复执行相同的工作。

观察结果：智能体和 RAG 工作负载中的提示词几乎总是共享长前缀。系统提示词、工具模式、少样本示例、检索头、对话历史——所有这些都在请求间重复。如果将该前缀的 KV 缓存存储一次并复用，则无需再次填充。

RadixAttention 正是如此。Token 在基数树中被索引；每个节点拥有从其根路径上 token 序列的 KV 块。新请求遍历该树：任何 token 匹配的节点复用该节点的 KV 块。填充成本仅与新后缀成比例，而非整个提示词。

挑战在于调度。如果两个请求共享一个 2,000 token 的前缀，而第三个请求仅共享同一前缀的 200 个 token，则希望将两个长共享请求一起服务，以使长前缀留在 HBM 中。FCFS 则相反——它先服务先到的请求，可能在下一个长前缀请求到达前就将热点分支驱逐。

## 核心概念

### 作为 KV 索引的基数树

基数树（紧凑字典树）存储 token 序列。每个节点拥有一个 token 范围以及为该范围计算的 KV 块。子节点扩展一个或多个 token。

```
root
 |- "You are a helpful assistant..."  (2,000 tokens, 124 KV blocks)
      |- "Context: <doc A>..."        (500 tokens, 31 blocks)
           |- "Question: Alice..."    (80 tokens, 5 blocks)
           |- "Question: Bob..."      (95 tokens, 6 blocks)
      |- "Context: <doc B>..."        (520 tokens, 33 blocks)
```

新请求包含系统提示词 + "上下文：<文档 A>" + "问题：Carol"。调度器遍历：系统前缀匹配（复用 124 块），文档 A 分支匹配（复用 31 块），然后仅为 "问题：Carol" 分配新块（4 块）。填充成本：4 块新 token。无树时：160 块。填充节省约 40 倍。

### 缓存感知调度

如果缓存频繁抖动，基于基数树的复用便无意义。两个关键策略：

1. **深度优先分发**。从队列中选择下一个请求时，优先选择与当前运行集合在同一分支的请求。这使热点分支保持常驻。
2. **分支级别 LRU，非块级别**。驱逐整个分支（从最短使用的叶子开始），而非单个块，以使缓存形状与基数树形状匹配。

FCFS 违反了这两条。一个共享 2,000 token 的请求排在共享 50 token 的请求之后，然后 2,000 token 分支被驱逐以接纳 50 token 请求。

### 应记忆的基准测试数字

- Llama 3.1 8B，H100，ShareGPT 1K 提示词：SGLang 约 16,200 tok/s vs vLLM 约 12,500 tok/s（约 29% 优势）。
- 前缀密集 RAG（相同系统 + 相同文档，变体问题）：SGLang 最高达 6.4 倍。
- 语音克隆工作负载：86.4% 前缀缓存命中率。
- SGLang 客户的生产命中率：50-99%，取决于提示词纪律。
- 2026 年已部署于 40 万+ GPU。

### 顺序陷阱

6.4 倍数字依赖于一致的提示词模板顺序。如果你的客户端在某些请求中构造提示词为 `[system, tools, context, history, question]`，在另一些中为 `[system, context, tools, history, question]`，则树无法找到共享前缀。人类看起来共享的前缀，对基数树而言是两个不同的序列。

工程师的杠杆：你的提示词模板就是一个缓存键。固定顺序。将所有不可变内容（系统、工具、模式）放在前面。检索上下文放在其后。用户问题放在最后。不要将动态内容穿插到前缀中。

研究中的真实案例：将动态内容移出可缓存前缀，使一次部署的缓存命中率从 7% 提升到 74%。

### RadixAttention 的胜负场景

胜：
- RAG（相同检索前导，变体问题）。
- 智能体（相同工具模式，变体查询）。
- 具有长系统提示词的聊天。
- 具有重复前导的语音/视觉工作负载。

败（恢复为 vLLM 级吞吐量）：
- 具有唯一提示词的单次生成（代码补全、无系统提示词的开放式聊天）。
- 每个请求在前缀中交错唯一内容的动态提示词。

### 为何这是调度问题，而不仅仅是内核问题

你可以将 KV 复用实现为内核技巧。SGLang 的洞察在于：只有调度器保持热点分支常驻，复用才有意义。"有就复用"的朴素策略会在混合负载下导致缓存抖动。基于基数树索引的调度器，才能将内核技巧转化为 29% 的生产优势。

### 与 vLLM 的相互作用

这两个系统并非严格竞争。2026 年，vLLM 增加了前缀缓存（`--enable-prefix-caching`）和一个缓存感知路由器（Rust 编写的 vLLM Router）。差距缩小但未完全消失——SGLang 的整个栈以基数树优先；vLLM 是嫁接的。对于以前缀复用为主的工作负载，SGLang 仍是默认选择。对于无强前缀模式的通用服务，vLLM 仍持平或更优。

```figure
roofline
```

## 使用它

`code/main.py` 实现了一个玩具级基数树(Radix-Tree)KV缓存，以及一个带有两种策略的调度器：FCFS(先来先服务)和缓存感知(Cache-Aware)。它在同一个工作负载上运行两种策略，报告前缀缓存命中率(Prefix-Cache Hit Rate)和吞吐量差值(Throughput Delta)。然后运行一个“乱序”(Scrambled Ordering)工作负载以展示6.4倍的性能崩溃。

## 发布

本课产生 `outputs/skill-radix-scheduler-advisor.md`。给定工作负载描述（提示模板形状(Prompt-Template Shape)、检索模式(Retrieval Pattern)、并发租户数量(Number of Concurrent Tenants)），它生成一个提示排序建议(Prompt-Ordering Prescription)以及是否采用SGLang的决策(Go/No-Go)。

## 练习

1. 运行 `code/main.py`。比较FCFS和缓存感知在相同工作负载下的表现。差值来自哪里——预填充节省(Prefill Savings)、解码节省(Decode Savings)还是队列延迟(Queue Delay)?
2. 修改工作负载，使提示随机排列 `code/main.py`。重新运行。命中率发生了什么变化？为什么？
3. 计算在Llama 3.1 8B上将一条2000令牌(Token)的系统提示作为基数树的一个分支常驻时的高带宽内存成本(HBM Cost)。与不使用前缀复用的16序列批处理(16-Sequence Batch)的成本进行比较。
4. 阅读SGLang的RadixAttention论文。用三句话解释为什么在前缀繁重负载(Prefix-Heavy Load)下树形LRU淘汰(Tree-Shaped LRU Eviction)优于块形LRU淘汰(Block-Shaped LRU Eviction)。
5. 一位客户报告只有8%的缓存命中率。列举三个可能的原因以及每个原因您会运行的诊断方法。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  RadixAttention  |  “SGLang那玩意儿”  |  KV缓存以基数树索引，使共享前缀能复用块  |
|  基数树  |  “紧凑特里树”  |  每个节点拥有一个令牌范围及其KV块的树  |
|  缓存感知调度器  |  “热分支优先”  |  偏好共享常驻分支的请求的调度器  |
|  前缀缓存命中率  |  “你的提示有多少是免费的”  |  从复用KV块获取的提示令牌的比例  |
|  FCFS  |  “先来先服务”  |  破坏前缀局部性(Prefix Locality)的默认调度  |
|  分支级LRU  |  “淘汰叶子”  |  匹配基数树形状的淘汰策略  |
|  提示模板排序  |  “缓存键”  |  提示的组件顺序决定了树可以共享什么  |
|  系统提示固定  |  “常驻前缀”  |  将不可变的系统部分固定以避免淘汰抖动  |

## 延伸阅读

- [SGLang GitHub](https://github.com/sgl-project/sglang) — 源代码和文档。
- [SGLang GitHub](https://github.com/sgl-project/sglang) — RadixAttention和调度细节。
- [SGLang GitHub](https://github.com/sgl-project/sglang) — 设计参考。
- [SGLang GitHub](https://github.com/sgl-project/sglang) — 基准测试数据和调度器原理。
- [SGLang GitHub](https://github.com/sgl-project/sglang) — vLLM自己的类基数实现，供比较。
