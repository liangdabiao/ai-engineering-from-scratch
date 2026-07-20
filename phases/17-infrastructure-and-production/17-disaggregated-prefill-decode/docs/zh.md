# 分离式预填充/解码 — NVIDIA Dynamo 与 llm-d

> 预填充是计算密集型；解码是内存密集型。在同一GPU上同时运行两者会浪费一种资源。分离式架构将其分配到不同的池中，并通过NIXL（RDMA/InfiniBand或TCP回退）在它们之间传输KV缓存。NVIDIA Dynamo（GTC 2025宣布，1.0 GA）位于vLLM/SGLang/TRT-LLM之上——其规划器分析器+ SLA规划器自动匹配预填充与解码的比例以满足SLO。NVIDIA发布了这一范围内的吞吐量提升——developer.nvidia.com (2025-06)显示，DeepSeek-R1 MoE在GB200 NVL72 + Dynamo上，在中延迟场景下提升了约6倍，而Dynamo产品页面（developer.nvidia.com，无日期）宣称GB300 NVL72 + Dynamo相比Hopper的MoE吞吐量提升高达50倍。"30x"这个数字是社区对全栈Blackwell + Dynamo + DeepSeek-R1报告的汇总；我们未找到任何单一主要来源明确声称30倍，因此应将其视为方向性说法。llm-d（Red Hat + AWS）是Kubernetes原生：预填充/解码/路由器作为独立服务，每个角色具有HPA。llm-d 0.5增加了分层KV卸载、缓存感知LoRA路由、UCCL网络、缩放到零。经济性：多个客户披露的内部汇总表明，当从共置服务切换到使用Dynamo的分离式服务且SLA不变时，可节省30-40%（$2M-class inference spend (i.e., $600-800K/年）；具体的$2M→$600-800K数字是内部综合数据，并非单一已发布的案例研究——应将其视为数量级参考，而不是参考文献引用。短提示（<512个令牌，短输出）无法证明传输成本合理。

**类型：** 学习
**语言：** Python (stdlib, 玩具分离式vs共置模拟器)
**前置条件：** 阶段 17 · 04 (vLLM服务内部), 阶段 17 · 08 (推理度量)
**时间：** ~75分钟

## 学习目标

- 解释为什么预填充和解码有不同的最佳GPU分配，并量化共置下的浪费。
- 绘制分离式架构图：预填充池、解码池、通过NIXL的KV传输、路由器。
- 指出分离式架构不划算的条件（短提示、短输出）。
- 区分NVIDIA Dynamo（栈上层）和llm-d（Kubernetes原生），并将每个匹配到操作场景。

## 问题

你在8个H100上运行Llama 3.3 70B。在混合工作负载下（长提示+短输出），GPU在解码期间闲置，因为大部分计算都花费在了预填充上。在不同的工作负载下（短提示+长输出），则相反。共置的预填充+解码意味着你过度配置了两者。

预算影响：20-40%的GPU时间浪费在了错误的资源上。你购买H100的计算能力来运行内存密集型的解码，或者购买H100的HBM带宽来运行计算密集型的预填充。两者都是昂贵的浪费。

分离式架构将预填充和解码分离到各自瓶颈优化的独立池中。KV缓存通过高带宽互连从预填充池传输到解码池。

## 核心概念

### 为何瓶颈不同

**预填充** — 在单次前向传播中在整个输入提示上运行transformer。矩阵乘法占主导；计算密集型。H100 FP8提供约2000 TFLOPS的有效吞吐量。批处理效率高——一次前向处理许多令牌。

**解码** — 一次生成一个令牌，每次迭代读取全部权重。内存带宽受限。HBM3提供约3 TB/s。批处理效率仅在高并发下良好——权重读取在批次中摊销。

将它们共置：你购买为两者优化的GPU。H100两者都擅长，但无论哪种方式成本相同。在规模上，你希望预填充池使用H100（计算密集型）；解码池使用H200（内存密集型），或使用激进量化。

### 架构

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL是NVIDIA的节点间传输。当可用时使用RDMA/InfiniBand，否则使用TCP回退。传输延迟是真实的——对于70B FP8上4K令牌提示的KV缓存，通常为20-80毫秒。这就是为什么短提示不证明分离式架构合理：传输成本超过了节省。

### Dynamo 与 llm-d

**NVIDIA Dynamo** (GTC 2025宣布，1.0 GA)：
- 位于vLLM、SGLang、TRT-LLM之上，作为编排器。
- 规划器分析器测量工作负载，SLA规划器自动配置预填充：解码比例。
- Rust核心，Python可扩展性。
- 吞吐量提升：NVIDIA报告DeepSeek-R1 MoE在GB200 NVL72 + Dynamo上，中延迟场景下提升6倍（developer.nvidia.com, 2025-06）；社区报告称"高达30倍"的全栈Blackwell + Dynamo + DeepSeek-R1缺乏单一主要来源，应视为方向性数据。
- GB300 NVL72 + Dynamo：根据Dynamo产品页面（developer.nvidia.com，无日期），相比Hopper的MoE吞吐量提升高达50倍。

**llm-d** (Red Hat + AWS, Kubernetes原生)：
- 预填充/解码/路由器作为独立的Kubernetes服务。
- 每个角色具有基于队列深度（预填充）和KV利用率（解码）信号的HPA。
- `topologyConstraint packDomain: rack`将预填充+解码集群打包在同一机架上，以实现高带宽KV传输。
- llm-d 0.5 (2026)：分层KV卸载、缓存感知LoRA路由、UCCL网络、缩放到零。

如果你想要一个托管式的栈上层编排器，请使用Dynamo。如果你想要Kubernetes原生原语并致力于CNCF生态系统，请使用llm-d。

### 经济性

内部综合数据（并非单一已发布的案例研究——数量级参考）：

- 每年200万美元的推理支出用于共置服务。
- 切换到使用Dynamo的分离式架构。
- 相同的请求量，相同的P99延迟SLA。
- 报告节省：$600K–$80万美元/年（减少30-40%）。
- 无需新硬件。

我们综合了多个客户披露的数据，而非单一可引用案例研究；最接近的已发布数据点是Baseten使用Dynamo KV路由实现TTFT快2倍/吞吐量高61%（baseten.co, 2025-10），以及VAST + CoreWeave预测在40-60% KV命中率下每个美元获得的令牌数增加60-130%（vastdata.com, 2025-12）。节省来自于为每个池合理调整规模；预填充密集型工作负载（RAG带8K+前缀）比平衡型工作负载受益更多。

### 何时不宜分离

- 提示<512个令牌且输出<200个令牌：传输成本超过收益。
- 小型集群（<4个GPU）：没有足够的池多样性。
- 团队无法运营两个具有按角色扩展的GPU池：Dynamo有帮助但并非易事。
- 没有RDMA网络：TCP传输成本更高。

### 路由器与阶段17·11集成

分离式路由器具有KV缓存感知能力（阶段17·11）。请求到达持有其前缀的解码池——如果不匹配，则流经预填充→解码。命中率和分离式架构相互叠加——缓存感知路由器决定了是否需要新的预填充。

### Blackwell上的MoE才是实际数字所在

GB300 NVL72 + Dynamo相比Hopper基线展示了50倍的MoE吞吐量。MoE专家路由在预填充上计算密集，但在解码上内存密集（专家缓存），因此分离式架构是双重胜利。2026年的前沿模型服务将MoE为主导（DeepSeek-V3，未来的GPT-5变体）。

### 你应该记住的数字

基准测试数字会漂移——NVIDIA和推理栈每季度发布更新结果。引用前请重新检查。

- DeepSeek-R1在GB200 NVL72 + Dynamo上：中延迟场景下约6倍于基线的吞吐量（developer.nvidia.com, 2025-06）；社区声称全栈Blackwell + Dynamo"高达30倍"是方向性汇总，缺乏单一主要来源。
- GB300 NVL72 + Dynamo：相比Hopper的MoE吞吐量提升高达50倍（developer.nvidia.com，无日期）。
- 节省参考（内部综合数据，非单一案例研究）：$600-800K/year off a $200万美元年支出，SLA不变。
- 分离式阈值：提示>512个令牌+输出>200个令牌。
- 通过NIXL的KV传输：70B FP8上4K提示KV为20-80毫秒。

## 使用它

`code/main.py`模拟共置与分离式服务。报告吞吐量、每请求成本以及提示长度交叉点。

## 发布

本课生成`outputs/skill-disaggregation-decider.md`。给定工作负载和集群，决定是否进行解聚。

## 练习

1. 运行`code/main.py`。在什么提示长度下，解聚优于共置？
2. 为P99前缀长度8K、输出300的RAG服务设计预填充池和解码池。
3. Dynamo vs llm-d：为纯Kubernetes环境且无Python运行时偏好的场景选择其中一个。
4. 计算KV传输开销：70B FP8模型的4K预填充约产生500 MB KV。在RDMA 100 GB/s下，传输时间=5 ms。在TCP 10 GB/s下，传输时间=50 ms。哪个对你的SLA更重要？
5. MoE专家路由改变了KV访问模式。在MoE每token激活不同专家的情况下，解聚表现如何？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  解聚服务  |  "分离预填充/解码"  |  为每个阶段分配单独的GPU池  |
|  NIXL  |  "NVIDIA传输"  |  Dynamo的节点间KV传输（RDMA/TCP）  |
|  NVIDIA Dynamo  |  "编排器"  |  位于vLLM/SGLang/TRT-LLM之上的栈协调器  |
|  llm-d  |  "Kubernetes原生"  |  Red Hat + AWS K8s解聚栈  |
|  Planner Profiler  |  "Dynamo自动配置"  |  测量工作负载，配置池比例  |
|  SLA Planner  |  "Dynamo策略"  |  自动匹配预填充:解码速率以满足SLO  |
|  `packDomain: rack`  |  "llm-d拓扑"  |  将预填充和解码部署在同一机架以实现快速KV传输  |
|  UCCL  |  "统一集合"  |  llm-d 0.5网络层，支持缩至零  |
|  MoE专家路由  |  "每token专家"  |  DeepSeek-V3模式；解聚有助于此  |

## 延伸阅读

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)
