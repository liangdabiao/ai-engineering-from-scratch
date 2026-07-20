# 多区域大语言模型(LLM)服务与KV缓存局部性

> 轮询负载均衡对缓存的LLM推理有严重的负面影响。请求未到达持有其前缀的节点时，需要支付全部预填充成本——长提示词下P50大约800毫秒，而缓存命中时约为80毫秒。到2026年，生产模式是使用缓存感知路由器（用Rust编写的vLLM Router、llm-d router），它消耗KV缓存事件并根据前缀哈希匹配进行路由。最近的研究（GORGO）将跨区域网络延迟作为路由目标的显式项。商业的“跨区域推理”产品（Bedrock跨区域推理、GKE多集群网关）将推理视为不透明——它们处理可用性，而非TTFT。摩根大通和梅奥诊所于2024年11月在us-east-1进行了故障转移演练，耗时约22分钟。灾难恢复现实：32%的LLM灾难恢复失败是因为团队备份了权重但忘记了分词器文件或量化配置。

**类型:** 学习
**语言:** Python（标准库，玩具级前缀缓存感知路由器模拟器）
**前置条件：** 阶段17·04（vLLM服务）、阶段17·06（SGLang RadixAttention）
**时间：** 约60分钟

## 学习目标

- 解释为什么轮询负载均衡会破坏缓存推理，并量化TTFT损失。
- 绘制缓存感知路由器的示意图：输入（KV缓存事件）、算法（前缀哈希匹配）、决策因子（GPU利用率）。
- 指出LLM灾难恢复失败的32%原因（缺少分词器文件/量化配置），并列出三项文件的灾难恢复清单。
- 区分商业跨区域产品（Bedrock CRI、GKE多集群网关）与KV感知路由。

## 问题

您的服务运行在us-east-1、us-west-2和eu-west-3。您在前面部署了一个ALB并使用轮询方式。生产环境中前缀缓存命中率降至8%。TTFT P50激增三倍。您的vLLM日志显示每个请求都在支付完整的预填充成本。

轮询对于无状态服务是最优的。LLM推理本质上是状态性的——KV缓存编码了模型所看到的一切。盲目路由就是路由到错误的缓存中。

另外，您的团队有一个灾难恢复计划。您将模型权重备份到跨区域S3。某个区域发生故障；您尝试故障转移；副本拒绝启动。您忘记了tokenizer.json、量化配置和RoPE缩放配置存放在另一个未同步的存储桶中。

多区域LLM服务是一个缓存问题、路由问题和灾难恢复卫生问题——而不是负载均衡问题。

## 核心概念

### 缓存感知路由

请求到达时带有提示词。路由器对前缀（例如前512个令牌）进行哈希处理；它询问每个副本“您是否缓存了这个前缀？”。副本在分配和驱逐块时通过发布/订阅通道发布KV缓存事件。路由器选择匹配的副本，如果没有匹配的副本，则回退到基于GPU利用率的决策因子。

**vLLM Router**（Rust，2026年生产栈）：订阅`kv.cache.block_added`事件，维护前缀哈希到副本索引的映射，以O(1)时间路由。当没有匹配时回退到最小队列深度。

**llm-d router**：模式相同，原生支持Kubernetes。通过ControlPlane API发布事件。

**SGLang RadixAttention**（阶段17·06）是副本内的等效机制。跨副本路由严格属于上游。

### 数字

2K令牌提示词，Llama 3.3 70B FP8，H100的TTFT P50：
- 缓存命中（相同副本，前缀驻留）：约80毫秒。
- 缓存未命中（冷预填充）：约800毫秒。

10倍差距。如果您的路由器在副本间达到60-80%的前缀缓存命中率，则近似于单副本性能但具有N副本容量。如果命中率只有10%，则近似于朴素扩展。

### 跨区域有一个新的约束——网络延迟

区域间往返时延：
- us-east-1 ↔ us-west-2：约65毫秒。
- us-east-1 ↔ eu-west-1：约75毫秒。
- us-east-1 ↔ ap-southeast-1：约220毫秒。

如果将请求从us-east-1路由到ap-southeast-1的热前缀，则节省的预填充时间（800毫秒→80毫秒）被440毫秒的往返时间所掩盖。GORGO（2026年研究）使这一点明确——最小化`prefill_time + network_latency`的联合目标，而不仅仅是预填充。通常的答案是保持区域内部路由，除非是巨大的多MB前缀，此时预填充占主导地位。

### 商业“跨区域推理”在这里无济于事

AWS Bedrock跨区域推理在容量压力下自动将请求路由到其他区域。它优化可用性，而非TTFT，并将推理视为不透明。GKE多集群网关也是如此——服务级故障转移，不感知KV缓存。

即使使用这些，您仍然需要一个应用层的缓存感知路由器。它们处理“us-east-1着火了”的情况。缓存感知路由器处理TTFT问题。

### 灾难恢复卫生——32%的文件缺失问题

广泛引用的2026年统计数据：32%的LLM灾难恢复失败是因为团队备份了权重但忘记了：

- `tokenizer.json`或`tokenizer.model`
- 量化配置（`tokenizer.json`、AWQ缩放因子、GPTQ零点）
- 模型特定配置（RoPE缩放、注意力掩码、聊天模板）
- 引擎配置（`tokenizer.json`、采样默认值、LoRA适配器清单）

解决方案是最少三项文件的灾难恢复清单：

1. HF模型仓库下的所有文件（权重+配置+分词器）。
2. 引擎特定的服务配置。
3. 部署清单（K8s YAML、Dockerfile、依赖锁定文件）。

另外：每季度进行一次灾难恢复演练。摩根大通2024年11月的us-east-1演练之所以在22分钟内恢复，只是因为手册经过预演。

### 数据驻留是正交问题

欧盟客户的PHI不能离开欧盟。如果您的缓存感知路由器将巴黎发出的请求发送到us-east-1进行前缀匹配，则无论TTFT收益如何，您都违反了GDPR。在优化缓存之前，先按驻留边界划分路由器。

### 你应该记住的数字

- 缓存命中与未命中TTFT差距：约10倍（2K提示下80 ms vs 800 ms）。
- 跨区域RTT美国-欧盟：约75 ms。
- DR失败：32%未命中tokenizer/量化配置。
- 摩根大通us-east-1故障转移2024年11月：22分钟（30分钟SLA）。

## 使用它

`code/main.py`在跨区域工作负载上模拟三种路由策略（轮询、缓存感知区域、缓存感知全局）。报告缓存命中率、TTFT P50/P99以及跨区域费用。

## 发布

本课生成`outputs/skill-multi-region-router.md`。给定区域、数据驻留约束和SLA，设计路由计划。

## 练习

1. 运行`code/main.py`。在给定75 ms RTT的情况下，当提示长度达到多少时，跨区域路由会优于仅本地路由？
2. 你的缓存命中率从70%下降到12%。诊断三个可能的原因以及可以确认每个原因的观测指标。
3. 为在vLLM中提供服务的70B AWQ量化模型设计一个DR清单，该模型带有5个LoRA适配器。列出每个文件和配置。
4. 讨论Bedrock跨区域推理对于具有严格TTFT SLO的金融科技公司来说是否“足够”。引用具体行为。
5. 一个来自巴黎的请求匹配了us-east-1中的前缀。你会路由它吗？写出策略。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
| 缓存感知路由  |  "智能负载均衡器"  |  根据前缀哈希匹配路由到持有KV缓存的副本 |
| KV缓存事件  |  "缓存发布/订阅"  |  副本发布块添加/驱逐；路由器建立索引 |
| 前缀哈希  |  "缓存键"  |  前N个令牌的哈希用作路由器查找 |
| GORGO  |  "跨区域路由研究"  |  arXiv 2602.11688；网络延迟作为显式项 |
| 跨区域推理  |  "Bedrock CRI"  |  AWS产品；可用性故障转移，不关注TTFT |
| DR清单  |  "备份列表"  |  恢复所需的所有文件——不仅仅是权重 |
| 数据驻留  |  "GDPR边界"  |  关于哪个区域可以查看用户数据的法律约束 |
| RTT  |  "往返时间"  |  网络延迟；美国-欧盟75 ms，美国-亚太220 ms |
| LLM感知负载均衡器  |  "缓存命中负载均衡器"  |  缓存感知路由器作为一个产品类别 |

## 延伸阅读

- [BentoML — Multi-cloud and cross-region inference](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [BentoML — Multi-cloud and cross-region inference](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference) — 跨区域KV缓存重用，包含网络延迟项。
- [BentoML — Multi-cloud and cross-region inference](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [BentoML — Multi-cloud and cross-region inference](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference) — 可用性故障转移文档。
- [BentoML — Multi-cloud and cross-region inference](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference) — 缓存感知路由器源代码。
