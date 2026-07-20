# 使用LMCache KV卸载的vLLM生产栈

> vLLM的生产栈（production-stack）是参考性的Kubernetes部署——路由器(router)、引擎(engine)和可观测性(observability)组件已集成在一起。LMCache是KV卸载层，将KV缓存从GPU内存中提取出来，并在不同查询和引擎之间重用（先使用CPU DRAM，再使用磁盘/Ceph）。vLLM 0.11.0的KV卸载连接器（2026年1月）通过连接器API（v0.9.0+）实现了异步和可插拔。卸载延迟对用户不可见。即使没有共享前缀，LMCache也很有价值——当GPU的KV槽位耗尽时，被抢占的请求可以从CPU恢复，而无需重新计算预填充(prefill)。已发布的基准测试在16x H100（80GB HBM）跨4个a3-highgpu-4g上运行：当KV缓存超出HBM时，原生CPU卸载和LMCache都能显著提高吞吐量；在KV占用较小时，所有配置与基线匹配，仅带来少量开销。

**类型：** 学习
**语言：** Python（stdlib，玩具KV溢出模拟器）
**前置知识：** 阶段17·04（vLLM服务内部机制），阶段17·06（SGLang/RadixAttention）
**时间：** 约60分钟

## 学习目标

- 绘制vLLM生产栈各层：路由器、引擎、KV卸载、可观测性。
- 解释KV卸载连接器API（v0.9.0+）以及0.11.0异步路径如何隐藏卸载延迟。
- 量化LMCache CPU-DRAM在何时有帮助（KV > HBM）以及何时增加开销（KV足够小可容纳于HBM）。
- 根据部署约束，在原生vLLM CPU卸载和LMCache连接器之间进行选择。

## 问题

您的vLLM服务显示GPU的HBM使用率达到100%，且每当并发度增加时就会出现抢占事件。请求被逐出、重新排队，并且您在一分钟内重复预填充相同的2K token提示。GPU计算资源浪费在冗余的预填充上；实际有效吞吐量远低于原始吞吐量。

增加更多GPU的成本是线性的。增加更多HBM则不可能。但CPU DRAM很便宜——单个插槽就有512GB以上，虽然延迟比HBM差几个数量级，但对于“暂时性温”的KV缓存来说已经足够。

LMCache将KV缓存提取到CPU DRAM，使被抢占的请求快速恢复，同时跨引擎重复的前缀无需每个引擎重新预填充即可共享缓存。

## 核心概念

### vLLM生产栈

`github.com/vllm-project/production-stack`是参考性的Kubernetes部署：

- **路由器(Router)**——缓存感知（阶段17·11）。消费KV事件。
- **引擎(Engines)**——vLLM工作节点。每个GPU或每个TP/PP组一个。
- **KV缓存卸载**——LMCache部署或原生连接器。
- **可观测性(Observability)**——Prometheus抓取、Grafana仪表盘、OTel追踪。
- **控制平面(Control plane)**——服务发现、配置、滚动更新。

以Helm chart和operator形式发布。

### KV卸载连接器API（v0.9.0+）

vLLM 0.9.0引入了连接器API，用于可插拔的KV缓存后端。您的引擎将块卸载到连接器；连接器存储它们（RAM、磁盘、对象存储、LMCache）。当请求需要块时，连接器将其加载回来。

vLLM 0.11.0（2026年1月）增加了异步卸载路径——卸载可以在后台进行，因此引擎在常见情况下不会阻塞。端到端延迟和吞吐量仍然取决于工作负载形状、KV缓存命中率和系统压力；vLLM自己的说明指出，自定义内核卸载在低命中率下会降低吞吐量，并且异步调度与推测解码已知存在交互问题。

### 原生CPU卸载与LMCache对比

**原生vLLM CPU卸载**：引擎本地。将KV块存储在主存RAM中。实现快速，零网络跳。不跨引擎。

**LMCache连接器**：集群级。将块存储在共享的LMCache服务器（CPU DRAM + Ceph/S3层）中。任何引擎都可以访问块。已发布16x H100基准测试。

当单个引擎面临HBM压力时，选择原生卸载。当多个引擎共享前缀时（RAG使用通用系统提示、多租户共享模板），选择LMCache。

### 基准测试行为

16x H100（80 GB HBM）分布在4个a3-highgpu-4g上的测试：

- 低KV占用（短提示、低并发）：所有配置与基线匹配，LMCache增加约3-5%开销。
- 中等占用：LMCache开始有助于跨引擎的前缀重用。
- KV超出HBM：原生CPU卸载和LMCache都显著提高吞吐量；LMCache增益更大，因为跨引擎共享。

### 何时LMCache起决定性作用

- 多租户服务，其中系统提示在租户间共享。
- RAG，其中文档块在查询中重复出现。
- 相同基座模型上的微调变体（LoRA），基座模型的KV重用减少了冗余工作。
- 抢占密集型工作负载：从CPU恢复比重做预填充更便宜。

### 何时不启用

- HBM压力小——您会付出开销而无收益。
- 短上下文（<1K token）——传输时间超过重新预填充。
- 单租户单提示工作负载——没有可捕获的重用。

### 与分离式服务的集成

阶段17·17分离式服务+LMCache共同作用：KV从预填充池传输到解码池，如果不使用则存入LMCache；后续查询从LMCache拉取。阶段17·11缓存感知路由器可以将请求路由到其本地或LMCache共享缓存匹配的引擎。

### 你应该记住的数字

- vLLM 0.9.0：连接器API发布。
- vLLM 0.11.0（2026年1月）：异步卸载路径；端到端延迟影响取决于工作负载、KV命中率和系统压力（非绝对保证）。
- 16x H100基准测试：当KV占用超出HBM时，LMCache有帮助。
- HBM压力小：3-5%开销而无收益。

```figure
zero-sharding
```

## 使用它

`code/main.py`模拟了有无LMCache的抢占密集型工作负载。报告了避免的重新预填充次数、吞吐量增益以及盈亏平衡的HBM利用率。

## 发布

本节课产出`outputs/skill-vllm-stack-decider.md`。根据工作负载形状和vLLM部署，决定使用原生、LMCache还是都不使用。

## 练习

1. 运行`code/main.py`。LMCache在什么HBM利用率下开始生效？
2. 一个租户在200次查询/小时中共享一个6K令牌的系统提示。计算每个租户的预期LMCache节省量。
3. LMCache服务器是单点故障。设计高可用性策略（副本、回退到原生模式）。
4. LMCache将数据存储到旋转磁盘上的Ceph。对于70B FP8（500 MB）的4K令牌KV，读取时间与重新预填充相比如何？
5. 论证vLLM 0.11.0异步路径是否“免费”——开销隐藏在哪里？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
| 生产栈  |  "参考部署"  |  vLLM的Kubernetes Helm chart + operator  |
| 连接器API  |  "KV后端接口"  |  vLLM 0.9.0+可插拔KV存储接口  |
| 原生CPU卸载  |  "引擎本地溢出"  |  在同一引擎的主机RAM中存储KV  |
| LMCache  |  "集群KV缓存"  |  跨引擎KV缓存服务器，位于CPU DRAM + 磁盘  |
| 0.11.0异步  |  "非阻塞卸载"  |  隐藏在引擎流后面的卸载  |
| 抢占  |  "驱逐腾出空间"  |  HBM满时的KV缓存洗牌  |
| 前缀复用  |  "相同系统提示"  |  多个查询共享开头；缓存命中  |
| Ceph层  |  "磁盘层"  |  缓存层次结构中DRAM下的持久存储  |

## 延伸阅读

- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html) — Helm chart + operator.
- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html) — 连接器实现。
- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html) — 异步路径详细信息。
