# LLM API 负载测试 —— 为什么 k6 和 Locust 会骗人

> 传统的负载测试工具并非为流式响应、可变输出长度、令牌级指标或GPU饱和而设计。两个陷阱容易坑到大多数团队。GIL陷阱：Locust的令牌级测量在Python GIL下运行分词，在高并发下与请求生成竞争；分词积压会导致报告的令牌间延迟膨胀——瓶颈在客户端，而非服务端。提示一致性问题：在循环测试中使用相同提示只测试了令牌分布中的一个点；真实流量具有可变长度和多样的前缀匹配。LLMPerf通过`--mean-input-tokens`和`--stddev-input-tokens`解决了这个问题。2026年的工具映射：专用LLM工具（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）用于令牌级精度；**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025年9月）**——支持流式、Kubernetes原生分布式，通过TestRun/PrivateLoadZone CRD实现，最适合CI/CD门控；Vegeta用于Go恒定速率饱和；Locust 2.43.3仅配合LLM-Locust扩展支持流式。负载模式：稳态、斜坡、突发（自动缩放测试）、浸泡（内存泄漏）。

**类型：** 构建
**语言：** Python（标准库、玩具级真实提示生成器 + 延迟收集器）
**前置条件：** 阶段17·08（推理指标）、阶段17·03（GPU自动缩放）
**时间：** 约75分钟

## 学习目标

- 解释导致通用负载测试工具对LLM API失效的两种反模式（GIL陷阱、提示一致性问题）。
- 为特定目的选择工具：LLMPerf（基准测试运行）、k6 + 流式扩展（CI门控）、guidellm（大规模合成）、GenAI-Perf（NVIDIA参考）。
- 设计四种负载模式（稳态、斜坡、突发、浸泡）并指出每种模式捕获的故障模式。
- 使用输入令牌的均值与标准差而非固定长度构建真实的提示分布。

## 问题

你用k6测试了LLM端点，500个并发用户。它扛住了。你上线了。在生产环境中，实际只有200个用户，服务就崩溃了——P99 TTFT爆炸，GPU满载。

发生了两件事。第一，k6发送了500个完全相同的提示——你的请求合并和前缀缓存使得看起来你同时在处理500个解码，但实际上只处理了一个。第二，k6不像人眼那样追踪流响应中的令牌间延迟；它只看到一条HTTP连接，而不是500个令牌以不同间隔到达。

LLM的负载测试是一门独立的学科。

## 核心概念

### GIL陷阱（Locust）

Locust使用Python，在GIL下客户端运行分词。在高并发下，分词器在请求生成后面排队。报告的令牌间延迟包含客户端分词积压。你以为服务端慢，其实是测试工具的问题。

修复：LLM-Locust扩展将分词移至独立进程，或使用编译语言工具（k6、使用tokenizers.rs的LLMPerf）。

### 提示一致性问题

所有已知负载测试工具都允许配置一个提示。在10000次迭代的循环测试中，每次发送完全相同的提示。服务端每次看到相同前缀——前缀缓存命中率接近100%，吞吐量看起来很棒。

修复：从提示分布中采样。LLMPerf使用`--mean-input-tokens 500 --stddev-input-tokens 150`——不同的长度、不同的内容。

### 四种负载模式

1. **稳态**——恒定RPS持续30-60分钟。捕获：基准性能回归。
2. **斜坡**——在15分钟内将RPS从0线性增加到目标值。捕获：容量拐点、预热异常。
3. **突发**——突然增加3-10倍RPS持续2分钟，然后恢复。捕获：自动缩放延迟、队列饱和、冷启动影响。
4. **浸泡**——稳态持续4-8小时。捕获：内存泄漏、连接池漂移、可观测性溢出。

### 2026年工具映射

**LLMPerf**（Anyscale）——Python但使用Rust后端分词。均值/标准差提示。支持流式。性能运行的最佳默认选择。

**NVIDIA GenAI-Perf**——NVIDIA的参考实现。使用Triton客户端；全面的指标覆盖。注意其ITL不包括TTFT；LLMPerf包含。同一个服务器，两个工具会产生不同的TPOT。

**LLM-Locust**（TrueFoundry）——修复GIL陷阱的Locust扩展。熟悉的Locust DSL + 流式指标。

**guidellm**——大规模合成基准测试。

**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025年9月）**：
- k6本身（Go，编译，无GIL）增加了流式感知指标。
- k6 Operator使用TestRun / PrivateLoadZone CRD实现Kubernetes原生分布式测试。
- 最适合CI/CD门控和SLA测试。

**Vegeta**——Go语言，比k6更简单。恒定速率HTTP饱和。不感知LLM，但适用于网关/速率限制测试。

**Locust 2.43.3原版**——存在LLM的GIL陷阱。仅配合LLM-Locust扩展。

### CI中的SLA门控

在PR上运行k6：

- 每个基线RPS运行30-50次迭代。
- 门控条件：P50/P95 TTFT、5xx错误率<5%、TPOT低于阈值。
- 超出则构建失败。

### 真实的提示分布

从真实流量样本（如果有）或公开分布（例如聊天用ShareGPT提示、代码用HumanEval）构建。将均值与标准差输入LLMPerf。无论如何都避免单提示循环。

### 你应该记住的数字

- k6 Operator 1.0 GA：2025年9月。
- k6 v2026.1.0：支持流式指标。
- 典型LLMPerf运行：在并发度X下发送100-1000个请求。
- 典型CI门控：每个PR迭代30-50次。
- 四种模式：稳态、斜坡、突发、浸泡。

## 使用它

`code/main.py`模拟了具有真实提示分布的负载测试，测量有效TPOT，并演示了提示一致性陷阱。

## 发布

本课产生`outputs/skill-load-test-plan.md`。给定工作负载和服务等级协议，选择工具并设计四种负载模式。

## 练习

1. 运行`code/main.py`。比较均匀分布与真实分布——差距在哪里？
2. 为CI门编写k6脚本：在100并发下，TTFT P95 < 800 ms，运行时间5分钟。
3. 你的长时间测试显示内存以50 MB/小时增长。指出三种原因以及用于区分它们的检测手段。
4. 尖峰测试从10 RPS到100 RPS。如果使用Karpenter + vLLM生产堆栈（阶段17·03+18），预期恢复时间是多少？
5. GenAI-Perf报告TPOT=6ms；LLMPerf在同一服务器上报告TPOT=11ms。请解释。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  LLMPerf  |  "LLM框架"  |  Anyscale基准测试工具，支持流式  |
|  GenAI-Perf  |  "NVIDIA工具"  |  NVIDIA参考框架  |
|  LLM-Locust  |  "用于LLM的Locust"  |  修复GIL陷阱的Locust扩展  |
|  guidellm  |  "合成基准"  |  大规模合成工具  |
|  k6 Operator  |  "K8s k6"  |  基于CRD的分布式k6  |
|  GIL陷阱  |  "Python客户端开销"  |  分词积压会夸大报告的延迟  |
|  提示均匀性陷阱  |  "单提示谎言"  |  使用相同提示的循环会命中缓存，夸大吞吐量  |
|  稳态  |  "恒定负载"  |  在N分钟内保持恒定RPS  |
|  斜坡  |  "线性上升"  |  在持续时间内从0到目标  |
|  尖峰  |  "突发测试"  |  突然倍增然后恢复  |
|  浸泡  |  "长时间测试"  |  持续数小时以检测泄漏  |

## 延伸阅读

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)
