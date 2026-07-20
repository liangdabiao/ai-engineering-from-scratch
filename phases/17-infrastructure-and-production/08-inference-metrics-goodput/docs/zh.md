# 推理指标 — TTFT、TPOT、ITL、Goodput、P99

> 四个指标决定推理部署是否有效。TTFT 是预填充加排队加网络。TPOT（等价于 ITL）是受内存限制的每个令牌解码成本。端到端延迟是 TTFT 加上 TPOT 乘以输出长度。吞吐量是整个集群每秒处理的令牌数。但对于产品而言，真正重要的是有效吞吐量(Goodput)——同时满足所有 SLO 的请求比例。高吞吐量但低有效吞吐量意味着你在处理那些永远无法按时到达用户的令牌。2026 年 Llama-3.1-8B-Instruct 在 TRT-LLM 上的参考数值：平均 TTFT 162 ms，平均 TPOT 7.33 ms，平均 E2E 1,093 ms。始终报告 P50、P90、P99——绝不仅仅是平均值。注意测量陷阱：GenAI-Perf 在 ITL 计算中排除 TTFT，LLMPerf 包含它；两个工具对同一运行的 TPOT 结果不一致。

**类型：** 学习
**语言：** Python（标准库，简易百分位计算器和有效吞吐量报告器）
**前置要求：** 阶段 17 · 04（vLLM 服务内部机制）
**时间：** ~60 分钟

## 学习目标

- 精确定义 TTFT、TPOT、ITL、E2E、吞吐量和有效吞吐量，并说明每个指标衡量哪个组件。
- 解释为什么平均值对于 LLM 服务是错误的统计量，以及如何解读 P50/P90/P99。
- 构建一个多约束 SLO（例如 TTFT<500 ms 且 TPOT<15 ms 且 E2E<2 s），并据此计算有效吞吐量。
- 说出两个对同一运行 TPOT 结果不一致的基准测试工具，并解释原因。

## 问题

"我们的吞吐量是每秒 15,000 个令牌。"那又怎样？如果 40% 的请求端到端超过 2 秒，用户就会放弃会话。仅凭吞吐量并不能告诉你产品是否有效。

推理有多种延迟维度，每种失败方式不同。预填充是计算密集型的，随提示长度扩展。解码是内存密集型的，随批大小扩展。排队延迟是操作问题。网络是物理距离问题。你需要为每个维度设置不同的指标，还需要百分位数，以及一个单一的复合指标来表明"用户是否得到了他们期望的"——这就是有效吞吐量。

## 核心概念

### TTFT——首个令牌时间

`TTFT = queue_time + network_request + prefill_time`

当提示较长时，预填充占主导。在 H100 上的 Llama-3.3-70B FP8 中，32k 提示需要约 800 ms 的纯预填充。排队时间是负载下的调度器行为。网络请求是包括 TLS 的线路时间。TTFT 是用户在流式返回之前看到的延迟。

### TPOT / ITL——令牌间延迟

同一个量的多个名称。`TPOT`（每个输出令牌时间），`ITL`（令牌间延迟），`decode latency per token`——都一样。它是第一个令牌之后连续流式令牌之间的时间。

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

在同一个带分块预填充的 Llama-3.3-70B H100 堆栈上，TPOT 平均值约 7 ms。没有分块预填充时，在相邻序列的长预填充期间，TPOT 可能峰值达到 50 ms。关注 P99，而不是平均值。

### E2E 延迟

`E2E = TTFT + TPOT * output_tokens + network_response`

对于长输出（>500 令牌），E2E 由 TPOT 主导。对于短输出但长提示，E2E 由 TTFT 主导。报告按输出长度分层的 E2E。

### 吞吐量

`throughput = total_output_tokens / elapsed_time`

聚合指标。告诉你集群效率。但不能告诉你单个请求的健康状况。

### 有效吞吐量——你真正关心的指标

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

SLO 是多约束的。一个请求只有满足所有约束才算"有效"。有效吞吐量是有效请求的比例。高吞吐量但 60% 有效吞吐量是失败。较低吞吐量但 99% 有效吞吐量是目标。

2026 年，有效吞吐量是 MLPerf 推理 v6.0 提交以及 AI 平台提供商内部 SLA 跟踪中使用的指标。

### 为什么平均值是错误的统计量

LLM 延迟分布是右偏的。一个带有长预填充邻居的解码批次可能以 TPOT ~7 ms 提供 500 个令牌，而以 TPOT ~60 ms 提供 20 个令牌。平均 TPOT 是 9 ms。P99 TPOT 是 65 ms。用户经常遇到 P99——这就是他们离开的原因。

始终报告三元组（P50，P90，P99）。对于用户体验，P99 是你优化的目标。

### 参考数值——Llama-3.1-8B-Instruct 在 TRT-LLM 上，2026 年

- 平均 TTFT：162 ms
- 平均 TPOT：7.33 ms
- 平均 E2E：1,093 ms
- P99 TPOT：根据分块预填充配置在 10-25 ms 之间变化。

这些是 NVIDIA 发布的参考点。它们会随模型大小（70B 将显示 3-5 倍）、硬件（H100 与 B200 相比约 3 倍）和负载而变化。

### 测量陷阱

2026 年两个最常用的基准测试工具对同一运行的 TPOT 结果不一致：

- **NVIDIA GenAI-Perf**：在 ITL 计算中排除 TTFT。ITL 从第 2 个令牌开始。
- **LLMPerf**：包含 TTFT。ITL 从第 1 个令牌开始。

对于一个 TTFT 为 500 ms、总解码时间 700 ms 内 100 个输出令牌的请求，GenAI-Perf 报告 `ITL = 700/99 = 7.07 ms`，LLMPerf 报告 `ITL = 1200/100 = 12.00 ms`。工具的选择会改变数值。

始终说明使用哪个工具。始终发布定义。

### 构建 SLO

2026 年 70B 聊天模型合理面向消费者的 SLO：

- TTFT P99 <= 800 毫秒。
- TPOT P99 <= 25 毫秒。
- 对于 <300 token 的输出，E2E P99 <= 3 秒。
- Goodput 目标 >= 99%。

企业级SLO会收紧TTFT（200-400毫秒）并放宽E2E。关键在于将它们写下来，测量所有三项，并将goodput作为一个单一复合指标进行跟踪。

### 如何测量

- 运行真实流量或逼真的合成流量（使用`--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`的LLMPerf）。
- 基准测试运行的目标并发量为峰值并发量的2倍。
- 运行30-50次迭代，取合并样本的百分位数。
- 发布时附带工具名称、工具版本、模型、硬件、并发量、提示分布。

```figure
throughput-latency
```

## 使用它

`code/main.py` 是一个简易的goodput计算器。生成合成延迟分布，应用SLO，并计算goodput。它还在同一轨迹上展示了GenAI-Perf与LLMPerf在TPOT上的差异。

## 发布

本课产出`outputs/skill-slo-goodput-gate.md`。给定工作负载和SLO，它生成一个CI/CD就绪的基准测试配方，该配方基于goodput而非吞吐量来控制部署。

## 练习

1. 运行`code/main.py`。生成一个带有1%尾部尖峰（tail spike）的分布。当P99 TPOT从30毫秒收紧到15毫秒时，goodput如何变化？
2. 某供应商声称“在Llama 3.3 70B H100上达到15,000 tok/s”。在相信这个数据之前，请提出三个问题。
3. 为什么分块预填充（chunked prefill）能保护P99 TPOT但不能保护平均TPOT？
4. 为语音助手构建一个消费者SLO（首个token是听到的，而不是读到的）。哪个指标对用户最可见？
5. 阅读LLMPerf的README和GenAI-Perf的文档。找出两个工具在其他三个指标上的分歧。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  TTFT  |  “首个token耗时”  |  队列+网络+预填充；长提示时主要由预填充主导 |
|  TPOT  |  “每个输出token耗时”  |  首个token之后每个token的内存受限解码开销 |
|  ITL  |  “token间延迟”  |  在大多数工具中与TPOT相同（并非全部——请参见GenAI-Perf） |
|  E2E  |  “端到端”  |  TTFT + TPOT * 输出长度；再加上响应侧的网络时间 |
|  Throughput  |  “tok/s”  |  集群效率；没有延迟百分位数则无意义 |
|  Goodput  |  “SLO满足率”  |  同时满足所有SLO约束的请求比例 |
|  P99  |  “尾部”  |  百分之一的最坏情况延迟；用户体验指标 |
|  SLO多重约束  |  “联合”  |  所有三个延迟界限的与运算；任一违反则请求失败 |
|  GenAI-Perf vs LLMPerf  |  “工具陷阱”  |  工具在ITL是否包含TTFT上存在分歧 |

## 延伸阅读

- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — TTFT、ITL、TPOT的规范定义。
- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — 替代定义和测量方法。
- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — 在实际部署上的应用测量。
- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — 基于Ray的开源基准测试。
- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — NVIDIA的基准测试工具。
- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — 业界公认的基于goodput的基准测试。
