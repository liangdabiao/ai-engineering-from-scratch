# 结业项目14 — 推测解码推理服务器

> vLLM 0.7中的EAGLE-3在实际流量上实现了2.5-3倍吞吐量。P-EAGLE（AWS 2026）进一步推动了并行推测。SGLang的SpecForge大规模训练了草稿头。Red Hat的Speculators hub发布了常见开源模型的对齐草稿。TensorRT-LLM使推测解码在NVIDIA上成为一等公民。2026年的生产服务栈是vLLM或SGLang搭配EAGLE系列草稿、FP8或INT4量化，以及基于队列等待的HPA。本结业项目旨在服务两个开源模型，实现2.5倍以上基线吞吐量，并提供完整的尾延迟报告。

**类型:** 结业项目
**语言:** Python (服务)，C++ / CUDA (内核检查)，YAML (配置)
**前置条件:** 阶段3 (深度学习)，阶段7 (Transformer)，阶段10 (从头构建LLM)，阶段17 (基础设施)
**涉及的阶段:** P3 · P7 · P10 · P17
**时间:** 30小时

## 问题

推测解码在2026年已成为通用技术。EAGLE-3草稿头在目标模型的隐藏状态上训练，并预测后续N个token；目标模型在单次前向传播中验证。60-80%的接受率转化为2-3倍的端到端吞吐量。vLLM 0.7原生集成了这一功能。SGLang + SpecForge提供了训练管线。Red Hat的Speculators发布了针对Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B的对齐草稿。

关键在于服务运维，而非模型本身。接受率随流量分布（ShareGPT对比代码对比领域数据）而变化。拒绝情况下的尾延迟比不使用推测时更差——你必须报告多个批次大小下的p99，而不仅仅是稳态的token/秒。每百万token成本与Anthropic/OpenAI API的对比是可信度的关键杠杆。

## 概念

推测解码有两个层次。一个**草稿**模型（EAGLE-3头、n-gram或更小的目标对齐模型）每步提出k个候选token。**目标**模型在单次前向传播中验证所有k个token；任何被接受的前缀都会替换贪婪路径。接受率取决于草稿-目标对齐程度和输入分布。

EAGLE-3在大多数流量上优于n-gram草稿。P-EAGLE运行并行推测以实现更深的草稿树。权衡：拒绝时的P99延迟更高，因为验证前向传播更大。服务配置必须报告按批次大小分组的延迟以暴露这一点。

部署采用Kubernetes。vLLM 0.7每个GPU或张量并行分片运行一个副本。HPA基于队列等待而非CPU进行自动伸缩。FP8（Marlin）和INT4（AWQ）量化将GPU内存控制在H100/H200范围内。端到端报告包括吞吐量、接受率、批次大小1/8/32下的p50/p99，以及每百万token成本。

## 架构

```
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## 技术栈

- 服务: vLLM 0.7 或 SGLang 0.4
- 推测方法: EAGLE-3草稿头, P-EAGLE并行推测, n-gram回退
- 草稿训练: SpecForge (SGLang) 或 Red Hat Speculators
- 目标模型: Llama 3.3 70B, Qwen3-Coder-30B MoE, GPT-OSS-120B
- 量化: FP8 (Marlin), INT4 AWQ
- 部署: Kubernetes + NVIDIA设备插件；基于队列等待指标的HPA
- 评估: ShareGPT, MT-Bench-v2, GSM8K, HumanEval用于领域分布接受率测量
- 参考: TensorRT-LLM推测解码作为供应商基线

## 动手构建

1. **目标模型准备。** 选择Llama 3.3 70B。通过Marlin量化为FP8。在vLLM 0.7下部署到1xH100（或2路张量并行）。

2. **草稿来源。** 从Red Hat Speculators拉取一个对齐的EAGLE-3草稿头（或通过SpecForge训练一个）。加载到vLLM的推测解码配置中。

3. **基线数据。** 推测前：批次1/8/32下的token/s、p50/p99延迟、GPU利用率。发布。

4. **启用EAGLE-3。** 切换配置；重新运行相同的基准测试。报告加速比、接受率、p99尾延迟差异。

5. **P-EAGLE。** 启用并行推测；测量更深的草稿树与串行EAGLE-3的对比。报告P-EAGLE何时有益何时有害的拐点。

6. **领域流量。** 通过同一服务器运行ShareGPT、HumanEval以及特定领域流量。测量每个分布的接受率。识别草稿何时发生漂移。

7. **第二个目标模型。** 在Qwen3-Coder-30B MoE上运行相同管线。草稿更棘手（MoE路由噪声）。报告。

8. **K8s HPA。** 在K8s下部署，HPA跟踪`queue_wait_ms`。演示负载增加三倍时的横向扩展。

9. **成本对比。** 计算每百万token的成本，与Anthropic Claude Sonnet 4.7和OpenAI GPT-5.4在相同评估上的比较。发布。

## 使用它

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## 发布

`outputs/skill-inference-server.md`描述了交付物。一个经过测量的推测解码服务栈、完整的基准测试报告以及K8s部署。

|  权重  |  标准  |  衡量方式  |
|:-:|---|---|
|  25  |  相对于基线的实测加速比  |  两个模型在匹配质量下实现2.5倍以上吞吐量  |
|  20  |  实际流量下的接受率  |  每个分布的接受率报告  |
|  20  |  P99尾延迟控制  |  有/无推测时批次1/8/32下的p99  |
|  20  |  运维  |  K8s部署、基于队列等待的HPA、平滑上线  |
|  15  |  报告与方法论  |  清晰解释更改了什么以及为什么  |
|  **100**  |   |   |

## 练习

1. 测量草稿比目标落后一个版本时接受率的下降（例如，Llama 3.3 -> 3.4漂移）。构建监控告警。

2. 实现n-gram回退：如果EAGLE-3接受率低于阈值，切换到n-gram草稿。报告可靠性提升。

3. 运行受控的MoE实验：相同的Qwen3-Coder-30B，注入路由噪声与无噪声对比。测量草稿接受率敏感性。

4. 扩展到H200（141 GB）。报告每副本模型大小的提升空间，以及是否能够服务未量化的Llama 3.3 70B。

5. 在相同的H100硬件上对TensorRT-LLM推测解码进行基准测试。报告它在哪些方面优于vLLM。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
| 草稿模型 | "推测器(Speculator)" | 为目标模型提出N个令牌(tokens)以供验证的小模型 |
| EAGLE-3 | "2026年草稿架构" | 基于目标隐藏状态训练的草稿头(draft head)；约75%接受率 |
| P-EAGLE | "并行推测(Parallel speculation)" | 在一次目标传递中验证的草稿分支树 |
| 接受率(Acceptance rate) | "命中率(Hit rate)" | 无需重采样即被接受的草稿令牌比例 |
| 量化(Quantization) | "FP8 / INT4" | 降低权重的精度以在GPU内存中容纳更多模型 |
| 队列等待(Queue wait) | "HPA指标(HPA metric)" | 请求在开始推理前于待办队列中等待的时间 |
| Speculators hub | "对齐的草稿(Aligned drafts)" | Red Hat Neural Magic 提供的针对常见开源模型的EAGLE草稿中心 |

## 延伸阅读

- [vLLM EAGLE and P-EAGLE documentation](https://docs.vllm.ai) — 参考服务栈
- [vLLM EAGLE and P-EAGLE documentation](https://docs.vllm.ai) — 并行推测解码论文+集成
- [vLLM EAGLE and P-EAGLE documentation](https://docs.vllm.ai) — 草稿头训练流程
- [vLLM EAGLE and P-EAGLE documentation](https://docs.vllm.ai) — 对齐草稿中心
- [vLLM EAGLE and P-EAGLE documentation](https://docs.vllm.ai) — 供应商替代方案
- [vLLM EAGLE and P-EAGLE documentation](https://docs.vllm.ai) — 商业参考
- [vLLM EAGLE and P-EAGLE documentation](https://docs.vllm.ai) — 方法论文
- [vLLM EAGLE and P-EAGLE documentation](https://docs.vllm.ai) — 代码和基准测试
