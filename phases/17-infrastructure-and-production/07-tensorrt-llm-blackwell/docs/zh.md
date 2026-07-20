# Blackwell 上的 TensorRT-LLM，采用 FP8 和 NVFP4

> TensorRT-LLM 是 NVIDIA 独占的，但在 Blackwell 上胜出。在 GB200 NVL72 上使用 Dynamo 编排时，SemiAnalysis InferenceX 测量到 H100 + vLLM 的成本为 $0.012 per million tokens on a 120B model in Q1-Q2 2026, against $0.09/M——存在 7 倍的经济差距。该堆栈由三种浮点体制复合而成：FP8 对 KV 缓存和注意力内核仍然至关重要，因为它具有所需的动态范围；NVFP4（4 位微缩放）处理权重和激活；多 Token 预测（MTP）和分离式预填充/解码再增加 2-3 倍。第 0 天模型支持直接加载 FP4 权重，无需训练后转换。2026 年工程团队的注意事项：TRT-LLM 是一个封闭的 NVIDIA 堆栈，因此采用它会以可移植性换取吞吐量。在做出承诺之前，根据您的模型和硬件组合进行计算。

**类型：** 学习
**语言：** Python（标准库，玩具 FP8/NVFP4 内存和成本计算器）
**先决条件：** 阶段 17 · 04（vLLM 服务内部机制），阶段 10 · 13（量化）
**时间：** ~75 分钟

## 学习目标

- 解释为什么即使权重使用 NVFP4，FP8 对 KV 缓存和注意力仍然至关重要。
- 计算前沿模型在 BF16、FP8 和 NVFP4 下的 HBM 占用空间，并推理节省来自何处。
- 列出 TRT-LLM 利用的 Blackwell 特定特性（第 0 天 FP4、MTP、分离式服务、全连接原语）。
- 决定何时 TRT-LLM 的 NVIDIA 锁定值得与 Hopper 上的 vLLM 相比 7 倍的成本差距。

## 问题

2026 年推理经济的前沿是“每美元多少个 Token”。答案取决于四个堆叠选择：硬件代次（Hopper H100/H200 与 Blackwell B200/GB200）、精度（BF16 → FP8 → NVFP4）、服务引擎（vLLM 与 SGLang 与 TRT-LLM）以及编排（普通与分离式与 Dynamo）。

在 Hopper 上使用 vLLM，120B MoE 运行成本约为 $0.09 per million tokens. On Blackwell with TRT-LLM + Dynamo, the same model runs at ~$0.012——便宜 7 倍。其中部分差距来自硬件（Blackwell 每 GPU LLM 吞吐量是 Hopper 的 11-15 倍）。部分来自堆栈：FP4 权重、MTP 草稿、分离式预填充/解码以及用于 MoE 专家通信的 NVLink 5 全连接。

在 NVIDIA 堆栈之外无法复制这一点。这就是权衡——用可移植性换取经济性。理解哪些堆栈选择贡献了差距的哪一部分是本节课的重点。

## 核心概念

### 为什么 FP8 仍然是 KV 缓存的下限

2026 年的一个常见错误：假设 NVFP4 适用于所有地方。事实并非如此。KV 缓存需要 FP8（8 位浮点），因为它存储注意力键和值，这些键和值跨越很宽的动态范围。将 KV 量化到 FP4 会导致灾难性的精度损失——分布尾部下降，注意力分数崩溃。FP8 的指数位为 KV 缓存提供了所需的动态范围。

NVFP4（2025-2026）适用于权重和激活。微缩放：每个权重块都有自己的缩放因子，因此小块可以在没有逐张量缩放损失的情况下跨越不同的动态范围。对于激活，FP4 能够维持，因为层内激活的范围很小。

典型的 Blackwell 配置：

- 权重：NVFP4（4 位微缩放）。
- 激活：NVFP4。
- KV 缓存：FP8。
- 注意力累加器：FP32（softmax 稳定性）。

### TRT-LLM 使用的 Blackwell 特定原语

- **第 0 天 FP4 权重**：模型提供商直接提供 FP4 权重；TRT-LLM 无需训练后转换即可加载。无需 FP4 的 AWQ/GPTQ 步骤。
- **多 Token 预测（MTP）**：与 EAGLE（阶段 17 · 05）相同的思想，但集成到 TRT-LLM 构建中。
- **分离式服务**：预填充和解码在不同的 GPU 池上进行，KV 缓存通过 NVLink 或 InfiniBand 传输。与 Dynamo（阶段 17 · 20）相同的思想。
- **全连接通信原语**：NVLink 5 将 MoE 专家通信延迟降低至 Hopper 的 1/3。TRT-LLM 的 MoE 内核针对此进行了调优。
- **NVFP4 + MXFP8 微缩放**：Blackwell Tensor Core 上硬件加速的缩放因子处理。

### 您应该记住的数字

- HGX B200 在使用 TRT-LLM 的 GPT-OSS-120B 上达到 $0.02/M tokens。
- GB200 NVL72 通过 Dynamo（编排 TRT-LLM）达到 $0.012/M tokens。
- H100 + vLLM 在类似工作负载上约为 $0.09/M tokens。
- 三个月内 TRT-LLM 更新带来的吞吐量提升 2.8 倍（2026 年）。
- Blackwell 与 Hopper 相比，每 GPU LLM 吞吐量提升 11-15 倍。
- MLPerf 推理 v6.0（2026 年 4 月）：Blackwell 在每个提交任务中都占据主导地位。

### FP4 在质量上实际付出的代价

NVFP4 是激进的。在推理密集型工作负载（思维链、数学、长上下文代码生成）上，FP4 权重会明显退化。逐块校准可以缓解但无法消除。部署推理模型的团队通常采用 FP8 权重 + FP4 激活作为折衷方案，或者坚持使用全 FP8 的 H200。

规则：在承诺使用 NVFP4 权重之前，始终在评估集上验证任务质量。

### 为什么这是 NVIDIA 锁定决策

TRT-LLM 是 C++ + CUDA + 闭源内核。模型需要针对特定的 GPU SKU 进行编译。不支持 AMD、Intel 或 ARM。如果您的基础设施策略是多供应商，则 TRT-LLM 对于 TRT-LLM 服务层是不可行的——您仍然可以在混合硬件上使用 vLLM 进行服务。如果您是 NVIDIA 独占，则 7 倍差距为锁定付出了代价。

### 2026 年实用方案

对于每年超过 1 亿美元的推理账单，在 Hopper + vLLM 上运行会留下 7-10 倍的潜力。将成本主导的工作负载迁移到 Blackwell + TRT-LLM + Dynamo。为了模型迭代速度，在 H100 + vLLM 上保留实验层。在生产前验证每个 NVFP4 转换模型的性能。

### 分离式奖励

TRT-LLM 的分离式服务（独立的预填充和解码池）在阶段 17 · 20 中有详细介绍。在 Blackwell 上，乘数叠加：FP4 权重 × MTP 加速 × 分离式放置 × 缓存感知路由。7 倍数字假设了完整的堆栈。

```figure
pipeline-parallel
```

## 使用它

`code/main.py` 计算一个模型在三个堆栈下的 HBM 占用、解码吞吐量（内存受限区域）和每百万 Token 成本：H100 + BF16 + vLLM、H100 + FP8 + vLLM、B200 + NVFP4/FP8 + TRT-LLM。运行它以查看复合效应以及每个变化贡献的差距部分。

## 发布

本节课产生 `outputs/skill-trtllm-blackwell-advisor.md`。给定工作负载、模型大小和年度 Token 量，它决定 Blackwell + TRT-LLM 堆栈是否值得 NVIDIA 锁定。

## 练习

1. 运行 `code/main.py`。在一个 120B MoE（30% 激活参数）上，计算 H100 BF16、H100 FP8 和 B200 NVFP4/FP8 下受内存带宽限制的解码吞吐量。最大的提升来自哪里？
2. 一个客户每年在 H100 + vLLM 上花费 200 万美元。在 7 倍经济差距下，他们需要购买多少个 Blackwell GPU 才能在 12 个月内摊销迁移到 TRT-LLM 的成本？
3. 在 NVFP4 权重转换后，您在 MATH 上观察到精度下降 3 个点。给出两种恢复路径：一种质量优先（保留 FP8 权重），一种成本优先（使用领域内数据进行校准）。
4. 阅读 MLPerf v6.0 推理结果。哪个任务的 Blackwell 与 Hopper 差距最小，为什么？
5. 计算一个 405B 模型在 NVFP4 权重 + 128k 上下文的 FP8 KV 缓存下所需的 HBM。它能装进单个 GB200 NVL72 节点吗？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  FP8  |  "八位浮点"  |  8 位浮点；由于动态范围用于 KV 缓存和注意力  |
|  NVFP4  |  "四位微型"  |  NVIDIA 的 4 位微缩放 FP 格式；Blackwell 上的权重和激活  |
|  MXFP8  |  "MX 八位"  |  微缩放 FP8 变体；Blackwell Tensor Core 上硬件加速  |
|  Day-0 FP4  |  "发布 FP4 权重"  |  模型提供商已直接发布 FP4 权重的版本；无需训练后转换步骤  |
|  MTP  |  "多词元预测"  |  TRT-LLM 集成的投机解码草稿（阶段 17 · 05）  |
|  Disaggregated serving  |  "分离预填充/解码"  |  预填充和解码在独立的 GPU 池上执行；KV 通过 NVLink/IB 传输  |
|  All-to-all  |  "MoE 专家通信"  |  将词元路由到专家 GPU 的通信模式；NVLink 5 减少 3 倍  |
|  InferenceX  |  "SemiAnalysis 推理基准"  |  2026 年行业公认的每词元成本基准  |

## 延伸阅读

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — 2026 年 4 月 MLPerf 结果。
- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — NVLink 5 全互连和 MoE 内核。
- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — 官方引擎文档。
- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — TRT-LLM 之上的分离编排。
- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — 发布 Blackwell 数据的基准套件。
