# 生产量化 — AWQ, GPTQ, GGUF K-quants, FP8, MXFP4/NVFP4

> 量化格式并非通用选择，而是取决于硬件、推理引擎和工作负载。GGUF Q4_K_M 或 Q5_K_M 适用于CPU和边缘设备，通过llama.cpp和Ollama部署。GPTQ 在vLLM中当需要在同一基座上使用多LoRA时胜出。AWQ 配合Marlin-AWQ内核在7B规模模型上实现约741 tok/s，且在INT4下Pass@1最佳，是2026年数据中心生产环境的默认选择。FP8 在Hopper、Ada和Blackwell上保持折中地位——近乎无损且广泛支持。NVFP4和MXFP4（Blackwell微观缩放）较为激进，需要逐块验证。有两个陷阱需要注意：校准数据集必须匹配部署领域，且KV缓存与权重量化是分离的——AWQ课程的教训是“我的模型现在是4 GB”，却忘记了生产批次大小下10-30 GB的KV缓存。

**类型:** 学习
**语言:** Python (stdlib, 各格式的简易内存和吞吐量比较)
**前置条件:** 阶段10·13 (量化基础), 阶段17·04 (vLLM Serving内部原理)
**时间:** 约75分钟

## 学习目标

- 说出2026年六种生产量化格式及其适用场景。
- 根据硬件（CPU vs GPU, Hopper vs Blackwell）、引擎（vLLM, TRT-LLM, llama.cpp）和工作负载（常规聊天、推理、多LoRA）选择格式。
- 计算选定格式下节省的权重内存和未涉及的KV缓存大小。
- 说出会导致量化模型在领域流量上性能下降的校准数据集陷阱。

## 问题

量化减少了内存和HBM带宽，这正是解码所需要的。一个FP16 70B模型权重为140 GB。将权重量化为INT4（AWQ或GPTQ）后模型为35 GB——可放入一个H100，并留有KV缓存空间，这一点很重要，因为在128个并发序列和2k上下文下，KV缓存本身就需要20-30 GB。

但量化并非没有代价。激进的量化会降低质量，尤其是在推理密集型任务上。不同格式与不同引擎配合。不同硬件原生支持不同精度。2026年的格式动物园是真实存在的，你不能照搬别人的选择——必须基于自己的技术栈进行选择。

## 核心概念

### 六种格式

|  格式  |  比特  |  适用场景  |  引擎  |
|--------|------|-----------|---------|
|  GGUF Q4_K_M / Q5_K_M  |  4-5  |  CPU, 边缘设备, 笔记本电脑  |  llama.cpp, Ollama  |
|  GPTQ  |  4-8  |  vLLM上的多LoRA  |  vLLM, TGI  |
|  AWQ  |  4  |  数据中心GPU生产  |  vLLM (Marlin-AWQ), TGI  |
|  FP8  |  8  |  Hopper/Ada/Blackwell数据中心  |  vLLM, TRT-LLM, SGLang  |
|  MXFP4  |  4  |  Blackwell多用户  |  TRT-LLM  |
|  NVFP4  |  4  |  Blackwell多用户  |  TRT-LLM  |

### GGUF — CPU/边缘设备的默认选择

GGUF是一种文件格式，而非量化方案本身——它将K-quant变体（Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0）打包在一个容器中。Q4_K_M和Q5_K_M是生产默认值——在4-5比特下接近BF16质量。是CPU或边缘设备服务的最佳选择，因为llama.cpp是目前最快的CPU推理引擎。

vLLM中的吞吐量损失：7B模型约93 tok/s——该格式未针对GPU内核优化。仅在部署目标为CPU/边缘设备时使用GGUF。其他情况不要使用。

### GPTQ — vLLM中的多LoRA

GPTQ是一种带有校准过程的训练后量化算法。Marlin内核使其在GPU上速度很快（相比非Marlin GPTQ加速2.6倍）。7B模型约712 tok/s。

独特优势：GPTQ-Int4支持vLLM中的LoRA适配器。如果你在服务一个基座模型加上10-50个微调变体（每个作为LoRA），GPTQ是你的选择。截至2026年初，NVFP4尚不支持LoRA。

### AWQ — 数据中心GPU的默认选择

激活感知权重量化(Activation-aware Weight Quantization)。在量化过程中保护约1%最重要的权重。Marlin-AWQ内核：相比朴素实现加速10.9倍。7B模型约741 tok/s，INT4格式中Pass@1最佳。

除非你需要多LoRA（选择GPTQ）或激进的Blackwell FP4（选择NVFP4），否则为新GPU服务选择AWQ。

### FP8 — 可靠的折中选择

8位浮点数。近乎无损。广泛支持。Hopper Tensor Cores原生加速FP8。Blackwell继承。FP8是2026年质量不可妥协时（推理、医疗、代码生成）的安全默认选择。内存节省仅为INT4的一半，但质量风险远低于INT4。

### MXFP4 / NVFP4 — Blackwell的激进选择

微观缩放FP4。每个权重块有自己的缩放因子。激进但由Blackwell Tensor Cores硬件加速。每token字节数相比FP8减半——这是阶段17·07中的经济优势。

注意事项：
- 尚不支持LoRA（2026年初）。
- 在推理密集型工作负载上质量下降明显。
- 需在每个模型上使用你的评估集进行验证。

### 校准陷阱

AWQ和GPTQ需要一个校准数据集——通常是C4或WikiText。对于领域模型（代码、医学、法律），使用通用网络文本进行校准会让算法在决定保留哪些权重时做出错误判断。HumanEval上的Pass@1可能会下降几个百分点。

解决方法：使用领域内数据校准。通常数百个领域样本就足够了。在发布前在评估集上测试。

### KV缓存陷阱

AWQ将权重量化为4位。KV缓存是独立的，保持FP16/FP8。对于使用AWQ的70B模型：

- 权重：约35 GB（INT4，从140 GB压缩而来）。
- KV缓存，128并发×2k上下文：约20 GB。
- 激活值：约5 GB。
- 总计：约60 GB——可放入H100 80GB。

天真的认为“我将模型量化到4 GB”会忘记另外30-50 GB。要整体规划HBM预算。

另外，KV缓存量化（FP8 KV或INT8 KV）是另一种选择，有其自身的权衡——它直接影响注意力精度，并非无代价的胜利。

### AWQ INT4对推理有危害

思维链、数学、长上下文代码生成——这些任务在激进量化下明显受损。AWQ INT4在MATH上损失约3-5个百分点。对于推理密集型工作负载，使用FP8或BF16发布；接受内存成本。

### 2026年选择指南

- CPU/边缘部署：GGUF Q4_K_M。完成。
- GPU部署，常规对话，无LoRA：AWQ。
- GPU部署，多LoRA：GPTQ with Marlin。
- 推理工作负载：FP8。
- Blackwell数据中心，经过验证的质量：NVFP4 + FP8 KV。
- 不确定时：对每个候选格式运行1000样本评估。

```figure
gpu-memory-breakdown
```

## 使用它

`code/main.py` 计算多种模型规模下六种格式的内存占用（权重+KV+激活值）和相对吞吐量。显示KV缓存何时占主导，权重压缩何时有效，以及FP8何时是安全选择。

## 发布

本节课生成`outputs/skill-quantization-picker.md`。给定硬件、模型大小、工作负载类型和质量容忍度，选择一种格式并生成校准/验证计划。

## 练习

1. 运行`code/main.py`。对于一个70B模型，128并发，2k上下文，计算每种格式的总HBM。哪种格式可以装入一块H100 80GB？
2. 你有一个7B代码模型。选择一种格式并说明理由。如果你对质量容忍度的判断错误，恢复路径是什么？
3. 计算为医学领域模型校准AWQ所需的校准数据集大小。为什么数据并非越多越好？
4. 阅读Marlin-AWQ内核论文或发布说明。用三句话解释为什么AWQ在7B模型上达到741 tok/s，而原始GPTQ约为712。
5. 何时将AWQ权重与FP8 KV缓存结合使用有意义，而何时保持KV为BF16？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  GGUF  |  "llama.cpp格式"  |  捆绑了K-量化变体的文件格式；CPU/边缘默认  |
|  Q4_K_M  |  "Q4 K M"  |  4位K-量化中等；生产环境GGUF默认  |
|  GPTQ  |  "gee pee tee q"  |  带校准的后训练INT4；在vLLM中支持LoRA  |
|  AWQ  |  "a w q"  |  激活感知INT4；Marlin内核；INT4中最佳Pass@1  |
|  Marlin内核  |  "快速INT4内核"  |  用于Hopper上INT4的自定义CUDA内核；10倍加速  |
|  FP8  |  "8位浮点"  |  Hopper/Ada/Blackwell上的安全精度默认  |
|  MXFP4 / NVFP4  |  "微缩放4位"  |  Blackwell 4位浮点，带逐块缩放因子  |
|  校准数据集  |  "校准数据"  |  用于选择量化参数的输入文本；必须匹配领域  |
|  KV缓存量化  |  "KV INT8"  |  与权重分开的选择；影响注意力精度  |

## 延伸阅读

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — 对比基准。
- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — 各格式的吞吐量数据。
- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — 按格式选择的指南。
- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — 支持的格式和标志。
- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — 原始AWQ公式。
- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — 原始GPTQ公式。
