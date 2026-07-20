# 综合项目 07 — 端到端微调流水线（从数据到 SFT 到 DPO 到部署服务）

> 一个在自有数据上训练、基于自有偏好进行DPO对齐、量化、推测解码，并以可衡量的每百万token成本提供服务的8B模型。2026年的开源技术栈是Axolotl v0.8、TRL 0.15、用于迭代的Unsloth、用于量化的GPTQ/AWQ/GGUF、以及使用EAGLE-3进行服务的vLLM 0.7。最终目标是实现整个流程的可复现性——从YAML输入到服务端点输出——并在2026年模型开放框架(Model Openness Framework)下发布模型卡片。

**类型：** 结业项目
**语言：** Python（流程）、YAML（配置）、Bash（脚本）
**先决条件：** 阶段2（机器学习）、阶段3（深度学习）、阶段7（Transformer）、阶段10（从头构建大语言模型）、阶段11（大语言模型工程）、阶段17（基础设施）、阶段18（安全）
**涉及阶段：** P2 · P3 · P7 · P10 · P11 · P17 · P18
**时间：** 35小时

## 问题

2026年每个严肃的AI团队都会准备一个微调流程。不是因为他们在开发前沿基础模型，而是因为下游适配——领域监督微调、针对标注偏好进行DPO、用于推测解码的蒸馏草稿、结合EAGLE-3的服务——才是真正可衡量的优势所在。Axolotl v0.8处理多GPU的监督微调配置。TRL 0.15处理DPO和GRPO。Unsloth让你快速进行单GPU迭代。集成EAGLE-3的vLLM 0.7在不损失质量的前提下将解码吞吐量提升2-3倍。工具是现成的；关键在于YAML配置、数据卫生和评估纪律。

你将在一个8B基础模型（Llama 3.3、Qwen3或Gemma 3）上依次进行监督微调和DPO，使用任务特定数据，为服务进行量化，并通过lm-evaluation-harness、RewardBench-2、MT-Bench-v2和MMLU-Pro衡量收益。你将在2026年模型开放框架下生成一张模型卡片。关键在于可复现性——一条命令即可端到端地重新运行整个流程。

## 概念

该流程包含五个阶段。**数据**：去重（MinHash / Datatrove）、质量过滤（Nemotron-CC风格分类器）、PII清洗、针对公开基准污染的分裂卫生检查。**监督微调**：Axolotl YAML配置、8xH100上的ZeRO-3、余弦调度、序列打包、2-3个周期。**DPO或GRPO**：TRL配置、1个周期、偏好对由人工标注或模型判定、beta调优。**量化**：GPTQ + AWQ + GGUF以保证部署灵活性。**服务**：集成EAGLE-3推测解码头的vLLM 0.7（或使用SpecForge的SGLang）、K8s部署、基于队列等待时间的水平自动扩缩。

消融实验是交付物：在三个任务特定基准上对比监督微调、监督微调+DPO、监督微调+GRPO。服务指标：批大小为1/8/32时的token/秒、EAGLE-3接受率、每百万token成本。安全评估：Llama Guard 4通过率。模型卡片：偏见评估、可复现种子、数据许可。

## 架构

```
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## 技术栈

- 数据：使用Datatrove去重、Nemotron-CC分类器进行质量过滤、Presidio处理PII
- 基础模型：Llama 3.3 8B、Qwen3 14B或Gemma 3 12B
- 监督微调：Axolotl v0.8，配备ZeRO-3、Flash Attention 3、序列打包
- 偏好调优：TRL 0.15用于DPO或GRPO；Unsloth用于单GPU迭代
- 量化：GPTQ（Marlin）、AWQ、通过llama.cpp实现的GGUF
- 服务：集成EAGLE-3推测解码的vLLM 0.7（或SGLang 0.4 + SpecForge）
- 评估：lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro
- 安全评估：Llama Guard 4、ShieldGemma-2
- 基础设施：Kubernetes + NVIDIA设备插件、基于队列等待度量的水平自动扩缩
- 可观测性：W&B用于训练、Langfuse用于推理

## 动手构建

1. **数据流程。** 对原始语料运行Datatrove去重。应用Nemotron-CC风格质量分类器。Presidio清洗PII。使用显式随机种子写入训练/验证集划分。

2. **污染检查。** 对每个验证集划分，计算其与MMLU-Pro、MT-Bench-v2、RewardBench-2测试集的MinHash相似度。拒绝任何重叠。

3. **Axolotl监督微调。** 包含ZeRO-3、FA3、序列打包的YAML配置。在8xH100上运行2-3个周期。记录到W&B。

4. **TRL DPO / GRPO。** 使用监督微调检查点，在偏好对上运行一个周期的DPO（或使用数学/代码可验证奖励的GRPO）。对beta进行超参搜索。

5. **量化。** 生成三种量化版本：GPTQ-INT4-Marlin、AWQ-INT4、用于llama.cpp的GGUF-Q4_K_M。记录大小和标称吞吐量。

6. **使用推测解码进行服务。** vLLM 0.7配置，使用通过Red Hat Speculators训练的EAGLE-3草稿头。在批大小1/8/32下测量接受率和尾延迟。在与Anthropic/OpenAI相同的评估上报告每百万token成本。

7. **评估矩阵。** 在基础模型、监督微调、监督微调+DPO、监督微调+GRPO上运行lm-eval-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro。生成表格。

8. **安全评估。** 开发集上的Llama Guard 4通过率。ShieldGemma-2输出过滤器。

9. **模型卡片。** MOF 2026模板：数据、训练、评估、安全、许可、可复现性部分（含YAML配置和提交哈希）。

## 使用它

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## 发布

`outputs/skill-finetuning-pipeline.md` 描述了交付物。一条命令运行从数据到监督微调、DPO、量化、服务、评估的整个流程，并输出模型卡片和已服务的端点。

|  权重  |  标准  |  衡量方式  |
|:-:|---|---|
|  25  |  与基础模型相比的评估差值  |  在目标任务（MMLU-Pro、MT-Bench-v2、任务特定）上的可衡量提升  |
|  20  |  流程可复现性  |  一条命令以相同随机种子端到端重新运行  |
|  20  |  数据卫生  |  去重率、PII清洗覆盖率、污染检查通过  |
|  20  |  服务效率  |  批大小为1/8/32时的token/秒、EAGLE-3接受率、每百万token成本  |
|  15  |  模型卡片和安全评估  |  2026年MOF完整性 + Llama Guard 4通过率  |
|  **100**  |   |   |

## 练习

1. 在相同任务特定基准上运行监督微调、监督微调+DPO、监督微调+GRPO。报告哪种偏好方法胜出以及优势大小。

2. 将Llama 3.3 8B替换为Qwen3 14B。在匹配质量下测量每百万token成本。

3. 在领域数据与通用ShareGPT数据上测量EAGLE-3接受率。报告差异及其对延迟预算的影响。

4. 注入1%的污染（将MMLU-Pro答案泄露到训练数据中）并重新运行评估。观察MMLU-Pro准确率不切实际地飙升。构建一个能捕获此问题的污染检查CI门控。

5. 添加LoRA监督微调作为全参数微调的替代方案。在内存降低10倍的情况下衡量质量差距。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
|  Axolotl  |  "监督微调训练器"  |  统一的YAML驱动训练器，用于监督微调、DPO和蒸馏  |
|  TRL  |  "偏好调优器"  |  Hugging Face库，用于大语言模型上的DPO、GRPO、PPO  |
|  GRPO  |  "组相对策略优化"  |  DeepSeek R1的基于可验证奖励的强化学习方案  |
|  EAGLE-3  |  "推测性解码草稿"  |  预测前方N个token的草稿头；vLLM使用目标模型验证  |
|  MOF  |  "模型开放框架"  |  2026年标准，用于根据数据、代码、许可对模型发布进行分级  |
|  污染检查  |  "分割卫生"  |  基于MinHash的检测测试集泄露到训练集的方法  |
|  接受率  |  "EAGLE/MTP指标"  |  目标模型接受的草稿token比例  |

## 延伸阅读

- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/) — 参考SFT/DPO训练器
- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/) — DPO和GRPO参考实现
- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/) — 单GPU迭代参考
- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/) — GRPO方法论
- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/) — 参考服务栈
- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/) — 备选推测性解码训练器
- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/) — 开放发布分级标准
- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/) — 规范评估运行器
