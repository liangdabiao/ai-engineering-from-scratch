# Janus-Pro：用于统一多模态模型的解耦编码器

> 统一多模态模型存在一个不可避免的张力。理解需要语义特征——SigLIP或DINOv2输出的向量富含概念级别信息。生成需要重建友好的编码——VQ令牌能够重新组合成清晰的像素。这两个目标在单一编码器中无法兼容。Janus（深度求索，2024年10月）和Janus-Pro（深度求索，2025年1月）认为解决方案是停止尝试：将两个编码器解耦。在理解任务中通过SigLIP路由，在生成任务中通过VQ令牌化器路由，而Transformer主体则在任务之间共享。在7B参数规模下，Janus-Pro在GenEval上击败了DALL-E 3，同时在MMMU上与LLaVA持平。本课将解读为什么两个编码器比一个编码器更有效。

**类型：** 构建
**语言：** Python（标准库，双编码器路由 + 共享主体信号）
**先修知识：** 第12章·第13课（Transfusion），第12章·第14课（Show-o）
**时间：** 约120分钟

## 学习目标

- 解释为什么单一共享编码器会损害理解或生成质量。
- 描述Janus-Pro的路由：理解任务中输入侧使用SigLIP特征，生成任务中输入和输出侧都使用VQ令牌。
- 追踪使Janus-Pro超越Janus的数据混合规模扩展。
- 比较解耦（Janus-Pro）、耦合连续（Transfusion）和耦合离散（Show-o）架构。

## 问题

统一模型在理解和生成任务之间共享一个Transformer主体。先前的尝试（Chameleon、Show-o、Transfusion）都在两个方向上使用同一个视觉令牌化器。这个令牌化器是一种妥协：

- 针对重建优化（生成）：VQ-VAE捕获细粒度像素细节，但产生的令牌语义连贯性弱。
- 针对语义优化（理解）：SigLIP嵌入将“猫”图像与“猫”令牌聚类在一起，但无法实现良好的重建。

Show-o和Transfusion为此付出了可见的质量代价——在其中一个方向上表现不佳。Janus-Pro提出：当任务需求不同时，为何要求使用同一个令牌化器？

## 核心概念

### 解耦的视觉编码

Janus-Pro的架构将两个编码器分离：

- 理解路径。输入图像 → SigLIP-SO400m → 2层MLP → Transformer主体。
- 生成路径。输入图像（如果基于现有图像进行条件生成）→ VQ令牌化器 → 令牌ID → Transformer主体。
- 输出生成。由Transformer预测的图像令牌 → VQ解码器 → 像素。

Transformer主体是共享的。主体上游和下游的所有部分都是任务特定的。

输入通过提示格式进行消歧：`<understand>`标签路由至SigLIP；`<generate>`路由至VQ。或者路由由任务隐式决定。

### 为什么有效

理解损失获得SigLIP特征，而CLIP风格的预训练已将其调整为语义相似性。模型的感知基准测试优于Show-o/Transfusion，因为输入特征更适合该任务。

生成损失获得VQ令牌，而令牌化器已将其调整为重建。图像质量优于Show-o，因为VQ编码能够干净地重新组合成像素。

共享的Transformer主体看到两种输入分布（SigLIP和VQ），并学会同时处理两者。其主张是：足够的数据和足够的参数，主体能够吸收这种切换。

### 数据规模扩展——Janus vs Janus-Pro

Janus（原始版，arXiv 2410.13848）引入了解耦，但规模较小（1.3B参数，有限数据）。Janus-Pro（arXiv 2501.17811）进行了扩展：

- 7B参数（vs 1.3B）。
- 第一阶段（对齐）使用9000万图像-文本对，高于之前的7200万。
- 第二阶段（统一）使用7200万对，高于之前的2600万。
- 第三阶段增加了20万图像生成指令样本。

结果：Janus-Pro-7B在MMMU上与LLaVA持平（60.3 vs ~58），并在GenEval上击败DALL-E 3（0.80 vs 0.67）。一个开源模型，在统一谱系的两个方面都具备竞争力。

### JanusFlow——整流流变体

JanusFlow（arXiv 2411.07975）将VQ生成路径替换为整流流生成路径（连续）。分割变为SigLIP用于理解 + 整流流用于生成。质量上限进一步提升。架构仍然是解耦编码器-共享主体。

### 共享主体的任务

Transformer主体处理统一的序列，但接收两种输入分布。其任务是：

- 对于理解：消费SigLIP特征 + 文本令牌 → 自回归生成文本。
- 对于生成：消费文本令牌 + （可选图像VQ令牌） → 自回归生成图像VQ令牌。

主体没有每个块的模态特定权重。它是你会在Qwen或Llama内部找到的标准文本风格Transformer，外加两个输入适配器。

有趣的是，这意味着Janus-Pro的主体可以从预训练的大语言模型初始化。Janus-Pro确实从DeepSeek-MoE-7B初始化。这个选择很重要：大语言模型贡献了从头开始训练的统一模型难以达到的推理能力。

### 与InternVL-U的比较

InternVL-U（第12章第10课）是2026年的后续工作。它结合了：

- 原生多模态预训练（InternVL3骨干）。
- 解耦编码器路由（SigLIP输入，VQ + 扩散头输出）。
- 统一的理解 + 生成 + 编辑。

InternVL-U 将 Janus-Pro 的架构选择纳入一个更大的框架。解耦编码器(decoupled encoder)的思想如今已成为大规模统一模型的默认方案。

### 局限性

解耦编码器增加了架构复杂性。需要训练两个分词器(tokenizer)、维护两条输入路径、处理两组失败模式。对于不需要生成功能的产品，Janus-Pro 过于复杂——选择 LLaVA 系列的理解模型即可。

对于不需要理解功能的产品，Janus-Pro 又大材小用——选择 Stable Diffusion 3 或 Flux 模型即可。

对于同时需要的产品，Janus-Pro 目前是开放架构的参考标准。

## 使用它

`code/main.py` 模拟 Janus-Pro 的路由机制：

- 两个模拟编码器：类 SigLIP 编码器（生成256维语义向量）和类 VQ 编码器（生成整数码）。
- 一个提示路由器，根据任务标签(task tag)选择编码器。
- 一个共享主体（占位），处理无论由哪个编码器生成的令牌序列。
- 一个从阶段1（对齐）切换到阶段3（指令微调）的加权采样调度。

打印三个示例的路由路径：图像问答(image QA)、文生图(T2I)、图像编辑(image editing)。

## 发布

本课程产出 `outputs/skill-decoupled-encoder-picker.md`。给定一个希望以前沿质量实现统一生成与理解的产品，它会选择 Janus-Pro、JanusFlow 或 InternVL-U，并附带具体的数据规模推荐。

## 练习

1. Janus-Pro-7B 在 GenEval 上击败了 DALL-E 3。解释为什么一个 7B 的开放模型在生成方面能比肩前沿专有模型，但在理解方面则不然。

2. 实现一个路由器函数：给定提示文本，分类为 `understand` 或 `generate`。如何处理像“描述然后素描”这样的模糊提示？

3. JanusFlow 用整流流(rectified flow)替换了 VQ 路径。变换器主体现在输出什么？损失函数有什么变化？

4. 提出一个第四个任务，Janus-Pro 架构可以通过增加一个解耦编码器来处理该任务。示例：图像分割（DINO 风格）、深度估计（MiDaS 风格）。

5. 阅读 Janus-Pro 第4.2节关于数据规模的内容。与 Janus 相比，哪个数据阶段对 T2I 质量提升贡献最大？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
|  解耦编码(Decoupled encoding)  |  "两个视觉编码器"  |  每个方向独立的分词器或编码器：语义编码用于理解，重建编码用于生成  |
|  共享主体(Shared body)  |  "一个变换器"  |  单个变换器处理任一编码器的输出；无模态特定权重  |
|  用于理解的 SigLIP  |  "语义特征"  |  CLIP 系列视觉塔提供丰富的概念特征但重建效果差  |
|  用于生成的 VQ  |  "重建码"  |  向量量化(vector-quantized)的令牌，可干净地解码回像素  |
|  JanusFlow  |  "整流流变体"  |  采用连续流匹配(continuous flow-matching)生成头替代 VQ 的 Janus-Pro  |
|  路由标签(Routing tag)  |  "任务标签"  |  提示标记（`<understand>` / `<generate>`），用于选择输入编码器  |

## 延伸阅读

- [Wu et al. — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen et al. — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma et al. — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong et al. — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)
