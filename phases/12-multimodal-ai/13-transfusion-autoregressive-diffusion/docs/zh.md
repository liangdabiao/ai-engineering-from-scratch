# 输血(Transfusion)：在一个Transformer中实现自回归文本与扩散图像

> Chameleon和Emu3将所有赌注押在离散令牌上。它们有效，但量化瓶颈显而易见——图像质量低于连续空间扩散模型。输血(Transfusion)(Meta, Zhou等人，2024年8月)采取了相反的赌注：保持图像的连续性，完全弃用VQ-VAE，并用两个损失训练一个Transformer。文本令牌使用下一个令牌预测。图像块使用流匹配/扩散损失。两个目标优化相同的权重。Stable Diffusion 3 (MMDiT) 的架构是近亲。本节解读Transfusion论文，构建一个玩具双损失训练器，并追踪让一个Transformer同时处理两种任务的注意力掩码。

**类型：** 构建
**语言：** Python (标准库，基于MNIST量级的玩具双损失训练器)
**前置条件：** 阶段12 · 11 (Chameleon)，阶段8 (生成式AI)
**时间：** ~180分钟

## 学习目标

- 构建一个在一个骨干网络上运行两个损失（文本令牌的NTP，图像块的扩散MSE）的Transformer。
- 解释为什么图像块间的双向注意力加上文本令牌上的因果注意力是正确的掩码选择。
- 在计算、质量和代码复杂度上比较Transfusion风格（连续图像，扩散损失）与Chameleon风格（离散图像，NTP）。
- 指明MMDiT的贡献：每个块中模态特定的权重，残差流上的联合注意力。

## 问题

离散与连续图像令牌的争论比LLM更久远。连续表示（原始像素，VAE潜变量）保留了细节。离散令牌（VQ索引）适合Transformer的原生词汇，但在量化步骤中丢失了细节。

Chameleon / Emu3采用了离散方式：一个损失，一个架构，但图像保真度受限于分词器质量。

扩散模型采用连续方式：出色的图像质量，但与LLM分离，复杂的噪声调度工程，以及与文本生成缺乏干净的集成。

Transfusion提出：我们能不能两者兼得？保持图像的连续性，仍然训练一个模型，使用两个损失缝合到一个梯度步骤中。

## 核心概念

### 双损失架构

一个单一的仅解码器Transformer处理包含以下内容的序列：

- 文本令牌（离散，来自BPE词汇表）。
- 图像块（连续，16x16像素块通过线性投影映射到隐藏维度——与ViT编码器的输入相同）。
- `<image>`和`</image>`标记，标明连续块的所在位置。

前向传播运行一次。每个令牌的损失从两个头中选择一个：

- 对于文本令牌：在词汇表logits头上的标准交叉熵损失。
- 对于图像块：在连续块上的扩散损失——预测添加到每个块的噪声。

梯度流过共享的Transformer本体。两个损失同时改进共享权重。

### 注意力掩码：因果文本 + 双向图像

文本令牌必须是因果的——不能让文本令牌关注未来的文本，否则教师强制会破坏。然而，图像块代表单个快照；它们应该在同一图像块内双向相互关注。

掩码：

```
M[i, j] = 1 if:
  (i is text and j is text and j <= i)   # causal for text
  OR (i is image and j is image and same_image_block(i, j))   # bidirectional within image
  OR (i is text and j is image and j < i_image_end)   # text attends to previous images
  OR (i is image and j is text and j < i_image_start)   # image attends to preceding text
```

在训练和推理时实现为块三角掩码。

### Transformer内部的扩散损失

扩散损失是标准的：向图像块添加噪声，要求模型预测噪声（或等价地预测干净块）。Transfusion的版本使用流匹配——预测从有噪声到干净的流场。

在训练期间：
1. 对于每个图像块x0，采样一个随机时间步t。
2. 采样噪声ε，计算xt = (1-t) * x0 + t * ε（流匹配的线性插值）。
3. Transformer预测v_theta(xt, t)；损失 = MSE(v_theta(xt, t), ε - x0)。
4. 与来自同一序列的文本NTP损失一起反向传播。

在推理时，生成过程是：
- 文本令牌：标准的自回归采样。
- 图像块：扩散采样循环（通常10-30步），以前面的文本令牌为条件。

### MMDiT：Stable Diffusion 3的变体

Stable Diffusion 3 (Esser等人，2024年3月）在几乎同一时间推出了MMDiT（多模态扩散Transformer）。这些架构是兄弟关系。

MMDiT的关键区别：

- 每个块具有模态特定的权重。每个Transformer块具有分离的Q、K、V和MLP权重，用于文本令牌和图像块。注意力是联合的（跨模态）；其他一切都是模态特定的。
- 整流流训练。一种特定的流匹配变体，具有已知的采样和比DDPM更简单的数学。
- 规模。MMDiT是SD3（2B和8B参数变体）的骨干。Transfusion的论文扩展到7B。

两者都收敛到同一个核心思想：一个Transformer对文本运行NTP，对连续图像表示运行扩散。

### 为什么这比Chameleon风格更胜一筹

连续扩散与离散NTP在图像生成上的质量差距是可测量的。Transfusion论文报告如下：

- 在7B参数规模下，在FID指标上比同等规模的Chameleon风格模型领先3-5个点。
- 不需要训练分词器——图像编码器更简单（线性投影到隐藏层，与ViT的输入层相同）。
- 推理时可以对图像块去噪进行并行化，而自回归图像token则不行。

缺点：Transfusion是一个双损失模型，使得训练动态更棘手。损失权重需要调节。NTP和扩散之间的调度不匹配可能导致其中一个头部主导。

### 后续发展

Janus-Pro（第12.15课）通过解耦视觉编码器用于理解和生成——一个使用SigLIP，另一个使用VQ——同时共享Transformer主体，改进了Transfusion的思想。Show-o（第12.14课）将扩散替换为离散扩散（掩码预测）。在Transfusion之后，统一生成家族迅速分支。

2026年生产级的能生成图像的VLM——Gemini 3 Pro、GPT-5、Claude Opus 4.7的图像生成路径——几乎肯定使用了这个家族的某个变种。具体细节是专有的。

## 使用它

`code/main.py` 在类似MNIST的小问题上构建了一个玩具Transfusion：

- 文本描述是描述数字（0-9）的短整数序列。
- 图像是4x4字节网格。
- 一对共享权重的线性投影充当Transformer的替代品；文本上的NTP损失，噪声块上的MSE损失。
- 训练循环交替两个损失，注意力掩码是显式的。
- 生成在一次前向传播中产生文本描述和4x4图像。

Transformer是玩具模型。双损失流程、注意力掩码构建和推理循环才是真正的产物。

## 发布

本课产出`outputs/skill-two-loss-trainer-designer.md`。给定一个新的多模态训练任务（文本+图像、文本+音频、文本+视频），它设计双损失调度（损失权重、掩码形状、共享块与特定模态块的选择）并标记实现风险。

## 练习

1. 一个Transfusion风格的模型训练70%的文本token和30%的图像块。图像扩散损失的量级大约是文本NTP损失的10倍。什么损失权重能使它们平衡？

2. 为序列实现块三角掩码：`[T, T, <image>, P, P, P, P, </image>, T]`。将每个条目标记为0或1。

3. MMDiT具有特定模态的QKV权重。与Transfusion的完全共享Transformer相比，这会增加多少参数开销？在7B参数规模下，这值得吗？

4. 生成：给定文本提示，模型运行NTP 50个token，然后到达`<image>`，然后在256个块上运行20步去噪扩散。总共有多少次前向传播？

5. 阅读SD3论文第3节。描述整流流（Rectified Flow）以及为什么它比DDPM收敛所需的推理步骤更少。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
| 双损失训练  |  "NTP + 扩散"  |  同一个Transformer在同一个梯度步骤中优化文本token上的交叉熵和连续图像块上的MSE损失 |
| 流匹配  |  "整流流"  |  扩散变体，预测从噪声到干净数据的速度场；数学比DDPM更简单 |
| MMDiT  |  "多模态DiT"  |  Stable Diffusion 3的架构：联合注意力、特定模态的MLP和归一化层 |
| 块三角掩码  |  "因果文本 + 双向图像"  |  注意力掩码在文本上是因果的，但在图像区域内是双向的 |
| 连续图像表示  |  "无VQ"  |  图像块作为实值向量，而不是整数码本索引 |
| 速度预测  |  "v-参数化"  |  网络输出是噪声与数据之间的速度场，而不是噪声本身 |

## 延伸阅读

- [Zhou et al. — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser et al. — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao et al. — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
