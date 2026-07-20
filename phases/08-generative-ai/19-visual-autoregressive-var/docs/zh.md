# 视觉自回归建模(VAR)：下一尺度预测

> 扩散模型在时间上迭代采样（去噪步骤）。VAR在尺度上迭代采样——它预测1x1令牌(token)，然后2x2，然后4x4，直到最终分辨率，每个尺度以前一个尺度为条件。2024年的论文表明，VAR在图像生成上匹配GPT风格的缩放定律(Scaling Law)，并在相同计算预算下击败了DiT。本节课构建核心机制。

**类型：**构建
**语言：**Python（使用PyTorch）
**先修课程：**阶段7第03课（多头注意力Multi-Head Attention），阶段8第06课（DDPM）
**时间：**约90分钟

## 问题

自回归生成主导了语言建模，因为它可预测地缩放：更多计算、更多参数、更低困惑度(Perplexity)、更好输出。2024年之前，图像生成有两个主要的自回归(AR)尝试：PixelRNN/PixelCNN（逐像素）和DALL-E 1 / Parti / MuseGAN（在VQ-VAE编码上逐令牌）。

两者都遭受了生成顺序问题。像素和令牌以2D网格排列，但自回归模型必须以1D光栅顺序访问它们。早期角落的像素不知道图像最终会变成什么。生成质量的缩放比文本上的GPT更差，并且在匹配的计算量下从未达到扩散模型的质量。

VAR通过改变生成内容来修复生成顺序问题。VAR不是在空间中逐个预测图像令牌，而是以递增分辨率预测整个图像。步骤1：预测1x1令牌（整体图像"摘要"）。步骤2：预测2x2令牌网格（更粗糙的特征）。步骤3：预测4x4网格。步骤K：预测最终的(H/8)x(W/8)网格。

每个尺度关注所有先前尺度（在"尺度顺序"上因果地），并在其自身尺度内并行。顺序问题消失了：尺度k的整个图像在一次Transformer传递中生成。

## 核心概念

### VQ-VAE多尺度分词器(Tokenizer)

VAR需要一个**多尺度离散分词器**。对于图像x，它产生一系列分辨率逐渐提高的令牌网格：

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

每个z_k使用相同的码本(codebook)（典型大小4096-16384）。每个尺度的分词不是独立的——它被训练成使得每个尺度的残差之和重建f：

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

这是一种**残差VQ**变体。尺度k捕获尺度1..k-1遗漏的内容。解码器取所有尺度嵌入之和并生成图像。

多尺度VQ分词器训练一次（如VQGAN），然后冻结。所有生成工作都由顶部的自回归模型完成。

### 下一尺度预测(Next-Scale Prediction)

生成模型是一个Transformer，它看到所有先前尺度的令牌并预测下一尺度的令牌。

输入序列结构：
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

位置嵌入编码尺度索引和尺度内的空间位置。注意力在尺度顺序上是因果的：尺度k、位置(i,j)的令牌可以关注尺度1..k的所有令牌，以及尺度k自身中在所用任何尺度内顺序中较早出现的令牌（VAR使用固定的位置注意力，没有尺度内因果性——一个尺度内的所有位置并行预测）。

训练损失：在每个尺度k，给定所有先前尺度的令牌预测令牌z_k。离散VQ码上的交叉熵损失。与GPT结构相同，只是"序列"现在按尺度结构化。

### 生成

推理时：
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

对于K=10个尺度，生成需要10次Transformer前向传递。每次传递并行产生其整个尺度——一个尺度内没有逐令牌自回归。对于256x256图像，这大约是10次传递，而DiT需要28-50次。

### 为什么下一尺度胜过下一令牌

三个结构性优势：
1. **由粗到细符合自然图像统计。**人类视觉感知和图像数据集都表现出尺度依赖的规律性：低频结构稳定且可预测；高频细节依赖于低频内容。下一尺度预测利用了这一点。
2. **尺度内并行生成。**与GPT风格的令牌自回归不同，VAR在一个步骤中产生一个尺度的所有令牌。有效生成长度是对数尺度而不是线性。
3. **无生成顺序偏差。**尺度k的令牌可以看到尺度k-1的所有令牌；没有"左侧"或"上方"偏差迫使早期令牌在晚期上下文可用之前做出决定。

### 缩放定律(Scaling Law)

Tian等人证明，VAR在ImageNet上的FID遵循幂律缩放曲线——就像GPT在困惑度上一样。参数或计算量加倍可靠地使误差减半。这是第一个像语言模型一样清晰地展示这种缩放行为的图像生成模型。结果是VAR尺度的预测可以从计算量中预测，而不是针对每个架构的经验猜测。

### 与扩散的关系

VAR和扩散共享相同的数据压缩理念：两者都将生成问题分解为一系列更简单的子问题。

- 扩散：逐渐添加噪声，学习撤销一步。
- VAR：逐渐增加分辨率，学习预测下一个尺度。

它们是解决问题的不同轴。两者都产生易处理的条件分布。经验上，VAR在推理时更快（更少的传递，所有尺度内并行），并在类条件ImageNet上匹配或击败DiT。文本条件VAR（VARclip, HART）是一个活跃的研究方向。

## 动手构建

在`code/main.py`中，你将：
1. 在一个合成的“图像”数据集（二维高斯环）上构建一个小型**多尺度VQ分词器**。
2. 训练一个**VAR风格变换器**来按尺度预测下一个分词的token。
3. 通过调用变换器4次（4个尺度）并解码来进行采样。
4. 验证尺度排序训练使得每个尺度内的生成是并行的。

这是一个玩具实现。关键在于看到尺度结构的注意力掩码和尺度内并行生成实际工作。

## 发布

本节课涉及`outputs/skill-var-tokenizer-designer.md`——设计多尺度分词器的技能：尺度数量、尺度比例、码本大小、残差共享、解码器架构。

## 练习

1. **尺度数量消融实验。** 使用4、6、8、10个尺度训练VAR。衡量重建质量与自回归步骤次数的关系。尺度越多=残差越细=质量越好但步骤更多。

2. **码本大小。** 使用码本大小为512、4096、16384训练分词器。更大的码本带来更好的重建但预测更难。找到拐点。

3. **尺度内并行性检查。** 对于训练好的VAR，显式测量注意力模式。在尺度k内，模型是否关注跨尺度位置而不关注尺度内位置？验证掩码实现。

4. **VAR与DiT的缩放比较。** 对于相同的ImageNet类别条件任务，在匹配参数预算下（如33M、130M、458M）训练VAR和DiT。绘制FID与计算量的关系。在每个规模上VAR应该领先于DiT——在小规模上重现论文结果。

5. **文本条件控制。** 扩展VAR以通过adaLN将文本嵌入（CLIP池化）作为额外条件输入。这是HART方法。在文本对齐采样上FID改善多少？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|----------------------|
|  VAR  |  "视觉自回归"  |  通过在VQ token网格金字塔上按尺度预测下一步来生成图像  |
|  按尺度预测  |  "先粗后细"  |  模型在递增的分辨率尺度上预测token，并基于所有先前尺度进行条件化  |
|  多尺度VQ分词器  |  "残差VQ"  |  产生K个递增分辨率token网格的VQ-VAE，解码器将所有尺度求和  |
|  尺度k  |  "金字塔层级k"  |  K个分辨率层级之一，从k=1时的1x1到k=K时的(H/p)x(W/p)  |
|  尺度内并行  |  "每尺度一次前向"  |  尺度k的所有token在一次变换器前向中预测，而非自回归  |
|  跨尺度因果  |  "尺度排序注意力"  |  尺度k的token可以关注尺度1..k的所有内容，但不能关注尺度k+1..K  |
|  残差VQ  |  "加性分词"  |  每个尺度的token编码较低尺度留下的残差；解码器将所有尺度嵌入求和  |
|  VAR缩放定律  |  "图像GPT缩放"  |  FID遵循计算量上的可预测幂律，类似于语言模型的困惑度  |
|  HART  |  "混合VAR+文本"  |  结合了MaskGIT风格的迭代解码与VAR尺度结构的文本条件VAR变体  |
|  尺度位置嵌入  |  "(尺度, 行, 列)三元组"  |  位置编码同时携带尺度索引和尺度内的空间坐标  |

## 延伸阅读

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905)——VAR论文，标准参考文献
- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905)——DiT，扩散比较基线
- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905)——VQGAN，VAR多尺度分词器所扩展的分词器家族
- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905)——VQ-VAE，离散图像分词的基础
- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905)——文本条件VAR
