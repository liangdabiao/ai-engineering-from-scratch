# 条件生成对抗网络与Pix2Pix

> 2014-2017年的第一个重大突破是控制生成对抗网络(Generative Adversarial Network, GAN)生成的内容。添加一个标签、一张图片或一个句子。Pix2Pix实现了图像版本，并且在狭窄的图像到图像任务上，它仍然胜过所有通用的文本到图像生成模型。

**类型:** 构建
**语言:** Python
**前置知识:** 第8阶段·03 (生成对抗网络), 第4阶段·06 (U-Net), 第3阶段·07 (卷积神经网络)
**时间:** ~75分钟

## 问题

无条件的生成对抗网络(Unconditional GAN)可以生成任意的人脸。在演示中有用，但在实际应用中毫无用处。你想要的是：*将草图映射为照片*，*将地图映射为航拍照片*，*将白天场景映射为夜晚*，*为灰度图像上色*。在所有这些情况中，你得到一个输入图像 `x` ，必须输出具有某种语义对应关系的 `y` 。每个 `x` 对应多个合理的 `y` 。均方误差(Mean-squared error)会将它们模糊成一团。对抗性损失(Adversarial loss)不会，因为"看起来真实"是锐利的。

条件生成对抗网络(Conditional GAN) (Mirza & Osindero, 2014) 将条件 `c` 作为输入同时提供给 `G` 和 `D` 。Pix2Pix (Isola et al., 2017) 对此进行了专门化：条件是一个完整的输入图像，生成器(Generator)是U-Net，判别器(Discriminator)是基于块(patch-based)的分类器(PatchGAN)，损失是对抗性损失(Adversarial loss)加L1损失(L1 loss)。这个配方在狭窄的图像到图像领域上，甚至到2026年仍优于从头开始训练的文本到图像模型，因为它是在*成对数据*上训练的——你拥有恰好所需的信号。

## 核心概念

![Pix2Pix: U-Net generator, PatchGAN discriminator](../assets/pix2pix.svg)

**条件生成器(Conditional G).** `G(x, z) → y` 。在Pix2Pix中，`z` 是G内部的随机失活(Dropout)（不添加输入噪声——Isola发现显式噪声会被忽略）。

**条件判别器(Conditional D).** `D(x, y) → [0, 1]` 。输入是*对*(条件，输出)。这是关键区别：D必须判断 `y` 是否与 `x` 一致，而不仅仅是 `y` 看起来是否真实。

**U-Net生成器.** 带有跳跃连接(Skip connections)的编码器-解码器，跨越瓶颈层。在输入和输出共享低级结构（边缘、轮廓）的任务中至关重要。没有跳跃连接，高频细节会消失。

**PatchGAN判别器.** D不输出单一的实数/假数评分，而是输出一个 `N×N` 网格，其中每个单元格判断大约70×70像素的感受野(Receptive field)。取平均。这是一个马尔可夫随机场(Markov random field)假设：真实感是局部的。训练更快，参数更少，输出更锐利。

**损失函数(Loss).**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1项稳定训练，并将G推向已知目标。L1比L2产生更锐利的边缘（中位数 vs 均值）。`λ = 100` 是Pix2Pix的默认值。

## CycleGAN——当没有成对数据时

Pix2Pix需要成对的 `(x, y)` 数据。CycleGAN (Zhu et al., 2017) 通过引入一个额外的损失——*循环一致性损失(Cycle consistency loss)*来消除这个要求。两个生成器 `G: X → Y` 和 `F: Y → X` 。训练它们使得 `F(G(x)) ≈ x` 且 `G(F(y)) ≈ y` 。这让你可以在没有成对示例的情况下，将马转换为斑马，将夏季转换为冬季。

到了2026年，非成对的图像到图像转换大多通过扩散模型(Diffusion)（ControlNet、IP-Adapter）而非CycleGAN完成，但循环一致性思想几乎存在于每一篇非成对领域自适应论文中。

## 动手构建

`code/main.py` 实现了一个基于一维数据的微型条件生成对抗网络(Conditional GAN)。条件 `c` 是一个类别标签（0或1）。任务：为给定类别产生条件分布下的一个样本。

### 第一步：将条件附加到G和D的输入中

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

独热编码(One-hot encoding)是最简单的方法。更大的模型使用学习到的嵌入(Learned embeddings)、FiLM调制或交叉注意力(Cross-attention)。

### 第二步：训练条件模型

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

生成器必须匹配*针对给定条件*的真实分布，而不是边缘分布。

### 第三步：按类别验证输出

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## 陷阱

- **条件被忽略.** G学习边缘分布，D从不惩罚因为条件信号太弱。解决方法：更激进地调节D（早期层，不仅仅是后期层），使用投影判别器(Projection discriminator) (Miyato & Koyama 2018)。
- **L1权重过低.** G漂移到任意的看似真实的输出，而非忠实输出。对于Pix2Pix风格的任务，从λ≈100开始。
- **L1权重过高.** G产生模糊的输出，因为L1仍然是一个L_p范数。训练稳定后逐渐减小。
- **D中的真实信息泄露.** 将 `(x, y)` 作为D输入进行拼接，而不仅仅是 `y` 。否则D无法检查一致性。
- **每类模式崩溃.** 每个类别可能独立崩溃。运行类别条件多样性检查。

## 使用它

2026年图像到图像任务的现状：

|  任务  |  最佳方法  |
|------|---------------|
|  草图→照片，同领域，成对数据  |  Pix2Pix / Pix2PixHD（仍然快速，仍然锐利）  |
|  草图→照片，非成对  |  带有涂鸦条件模型(Scribble conditioning model)的ControlNet  |
|  语义分割→照片  |  SPADE / GauGAN2 或 SD + ControlNet-Seg  |
|  风格迁移  |  带有IP-Adapter或LoRA的扩散模型；生成对抗网络方法已过时  |
|  深度图→照片  |  基于稳定扩散(Stable Diffusion)的ControlNet-Depth  |
|  超分辨率  |  Real-ESRGAN (生成对抗网络)、ESRGAN-Plus 或 SD-Upscale (扩散模型)  |
|  上色  |  ColTran、基于扩散模型的上色器、或Pix2Pix-color  |
|  白天 → 夜晚，季节，天气  |  基于CycleGAN或ControlNet  |

Pix2Pix仍然是正确的工具，当(a)你有数千个配对样本，(b)任务狭窄且可重复，(c)你需要快速推理。在通用的开放域任务上，扩散模型更胜一筹。

## 发布

保存`outputs/skill-img2img-chooser.md`。技能接受任务描述、数据可用性（配对与未配对，N个样本）以及延迟/质量预算，然后输出：方法（Pix2Pix、CycleGAN、ControlNet变体、SDXL + IP-Adapter）、训练数据需求、推理成本和评估协议（LPIPS、FID、任务特定）。

## 练习

1. **简单。** 修改`code/main.py`以添加第三类。确认G仍然将每个类别的噪声映射到正确的模式。
2. **中等。** 在一维设置中用感知风格损失（perceptual-style loss）替换L1（例如，一个小的冻结D作为特征提取器）。这会改变条件分布的清晰度吗？
3. **困难。** 在一维设置中勾画一个CycleGAN：两个分布，两个生成器，循环一致性损失（cycle loss）。证明它可以在没有配对数据的情况下学习在它们之间映射。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  条件GAN  |  "带标签的GAN"  |  G(z, c), D(x, c)。两个网络都看到条件。  |
|  Pix2Pix  |  "图像到图像的GAN"  |  配对的cGAN，使用U-Net生成器G和PatchGAN判别器D + L1损失。  |
|  U-Net  |  "带跳跃连接的编码器-解码器"  |  对称卷积网络；跳跃连接保留了高频信息。  |
|  PatchGAN  |  "局部真实性分类器"  |  D输出每个图块的分数，而不是全局分数。  |
|  CycleGAN  |  "未配对的图像翻译"  |  两个G + 循环一致性损失；没有配对数据。  |
|  SPADE  |  "GauGAN"  |  用语义图归一化中间激活；分割到图像。  |
|  FiLM  |  "特征级线性调制"  |  来自条件的每个特征仿射变换；廉价的条件控制。  |

## 生产注意：Pix2Pix作为延迟受限的基线

当你拥有配对数据和狭窄任务时（草图→渲染，语义图→照片，白天→夜晚），Pix2Pix的一次推理在延迟上比扩散模型快一个数量级。生产比较通常如下：

|  路径  |  步骤  |  在单个L4上512²的典型延迟  |
|------|-------|----------------------------------------|
|  Pix2Pix（U-Net前向）  |  1  |  ~30毫秒  |
|  SD-Inpaint或SD-Img2Img  |  20  |  ~1.2秒  |
|  SDXL-Turbo Img2Img  |  1-4  |  ~0.15-0.35秒  |
|  ControlNet + SDXL基础模型  |  20-30  |  ~3-5秒  |

Pix2Pix在静态批次中胜出（每个请求的FLOPs相同）。扩散模型在质量和泛化能力上胜出。现代策略通常是针对狭窄任务发布Pix2Pix风格的蒸馏模型，并为尾部输入提供扩散模型回退。

## 延伸阅读

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — cGAN论文。
- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — Pix2Pix。
- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — CycleGAN。
- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — Pix2PixHD。
- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — SPADE / GauGAN。
- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — 投影判别器（projection D）。
