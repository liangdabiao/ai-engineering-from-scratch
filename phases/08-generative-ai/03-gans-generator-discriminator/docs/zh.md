# 生成对抗网络——生成器 vs 判别器

> Goodfellow在2014年的巧妙之处在于完全跳过了密度。两个网络。一个生成假数据。一个识别假数据。它们相互对抗，直到假数据与真实数据无法区分。这本来不该奏效。事实上常常也不奏效。但当它奏效时，生成的样本在狭窄领域内仍然是文献中最清晰的。

**类型：**构建
**语言：**Python
**先修知识：**阶段3 · 02（反向传播），阶段3 · 08（优化器），阶段8 · 02（变分自编码器）
**时间：**约75分钟

## 问题

变分自编码器产生模糊的样本，因为其均方误差解码器损失对于*平均*图像是贝叶斯最优的——而多个合理数字的平均值是一个模糊的数字。你需要一个奖励*合理性*的损失，而不是像素级接近任何单一目标。合理性没有封闭形式。你必须学习它。

Goodfellow的想法：训练一个分类器`D(x)`来区分真实图像和假图像。训练一个生成器`G(z)`来欺骗`D`。对于`G`的损失信号就是`D`当前认为使某物看起来真实的东西。这个信号随着`G`的改进而更新，不断追逐移动目标。如果两个网络收敛，`G`就学会了数据分布，而从未明确写出`log p(x)`。

这就是对抗训练。数学上是一个极小极大博弈：

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

在2026年，生成对抗网络已不再是最先进的生成器（扩散和流匹配夺走了这个桂冠）。但StyleGAN 2/3仍然是有史以来最清晰的人脸模型，生成对抗网络的判别器被用作扩散训练中的*感知损失*，而对抗训练驱动了快速的单步蒸馏（SDXL-Turbo、SD3-Turbo、LCM），让你能够实现实时扩散。

## 核心概念

![GAN training: generator and discriminator in minimax](../assets/gan.svg)

**生成器`G(z)`。** 将噪声向量`z ~ N(0, I)`映射到样本`x̂`。一个解码器形状的网络（全连接或转置卷积）。

**判别器`D(x)`。** 将样本映射到一个标量概率（或分数）。真实→1，假→0。

**损失。** 两次交替更新：

- **训练`D`：** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`。二元交叉熵，真实=1，假=0。
- **训练`D`：** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`。这是Goodfellow使用的*非饱和*形式（原始`G`在`loss_G = -log D(G(z))`自信时会饱和并杀死梯度）。

**训练循环。** `D`一步，`G`一步。重复。

**为什么它有效。** 如果`G`完美匹配`p_data`，那么`D`无法做得比随机更好，处处输出0.5；`G`不再获得梯度。均衡。

**为什么它失效。** 模式崩溃（`G`找到一个`D`无法分类的模式，并永远生成它），梯度消失（`D`学得太快，`log D`饱和），训练不稳定（学习率、批量大小，任何因素）。

## 使GANs奏效的变体

|  年份  |  创新  |  修复  |
|------|------------|-----|
|  2015  |  DCGAN  |  卷积/反卷积，批归一化，LeakyReLU——首个稳定架构。  |
|  2017  |  WGAN, WGAN-GP  |  用Wasserstein距离+梯度惩罚替代BCE。修复梯度消失。  |
|  2017  |  谱归一化  |  Lipschitz约束判别器。在2026年的判别器中仍在使用。  |
|  2018  |  渐进式GAN  |  先训练低分辨率，再添加层。首个百万像素结果。  |
|  2019  |  StyleGAN / StyleGAN2  |  映射网络+自适应实例归一化。固定领域照片真实感的最新技术。  |
|  2021  |  StyleGAN3  |  无混叠、平移等变——在2026年仍是人脸黄金标准。  |
|  2022  |  StyleGAN-XL  |  条件式、类别感知、更大规模。  |
|  2024  |  R3GAN  |  以更强正则化重新品牌化；在1024²上无需技巧即可工作。  |

```figure
gan-minimax
```

## 动手构建

`code/main.py`在1维数据上训练一个微型GAN：两个高斯分布的混合。生成器和判别器是单隐藏层MLP。我们手动实现前向、反向和极小极大循环。目标是看到两个关键失败模式（模式崩溃+梯度消失）发生时的情况。

### 第1步：非饱和损失

标准的Goodfellow损失`log(1 - D(G(z)))`在D以高置信度将G的假样本判为假时变为0。此时G的梯度基本为零——G无法改进。非饱和形式`-log D(G(z))`有相反的渐近线：当D自信时它会激增，给G一个强信号。

```python
def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### 第2步：生成器每步对应一个判别器步

```python
for step in range(steps):
    # train D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # train G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # fresh fakes
    update_G(fake_batch)
```

为G提供新鲜的假样本，否则梯度会过时。

### 第3步：观察模式崩溃

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

典型症状：两种实模态之一停止生成。判别器不再纠正它，因为它从未被视为假样本。

## 陷阱

- **判别器过强。** 将D的学习率降低2-5倍，或添加实例/层噪声。如果D准确率超过95%，则G失效。
- **生成器记忆了一个模态。** 向D输入添加噪声，使用小批量判别器层，或切换到WGAN-GP。
- **批归一化泄露统计信息。** 真实批和假批流经同一个BN层会混合它们的统计信息。改用实例归一化或谱归一化。
- **Inception分数作弊。** FID和IS在样本量低时噪声很大。评估时使用≥10k样本。
- **条件任务中一次采样不可靠。** 你仍需要CFG尺度、截断技巧和重采样来获得可用输出。

## 使用它

2026年GAN技术栈：

|  情况  |  选择  |
|-----------|------|
|  逼真的人脸，固定姿态  |  StyleGAN3（最清晰，最小） |
|  动漫/风格化人脸  |  StyleGAN-XL或Stable Diffusion LoRA |
|  图像到图像翻译  |  Pix2Pix / CycleGAN（阶段8 · 04）或ControlNet（阶段8 · 08） |
|  快速单步文本到图像  |  扩散模型的对抗蒸馏（SDXL-Turbo, SD3-Turbo） |
|  扩散训练器中的感知损失  |  图像裁剪上的小型GAN判别器 |
|  任何多模态、开放式的任务  |  不要用——使用扩散或流匹配 |

GANs清晰但狭窄。一旦你的领域开放——照片、任意文本提示、视频——切换到扩散。对抗技巧作为组件（感知损失、蒸馏）存活下来，而不是独立的生成器。

## 发布

保存 `outputs/skill-gan-debugger.md`。技能接收一个失败的GAN运行（损失曲线、样本网格、数据集大小），并输出一个按可能性排序的原因列表、一行修复建议以及重新运行协议。

## 练习

1. **简单。** 使用默认设置运行 `code/main.py`。然后设置 `D_LR = 5 * G_LR` 并重新运行。G的损失下降为常数的速度有多快？
2. **中等。** 将Goodfellow BCE损失替换为WGAN损失：`code/main.py`, `D_LR = 5 * G_LR`，并将D的权重裁剪到 `loss_D = E[D(fake)] - E[D(real)]`。训练更稳定吗？比较挂钟收敛时间。
3. **困难。** 将一维示例扩展到二维数据（环上的8个高斯混合）。跟踪生成器在1k、5k、10k步时捕获了多少个模态。实现小批量判别并重新测量。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  生成器  |  "G"  |  噪声到样本网络，`G: z → x̂`。 |
|  判别器  |  "D"  |  分类器 `D: x → [0, 1]`，真实vs虚假。 |
|  极小极大  |  "游戏"  |  联合目标的 `min_G max_D`。 |
|  非饱和损失  |  "修正"  |  对G使用 `-log D(G(z))` 而非 `log(1 - D(G(z)))`。 |
|  模式崩塌  |  "G记住了某一种东西"  |  生成器只产生少量不同的输出，尽管数据多样。 |
|  WGAN  |  "Wasserstein"  |  用推土机距离+梯度惩罚替换BCE；梯度更平滑。 |
|  谱归一化  |  "Lipschitz技巧"  |  约束D的权重范数以限制其斜率；稳定训练。 |
|  StyleGAN  |  "好用的那个"  |  映射网络+AdaIN；人脸最佳，即使在2026年。 |

## 生产提示：单步推理是GAN的持久优势

对于开放域生成，GANs不再在样本质量上胜出，但它们在推理成本上仍占优势。在生产推理文献词汇中，GAN具有：

- **没有预填充和解码阶段。** 一次 `G(z)` 前向传播。TTFT ≈ 总延迟。
- **没有KV缓存压力。** 唯一的状态是权重。批大小受激活内存限制，而非缓存。
- **简单的连续批处理。** 由于每个请求消耗相同的固定FLOPs，服务器目标占用率的静态批通常是最优的。不需要动态调度器。

这就是为什么GAN蒸馏（SDXL-Turbo, SD3-Turbo, ADD, LCM）是2026年快速文本到图像的主导技术：它将20-50步的扩散流程压缩为1-4步GAN风格的前向传播，同时保持扩散基底的分布。对抗损失作为训练时的旋钮存活下来，用于将慢速生成器变成快速生成器。

## 延伸阅读

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — 原始GAN论文。
- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — 第一个稳定架构。
- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — WGAN。
- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — SN。
- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — StyleGAN2。
- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — StyleGAN3。
- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — SDXL-Turbo。
