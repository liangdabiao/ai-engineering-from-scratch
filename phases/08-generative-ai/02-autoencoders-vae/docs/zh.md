# 自动编码器与变分自动编码器(VAE)

> 普通自动编码器先压缩再重构。它记忆数据，不会生成。加入一个小技巧——迫使编码看起来像高斯分布——你就得到一个采样器。这个技巧，即`z = μ + σ·ε`的重参数化，是你在2026年使用的每个潜在扩散和流匹配图像模型在输入处都有一个VAE的原因。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段3·02（反向传播）、阶段3·07（CNN）、阶段8·01（分类学）
**时间：** 约75分钟

## 问题

将784像素的MNIST数字压缩为16个数字的编码，然后重构。普通自动编码器可以完美地完成重构MSE，但编码空间是一团糟。在编码空间中随机选点解码，得到的是噪声。它没有采样器。这是一个装扮成生成模型的压缩模型。

你真正想要的是：(a) 编码空间是一个干净、平滑的分布，可以从中采样——比如各向同性高斯分布`N(0, I)`，(b) 解码任意样本都能产生一个合理的数字，(c) 编码器和解码器仍然能很好地压缩。三个目标，一个架构，一个损失函数。

Kingma在2013年提出的VAE通过训练编码器输出一个*分布*`q(z|x) = N(μ(x), σ(x)²)`，通过KL惩罚将该分布推向先验`N(0, I)`，然后在解码前从`z`采样`q(z|x)`来解决这个问题。推理时，去掉编码器，采样`z ~ N(0, I)`，解码。KL惩罚正是迫使编码空间结构化的关键。

到2026年，VAE很少独立部署——它们在原始图像质量上已被扩散模型超越——但它们仍然是每个潜在扩散模型（SD 1/2/XL/3、Flux、AudioCraft）的首选编码器。学习VAE，你就学习了所用每个图像流水线的不可见的第一层。

## 核心概念

![Autoencoder vs VAE: the reparameterization trick](../assets/vae.svg)

**自动编码器。** `z = encoder(x)`，`x̂ = decoder(z)`，损失 = `||x - x̂||²`。编码空间无结构。

**VAE编码器。** 输出两个向量：`μ(x)`和`log σ²(x)`。它们定义了`q(z|x) = N(μ, diag(σ²))`。

**重参数化技巧。** 从`q(z|x)`采样不可微分。将采样重写为`z = μ + σ·ε`，其中`ε ~ N(0, I)`。现在`z`是`(μ, σ)`加上非参数噪声的确定性函数——梯度流过`μ`和`σ`。

**损失函数。** 证据下界(ELBO)，两项：

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

重构项将`x̂`推向`x`。KL项将`q(z|x)`推向先验。它们相互权衡。小的β (<1) = 更清晰的样本，编码空间更少高斯性。大的β (>1) = 更干净的编码空间，更模糊的样本。β-VAE (Higgins 2017)使这个调节旋钮闻名，并开启了解耦表示学习的研究。

**采样。** 推理时：抽取`z ~ N(0, I)`，通过解码器前向传播。一次前向传播——无需像扩散那样的迭代采样。

```figure
vae-latent-grid
```

## 动手构建

`code/main.py`实现了一个小型VAE，无需numpy或torch。输入是8维合成数据，来自8维中的2分量高斯混合。编码器和解码器都是单隐藏层MLP。我们实现tanh激活、前向传播、损失函数和手写反向传播。教学目的，非生产代码。

### 第一步：编码器前向

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

使用`log σ²`而非`σ`，这样网络输出无约束（对σ使用softplus是陷阱——梯度在σ≈0时消失）。

### 第二步：重参数化和解码

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### 第三步：ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

由于两个分布都是高斯分布，使用精确闭式KL。不要数值积分。2026年仍有人使用蒙特卡洛KL估计——无谓地慢3倍。

### 第四步：生成

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

这就是生成模型。五行代码。

## 陷阱

- **后验坍塌。** KL项如此激进地驱动`q(z|x) → N(0, I)`，以至于`z`不携带关于`x`的信息。修复：β退火（从β=0开始，逐步增加到1）、自由位(free bits)，或跳过不活跃维度的KL。
- **模糊样本。** 高斯解码器似然意味着MSE重构，这对L2（均值）是贝叶斯最优的——一组合理数字的均值是一个模糊的数字。修复：离散解码器（VQ-VAE、NVAE），或仅将VAE用作编码器并在潜在空间上堆叠扩散（这就是Stable Diffusion的做法）。
- **β太大、太早。** 见后验坍塌。从β≈0.01开始并逐渐增加。
- **潜在维度太小。** MNIST用16维，ImageNet 256²用256维，ImageNet 1024²用2048维。Stable Diffusion的VAE将512×512×3压缩为64×64×4（空间面积32倍下采样，通道数32倍）。

## 使用它

2026年VAE堆栈：

|  情况  |  选择  |
|-----------|------|
|  用于扩散的图像-潜在编码器  |  Stable Diffusion VAE (`sd-vae-ft-ema`) 或 Flux VAE  |
|  音频-潜在编码器  |  Encodec (Meta)、SoundStream 或 DAC (Descript)  |
|  视频潜在编码器  |  Sora的时空块、Latte VAE、WAN VAE  |
|  解耦表示学习  |  β-VAE、FactorVAE、TCVAE  |
|  离散潜在变量（用于Transformer建模）  |  VQ-VAE、RVQ (ResidualVQ)  |
|  用于生成的连续潜在变量  |  普通VAE，然后在潜在空间中条件化流/扩散模型  |

潜在扩散模型是一个VAE，其中扩散模型位于编码器和解码器之间。VAE负责粗压缩，扩散模型负责重担。视频（VAE + 视频扩散DiT）和音频（Encodec + MusicGen transformer）采用相同模式。

## 发布

保存 `outputs/skill-vae-trainer.md`。

技能包括：数据集概况 + 潜在维度目标 + 下游用途（重建、采样或潜在扩散输入），输出：架构选择（普通/β/VQ/RVQ）、β调度、潜在维度、解码器似然（高斯 vs 分类）以及评估计划（重建MSE、每维KL、`q(z|x)` 与 `N(0, I)` 之间的弗雷歇距离）。

## 练习

1. **简单。** 将 `code/main.py` 中的 `β` 改为 `0.01`、`0.1`、`1.0`、`5.0`。记录最终的重建MSE和KL。对于你的合成数据，哪个β是帕累托最优的？
2. **中等。** 将高斯解码器似然替换为伯努利似然（交叉熵损失）。在相同合成数据的二值化版本上比较样本质量。
3. **困难。** 将 `β` 扩展为迷你VQ-VAE：用代码本中K=32个条目的最近邻查找替换连续的 `code/main.py`。比较重建MSE并报告使用了多少个代码本条目（代码本崩溃是真实存在的）。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  自编码器  |  编码-解码网络  |  `x → z → x̂`，学习MSE。非生成式。  |
|  VAE  |  带有采样器的自编码器  |  编码器输出分布，KL惩罚塑造代码空间。  |
|  ELBO  |  证据下界  |  `log p(x) ≥ recon - KL[q(z\ | x) \ | \ |  p(z)]`; tight when `q = p(z\ | x)`。  |
|  重参数化  |  `z = μ + σ·ε`  |  将随机节点重写为确定性加纯噪声。使采样可通过反向传播。  |
|  先验  |  `p(z)`  |  潜在变量的目标分布，通常为 `N(0, I)`。  |
|  后验坍缩  |  "KL项获胜"  |  编码器忽略 `x`，输出先验；解码器必须进行幻觉。  |
|  β-VAE  |  可调KL权重  |  `loss = recon + β·KL`。β越高表示解缠结越好但更模糊。  |
|  VQ-VAE  |  离散潜在变量  |  用最近的代码本向量替换连续 `z`；支持Transformer建模。  |

## 生产注意：VAE是扩散服务器中最热门的路径

在Stable Diffusion / Flux / SD3管道中，VAE每个请求被调用两次——一次编码（如果进行img2img / 修复），一次解码。在1024²分辨率下，解码过程通常是整个管道中激活内存的单个最大峰值，因为它将 `128×128×16` 潜在变量上采样回 `1024×1024×3`。两个实际后果：

- **切片或分块解码。** `diffusers` 暴露了 `pipe.vae.enable_slicing()` 和 `pipe.vae.enable_tiling()`。分块用轻微的接缝伪影换取 `O(tile²)` 内存而非 `O(H·W)`。在消费级GPU上处理1024²+图像时必不可少。
- **bf16解码器，最终缩放的fp32数值精度。** SD 1.x VAE以fp32发布，在1024²+分辨率下强制转换为fp16时会*静默产生NaN*。SDXL提供了 `diffusers`——始终优先使用fp16修复变体或使用bf16。

## 延伸阅读

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)——VAE论文。
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)——解缠结β-VAE。
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)——VQ-VAE。
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)——最先进的图像VAE。
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)——Stable Diffusion；VAE作为编码器。
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)——Encodec，音频VAE标准。
