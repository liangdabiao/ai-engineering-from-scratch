# 流匹配与纠正流

> 扩散模型需要20-50步采样，因为它们从噪声到数据走的是弯曲路径。流匹配(Lipman等人, 2023)和纠正流(Liu等人, 2022)则训练直线路径。路径越直，步数越少，推理越快。Stable Diffusion 3、Flux.1和AudioCraft 2都在2024年切换到了流匹配。

**类型：** 构建
**语言：** Python
**先修知识：** 阶段8·06 (DDPM)，阶段1·微积分
**时间：** ~45分钟

## 问题

DDPM的逆向过程是从`N(0, I)`回到数据分布的1000步随机漫步。DDIM将其压缩到20-50步确定性步数。你想要更少的步数——理想情况下是一步。障碍在于求解逆向过程的ODE是刚性的；路径是弯曲的。

如果你能训练模型使得从噪声到数据的路径是*直线*，那么从`t=1`到`t=0`的单步欧拉就能工作。流匹配直接构建了这一点：定义从`x_1 ∼ N(0, I)`到`x_0 ∼ data`的直线插值，训练向量场`v_θ(x, t)`匹配其时间导数，在推理时积分。

纠正流(Liu 2022)更进一步：通过再流(reflow)程序迭代地拉直路径，产生逐步接近线性的ODE。经过两次再流迭代，2步采样器即可匹配50步DDPM的质量。

## 核心概念

![Flow matching: straight-line interpolation between noise and data](../assets/flow-matching.svg)

### 直线流

定义：

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

其中`x_0 ~ data`和`x_1 ~ N(0, I)`。沿该直线的时间导数是常数：

```
dx_t / dt = x_1 - x_0
```

定义神经向量场`v_θ(x_t, t)`并训练它匹配此导数：

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

这就是**条件流匹配**损失(Lipman 2023)。训练是无模拟的：你永远不需要展开ODE。只需采样`(x_0, x_1, t)`并进行回归。

### 采样

在推理时，将学习到的向量场*反向*积分：

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

从`x_1 ~ N(0, I)`开始，欧拉步逐步下降到`t=0`。

### 纠正流(Liu 2022)

直线流可行，但学习到的路径*实际上并不直*——它们弯曲，因为许多`x_0`可以映射到同一个`x_1`。纠正流的再流步骤：

1. 训练流模型v_1，随机配对。
2. 通过从`x_1`积分v_1到其终点`x_0`来采样N对`(x_1, x_0)`。
3. 在这些配对样本上训练v_2。由于这些配对现在是"ODE匹配的"，它们之间的直线插值真正更平坦。
4. 重复。

实际上，2次再流迭代就能得到接近线性的结果，从而实现2-4步推理。SDXL-Turbo、SD3-Turbo、LCM都是从流匹配蒸馏的模型。

### 为什么这在2024年赢得了图像领域

三个原因：

1. **无模拟训练**——训练期间无需展开ODE，实现简单。
2. **更好的损失几何**——直线路径有一致的信号噪声比，而DDPM的ε损失在调度边界处SNR较差。
3. **更快的推理**——4-8步达到SDXL-Turbo质量；一致性蒸馏可达1步。

## 流匹配与DDPM——精确的联系

具有高斯条件路径的流匹配是*具有特定噪声调度*的扩散。选择`x_t = α(t) x_0 + σ(t) x_1`调度，流匹配恢复出具有`v = α'·x_0 - σ'·x_1`的Stratonovich重表述扩散。两者对于高斯路径是代数等价的。

流匹配带来的：目标的*清晰度*(一个简单的速度)，更干净的损失，以及实验非高斯插值的自由。

## 动手构建

`code/main.py`在双峰高斯混合上实现一维流匹配。向量场`v_θ(x, t)`是一个小型MLP，用直线目标进行训练。在推理时，分别积分1、2、4和20步欧拉，并比较样本质量。

### 步骤1：训练损失

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # backprop + update
```

### 步骤2：多步推理

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### 步骤3：比较步数

预计4步采样器已经匹配20步的质量——这对延迟来说意义重大。

## 陷阱

- **时间参数化。** 流匹配使用`t ∈ [0, 1]`，在数据处`t=0`，噪声处`t=1`。DDPM使用`t ∈ [0, T]`，在数据处`t=0`，噪声处`t=T`。方向相同，尺度不同。论文经常搞错这一点。
- **调度选择。** 纠正流的直线是"那个"流匹配调度，但你可以使用余弦或logit-normal t采样(SD3就是这样做的)以获得更好的尺度覆盖。
- **再流成本。** 为再流生成配对数据集需要对每个样本进行一次完整的推理。只有在真正需要1-2步推理时才进行再流。
- **无分类器指导仍然适用。** 只需在线性组合中将ε替换为v：`t ∈ [0, 1]`。

## 使用它

|  用例  |  2026技术栈  |
|----------|-----------|
|  文本到图像，最佳质量  |  流匹配：SD3, Flux.1-dev  |
| 文生图，1-4步  |  蒸馏流匹配(Flow matching)：Flux.1-schnell, SD3-Turbo, SDXL-Turbo |
| 实时推理  |  基于流匹配基座的一致性蒸馏(Consistency distillation) (LCM, PCM) |
| 音频生成  |  流匹配(Flow matching)：Stable Audio 2.5, AudioCraft 2 |
| 视频生成  |  流匹配与扩散混合 (Sora, Veo, Stable Video) |
| 科学/物理（粒子轨迹、分子）  |  流匹配 + 等变向量场 |

每当2025-2026年间有论文声称"比扩散(Diffusion)更快"，几乎都是流匹配+蒸馏。

## 发布

保存 `outputs/skill-fm-tuner.md`。技能获取一个扩散风格的模型规范，并将其转换为流匹配训练配置：调度选择、时间采样分布（均匀/对数正态）、优化器、重流(Reflow)计划、目标步数、评估协议。

## 练习

1. **简单。** 运行 `code/main.py`，比较1步与20步的MSE与真实数据分布。
2. **中等。** 将均匀 `code/main.py` 采样切换为对数正态（集中在中间t值采样）。模型质量是否提高？
3. **困难。** 实现一次重流迭代：通过对第一个模型积分生成配对的 (x_0, x_1)，在配对数据上训练第二个模型，并比较1步采样质量。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
| 流匹配  |  "直线扩散"  |  训练 `v_θ(x, t)` 匹配沿插值路径的 `x_1 - x_0`。 |
| 已校正流(Rectified flow)  |  "重流"  |  使学到的流直线化的迭代过程。 |
| 速度场(Velocity field)  |  "v_θ"  |  模型的输出——移动 `x_t` 的方向。 |
| 直线插值(Straight-line interpolant)  |  "路径"  |  `x_t = (1-t)·x_0 + t·x_1`；平凡目标导数。 |
| 欧拉采样器(Euler sampler)  |  "一阶ODE求解器"  |  最简单的积分器；当路径是直线时效果很好。 |
| 对数正态 t  |  "SD3采样"  |  将 `t` 采样集中在梯度最强的中间值附近。 |
| 一致性蒸馏(Consistency distillation)  |  "1步采样器"  |  训练学生模型将任何 `x_t` 直接映射到 `x_0`。 |
| 带速度的CFG  |  "v-CFG"  |  `v_cfg = (1+w) v_cond - w v_uncond`；相同技巧，新变量。 |

## 生产提示：Flux.1-schnell 是流匹配的最快版本

流匹配在生产上的胜利是 Flux.1-schnell —— 一个流匹配的DiT被蒸馏到1-4推理步，同时保持Flux-dev级别的质量。Niels的"在8GB机器上运行Flux"笔记本是参考部署方案：T5+CLIP编码，量化MMDiT去噪（schnell用4步，dev用50步），VAE解码。成本核算：

| 变体  |  步数  |  L4上1024²延迟  |  总FLOPs（相对） |
|---------|-------|------------------------|------------------------|
| Flux.1-dev (原始)  |  50  |  ~15秒  |  1.0× |
| Flux.1-schnell  |  4  |  ~1.2秒  |  0.08× (快12倍) |
| SDXL-base  |  30  |  ~4秒  |  0.25× |
| SDXL-Lightning 2步  |  2  |  ~0.3秒  |  0.03× |

生产规则：**流匹配基座 + 蒸馏 = 2026年快速文生图的默认方案。** 每个主要厂商都发布这种组合：SD3-Turbo (SD3 + 流 + 蒸馏), Flux-schnell (Flux-dev + 已校正流直线化), CogView-4-Flash。纯扩散基座仅存在于旧版检查点。

## 延伸阅读

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — 已校正流(Rectified flow)。
- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — 流匹配(Flow matching)。
- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — SD3，大规模已校正流。
- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — 涵盖FM+扩散的通用框架。
- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — 扩散/流的一步蒸馏。
- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — 涡轮变体。
- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — 生产中的流匹配。
