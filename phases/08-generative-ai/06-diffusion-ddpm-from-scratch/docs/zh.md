# 扩散模型——从头实现DDPM

> Ho, Jain, Abbeel (2020) 提供了一个令人无法抗拒的配方：用噪声在数千个微小步骤中破坏数据；训练一个神经网络来预测噪声；在推理时逆转这一过程。如今，所有主流图像、视频、3D和音乐模型都运行在这个循环上，可能还会叠加流匹配或一致性技巧。

**类型：** 构建
**语言：** Python
**前置知识：** 第三阶段·02（反向传播）、第八阶段·02（VAE）
**时间：** ~75分钟

## 问题

你想要一个 `p_data(x)` 的采样器。GANs 玩的是 minimax 游戏，常常发散。VAEs 从高斯解码器产生模糊样本。你真正想要的是一个训练目标，它（a）是单一稳定的损失（没有鞍点，没有 minimax），（b）是 `log p(x)` 的下界（这样你就有似然），并且（c）能产生与 SOTA 质量匹配的样本。

Sohl-Dickstein 等人（2015）给出了一个理论答案：定义一个马尔可夫链 `q(x_t | x_{t-1})`，逐步添加高斯噪声，并训练一个反向链 `p_θ(x_{t-1} | x_t)` 去噪。Ho, Jain, Abbeel（2020）证明损失可以简化为一行——预测噪声——并整理了数学。2020年这还是一个好奇心产物；2021年它产生了最先进的样本；2022年它变成了 Stable Diffusion；2026年它已成为基石。

## 核心概念

![DDPM: forward noise, reverse denoise](../assets/ddpm.svg)

**前向过程 `q`。** 在 `T` 个小步骤中添加高斯噪声。闭式解——数学可处理的原因——是累积步骤也是高斯分布：

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

其中 `α̅_t = ∏_{s=1..t} (1 - β_s)`，对于 `β_t` 的调度。在 T=1000 步上线性选取 `β_t` 从 1e-4 到 0.02，且 `x_T` 近似为 `N(0, I)`。

**反向过程 `p_θ`。** 学习一个神经网络 `ε_θ(x_t, t)`，预测已添加的噪声。给定 `x_t`，通过以下方式去噪：

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

其中 `σ_t` 要么是 `sqrt(β_t)`，要么是学习到的方差。这个表达式看起来难看，但只是代数——求解给定后验 `q(x_{t-1} | x_t, x_0)` 的 `x_{t-1}`，并将 `x_0` 替换为其噪声预测的估计。

**训练损失。**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

从数据中采样 `x_0`，随机选取 `t`，采样 `ε ~ N(0, I)`，通过闭式解一步计算带噪 `x_t`，并对噪声进行回归。一个损失，没有 minimax，没有 KL，没有重参数化技巧。

**采样。** 从 `x_T ~ N(0, I)` 开始。从 `t = T` 到 `1` 迭代反向步骤。完成。

## 为什么有效

三种直觉：

1. **去噪容易；生成困难。** 在 `t=T` 时，数据是纯噪声——网络只需解决一个平凡问题。在 `t=0` 时，网络只需清除几个像素。在中间的 `t`，问题困难，但网络有来自所有噪声级别的多个梯度流过相同的权重。

2. **伪装下的得分匹配。** Vincent（2011）证明预测噪声等价于估计 `∇_x log q(x_t | x_0)`，即*得分(score)*。反向 SDE 利用这个得分沿密度梯度向上行走——朝向高概率区域的有引导随机游走。

3. **ELBO 简化为简单的 MSE。** 完整的变分下界在每个时间步有一个 KL 项。采用 DDPM 的参数化，这些 KL 项简化为特定系数下的噪声预测 MSE；Ho 去掉了这些系数（称为“简单”损失），而质量*提高了*。

```figure
diffusion-denoise
```

## 动手构建

`code/main.py` 实现了一个一维 DDPM。数据是双峰混合。“网络”是一个微型 MLP，接收 `(x_t, t)` 并输出预测噪声。训练是单行损失函数。采样迭代反向链。

### 第一步：前向调度（闭式解）

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### 第二步：一步采样 `x_t`

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### 第三步：一步训练

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### 第四步：反向采样

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

对于一个具有40个时间步和24单元MLP的一维问题，大约200个epoch就能学会双峰混合。

## 时间条件

网络需要知道它正在去噪哪个时间步。两种标准选择：

- **正弦嵌入。** 类似Transformer位置编码。`embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`。通过MLP处理，广播到网络中。
- **FiLM/组归一化条件。** 将嵌入投影到每个通道的缩放/偏置（FiLM）在每个块中。

我们的玩具代码使用正弦嵌入后拼接。生产级U-Net使用FiLM。

## 陷阱

- **调度非常重要。** 线性 `β` 是DDPM的默认设置，但余弦调度（Nichol & Dhariwal, 2021）在相同计算量下给出更好的FID。如果质量停滞，可以切换调度。
- **时间步嵌入很脆弱。** 将原始 `β` 作为浮点数传入对玩具一维有效，但对图像会失败；始终使用合适的嵌入。
- **V预测 vs. ε预测。** 在窄范围（非常小或非常大的t）内，`β` 的信噪比较差。V预测（`t`）更稳定；SDXL、SD3和Flux使用它。
- **无分类器引导。** 在推理时，同时计算有条件与无条件的 `β`，然后以 `ε` 进行 `t`。在第08课中介绍。
- **1000步太多了。** 生产中使用DDIM（20-50步）、DPM-Solver（10-20步）或蒸馏（1-4步）。见第12课。

## 使用它

|  角色  |  2026年的典型栈  |
|------|-----------------------|
|  图像像素空间扩散（小型、玩具型）  |  DDPM + U-Net  |
|  图像潜在扩散  |  VAE 编码器 + U-Net 或 DiT（第07课）  |
| 视频潜在扩散 | 时空DiT (Sora, Veo, WAN) |
| 音频潜在扩散 | Encodec + 扩散变换器 |
| 科学（分子、蛋白质、物理）||| 等变扩散 (EDM, RFdiffusion, AlphaFold3) |  |

扩散是通用的生成骨干。流匹配（第13课）是2024-2026年的竞争者，通常在同质量下在推理速度上胜出。

## 发布

保存`outputs/skill-diffusion-trainer.md`。技能接收数据集+计算预算，输出：调度方案(linear/cosine/sigmoid)、预测目标(ε/v/x)、步数、指导尺度、采样器族以及评估协议。

## 练习

1. **简单。** 在`code/main.py`中将T从40改为10。样本质量（输出的视觉直方图）如何下降？在哪个T值下双模结构崩溃？
2. **中等。** 从ε预测切换到v预测。重新推导反向步骤。比较最终样本质量。
3. **困难。** 添加无分类器指导。以类别标签`code/main.py`为条件，在训练期间丢弃它10%的时间，在采样时使用`c ∈ {0, 1}`。测量在`ε = (1+w)·ε_cond - w·ε_uncond`下的条件模式命中率。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
| 前向过程 | "加噪" | 固定的马尔可夫链`q(x_t \ | x_{t-1})`，破坏数据。 |
| 反向过程 | "去噪" | 学习到的链`p_θ(x_{t-1} \ | x_t)`，重建数据。 |
| β调度方案 | "噪声阶梯" | 每步方差；线性、余弦或S型。 |
| α̅ | "Alpha bar" | 累积乘积`∏(1 - β)`；从`x_0`给出封闭形式`x_t`。 |
| 简单损失 | "噪声上的MSE" | `\ | \ | ε - ε_θ(x_t, t)\ | \ | ²`；所有变分推导都归结于此。 |
| ε预测 | "预测噪声" | 输出是添加的噪声；标准DDPM。 |
| v预测 | "预测速度" | 输出是`α·ε - σ·x`；在t上更好的条件作用。 |
| DDPM | "那篇论文" | Ho等人2020；线性β，1000步，U-Net。 |
| DDIM | "确定性采样器" | 非马尔可夫采样器，20-50步，相同的训练目标。 |
| 无分类器指导 (CFG) | "CFG" | 混合条件和无条件噪声预测以放大条件作用。 |

## 生产注意：扩散推理是一个步数问题

DDPM论文运行T=1000个反向步骤。没有人会在生产中那样部署。每个真实的推理栈都会选择三种策略之一——每种都清晰地映射到“延迟来自哪里”的生产框架中：

1. **更快的采样器，相同的模型。** DDIM (20-50步), DPM-Solver++ (10-20), UniPC (8-16)。直接替换反向循环；训练好的`ε_θ`权重保持不变。延迟降低20-50倍。
2. **蒸馏。** 训练一个学生模型在更少的步数上匹配教师：渐进蒸馏 (2→1), 一致性模型 (任意→1-4), LCM, SDXL-Turbo, SD3-Turbo。延迟再降低5-10倍，需要重新训练。
3. **缓存和编译。** `ε_θ`, TensorRT-LLM的扩散后端, `torch.compile(unet, mode="reduce-overhead")`/SDPA注意力, bf16权重。每步延迟降低约2倍。可与(1)和(2)叠加使用。

对于生产环境中的扩散服务器，预算讨论与生产文献中描述LLM的相同：延迟是`num_steps × step_cost + VAE_decode`，吞吐量是`batch_size × (num_steps × step_cost)^-1`。TTFT很小（一步）；TPOT等效于完整响应时间，因为从用户角度看图像生成是“一次性”的。

## 延伸阅读

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — 扩散论文，超前于时代。
- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — DDPM。
- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — DDIM，更少步数。
- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — 余弦调度方案，学习方差。
- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — 分类器指导。
- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — CFG。
- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — 统一符号，最简洁的方案。
