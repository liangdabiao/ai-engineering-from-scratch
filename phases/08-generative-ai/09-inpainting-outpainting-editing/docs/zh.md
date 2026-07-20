# 修复(Inpainting)、扩展(Outpainting)与图像编辑

> 文生图创造新事物。修复(Inpainting)修正旧事物。在生产中，70%的可收费图像工作是编辑——更换背景、移除Logo、扩展画布、重绘一只手。修复(Inpainting)是扩散模型展现价值之处。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段8·07（潜在扩散），阶段8·08（ControlNet & LoRA）
**时间：** 约75分钟

## 问题

客户发来一张完美的产品照片，背景中却有块干扰性的招牌。你想擦除这块招牌，并让其他所有像素保持不变。你不能从头运行文生图——结果会有不同的颜色、不同的光照、不同的产品角度。你希望*仅*重绘被遮罩的区域，并且希望重绘结果尊重周围的上下文。

这就是修复(Inpainting)。变体：

- **修复(Inpainting).** 在遮罩内重绘，保留外部像素。
- **扩展(Outpainting).** 在遮罩外（或画布外）重绘，保留内部。
- **图像编辑.** 重绘整个图像但保持与原图的语义或结构忠实度（SDEdit, InstructPix2Pix）。

2026年的每个扩散管道都内置了修复(Inpainting)模式。Flux.1-Fill, Stable Diffusion Inpaint, SDXL-Inpaint, DALL-E 3 Edit。它们基于相同原理工作。

## 核心概念

![Inpainting: mask-aware denoising with context-preserving reinjection](../assets/inpainting.svg)

### 朴素方法（以及为何错误）

使用遮罩运行标准文生图。在每个采样步骤，用前向扩散后的干净图像替换噪声潜在变量中未遮罩的区域。它...效果很差。边界伪影渗入，因为模型对遮罩区域内是什么没有信息。

### 正确的修复(Inpainting)模型

训练一个修改的U-Net，接受9个输入通道而不是4个：

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

额外的通道是VAE编码的源图像的副本加上一个单通道遮罩。训练时，你随机遮罩图像区域，训练模型仅去噪遮罩区域，而未遮罩区域作为干净的条件信号给出。推理时，模型能“看到”遮罩区域周围的内容，并生成连贯的补全结果。

SD-Inpaint, SDXL-Inpaint, Flux-Fill 都使用这种9通道（或类似）输入。扩散器(Diffusers) `StableDiffusionInpaintPipeline`, `FluxFillPipeline`。

### SDEdit (Meng等人, 2022) — 自由编辑

向源图像添加噪声到某个中间步数 `t`，然后使用新提示从 `t` 运行反向链降到0。无需重新训练。起始步数 `t` 的选择在忠实度与创作自由之间权衡：

- `t/T = 0.3` → 几乎与源图相同，细微风格变化
- `t/T = 0.3` → 中等编辑，保留粗略结构
- `t/T = 0.3` → 从近噪声生成，最小源保留

### InstructPix2Pix (Brooks等人, 2023)

在 `(input_image, instruction, output_image)` 三元组上微调扩散模型。推理时，以输入图像和文本指令（“让它变成日落”，“添加一条龙”）为条件。两个CFG尺度：图像尺度和文本尺度。

### RePaint (Lugmayr等人, 2022)

保持标准无条件扩散模型。在每个反向步骤，重新采样——偶尔跳回更嘈杂的状态并重新生成。避免边界伪影。用于没有训练好的修复(Inpainting)模型时。

## 动手构建

`code/main.py` 在5维数据上实现了一个玩具1维修复(Inpainting)方案。我们在5维混合数据上训练DDPM，每个样本是来自两个簇之一的5个浮点数。推理时，我们“遮罩”5个维度中的2个，在每个步骤注入未遮罩三个维度的噪声前向版本，并仅重绘被遮罩的维度。

### 步骤1: 5维DDPM数据

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### 步骤2: 在所有5个维度上训练去噪器

标准DDPM。网络输出5维噪声预测，对应5维噪声输入。

### 步骤3: 推理时，遮罩感知反向

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # replace unmasked dims with a freshly noised version of the clean source
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...then run the normal reverse step on x_t
```

这是朴素方法，适用于玩具1维数据。真实图像修复(Inpainting)使用9通道输入，因为纹理连贯性更重要。

### 步骤4: 扩展(Outpainting)

扩展(Outpainting)是遮罩反转的修复(Inpainting)：遮罩新的（先前不存在的）画布，用原图填充其余部分。训练目标相同。

## 陷阱

- **接缝.** 朴素方法留下可见边界，因为梯度信息无法跨遮罩流动。修复：将遮罩膨胀8-16像素，或使用正确的修复(Inpainting)模型。
- **遮罩泄漏.** 如果条件图像的未遮罩区域质量低或有噪声，会污染遮罩内的生成。轻微去噪或模糊。
- **CFG与遮罩大小相互作用.** 小遮罩上的高CFG = 饱和斑块。对小幅编辑降低CFG。
- **SDEdit忠实度悬崖.** 从 `t/T = 0.5` 到 `t/T = 0.6` 可能丢失主体身份。扫描并检查点。
- **提示不匹配.** 提示应描述*整个*图像，而不仅是新内容。使用“一只猫坐在椅子上”而不是“一只猫”。

## 使用它

|  任务  |  管道  |
|------|----------|
|  移除物体，小遮罩  |  SD-Inpaint 或 Flux-Fill，标准提示词  |
|  替换天空  |  SD-Inpaint + "黄昏的蓝天"  |
|  扩展画布  |  SDXL 外绘画模式（8像素羽化）或 Flux-Fill 配合外绘画遮罩  |
|  重新生成手/脸  |  SD-Inpaint，提示词重新描述主体 + ControlNet-Openpose  |
|  改变某一区域的风格  |  SDEdit 在 `t/T=0.5` 的遮罩区域上操作  |
|  "让它变成黄昏"  |  InstructPix2Pix 或 Flux-Kontext  |
|  背景替换  |  SAM 遮罩 → SD-Inpaint  |
|  超高保真度  |  针对最困难情况使用 Flux-Fill 或 GPT-Image（托管）  |

SAM（Meta 的 Segment Anything，2023）加上扩散修补是 2026 年的背景去除流程。SAM 2（2024）适用于视频。

## 发布

保存 `outputs/skill-editing-pipeline.md`。技能接受原始图像 + 编辑描述 + 可选遮罩（或 SAM 提示）并输出：遮罩生成方式、基础模型、CFG 尺度（图像 + 文本）、SDEdit-t 或修补模式，以及 QA 检查清单。

## 练习

1. **简单。** 在 `code/main.py` 中，将遮罩的维度比例从 0.2 变化到 0.8。在什么比例下，修补质量（遮罩维度中的残差）等于无条件生成？
2. **中等。** 实现 RePaint：每 10 个逆向步骤，跳回 5 步（添加噪声）并重新去噪。测量它是否减少了遮罩边缘的边界残差。
3. **困难。** 使用 Hugging Face diffusers 比较：SD 1.5 Inpaint + ControlNet-Openpose 与 Flux.1-Fill 在 20 个面部重生成任务上的表现。分别评分姿态一致性和身份保持。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  内绘画  |  "填充孔洞"  |  在遮罩内部重新生成；保留外部像素。  |
|  外绘画  |  "扩展画布"  |  在画布外部重新生成；保留内部。  |
|  9通道 U-Net  |  "真正的修补模型"  |  输入为 `noisy \ |  encoded-source \ |  mask` 的 U-Net。  |
|  SDEdit  |  "带有噪声级别的图生图"  |  对时间 `t` 加噪，用新提示词去噪。  |
|  InstructPix2Pix  |  "仅文本编辑"  |  在（图像，指令，输出）三元组上微调的扩散模型。  |
|  RePaint  |  "无需重新训练"  |  在逆向过程中周期性重新加噪以减少接缝。  |
|  SAM  |  "分割一切"  |  通过点击或框选生成遮罩；与修补配合使用。  |
|  Flux-Kontext  |  "带有上下文的编辑"  |  Flux 变体，接受参考图像加指令进行编辑。  |

## 生产注意：编辑流程对延迟敏感

编辑图像的用户期望往返时间低于 5 秒。在 L4 上，30 步的 SDXL-Inpaint 在 1024² 分辨率下需要 3-4 秒，加上 SAM 遮罩生成（约 200 毫秒）和 VAE 编码/解码（合计约 500 毫秒）。在生产框架中，这受 TTFT 限制而非吞吐量限制——批大小为 1，低并发，最小化每个阶段：

- **SAM-H 是较慢的。** SAM-H 在 1024² 下约 200 毫秒；SAM-ViT-B 约 40 毫秒，质量略有下降。SAM 2（视频）增加了时间开销；不要将其用于单图像编辑。
- **尽可能跳过编码。** `pipe.image_processor.preprocess(img)` 编码为潜变量。如果您有之前生成（常见于迭代编辑 UI）的潜变量，通过 `latents=...` 直接传递它们，跳过一次 VAE 编码。
- **遮罩扩张也影响吞吐量。** 小遮罩意味着大部分 U-Net 前向传递被浪费（未遮罩像素无论如何被钳制）。`pipe.image_processor.preprocess(img)` 的 `latents=...` 运行完整的 U-Net；只有 9 通道的真正修补变体利用遮罩计算。
- **Flux-Kontext 是 2025 年的答案。** 在 `pipe.image_processor.preprocess(img)` 上单次前向传递——没有单独遮罩，没有 SDEdit 噪声扫描。在 H100 上，它在约 1.5 秒内完成编辑。架构教训：合并阶段。

## 延伸阅读

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865)——无训练修补。
- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865)——SDEdit。
- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865)——文本指令编辑。
- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865)——SAM，遮罩来源。
- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865)——视频 SAM。
- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865)——注意力级别编辑。
- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865)——2024 年工具。
