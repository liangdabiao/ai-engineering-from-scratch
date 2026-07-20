# 视频生成

> 图像是2维张量。视频是3维张量。理论是相同的；计算量则困难10到100倍。OpenAI的Sora（2024年2月）证明了这是可能的。到2026年，Veo 2、Kling 1.5、Runway Gen-3、Pika 2.0和WAN 2.2都支持从文本生成1080p的生产级视频——而开源权重堆栈（CogVideoX、HunyuanVideo、Mochi-1、WAN 2.2）落后了12个月。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段8·07（潜在扩散）、阶段7·09（ViT）、阶段8·06（DDPM）
**时间：** 约45分钟

## 问题

一段10秒1080p、24fps的视频包含240帧1920×1080×3像素。每个片段约1.5GB原始数据。像素空间的扩散不可行。你需要：

1. **时空压缩。** 一个VAE将视频（而非帧）编码为一系列时空图块。
2. **时间连贯性。** 帧需要在几秒内共享内容、光照和物体身份。网络必须建模运动。
3. **计算预算。** 在相同模型大小下，视频训练比图像训练昂贵10到100倍。
4. **条件输入。** 文本、图像（首帧）、音频或其他视频。大多数生产模型都接受这四种输入。

解决这一问题的架构是应用于时空图块的**扩散Transformer (DiT)**，在大型（提示、描述、视频）数据集上训练。与第6课相同的扩散损失。

## 核心概念

![Video diffusion: patchify, DiT, decode](../assets/video-generation.svg)

### 图块化

使用3D VAE（学习的时空压缩）对视频进行编码。潜在变量形状为`[T_latent, H_latent, W_latent, C_latent]`。分割成大小`[t_p, h_p, w_p]`的图块。对于Sora风格的模型，`t_p = 1`（每帧图块）或`t_p = 2`（每两帧）。一段10秒1080p视频压缩到约20,000-100,000个图块。

### 时空DiT

一个Transformer处理扁平化的图块序列。每个图块都有一个3D位置嵌入（时间+y+x）。注意力通常是分解的：

- **空间注意力**：每个帧内的图块。
- **时间注意力**：同一空间位置跨帧的图块。
- **全3D注意力**昂贵16-100倍，仅在低分辨率或研究中才使用。

### 文本条件输入

与大型文本编码器的交叉注意力（Sora使用T5-XXL，CogVideoX-5B使用T5-XXL）。长提示很重要——Sora的训练集有GPT生成的密集重新描述，每个片段平均200个token。

### 训练

标准扩散损失（ε或v预测）在时空潜在变量上。数据：网络视频+约1亿个精选片段+合成文本描述。计算量：即使是小型研究运行也需要10,000+GPU小时；Sora规模需要100,000+。

## 2026年生产格局

|  模型  |  日期  |  最大时长  |  最大分辨率  |  开源权重？  |  备注  |
|-------|------|--------------|---------|---------------|---------|
|  Sora (OpenAI)  |  2024-02  |  60秒  |  1080p  |  否  |  首个展示规模级世界模拟器属性的模型  |
|  Sora Turbo  |  2024-12  |  20秒  |  1080p  |  否  |  生产级Sora，推理速度提升5倍  |
|  Veo 2 (Google)  |  2024-12  |  8秒  |  4K  |  否  |  2025年最高质量+物理效果  |
|  Veo 3  |  2025年第三季度  |  15秒  |  4K  |  否  |  原生音频和更强的摄像头控制  |
|  Kling 1.5 / 2.1 (快手)  |  2024-2025  |  10秒  |  1080p  |  否  |  2025年第一季度最佳人体运动  |
|  Runway Gen-3 Alpha  |  2024-06  |  10秒  |  768p  |  否  |  基于专业视频工具  |
|  Pika 2.0  |  2024-10  |  5秒  |  1080p  |  否  |  最强人物一致性  |
|  CogVideoX (THUDM)  |  2024  |  10秒  |  720p  |  是 (2B, 5B)  |  首个开源5B级视频模型  |
|  HunyuanVideo (腾讯)  |  2024-12  |  5秒  |  720p  |  是 (13B)  |  2024年末开源SOTA  |
|  Mochi-1 (Genmo)  |  2024-10  |  5.4秒  |  480p  |  是 (10B)  |  许可最宽松  |
|  WAN 2.2 (阿里巴巴)  |  2025-07  |  5秒  |  720p  |  是  |  2025年中最强开源模型  |

开源权重缩小差距的速度比图像领域更快：到2026年中，HunyuanVideo + WAN 2.2 LoRA已经为大多数开源工作流提供动力。

## 动手构建

`code/main.py`模拟了核心的时空DiT想法：将小型合成视频图块化，为每个图块添加位置嵌入，并通过基于Transformer的注意力机制对整个序列进行去噪。不使用numpy；纯Python。我们展示了即使在1维场景中，当相邻帧的图块共享去噪器和位置嵌入时，时间连贯性也会出现。

### 步骤1：将合成的1D“视频”进行分块(patchify)

```python
def make_video(T_frames=8, rng=None):
    # a "video" is a sequence of 1-D values following a smooth trajectory
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### 步骤2：每帧的位置嵌入(position embedding)

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### 步骤3：去噪器(denoiser)看到整个序列

我们的微型网络不是独立去噪每一帧，而是将所有帧的值及其位置嵌入连接起来，并联合预测所有帧的噪声。

### 步骤4：时间一致性测试(temporal coherence test)

训练后，采样一个视频。测量帧与帧之间的差异(delta)。如果模型学习了时间结构，这些差异会小于独立采样每一帧时的差异。

## 陷阱

- **独立逐帧采样 = 闪烁(flicker)。** 如果对每帧单独运行图像扩散(image diffusion)，输出会闪烁，因为每帧噪声是独立的。视频扩散通过注意力(attention)或共享噪声将帧耦合来解决此问题。
- **朴素3D注意力 = 内存溢出(OOM)。** 在10秒1080p潜在表示(latent)上进行全3D注意力需要数千亿次操作。分解为空间+时间。
- **数据描述的重要性大于规模。** Sora相比先前工作的主要升级是训练了约10倍更详细的描述（GPT-4重新标注的片段）。OpenAI的技术报告明确指出了这一点。
- **首帧条件化(First-frame conditioning)。** 大多数生产模型也接受一张图像作为第一帧。这就是“图像到视频”(image-to-video)模式；训练包含此变体。
- **物理漂移(Physics drift)。** 长片段（>10秒）会累积细微的不一致性。滑动窗口生成+关键帧锚定(keyframe anchoring)有助于解决。

## 使用它

|  用例 | 2026年优选  |
|----------|-----------|
|  最高质量的文本到视频，托管 | Veo 3或Sora  |
|  相机控制电影效果 | Runway Gen-3（运动笔刷）  |
|  跨片段角色一致性 | Pika 2.0或Kling 2.1  |
|  开放权重，快速微调 | WAN 2.2 + LoRA  |
|  图像到视频 | WAN 2.2-I2V, Kling 2.1 I2V, 或 Runway  |
|  音频到视频唇形同步 | Veo 3（原生音频）或专用唇形同步模型  |
|  视频编辑 | Runway Act-Two, Kling运动笔刷, Flux-Kontext（静态帧）  |

在质量相同的情况下，每秒视频成本在2024到2026年间下降了20倍。

## 发布

保存 `outputs/skill-video-brief.md`。技能接受视频简介（时长、宽高比、风格、镜头规划、主体一致性、音频）并输出：模型+托管、提示框架(prompt scaffolding)（镜头语言、主体描述、运动描述）、种子(seed)+可复现协议(reproducibility protocol)，以及逐帧质量检查清单(frame-level QA checklist)。

## 练习

1. **简单(Easy)。** 在`code/main.py`中，比较(a)独立逐帧采样和(b)联合序列采样的帧间差异。报告差异的均值和方差。
2. **中等(Medium)。** 添加首帧条件：将帧0固定为给定值，采样其余帧。测量固定值的传播情况。
3. **困难(Hard)。** 使用HuggingFace diffusers在本地GPU上运行CogVideoX-2B。对6秒片段在720p下进行20推理步骤计时。分析时空注意力(spatiotemporal attention)以识别瓶颈。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  视频VAE | "3-D VAE" | 将`(T, H, W, C)`压缩为时空潜在表示(spatiotemporal latent)的编码器。 |
|  分块(Patches) | "令牌(tokens)" | 潜在表示的固定大小3D块；输入到DiT。 |
|  分解注意力(Factorized attention) | "空间+时间" | 先对空间运行注意力，再对时间运行；跳过全3D注意力。 |
|  图像到视频 (I2V) | "让这张照片动起来" | 模型接受图像+文本，输出从该图像开始的视频。 |
|  关键帧条件化(Keyframe conditioning) | "锚定帧" | 固定特定帧以控制视频的叙事弧线。 |
|  运动笔刷(Motion brush) | "方向提示" | 用户将运动向量绘制到图像上的UI输入。 |
|  重新标注(Re-captioning) | "密集描述" | 使用LLM用详细提示重新标注训练片段。 |
|  闪烁(Flicker) | "时间伪影" | 帧间不一致性；通过耦合去噪修复。 |

## 生产说明：视频潜在表示是一个内存带宽问题

一个10秒1080p、24fps的片段是240帧×1920×1080×3≈1.5 GB原始像素。经过4倍视频VAE压缩（`2 × spatial × 2 × temporal`）后，每次请求的潜在表示约为100 MB。将其通过时空DiT运行30步，batch为1，则每一步通过HBM移动约3 GB——瓶颈是内存带宽，而非FLOPs。

三个生产调节旋钮(production knobs)，全部直接来自生产推理文献推理章节：

- **TP跨DiT。** 文本到视频模型通常参数≥10B。在4块H100上TP=4是标准配置；对于405B类模型，PP=2×TP=2。每一步延迟大致随TP线性下降，直到全规约(all-reduce)瓶颈。
- **帧批处理=连续批处理。** 在生成时，视频概念上是由注意力连接的一批帧。连续批处理（飞行中调度）适用：如果模型架构允许滑动窗口生成，则在返回帧`t-1`的同时开始渲染帧`t+1`。
- **片段级预填充缓存。** 对于图像到视频，首帧条件化类似于LLM的提示预填充：计算一次，在时间解码器传递中重用。这实际上是视频的KV-cache。

## 延伸阅读

- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — Sora 技术报告。
- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — CogVideoX。
- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — HunyuanVideo。
- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — Mochi-1。
- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — 2025 年中期的开放 SOTA。
- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — 开创性的视频扩散论文。
- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — Stable Video Diffusion 的前身。
