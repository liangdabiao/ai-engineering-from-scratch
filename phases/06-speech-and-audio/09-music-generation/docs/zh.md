# 音乐生成 — MusicGen、Stable Audio、Suno 与许可地震

> 2026 年音乐生成：Suno v5 和 Udio v4 主导商业领域；MusicGen、Stable Audio Open 和 ACE-Step 引领开源。技术问题基本解决。法律问题（华纳音乐 5 亿美元和解、环球音乐集团和解）在 2025-2026 年重塑了该领域。

**类型：** 构建
**语言：** Python
**前提条件：** 阶段 6 · 02（频谱图），阶段 4 · 10（扩散模型）
**时间：** 约 75 分钟

## 问题

文本 → 一段 30 秒到 4 分钟的音乐片段，包含歌词、人声和结构。三个子问题：

1. **器乐生成。** 类似“带有温暖键盘的 lo-fi 嘻哈鼓点”的文本 → 音频。MusicGen、Stable Audio、AudioLDM。
2. **歌曲生成（带人声+歌词）。** “关于德克萨斯雨夜的乡村歌曲” → 完整歌曲。Suno、Udio、YuE、ACE-Step。
3. **条件/可控。** 扩展已有片段、重新生成桥段、切换风格、分离音轨或修复。Udio 的修复+音轨分离是 2026 年的对标功能。

## 核心概念

![Music generation: token-LM vs diffusion, the 2026 model map](../assets/music-generation.svg)

### 基于神经编解码令牌的令牌语言模型

Meta 的 **MusicGen**（2023 年，MIT 许可证）及其众多衍生模型：以文本/旋律嵌入为条件，自回归预测 EnCodec 令牌（32 kHz，4 个码书），用 EnCodec 解码。参数规模 3 亿到 33 亿。强大的基准模型；超过 30 秒时表现不佳。

**ACE-Step**（开源，4B XL 于 2026 年 4 月发布）将此扩展到全歌曲歌词条件生成。开源社区最接近 Suno 的作品。

### 基于梅尔谱或潜变量的扩散

**Stable Audio（2023）** 和 **Stable Audio Open（2024）**：对压缩音频进行潜变量扩散。擅长循环、声音设计、环境纹理。不太擅长结构化的完整歌曲。

**AudioLDM / AudioLDM2**：通过 T2I 风格的潜变量扩散实现文本到音频，泛化到音乐、音效、语音。

### 混合（生产级）— Suno、Udio、Lyria

闭源权重。可能是自回归编解码语言模型 + 基于扩散的声码器，带有专门的语音/鼓点/旋律头。Suno v5（2026 年）是 ELO 1293 质量领导者。Udio v4 增加了修复+音轨分离（贝斯、鼓、人声可单独下载）。

### 评估

- **FAD（弗雷歇音频距离）。** 使用 VGGish 或 PANNs 特征在嵌入层面衡量生成音频与真实音频分布的距离。越低越好。MusicGen small 在 MusicCaps 上为 4.5 FAD；当前最优约 3.0。
- **音乐性（主观）。** 人类偏好。Suno v5 以 ELO 1293 领先。
- **文本-音频对齐。** 提示与输出之间的 CLAP 分数。
- **音乐性伪影。** 节拍错位的过渡、人声短语漂移、超过 30 秒后结构丢失。

## 2026 年模型地图

|  模型  |  参数  |  时长  |  人声  |  许可证  |
|-------|--------|--------|--------|---------|
|  MusicGen-large  |  33 亿  |  30 秒  |  否  |  MIT  |
|  Stable Audio Open  |  12 亿  |  47 秒  |  否  |  Stability 非商业许可证  |
|  ACE-Step XL（2026 年 4 月） |  40 亿  |  > 2 分钟  |  是  |  Apache-2.0  |
|  YuE  |  70 亿  |  > 2 分钟  |  是，多语言  |  Apache-2.0  |
|  Suno v5（闭源） |  ?  |  4 分钟  |  是，ELO 1293  |  商业许可证  |
|  Udio v4（闭源） |  ?  |  4 分钟  |  是 + 音轨  |  商业许可证  |
|  Google Lyria 3（闭源） |  ?  |  实时  |  是  |  商业许可证  |
|  MiniMax Music 2.5  |  ?  |  4 分钟  |  是  |  商业 API  |

## 法律格局（2025-2026）

- **华纳音乐诉 Suno 的和解。** 5 亿美元。WMG 现在对 Suno 上的 AI 肖像、音乐版权和用户生成曲目拥有监督权。类似地，环球音乐集团在 Udio 上达成和解。
- **欧盟人工智能法案** + **加利福尼亚州 SB 942**：人工智能生成的音乐必须披露。
- **Riffusion / MusicGen** 在 MIT 许可证下没有合规负担，但也没有商业人声。

安全发布模式：

1. 仅生成器乐（MusicGen、Stable Audio Open、MIT/CC0 输出）。
2. 使用商业 API（Suno、Udio、ElevenLabs Music）并按每次生成获得许可。
3. 在自有或授权目录上训练（大多数企业最终选择此方式）。
4. 为生成内容添加水印和元数据标签。

## 动手构建

### 第 1 步：使用 MusicGen 生成

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

三种尺寸：`small`（3 亿，快速）、`medium`（15 亿）、`large`（33 亿）。小尺寸足以判断“想法是否成立”。

### 步骤2：旋律条件(Conditioning)

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody 接收色度图(Chromagram)并保留曲调，同时交换音色。适用于"把这个旋律变成弦乐四重奏"。

### 步骤3：FAD评估

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

计算 VGGish 嵌入距离。适用于流派级别的回归测试；不能替代人工听评。

### 步骤4：整合到LLM音乐工作流中

结合第7-8课的思想：

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 使用它

|  目标  |  技术栈  |
|------|-------|
|  乐器音效设计  |  Stable Audio Open  |
|  游戏/自适应音乐  |  Google Lyria RealTime (封闭)  |
|  含人声完整歌曲 (商业)  |  Suno v5 或 Udio v4（明确许可）  |
|  含人声完整歌曲 (开放)  |  ACE-Step XL 或 YuE  |
|  短广告曲  |  基于哼唱参考进行旋律条件(Melody Conditioning)的 MusicGen  |
|  音乐视频背景  |  MusicGen + Stable Video Diffusion  |

## 2026年仍存在的陷阱

- **版权清洗提示。** "泰勒·斯威夫特风格的歌曲"——商业版 Suno/Udio 现已过滤此类提示，开放模型则不会。请自行添加过滤列表。
- **重复/超过30秒后漂移。** 自回归模型会循环。对多次生成进行交叉淡入淡出，或使用 ACE-Step 实现结构连贯性。
- **速度漂移。** 模型会偏离 BPM。在提示词中加入 BPM 标签，并用 librosa 的 `beat_track` 进行后过滤。
- **人声清晰度。** Suno 表现优异；开放模型在人声上往往模糊不清。如果歌词很重要，请使用商业 API 或进行微调。
- **单声道输出。** 开放模型生成单声道或伪立体声。通过合适的立体声重建（ezst、Cartesia 的立体声扩散）进行升级。

## 发布

保存为 `outputs/skill-music-designer.md`。选择模型、许可策略、长度/结构计划以及用于音乐生成部署的披露元数据。

## 练习

1. **简单。** 运行 `code/main.py`。它会生成一个"生成式"和弦进行+鼓点模式，以 ASCII 符号表示——一个音乐生成卡通。如果需要，可以通过任何 MIDI 渲染器播放。
2. **中等。** 安装 `code/main.py`，用 MusicGen-small 针对4个流派提示生成10秒片段，对照参考流派集测量 FAD。
3. **困难。** 使用 ACE-Step（或 MusicGen-melody），用不同的音色提示生成同一曲调的三个变体。计算与提示词的 CLAP 相似度以验证对齐。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  FAD  |  Audio FID  |  真实与生成的嵌入分布之间的 Fréchet 距离。  |
|  色度图(Chromagram)  |  旋律音高  |  每帧12维向量；旋律条件(Melody Conditioning)的输入。  |
|  分轨(Stems)  |  乐器轨道  |  分离的贝斯/鼓/人声/旋律，WAV格式。  |
|  填充(Inpainting)  |  重新生成一段  |  遮罩一个时间窗口；模型仅重新生成该部分。  |
|  CLAP  |  文本-音频 CLIP  |  对比音频-文本嵌入；评估文本-音频对齐。  |
|  EnCodec  |  音乐编解码器  |  Meta 的神经编解码器，被 MusicGen 使用；32 kHz，4个码本。  |

## 延伸阅读

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) —— 开放自回归基准。
- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) —— 音效设计默认选择。
- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) —— 开放4B参数完整歌曲生成器，2026年4月。
- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) —— 商业质量领导者。
- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) —— 音乐+音效的潜在扩散模型。
- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) —— 2025年11月前例。
