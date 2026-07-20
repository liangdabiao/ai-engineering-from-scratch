# 神经音频编解码器 — EnCodec、SNAC、Mimi、DAC 与语义-声学分离

> 2026年的音频生成几乎全是令牌(Token)的天下。EnCodec、SNAC、Mimi和DAC将连续波形转换为离散序列，使Transformer能够预测。语义与声学令牌的分离——第一个码本作为语义，其余作为声学——是自Transformer以来音频领域最重要的架构变革。

**类型:** 学习
**语言:** Python
**前置知识:** 阶段6·02（频谱图），阶段10·11（量化），阶段5·19（子词令牌化）
**时间:** 约60分钟

## 问题

语言模型处理离散令牌(Token)，而音频是连续的。如果你想要一个类似LLM的语音/音乐模型——MusicGen、Moshi、Sesame CSM、VibeVoice、Orpheus——你首先需要一个**神经音频编解码器(Neural Audio Codec)**：一个学习得到的编码器，将音频离散化为少量令牌组成的词汇表，以及一个匹配的解码器来重构波形。

目前已出现两大类别：

1. **重构优先的编解码器** — EnCodec、DAC。优化感知音频质量。令牌是“声学”的——它们捕获所有信息，包括说话人身份、音色、背景噪声。
2. **语义优先的编解码器** — Mimi（Kyutai）、SpeechTokenizer。强制第一个码本编码语言/语音内容（通常通过从WavLM蒸馏得到）。后续码本是声学细节。

2024-2026年的洞察：**纯粹的重构编解码器在尝试从文本生成时会产生模糊的语音。** 基于编解码器令牌的LLM不得不在同一个码本中同时学习语言结构和声学结构，这难以扩展。将它们分离——语义码本0，声学码本1-N——正是Moshi和Sesame CSM成功的原因。

## 核心概念

![Four codec landscape: EnCodec, DAC, SNAC (multi-scale), Mimi (semantic+acoustic)](../assets/codec-comparison.svg)

### 核心技巧：残差向量量化(Residual Vector Quantization, RVQ)

与使用单个大型码本（这需要数百万个码字才能达到良好质量）不同，所有现代音频编解码器都使用**RVQ**：一系列小型码本的级联。第一个码本对编码器输出进行量化；第二个对残差进行量化；以此类推。每个码本包含1024个码字。8个码本的有效词汇量是1024^8 = 10^24。

推理时，解码器对所有选中的每帧码字求和以进行重构。

### 2026年重要的四种编解码器

**EnCodec（Meta，2022）** 基准模型。基于波形的编码器-解码器，RVQ瓶颈。24 kHz，最多32个码本，默认4个码本@1.5 kbps。使用`1D conv + transformer + 1D conv`架构。被MusicGen使用。

**DAC（Descript，2023）** 使用L2归一化码本、周期激活函数和改进的损失函数的RVQ。所有开源编解码器中重构保真度最高——使用12个码本时有时与原语音难以区分。44.1 kHz全频带。

**SNAC（Hubert Siuzdak，2024）** 多尺度RVQ——粗码本以低于细码本的帧率运行。有效地对音频进行分层建模：约12 Hz的粗略“草图”加上50 Hz的细节。被Orpheus-3B使用，因为分层结构很好地映射到基于LM的生成。

**Mimi（Kyutai，2024）** 2026年的游戏规则改变者。12.5 Hz帧率（极低），8个码本@4.4 kbps。码本0**从WavLM蒸馏得到**——训练用于预测WavLM的语音内容特征。码本1-7是声学残差。这种分离驱动了Moshi（第15课）和Sesame CSM。

### 帧率对语言建模至关重要

更低的帧率 = 更短的序列 = 更快的语言模型。

|  编解码器  |  帧率  |  1秒 = N帧  |  适用场景  |
|-------|-----------|----------------|---------|
|  EnCodec-24k  |  75 Hz  |  75  |  音乐、通用音频  |
|  DAC-44.1k  |  86 Hz  |  86  |  高保真音乐  |
|  SNAC-24k (粗)  |  ~12 Hz  |  12  |  AR-LM高效  |
|  Mimi  |  12.5 Hz  |  12.5  |  流式语音  |

在12.5 Hz下，一个10秒的话语只有125个编解码器帧——Transformer可以轻松预测它们。

### 语义令牌与声学令牌

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **语义令牌（Mimi中的码本0）** 编码所说的内容——音素、单词、语义。通过辅助预测损失从WavLM蒸馏得到。
- **声学令牌（码本1-7）** 编码音色、说话人身份、韵律、背景噪声、精细细节。

自回归语言模型首先预测语义令牌（以文本为条件），然后预测声学令牌（以语义+说话人参考为条件）。这种分解是现代TTS能够零样本克隆语音的原因：语义模型处理内容；声学模型处理音色。

### 2026年重构质量（每秒比特数，码率越低越好）

|  编解码器  |  码率  |  PESQ  |  ViSQOL  |
|-------|---------|------|--------|
|  Opus-20kbps  |  20 kbps  |  4.0  |  4.3  |
|  EnCodec-6kbps  |  6 kbps  |  3.2  |  3.8  |
| DAC-6kbps  |  6 kbps  |  3.5  |  4.0  |
| SNAC-3kbps  |  3 kbps  |  3.3  |  3.8  |
| Mimi-4.4kbps  |  4.4 kbps  |  3.1  |  3.7  |

传统编解码器如Opus在感知质量上仍然每比特领先。神经编解码器在**离散令牌**（Opus不产生）和**生成模型质量**（语言模型能用这些令牌做什么）方面获胜。

## 动手构建

### 第1步：用EnCodec编码

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

`n_codebooks=8` 在6 kbps下。每个码是0-1023（10比特）。

### 第2步：解码并测量重建

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### 第3步：语义-声学分离（Mimi风格）

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

语义码本0与WavLM对齐。你可以训练一个文本到语义的Transformer——词汇量比直接到音频小得多。然后一个单独的声学到波形解码器以说话人参考为条件。

### 第4步：为什么AR语言模型在编解码器令牌上有效

对于Mimi的12.5 Hz × 8码本下的10秒语音片段：

```
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000个令牌对Transformer来说是微不足道的上下文。一个256M参数的Transformer在现代GPU上可以在毫秒内生成10秒语音。

## 使用它

问题映射→编解码器：

|  任务  |  编解码器  |
|------|-------|
|  通用音乐生成  |  EnCodec-24k  |
|  最高保真度重建  |  DAC-44.1k  |
|  语音上的AR语言模型（TTS）  |  SNAC或Mimi  |
|  流式全双工语音  |  Mimi（12.5 Hz）  |
|  带文本的音效库  |  EnCodec + T5条件  |
|  细粒度音频编辑  |  DAC + 修复  |

经验法则：**如果你在构建生成模型，从Mimi或SNAC开始。如果你在构建压缩流程，使用Opus。**

## 陷阱

- **码本太多。**增加码本线性提高保真度，但也线性增加语言模型序列长度。停在8-12个。
- **帧率不匹配。**在12.5 Hz Mimi上训练语言模型，然后在50 Hz EnCodec上微调会无声失败。
- **假设所有码本相等。**在Mimi中，码本0携带内容；丢失它会破坏可懂度。丢失码本7几乎察觉不到。
- **仅以重建质量为指标。**一个编解码器可能有很好的重建，但如果语义结构糟糕，对基于语言模型的生成毫无用处。

## 发布

另存为`outputs/skill-codec-picker.md`。为给定的生成或压缩任务选择一个编解码器。

## 练习

1. **简单。**运行`code/main.py`。它实现了一个玩具标量+残差量化器，并在添加码本时测量重建误差。
2. **中等。**安装`code/main.py`并在一个保留的语音片段上比较1、4、8、32个码本。绘制PESQ或MSE与比特率的关系图。
3. **困难。**加载Mimi。编码一个片段。将码本0替换为随机整数；解码。然后类似地替换码本7。比较两种破坏——码本0破坏应破坏可懂度；码本7破坏几乎不会改变任何东西。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  RVQ  |  残差量化  |  小码本的级联；每个量化前一个残差。  |
|  帧率  |  编解码器速度  |  每秒多少令牌帧。越低=语言模型越快。  |
|  语义码本  |  码本0（Mimi）  |  从SSL特征蒸馏的码本；编码内容。  |
|  声学码本  |  其他所有  |  音色、韵律、噪声、细节。  |
|  PESQ / ViSQOL  |  感知质量  |  与MOS相关的客观指标。  |
|  EnCodec  |  Meta编解码器  |  RVQ基线；被MusicGen使用。  |
| Mimi | Kyutai codec | 12.5 Hz帧率；语义-声学分离；驱动Moshi。 |

## 延伸阅读

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) — RVQ基线。
- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) — 最高保真度的开放编解码器。
- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) — 多尺度RVQ。
- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) — 语义-声学分离，WavLM蒸馏。
- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) — 两阶段语义/声学范式。
- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) — 原始可流式RVQ编解码器。
