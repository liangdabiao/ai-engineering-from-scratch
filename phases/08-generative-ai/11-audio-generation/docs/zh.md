# 音频生成

> 音频是16-48 kHz的一维信号。一个五秒的片段包含80-240k个样本。没有Transformer能直接处理这种长度的序列。2026年每个生产级音频模型的解决方案都一样：神经编解码器(Encodec、SoundStream、DAC)将音频压缩为50-75 Hz的离散令牌(token)，然后由Transformer或扩散模型生成令牌。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段6·02（音频特征）、阶段6·04（ASR）、阶段8·06（DDPM）
**时间：** 约45分钟

## 问题

三项音频生成任务：

1. **文本转语音(TTS)。** 给定文本，产生语音。干净的语音是窄带的，具有强语音结构——通过令牌上的Transformer很好地解决。例如：VALL-E（微软）、NaturalSpeech 3、ElevenLabs、OpenAI TTS。
2. **音乐生成。** 给定提示（文本、旋律、和弦进行、流派），产生音乐。分布更广泛。例如：MusicGen（Meta）、Stable Audio 2.5、Suno v4、Udio、Riffusion。
3. **音频特效/声音设计。** 给定提示，产生环境音或拟音。例如：AudioGen、AudioLDM 2、Stable Audio Open。

这三者都运行在相同的基础上：神经音频编解码器 + 令牌自回归(token-AR)或扩散生成器。

## 核心概念

![Audio generation: codec tokens + transformer or diffusion](../assets/audio-generation.svg)

### 神经音频编解码器

Encodec（Meta，2022）、SoundStream（Google，2021）、Descript Audio Codec（DAC，2023）。卷积编码器将波形压缩为每个时间步的向量；残差向量量化(RVQ)将每个向量转换为K个码本索引的级联。解码器反转该过程。24 kHz音频在2 kbps下使用8个RVQ码本，75 Hz = 600令牌/秒。

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### 两种生成范式建立在此之上

**令牌自回归(Token-autoregressive)。** 将RVQ令牌展平为序列，运行仅解码器Transformer。MusicGen使用“延迟并行”方法，通过每个流的偏移并行发出K个码本流。VALL-E从文本提示和3秒语音样本生成语音令牌。

**潜在扩散(Latent diffusion)。** 将编解码器令牌打包为连续潜在变量，或用分类扩散建模。Stable Audio 2.5在连续音频潜在变量上使用流匹配(flow matching)。AudioLDM 2使用文本到梅尔频谱再到音频的扩散。

2024-2026年的趋势：流匹配在音乐领域胜出（推理更快，样本更干净），而令牌自回归在语音领域仍占主导地位，因为它天然是因果的且适合流式传输。

## 生产环境概览

|  系统  |  任务  |  主干网络  |  延迟  |
|--------|------|----------|---------|
|  ElevenLabs V3  |  TTS  |  令牌自回归 + 神经声码器  |  首令牌约300ms  |
|  OpenAI GPT-4o audio  |  全双工语音  |  端到端多模态自回归  |  约200ms  |
|  NaturalSpeech 3  |  TTS  |  潜在流匹配  |  非流式  |
|  Stable Audio 2.5  |  音乐 / 音效  |  DiT + 音频潜在流匹配  |  1分钟片段约10s  |
|  Suno v4  |  完整歌曲  |  未公开；疑似令牌自回归  |  每首歌约30s  |
|  Udio v1.5  |  完整歌曲  |  未公开  |  每首歌约30s  |
|  MusicGen 3.3B  |  音乐  |  基于Encodec 32kHz的令牌自回归  |  实时  |
|  AudioCraft 2  |  音乐 + 音效  |  流匹配  |  5秒片段约5s  |
|  Riffusion v2  |  音乐  |  频谱图扩散  |  约10s  |

## 动手构建

`code/main.py` 模拟了核心思想：在合成“音频令牌”序列上训练一个极小的下一个令牌预测Transformer，这些序列由两种不同的“风格”生成（风格A交替低和高令牌，风格B单调斜坡）。以风格为条件进行采样。

### 第一步：合成音频令牌

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### 第二步：训练一个极小的令牌预测器

一个以风格为条件的二元模型风格预测器。要点在于模式：编解码器令牌 → 交叉熵训练 → 自回归采样。

### 第三步：条件采样

给定风格令牌和起始令牌，从预测分布中采样下一个令牌。持续20-40个令牌。

## 陷阱

- **编解码器质量限制输出质量。** 如果编解码器不能忠实地表示声音，无论生成器质量多高都无济于事。DAC是目前最好的开源方案。
- **RVQ误差积累。** 每个RVQ层对前一层残差建模。第一层的误差会传播。在高层用温度0采样有帮助。
- **音乐结构。** 30秒的令牌在75 Hz下超过20k个令牌。对Transformer来说很困难。MusicGen使用滑动窗口和提示延续；Stable Audio使用更短的片段和交叉淡化。
- **边界伪影。** 生成片段之间的交叉淡化需要仔细的重叠相加。
- **干净数据需求。** 音乐生成器需要数万小时的有许可音乐。Suno/Udio的RIAA诉讼（2024年）使这一问题浮出水面。
- **语音克隆伦理。** 3秒样本加文本提示足以让VALL-E/XTTS/ElevenLabs克隆声音。每个生产模型都需要滥用检测和退出名单。

## 使用它

|  任务  |  2026 年技术栈  |
|------|------------|
|  商业 TTS  |  ElevenLabs、OpenAI TTS 或 Azure Neural  |
|  语音克隆（已确认同意）  |  XTTS v2（开源）或 ElevenLabs Pro  |
|  背景音乐，快速  |  Stable Audio 2.5 API、Suno 或 Udio  |
|  带歌词的音乐  |  Suno v4 或 Udio v1.5  |
|  音效 / 拟音  |  AudioCraft 2、ElevenLabs SFX 或 Stable Audio Open  |
|  实时语音智能体  |  GPT-4o 实时或 Gemini Live  |
|  开源权重音乐研究  |  MusicGen 3.3B、Stable Audio Open 1.0、AudioLDM 2  |
|  配音 / 翻译  |  HeyGen、ElevenLabs Dubbing  |

## 发布

保存 `outputs/skill-audio-brief.md`。技能获取音频简介（任务、时长、风格、声音、许可证）并输出：模型 + 托管、提示格式（流派标签、风格描述符、结构标记）、编解码器 + 生成器 + 声码器链、种子协议和评估计划（MOS / CLAP 得分 / TTS 的 CER / 用户 A/B 测试）。

## 练习

1. **简单.** 运行 `code/main.py` 并明确设置风格。验证生成的序列是否符合该风格的模式。
2. **中等.** 添加延迟并行解码：模拟 2 个令牌流，它们必须保持 1 步偏移。训练一个联合预测器。
3. **困难.** 使用 HuggingFace transformers 在本地运行 MusicGen-small。用三个不同的提示生成一个 10 秒的片段；进行 A/B 测试以评估风格遵循度。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  编解码器  |  "神经压缩"  |  音频编码器/解码器；典型输出为 50-75 Hz 令牌。  |
|  RVQ  |  "残差矢量量化"  |  级联的 K 个量化器；每个量化器建模前一个量化器的残差。  |
|  令牌  |  "一个编解码器符号"  |  码本中的离散索引；典型大小为 1024 或 2048。  |
|  延迟并行  |  "偏移码本"  |  以交错偏移发射 K 个令牌流，以减少序列长度。  |
|  流匹配  |  "2024 年音频领域的胜者"  |  扩散模型的路径更直的替代方案；采样速度更快。  |
|  语音提示  |  "3 秒样本"  |  引导克隆语音的说话人嵌入或令牌前缀。  |
|  梅尔频谱图  |  "可视化表示"  |  对数幅度感知频谱图；被许多 TTS 系统使用。  |
|  声码器  |  "梅尔转波形"  |  将梅尔频谱图转换回音频的神经组件。  |

## 生产注意：音频是一个流式问题

音频是用户期望**随生成过程实时到达**而非一次性全部到达的输出模态。在生产术语中，这意味着 TPOT（每个输出令牌的时间）很重要，因为用户的收听速度是目标吞吐量——而不是他们的阅读速度。对于以约 75 令牌/秒（Encodec）进行令牌化的 16kHz 音频，服务器必须为每个用户生成 ≥75 令牌/秒，以保持播放流畅。

两个架构上的后果：

- **流匹配音频模型无法轻松实现流式传输。** Stable Audio 2.5 和 AudioCraft 2 一次性渲染固定长度的片段。要实现流式传输，你需要对片段进行分块并重叠边界——类似于滑动窗口扩散——与编解码器自回归模型相比，会增加 100-300 毫秒的延迟开销。

如果产品是“实时语音聊天”或“实时音乐续写”，请选择编解码器自回归路径。如果是“提交时渲染 30 秒片段”，则流匹配在质量和总延迟方面胜出。

## 延伸阅读

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — 编解码器标准。
- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — 第一个广泛使用的神经音频编解码器。
- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — DAC。
- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — VALL-E。
- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — MusicGen。
- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — AudioLDM 2。
- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — 2025 年采用流匹配的文本到音乐。
