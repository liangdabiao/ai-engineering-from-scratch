# 音频变换器——Whisper 架构

> 音频是频率随时间变化的图像。Whisper 是一种视觉变换器（ViT），它“吃掉”梅尔频谱图并“说出”结果。

**类型：** 学习
**语言：** Python
**前置知识：** 阶段7·05（完整变换器），阶段7·08（编码器-解码器），阶段7·09（ViT）
**时间：** 约45分钟

## 问题

在 Whisper（OpenAI, Radford et al. 2022）出现之前，最先进的自动语音识别（ASR）是 wav2vec 2.0 和 HuBERT——自监督特征提取器加微调头部。质量高，但数据流水线昂贵，领域脆弱。多语种语音识别需要为每个语系训练单独的模型。

Whisper 做了三个赌注：

1. **在所有数据上训练。** 从互联网上收集了 68 万小时的弱标注音频，覆盖 97 种语言。没有干净的学术语料库。没有音素标注。
2. **多任务单一模型。** 一个解码器通过任务令牌联合训练，执行转录、翻译、语音活动检测、语言识别和时间戳生成。
3. **标准的编码器-解码器变换器。** 编码器接收对数梅尔频谱图。解码器自回归地生成文本令牌。没有声码器，没有 CTC，没有 HMM。

结果：Whisper large-v3 在口音、噪声和缺乏干净标注数据的语言上表现稳健。它是 2026 年所有开源语音助手和大多数商业语音助手的默认语音前端。

## 核心概念

![Whisper pipeline: audio → mel → encoder → decoder → text](../assets/whisper.svg)

### 步骤 1——重采样 + 加窗

音频为 16 kHz。裁剪/填充至 30 秒。计算对数梅尔频谱图：80 个梅尔频带，10 毫秒步长 → 约 3000 帧 × 80 个特征。这就是 Whisper“看到”的“输入图像”。

### 步骤 2——卷积主干

两个卷积层，核大小为 3，步长为 2，将 3000 帧减少至 1500 帧。在不增加大量参数的情况下将序列长度减半。

### 步骤 3——编码器

一个 24 层（large 模型）的变换器编码器，处理 1500 个时间步。正弦位置编码，自注意力，GELU 前馈网络。生成 1500 × 1280 的隐藏状态。

### 步骤 4——解码器

一个 24 层的变换器解码器。它自回归地从 BPE 词汇表中生成令牌，该词汇表是 GPT-2 的超集，并包含一些音频特定的特殊令牌。

### 步骤 5——任务令牌

解码器提示以控制令牌开始，告诉模型要做什么：

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

或者

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

模型在此约定下训练。您通过前缀控制任务。这相当于 2026 年的指令微调，但应用于语音。

### 步骤 6——输出

束搜索（宽度为 5），带有对数概率阈值。当 `<|notimestamps|>` 令牌不存在时，每 0.02 秒音频预测一次时间戳。

### Whisper 模型尺寸

|  模型  |  参数量  |  层数  |  隐藏层维度  |  注意力头数  |  显存 (fp16)  |
|-------|--------|--------|---------|-------|-------------|
|  Tiny  |  39M  |  4  |  384  |  6  |  ~1 GB  |
|  Base  |  74M  |  6  |  512  |  8  |  ~1 GB  |
|  Small  |  244M  |  12  |  768  |  12  |  ~2 GB  |
|  Medium  |  769M  |  24  |  1024  |  16  |  ~5 GB  |
|  Large  |  1550M  |  32  |  1280  |  20  |  ~10 GB  |
|  Large-v3  |  1550M  |  32  |  1280  |  20  |  ~10 GB  |
|  Large-v3-turbo  |  809M  |  32  |  1280  |  20  |  ~6 GB（4层解码器）  |

Large-v3-turbo（2024 年）将解码器从 32 层削减至 4 层。解码速度提升 8 倍，词错误率（WER）增加不到 1 个百分点。这种解码速度的提升使得 Whisper-turbo 成为 2026 年实时语音助手的默认选择。

### Whisper 所不具备的功能

- 无说话人识别（谁在说话）。需搭配 pyannote 使用。
- 原生不支持实时流式传输——30秒窗口是固定的。现代封装器（如 `faster-whisper`、`WhisperX`）通过 VAD + 重叠实现流式传输。
- 若无外部分块，无法处理超过30秒的长时上下文。实际使用中效果良好，因为人类语音通常不需要长距离上下文来完成转录。

### 2026年现状

|  任务  |  模型  |  备注  |
|------|-------|-------|
|  英语自动语音识别  |  Whisper-turbo, Moonshine  |  Moonshine 在边缘设备上快4倍  |
|  多语种自动语音识别  |  Whisper-large-v3  |  97种语言  |
|  流式自动语音识别  |  faster-whisper + VAD  |  可实现150毫秒延迟目标  |
|  文本转语音  |  Piper, XTTS-v2, Kokoro  |  编码器-解码器模式，但形状类似Whisper  |
|  音频+语言  |  AudioLM, SeamlessM4T  |  单一Transformer中的文本令牌+音频令牌  |

## 动手构建

详见 `code/main.py`。我们不训练Whisper——我们构建对数梅尔频谱图流水线 + 任务令牌提示格式化器。这些才是你在生产环境中实际接触的部分。

### 步骤1：合成音频

生成一个440 Hz、16 kHz采样的1秒正弦波。共16,000个样本。

### 步骤2：对数梅尔频谱图（简化版）

完整梅尔频谱图需要FFT。我们采用简化分帧+每帧能量版本，展示流水线而不需要依赖 `librosa`：

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

帧长25毫秒，帧移10毫秒。匹配Whisper的加窗方式。为便于教学，每帧能量代替梅尔频带。

### 步骤3：填充至30秒

Whisper总是处理30秒的块。将频谱图填充（或截断）至3,000帧。

### 步骤4：构建提示令牌

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

这就是完整的任务控制面。一个4令牌前缀。

## 使用它

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

更快速、兼容OpenAI：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026年何时选择Whisper：**

- 使用单一模型进行多语种自动语音识别。
- 对嘈杂、多样音频的鲁棒转录。
- 研究/原型自动语音识别——最快的起点。

**何时选择其他方案：**

- 边缘设备上的超低延迟流式传输——Moonshine在相同质量下优于Whisper。
- 需要小于200毫秒的实时对话AI——专用流式自动语音识别。
- 说话人识别——Whisper不支持此功能；需集成 pyannote。

## 发布

详见 `outputs/skill-asr-configurator.md`。该技能为新的语音应用选择自动语音识别模型、解码参数和预处理流水线。

## 练习

1. **简单.** 运行 `code/main.py`。确认16 kHz、10毫秒帧移的1秒信号帧数约为100帧。30秒：约3,000帧。
2. **中等.** 使用 `code/main.py` 构建完整对数梅尔频谱图。验证80个梅尔频带与 `numpy.fft` 在数值误差内一致。
3. **困难.** 实现流式推理：将音频分块为10秒窗口（重叠2秒），对每个块运行Whisper，合并转录文本。在5分钟播客样本上测量与单次处理的词错误率。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  梅尔频谱图  |  "音频图像"  |  二维表示：一轴为频率带，另一轴为时间帧；每个单元是对数缩放的能量。  |
|  对数梅尔频谱  |  "Whisper所见"  |  经过对数处理的梅尔频谱图；近似人类对响度的感知。  |
|  帧  |  "一个时间切片"  |  25毫秒的样本窗口；以10毫秒步长重叠。  |
|  任务令牌  |  "语音的提示前缀"  |  解码器提示中的特殊令牌，如 `<\ | transcribe\ | >` / `<\ | translate\ | >`。  |
| 语音活动检测(VAD)  |  “寻找语音”  |  在ASR前消除静音的模块；大幅降低成本。 |
| CTC  |  “连接主义时序分类(Connectionist Temporal Classification)”  |  经典的免对齐训练ASR损失函数；Whisper并未使用它。 |
| Whisper-turbo  |  “小解码器，完整编码器”  |  large-v3编码器 + 4层解码器；解码速度提升8倍。 |
| Faster-whisper  |  “生产级封装”  |  CTranslate2重实现；int8量化；比OpenAI参考实现快4倍。 |

## 延伸阅读

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper论文。
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 参考代码与模型权重。阅读 [OpenAI Whisper repo](https://github.com/openai/whisper) 可在约400行代码中从上到下查看Conv1D前端、编码器和解码器。
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 束搜索与任务令牌逻辑（步骤5-6所述）在此处；500行，完全可读。
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 前身；在某些场景下仍为SOTA特征。
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 生产级封装，比参考实现快4倍。
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 2024年边缘端友好型ASR，形状类似Whisper但更小。
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 标准微调方案，包含梅尔频谱预处理和令牌时间戳处理。
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 完整实现（编码器、解码器、交叉注意力、生成），与课程架构图一致。
