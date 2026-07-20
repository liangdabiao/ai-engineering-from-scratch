# 音频基础——波形、采样、傅里叶变换

> 波形是原始信号。频谱图是表示形式。Mel特征是机器学习友好形式。每个现代ASR和TTS流水线都沿着这个阶梯前进，第一级是理解采样和傅里叶变换。

**类型：** 学习
**语言：** Python
**先决条件：** 阶段1·06（向量与矩阵），阶段1·14（概率分布）
**时间：** ~45分钟

## 问题

麦克风产生压力-时间信号。你的神经网络消耗张量。两者之间有一堆约定，违反这些约定会产生隐蔽的bug：模型训练良好但WER翻倍，或TTS发出嘶嘶声，或语音克隆系统记住了麦克风而非说话者。

语音系统中的每个bug都可追溯到以下三个问题之一：

1. 数据是以什么采样率录制的，模型期望什么采样率？
2. 信号是否发生了混叠？
3. 你是在原始样本上操作还是在频率表示上操作？

这些做对了，阶段6的其余部分就易于处理。做错了，即使Whisper-Large-v4也会输出垃圾。

## 核心概念

![Waveform, sampling, DFT, and frequency bins visualized](../assets/audio-fundamentals.svg)

**波形。** `[-1.0, 1.0]`中的一维浮点数数组。以样本编号索引。要转换为秒，除以采样率：`t = n / sr`。一个16 kHz的10秒片段是一个160,000个浮点数的数组。

**采样率(sr)。** 每秒样本数。2026年的常见采样率：

|  速率  |  用途  |
|------|-----|
|  8 kHz  |  电话，传统VoIP。4 kHz的奈奎斯特频率消灭了辅音。避免用于ASR。  |
|  16 kHz  |  ASR标准。Whisper、Parakeet、SeamlessM4T v2都使用16 kHz。  |
|  22.05 kHz  |  旧模型TTS声码器训练。  |
|  24 kHz  |  现代TTS（Kokoro、F5-TTS、xTTS v2）。  |
|  44.1 kHz  |  CD音频，音乐。  |
|  48 kHz  |  电影、专业音频、高保真TTS（VALL-E 2、NaturalSpeech 3）。  |

**奈奎斯特-香农定理。** `sr`的采样率可以明确表示高达`sr/2`的频率。`sr/2`边界是*奈奎斯特频率*。高于奈奎斯特频率的能量会被*混叠*——折叠到较低频率——并破坏信号。在降采样前始终使用低通滤波器。

**位深度。** 16位PCM（有符号int16，范围±32,767）是通用交换格式。音乐用24位，内部DSP用32位浮点数。像`soundfile`这样的库读取int16，但在`[-1, 1]`中暴露float32数组。

**傅里叶变换。** 任何有限信号都是不同频率正弦波之和。离散傅里叶变换(DFT)对`N`个样本计算`N`个复数系数——每个频率槽一个。`bin k`映射到频率`k · sr / N`Hz。幅度是该频率下的振幅，角度是相位。

**FFT。** 快速傅里叶变换：当`N`是2的幂时，用于DFT的`O(N log N)`算法。每个音频库都在底层使用FFT。在16 kHz下进行1024样本FFT给出512个可用频率槽，横跨0–8 kHz，分辨率15.6 Hz。

**帧加窗。** 我们不对整个片段做FFT。我们将其切分成重叠的*帧*（通常25 ms，步长10 ms），将每帧乘以一个窗函数（汉宁窗、汉明窗）以消除边缘不连续性，然后对每帧做FFT。这就是短时傅里叶变换(STFT)。第02课从这里开始。

```figure
mel-scale
```

## 动手构建

### 第1步：读取一个片段并绘制波形

`code/main.py`仅使用标准库`wave`模块以保持演示无依赖。在生产环境中，你将使用`soundfile`或`torchaudio.load`（两者都返回`(waveform, sr)`元组）：

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### 第2步：从基本原理合成正弦波

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

一个440 Hz的正弦波（音乐会A音）在16 kHz下持续1秒是16,000个浮点数。使用`wave.open(..., "wb")`以16位PCM编码写入。

### 第3步：手动计算DFT

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)` — 适用于`N=256`确认正确性，对实际音频无用。实际代码调用`numpy.fft.rfft`或`torch.fft.rfft`。

### 第4步：找出主频率

幅度峰值索引`k_star`映射到频率`k_star * sr / N`。在440 Hz正弦波上运行此步骤应返回槽`440 * N / sr`处的峰值。

### 第5步：演示混叠

以10 kHz采样7 kHz正弦波（奈奎斯特(Nyquist)=5 kHz）。7 kHz音调高于奈奎斯特频率，折叠到`10 − 7 = 3 kHz`。FFT峰值出现在3 kHz。这是经典的混叠演示，也是每个DAC/ADC都配备砖墙低通滤波器的原因。

## 使用它

你将在2026年实际交付的技术栈：

|  任务 | 库 | 原因  |
|------|---------|-----|
|  读/写WAV/FLAC/OGG | `soundfile` (libsndfile封装) | 最快、稳定，返回float32。  |
|  重采样 | `torchaudio.transforms.Resample` 或 `librosa.resample` | 内置正确的抗混叠功能。  |
|  STFT / Mel | `torchaudio` 或 `librosa` | GPU友好；PyTorch生态系统。  |
|  实时流处理 | `sounddevice` 或 `pyaudio` | 跨平台PortAudio绑定。  |
|  检查文件 | `ffprobe` 或 `soxi` | 命令行界面，快速，报告采样率/通道/编解码器。  |

决策规则：**匹配任何其他内容之前，先匹配采样率**。Whisper期望16 kHz单声道float32。如果传入44.1 kHz立体声，你会得到看起来像模型bug的垃圾数据。

## 发布

保存为`outputs/skill-audio-loader.md`。这项技能帮助你检查音频输入是否满足下游模型的期望，并在不匹配时正确重采样。

## 练习

1. **简单。** 以16 kHz合成一个1秒的混合音，包含220 Hz、440 Hz和880 Hz。运行DFT。确认在预期的频率柱上有三个峰值。
2. **中等。** 以48 kHz录制一段3秒的语音WAV。使用`torchaudio.transforms.Resample`（带抗混叠）降采样到16 kHz，然后使用朴素抽取（每三个样本取一个）降采样到16 kHz。对两者进行FFT。混叠出现在哪里？
3. **困难。** 仅使用`torchaudio.transforms.Resample`和第3步的DFT从头构建STFT。帧大小400，步长160，汉宁窗。用`math`绘制幅度谱。这是第02课的频谱图。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  采样率 | 每秒样本数 | ADC测量信号的频率（Hz）。  |
|  奈奎斯特(Nyquist) | 能表示的最高频率 | `sr/2`；高于它的能量会混叠回来。  |
|  位深 | 每个样本的分辨率 | `int16` = 65,536级；`float32` = `[-1, 1]`中的24位精度。  |
|  DFT | 序列的傅里叶变换 | `N`个样本 → `N`个复频率系数。  |
|  FFT | 快速DFT | `O(N log N)`算法，要求`N`为2的幂。  |
|  频率柱(Bin) | 频率列 | `k · sr / N` Hz；分辨率 = `sr / N`。  |
|  STFT | 频谱图的幕后原理 | 随时间进行分帧加窗的FFT。  |
|  混叠(Aliasing) | 奇怪的频率鬼影 | 高于奈奎斯特频率的能量镜像到较低频率柱。  |

## 延伸阅读

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — 采样定理背后的论文。
- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — 免费的经典DSP教科书。
- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — 附带代码的实用教程。
- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — 解释为什么真实世界的音频不是干净正弦波的参考资料。
- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — 十分钟内弄清频率柱的直觉。
