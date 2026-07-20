# 语谱图(Spectrograms)、梅尔刻度(Mel Scale)与音频特征(Audio Features)

> 神经网络不擅长处理原始波形。它们处理语谱图。它们处理梅尔语谱图(mel spectrograms)更佳。2026年的每一个ASR、TTS和音频分类器都取决于这一预处理选择。

**类型：**构建
**语言：**Python
**先决条件：**阶段6 · 01（音频基础）
**时长：**约45分钟

## 问题

取一个10秒16kHz的片段。那是160,000个浮点数，全部在`[-1, 1]`中，几乎与标签“狗吠”或“单词cat”完全不相关。原始波形包含信息，但模型难以提取的形式。两个相同的音素相隔100毫秒发音，其原始样本完全不同。

语谱图解决了这个问题。它压缩了人类感知忽略的时间细节（微秒级抖动），并保留了感知关注的结构（哪些频率有能量，在约10–25毫秒的时间窗口内）。

梅尔语谱图更进一步。人类对数地感知音高：100 Hz与200 Hz听起来与1000 Hz和2000 Hz的“距离相同”。梅尔刻度扭曲频率轴以匹配。从2010年到2026年，梅尔刻度语谱图是语音机器学习中唯一最重要的特征。

## 核心概念

![Waveform to STFT to mel spectrogram to MFCC ladder](../assets/mel-features.svg)

**STFT（短时傅里叶变换）。**将波形切分成重叠的帧（典型：25 ms窗口，10 ms步长= 400个样本/ 16 kHz下160个样本）。将每一帧乘以窗函数（默认使用汉宁窗；汉明窗略有不同的权衡）。对每一帧进行FFT。将幅度频谱堆叠成形状为`(n_frames, n_freq_bins)`的矩阵。这就是你的语谱图。

**对数幅度。**原始幅度跨越5-6个数量级。使用`log(|X| + 1e-6)`或`20 * log10(|X|)`来压缩动态范围。每一个生产流水线都使用对数幅度，而不是原始幅度。

**梅尔刻度(Mel scale)。**频率`f`（以Hz为单位）通过`m = 2595 * log10(1 + f / 700)`映射到梅尔`m`。映射在1 kHz以下大致为线性，之上大致为对数。覆盖0–8 kHz的80个梅尔区间是标准的ASR输入。

**梅尔滤波器组(Mel filterbank)。**一组在梅尔刻度上等间距的三角滤波器。每个滤波器是相邻FFT区间的加权和。将STFT幅度与滤波器组矩阵相乘，一次矩阵乘法得到梅尔语谱图。

**对数梅尔语谱图(Log-mel spectrogram)。**`log(mel_spec + 1e-10)`。Whisper的输入。Parakeet的输入。SeamlessM4T的输入。2026年的通用音频前端。

**MFCC。**取对数梅尔语谱图，应用DCT（II型），保留前13个系数。去相关特征并进一步压缩。在约2015年之前是主导特征，之后基于原始对数梅尔谱的CNN/Transformer方法迎头赶上。仍用于说话人识别（x-vectors, ECAPA）。

**分辨率权衡。**更大的FFT = 更好的频率分辨率但更差的时间分辨率。25 ms / 10 ms是音频机器学习的默认值；50 ms / 12.5 ms用于音乐；5 ms / 2 ms用于瞬态检测（鼓点、爆破音）。

```figure
spectrogram-window
```

## 动手构建

### **第一步：对波形分帧**

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

一个10秒16kHz的片段，使用`frame_len=400, hop=160`产生998帧。

### **第二步：汉宁窗**

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

在FFT之前逐元素相乘。消除因在非零端点截断而引起的频谱泄漏。

### **第三步：STFT幅度**

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

生产中使用`torch.stft`或`librosa.stft`（基于FFT，向量化）。这里的循环是教学性的；它在`code/main.py`中对短片段运行。

### **第四步：梅尔滤波器组**

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

覆盖0–8 kHz的80个梅尔区间，使用`n_fft=400`得到一个`(80, 201)`矩阵。将`(n_frames, 201)`的STFT幅度乘以转置得到`(n_frames, 80)`的梅尔语谱图。

### **第五步：对数梅尔**

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

常见替代方案：`librosa.power_to_db`（参考归一化的分贝），`10 * log10(power + eps)`。Whisper使用更复杂的裁剪+归一化程序（参见Whisper的`log_mel_spectrogram`）。

### **第六步：MFCC**

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

对每个对数梅尔帧应用DCT，保留前13个系数。这就是你的MFCC矩阵。通常丢弃第一个系数（它编码总体能量）。

## 使用它

2026年技术栈：

|  任务  |  特征  |
|------|----------|
|  ASR (Whisper, Parakeet, SeamlessM4T)  |  80个对数梅尔，10 ms步长，25 ms窗口  |
|  TTS声学模型 (VITS, F5-TTS, Kokoro)  |  80个梅尔，5–12 ms步长以实现精细的时间控制  |
|  音频分类 (AST, PANNs, BEATs)  |  128个对数梅尔，10 ms步长  |
|  说话人嵌入 (ECAPA-TDNN, WavLM)  |  80个对数梅尔或原始波形SSL  |
| 音乐（MusicGen, Stable Audio 2）||| EnCodec离散令牌（非梅尔谱） |  |
| 关键词检测(Keyword spotting) | 适用于微型设备的40个MFCC |

经验法则：**如果你不处理音乐，从80个对数梅尔(Log-mel)开始。**任何偏离都需要举证。

## 2026年仍存在的陷阱

- **梅尔数不匹配。**训练时使用80个梅尔，推理时使用128个梅尔。静默失败。在两端记录特征形状。
- **采样率不匹配（上游）。**在22.05 kHz下计算的梅尔与16 kHz下的不同。在特征提取*之前*修复采样率。
- **分贝 vs 对数。**Whisper期望的是对数梅尔(Log-mel)，而非分贝梅尔。一些Hugging Face流水线会自动检测；你的自定义代码不会。
- **归一化漂移。**训练时逐句归一化，推理时全局归一化。这是使词错误率(WER)翻倍的生产错误。
- **填充泄露。**在剪辑末尾进行零填充会导致尾部帧出现平坦频谱。应使用对称填充或重复填充。

## 发布

保存为`outputs/skill-feature-extractor.md`。该技能会为给定模型目标选择特征类型、梅尔数、帧/跳跃步长以及归一化方式。

## 练习

1. **简单。**运行`code/main.py`。它会合成一个线性调频脉冲（频率从200Hz扫到4000Hz），并打印每帧的argmax梅尔频带。可选地绘制图形并确认其与扫描匹配。
2. **中等。**在`n_mels`中使用`code/main.py`，在`frame_len`中使用`{40, 80, 128}`重新运行。测量沿时间轴的尖锐峰值带宽。哪种组合最能解析线性调频脉冲？
3. **困难。**实现`code/main.py`，并在AudioMNIST上使用一个微型CNN分类器比较ASR准确率，使用（a）原始对数梅尔(Log-mel)，（b）带`n_mels`的分贝梅尔(Db-mel)，（c）MFCC-13 + 一阶差分 + 二阶差分。报告top-1准确率。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
| 帧(Frame) | 切片 | 输入一次FFT的25毫秒波形片段。 |
| 跳跃步长(Hop) | 步幅(Stride) | 连续帧之间的样本数；10毫秒是ASR默认值。 |
| 窗口(Window) | 汉宁/汉明之类 | 将帧边缘渐变为零的逐点乘数。 |
| 短时傅里叶变换(STFT) | 语谱图生成器 | 分帧加窗FFT；生成时间×频率矩阵。 |
| 梅尔(Mel) | 扭曲频率 | 对数感知尺度；`m = 2595·log10(1 + f/700)`。 |
| 滤波器组(Filterbank) | 矩阵 | 将STFT投影到梅尔频带的三角形滤波器。 |
| 对数梅尔(Log-mel) | Whisper的输入 | `log(mel_spec + eps)`；2026年标准化。 |
| 梅尔频率倒谱系数(MFCC) | 传统特征 | 对数梅尔的离散余弦变换(DCT)；13个系数，去相关。 |

## 延伸阅读

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — MFCC论文。
- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — 原始梅尔尺度。
- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — 阅读参考实现。
- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — 关于[Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/)、[OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py)以及跳跃步长/窗口的参考。
- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — 用于Parakeet + Canary模型的生产级流水线。
