# 语音活动检测与轮次管理——Silero、Cobra 与刷新技巧

> 每个语音智能体成败取决于两个判断：用户当前是否在说话？他们是否说完了？VAD 回答第一个问题。轮次检测（VAD + 静音拖尾 + 语义端点模型）回答第二个。任何一个出错，你的助手要么打断用户，要么永远不闭嘴。

**类型：** 构建
**语言：** Python
**先决条件：** 阶段 6 · 11（实时音频）、阶段 6 · 12（语音助手）
**时间：** ~45 分钟

## 问题

语音智能体在每个 20 毫秒块上做出三个不同的决策：

1. **这个帧是语音吗？** —— VAD。二值，逐帧。
2. **用户是否开始了新的发言？** —— 起始检测。
3. **用户是否结束了？** —— 端点检测（轮次结束）。

朴素方法（能量阈值）在任何噪声中都会失败——交通、键盘、人群嘈杂。2026 年的方案：Silero VAD（开源的、深度学习的）+ 轮次检测模型（语义端点）+ VAD 校准的静音拖尾。

## 核心概念

![VAD cascade: energy → Silero → turn-detector → flush trick](../assets/vad-turn-taking.svg)

### 三层 VAD 级联

**第一层：能量门。** 最便宜。阈值 RMS 设为 -40 dBFS。过滤明显静音，但会对任何高于阈值的噪声触发。

**第二层：Silero VAD**（2020-2026，MIT）。100 万个参数。在 6000 多种语言上训练。在单 CPU 线程上，每 30 毫秒块运行约 1 毫秒。在 5% 假阳性率下真阳性率 87.7%。开源默认选择。

**第三层：语义轮次检测器。** LiveKit 的轮次检测模型（2024-2026）或你自己的小型分类器。区分“句中停顿”与“说话结束”。使用语言上下文（语调 + 最近词语），而不仅仅是静音。

### 关键参数及其默认值

- **阈值。** Silero 输出概率；分类语音为大于 0.5（默认）或大于 0.3（灵敏）。更低阈值 = 更少首词截断，更多误报。
- **最短语音时长。** 拒绝短于 250 毫秒的语音——通常是咳嗽或椅子噪声。
- **静音拖尾（端点检测）。** VAD 返回 0 后，等待 500-800 毫秒再宣布轮次结束。太短 → 打断用户。太长 → 感觉迟缓。
- **预卷缓冲。** 在 VAD 触发前保留 300-500 毫秒的音频。防止“hey”被截断。

### 刷新技巧（Kyutai，2025）

流式 STT 模型有一个超前延迟（Kyutai STT-1B 为 500 毫秒，STT-2.6B 为 2.5 秒）。通常你要在语音结束后等待这么长时间才能得到转录文本。刷新技巧：当 VAD 触发语音结束时，**向 STT 发送一个刷新信号**，强制立即输出。STT 以约 4 倍实时速度处理，因此 500 毫秒的缓冲在约 125 毫秒内完成。

端到端：125 毫秒 VAD + 刷新 STT = 对话延迟。

### 2026 年 VAD 对比

|  VAD  |  5% 假阳性率下的真阳性率  |  延迟  |  许可证  |
|-----|--------------|---------|---------|
|  WebRTC VAD（谷歌，2013）  |  50.0%  |  30 毫秒  |  BSD  |
|  Silero VAD（2020-2026）  |  87.7%  |  ~1 毫秒  |  MIT  |
|  Cobra VAD（Picovoice）  |  98.9%  |  ~1 毫秒  |  商业  |
|  pyannote 分割  |  95%  |  ~10 毫秒  |  类似 MIT  |

Silero 是正确的默认选择。Cobra 是合规性/准确性的升级。仅能量 VAD 在 2026 年的生产中已无立足之地。

## 动手构建

### 步骤 1：能量门

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### 步骤 2：Python 中的 Silero VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### 步骤 3：轮次结束状态机

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### 步骤 4：刷新技巧骨架

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT（Kyutai、Deepgram、AssemblyAI）必须支持刷新才能使其工作。Whisper 流式不支持——它基于块，总是等待块。

## 使用它

|  情况  |  VAD 选择  |
|-----------|-----------|
|  开源、快速、通用  |  Silero VAD  |
|  商业呼叫中心  |  Cobra VAD  |
| 设备端（手机）||| Silero VAD ONNX |  |
| 研究/说话人分离 (diarization) | pyannote segmentation |
| 零依赖回退 | WebRTC VAD (传统) |
| 需要高质量回合结束检测 | Silero + LiveKit 回合检测器分层 |

经验法则：除非别无选择，否则永远不要仅依赖能量型 VAD。

## 陷阱

- **固定阈值。** 安静环境下有效，嘈杂环境下失效。要么在设备端校准，要么切换为 Silero。
- **静音挂起时间过短。** 代理会打断说话人的句子。500-800 毫秒是对话语音的最佳区间。
- **挂起时间过长。** 反应迟钝。与目标用户进行 A/B 测试。
- **无预卷缓冲。** 用户语音的前 200-300 毫秒会丢失。务必保留滚动预卷。
- **忽略语义端点检测。** “嗯，让我想想……” 包含长时间停顿。用户讨厌在思考时被打断。使用 LiveKit 的回合检测器或类似方案。

## 发布

保存为 `outputs/skill-vad-tuner.md`。为工作负载选择 VAD 模型、阈值、挂起时间、预卷和回合检测策略。

## 练习

1. **简单。** 运行 `code/main.py`。它会模拟一段“语音+静音+语音+咳嗽”的序列，并测试三个 VAD 层级。
2. **中等。** 安装 `code/main.py`，处理一段 5 分钟录音，调整阈值以最小化首词截断和误触发。报告精确率/召回率。
3. **困难。** 构建一个小型回合检测器：Silero VAD + 一个基于最后 10 个词嵌入（使用 sentence-transformers）的 3 层 MLP。在人工标注的回合结束数据集上训练，F1 分数比纯 Silero 提升 10%。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  VAD  |  语音检测器  |  逐帧二分类：这是语音吗？ |
|  回合检测  |  端点检测  |  VAD + 静音挂起 + 语义端点检测 |
|  静音挂起  |  语音后等待  |  声明回合结束前的等待时间；500-800 毫秒 |
|  预卷  |  语音前缓冲  |  在 VAD 触发前保留 300-500 毫秒音频 |
|  刷新技术  |  Kyutai 技巧  |  VAD → 刷写 STT → 125 毫秒延迟，而非 500 毫秒 |
|  语义端点  |  “他们打算停吗？”  |  基于词汇而非仅静音的机器学习分类器 |
|  TPR @ FPR 5%  |  ROC 点  |  标准 VAD 基准；Silero 87.7%，WebRTC 50% |

## 延伸阅读

- [Silero VAD](https://github.com/snakers4/silero-vad)——参考开源 VAD。
- [Silero VAD](https://github.com/snakers4/silero-vad)——商业精度领先者。
- [Silero VAD](https://github.com/snakers4/silero-vad)——亚 200 毫秒工程技巧。
- [Silero VAD](https://github.com/snakers4/silero-vad)——生产环境中的语义端点检测。
- [Silero VAD](https://github.com/snakers4/silero-vad)——传统基线。
- [Silero VAD](https://github.com/snakers4/silero-vad)——说话人分离级别的分割。
