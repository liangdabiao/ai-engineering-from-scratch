# 实时音频处理

> 批处理管道处理一个文件。实时管道处理接下来的20毫秒，在下一个20毫秒到来之前完成。每个对话式AI、广播演播室和电话机器人的生死都取决于这个延迟预算。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段6·02（频谱图），阶段6·04（自动语音识别），阶段6·07（文本转语音）
**时间：** 约75分钟

## 问题

您希望一个语音助手感觉真实。人类对话的轮流延迟大约是230毫秒（从静默到响应）。超过500毫秒会感觉像机器人；超过1500毫秒会感觉有问题。2026年一个完整的**听→理解→响应→说**循环的预算是：

|  阶段  |  预算  |
|-------|--------|
|  麦克风→缓冲区  |  20毫秒  |
|  语音活动检测  |  10毫秒  |
|  自动语音识别（流式）  |  150毫秒  |
|  大语言模型（第一个令牌）  |  100毫秒  |
|  文本转语音（第一个数据块）  |  100毫秒  |
|  渲染→扬声器  |  20毫秒  |
|  **总计**  |  **约400毫秒**  |

Moshi（Kyutai，2024年）实现了200毫秒的全双工。GPT-4o-realtime（2024年）约为320毫秒。2022年的级联管道延迟为2500毫秒。10倍的改进来自于三种技术：（1）全程流式，（2）带有部分结果的异步流水线，（3）可中断的生成。

## 核心概念

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**帧/数据块/窗口。** 实时音频以固定大小的块流动。常见选择：20毫秒（16 kHz下320个样本）。下游所有内容必须跟上这个节奏。

**环形缓冲区。** 固定大小的循环缓冲区。生产者线程写入新帧，消费者线程读取。避免在热路径中分配内存。大小≈最大延迟×采样率；一个2秒16 kHz的环形缓冲区=32,000个样本。

**语音活动检测。** 在无人说话时关闭下游工作。Silero VAD 4.0（2024年）在CPU上每个30毫秒帧运行时间小于1毫秒。`webrtcvad` 是较老的替代方案。

**流式自动语音识别。** 在音频到达时输出部分转录文本的模型。Parakeet-CTC-0.6B在流式模式下（NeMo，2024年）在320毫秒延迟下实现2-5%的词错误率。Whisper-Streaming（Macháček等，2023年）将Whisper分块以实现接近流式的约2秒延迟。

**中断。** 当用户在助手说话时说话，您必须（a）检测打断，（b）停止文本转语音，（c）丢弃剩余的大语言模型输出。所有这些需要在100毫秒内完成，否则用户会感觉助手失聪。

**WebRTC Opus传输。** 20毫秒帧，48 kHz，自适应比特率8-128 kbps。浏览器和移动设备的标准。LiveKit、Daily.co、Pion是2026年构建语音应用的栈。

**抖动缓冲区。** 网络数据包可能乱序/延迟到达。抖动缓冲区重新排序并平滑；太小→可听间隙，太大→延迟。典型值为60-80毫秒。

### 常见陷阱

- **线程竞争。** Python的GIL加上重型模型可能会饿死音频线程。使用C回调音频库（sounddevice、PortAudio）并让Python远离热路径。
- **采样率转换延迟。** 管道内部的重新采样会增加5-20毫秒。要么提前重新采样，要么使用零延迟重采样器（PolyPhase、`soxr_hq`）。
- **文本转语音预热。** 即使是像Kokoro这样的快速文本转语音，首次请求也有100-200毫秒的预热。在第一次真实轮次之前缓存模型并用虚拟运行预热。
- **回声消除。** 没有回声消除，文本转语音输出会重新进入麦克风并触发自动语音识别识别机器人自己的声音。WebRTC AEC3是开源默认选项。

```figure
nyquist-aliasing
```

## 动手构建

### 步骤1：环形缓冲区

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

容量决定最大缓冲延迟。16 kHz下32,000个样本=2秒。

### 步骤2：语音活动检测门控

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

在生产环境中替换为Silero VAD：

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### 步骤3：流式自动语音识别

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### 步骤4：中断处理器

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

依赖于异步I/O和可取消的文本转语音流。WebRTC peerconnection.stop()在音频轨道上是标准方法。

## 使用它

2026年技术栈：

|  层  |  选择  |
|-------|------|
|  传输  |  LiveKit (WebRTC) 或 Pion (Go)  |
|  语音活动检测(VAD)  |  Silero VAD 4.0  |
|  流式语音识别(ASR)  |  Parakeet-CTC-0.6B 或 Whisper-Streaming  |
|  大语言模型(LLM)首字延迟  |  Groq, Cerebras, vLLM-streaming  |
|  流式语音合成(TTS)  |  Kokoro 或 ElevenLabs Turbo v2.5  |
|  回声消除  |  WebRTC AEC3  |
|  端到端原生  |  OpenAI Realtime API 或 Moshi  |

## 陷阱

- **缓冲 500 毫秒以确保安全。** 缓冲区 *就是* 你的延迟下限。缩减它。
- **未固定线程。** 音频回调放在优先级低于用户界面(UI)的线程上 = 负载下出现卡顿。
- **语音合成(TTS)分块过小。** 低于 200 毫秒的分块会使声码器伪影变得可闻。320 毫秒的分块是最佳点。
- **无抖动缓冲。** 真实网络是有抖动的；没有平滑处理就会出现爆音。
- **单次错误处理。** 音频流水线必须是崩溃安全的。一个异常就会终止会话。

## 发布

保存为 `outputs/skill-realtime-designer.md`。设计一个实时音频流水线，为每个阶段指定具体的延迟预算。

## 练习

1. **简单。** 运行 `code/main.py`。模拟一个环形缓冲区 + 能量语音活动检测(VAD)；打印模拟 10 秒流的各阶段延迟。
2. **中等。** 使用 `code/main.py`，构建一个直通循环，以 20 毫秒帧处理麦克风输入，并在每帧打印语音活动检测(VAD)状态。
3. **困难。** 使用 `code/main.py` 构建全双工回声测试：浏览器 → WebRTC → Python → WebRTC → 浏览器。使用 1 kHz 脉冲测量端到端延迟。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  环形缓冲区  |  循环队列  |  固定大小、无锁（或单生产者单消费者(SPSC)锁定的）先进先出(FIFO)，用于音频帧。  |
|  语音活动检测(VAD)  |  静音门控  |  使用模型或启发式方法标记语音与非语音。  |
|  流式语音识别(ASR)  |  实时语音转文字(STT)  |  音频到达时输出部分文本；有限的前瞻。  |
|  抖动缓冲  |  网络平滑器  |  对乱序数据包进行重排序的队列；通常 60–80 毫秒。  |
|  回声消除(AEC)  |  回声消除  |  减去扬声器到麦克风的反馈路径。  |
|  打断  |  用户中断  |  系统在语音合成(TTS)播放期间检测到用户语音；必须取消播放。  |
|  全双工  |  同时双向通信  |  用户和机器人可以同时说话；Moshi 是全双工的。  |

## 延伸阅读

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — 分块近流式 Whisper。
- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — 全双工 200 毫秒延迟。
- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — 生产级音频代理编排。
- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — 亚毫秒级语音活动检测(VAD)，Apache 2.0。
- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — 开源回声消除。
