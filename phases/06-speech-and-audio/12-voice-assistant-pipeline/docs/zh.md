# 构建语音助手流水线 —— 第 6 阶段毕业设计

> 整合第 01-11 课的所有内容。构建一个能听、能思考、能回应的语音助手。在 2026 年，这是一个已经解决的工程问题，而非研究问题——但集成细节决定了它能否真正交付。

**类型：** 构建
**语言：** Python
**前置知识：** 第 6 阶段 · 04, 05, 06, 07, 11；第 11 阶段 · 09（函数调用）；第 14 阶段 · 01（智能体循环）
**预计用时：** ~120 分钟

## 问题

构建一个端到端助手：

1. 捕获麦克风输入（16 kHz 单声道）。
2. 检测用户语音的开始/结束。
3. 流式转录。
4. 将转录文本传给可调用工具（计时器、天气、日历）的 LLM。
5. 将 LLM 文本流式传给 TTS。
6. 将音频播放给用户。
7. 若用户在回应中途打断，则停止播放。

延迟目标：在笔记本电脑 CPU 上，用户说完话后 800 ms 内输出首个 TTS 音频字节。质量目标：无漏词，无声时无幻觉字幕，无声音克隆泄漏，无提示注入成功。

## 核心概念

![Voice assistant pipeline: mic → VAD → STT → LLM+tools → TTS → speaker](../assets/voice-assistant.svg)

### 七个组件

1. **音频捕获。** 麦克风 → 16 kHz 单声道 → 20 ms 数据块。通常用 Python 的 `sounddevice`，生产环境用原生 AudioUnit/ALSA/WASAPI。
2. **VAD（第 11 课）。** Silero VAD，阈值 0.5，最小语音时长 250 ms，静默挂起 500 ms。输出“开始”和“结束”信号。
3. **流式 STT（第 4-5 课）。** Whisper-streaming、Parakeet-TDT 或 Deepgram Nova-3（API）。输出部分和最终转录文本。
4. **带工具调用的 LLM。** GPT-4o / Claude 3.5 / Gemini 2.5 Flash。工具使用 JSON Schema。流式输出 token。
5. **流式 TTS（第 7 课）。** Kokoro-82M（最快开源）或 Cartesia Sonic（商业）。在 LLM 输出 20 个 token 后开始 TTS。
6. **播放。** 扬声器输出；低带宽网络使用 opus 编码。
7. **打断处理器。** 若在 TTS 播放期间 VAD 触发，则停止播放，取消 LLM，重启 STT。

### 会遇到的三种失败模式

1. **首个词被切掉。** VAD 启动慢了一拍。用户开头的“嘿”丢失。将启动阈值设为 0.3 而非 0.5。
2. **回应中途打断混乱。** 用户打断后 LLM 仍在生成；助手与用户同时说话。将 VAD 连接到取消 LLM 的逻辑。
3. **静默幻觉。** Whisper 在静默预热帧上输出“Thanks for watching”。务必使用 VAD 门控。

### 2026 年生产参考技术栈

|  技术栈  |  延迟  |  许可证  |  备注  |
|-------|---------|---------|-------|
|  LiveKit + Deepgram + GPT-4o + Cartesia  |  350-500 ms  |  商业 API  |  2026 年行业默认方案  |
|  Pipecat + Whisper-streaming + GPT-4o + Kokoro  |  500-800 ms  |  主要开源自建  |  适合 DIY  |
|  Moshi（全双工）  |  200-300 ms  |  CC-BY 4.0  |  单一模型；不同架构，见第 15 课  |
|  Vapi / Retell（托管服务）  |  300-500 ms  |  商业  |  最快上线；定制有限  |
|  Whisper.cpp + llama.cpp + Kokoro-ONNX  |  离线  |  开源  |  隐私/边缘设备  |

## 动手构建

### 步骤 1：带分块的麦克风捕获（伪代码）

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### 步骤 2：VAD 门控的语音段捕获

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### 步骤 3：流式 STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### 步骤 4：LLM 循环内的工具调用

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### 步骤 5：打断处理

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## 使用它

参见 `code/main.py`，其中有一个可运行的模拟，连接了所有七个组件并使用存根模型，因此即使没有硬件也能看到流水线形状。对于实际实现，将存根替换为：

- `silero-vad` (`pip install silero-vad`)
- `silero-vad` 或 `pip install silero-vad`
- `silero-vad` (`pip install silero-vad`) 或 `deepgram-sdk`
- `silero-vad` 或 `pip install silero-vad`
- `silero-vad` 用于 I/O

## 陷阱

- **永久记录 PII。** 完整音频在大多数司法管辖区属于个人身份信息。保留 30 天，静态加密。
- **没有打断功能。** 用户会打断。你的助手必须能停止说话。
- **阻塞的 TTS。** 同步 TTS 会阻塞事件循环。使用异步或独立线程。
- **无工具调用错误处理。** 工具会失败。LLM 必须收到错误并重试一次，然后优雅降级。
- **过度幻觉过滤。** 过滤过头，助手会重复“我无法帮助这个”。过滤不足则什么都说。在保留集上校准。
- **没有唤醒词选项。** 始终监听是隐私负担。添加唤醒词门控（Porcupine 或 openWakeWord）。

## 发布

保存为 `outputs/skill-voice-assistant-architect.md`。根据预算、规模、语言和合规性约束，给出完整技术栈规范。

## 练习

1. **简单版。** 运行 `code/main.py`。它用存根模块模拟一个完整的端到端交互轮次，并打印每阶段延迟。
2. **中等版。** 在预录的 `code/main.py` 上使用真实 Whisper 模型替换 STT 存根。测量词错误率 (WER) 和端到端延迟。
3. **困难版。** 添加工具调用：实现 `code/main.py`（任意 API）和 `.wav`。让 LLM 通过工具路由，验证当用户说“设置一个 5 分钟计时器”时，正确的函数被触发并且语音回复确认了该操作。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  轮次  |  用户与助手的一次往返  |  一次 VAD 限定的用户语音 + 一次 LLM-TTS 响应。  |
|  打断  |  中断  |  助手说话时用户说话；助手停止。  |
|  唤醒词  |  “嘿，助手”  |  短关键词检测器；Porcupine、Snowboy、openWakeWord。  |
| 结束检测  |  回合结束  |  VAD + 最小静默决策以确定用户已说完。 |
| 预卷  |  语音前缓冲区  |  保留VAD触发前200-400毫秒的音频以避免第一个字被截断。 |
| 工具调用  |  函数调用  |  LLM发出JSON；运行时调度；结果回馈到循环中。 |

## 延伸阅读

- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) — 生产级参考。
- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) — 适合DIY的框架。
- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) — 托管语音原生路径。
- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) — 全双工参考（第15课）。
- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) — 唤醒词门控。
- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) — LLM函数调用。
