## §18 语音 Agent：不是文本循环套个 TTS

### 18.1 一句判断：语音是独立品类

2026 年，语音 Agent 是production 一等公民。

它不是"文本循环末尾加个语音合成"。

延迟预算极狠：端到端要到 450–600ms，用户才不觉得卡。

Vapi 在优化栈上跑到 450–600ms。Retell 跨 180 通测试约 600ms。

超过 1500ms，用户就觉得"这东西坏了"。

> **标叔的经验**：第一次被延迟教做人
>
> 我做过一个客服语音，STT + LLM + TTS 一串，端到端 1.8 秒。用户平均每句说完等两秒。上线三天，挂断率 40%。砍到 700ms 后降到 12%。

### 18.2 Pipecat：帧管线

Pipecat（pipecat-ai/pipecat）是 Python 帧框架。

核心是 `Frame` → `FrameProcessor` 链。

两个方向：

- **DOWNSTREAM**：源 → 汇。音频进，语音出。
- **UPSTREAM**：反馈与控制。取消、指标、打断。

典型五段：

```text
VAD (Silero) → STT → LLM → TTS → transport
```

VAD 判"人在说话"。STT 转文字。LLM 想。TTS 念。transport 传出去。

transport 支持 Daily、LiveKit、WebSocket、WhatsApp。

`PipelineTask` 管生命周期：`on_pipeline_started`、`on_idle_timeout` 等事件可挂观察者做指标与追踪。

### 18.3 LiveKit：WebRTC 优先

LiveKit Agents（livekit/agents）把模型经 WebRTC 桥给用户。

两个语音类：

- **MultimodalAgent**：直接音频进音频出（OpenAI Realtime 那种）。
- **VoicePipelineAgent**：STT → LLM → TTS 级联，文字级可控。

它有语义性"话轮检测"（transformer 模型判你讲完没）。

原生接 MCP，电话走 SIP。50+ 模型免密钥，200+ 走插件。

| 维度 | Pipecat | LiveKit | 标叔的结论 |
|------|---------|---------|-----------|
| 定位 | Python 帧框架 | WebRTC 平台 | 要控制选前者 |
| 音频路径 | 级联为主 | 可直连音频 | 直连更省延迟 |
| 可控性 | 自定义 processor | 类封装好 | 深度定制选 Pipecat |
| 电话 | 经 transport | 原生 SIP | 打电话选 LiveKit |
| 上手 | 中 | 快 | 求快上 LiveKit |

### 18.4 延迟账：每一段都吃预算

每一段都加 50–200ms。上线前先加一遍。

- VAD：20–60ms
- STT 部分结果：100–250ms
- LLM 首个 token：150–400ms
- TTS 首个音频：100–200ms
- transport 往返：30–80ms

高端栈 450–600ms。普通栈 800–1200ms。

> **注意**：打断（barge-in）不做会出丑
>
> 用户中途插话，Agent 还在念。Pipecat 用 UPSTREAM 取消帧停掉 TTS。LiveKit 同理。不做，体验直接崩。

### 18.5 手写一个帧管线

不接真模型，用脚本处理器演示流向与打断：

```python
# 帧管线：VAD → STT → LLM → TTS → 传输
class Pipeline:
    def __init__(self):
        self.stages = []          # 处理器链

    def add(self, proc):
        self.stages.append(proc)

    def push(self, frame):
        for stage in self.stages:
            frame = stage.process(frame)   # 逐段流转
            if frame is None:
                return                      # 取消帧到此截断
        return frame

class CancelFrame: pass                    # UPSTREAM 打断帧

class TTS:
    def process(self, frame):
        if isinstance(frame, CancelFrame):
            return None                     # 被打断，停止念
        return frame
```

跑一遍：正常流顺下来；插一个 `CancelFrame`，TTS 中途停。

### 18.6 先给结论

语音 Agent 拼的是延迟，不是模型。

先画延迟账，再选 Pipecat 还是 LiveKit。打断必须做。

[向前桥接] 会说话了。但真上线，钱和稳定性才是生死线。
