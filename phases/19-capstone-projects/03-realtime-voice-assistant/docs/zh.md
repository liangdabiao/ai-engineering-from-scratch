# 顶点项目 03 — 实时语音助手（ASR 到 LLM 到 TTS）

> 一个感觉合适的语音代理(agent)需要端到端延迟在800ms以下，知道用户何时停止说话，支持打断(barge-in)，并且能够调用工具而不卡顿。Retell、Vapi、LiveKit Agents和Pipecat都在2026年达到了这个标准。它们采用相同的架构：流式ASR、轮流检测器(turn-detector)、流式LLM和流式TTS，所有组件通过WebRTC连接，每一跳都严格控制延迟预算。构建这样一个系统，测量词错误率(WER)、平均意见分(MOS)和错误中断率(false-cutoff rate)，并在丢包条件下运行测试。

**类型：** 结业项目
**语言：** Python（代理+流水线）、TypeScript（Web客户端）
**前置要求：** 阶段6（语音和音频）、阶段7（transformers）、阶段11（LLM工程）、阶段13（工具）、阶段14（代理）、阶段17（基础设施）
**涉及的阶段：** P6 · P7 · P11 · P13 · P14 · P17
**所需时间：** 30小时

## 问题

语音是2025-2026年人工智能用户界面中发展最快的类别。技术天花板每个季度都在降低。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0和Pipecat 0.0.70都使得首次音频输出(sub-800ms first-audio-out)成为可能。标准不仅仅是延迟，更是交互感受：不打断用户、不被用户打断、从中句中断中恢复、在对话中途调用工具而不卡顿音频、在抖动的移动网络下保持稳定。

你无法通过拼凑三个REST调用达到目标。架构必须是端到端的流式流水线。构建它后，失败模式就会显现：针对电话音频调优的VAD在背景电视噪音下误触发，等待标点符号但永远不会出现的轮流检测器，以及缓冲400ms后才输出的TTS。结业项目就是在负载下逐一修复这些问题，并发布延迟和质量报告。

## 概念

流水线包含五个流式阶段：**音频输入**（来自浏览器或PSTN的WebRTC）、**ASR**（来自Deepgram Nova-3或faster-whisper的流式部分转录）、**轮流检测**（VAD加上一个读取部分转录以寻找完成线索的小型轮流检测模型）、**LLM**（一旦判断轮流结束，立即流式输出tokens）、**TTS**（在第一个LLM token之后约200ms内流式输出音频）。

三个横切关注点。**打断**：当用户在代理说话时开始说话，TTS取消并立即启用ASR。**工具使用**：对话中途的函数调用（天气、日历）必须在侧通道上运行，而不卡顿音频；如果延迟超过300ms，代理预填充一个确认token（"稍等..."）。**背压**：在丢包情况下，部分转录被暂存，VAD提高语音门限阈值，代理避免在未确认的消息上说话。

测量标准是定量的。在15 dB信噪比(SNR)的Hamming VAD基准测试上，WER低于8%。在100次测量通话中，首次音频输出的p50低于800ms。错误中断率低于3%。TTS的MOS高于4.2。单个g5.xlarge实例上50路并发通话。这些数字是交付物。

## 架构

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (or Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      per 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## 技术栈

- 传输：LiveKit Agents 1.0（WebRTC）加上Twilio PSTN网关；Pipecat 0.0.70作为备选框架
- ASR：Deepgram Nova-3（流式，首次部分结果低于300ms）或自托管的faster-whisper Whisper-v3-turbo
- VAD：Silero VAD v5加上LiveKit轮流检测器（读取部分转录的小型transformer）
- LLM：OpenAI GPT-4o-realtime（紧密集成）、Gemini 2.5 Flash Live，或级联的Claude Haiku 4.5（流式补全，独立音频路径）
- TTS：Cartesia Sonic-2（最低首字节延迟）、ElevenLabs Flash v3，或自托管的开源Orpheus
- 工具：FastMCP侧通道用于天气/日历/预订；如果工具耗时超过300ms，代理预发出填充语
- 可观测性：OpenTelemetry语音跨度(voice spans)、Langfuse语音追踪带音频回放
- 部署：单个g5.xlarge（24GB VRAM）用于自托管的Whisper + Orpheus；托管API以获得最低延迟

## 动手构建

1. **WebRTC会话。** 创建一个LiveKit房间和一个流传输麦克风音频的Web客户端。在服务器上，附加一个加入房间的代理worker。

2. **ASR流式处理。** 将20ms PCM帧送入Deepgram Nova-3（或GPU上的faster-whisper）。订阅部分和最终转录结果。记录每个部分结果的延迟。

3. **VAD和轮流检测器。** 在帧流上运行Silero VAD v5。在语音结束事件发生时，将最新部分转录送入LiveKit轮流检测器。仅当VAD检测到500ms静音且轮流检测器完成得分>0.6时，才确认"轮流结束"。

4. **LLM流式输出。** 在轮流结束后，使用当前对话加上最终转录启动LLM调用。流式输出tokens。在第一个token处，交给TTS。

5. **TTS流式输出。** Cartesia Sonic-2流式返回音频块。第一个音频块必须在第一个LLM token后的200ms内离开服务器。将音频块发送到LiveKit房间；客户端通过WebRTC抖动缓冲区播放。

6. **打断。** 当VAD在TTS播放时检测到新的用户语音，立即取消TTS流，丢弃剩余的LLM输出，并重新启用ASR。发布一个`tts_canceled`跨度(span)。

7. **工具侧通道。** 将天气和日历注册为函数调用工具。当被调用时，并发执行调用；如果在300ms内未解析，让LLM发出"稍等，让我查一下"作为填充语；一旦工具返回则继续。

8. **评估框架。** 记录100次通话。计算WER（与保留转录对比）、错误中断率（用户说话中途TTS被取消）、首次音频输出的p50、TTS MOS（人工或NISQA），以及抖动丢包测试（丢弃3%的数据包）。

9. **负载测试。** 在单个g5.xlarge上使用模拟呼叫器驱动50路并发通话。测量持续首次音频输出的p95。

## 使用它

```
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## 发布

`outputs/skill-voice-agent.md`是交付物。给定一个领域（客户支持、调度或自助终端），它构建一个LiveKit代理，其ASR/VAD/LLM/TTS流水线已针对测量标准进行调优。评分标准：

|  权重  |  标准  |  衡量方式  |
|:-:|---|---|
|  25  |  端到端延迟  |  在100次记录的通话中，首次音频输出的p50低于800ms  |
|  20  |  轮流交互质量  |  在Hamming VAD基准测试上错误中断率低于3%  |
|  20  |  工具使用正确性  |  对话中途的工具调用返回正确数据且不卡顿音频  |
|  20  |  丢包条件下的可靠性  |  注入3%丢包后WER和轮流交互的稳定性  |
|  15  |  评估框架完整性  |  可复现的测量，附带公开配置  |
|  **100**  |   |   |

## 练习

1. 在g5.xlarge上将Deepgram Nova-3替换为faster-whisper v3 turbo。测量延迟和WER的差距。识别CPU与GPU决策的关键点。

2. 添加中断仲裁策略：当用户在工具调用期间打断时，代理应做什么？比较三种策略（硬取消、完成工具后停止、排队下一个轮流）。

3. 运行对抗性轮流检测器测试：让用户在句中长时间停顿。调整VAD静音阈值和轮流检测器得分阈值，以在不超过900ms的前提下实现最低错误中断。

4. 通过Twilio在PSTN上部署相同的代理。比较PSTN首次音频输出与WebRTC。解释抖动缓冲区(jitter-buffer)和编解码器(codec)的差异。

5. 为非英语语言（日语、西班牙语）添加语音活动检测。测量Silero VAD v5的误触发率与针对特定语言的微调版本对比。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
|  轮流检测  |  "话语结束"  |  一个分类器，根据VAD静音和部分转录判断用户是否说完  |
|  打断（Barge-in）  |  "中断处理"  |  当VAD检测到用户新语音时，取消TTS播放中的内容 |
|  首次音频输出（First-audio-out）  |  "延迟"  |  从用户停止说话到第一个音频包离开服务器的时间 |
|  语音活动检测（VAD）  |  "语音门控"  |  将音频帧分类为语音或静音的模型；Silero VAD v5是2026年默认设置 |
|  抖动缓冲（Jitter buffer）  |  "音频平滑"  |  客户端侧的缓冲，短暂保存数据包以吸收网络波动 |
|  填充词（Filler）  |  "确认令牌"  |  当工具响应缓慢时，智能体发出的简短短语以避免静音 |
|  平均意见得分（MOS）  |  "平均意见得分"  |  感知语音质量评分；NISQA是其自动化代理 |

## 延伸阅读

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — 参考WebRTC智能体框架
- [LiveKit Agents 1.0](https://github.com/livekit/agents) — 备选的Python优先流式智能体框架
- [LiveKit Agents 1.0](https://github.com/livekit/agents) — 集成语音模型的参考
- [LiveKit Agents 1.0](https://github.com/livekit/agents) — 流式ASR参考
- [LiveKit Agents 1.0](https://github.com/livekit/agents) — 语音活动检测参考模型
- [LiveKit Agents 1.0](https://github.com/livekit/agents) — 低延迟TTS参考
- [LiveKit Agents 1.0](https://github.com/livekit/agents) — 生产级语音智能体架构
- [LiveKit Agents 1.0](https://github.com/livekit/agents) — 备选生产参考
