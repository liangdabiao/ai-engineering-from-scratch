# 语音代理：Pipecat 和 LiveKit

> 语音代理是2026年的一级生产类别。Pipecat 提供基于 Python 的帧式流水线（VAD → STT → LLM → TTS → 传输层）。LiveKit Agents 通过 WebRTC 将 AI 模型桥接到用户。对于高级堆栈，生产延迟目标为端到端 450–600ms。

**类型：** 学习
**语言：** Python (标准库)
**前置知识：** 阶段14 · 01 (代理循环)，阶段14 · 12 (工作流模式)
**时间：** ~60 分钟

## 学习目标

- 描述 Pipecat 基于帧的流水线：DOWNSTREAM（源→汇）和 UPSTREAM（控制）。
- 列举规范的语音流水线阶段以及 Pipecat 支持的传输层。
- 解释 LiveKit Agents 的两类语音代理（MultimodalAgent、VoicePipelineAgent）以及各自适用场景。
- 总结2026年生产延迟预期及其如何驱动架构选择。

## 问题

语音代理不是带有 TTS 的文本循环。延迟预算很苛刻（~600ms），部分音频是默认情况，话轮检测是一个模型，传输层从电信 SIP 到 WebRTC。要么构建基于帧的流水线（Pipecat），要么依赖平台（LiveKit）。

## 核心概念

### Pipecat（pipecat-ai/pipecat）

- 基于 Python 的帧式流水线框架。
- `Frame` → `FrameProcessor` 链。
- 两个流向：
  - **DOWNSTREAM** — 源 → 汇（音频输入，TTS输出）。
  - **UPSTREAM** — 反馈和控制（取消、指标、打断）。
- `Frame` 通过事件（`FrameProcessor`、`PipelineTask`、`on_pipeline_started`）和用于指标/追踪/RTVI 的观察器管理生命周期。

典型流水线：

```
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

传输层：Daily、LiveKit、SmallWebRTCTransport、FastAPI WebSocket、WhatsApp。

Pipecat Flows 增加了结构化对话（状态机）。Pipecat Cloud 是托管运行时。

### LiveKit Agents（livekit/agents）

- 通过 WebRTC 将 AI 模型桥接到用户。
- 关键概念：`Agent`、`AgentSession`、`entrypoint`、`AgentServer`。
- 两类语音代理：
  - **MultimodalAgent** — 通过 OpenAI Realtime 或等同的直接音频。
  - **VoicePipelineAgent** — STT → LLM → TTS 级联；提供文本级控制。
- 通过 Transformer 模型进行语义话轮检测。
- 原生 MCP 集成。
- 通过 SIP 实现电话。
- 通过 LiveKit Inference 提供 50+ 模型无需 API 密钥；通过插件提供 200+ 更多模型。

### 商业平台

Vapi（优化高级堆栈上约 450–600ms）和 Retell（180 次测试通话中端到端约 600ms）基于这些构建。当需要托管语音堆栈但没有 WebRTC 团队时，选择平台。

### 这种模式出错的地方

- **无打断处理。** 用户打断时，代理继续说话。需要 Pipecat 中的 UPSTREAM 取消帧，LiveKit 中类似。
- **忽略 STT 置信度。** 低置信度转录内容被当作真理输入 LLM。需根据置信度设置门限或请求确认。
- **TTS 中间句子截断。** 当流水线在话语中途取消时，TTS 需要获知或截断音频。
- **忽略延迟预算。** 每个组件增加 50–200ms。在发布之前需累加链。

### 典型的2026年延迟

- VAD：20–60ms
- STT 部分：100–250ms
- LLM 第一个令牌：150–400ms
- TTS 第一个音频：100–200ms
- 传输 RTT：30–80ms

端到端 450–600ms 为高级。800–1200ms 较常见。超过 1500ms 则感觉卡顿。

## 动手构建

`code/main.py` 是一个基于帧的玩具流水线，包含：

- `Frame` 类型（音频、转录、文本、tts_音频、控制）。
- 带有 `Processor` 的 `Frame` 接口。
- 一个五阶段流水线（VAD → STT → LLM → TTS → 传输层）作为脚本处理器。
- 一个 UPSTREAM 取消帧以演示打断。

运行它：

```
python3 code/main.py
```

跟踪显示正常流程和打断取消（在话语中途停止 TTS）。

## 使用它

- **Pipecat** 用于完全控制——自定义处理器、Python 优先、可插拔提供商。
- **LiveKit Agents** 用于优先 WebRTC 的部署和电话。
- **Vapi / Retell** 用于托管语音代理，无需 WebRTC 团队。
- **OpenAI Realtime / Gemini Live** 用于直接音频输入/输出（MultimodalAgent）。

## 发布

`outputs/skill-voice-pipeline.md` 搭建了一个 Pipecat 形状的语音流水线，包含 VAD + STT + LLM + TTS + 传输层以及打断处理。

## 练习

1. 向你的玩具流水线添加指标观察器：统计每阶段每秒的帧数。延迟累积在哪里？
2. 实现置信度门限的 STT：低于阈值则请求“能重复一遍吗？”
3. 添加语义话轮检测：简单规则——如果转录以“？”结尾，则话轮结束。
4. 阅读 Pipecat 的传输层文档。将标准库传输层替换为 SmallWebRTCTransport 配置（存根）。
5. 在相同查询上测量 OpenAI Realtime 与 STT+LLM+TTS 级联。文本级控制带来多少延迟成本？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  帧  |  "事件"  |  流水线中类型化的数据单元（音频、转录、文本、控制）  |
|  处理器  |  "流水线阶段"  |  带有 process(frame) 的处理程序  |
|  DOWNSTREAM  |  "正向流"  |  源到汇：音频输入，语音输出  |
|  UPSTREAM  |  "反馈流"  |  控制：取消、指标、打断  |
|  VAD  |  "语音活动检测"  |  检测用户是否在说话  |
|  语义话轮检测  |  "智能话轮结束"  |  基于模型判断用户是否说完  |
| MultimodalAgent | "直接音频代理" | 音频输入，音频输出；中间没有文本 |
| VoicePipelineAgent | "级联代理" | STT + LLM + TTS; 文本级控制 |

## 延伸阅读

- [Pipecat docs](https://docs.pipecat.ai/getting-started/introduction) — 基于帧的流水线，处理器，传输
- [Pipecat docs](https://docs.pipecat.ai/getting-started/introduction) — WebRTC + 语音原语
- [Pipecat docs](https://docs.pipecat.ai/getting-started/introduction) — 托管语音平台
- [Pipecat docs](https://docs.pipecat.ai/getting-started/introduction) — 托管语音，延迟基准测试的
