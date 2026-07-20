# 流式语音到语音——Moshi、Hibiki与全双工对话

> 2024-2026 年重新定义了语音 AI。Moshi 推出单一模型，能够同时听和说，延迟为 200 毫秒。Hibiki 逐块进行语音到语音翻译。两者都摒弃了 ASR → LLM → TTS 流水线，采用基于 Mimi 编解码器令牌的统一全双工架构。这是新的参考设计。

**类型：** 学习
**语言：** Python
**前置条件：** 阶段 6 · 13（神经音频编解码器）、阶段 6 · 11（实时音频）、阶段 7 · 05（完整 Transformer）
**时间：** ~75 分钟

## 问题

从第 11 课和第 12 课构建的每个语音代理都有大约 300-500 毫秒的固有延迟下限：VAD 触发、STT 处理、LLM 推理、TTS 生成。每个阶段都有其自身的最低延迟。你可以调整和并行化，但流水线的形状限制了上限。

Moshi（Kyutai，2024-2026）提出了一个不同的问题：如果没有流水线呢？如果一个模型直接、连续地接收音频输入并输出音频，而文本作为中间“内心独白”而不是必需阶段，会怎么样？

答案是**全双工语音到语音**。理论延迟 160 毫秒（80 毫秒 Mimi 帧 + 80 毫秒声学延迟）。在单个 L4 GPU 上实际延迟 200 毫秒。这是最佳流水线语音代理所能实现的一半。

## 核心概念

![Moshi architecture: two parallel Mimi streams + inner-monologue text](../assets/moshi-hibiki.svg)

### Moshi 架构

**输入。** 两个 Mimi 编解码器流，均为 12.5 Hz × 8 个码本：

- 流 1：用户音频（Mimi 编码，持续到达）
- 流 2：Moshi 自己的音频（由 Moshi 生成）

**Transformer。** 一个 7B 参数的时间 Transformer 处理两个流和一个文本“内心独白”流。在每个 80 毫秒步长中，它：

1. 消耗最新的用户 Mimi 令牌（8 个码本）。
2. 消耗最近的 Moshi Mimi 令牌（8 个码本，如已生成）。
3. 生成下一个 Moshi 文本令牌（内心独白）。
4. 生成下一个 Moshi Mimi 令牌（通过一个小型深度 Transformer 生成 8 个码本）。

所有三个流——用户音频、Moshi 音频、Moshi 文本——并行运行。Moshi 可以在说话的同时听到用户的声音；当用户打断时可以自我打断；可以在不破坏主要话语的情况下进行反向通道（“嗯哼”）。

**深度 Transformer。** 在一个帧内，8 个码本不是并行预测的——它们存在码本间依赖关系。一个小的 2 层“深度 Transformer”在 80 毫秒内顺序预测它们。这是 AR 编解码器语言模型的标准分解方法（也用于 VALL-E、VibeVoice）。

### 为什么内心独白文本有帮助

没有显式文本，模型必须隐式地在其声学流中对语言进行建模。Moshi 的见解：强制它在音频旁边发出文本令牌。文本流本质上是 Moshi 所说内容的转录。这提高了语义连贯性，使得更容易替换语言模型头，并免费提供转录文本。

### Hibiki：流式语音到语音翻译

相同的架构，在翻译对上进行训练。源音频输入，目标语言音频输出，连续进行。Hibiki-Zero（2026 年 2 月）消除了对词级对齐训练数据的需求——使用句子级数据 + GRPO 强化学习进行延迟优化。

最初支持四种语言对；可以用约 1000 小时适应新语言。

### 更广泛的 Kyutai 技术栈（2026）

- **Moshi** — 全双工对话（法语优先，英语支持良好）
- **Hibiki / Hibiki-Zero** — 同声传译
- **Kyutai STT** — 流式 ASR（500 毫秒或 2.5 秒前瞻）
- **Kyutai Pocket TTS** — 1 亿参数 TTS，在 CPU 上运行（2026 年 1 月）
- **Unmute** — 结合这些功能的完整流水线，部署在公共服务器上

在 L40S GPU 上的吞吐量：64 个并发会话，3 倍实时。

### Sesame CSM——同类产品

Sesame CSM（2025）使用了类似的想法——一个 Llama-3 主干和一个 Mimi 编解码器头。但 CSM 是单向的（接收上下文+文本，生成语音）而不是全双工。它是市场上最好的“语音存在感”TTS；与 Moshi 的全双工能力不太一样。

### 2026 年性能数据

|  模型  |  延迟  |  用例  |  许可证  |
|-------|---------|----------|---------|
|  Moshi  |  200 毫秒（L4）  |  全双工英语/法语对话  |  CC-BY 4.0  |
|  Hibiki  |  12.5 赫兹帧率  |  法语↔英语流式翻译  |  CC-BY 4.0  |
|  Hibiki-Zero  |  同上  |  5 种语言对，无对齐数据  |  CC-BY 4.0  |
|  Sesame CSM-1B  |  200 毫秒 TTFA  |  上下文条件 TTS  |  Apache-2.0  |
|  GPT-4o Realtime  |  ~300 毫秒  |  闭源，OpenAI API  |  商业  |
| Gemini 2.5 Live  |  ~350 ms  |  封闭，Google API  |  商业 |

## 动手构建

### 步骤1：接口

Moshi 暴露一个 WebSocket 服务器，它接收 80 毫秒的 Mimi 编码音频块，并返回 80 毫秒的 Mimi 编码音频块。双向进行。持续不断。

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### 步骤2：全双工循环

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

两个方向同时运行。Python asyncio 或 Rust futures 是标准传输方式。

### 步骤3：训练目标（概念性）

对于每个 80 毫秒帧 `t`：

- 输入：`user_mimi[0..t]`, `moshi_mimi[0..t-1]`, `moshi_text[0..t-1]`
- 预测：`user_mimi[0..t]`, 然后 `moshi_mimi[0..t-1]`

文本在音频之前被预测（内部独白）；音频在深度变换器内按码本顺序预测。

### 步骤4：Moshi 的优势与劣势

Moshi 的优势：

- 在廉价硬件上端到端低于 250 毫秒。
- 自然的反馈和打断。
- 无需管道粘合代码。

Moshi 的劣势：

- 工具调用（未针对此训练；你需要单独的 LLM 路径）。
- 长推理（Moshi 是一个约 8B 的对话模型，不是 Claude/GPT-4）。
- 小众话题的事实准确性。
- 大多数生产级企业用例（2026 年仍使用管道）。

## 使用它

|  情况  |  选择  |
|-----------|------|
|  最低延迟语音伴侣  |  Moshi |
|  实时翻译通话  |  Hibiki |
|  语音演示/研究  |  Moshi, CSM |
|  带工具的企业代理  |  管道（第12课），而非 Moshi |
|  上下文中的自定义语音 TTS  |  Sesame CSM |
|  语音到语音，任意语言  |  GPT-4o Realtime 或 Gemini 2.5 Live（商业） |

## 陷阱

- **有限的工具调用。** Moshi 是一个对话模型，不是代理框架。结合管道以使用工具。
- **特定语音条件。** Moshi 使用单一训练角色；克隆需要单独的训练运行。
- **语言覆盖。** 法语+英语优秀；其他语言有限。Hibiki-Zero 有帮助，但仍需训练数据。
- **资源成本。** 完整的 Moshi 会话占用一个 GPU 插槽；不是便宜的共享租户部署模式。

## 发布

保存为 `outputs/skill-duplex-pipeline.md`。为语音代理工作负载选择管道还是全双工架构，并说明理由。

## 练习

1. **简单。** 运行 `code/main.py`。它象征性地模拟双流+内部独白架构。
2. **中等。** 从 HuggingFace 拉取 Moshi，运行服务器，测试一次对话。测量从用户说话结束到 Moshi 开始响应的挂钟延迟。
3. **困难。** 使用你的第12课管道代理，并在 20 个匹配的测试话语上与 Moshi 比较 P50 延迟。写出管道在架构上仍然获胜的情况。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  全双工  |  同时听说  |  同一模型上同时激活两个音频流。 |
|  内部独白  |  模型的文本流  |  Moshi 在其音频输出旁同时发出文本 token。 |
|  深度变换器  |  码本间预测器  |  在一个 80 毫秒帧内预测 8 个码本的小型变换器。 |
|  Mimi  |  Kyutai 的编解码器  |  12.5 Hz × 8 码本；语义+声学；驱动 Moshi。 |
|  流式 S2S  |  音频到音频实时  |  逐块翻译/对话，无管道阶段。 |
|  反馈通道  |  "嗯"反应  |  Moshi 可以发出小确认而不打断其发言轮次。 |

## 延伸阅读

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — 论文。
- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — 无对齐数据的流式翻译。
- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — CSM 规范。
- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — 安装+服务器。
- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — 封闭商业同行。
- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — 底层的 STT/TTS 框架。
