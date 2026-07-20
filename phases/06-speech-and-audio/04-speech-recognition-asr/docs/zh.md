# 语音识别 (ASR) — CTC, RNN-T, Attention

> 语音识别是在每个时间步上进行音频分类，通过一个了解英语和静音的序列模型将它们粘合在一起。CTC、RNN-T 和注意力是三种实现方式。选择一种并理解其原因。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段6·02（频谱图与梅尔谱），阶段5·08（用于文本的CNN和RNN），阶段5·10（注意力机制）
**时间：** 约45分钟

## 问题

你有一个10秒16kHz的片段。你想要一个字符串："turn on the kitchen lights"。挑战在于结构：音频帧与字符并非一一对应。单词"okay"可能耗时200毫秒或1200毫秒。静音分隔了话语。某些音素比其他音素更长。输出令牌(token)的数量事先未知。

三种公式化解法：

1. **CTC（连接时序分类）. ** 逐帧发射令牌概率，包括一个特殊的 *空白*。解码时合并重复和空白。非自回归，速度快。被wav2vec 2.0、MMS使用。
2. **RNN-T（循环神经网络换能器）. ** 联合网络根据编码器帧和先前令牌预测下一个令牌。可流式处理。被Google的设备端ASR、NVIDIA Parakeet使用。
3. **注意力编码器-解码器. ** 编码器将音频压缩为隐藏状态，解码器通过交叉注意力自回归生成令牌。被Whisper、SeamlessM4T使用。

2026年，LibriSpeech test-clean上的SOTA词错误率(WER)为1.4%（Parakeet-TDT-1.1B，NVIDIA）和1.58%（Whisper-Large-v3-turbo）。差异很小；部署差异却很大。

## 核心概念

![Three ASR formulations: CTC, RNN-T, attention-encoder-decoder](../assets/asr-formulations.svg)

**CTC直觉.** 令编码器输出`T`帧级别分布在`V+1`个令牌（V个字符+空白）上。对于长度为`U < T`的目标字符串`y`，任何坍缩为`y`的帧对齐都被计入。CTC损失对所有此类对齐求和。推理：逐帧argmax，合并重复，移除空白。

优点：非自回归、可流式处理、零前瞻。缺点：*条件独立假设*——每一帧预测独立于其他帧，因此没有内部语言模型。通过波束搜索或浅融合使用外部语言模型(LM)来修复。

**RNN-T直觉.** 增加一个*预测器*网络，用于嵌入令牌历史；以及一个*联合器*，将预测器状态与编码器帧结合，形成在`V+1`上的联合分布（其中`+1`是空/无发射）。显式建模了CTC忽略的条件依赖。可流式处理，因为每一步仅依赖于过去的帧和过去的令牌。

优点：可流式处理+内部语言模型。缺点：训练更复杂且内存消耗大（3D损失网格）；RNN-T损失核本身就是一个完整的库类别。

**注意力编码器-解码器.** 编码器（6-32个Transformer层）对对数梅尔谱帧进行处理。解码器（6-32个Transformer层）通过交叉注意力关注编码器输出，以自回归方式生成令牌。无对齐约束——注意力可以关注音频的任何位置。除非限制注意力（分块Whisper-Streaming，2024），否则不可流式处理。

优点：在离线ASR上质量最高，易于使用标准序列到序列工具训练。缺点：自回归延迟与输出长度成正比；未经工程处理无法流式处理。

### 词错误率(WER)：唯一数字

**词错误率** = `(S + D + I) / N`，其中S=替换数，D=删除数，I=插入数，N=参考词数。在词级别上匹配莱文斯坦编辑距离。越低越好。词错误率超过20%通常不可用；低于5%则达到朗读语音的人类水平。2026年在标准基准上的数据：

|  模型  |  LibriSpeech test-clean  |  LibriSpeech test-other  |  参数量  |
|-------|------------------------|------------------------|------|
|  Parakeet-TDT-1.1B  |  1.40%  |  2.78%  |  1.1B参数  |
|  Whisper-Large-v3-turbo  |  1.58%  |  3.03%  |  809M  |
|  Canary-1B Flash  |  1.48%  |  2.87%  |  1B  |
|  Seamless M4T v2  |  1.7%  |  3.5%  |  2.3B  |

所有这些都是基于编码器-解码器或RNN-T的。纯CTC系统（wav2vec 2.0）在test-clean上约为1.8–2.1%。

## 动手构建

### 步骤1：贪婪CTC解码

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: list of per-frame probability vectors
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

两条规则：合并连续重复，丢弃空白。示例：`a a _ _ a b b _ c` → `a a b c`。

### 步骤2：波束搜索CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

生产环境使用前缀树波束搜索与语言模型融合；这是概念骨架。

### 步骤3：词错误率(WER)

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### 步骤4：使用Whisper进行推理

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

2026年最强通用ASR的一行指令。在24GB GPU上以约20倍实时运行。

### 步骤5：使用Parakeet或wav2vec 2.0进行流式处理

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

流式ASR需要分块编码器注意力和结转状态；使用支持它的库（NeMo用于Parakeet，`transformers`管道与`chunk_length_s`）。

## 使用它

2026年技术栈：

|  情况  |  选择  |
|-----------|------|
| 英文，离线，最高质量 | Whisper-large-v3-turbo |
| 多语言，鲁棒 | SeamlessM4T v2 |
| 流式，低延迟 | Parakeet-TDT-1.1B 或 Riva |
| 边缘设备，移动端，延迟<500ms | Whisper-Tiny 量化版或 Moonshine (2024) |
| 长音频 | 基于VAD分块的Whisper (WhisperX) |
| 特定领域（医疗、法律） | 微调 wav2vec 2.0 + 领域语言模型融合 |

## 2026年仍存在的陷阱

- **不要VAD。** 对静音运行Whisper会产生幻觉（“感谢观看！”）。始终使用VAD进行门控。
- **字符级 vs 词级 vs 子词级WER。** 在归一化（小写，去除标点）*之后*报告词级WER。
- **语言ID漂移。** Whisper的自动语言ID会将嘈杂片段错误路由到日语或威尔士语；当你确定语言时，使用`language="en"`强制指定。
- **长片段无分块。** Whisper有30秒的窗口。超过此长度使用`language="en"`。

## 发布

保存为`outputs/skill-asr-picker.md`。针对给定部署目标选择模型、解码策略、分块和语言模型融合。

## 练习

1. **简单。** 运行`code/main.py`。它贪心地解码一个手工制作的CTC输出，并计算相对于参考的WER。
2. **中等。** 正确实现步骤2中的前缀树束搜索（考虑空白合并规则）。与贪心解码在10个示例的合成数据集上进行比较。
3. **困难。** 在`whisper-large-v3-turbo`上使用`code/main.py`。计算前100个话语的WER。与已发表数字比较。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
| CTC | 空白标记损失 | 对所有帧到标记对齐的边缘化；非自回归。 |
| RNN-T | 流式损失 | CTC + 下一标记预测器；处理词序。 |
| 注意力编码器-解码器 | Whisper风格 | 编码器 + 交叉注意力解码器；最佳离线质量。 |
| WER | 你报告的数字 | 词级的`(S+D+I)/N`。 |
| 空白 | 空 | CTC中表示“此帧无发射”的特殊标记。 |
| 语言模型融合 | 外部语言模型 | 在束搜索期间添加加权的语言模型对数概率。 |
| VAD | 静音门控 | 语音活动检测器；裁剪非语音部分。 |

## 延伸阅读

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — CTC论文。
- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — RNN-T论文。
- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — 2022年经典论文；2024年v3-turbo扩展版。
- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — 2026年Open ASR排行榜领军者。
- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — 涵盖25+模型的实时基准测试。
