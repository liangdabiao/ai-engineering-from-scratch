# 语音克隆与语音转换

> 语音克隆是用他人的声音朗读你的文本。语音转换是将你的声音转写成他人的声音，同时保留你说的话。两者都依赖于同一种分解：将说话者身份与内容分离。

**类型：** 构建
**语言：** Python
**前置知识：** 第6阶段·06（说话人识别），第6阶段·07（语音合成）
**时长：** 约75分钟

## 问题

到2026年，一段5秒的音频片段足以用消费级GPU高质量克隆任何人的声音。ElevenLabs、F5-TTS、OpenVoice v2、VoiceBox都提供了零样本或少样本克隆技术。这项技术是一把双刃剑：既是福音（无障碍语音合成、配音、辅助语音），也是武器（诈骗电话、政治深度伪造、知识产权盗窃）。

两个密切相关的任务：

- **语音克隆（语音合成侧）：** 文本 + 5秒参考语音 → 该语音的音频。
- **语音转换（语音侧）：** 源音频（人物A说X）+ 人物B的参考语音 → 人物B说X的音频。

两者都将波形分解为（内容、说话者、韵律），然后将一个源的内容与另一个源的说话者重新组合。

2026年你需要满足的关键约束：**水印和同意门在欧盟（AI法案，2026年8月生效）和加利福尼亚州（AB 2905，2025年生效）是法律要求的**。你的流程必须发出不可听见的水印，并拒绝非同意克隆。

## 核心概念

![Voice cloning vs conversion: factorize, swap speaker, recombine](../assets/voice-cloning.svg)

**零样本克隆。** 将一段5秒的片段传递给一个在数千个说话者上训练过的模型。说话者编码器将片段映射到一个说话者嵌入；语音合成解码器以该嵌入和文本为条件。

使用该技术的有：F5-TTS（2024）、YourTTS（2022）、XTTS v2（2024）、OpenVoice v2（2024）。

**少样本微调。** 录制5-30分钟的目标语音。对基础模型进行LoRA微调一小时。质量从“还行”跃升到“无法区分”。Coqui和ElevenLabs都支持这种模式；社区将其与F5-TTS一起使用。

**语音转换（VC）。** 两个家族：

- **识别-合成。** 运行类似ASR的模型来提取内容表示（例如，软音素后验概率、PPG），然后用目标说话者嵌入重新合成。对语言和口音鲁棒。由KNN-VC（2023）、Diff-HierVC（2023）使用。
- **解耦。** 训练一个自编码器，在瓶颈处的潜空间中分离内容、说话者和韵律。在推理时交换说话者嵌入。质量较低但速度更快。由AutoVC（2019）、VITS-VC变体使用。

**基于神经编解码器的克隆（2024+）。** VALL-E、VALL-E 2、NaturalSpeech 3、VoiceBox——将音频视为来自SoundStream/EnCodec的离散令牌，训练一个大型自回归或流匹配模型处理编解码器令牌。在短提示下质量与ElevenLabs相当。

### 伦理问题，而非附加组件

**水印。** PerTh和SilentCipher（2024）在音频中不可察觉地嵌入一个约16-32位的ID。能抵抗重新编码、流媒体和常见编辑。生产级开源。

**同意门。** 必须为每个克隆输出配对一个可验证的同意记录。“我，Rohit，于2026-04-22，授权此语音用于X目的。”存储在防篡改日志中。

**检测。** AASIST、RawNet2和Wav2Vec2-AASIST可作为检测器。ASVspoof 2025挑战赛公布了针对ElevenLabs、VALL-E 2和Bark输出的最先进检测器的等错误率为0.8–2.3%。

### 数值（2026年）

|  模型  |  零样本？  |  SECS（目标相似度）  |  WER（可理解性）  |  参数量  |
|-------|-----------|--------------------|--------------|--------|
|  F5-TTS  |  是  |  0.72  |  2.1%  |  335M  |
|  XTTS v2  |  是  |  0.65  |  3.5%  |  470M  |
|  OpenVoice v2  |  是  |  0.70  |  2.8%  |  220M  |
|  VALL-E 2  |  是  |  0.77  |  2.4%  |  370M  |
|  VoiceBox  |  是  |  0.78  |  2.1%  |  330M  |

SECS > 0.70 对大多数听众来说通常与目标无法区分。

## 动手构建

### 第1步：用识别-合成分解（main.py中仅代码演示）

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

概念上简单；实现工作量主要在`tts_model`和说话者编码器中。

### 第2步：使用F5-TTS进行零样本克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

参考转录必须与音频精确匹配；不匹配会破坏对齐。

### 步骤3：使用KNN-VC进行语音转换

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC运行WavLM提取源语音和目标集每个帧的嵌入向量，然后用池中最近邻替换每个源帧。非参数化，只需一分钟目标语音即可工作。

### 步骤4：嵌入水印

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # returns payload bytes
```

约32比特载荷，在MP3重新编码和轻度噪声后仍可检测。

### 步骤5：同意门控

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 使用它

2026年技术栈：

|  情况  |  选择  |
|-----------|------|
|  5秒零样本克隆，开源  |  F5-TTS或OpenVoice v2  |
|  商业生产级克隆  |  ElevenLabs Instant Voice Clone v2.5  |
|  语音转换（重写）  |  KNN-VC或Diff-HierVC  |
|  多说话人微调  |  StyleTTS 2 + 说话人适配器  |
|  跨语言克隆  |  XTTS v2或VALL-E X  |
|  深度伪造检测  |  Wav2Vec2-AASIST  |

## 陷阱

- **参考转录不匹配。** F5-TTS等要求参考文本与参考音频完全匹配，包括标点符号。
- **混响参考。** 回声会破坏克隆。录制时应无回音、近距离麦克风。
- **情感不匹配。** 如果训练参考是“欢快的”，那么克隆出的所有内容都会是欢快的。参考情感应与目标用途匹配。
- **语言泄漏。** 克隆英语说话者再让模型说法语，通常会带有口音；应使用跨语言模型（XTTS, VALL-E X）。
- **无水印。** 2026年8月起在欧盟无法合法发布。

## 发布

保存为`outputs/skill-voice-cloner.md`。设计一个包含同意门控+水印+质量目标的克隆或转换流水线。

## 练习

1. **简单。** 运行`code/main.py`。通过计算两个“说话人”在交换前后的余弦相似度，展示说话人嵌入的交换。
2. **中等。** 使用OpenVoice v2克隆自己的声音。测量参考和克隆之间的SECS。通过Whisper测量CER。
3. **困难。** 对20个克隆应用SilentCipher水印，经过128 kbps MP3编码和解码，检测载荷。报告比特准确率。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  零样本克隆  |  5秒足够  |  预训练模型+说话人嵌入；无需训练。  |
|  PPG  |  音素后验图(Phonetic posteriorgram)  |  逐帧ASR后验，用作语言无关的内容表示。  |
|  KNN-VC  |  最近邻转换  |  用目标集中最近的帧替换每个源帧。  |
|  神经编解码TTS  |  VALL-E风格  |  基于EnCodec/SoundStream令牌的自回归模型。  |
|  水印  |  不可听签名  |  嵌入音频的比特，可经受重新编码。  |
|  SECS  |  克隆保真度  |  目标与克隆说话人嵌入的余弦相似度。  |
|  AASIST  |  深度伪造检测器  |  反欺骗模型；检测合成语音。  |

## 延伸阅读

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 开源SOTA零样本克隆。
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885)和[Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) — 神经编解码TTS。
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 基于解耦的语音转换。
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 基于检索的语音转换。
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 生产级32位音频水印。
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 检测器与合成器的军备竞赛，2026年更新。
