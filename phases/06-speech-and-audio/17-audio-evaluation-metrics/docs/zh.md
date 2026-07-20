# 音频评估——WER、MOS、UTMOS、MMAU、FAD及开放排行榜

> 你无法交付无法衡量的东西。本课列出了2026年每项音频任务的指标：ASR（WER、CER、RTFx），TTS（MOS、UTMOS、SECS、WER-on-ASR-round-trip），音频语言（MMAU、LongAudioBench），音乐（FAD、CLAP）和说话人（EER）。以及进行对比的排行榜。

**类型：** 学习
**语言：** Python
**前提条件：** 阶段6 · 04、06、07、09、10；阶段2 · 09（模型评估）
**时间：** ~60分钟

## 问题

每个音频任务都有多个指标，每个指标衡量不同的维度。使用错误的指标会导致你交付一个在仪表盘上看起来很棒但在生产中表现糟糕的模型。以下是2026年的权威列表：

|  任务  |  主要指标  |  次要指标  |
|------|---------|-----------|
|  ASR  |  WER  |  CER · RTFx · 首词延迟  |
|  TTS  |  MOS / UTMOS  |  SECS · WER-on-ASR-round-trip · CER · TTFA  |
|  语音克隆  |  SECS（ECAPA余弦）  |  MOS · CER  |
|  说话人验证  |  EER  |  minDCF · 工作点处的FAR/FRR  |
|  说话人日志  |  DER  |  JER · 说话人混淆  |
|  音频分类  |  top-1 · mAP  |  macro F1 · 每类召回率  |
|  音乐生成  |  FAD  |  CLAP · 人工评测MOS  |
|  音频语言模型  |  MMAU-Pro  |  LongAudioBench · AudioCaps FENSE  |
|  流式S2S  |  延迟P50/P95  |  WER · MOS  |

## 核心概念

![Audio evaluation matrix — metrics vs tasks vs 2026 leaderboards](../assets/eval-landscape.svg)

### ASR指标

**WER（词错误率）。** `(S + D + I) / N`。在评分前转换为小写、去除标点、标准化数字。使用`jiwer`或OpenAI的`whisper_normalizer`。&lt; 5% = 与人类相当的朗读语音。

**CER（字符错误率）。** 相同公式，字符级别。用于词分割存在歧义的声调语言（普通话、粤语）。

**RTFx（逆实时系数）。** 每挂钟秒处理的音频秒数。越高越好。Parakeet-TDT达到3380×。Whisper-large-v3约为30×。

**首词延迟。** 从音频输入到第一个转录token的挂钟时间。对流式处理至关重要。Deepgram Nova-3：约150毫秒。

### TTS指标

**MOS（平均意见分）。** 1-5分人工评分。黄金标准但速度慢。每个样本收集20+名听者，每个模型100+个样本。

**UTMOS（2022-2026）。** 学习型MOS预测器。在标准基准上与人类MOS的相关性约为0.9。F5-TTS：UTMOS 3.95；真实值：4.08。

**SECS（说话人编码器余弦相似度）。** 用于语音克隆。参考语音与克隆输出之间的ECAPA嵌入余弦。&gt; 0.75 = 可识别的克隆。

**WER-on-ASR-round-trip。** 在TTS输出上运行Whisper，与输入文本计算WER。捕获可懂度回归。2026年SOTA：&lt; 2% CER。

**TTFA（首次音频时间）。** 挂钟延迟。Kokoro-82M：~100毫秒；F5-TTS：~1秒。

### 语音克隆专属

**SECS + MOS + CER** 作为三件套。SECS高但MOS低的克隆意味着音色对但自然度差；相反则意味着声音自然但说话人错误。

### 说话人验证

**EER（等错误率）。** 错误接受率等于错误拒绝率时的阈值。ECAPA在VoxCeleb1-O上：0.87%。

**minDCF（最小检测成本）。** 在选定工作点（通常为FAR=0.01）处的加权成本。比EER更贴近生产。

### 说话人日志(Diarization)

**DER（说话人日志错误率）。** `(FA + Miss + Confusion) / total_speaker_time`。漏检语音+虚警语音+说话人混淆，各自作为分数。AMI会议：DER约10-20%是现实的。pyannote 3.1 + Precision-2商业版：在良好录制的音频上DER<10%。

**JER（Jaccard错误率）。** DER的替代指标，对短片段偏差具有鲁棒性。

### 音频分类

多标签：所有类别上的**mAP（平均精度均值）**。AudioSet：BEATs-iter3的mAP为0.548。

多类互斥：**top-1、top-5准确率**。Speech Commands v2：Audio-MAE的top-1准确率为99.0%。

不平衡：**宏F1** + **每类召回率**。报告每类指标——聚合准确率会隐藏哪些类别失败。

### 音乐生成

**FAD（Fréchet音频距离）。** 真实音频与生成音频的VGGish嵌入分布之间的距离。MusicGen-small在MusicCaps上：4.5。MusicLM：4.0。数值越低越好。

**CLAP分数。** 使用CLAP嵌入的文本-音频对齐分数。>0.3表示合理的对齐。

**听感评测MOS。** 仍然是消费级音乐的最终评判标准。Suno v5在TTS Arena上的ELO为1293（基于配对人工偏好）。

### 音频语言基准

**MMAU（大规模多音频理解）。** 1万个音频问答对。

**MMAU-Pro。** 1800个困难项，四个类别：语音/声音/音乐/多音频。随机猜测在四选一中为25%。Gemini 2.5 Pro整体约60%；所有模型在多音频上约22%。

**LongAudioBench。** 带有语义查询的多分钟片段。Audio Flamingo Next击败Gemini 2.5 Pro。

**AudioCaps / Clotho。** 字幕生成基准。使用SPICE、CIDEr、FENSE指标。

### 流式语音到语音

**延迟P50/P95/P99。** 从用户语音结束到首次可闻响应的挂钟时间。Moshi：200毫秒；GPT-4o实时：300毫秒。

**输出上的WER/MOS**。

**打断响应能力。** 从用户打断到助手静音的时间。目标<150毫秒。

### 2026年排行榜

|  排行榜  |  赛道  |  网址  |
|------------|--------|-----|
|  Open ASR排行榜 (HF)  |  英语 + 多语言 + 长格式  |  `huggingface.co/spaces/hf-audio/open_asr_leaderboard`  |
|  TTS竞技场 (HF)  |  英语TTS  |  `huggingface.co/spaces/TTS-AGI/TTS-Arena`  |
|  Artificial Analysis语音  |  TTS + STT，基于配对投票的ELO  |  `artificialanalysis.ai/speech`  |
|  MMAU-Pro  |  LALM推理  |  `mmaubenchmark.github.io`  |
|  SpeakerBench / VoxSRC  |  说话人识别  |  `voxsrc.github.io`  |
|  MMAU音乐子集  |  音乐LALM  |  (在MMAU内)  |
|  HEAR基准  |  自监督音频  |  `hearbenchmark.com`  |

## 动手构建

### 步骤1：带归一化的WER

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### 步骤2：TTS往返WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### 步骤 3：用于语音克隆的 SECS

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### 步骤 4：用于音乐生成的 FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### 步骤 5：用于说话人验证的 EER（与第6课相同的代码）

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## 使用它

每次部署都配对一个固定的评估工具集，每次模型更新时运行。三条基本规则：

1. **评分前进行归一化。** 小写化、去除标点、数字展开。报告归一化规则。
2. **报告分布，而非平均值。** 延迟的 P50/P95/P99。分类的每类召回率。MMAU的每个类别。
3. **运行一个规范的公共基准测试。** 即使你的生产数据不同，在 Open ASR / TTS Arena / MMAU 上报告可以让审查者进行同类比较。

## 陷阱

- **UTMOS 外推。** 在 VCTK 风格的干净语音上训练；对噪声/克隆/情感语音评分不佳。
- **MOS 评分者偏差。** 20 名 Amazon Mechanical Turk 工作者 ≠ 20 名目标用户。如果风险高，请为领域评分者付费。
- **FAD 依赖参考集。** 在模型间与相同的参考分布进行比较。
- **聚合 WER。** 总体 5% 的 WER 可能掩盖了口音语音上 30% 的 WER。按人口统计切片报告。
- **公共基准测试饱和。** 大多数前沿模型在标准基准测试上接近天花板。构建一个反映你业务流量的内部保留集。

## 发布

保存为 `outputs/skill-audio-evaluator.md`。为任何音频模型发布选择指标、基准测试和报告格式。

## 练习

1. **简单。** 运行 `code/main.py`。在玩具输入上计算 WER / CER / EER / SECS / FAD-ish / MMAU-ish。
2. **中等。** 构建一个 TTS 往返 WER 评估工具。将你的 Kokoro 或 F5-TTS 输出通过 Whisper 转录。计算 50 个提示上的 WER。标记 WER > 10% 的提示。
3. **困难。** 在 MMAU-Pro 语音+多音频子集（各 50 项）上对你的第 10 课 LALM 选择进行评分。报告每类准确率，并与已发布数值比较。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  WER  |  ASR 得分  |  归一化后词级别的 `(S+D+I)/N`。  |
|  CER  |  字符错误率 (Character WER)  |  用于声调语言或字符级系统。  |
|  MOS  |  主观意见  |  1-5 分；20 名以上听者 × 100 个样本。  |
|  UTMOS  |  机器学习 MOS 预测器  |  学习模型；与人类 MOS 的相关系数约为 0.9。  |
|  SECS  |  语音克隆相似度  |  参考语音与克隆语音之间的 ECAPA 余弦相似度。  |
|  EER  |  说话人验证得分  |  虚警率等于拒识率时的阈值。  |
|  DER  |  说话人日志得分  |  (虚警 + 漏检 + 混淆) / 总时长。  |
|  FAD  |  音乐生成质量  |  基于 VGGish 嵌入的 Fréchet 距离。  |
|  RTFx  |  吞吐量  |  每秒墙钟时间处理的音频秒数。  |

## 延伸阅读

- [jiwer](https://github.com/jitsi/jiwer) — 带有归一化工具的 WER/CER 库。
- [jiwer](https://github.com/jitsi/jiwer) — 学习型 MOS 预测器。
- [jiwer](https://github.com/jitsi/jiwer) — 音乐生成标准。
- [jiwer](https://github.com/jitsi/jiwer) — 2026 年实时排名。
- [jiwer](https://github.com/jitsi/jiwer) — 人工投票 TTS 排行榜。
- [jiwer](https://github.com/jitsi/jiwer) — LALM 推理排行榜。
- [jiwer](https://github.com/jitsi/jiwer) — 音频自监督学习基准。
