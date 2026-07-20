# 说话人识别与验证(Speaker Recognition & Verification)

> ASR问的是“他说了什么？”说话人识别问的是“谁说的？”数学看起来一样——嵌入(embedding)加余弦——但每个生产决策都取决于单个EER数值。

**类型：** 构建
**语言：** Python
**先决条件：** 阶段6·02（语谱图与梅尔）、阶段5·22（嵌入模型）
**时间：** 约45分钟

## 问题

用户说出一句口令。你想知道：这是他们所声称的那个人吗（*验证*，1:1），还是你注册库中的第一个人（*识别*，1:N）？或者两者都不是——这是一个未知的说话人（*开放集*）？

2018年前：GMM-UBM + i-vector。EER尚可，但对信道变化（手机vs笔记本）和情感脆弱。2018–2022：x-vector（基于角度间隔训练的TDNN骨干网络）。2022年后：ECAPA-TDNN和WavLM-large嵌入。到2026年，该领域由三个模型和一个指标主导。

这个指标是**EER**——等错误率(Equal Error Rate)。设置决策阈值使得错误接受率(False Accept Rate)等于错误拒绝率(False Reject Rate)。交叉点即为EER。每篇论文、每个排行榜、每次采购招标都使用它。

## 核心概念

![Enrollment + verification pipeline with embedding + cosine + EER](../assets/speaker-verification.svg)

**流程。** 注册：录制目标说话人5–30秒；计算固定维度的嵌入（ECAPA-TDNN为192维，WavLM-large为256维）。验证：获取测试话语的嵌入；计算余弦相似度；与阈值比较。

**ECAPA-TDNN（2020年，到2026年仍占主导）。** 强调通道注意力、传播与聚合——时延神经网络(Time-Delay Neural Network)。带有压缩-激励(squeeze-excitation)的1D卷积块、多头注意力池化，后接线性层至192维。在VoxCeleb 1+2（2700个说话人，110万条话语）上使用加法角度间隔损失(Additive Angular Margin loss, AAM-softmax)训练。

**WavLM-SV（2022年后）。** 使用AAM损失微调预训练的WavLM-large SSL骨干网络。质量更高但更慢——300+ MB vs 15 MB。

**x-vector（基线）。** TDNN + 统计池化。经典；在CPU/边缘设备上仍然有用。

**AAM-softmax。** 标准softmax在角度空间中添加间隔`m`：正确类别`cos(θ + m)`。强制类间角度分离。典型`m=0.2`，尺度`s=30`。

### 评分

- **余弦** 在注册与测试嵌入之间。基于阈值的决策。
- **PLDA（概率线性判别分析）。** 将嵌入投影到潜在空间，其中同说话人与不同说话人有闭式似然比。在余弦基础上额外降低EER 10–20%。2020年前标准；现在仅在闭集设置中使用。
- **得分归一化。** `S-norm` 或 `AS-norm`：针对冒名者均值和标准差的队列归一化每个得分。跨域评估必不可少。

### 你应该知道的数字（2026年）

|  模型  |  VoxCeleb1-O EER  |  参数量  |  吞吐量 (A100)  |
|-------|-----------------|--------|-------------------|
|  x-vector (经典)  |  3.10%  |  5 M  |  400× RT  |
|  ECAPA-TDNN  |  0.87%  |  15 M  |  200× RT  |
|  WavLM-SV large  |  0.42%  |  316 M  |  20× RT  |
|  Pyannote 3.1 分割+嵌入  |  0.65%  |  6 M  |  100× RT  |
|  ReDimNet (2024)  |  0.39%  |  24 M  |  100× RT  |

### 说话人日志(Diarization)

多说话人片段中的“谁什么时候说话”。流程：VAD → 分割 → 嵌入每个片段 → 聚类（凝聚或谱聚类） → 平滑边界。现代堆栈：`pyannote.audio` 3.1，它将说话人分割+嵌入+聚类整合到一个调用中。2026年AMI上的SOTA DER约为15%（从2022年的23%下降）。

## 动手构建

### 步骤1：从MFCC统计量构建玩具嵌入

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

远不是SOTA——仅用于教学。`code/main.py`将其作为合成说话人数据上的概念验证。

### 步骤2：余弦相似度+阈值

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### 步骤3：从相似度对计算EER

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

返回(eer, threshold_at_eer)。报告两者。

### 步骤4：使用SpeechBrain进行生产级实现

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data
```

### 步骤5：使用pyannote进行说话人日志

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 使用它

2026年技术栈：

|  情况  |  选择  |
|-----------|------|
|  闭集1:1验证，边缘设备  |  ECAPA-TDNN + 余弦阈值  |
| 开放集验证，云端  |  WavLM-SV + AS-norm |
| 说话人日志（会议、播客）  |  `pyannote/speaker-diarization-3.1` |
| 反欺骗（重放/深度伪造检测）  |  AASIST 或 RawNet2 |
| 微型嵌入式（关键词唤醒+注册）  |  Titanet-Small (NeMo) |

## 陷阱

- **通道不匹配。** 在 VoxCeleb（网络视频）上训练的模型 ≠ 电话通话音频。始终在目标通道上评估。
- **短语音段。** 测试音频低于3秒时，EER会急剧下降。
- **带噪声的注册。** 一个带噪声的注册会污染锚点。使用至少3个干净样本并取平均。
- **跨条件固定阈值。** 始终在目标域中保留的开发集上调整阈值。
- **对未归一化的嵌入使用余弦相似度。** 首先进行L2归一化；否则幅度会主导结果。

## 发布

保存为 `outputs/skill-speaker-verifier.md`。选择模型、注册协议、阈值调整方案和欺诈防护措施。

## 练习

1. **简单。** 运行 `code/main.py`。构建合成“说话人”（不同的语调特征），注册，计算100对测试列表上的EER。
2. **中等。** 在30个VoxCeleb1语料（5个说话人×每个6段）上使用SpeechBrain ECAPA。使用余弦对比PLDA计算EER。
3. **困难。** 使用 `code/main.py` 构建完整的 注册→日志→验证 流水线。在AMI开发集上评估DER。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  EER  |  主要指标  |  误接受率等于误拒绝率的阈值。 |
|  验证  |  1:1  |  “这是Alice吗？” |
|  辨认  |  1:N  |  “谁在说话？” |
|  开放集  |  未知可能  |  测试集可能包含未注册的说话人。 |
|  注册  |  注册  |  计算说话人的参考嵌入。 |
|  AAM-softmax  |  损失函数  |  带有附加角度间隔的Softmax；强制类别分离。 |
|  PLDA  |  经典评分  |  概率线性判别分析；基于嵌入的似然比评分。 |
|  DER  |  说话人日志指标  |  说话人日志错误率 — 漏报+误报+混淆。 |

## 延伸阅读

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — 经典深度嵌入论文。
- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — 2020–2026年主流架构。
- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — 用于说话人验证和日志的SSL主干网络。
- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — 生产级日志+嵌入堆栈。
- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — 各模型当前EER排名。
