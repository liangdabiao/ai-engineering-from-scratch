# 音频分类——从基于MFCC的k-NN到AST和BEATs

> 从“狗叫 vs 警笛”到“这是哪种语言”，一切皆属于音频分类。其特征是梅尔频谱。架构每十年一更迭。评估指标始终为AUC、F1和各类别召回率。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段6·02（频谱图和梅尔频谱），阶段3·06（CNN），阶段5·08（文本的CNN和RNN）
**时间：** ~75分钟

## 问题

你拿到一段10秒的音频片段。你想知道：“这是什么？” 城市声音（警笛、电钻、狗叫）、语音命令（是/否/停止）、语言识别（英/西/阿）、说话人情绪（愤怒/中性）、或环境声音（室内/室外、嘈杂人声）。所有这些都是*音频分类*，而到2026年，基线架构已经成熟：对数梅尔频谱→CNN或Transformer→softmax。

核心难点不在于网络。而在于数据。音频数据集存在严重的类别不平衡、强烈的领域偏移（干净vs嘈杂）和标签噪声（谁来判断“城市嘈杂人声”vs“餐厅噪音”？）。80%的问题在于数据整理、增强和评估，而非用Transformer替换CNN。

## 核心概念

![Audio classification ladder: k-NN on MFCCs to AST to BEATs](../assets/audio-classification.svg)

**基于MFCC的k-NN（上世纪90年代的基线）。** 对每个片段展开MFCC，计算与标注库的余弦相似度，返回前K个的多数投票。在干净的小数据集（Speech Commands、ESC-50）上出奇地强。无需GPU即可运行。

**基于对数梅尔频谱的2D CNN（2015-2019年）。** 将对数梅尔频谱视为图像。应用ResNet-18或VGG风格。沿时间轴进行全局平均池化。对类别进行softmax。在2026年的大多数Kaggle竞赛中仍是基准。

**音频频谱图Transformer，AST（2021-2024年）。** 将对数梅尔频谱分成块（例如16×16块），添加位置嵌入，输入ViT。在AudioSet上达到监督学习的最高水平（mAP 0.485）。

**BEATs和WavLM-base（2024-2026年）。** 在数百万小时数据上进行自监督预训练。用原来所需监督数据的1-10%在你的任务上微调。到2026年，这是非语音音频的默认起点。BEATs-iter3在AudioSet上比AST高出1-2个mAP，同时仅用1/4的计算量。

**Whisper编码器作为冻结骨干网络（2024年）。** 使用Whisper的编码器，去掉解码器，附加线性分类器。在语言识别和简单事件分类上接近SOTA，无需任何音频增强。这是“免费午餐”基线。

### 类别不平衡是真正的挑战

ESC-50：50类，每类40个片段——平衡，简单。UrbanSound8K：10类，不平衡比例10:1。AudioSet：632类，长尾分布100,000:1。有效的技术包括：

- 训练时平衡采样（不用于评估）。
- Mixup：线性插值两个片段（及其标签）作为增强。
- SpecAugment：随机掩码时间和频率段。简单，但至关重要。

### 评估

- 多类互斥（Speech Commands）：top-1准确率、top-5准确率。
- 多类多标签（AudioSet、UrbanSound风格）：平均精度均值（mAP）。
- 严重不平衡：各类别召回率 + 宏平均F1。

2026年你应该知道的数字：

|  基准  |  基线  |  2026 SOTA  |  来源  |
|-----------|----------|-----------|--------|
|  ESC-50  |  82%（AST）  |  97.0%（BEATs-iter3）  |  BEATs论文（2024）  |
|  AudioSet mAP  |  0.485（AST）  |  0.548（BEATs-iter3）  |  HEAR排行榜2026  |
|  Speech Commands v2  |  98%（CNN）  |  99.0%（Audio-MAE）  |  HEAR v2结果  |

## 动手构建

### 步骤1：特征提取

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### 步骤2：固定长度汇总

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

简单但强大：对13维MFCC取时间维度的均值和方差，得到26维的固定嵌入。运行即出。直到2017年，它在ESC-50上仍能击败最先进的神经网络基线。

### 步骤3：k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### 步骤4：升级为基于对数梅尔频谱的CNN

在PyTorch中：

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

3M参数。在ESC-50上用单张RTX 4090训练约10分钟。准确率80%以上。

### 步骤5：2026年的默认做法——微调BEATs

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

对于BEATs，通过`microsoft/BEATs-base`库使用`beats`；transformers API形状相同。

## 使用它

2026年技术栈：

|  情况  |  从何开始  |
|-----------|-----------|
|  极小数据集（<1000个片段）  |  基于MFCC均值的k-NN（你的基线）+ 音频增强  |
|  中等数据集（1K–100K）  |  BEATs或AST微调  |
|  大数据集（>100K）  |  从头训练或微调Whisper-encoder  |
|  实时，边缘计算  |  40-MFCC CNN，量化为int8（KWS风格）  |
|  多标签（AudioSet）  |  BEATs-iter3，使用BCE损失 + mixup + SpecAugment  |
|  语言识别  |  MMS-LID，SpeechBrain VoxLingua107基线  |

决策规则：**从冻结的骨干网络开始，而不是全新模型**。微调BEATs头部，几小时内即可达到SOTA的95%，无需数周。

## 发布

保存为`outputs/skill-classifier-designer.md`。针对给定的音频分类任务，选择架构、数据增强、类别平衡策略和评估指标。

## 练习

1. **简单。**运行`code/main.py`。它在4类合成数据集（不同音高纯音）上训练k-NN MFCC基线。报告混淆矩阵。
2. **中等。**将`code/main.py`替换为[均值、方差、偏度、峰度]。在相同合成数据集上，4矩池化是否优于均值+方差？
3. **困难。**使用`code/main.py`，在ESC-50折1上训练2D CNN。报告5折交叉验证准确率。添加SpecAugment（时间掩码=20，频率掩码=10）并报告差值。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  AudioSet  |  音频领域的ImageNet  |  谷歌的200万片段、632类弱标注YouTube数据集。  |
|  ESC-50  |  小型分类基准  |  50类×40个环境声音片段。  |
|  AST  |  音频频谱图Transformer  |  应用于log-mel patches的ViT；2021年SOTA。  |
|  BEATs  |  自监督音频  |  微软模型，iter3截至2026年领跑AudioSet。  |
|  Mixup  |  配对数据增强  |  `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`。  |
|  SpecAugment  |  基于掩码的数据增强  |  将频谱图的随机时间和频带置零。  |
|  mAP  |  主要多标签指标  |  跨类别和阈值的平均精度均值。  |

## 延伸阅读

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — 2021–2024年的主流架构。
- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — 2024年后的默认选择。
- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — 占主导地位的音频数据增强方法。
- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — 持续使用的50类基准。
- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — 632类YouTube分类法；仍是黄金标准。
