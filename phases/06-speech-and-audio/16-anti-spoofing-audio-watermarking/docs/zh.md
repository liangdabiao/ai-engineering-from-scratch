# 语音反欺骗与音频水印 — ASVspoof 5、AudioSeal、WaveVerify

> 语音克隆的防御技术发展比攻击慢。2026年生产级语音系统需要两样东西：一个检测器（AASIST、RawNet2），用于分类真实与伪造语音；以及一个水印（AudioSeal），能够抵抗压缩和编辑。两者都要部署，否则不要发布语音克隆。

**类型：** 构建
**语言：** Python
**前置知识：** 第6阶段·06（说话人识别）、第6阶段·08（语音克隆）
**时间：** 约75分钟

## 问题

三种相关的防御手段：

1. **反欺骗/深度伪造检测。** 给定一段音频，它是合成还是真实的？ASVspoof基准（ASVspoof 2019 → 2021 → 5）是黄金标准。
2. **音频水印。** 在生成的音频中嵌入一个不可感知的信号，检测器稍后可以提取出来。AudioSeal（Meta）和WavMark是开放选择。
3. **认证溯源。** 对音频文件+元数据进行加密签名。C2PA/内容真实性倡议（Content Authenticity Initiative）。

检测处理不合作的攻击者。水印处理合规性——AI生成的音频应能被识别。2026年两者都是必需的。

## 核心概念

![Anti-spoofing vs watermarking vs provenance — three defense layers](../assets/spoofing-watermark.svg)

### ASVspoof 5——2024-2025基准

与之前版本的最大变化：

- **众包数据**（非录音室清洁）——真实条件。
- **约2000个说话人**（之前约100个）。
- **32种攻击算法。** 包括TTS、语音转换（Voice Conversion）和对抗扰动。
- **两个赛道。** 对策（CM）独立检测；针对生物识别系统的抗欺骗ASV（SASV）。

在ASVspoof 5上的当前最佳：约7.23% EER。在较旧的ASVspoof 2019 LA上：0.42% EER。实际部署：在野外片段上预计5-10% EER。

### AASIST和RawNet2——检测模型家族

**AASIST**（2021年，更新至2026年）。基于谱特征的图注意力。当前ASVspoof 5对策任务的最佳模型。

**RawNet2。** 基于原始波形的卷积前端+TDNN骨干。较简单的基线；微调后仍具竞争力。

**NeXt-TDNN + SSL特征。** 2025变体：ECAPA风格+WavLM特征+焦点损失。在ASVspoof 2019 LA上达到0.42% EER。

### AudioSeal——2024年水印默认选择

Meta的**AudioSeal**（2024年1月，v0.2于2024年12月）。关键设计：

- **局部化。** 以16 kHz采样率（1/16000秒）逐帧检测水印。
- **生成器与检测器联合训练。** 生成器学习嵌入不可听信号；检测器学习通过增强变换找到它。
- **鲁棒。** 抵抗MP3/AAC压缩、EQ、±10%速度变化、噪声混合（+10 dB信噪比）。
- **快速。** 检测器运行速度为实时485倍；比WavMark快1000倍。
- **容量。** 16比特有效载荷（可编码模型ID、生成时间戳、用户ID），可嵌入每个话语。

### WavMark

AudioSeal之前的开放基线。可逆神经网络，32比特/秒。问题：

- 同步暴力破解速度慢。
- 可通过高斯噪声或MP3压缩移除。
- 不适用于实时场景。

### WaveVerify（2025年7月）

解决了AudioSeal的弱点——特别是时间操作（反转、变速）。使用FiLM生成器+专家混合（Mixture-of-Experts）检测器。在标准攻击上与AudioSeal竞争；处理时间编辑。

### 攻击者利用的差距

根据AudioMarkBench：“在音高偏移下，所有水印的比特恢复准确率低于0.6，表明几乎完全移除。”**音高偏移是通用攻击。** 2026年没有一种水印对激进音高修改完全鲁棒。这就是为什么你需要检测（AASIST）与水印结合。

### C2PA / 内容真实性倡议（Content Authenticity Initiative）

不是一种机器学习技术——而是一个清单格式。音频文件携带关于创建工具、作者、日期的加密签名元数据。Audobox / Seamless使用它。有利于溯源；但如果恶意行为者重新编码并移除元数据，则无效。

## 动手构建

### 第一步：一个简单的谱特征检测器（示例）

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

合成语音通常具有异常平坦的高频能量。生产级检测器使用AASIST，而不是这个。但直觉是成立的。

### 第二步：AudioSeal嵌入+检测

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: float in [0, 1] — probability of watermark presence
# decoded_payload: 16 bits; match against embedded payload
```

### 第三步：评估——EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### 第4步：生产集成

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

每个生成版本都附带：(1) 水印，(2) 签名清单，(3) 符合保留策略的审计日志。

## 使用它

|  用例 | 防御措施  |
|----------|---------|
|  语音合成/声音克隆  |  每个输出都嵌入AudioSeal（不可协商）  |
|  生物特征语音解锁  |  AASIST + ECAPA集成；活体挑战  |
|  呼叫中心欺诈检测  |  对20%呼入电话样本应用AASIST  |
|  播客真实性验证  |  上传时进行C2PA签名，若为AI生成则添加AudioSeal  |
|  研究/训练检测器  |  ASVspoof 5训练/开发/评估集  |

## 陷阱

- **水印从未运行检测器。**毫无意义。在CI中部署检测器。
- **未校准的检测。**在ASVspoof LA上训练的AASIST过拟合；实际准确率下降。在你的领域进行校准。
- **音调偏移漏洞。**剧烈的音调偏移会移除大部分水印。需要备用检测方案。
- **元数据剥离与重托管。**C2PA可通过重新编码轻易绕过。始终同时添加加密与感知防御（水印）。
- **将活体检测视为检测手段。**要求用户说出随机短语。可防止重放攻击，但无法防御实时克隆。

## 发布

另存为`outputs/skill-spoof-defender.md`。为语音生成部署选取检测模型、水印、出处清单及操作手册。

## 练习

1. **简单。**运行`code/main.py`。玩具检测器+玩具水印嵌入/检测合成音频。
2. **中等。**安装`code/main.py`，在TTS输出中嵌入16位有效载荷，重新解码。用噪声破坏音频并测量位恢复准确率。
3. **困难。**在ASVspoof 2019 LA上微调RawNet2或AASIST。测量EER。在保留的F5-TTS生成片段上测试——观察域外检测性能下降情况。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  ASVspoof  |  基准  |  两年一次挑战；2024年为ASVspoof 5。  |
|  CM (对抗措施)  |  检测器  |  分类器：真实语音 vs 合成/转换语音。  |
|  SASV  |  说话人验证+对抗措施  |  集成生物特征与欺骗检测。  |
|  AudioSeal  |  Meta水印  |  局部化，16位有效载荷，比WavMark快485倍。  |
|  位恢复准确率  |  水印存活率  |  攻击后恢复的有效载荷比特比例。  |
|  C2PA  |  出处清单  |  关于创作/作者身份的加密元数据。  |
|  AASIST  |  检测器家族  |  基于图注意力的抗欺骗最新技术。  |

## 延伸阅读

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825)——当前基准。
- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825)——默认水印。
- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825)——针对时间攻击的MoE检测器。
- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825)——最新检测骨干网络。
- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825)——鲁棒性评估。
- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825)——出处清单格式。
