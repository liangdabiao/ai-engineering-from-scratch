# 水印 — SynthID、Stable Signature、C2PA

> 三种技术构成2026年AI生成内容溯源体系。SynthID（Google DeepMind）——2023年8月推出图像水印，2024年5月扩展至文本+视频（Gemini + Veo），2024年10月通过负责任生成式AI工具包开源文本水印，2025年11月随Gemini 3 Pro发布统一多媒体检测器。文本水印通过隐式调整下一个词元采样概率实现；图像/视频水印可抵抗压缩、裁剪、滤镜、帧率变化。Stable Signature（Fernandez等，ICCV 2023，arXiv:2303.15435）——微调潜在扩散解码器，使每个输出包含固定消息；在虚警率<1e-6条件下，裁剪至原内容10%的生成图像检测率>90%。后续论文"Stable Signature is Unstable"（arXiv:2405.07145，2024年5月）显示，微调可移除水印且保持质量。C2PA——加密签名、防篡改元数据标准（C2PA 2.2解释器2025）。水印与C2PA互补：元数据可被剥离但携带更丰富的溯源信息；水印在转码过程中持续存在但信息量较少。

**类型：** 构建
**语言：** Python（标准库，词元水印嵌入+检测）
**先修知识：** 阶段10·04（采样），阶段01·09（信息论）
**时间：** ~75分钟

## 学习目标

- 描述词元级水印（SynthID文本风格）及其可检测机制。
- 描述Stable Signature及其在2024年被移除攻击攻破的情况。
- 说明C2PA的作用及其与水印互补的原因。
- 描述主要局限：模型特定信号、改写鲁棒性、意义保持攻击（arXiv:2508.20228）。

## 问题

2023-2024年，深度伪造和AI生成内容大规模进入政治和消费者领域。水印是提议的技术溯源信号：在创建时标记生成内容，后续可检测。2025年证据表明：没有水印是无条件鲁棒的，但与C2PA元数据分层组合后，可提供可用的溯源方案。

## 核心概念

### 文本水印（SynthID文本风格）

Kirchenbauer等人2023年提出的机制，经Google产品化：

1. 在每个解码步骤，对前K个词元进行哈希，产生词汇表的伪随机划分（"绿色"与"红色"集合）。
2. 通过向绿色logits添加δ来偏向绿色集合的采样。
3. 生成内容中绿色词元数量高于随机概率。

检测：对每个前缀重新哈希，统计生成内容中绿色词元数量，计算z分数。水印文本z分数>0，人类文本z分数≈0。

性质：
- 对读者不可察觉（δ足够小，质量损失微小）。
- 在访问词汇表划分函数时可检测。
- 对改写不鲁棒——重写文本会破坏信号。

SynthID文本于2024年10月通过Google负责任生成式AI工具包开源。

### Stable Signature（图像）

Fernandez等人，ICCV 2023。微调潜在扩散解码器，使每张生成图像在潜在表示中嵌入固定二进制消息。通过神经解码器从潜在表示中解码检测。在虚警率<1e-6条件下，裁剪至原内容10%的图像检测率>90%。

2024年5月论文"Stable Signature is Unstable"（arXiv:2405.07145）：对解码器进行微调可移除水印且保持图像质量。对抗性后生成微调成本低廉；水印的对抗鲁棒性有限。

### SynthID统一检测器（2025年11月）

随Gemini 3 Pro推出：多模态检测器，通过单一API读取文本、图像、音频和视频中的SynthID信号。统一了Google溯源技术栈。

### C2PA

内容溯源与真实性联盟（Coalition for Content Provenance and Authenticity）。加密签名、防篡改元数据标准。C2PA 2.2解释器（2025）。C2PA清单记录溯源声明（创建者、时间、变换历史），由创建者密钥签名。

与水印互补：
- 元数据可被剥离；水印则不易被移除。
- 元数据丰富（完整溯源链）；水印携带少量信息。
- C2PA依赖平台采纳；水印自动嵌入。

Google在搜索、广告和"关于此图像"中集成了两者。

### 局限性

- **模型特定。** SynthID水印标记来自启用SynthID模型的生成内容。未启用SynthID模型的生成内容无水印，因此"无SynthID信号"不能证明真实性。
- **改写。** 文本水印无法抵御意义保持的改写。
- **变换攻击。** arXiv:2508.20228（2025）展示了意义保持攻击，可同时破坏文本水印和许多图像水印。
- **微调移除。** 根据"Stable Signature is Unstable"，后生成微调可移除嵌入式水印。

### 欧盟AI法案第50条

AI生成内容标签透明度准则（第一稿2025年12月，第二稿2026年3月，预计最终稿2026年6月，详见[European Commission status page](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content)）。截至2026年4月，准则仍为草案，时间表可能变更。监管层要求技术层实现。深度伪造必须被标注。

### 这在阶段18中的位置

第22-23课讨论模型输出内容（私有数据、溯源信号）。第27课涵盖训练数据治理。第24课是要求这些技术措施的监管框架。

## 使用它

`code/main.py`构建一个玩具文本水印。词元为整数0..N-1；水印采样偏向哈希定义的绿色集合。检测器计算绿色词元z分数。可观察1000词元生成内容的检测效果，看到改写破坏信号，并测量人类文本的假阳性率。

## 发布

本课生成`outputs/skill-provenance-audit.md`。给定一个带有溯源声明的部署内容，审计：水印机制（如有）、C2PA签名链（如有）、各自的对抗鲁棒性以及每模态覆盖情况。

## 练习

1. 运行`code/main.py`。报告水印1000词元生成内容与人类撰写文本的z分数。在95%置信阈值下识别假阳性率。

2. 实现一个改写攻击，将30%的词元替换为同义词。重新测量z分数。

3. 阅读Kirchenbauer等人2023年第6节关于鲁棒性的内容。为什么文本水印在改写下失败，而图像水印却能抵抗裁剪？

4. 设计一个使用SynthID-text + C2PA元数据的部署。描述消费者看到的溯源链。指出每个组件的一个失效模式。

5. 2024年的“Stable Signature is Unstable”结果表明微调会移除图像水印。设计一个限制这种攻击的部署控制措施——例如，要求对微调检查点的发布进行签名。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
|  SynthID  |  "Google's watermark"  |  跨模态溯源信号；文本、图像、音频、视频  |
|  Token watermark  |  "Kirchenbauer-style"  |  通过绿色token的z分数可检测的偏差采样文本水印  |
|  Stable Signature  |  "image watermark"  |  微调解码器水印；ICCV 2023  |
|  C2PA  |  "the metadata standard"  |  加密签名的防篡改溯源元数据  |
|  Paraphrase robustness  |  "does rewording break it"  |  文本水印属性；目前有限  |
|  Fine-tune removal  |  "adversarial unwatermark"  |  通过解码器微调移除图像水印的攻击  |
|  Cross-modal detector  |  "unified SynthID"  |  2025年11月跨模态的统一API  |

## 延伸阅读

- [Kirchenbauer et al. — A Watermark for Large Language Models (ICML 2023, arXiv:2301.10226)](https://arxiv.org/abs/2301.10226) — token水印机制
- [Kirchenbauer et al. — A Watermark for Large Language Models (ICML 2023, arXiv:2301.10226)](https://arxiv.org/abs/2301.10226) — 图像水印论文
- [Kirchenbauer et al. — A Watermark for Large Language Models (ICML 2023, arXiv:2301.10226)](https://arxiv.org/abs/2301.10226) — 移除攻击
- [Kirchenbauer et al. — A Watermark for Large Language Models (ICML 2023, arXiv:2301.10226)](https://arxiv.org/abs/2301.10226) — 跨模态水印
- [Kirchenbauer et al. — A Watermark for Large Language Models (ICML 2023, arXiv:2301.10226)](https://arxiv.org/abs/2301.10226) — 元数据标准
