# OCR与文档理解

> OCR是一个三阶段流水线——检测文本框，识别字符，然后进行布局排列。每个现代OCR系统都会重新排序这些阶段或将它们合并。

**类型：** 学习 + 使用
**语言：** Python
**前置条件：** 第4阶段第6课（检测），第7阶段第2课（自注意力）
**时间：** 约45分钟

## 学习目标

- 追踪经典OCR流水线（检测 -> 识别 -> 布局）和现代端到端替代方案（Donut、Qwen-VL-OCR）
- 实现CTC（联结主义时间分类）损失函数，用于序列到序列的OCR训练
- 无需训练，使用PaddleOCR或EasyOCR进行生产级文档解析
- 区分OCR、布局解析和文档理解——并为每个任务选择合适的工具

## 问题

满篇文字的图像无处不在：收据、发票、身份证件、扫描书籍、表格、白板、路标、截图。从中提取结构化数据——不仅仅是字符，还包括“这是总金额”——是最具价值的应用视觉问题之一。

该领域分为三个技能层次：

1. **OCR本身**：将像素转换为文本。
2. **布局解析**：将OCR输出分组为区域（标题、正文、表格、页眉）。
3. **文档理解**：从布局中提取结构化字段（“invoice_total = 42.50美元”）。

每一层都有经典和现代方法，而“我想从图像中获取文本”与“我需要这张收据中的总金额”之间的差距比大多数团队意识到的要大。

## 核心概念

### 经典流水线

```mermaid
flowchart LR
    IMG["Image"] --> DET["Text detection<br/>(DB, EAST, CRAFT)"]
    DET --> BOX["Word/line<br/>bounding boxes"]
    BOX --> CROP["Crop each region"]
    CROP --> REC["Recognition<br/>(CRNN + CTC)"]
    REC --> TXT["Text strings"]
    TXT --> LAY["Layout<br/>ordering"]
    LAY --> OUT["Reading-order text"]

    style DET fill:#dbeafe,stroke:#2563eb
    style REC fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

- **文本检测** 生成每行或每词的四边形。
- **识别** 将每个区域裁剪为固定高度，运行CNN + BiLSTM + CTC生成字符序列。
- **布局** 重建阅读顺序（拉丁语系从上到下、从左到右；阿拉伯语、日语则不同）。

### 一句话概括CTC

OCR识别从固定尺寸的特征图生成可变长度的序列。CTC（Graves等人，2006）允许在不进行字符级对齐的情况下训练。模型在每个时间步输出（词汇表+空白）上的概率分布；CTC损失通过合并所有能规约为目标文本（合并重复并移除空白）的对齐来边缘化。

```
raw output: "h h h _ _ e e l l _ l l o _ _"
after merge repeats and remove blanks: "hello"
```

CTC是CRNN在2015年成功并至今仍训练大多数生产级OCR模型的原因。

### 现代端到端模型

- **Donut**（Kim等人，2022）——ViT编码器+文本解码器；读取图像并直接输出JSON。无需文本检测器，无需布局模块。
- **TrOCR**——用于行级OCR的ViT+变换器解码器。
- **Qwen-VL-OCR / InternVL**——全视觉语言模型，针对OCR任务微调；2026年在复杂文档上最佳准确率。
- **PaddleOCR**——经典DB+CRNN流水线，封装为成熟的生产包；仍然是开源的主力工具。

端到端模型需要更多数据和计算，但避免了多阶段流水线的错误累积。

### 布局解析

对于结构化文档，运行布局检测器（LayoutLMv3、DocLayNet），为每个区域标注：标题、段落、图形、表格、脚注。阅读顺序随后变为“按布局顺序遍历区域，拼接”。

对于表单，使用**键值提取**模型（Donut用于视觉丰富文档，LayoutLMv3用于普通扫描件）。它们接收图像+检测到的文本+位置，预测结构化的键值对。

### 评估指标

- **字符错误率（CER）**——莱文斯坦距离/参考文本长度。越小越好。生产目标：在清晰扫描件上低于2%。
- **词错误率（WER）**——在词级别相同。
- **结构化字段上的F1**——用于键值任务；衡量`{invoice_total: 42.50}`是否正确出现。
- **JSON上的编辑距离**——用于端到端文档解析；Donut论文引入了归一化树编辑距离。

## 动手构建

### 第一步：CTC损失+贪婪解码器

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    log_probs:      (T, N, C) log-softmax over vocab including blank at index 0
    targets:        (N, S) int targets (no blanks)
    input_lengths:  (N,) per-sample time steps used
    target_lengths: (N,) per-sample target length
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def greedy_ctc_decode(log_probs, blank=0):
    """
    log_probs: (T, N, C) log-softmax
    returns: list of index sequences (blanks removed, repeats merged)
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out
```

`F.ctc_loss`在可用时使用高效的CuDNN实现。贪婪解码器比波束搜索更简单，且通常CER优于波束搜索约1%。

### 第二步：微型CRNN识别器

最小化CNN+BiLSTM用于行级OCR。

```python
class TinyCRNN(nn.Module):
    def __init__(self, vocab_size=40, hidden=128, feat=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, 1, 1), nn.BatchNorm2d(feat * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(feat * 4, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(feat * 4, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (N, 1, H, W)
        f = self.cnn(x)                # (N, C, H', W')
        f = f.mean(dim=2).transpose(1, 2)  # (N, W', C)
        h, _ = self.rnn(f)
        return F.log_softmax(self.head(h).transpose(0, 1), dim=-1)  # (W', N, vocab)
```

固定高度输入（CNN将高度池化至1）。宽度作为CTC的时间维度。

### 第三步：合成OCR

生成黑底白字的数字字符串，用于端到端冒烟测试。

```python
import numpy as np

def synthetic_line(text, height=32, char_width=16):
    W = char_width * len(text)
    img = np.ones((height, W), dtype=np.float32)
    for i, c in enumerate(text):
        x = i * char_width
        shade = 0.0 if c.isalnum() else 0.5
        img[6:height - 6, x + 2:x + char_width - 2] = shade
    return img


def build_batch(strings, vocab):
    H = 32
    W = 16 * max(len(s) for s in strings)
    imgs = np.ones((len(strings), 1, H, W), dtype=np.float32)
    target_lengths = []
    targets = []
    for i, s in enumerate(strings):
        imgs[i, 0, :, :16 * len(s)] = synthetic_line(s)
        ids = [vocab.index(c) for c in s]
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.from_numpy(imgs), torch.tensor(targets), torch.tensor(target_lengths)


vocab = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
imgs, targets, lengths = build_batch(["hello", "world"], vocab)
print(f"images: {imgs.shape}   targets: {targets.shape}   lengths: {lengths.tolist()}")
```

真实的OCR数据集会添加字体、噪声、旋转、模糊和颜色。上述流水线完全相同。

### 第四步：训练草图

```python
model = TinyCRNN(vocab_size=len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    strings = ["abc" + str(step % 10)] * 4 + ["xyz" + str((step + 1) % 10)] * 4
    imgs, targets, target_lens = build_batch(strings, vocab)
    log_probs = model(imgs)  # (W', 8, vocab)
    input_lens = torch.full((8,), log_probs.size(0), dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens, blank=0)
    opt.zero_grad(); loss.backward(); opt.step()
```

在这个简单的合成数据上，损失应在200步内从约3下降到约0.2。

## 使用它

三种生产路径：

- **PaddleOCR** — 成熟、快速、多语言。一行使用：`paddleocr.PaddleOCR(lang="en").ocr(image_path)`。
- **EasyOCR** — 原生Python、多语言、PyTorch主干。
- **Tesseract** — 经典；在模型难以处理老旧扫描文档时仍有用。

对于端到端文档解析，使用Donut或VLM：

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
```

对于具有可重复结构的收据、发票和表格，微调Donut。对于任意文档或需要推理的OCR，像Qwen-VL-OCR这样的VLM是当前默认选择。

## 发布

本課(lesson)产出：

- `outputs/prompt-ocr-stack-picker.md` — 根据文档类型、语言和结构选择Tesseract / PaddleOCR / Donut / VLM-OCR的提示。
- `outputs/prompt-ocr-stack-picker.md` — 从头编写贪婪和波束搜索CTC解码器（包括长度归一化）的技能。

## 练习

1. **(简单)** 在5位随机数字字符串上训练TinyCRNN 500步。报告在保留集上的CER。
2. **(中等)** 用波束搜索（beam_width=5）替换贪婪解码。报告CER差值。波束搜索在哪些输入上胜出？
3. **(困难)** 在一组20张收据上使用PaddleOCR，提取行项目，并计算{item_name, price}对与手工标注真实值的F1分数。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|----------------------|
|  OCR  |  "来自像素的文本"  |  将图像区域转化为字符序列  |
|  CTC  |  "无需对齐的损失"  |  无需每时间步标签即可训练序列模型的损失；通过对齐进行边缘化  |
|  CRNN  |  "经典OCR模型"  |  卷积特征提取器 + BiLSTM + CTC；2015基线模型，仍在生产中使用  |
|  Donut  |  "端到端OCR"  |  ViT编码器 + 文本解码器；直接从图像输出JSON  |
|  布局解析  |  "查找区域"  |  检测并标注文档中的标题/表格/图形/段落区域  |
|  阅读顺序  |  "文本序列"  |  将识别出的区域排序成句子；对于拉丁文字简单，对于混合布局复杂  |
|  CER / WER  |  "错误率"  |  字符或词粒度的莱文斯坦距离/参考长度  |
|  VLM-OCR  |  "会读的LLM"  |  为OCR任务训练或提示的视觉语言模型；当前在复杂文档上的SOTA  |

## 延伸阅读

- [CRNN (Shi et al., 2015)](https://arxiv.org/abs/1507.05717) — 原始的CNN+RNN+CTC架构
- [CRNN (Shi et al., 2015)](https://arxiv.org/abs/1507.05717) — 原始的CTC论文；充满了算法思想
- [CRNN (Shi et al., 2015)](https://arxiv.org/abs/1507.05717) — 无OCR文档理解Transformer
- [CRNN (Shi et al., 2015)](https://arxiv.org/abs/1507.05717) — 开源生产级OCR栈
