# 视觉变换器(Vision Transformers, ViT)

> 一张图像是一个补丁网格。一个句子是一个词元网格。同样的Transformer处理两者。

**类型：** 构建
**语言：** Python
**前提条件：** 阶段7 · 05（完整Transformer）、阶段4 · 03（CNN）、阶段4 · 14（视觉Transformer简介）
**时间：** ≈45分钟

## 问题

2020年之前，计算机视觉意味着卷积。ImageNet、COCO和检测基准上的所有SOTA都使用CNN骨干网络。Transformer是为语言设计的。

Dosovitskiy等人（2020）——《一张图像等于16x16个词》——表明你可以完全丢弃卷积。将图像切成固定大小的补丁，将每个补丁线性投影到嵌入，将序列送入一个普通的Transformer编码器。在足够大的规模下（ImageNet-21k预训练或更大），ViT匹配或超越了基于ResNet的模型。

ViT是2026年更广泛模式的开端：一种架构，多种模态。Whisper将音频词元化。ViT将图像词元化。机器人的动作词元。视频的像素词元。Transformer不在乎——给它一个序列，它就学习。

到2026年，ViT及其后代（DeiT、Swin、DINOv2、ViT-22B、SAM 3）占据了大部分视觉领域。CNN仍在边缘设备和延迟敏感任务上胜出。其他所有任务在堆栈的某处都有一个ViT。

## 核心概念

![Image → patches → tokens → transformer](../assets/vit.svg)

### 步骤1——分块

将一张`H × W × C`图像分成一个`N × (P·P·C)`的扁平补丁序列。典型设置：`224 × 224`图像，`16 × 16`补丁→196个补丁，每个补丁768个值。

```
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

补丁大小是一个杠杆。较小的补丁=更多的词元，更好的分辨率，二次注意力代价。较大的补丁=更粗糙，更便宜。

### 步骤2——线性嵌入

一个单一的学习矩阵将每个扁平补丁投影到`d_model`。等价于卷积核大小`P`和步幅`P`。在PyTorch中，这实际上就是`nn.Conv2d(C, d_model, kernel_size=P, stride=P)`——一个两行的实现。

### 步骤3——前置`[CLS]`词元，添加位置嵌入

- 前置一个可学习的`[CLS]`词元。其最终隐藏状态是用于分类的图像表示。
- 添加可学习的位置嵌入（原始ViT）或正弦二维嵌入（后来的变体）。
- 在2024年及以后，RoPE扩展到二维用于位置，有时没有显式嵌入。

### 步骤4——标准Transformer编码器

堆叠L个`LayerNorm → Self-Attention → + → LayerNorm → MLP → +`块。与BERT相同。没有特定于视觉的层。这是本文的教学要点。

### 步骤5——头部

对于分类：取`[CLS]`隐藏状态→线性→softmax。对于DINOv2或SAM，丢弃`[CLS]`，直接使用补丁嵌入。

### 重要的变体

|  模型  |  年份  |  变化  |
|-------|------|--------|
|  ViT  |  2020  |  原始版本。固定补丁大小，完全全局注意力。  |
|  DeiT  |  2021  |  蒸馏；仅在ImageNet-1k上可训练。  |
|  Swin  |  2021  |  带移位窗口的分层结构。固定的次二次代价。  |
|  DINOv2  |  2023  |  自监督（无标签）。最好的通用视觉特征。  |
|  ViT-22B  |  2023  |  220亿参数；缩放法则适用。  |
|  SigLIP  |  2023  |  ViT + 语言配对，sigmoid对比损失。  |
|  SAM 3  |  2025  |  分割一切；ViT-大型+可提示掩码解码器。  |

### 为什么花费了时间

ViT需要*大量*数据才能匹敌CNN，因为它没有CNN的归纳偏置（平移不变性、局部性）。如果没有超过1亿张标注图像或强大的自监督预训练，在相同计算量下CNN仍然胜出。DeiT在2021年通过蒸馏技巧解决了这一问题；DINOv2在2023年通过自监督彻底解决了。

## 动手构建

参见`code/main.py`。纯标准库的分块+线性嵌入+合理性检查。没有训练——任何实际规模的ViT都需要PyTorch和数小时的GPU时间。

### 步骤1：假图像

一个24×24的RGB图像，表示为行列表，每行为`(R, G, B)`元组。我们使用6×6的补丁(patch) → 16个补丁，每个补丁为108维嵌入向量。

### 第2步：分块(Patchify)

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

光栅顺序：按行主序(row-major)遍历网格。所有ViT都采用这种顺序。

### 第3步：线性嵌入(Linear Embed)

将每个扁平化补丁乘以一个随机的`(patch_flat_size, d_model)`矩阵。验证在预先添加`[CLS]`后输出形状为`(N_patches + 1, d_model)`。

### 第4步：为实际的ViT计算参数数量

打印ViT-Base的参数数量：12层，12个注意力头，d=768，补丁大小=16。与ResNet-50（约25M）进行比较。ViT-Base约为86M。ViT-Large约为307M。ViT-Huge约为632M。

## 使用它

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation
```

**DINOv2嵌入是2026年图像特征的默认选择。**冻结主干网络(backbone)，训练一个小型头部。适用于分类、检索、检测、字幕生成。Meta的DINOv2检查点(checkpoint)在所有非文本视觉任务上优于CLIP。

**补丁大小选择。**小型模型使用16×16（ViT-B/16）。密集预测（分割）使用8×8或14×14（SAM, DINOv2）。非常大的模型使用14×14。

## 发布

参见`outputs/skill-vit-configurator.md`。该技能根据数据集大小、分辨率和计算预算，为新视觉任务选择ViT变体和补丁大小。

## 练习

1. **简单。**运行`code/main.py`。验证补丁数量等于`(H/P) * (W/P)`且扁平化补丁维度等于`P*P*C`。
2. **中等。**实现二维正弦位置嵌入(sinusoidal positional embeddings)——为每个补丁的`code/main.py`和`(H/P) * (W/P)`分别生成两个独立的正弦编码，然后将二者拼接。将它们馈入一个小型PyTorch ViT，并在CIFAR-10上对比使用可学习位置嵌入(learnable positional embeddings)的精度。
3. **困难。**构建一个3层ViT（PyTorch），在1000张MNIST图像上训练，使用4×4补丁。测量测试精度。然后对相同的1000张图像添加DINOv2预训练（简化版：仅训练编码器从掩蔽补丁预测补丁嵌入）。精度是否提升？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
| 补丁(Patch)  |  "视觉变换器词元(vision-transformer token)"  |  图像中`P × P × C`区域的像素值的扁平向量。 |
| 分块(Patchify)  |  "切割+展平(Chop + flatten)"  |  将图像切割成不重叠的补丁，将每个补丁展平为向量。 |
| `[CLS]`词元(token)  |  "图像摘要(The image summary)"  |  预先添加的可学习词元；其最终嵌入是图像表示。 |
| 归纳偏置(Inductive bias)  |  "模型假设的内容(What the model assumes)"  |  ViT比CNN的先验(prior)更少；需要更多数据来弥补差距。 |
| DINOv2  |  "自监督ViT(Self-supervised ViT)"  |  使用图像增强+动量教师(momentum teacher)在无标签情况下训练。2026年最佳通用图像特征。 |
| SigLIP  |  "CLIP的继承者(CLIP's successor)"  |  ViT + 文本编码器，使用sigmoid对比损失(sigmoid contrastive loss)训练；在相同计算量下优于CLIP。 |
| Swin  |  "窗口化ViT(Windowed ViT)"  |  分层ViT，采用局部注意力+移位窗口；亚二次复杂度(sub-quadratic)。 |
| 注册词元(Register tokens)  |  "2023年技巧"  |  少量额外的可学习词元，用于吸收注意力汇聚(attention sinks)；提升DINOv2特征。 |

## 延伸阅读

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT论文。
- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — DeiT。
- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — Swin。
- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — DINOv2。
- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — 针对DINOv2的注册词元修复。
