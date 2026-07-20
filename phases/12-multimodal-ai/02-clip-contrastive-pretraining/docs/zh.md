# CLIP与对比视觉-语言预训练

> OpenAI的CLIP（2021）证明了一个足以驱动未来五年的简单想法：仅使用嘈杂的网络图像-标题对和对比损失，将图像编码器和文本编码器对齐到同一向量空间。零监督标签。4亿对。得到的嵌入空间可以进行零样本分类、图像-文本检索，并作为每个2026年视觉语言模型的视觉塔。SigLIP 2（2025）用sigmoid取代了softmax，并以更低的成本超越了CLIP。本课将讲解从InfoNCE到sigmoid成对损失的数学原理，并在stdlib Python中构建训练步骤。

**类型：** 构建
**语言：** Python（stdlib，InfoNCE + sigmoid损失实现）
**先修知识：** 第12阶段·01（ViT块），第7阶段（Transformer）
**时间：** 约180分钟

## 学习目标

- 从互信息推导InfoNCE损失，并实现一个数值稳定的向量化版本。
- 解释为什么sigmoid成对损失（SigLIP）能够扩展到批次32768以上，而无需softmax所需的all-gather开销。
- 通过构建文本模板（`a photo of a {class}`）并取余弦相似度的argmax，运行零样本ImageNet分类。
- 命名CLIP/SigLIP预训练提供的四个杠杆：批次大小、温度、提示模板、数据质量。

## 问题

CLIP之前的视觉是监督学习。收集带标签的数据集（ImageNet：120万张图片，1000个类别），训练一个CNN，然后发布。标签昂贵，标签偏向于标注者能够达成一致的内容，并且标签不能在没有微调的情况下迁移到新任务。

图像-标题网络上有超过十亿个松散标签的对可供免费使用。一张金毛犬的图片，其alt文本“my dog Max in the park”就携带了监督信号——文本描述了图像。问题在于：你能将其转化为有用的训练吗？

CLIP的回答是：将图像-标题对视为一个匹配任务。给定一批N张图像和N个标题，学习将每张图像与其自身标题匹配，排除N-1个干扰项。监督信号是“这两者属于一起；这N-1个不属于。”没有类别标签。没有人工标注。只有对比损失。

得到的嵌入空间实现了超出CLIP训练目标的功能。ImageNet零样本有效，因为“一张猫的照片”嵌入到从未明确标记为猫的猫图片附近。这正是催生每个2026年视觉语言模型的赌注。

## 核心概念

### 双编码器

CLIP有两个塔：

- 图像编码器`f`：ViT或ResNet，每张图像输出一个D维向量。
- 文本编码器`f`：小型Transformer，每个标题输出一个D维向量。

两个塔都将输出归一化到单位长度。由于都是单位范数，相似度是`cos(f(x), g(y)) = f(x)^T g(y)`。

对于一批N个（图像，标题）对，构建形状为`(N, N)`的相似度矩阵`S`：

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

其中`tau`是一个可学习的温度（CLIP初始化为0.07；在log空间中学习）。

### InfoNCE损失

CLIP在行和列上使用对称的交叉熵：

```
loss_i2t = CE(S, labels=identity)     # each image's positive is its own caption
loss_t2i = CE(S^T, labels=identity)   # each caption's positive is its own image
loss = (loss_i2t + loss_t2i) / 2
```

这就是InfoNCE。交叉熵中的softmax迫使每张图像与其标题的匹配程度超过批中的其他所有标题。“负样本”是批中的所有其他项。更大的批次=更多的负样本=更强的信号。CLIP在批次32k下训练；规模很重要。

### 温度

`tau`控制softmax的锐度。低τ→尖锐分布，难负样本挖掘效果。高τ→平滑，所有样本都有贡献。CLIP学习log(1/τ)，进行裁剪以防止坍塌。SigLIP 2固定了初始τ，并使用可学习的偏置代替。

### 为什么sigmoid扩展性更好（SigLIP）

Softmax需要整个相似度矩阵同步。在分布式训练中，你必须将所有嵌入全部收集到每个副本，然后进行softmax。这在通信量上与全局规模成二次方关系。

SigLIP用逐元素的sigmoid替换softmax：对于每一对`(i, j)`，损失是一个二分类问题——“这些是匹配对吗？”正类标签是对角线元素，其他所有都是负类。损失为：

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1` 如果 `i == j`，否则为0。每对的损失是独立的。无需all-gather。每个GPU计算其本地块并求和。SigLIP 2可以低成本扩展到批次32k-512k，而CLIP则需要成比例的更多通信。

### 零样本分类

给定N个类别名称，为每个类别构建一个文本模板：

```
"a photo of a {class}"
```

用文本编码器嵌入每个模板。用图像编码器嵌入图像。Argmax余弦相似度=预测类别。不对目标类别进行训练。

提示模板很重要。CLIP的原始论文每类使用了80个模板（普通、艺术、照片、绘画等），并对嵌入取平均。提升了3个ImageNet点。现代用法通常选择一个或两个模板。

### 线性探测和微调

零样本是一个基线。线性探测（在冻结的CLIP特征之上为目标类别训练一个线性层）在域内任务上优于零样本。完全微调在域内优于线性探测，但可能损害零样本迁移。三种模式各有三种权衡。

### SigLIP 2：NaFlex和密集特征

SigLIP 2（2025）新增：
- NaFlex：单一模型处理可变的宽高比和分辨率。
- 为分割和深度估计提供更好的密集特征，旨在作为视觉语言模型(VLMs)中的冻结主干网络使用。
- 多语言：在100多种语言上训练，而CLIP仅支持英语。
- 10亿参数规模，而CLIP的最大规模为4亿参数。

在2026年的开放视觉语言模型中，SigLIP 2 SO400m/14是默认的视觉塔。CLIP仍然是纯图像-文本检索的默认选择，当特定的LAION-2B训练分布与你的查询模式匹配时。

### ALIGN, BASIC, OpenCLIP, EVA-CLIP

ALIGN（Google，2021）：与CLIP相同的理念，18亿对规模，90%噪声。证明了噪声数据可以扩展。OpenCLIP（LAION）：在LAION-400M/2B上对CLIP的开源复现，多种规模，是首选的开源检查点。EVA-CLIP：从掩码图像建模初始化；是视觉语言模型的强大主干。BASIC：Google的CLIP+ALIGN混合体。同属一个家族，不同的数据和调优。

### 零样本天花板

CLIP类模型在ImageNet零样本上的表现上限约为76%（CLIP-G, OpenCLIP-G）。超过这个水平需要更大的数据（SigLIP 2达到80%以上）或架构改变（监督头，更多参数）。基准测试正在饱和；真正的价值在于下游视觉语言模型所使用的嵌入空间。

```figure
multimodal-fusion
```

## 使用它

`code/main.py` 实现：

1. 一个玩具双编码器（基于哈希的图像特征，文本字符特征），让你无需numpy就能看到InfoNCE的形状。
2. 纯Python实现的InfoNCE损失（通过log-sum-exp实现数值稳定性）。
3. 用于比较的Sigmoid成对损失。
4. 一个零样本分类例程：计算与一组文本提示的余弦相似度，取argmax进行预测。

运行它并观察损失曲线。绝对值是玩具；形状与真实CLIP训练器产生的相符。

## 发布

本课生成`outputs/skill-clip-zero-shot.md`。给定一组图像（通过路径）和一个目标类别列表，它使用CLIP模板构建文本提示，用指定的检查点（例如`openai/clip-vit-large-patch14`）对两侧进行嵌入，并返回top-1/top-5预测及其相似度分数。该技能拒绝声称提示列表中不存在的类别。

## 练习

1. 手动实现一批4对样本的InfoNCE。构造4x4相似度矩阵，运行softmax，提取对角线，计算交叉熵。用这个手算结果验证你的Python实现。

2. SigLIP除了温度参数外还使用了一个偏置参数`b`：`S'[i,j] = S[i,j]/tau + b`。当批次中存在严重的类别不平衡（每行负样本远多于正样本）时，`b`起到什么作用？阅读SigLIP第3节（arXiv:2303.15343）。

3. 构建一个猫狗零样本分类器。尝试两个提示模板：`a photo of a {class}`和`a picture of a {class}`。在100张测试图像上测量准确率。模板集成是否优于单个模板？

4. 计算在32k批次下512个GPU运行时，softmax InfoNCE与sigmoid成对损失的通信成本。哪个是O(N)缩放，哪个是O(N^2)？引用SigLIP第4节。

5. 阅读OpenCLIP缩放定律论文（arXiv:2212.07143，Cherti等人）。从图中重现他们关于数据缩放的结论：在固定模型大小下，ImageNet零样本准确率与训练数据量之间的对数线性关系是什么？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  InfoNCE  |  "对比损失"  |  批次相似度矩阵上的交叉熵；每个样本的正样本是其配对项，负样本是其他所有项  |
|  Sigmoid损失  |  "SigLIP损失"  |  每对二元交叉熵；无softmax，无全收集，在分布式训练中扩展成本低  |
|  温度  |  "tau"  |  在softmax/sigmoid之前缩放logits的标量；控制分布的尖锐程度  |
|  零样本  |  "无微调分类"  |  使用文本提示构建类别嵌入并通过余弦相似度分类；不对目标类别进行训练  |
|  提示模板  |  "一张...的照片"  |  围绕类别名称的文本支架；对零样本准确率有1-5个点的影响  |
|  双编码器  |  "双塔"  |  一个图像编码器+一个文本编码器，在共享的D维空间中输出  |
|  难负样本  |  "强干扰项"  |  与正样本足够相似，模型必须努力才能区分它们的负样本  |
|  线性探测  |  "冻结+一层"  |  仅在冻结特征之上训练线性分类器；衡量特征质量  |
|  NaFlex  |  "原生灵活分辨率"  |  SigLIP 2能够以任何宽高比和分辨率输入图像而无需调整大小  |
|  温度缩放  |  "对数参数化的tau"  |  CLIP对`log(1/tau)`进行参数化以确保梯度行为；进行裁剪以防止tau接近零  |

## 延伸阅读

- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — CLIP论文。
- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — SigLIP。
- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — 多语言+NaFlex。
- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — 使用噪声网络数据进行扩展。
- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — OpenCLIP缩放定律。
