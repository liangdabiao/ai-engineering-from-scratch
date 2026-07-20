# LLaVA 与视觉指令微调

> LLaVA（2023年4月）是地球上被复制最多的多模态架构。它用2层MLP替换了BLIP-2的Q-Former，用简单的标记拼接替换了Flamingo的门控交叉注意力，并在GPT-4从纯文本描述生成的15.8万个视觉指令轮次上训练。在2023年至2026年间构建VLM的实践者都在构建LLaVA的变体。LLaVA-1.5添加了AnyRes。LLaVA-NeXT提高了分辨率。LLaVA-OneVision将图像、多图像和视频统一到一个配方中。本节课解读该配方，实现投影器(Projector)，并解释为什么“简单胜出”。

**类型：** 构建
**语言：** Python（标准库，投影器 + 指令模板构建器）
**前置知识：** 第12阶段·02（CLIP），第11阶段（LLM工程——指令微调）
**时间：** 约180分钟

## 学习目标

- 构建一个2层MLP投影器，将ViT图像块嵌入（维度1024）映射到LLM的嵌入维度（维度4096）。
- 走一遍LLaVA的两阶段配方：(1) 在55.8万对描述对上投影器对齐，(2) 在GPT-4生成的15.8万轮次上进行视觉指令微调。
- 构建LLaVA格式的提示，包含图像标记占位符、系统提示和用户/助手轮次。
- 解释为什么社区从Q-Former转向MLP，尽管Q-Former在标记预算上占优。

## 问题

BLIP-2的Q-Former（第12.03课）将图像压缩为32个标记。干净、高效，在基准测试中表现良好。但它有两个问题。

首先，Q-Former是可训练的，但其损失并非最终任务。第一阶段训练ITC+ITM+ITG。第二阶段训练LM损失。查询学习一些中间表示，然后LLM必须解码这些表示。信息在瓶颈中丢失。

其次，Q-Former有1.88亿参数，且在LLaVA的2023年规模下，必须与目标LLM协同设计。更换LLM，重新训练Q-Former。更换视觉编码器，重新训练。每种组合都是一个独立的研发项目。

LLaVA的答案简单得令人尴尬：取ViT的576个图像块标记，通过一个2层MLP（`1024 → 4096 → 4096`）传递每个标记，然后将所有576个标记送入LLM的输入序列。没有瓶颈。没有基于奇怪目标的第一阶段预训练。仅在一个直接的LM损失上训练MLP。

数据从何而来？LLaVA的第二个洞见：使用GPT-4（仅文本）生成指令数据。将图像的COCO描述和边界框数据喂给GPT-4，让其生成对话、描述和复杂的推理问题。免费获得15.8万轮指令-回答。无需人工标注。

结果：一个在8个A100上运行一天、在MMMU上击败Flamingo、并发布社区可扩展的开放检查点的VLM。到2023年底，它已衍生出50多个分支。

## 核心概念

### 架构

LLaVA-1.5（13B参数）：
- 视觉编码器：CLIP ViT-L/14 @ 336（第一阶段冻结，第二阶段可选解冻）。
- 投影器：2层MLP，GELU激活函数，`1024 → 4096 → 4096`。
- LLM：Vicuna-13B（后来为Llama-3.1-8B）。

图像+文本提示的前向传播：

```
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

图像占用LLM上下文的576个标记。在2048上下文中，剩下1472个标记用于文本。在32k上下文中，这几乎可以忽略不计。

### 第一阶段：投影器对齐

冻结ViT。冻结LLM。仅训练2层MLP。数据集：55.8万对图像-描述对（LAION-CC-SBU）。损失：基于投影的图像标记的条件语言建模损失。

在批次大小为128的单个周期内，数小时即可完成。投影器学习将ViT空间映射到LLM空间。无需任务特定监督。

### 第二阶段：视觉指令微调

解冻投影器（仍可训练）。解冻LLM（通常完全解冻，有时使用LoRA）。在15.8万轮视觉指令上训练。

指令数据的技巧在于。Liu等人通过以下方式生成：
1. 取一张COCO图像。
2. 提取文本描述（5个人工描述+边界框列表）。
3. 发送给GPT-4，使用三种提示模板：
   - 对话：“生成用户和助手关于这张图像的来回对话。”
   - 详细描述：“给出丰富、详细的图像描述。”
   - 复杂推理：“提出一个需要对图像进行推理的问题，然后回答它。”
4. 将GPT-4的输出解析为（指令，回答）对。

这些操作均未直接接触图像——仅使用文本描述。GPT-4会幻觉出看似合理的图像内容。有些噪声，但行得通：15.8万轮次足以解锁对话能力。

### 为什么社区复制了这种方法

- 无需调整第一阶段特定损失。全程使用LM损失。
- 投影器数小时即可训练完成，而非数天。
- LLM可以互换（LLaVA-Llama2、LLaVA-Mistral、LLaVA-Llama3），只需重新训练投影器。
- 视觉指令数据管道使用GPT-4，并且为新的领域重新生成成本低廉。

### LLaVA-1.5 与 LLaVA-NeXT

LLaVA-1.5（2023年10月）添加了：
- 学术任务数据（VQA、OKVQA、RefCOCO）混入指令微调。
- 更好的系统提示。
- 2048 → 32k上下文。

LLaVA-NeXT（2024年1月）添加了：
- AnyRes：将高分辨率图像拆分为2x2或1x3的336x336裁剪块网格，外加一张全局低分辨率缩略图。每个裁剪块产生576个标记；每张图像总共约2880个视觉标记。OCR和图表任务性能跃升。
- 更好的指令数据混合，使用ShareGPT4V（高质量GPT-4V描述）。
- 更强的基座LLM（Mistral-7B、Yi-34B）。

### LLaVA-OneVision

第12.08课深入讲解OneVision。简短版：同一投影器，但训练课程涵盖单图像、多图像和视频，共用一个可视token预算。

### 与Q-Former的对比

|   |  Q-Former (BLIP-2)  |  MLP (LLaVA)  |
|---|---|---|
|  每张图像的可视token数  |  32  |  576 (基础) 或 2880 (AnyRes)  |
|  可训练参数  |  188M + 语言模型  |  40M + 语言模型  |
|  阶段1损失  |  ITC+ITM+ITG  |  仅语言模型  |
|  大语言模型替换性  |  需重新训练  |  最小重新训练即可替换  |
|  多图像  |  别扭  |  自然（拼接）  |
|  视频  |  别扭  |  自然（逐帧拼接）  |
|  Token预算  |  小  |  大  |

MLP因简单性和token灵活性胜出。Q-Former在token预算上占优。到2023年底，token预算不再是约束（大语言模型上下文增长到32k-128k+），简单性占主导。

### 提示格式

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>`是一个占位符token。在分词前，它被替换为576个可视token（AnyRes下为2880）。分词器会看到比训练时稍长的序列，但大语言模型能处理这种新颖输入，因为阶段1教会了它。

### 参数经济性

LLaVA-1.5-7B分解：
- CLIP ViT-L/14 @ 336: 303M（阶段1冻结，阶段2常解冻）。
- 投影器（2层线性层）：约22M可训练。
- Llama-7B: 7B。
- 总计：7.3B参数。阶段2可训练：完整7B + 22M投影器。

阶段2的训练成本：8xA100上约20小时。这是关键数字——一天，一个节点，可复现。这就是LLaVA传播的原因。

## 使用它

`code/main.py` 实现：

1. 纯Python实现的2层MLP投影器（玩具规模：维度16 → 32 → 32）。
2. 提示构建流程：系统提示 + 用N个投影token替换`<image>` + 用户轮次 + 助手生成占位符。
3. 可视化工具：展示576个可视token在大语言模型上下文中的占比（2k / 32k / 128k上下文消耗百分比）。

## 发布

本课产生`outputs/skill-llava-vibes-eval.md`。给定一个LLaVA系列检查点，运行一个10提示的vibes评估套件（3个描述、3个VQA、2个推理、2个拒绝），并输出一个人类可读的评分卡。不是基准测试；是一个冒烟测试，确认投影器和语言模型连接良好。

## 练习

1. 计算在`1024 → 4096 → 4096`处2层MLP投影器的可训练参数数量。带GELU和偏置，它占LLaVA-13B的多少比例？

2. 为“拒绝”案例构建一个LLaVA提示——图像包含一个私人个体。写出预期的助手回复。为什么LLaVA应该零样本拒绝，需要哪些训练数据来加强拒绝？

3. 阅读LLaVA-NeXT博客的AnyRes部分。计算1344x672图像在AnyRes下的可视token数量。与336x336基础576个token比较。

4. LLaVA阶段1投影器使用语言模型损失在描述上训练。如果跳过阶段1直接进入阶段2（视觉指令微调）会发生什么？引用Prismatic VLMs消融实验（arXiv:2402.07865）来回答。

5. LLaVA-Instruct-150k使用GPT-4和COCO描述生成指令。针对一个新领域（医学X光、卫星图像），描述生成领域指令的四步数据流水线。每一步可能出现什么问题？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  投影器  |  "MLP桥梁"  |  2层MLP带GELU，将ViT维度映射到大语言模型维度  |
|  图像token  |  "<image>占位符"  |  推理前由N个投影可视token替换的提示标记  |
|  视觉指令微调  |  "LLaVA阶段2"  |  在GPT-4生成的（图像，指令，回复）三元组上训练  |
|  阶段1对齐  |  "投影器预训练"  |  冻结ViT和大语言模型，用描述的语言模型损失训练投影器  |
|  AnyRes  |  "多裁剪平铺"  |  将高分辨率图像分割成平铺网格，并拼接每个平铺的可视token  |
|  LLaVA-Instruct  |  "GPT-4生成"  |  从COCO描述+GPT-4合成的15.8万条指令-回复对  |
| 视觉编码器冻结 | "主干网络锁定" | 在阶段1中CLIP权重不更新，有时在阶段2中也不更新 |
| ShareGPT4V | "更好的描述" | 由GPT-4V生成的100万条密集描述，用于更高质量的对齐 |
| 视觉问答(VQA) | "视觉问答(Visual question answering)" | 回答关于图像的自由形式问题的任务 |
| Prismatic VLMs | "设计空间论文" | Karamcheti 2024消融实验，系统测试了投影仪和数据选择 |

## 延伸阅读

- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — LLaVA论文。
- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — LLaVA-1.5。
- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — 密集描述数据集。
- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — 设计空间消融实验。
- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — 统一单图像、多图像、视频。
