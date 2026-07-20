# 从头构建Transformer——顶石项目

> 十三课。一个模型。无捷径。

**类型：**构建
**语言：**Python
**前置条件：**阶段7·第01至13课。不要跳过。
**时间：**约120分钟

## 问题

你已经阅读了每一篇论文。你已经实现了注意力机制、多头拆分、位置编码、编码器和解码器块、BERT和GPT损失、MoE、KV缓存。现在让它们协同工作在一个真实任务上。

顶石项目：在字符级语言建模任务上端到端训练一个小的仅解码器Transformer。它读取莎士比亚作品。它生成新的莎士比亚作品。它足够小，可以在笔记本上10分钟内完成训练。它足够正确，换成更大的数据集和更长的训练就能得到真正的语言模型。

这是课程的"nanoGPT"。它并非原创——Karpathy在2023年的nanoGPT教程是每个学生至少编写一次的参考实现。我们借鉴其框架并根据所学内容重新改造。

## 核心概念

![Transformer-from-scratch block diagram](../assets/capstone.svg)

架构注释：

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### 我们提供的内容

- `GPTConfig` — 统一配置所有超参数的地方。
- `GPTConfig` — 因果、批处理，带有可选的Flash样式路径（PyTorch的`MultiHeadAttention`）。
- `GPTConfig` — 现代FFN。
- `GPTConfig` — 预归一化，残差包裹的注意力+FFN。
- `GPTConfig` — 嵌入、堆叠块、LM头、generate()。
- 使用AdamW、余弦学习率、梯度裁剪的训练循环。
- 基于莎士比亚文本的字符级分词器。

### 我们不提供的内容

- RoPE——在第04课中概念性地实现。这里为了简单起见，我们使用可学习的位置嵌入。练习要求你替换为RoPE。
- 生成时的KV缓存——每个生成步骤重新计算整个前缀的注意力。较慢但更简单。练习要求你添加KV缓存。
- Flash Attention——PyTorch 2.0+ 在输入匹配时自动调度；我们使用`F.scaled_dot_product_attention`。
- MoE——每块单个FFN。你在第11课中看到了MoE。

### 目标指标

在Mac M2笔记本上，一个4层、4头、d_model=128的GPT在`tinyshakespeare.txt`上训练2000步：

- 训练损失在约6分钟内从~4.2（随机）收敛到~1.5。
- 采样输出看起来像莎士比亚风格：古语、换行、诸如"ROMEO:"的专有名称出现。
- 验证损失（保留的最后10%文本）紧密跟踪训练损失；在此大小/预算下没有过拟合。

## 动手构建

本课使用PyTorch。安装`torch`（CPU版本即可）。参见`code/main.py`。脚本处理：

- 如果缺失则下载`tinyshakespeare.txt`（或读取本地副本）。
- 字节级字符分词器。
- 训练/验证分割为90/10。
- 在支持的硬件上使用bf16自动混合精度的训练循环。
- 训练完成后进行采样。

### 步骤1：数据

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65个独特字符。微小词汇量。适合4字节的vocab_size。没有BPE，没有分词器麻烦。

### 步骤2：模型

参见`code/main.py`。该块是来自第05课的标准实现——预归一化、RMSNorm、SwiGLU、因果MHA。4/4/128的参数数量：约80万。

### 步骤3：训练循环

获取一个随机批次的长度为256的token窗口。前向传播。移位一位交叉熵。反向传播。AdamW步骤。记录。重复。

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### 步骤4：采样

给定一个提示，重复前向传播，从top-p logits中采样，追加，继续。500个token后停止。

### 步骤5：读取输出

2000步后：

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

不是莎士比亚，但具有莎士比亚风格。约80万参数和笔记本上6分钟的明显胜利。

## 使用它

这个顶石项目是一个参考架构。三个扩展使其成为真实产品：

1. **替换分词器。**使用BPE（例如`tiktoken.get_encoding("cl100k_base")`）。词汇量从65跳升到约5万。模型容量需要扩大以补偿。
2. **在更大语料库上训练。**使用`tiktoken.get_encoding("cl100k_base")`或`OpenWebText`（HuggingFace）。在单个A100上训练125M参数的GPT处理100亿token大约需要24小时。
3. **添加RoPE + KV缓存 + Flash Attention。**下面的练习会逐步引导你。

最终得到一个125M参数的GPT，能够生成流利的英语。不是前沿模型。但相同的代码路径——只是规模更大——正是Karpathy、EleutherAI和Allen Institute在2026年用来训练研究检查点的方法。

## 发布

参见`outputs/skill-transformer-review.md`。该技能回顾了一个从头实现的Transformer，验证其在全部13个先前课程中的正确性。

## 练习

1. **简单.** 运行`code/main.py`。验证你训练模型最后一步的验证损失低于2.0。将`max_steps`从2000改为5000——验证损失是否持续改善？
2. **中等.** 用RoPE替换学习的位置嵌入。在`code/main.py`内部对Q和K施加旋转。训练并验证验证损失至少同样低。
3. **中等.** 在采样循环中实现KV缓存。生成500个token，分别使用和不使用缓存。在笔记本电脑上，实际耗时应提升5-20倍。
4. **困难.** 为模型添加第二个头，用于预测下一个+1的token（MTP——来自DeepSeek-V3的多token预测）。联合训练。是否有帮助？
5. **困难.** 将每个块的单个FFN替换为4专家的MoE。路由 + top-2路由。观察在匹配有效参数时验证损失如何变化。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  nanoGPT  |  "Karpathy的教程仓库"  |  最小的仅解码器Transformer训练代码，约300行代码；权威参考。  |
|  tinyshakespeare  |  "标准玩具语料库"  |  约1.1 MB文本；自2015年以来的每个字符级LM教程都使用它。  |
|  绑定嵌入  |  "共享输入/输出矩阵"  |  LM头权重 = 词嵌入矩阵的转置；节省参数，提高质量。  |
|  bf16自动混合精度  |  "训练精度技巧"  |  在bf16中运行前向/反向，优化器状态保持在fp32；自2021年以来的标准做法。  |
|  梯度裁剪  |  "阻止尖峰"  |  将全局梯度范数限制在1.0；防止训练崩溃。  |
|  余弦学习率调度  |  "2020年以后的默认设置"  |  学习率线性上升（预热），然后余弦形状衰减到峰值的10%。  |
|  MFU  |  "模型FLOP利用率"  |  实际FLOP / 理论峰值；2026年，40%密集，30% MoE是强的。  |
|  验证损失  |  "留出损失"  |  模型从未见过的数据上的交叉熵；过拟合检测器。  |

## 延伸阅读

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/)——经典的带注释实现。
