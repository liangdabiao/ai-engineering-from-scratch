# GPT — 因果语言建模

> BERT 同时看到两侧。GPT 只看到过去。三角掩码是现代人工智能中最具影响力的一行代码。

**类型:** 构建
**语言:** Python
**先修知识:** 第七阶段 02（自注意力），第七阶段 05（完整Transformer），第七阶段 06（BERT）
**时间:** 约75分钟

## 问题

语言模型回答一个问题：给定前 `t-1` 个词元，词元 `t` 的概率分布是什么？在这个信号——下一词元预测——上训练，你就能得到一个可以逐个词元生成任意文本的模型。

为了在整条序列上并行端到端训练，每个位置的预测必须仅依赖于前面的位置。否则模型会通过查看答案而轻易作弊。

因果掩码实现了这一点。它是一个上三角矩阵，在 softmax 之前将对齐分数加上 `-inf` 值。经过 softmax 后，这些位置变为 0。每个位置只能关注自身及之前的位置。由于你对整个序列只应用一次掩码，一次前向传播就能得到 N 个并行的下一词元预测。

GPT-1（2018）、GPT-2（2019）、GPT-3（2020）、GPT-4（2023）、GPT-5（2024）、Claude、Llama、Qwen、Mistral、DeepSeek、Kimi —— 它们都是仅解码器的因果 Transformer，具有相同的内核循环。只是规模更大、数据更好、RLHF 更优。

## 核心概念

![Causal mask creates a triangular attention matrix](../assets/causal-attention.svg)

### 掩码

给定长度为 `N` 的序列，构建一个 `N × N` 矩阵：

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

在 softmax 之前将 `M` 加到原始注意力分数上。`exp(-inf) = 0`，这样被掩码的位置贡献的权重为零。注意力矩阵的每一行都是仅针对之前位置的概率分布。

实现成本：一次 `torch.tril()` 调用。计算时间：纳秒级。对领域的影响：一切。

### 并行训练，串行推理

训练：对整个 `(N, d_model)` 序列进行一次前向传播，计算 N 个交叉熵损失（每个位置一个），求和，反向传播。沿序列维度并行。这就是 GPT 训练可扩展的原因——你可以在一次 GPU 前向传播中处理一个批次的一百万个词元。

推理：你逐个词元生成。输入 `[t1, t2, t3]`，得到 `t4`。输入 `[t1, t2, t3, t4]`，得到 `t5`。输入 `[t1, t2, t3, t4, t5]`，得到 `t6`。KV 缓存（第12课）保存了 `t1…tn` 的隐藏状态，这样你就不需要每一步重新计算。但推理时的串行深度等于输出长度。这就是自回归的代价，也是解码成为每个大语言模型延迟瓶颈的原因。

### 损失——位移一位

给定词元 `[t1, t2, t3, t4]`：

- 输入：`[t1, t2, t3]`
- 目标：`[t1, t2, t3]`

对于每个位置 `i`，计算 `-log P(target_i | inputs[:i+1])`。求和。这是整个序列的交叉熵。

你听说过的每一个 Transformer 语言模型都在这个损失上训练。预训练、微调、SFT——相同的损失，不同的数据。

### 解码策略

训练后，采样选择的重要性超出人们的想象。

|  方法  |  作用  |  何时使用  |
|--------|--------------|-------------|
|  贪婪  |  每一步取 argmax  |  确定性任务，代码补全  |
|  温度  |  将 logits 除以 T，然后采样  |  创意任务，T 越高多样性越大  |
|  Top-k  |  仅从概率最高的 k 个词元中采样  |  消除低概率尾部  |
|  Top-p（核采样）  |  从累积概率 ≥ p 的最小集合中采样  |  2020 年后的默认策略；适应分布形状  |
|  Min-p  |  保留概率 ≥ `p > min_p * max_p` 的词元  |  2024 年后的策略；比 top-p 更能拒绝长尾  |
|  推测解码  |  草稿模型提出 N 个词元，大模型验证  |  在相同质量下降低 2-3 倍延迟  |

在 2026 年，min-p + 温度 0.7 是开源权重模型的合理默认配置。推测解码是任何生产推理栈的准入门槛。

### 是什么让“GPT 配方”行之有效

1. **仅解码器。** 无编码器开销。每层只需一次注意力+前馈网络。
2. **扩展。** 1.24亿 → 15亿 → 1750亿 → 万亿。Chinchilla缩放法则(第13课)指导你如何分配算力。
3. **上下文学习。** 约在6B-13B参数规模涌现。模型无需微调即可遵循少样本示例。
4. **RLHF。** 基于人类偏好的后训练将原始预训练文本转化为聊天助手。
5. **Pre-norm + RoPE + SwiGLU。** 实现大规模稳定训练。

自GPT-2以来，核心架构变化不大。所有有趣的事都发生在数据、规模和后训练上。

```figure
causal-mask
```

## 动手构建

### 步骤1：因果掩码

参见`code/main.py`。一句话概括：

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

在softmax之前将其加到注意力分数上。这就是整个机制。

### 步骤2：一个2层GPT风格模型

堆叠两个解码器块（带掩码的自注意力+前馈网络，无交叉注意力）。添加词嵌入、位置编码和解嵌入（与词嵌入矩阵共享权重——自GPT-2以来的标准技巧）。

### 步骤3：端到端的下一个词预测

在20个词符的玩具词表上，每个位置生成logits。计算与目标（移位一位）的交叉熵损失。无梯度——这是前向传播的合理性检查。

### 步骤4：采样

实现贪心、温度、top-k、top-p、min-p。对固定提示运行每种方法并比较输出。采样函数约10行代码。

## 使用它

PyTorch, 2026风格：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

在内部，`generate()`运行前向传播，提取最后一个位置的logits，采样下一个词符，追加并重复。每个生产级LLM推理栈（vLLM、TensorRT-LLM、llama.cpp、Ollama、MLX）都实现同样的循环，并经过大量优化——批量预填充、连续批处理、KV缓存分页、推测解码。

**GPT vs BERT，一句话概括：** GPT预测`P(x_t | x_{<t})`。BERT预测`P(x_masked | x_unmasked)`。损失函数决定了模型是否能够生成。

## 发布

参见`outputs/skill-sampling-tuner.md`。该技能为新生成任务选择采样参数，并标记何时需要确定性解码。

## 练习

1. **简单。** 运行`code/main.py`并验证因果注意力矩阵在softmax后是下三角的。抽查：第3行应仅在0-3列有权重。
2. **中等。** 实现宽度为4的束搜索。比较在10个短提示上束搜索与贪心搜索的困惑度。束搜索总是更好吗？（提示：通常用于翻译，而非开放聊天。）
3. **困难。** 实现推测解码：使用一个小的2层模型作为草稿模型，6层模型作为验证模型。测量在100个长度为64的补全上的端到端加速比。确认输出与验证模型的贪心搜索匹配。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  因果掩码  |  "三角形"  |  加到注意力分数上的上三角`-inf`矩阵，使得位置`i`只能看到位置`≤ i`。  |
|  下一个词预测  |  "损失"  |  每个位置上模型分布与真实下一个词之间的交叉熵。  |
|  自回归  |  "逐个生成"  |  将输出作为输入反馈回去；只在训练时并行，生成时不行。  |
|  Logits  |  "Softmax前得分"  |  LM头在softmax前的原始输出；采样基于这些值进行。  |
|  温度  |  "创造力旋钮"  |  将logits除以T；T→0为贪心，T→∞为均匀分布。  |
|  Top-p  |  "核采样"  |  截断分布至概率和≥p的最小集合；从剩余部分采样。  |
|  Min-p  |  "比top-p更好"  |  保留满足`p ≥ min_p × max_p`的词符；根据分布尖锐度自适应截断。  |
|  推测解码  |  "草稿+验证"  |  廉价模型提议N个词符；大模型并行验证。  |
|  教师强制  |  "训练技巧"  |  训练时，输入真实的上一个词符而非模型预测。所有序列到序列语言模型的标准做法。  |

## 延伸阅读

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1。
- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-2。
- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-3与上下文学习。
- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — 推测解码论文。
- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — 标准因果LM参考代码。
