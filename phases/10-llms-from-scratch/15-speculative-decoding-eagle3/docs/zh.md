# 推测性解码(Speculative Decoding)与EAGLE-3

> 第7阶段·第16课证明了数学：利维坦拒绝规则(Leviathan rejection rule)精确地保留了验证器(Verifier)的分布(Distribution)。本课是2026年生产环境推测性解码(Speculative Decoding)的训练堆栈视图。EAGLE-3将草稿模型(Draft model)从一个廉价的近似转变为专门构建的微型网络，该网络在验证器自身的隐藏状态(Hidden states)上训练，然后添加了一个训练时测试循环，对齐其训练和推理分布。结果：端到端加速比3倍到6.5倍，聊天场景下每Token接受率超过0.9，无分布权衡。2026年每个生产推理堆栈都默认搭载它。

**类型：** 构建
**语言：** Python (标准库)
**先修课程：** 第7阶段·第16课（推测性解码数学），第10阶段·第12课（推理优化）
**时间：** 约75分钟

## 学习目标

- 用一句话陈述利维坦定理(Leviathan theorem)，并证明推测性循环(Speculative loop)产生的样本与验证器(Verifier)同分布。
- 回顾从原始推测性解码(vanilla spec-decoding)（利维坦，2023）经过EAGLE、EAGLE-2到EAGLE-3的两年进展，指出每一步移除的具体限制。
- 根据接受率`α`和草稿与验证器的成本比`c`计算预期加速比(Speedup)，并为每种情况选择最优草稿长度`N`。
- 从头实现完整的推测性循环：草稿、验证、从残差分布拒绝采样、在拒绝时回滚KV缓存(KV cache)、在完全接受时发出奖励Token(Bonus token)。

## 问题

在H100上，一个70B模型的自回归解码(Autoregressive decoding)速度大约为每秒35个Token。GPU远未饱和。内存带宽(Memory bandwidth)是瓶颈：每个Token从HBM加载70B的权重，执行一步算术，产生一个浮点数。计算单元大部分时间处于空闲状态。

推测性解码(Speculative decoding)将其转化为一个你可以实际解决的吞吐量(Throughput)问题。一个廉价的草稿在`N`次小型前向传播(Forward pass)中提出`N`个Token。验证器在前缀加上所有`N`个草稿上运行一次。如果验证器在位置`i`的分布与草稿一致（在我们将精确化的统计意义上），我们接受；否则，我们拒绝并从残差分布(Residual distribution)中采样一个修正。单个大模型前向传播产生多达`N+1`个被接受的Token，而不是一个。

关键的定理是Leviathan、Kalman、Matias（ICML 2023）：输出分布与直接从验证器采样得到的结果完全相同。不是近似，而是完全相同。这就是推测性解码在生产环境中可被接受的整个原因——它是一个纯粹的延迟优化，没有质量权衡。

第7阶段·第16课教给你的是数学。本课教给你的是训练堆栈。一个好的草稿比一个廉价的草稿多提供2倍的加速。EAGLE、EAGLE-2和EAGLE-3（Li等人，2024-2025）将"草稿=同一模型的更小版本"转变为精确的工程学科。2026年生产推理服务器默认使用EAGLE-3。

## 核心概念

### 不变量：利维坦拒绝采样(Leviathan rejection sampling)

令`p(t)`为给定某个前缀时草稿对下一个Token的分布，`q(t)`为验证器的分布。采样一个草稿Token `d ~ p`。以概率`min(1, q(d) / p(d))`接受。如果拒绝，则从残差分布`(q - p)_+ / ||(q - p)_+||_1`采样。最终样本的分布符合`q`。无论`p`有多差，这个结论都成立——它越差，你拒绝的次数越多，但输出仍然精确。

在`prefix + d_1 + ... + d_N`上使用一次验证器前向传播，将`N`次这样的调用背靠背堆叠。验证器同时返回`q_1, q_2, ..., q_{N+1}`。从左到右处理。在位置`j`第一次拒绝时，从`residual(q_j, p_j)`采样并停止。如果完全接受，则从`q_{N+1}`采样一个奖励Token。

### 什么决定了加速比

令`α`为每个草稿Token的期望接受率(Expected acceptance rate)。令`c = cost(draft) / cost(verifier)`为成本比(Cost ratio)。每次验证器前向传播期望接受的Token数为：

```
E[accepted] = (1 - α^(N+1)) / (1 - α)
```

每个接受Token的期望总挂钟时间(Wall time)为`(N * c + 1) / E[accepted]`。相对于`N`最小化该值，您就能得到最佳点。对于`α = 0.8, c = 0.05`：最优的`N`大约为5-7，加速比3.2×。对于`α = 0.95, c = 0.02`：最优的`N`大约为8-10，加速比接近5×。

最大的单一杠杆(Lever)是`α`。在固定的`N = 5`下，从`α = 0.6`（原始草稿）到`α = 0.9`（EAGLE-3）将每次验证器前向传播的期望接受Token数从2.2提高到4.1。相同的验证器近乎2倍的吞吐量提升。

### 两年进展

**原始推测性解码(Vanilla speculative, Leviathan, 2023).** 草稿模型是一个独立训练的来自同系列的小型LLM。容易集成，`α ≈ 0.6`，最大加速比约2倍。

**EAGLE-1 (Li等人, 2024).** 草稿是一个微型Transformer——通常一或两层——以验证器的最后一层隐藏状态作为输入，直接预测下一个Token。由于草稿看到了验证器的特征表示(Feature representation)，其分布更接近验证器的分布。`α`提升到0.7-0.8。

**EAGLE-2 (Li等人, 2024).** 添加了动态草稿树(Dynamic draft tree)：不是提出一个`N`个Token的单序列，而是提出一个小的候选树，在一次前向传播中（树注意力(Tree attention)）由验证器给每个候选打分，然后沿着最高概率路径行走。草稿长度每步变得自适应。每个接受路径Token的`α`提升到0.85以上。

**EAGLE-3 (Li等人, 2025, NeurIPS).** 还有两个变化。首先，完全丢弃特征预测损失(Feature-prediction loss)——EAGLE-1/2训练草稿以匹配验证器的隐藏状态，这限制了数据能带来的帮助。EAGLE-3直接以Token预测进行训练。其次，训练时测试(Training-time test, TTT)：在草稿训练期间，将草稿自身的先前预测作为输入在多步中反馈，就像推理时一样。这对齐了训练和测试分布，并阻止了错误累积。实测加速比：聊天场景高达6.5倍，在H100上的SGLang中批量64时吞吐量提升38%。

### KV缓存回滚(KV cache rollback)

验证在一次前向传播中将验证器的KV缓存扩展`N`个条目。如果拒绝发生在位置`j`，则位置`j-1`之后的缓存内容现在是错误的。两种常见实现：写入临时缓冲区并在接受时提交（vLLM, TensorRT-LLM），或者保留物理KV缓存加上逻辑长度并在拒绝时截断。无论哪种方式，回滚开销是每层每头的字节数，与前向传播成本相比可以忽略不计。

对于EAGLE-2的树搜索，验证器使用尊重树拓扑的非因果掩码(Non-causal mask)运行注意力。工程实现很繁琐，但计算是一个带自定义掩码的标准FlashAttention调用。

### 2026年的草稿架构(Draft architectures)

|  策略(Strategy)  |  草稿类型(Draft type)  |  `α`  |  加速比(Speedup)  |  训练成本(Training cost)  |
|----------|-----------|-----|---------|---------------|
|  原始(Vanilla)  |  独立的小型LLM  |  0.55-0.70  |  1.8-2.3×  |  无（重用现有小型模型）  |
|  Medusa  |  验证器上的额外LM头(Extra LM heads)  |  0.65-0.75  |  2-3×  |  约10亿SFT Token  |
|  EAGLE-1  |  基于隐藏状态的单层Transformer  |  0.70-0.80  |  2.5-3×  |  约600亿Token  |
|  EAGLE-2  |  EAGLE-1 + 动态草稿树(Dynamic draft tree)  |  0.80-0.88  |  3-4×  |  约600亿Token  |
|  EAGLE-3  |  多层特征融合 + 训练时测试(TTT)  |  0.88-0.92  |  3.5-6.5×  |  约600亿至2000亿Token  |
|  Lookahead  |  无草稿（雅可比迭代(Jacobi iteration)）  |  不适用  |  1.3-1.6×  |  无  |

在2026年的生产中：vLLM和SGLang在可用时默认使用EAGLE-3，否则使用EAGLE-2。TensorRT-LLM为Meta和NVIDIA的公开模型提供了最快的Medusa路径。llama.cpp针对CPU部署提供了原生草稿。

## 动手构建

参见`code/main.py`。这是完整的Leviathan投机循环，包含所有部分：草稿生成N个token、验证器并行执行、逐位置拒绝、残差采样、奖励token、KV回滚以及经验验证输出分布与直接从`q`采样一致。

### 第一步：拒绝规则

```python
def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
```

### 第二步：残差分布

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
```

### 第三步：完整的投机步骤

`spec_step`函数从`N`中草稿生成`p`个token，然后在一个并行的`q`评估中验证所有token。对每个草稿token应用拒绝规则，并在第一次拒绝时从残差中采样修正。如果全部接受，则从`q_{N+1}`中发出一个奖励token。

### 第四步：KV回滚记账

模拟器跟踪每个工作线程的逻辑`kv_length`。当接受`k`个草稿时，`kv_length += k`。如果在位置`j`拒绝，缓存已经写入了超过`j`的位置，但逻辑长度设置为`prefix_length + j + 1`——即修正token之后一个位置。后续读取会截断到逻辑长度。

### 第五步：Leviathan检查

运行50,000次投机步骤。统计接受token的经验分布。与从`q`直接采样的50,000个样本进行比较。卡方统计量应远低于临界值。定理在实践中成立。

### 第六步：加速比与α的关系

通过以不同幅度将`p`偏离`q`来扫描草稿质量。测量`α`，然后绘制每个验证器调用的预期token数作为`α`和`N`的函数。代码输出一个表格，显示EAGLE-3级别的草稿质量（`α ≈ 0.9`）如何实现每个验证器调用4-5个token。

## 使用它

使用EAGLE-3的生产级`vllm serve`：

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'
```

在H100上使用EAGLE-3、批次大小为64的SGLang：根据EAGLE-3论文，吞吐量大约是批次为64的普通解码的1.38倍。

何时使用投机解码：

- 任何交互式聊天工作负载，其中p50延迟比峰值吞吐量更重要。
- 代码生成和结构化输出（JSON、SQL）。`α`高于0.9，因为目标分布高度可预测。
- 长文本生成（数千个token）。摊销的加速比持续生效。

何时不使用：

- 非常小的模型（< 3B）。草稿并不比验证器便宜多少。
- 批次为1的极小型CPU部署。草稿模型的内存开销可能不值得。
- 极高温度的创意采样，此时`α`急剧下降。

## 发布

本课生成`outputs/skill-eagle3-tuner.md`。给定一个推理工作负载（模型、批次大小、目标延迟、任务类型），它推荐投机解码策略和调优参数（草稿系列、`N`、树深度、温度感知切换）。

## 练习

1. 运行`code/main.py`。确认Leviathan分布检查的卡方统计量在50,000个样本上低于95%临界值。

2. 将`N`从1扫描到10，`α`固定为0.9，`c`固定为0.04。绘制每个验证器调用的预期token数和每个token的实际墙钟时间。找到最小化墙钟时间的`N`。解释曲线的形状。

3. 修改代码以模拟EAGLE-2树搜索：在每一步，草稿提出一个形状为`[2, 2, 2]`的树（八个候选路径）。验证器运行一次，选择概率最高的接受路径。计算每个叶节点的`α`和每个验证器调用的总token数。与同等计算量下的线性链投机解码进行比较。

4. 实现一个批处理KV回滚模拟器，用于两个并发序列。序列A的所有草稿都被接受；序列B在位置2拒绝。显示每个序列的正确`kv_length`已更新，且没有浪费任何工作。

5. 阅读EAGLE-3论文的第4节（训练时测试）。用两句话解释为什么没有TTT的朴素草稿训练会受到暴露偏差的影响，以及为什么在训练期间将草稿自己的预测馈送回草稿可以解决这个问题。将其与seq2seq中的计划采样文献联系起来。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
| Leviathan规则  |  "min(1, q/p)"  |  伯努利接受/拒绝，概率为`min(1, q(d)/p(d))`，当在拒绝时从残差中采样时，精确保持验证器分布 |
| 残差分布  |  "(q - p)⁺，归一化"  |  `(q - p)_+`在零处截断并重新归一化——拒绝时应采样的正确分布 |
| 接受率α  |  "草稿正确的频率"  |  在拒绝规则下每个token的期望伯努利成功概率；决定所有加速比计算 |
| EAGLE-1  |  "隐藏状态草稿"  |  以验证器最后一层隐藏状态为条件的小型Transformer草稿（Li et al., 2024） |
| EAGLE-2  |  "动态草稿树"  |  EAGLE-1加上一个候选延续树，通过一次验证器传递中的树注意力进行评分 |
| EAGLE-3  |  "训练时测试"  |  丢弃特征预测损失，训练时使用草稿自己的输出进行直接token预测 |
| 训练时测试（TTT）  |  "暴露偏差修复"  |  在训练期间自回归地运行草稿，使训练和测试输入分布匹配——计划采样在序列到序列中的直接类比 |
| KV回滚  |  "撤销被拒绝的草稿"  |  一种记账机制，在拒绝后将验证器的KV缓存重置为已接受前缀的长度 |
| 额外令牌  |  "免费的"  |  当所有`N`草稿都被接受时，从`q_{N+1}`中额外采样一个，无需额外的验证器成本 |
| 树注意力  |  "一次性验证多个候选"  |  一种采用非因果掩码的注意力机制，该掩码尊重草稿树的拓扑结构；在前向传播中一次性计算树中每个节点的`q_i` |

## 延伸阅读

- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) — 基础论文与等价定理
- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) — 同时独立引入，带有简洁证明
- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) — EAGLE-1，基于隐藏状态条件的草稿
- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) — 动态树搜索
- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) — 2026年生产默认
- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) — 无草稿替代方法
- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) — 经典生产参考，包含所有策略
