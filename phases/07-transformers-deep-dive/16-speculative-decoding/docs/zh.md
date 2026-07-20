# 投机解码 — 草稿、验证、重复

> 自回归解码是串行的。每个 token 等待前一个。投机解码打破了链条：一个廉价模型草拟 N 个 token，昂贵模型在一个前向传播中验证所有 N 个。当草稿正确时，你为生成了 N 个 token 只付出了一次大模型前向传播的代价。

**类型：** 构建
**语言：** Python
**先决条件：** 阶段 7 · 07 (GPT 因果语言模型)，阶段 7 · 12 (KV 缓存与 Flash Attention)
**时间：** 约 60 分钟

## 问题

一个 70B 的大语言模型在 H100 上采样一个 token 大约需要 30 毫秒。一个 3B 的草稿模型大约需要 3 毫秒。如果我们让 3B 模型草拟 5 个 token，然后运行 70B 模型 *一次* 来验证所有 5 个，那么总共需要 `5×3 + 30 = 45 ms` 用于最多 5 个被接受的 token —— 而直接生成则是 `5×30 = 150 ms`。这就是完整的投机解码宣传：用少量的额外 GPU 内存（草稿模型）换取 2-4 倍的解码延迟降低。

这个技巧必须保持分布不变。投机采样，由 Leviathan 等人 (2023) 和 Chen 等人同时提出，保证输出序列与大模型自己生成的 **同分布**。没有质量权衡。只是更快。

在 2026 年的推理中，四种草稿-验证器家族占主导地位：

1. **原始投机 (Leviathan 2023)。** 分离的草稿模型（例如，Llama 3 1B）+ 验证器（例如，Llama 3 70B）。
2. **Medusa (Cai 2024)。** 验证器上的多个解码头并行预测位置 `t+1..t+k`。没有独立的草稿模型。
3. **EAGLE 家族 (Li 2024, 2025)。** 轻量级草稿，重用验证器的隐藏状态；接受率高于原始投机；典型 3-4 倍。
4. **前瞻解码 (Fu 2024)。** Jacobi 迭代；根本不需要草稿模型。自我投机。小众但无依赖。

2026 年的每个生产推理栈默认都集成了投机解码。vLLM、TensorRT-LLM、SGLang 和 llama.cpp 都至少支持原始投机 + EAGLE-2。

## 核心概念

### 核心算法

给定一个验证器 `M_q` 和一个更便宜的草稿 `M_p`：

1. 令 `x_1..x_k` 为已经解码的前缀。
2. **草稿**：使用 `x_1..x_k` 自回归地提出 `M_p`，草稿概率为 `d_{k+1}, d_{k+2}, ..., d_{k+N}`。
3. **并行验证**：对 `M_p` 运行一次 `x_1..x_k`，得到位置 `p_1..p_N` 的验证器概率 `d_{k+1}, d_{k+2}, ..., d_{k+N}`。
4. **从左到右接受/拒绝每个草稿 token**：对于每个 `x_1..x_k`，以概率 `M_p` 接受。
5. 在位置 `x_1..x_k` 第一次拒绝时：从归一化的“残差”分布 `M_p` 中采样 `d_{k+1}, d_{k+2}, ..., d_{k+N}`。丢弃 `p_1..p_N` 之后的所有草稿。
6. 在接受所有 `x_1..x_k` 后：从 `M_p` 中采样一个额外 token `d_{k+1}, d_{k+2}, ..., d_{k+N}`（免费奖励 token）。

残差分布技巧是数学上的洞见，它使得输出分布与 `M_q` 从头采样完全相同。

### 什么决定了加速比

令 `α` = 每个草稿 token 的预期接受率。令 `c` = 草稿与验证器的成本比。每一步：

- 朴素生成每个 token 需要一次大模型调用。
- 投机解码每 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` 个 token 需要一次大模型调用，当 `α` 很高时。

在 `α = 0.75` 和 `N = 5` 下的典型经验法则：大模型调用次数减少 3 倍。草稿成本便宜 5 倍。整体墙上时钟时间下降约 2.5 倍。

**α 取决于：**

- 草稿对验证器的近似程度。同一家族/同一训练数据可以显著提高 α。
- 解码策略。贪心草稿对贪心验证器：α 高。温度采样：更难匹配；接受率下降。
- 任务类型。代码和结构化输出接受更多（可预测）；自由形式的创意写作接受更少。

### Medusa — 无需草稿模型的草稿

Medusa 用验证器上的额外输出头取代了草稿模型。在位置 `t`：

```
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

每个头输出自己的 logits。推理时，从每个头采样获得候选序列，然后使用树注意力方案进行一次前向传播验证，该方案同时考虑所有候选延续。

优点：没有第二个模型。缺点：增加了可训练参数；需要监督微调阶段（约 1B token）；接受率略低于使用良好草稿的原始投机。

### EAGLE — 通过重用隐藏状态获得更好的草稿

EAGLE-1/2/3 (Li et al., 2024–2025) 使草稿模型成为一个微型 transformer（通常 1 层），它吸收验证器的最后一层隐藏状态。由于草稿看到验证器的特征表示，其预测与验证器的输出分布强烈相关。接受率从约 0.6（原始）上升到 0.85+。

EAGLE-3 (2025) 添加了候选延续的树搜索。vLLM 和 SGLang 将 EAGLE-2/3 作为 Llama 3/4 和 Qwen 3 的默认投机路径。

### KV 缓存的舞蹈

验证在一个前向传播中将 `N` 个草稿 token 送入验证器。这将验证器的 KV 缓存扩展了 `N` 个条目。如果某些草稿被拒绝，你必须将缓存回滚到被接受的前缀长度。

生产实现（vLLM 的 `--speculative-model`，TensorRT-LLM 的 LookaheadDecoder）使用临时 KV 缓冲区处理这个问题。先写入，在确认接受后提交。这在概念上不困难，但很繁琐。

## 动手构建

参见 `code/main.py`。我们实现了核心的投机采样算法（拒绝步骤 + 残差分布），包括：

- 一个“大模型”，它对一个手动编码的分布进行确定性 softmax（这样我们可以解析地验证接受数学）。
- 一个“草稿模型”，它是大模型的一个扰动。
- 一个接受/拒绝循环，产生与直接采样相同的边际分布。

### 第一步：拒绝步骤

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` 是一个均匀随机数。`q_prob` 是验证器对于所提议令牌的概率。`p_prob` 是草稿模型的概率。Leviathan(利维坦)定理指出，这个伯努利决策，加上在拒绝时从残差分布中采样，恰好保持了验证器的分布。

### 第二步：残差分布

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

将 `p` 从 `q` 中逐元素相减，将负值钳制为零，重新归一化。在任何拒绝发生时从中采样。

### 第三步：一次投机步骤

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

五个被接受 → 一个奖励令牌 → 在一次验证器前向传播中产生六个令牌。

### 第四步：衡量接受率

在不同草稿质量水平下运行10,000次投机步骤。绘制接受率与草稿和验证器分布之间的KL散度的关系图。你应该能看到一个清晰的单调关系。

### 第五步：验证分布等价性

经验性地：由投机循环产生的令牌直方图应与直接从验证器采样产生的直方图匹配。这就是Leviathan定理的实践。卡方检验确认在抽样误差范围内一致。

## 使用它

生产环境：

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

截至2026年中，TensorRT-LLM拥有最快的Medusa路径。`faster-whisper` 将投机解码封装用于Whisper-large，并带有一个小型的草稿模型。

**选择草稿模型：**

|  策略  |  何时选择  |  加速比  |
|----------|--------------|---------|
|  原始草稿(Llama 1B/3B系列)  |  快速原型，无需训练  |  1.8–2.3×  |
|  Medusa heads  |  可以微调验证器  |  2–3×  |
|  EAGLE-2 / 3  |  生产环境，最大速度  |  3–4×  |
|  前瞻解码  |  无草稿，无训练，无额外参数  |  1.3–1.6×  |

**何时不应使用投机解码：**

- 单序列生成长度为1–5个令牌。开销占主导。
- 高度创造性/高温采样（接受率α下降）。
- 内存受限的部署（草稿模型增加显存）。

## 发布

参见 `outputs/skill-spec-decode-picker.md`。该技能为新的推理工作负载选择投机解码策略（原始/Medusa/EAGLE/前瞻）和调优参数（N，草稿温度）。

## 练习

1. **简单。** 运行 `code/main.py`。确认投机令牌分布与验证器在50,000个令牌上的直接采样分布在卡方检验p > 0.05的情况下匹配。
2. **中等。** 绘制加速比（每大模型前向传播产生的令牌数）作为 `code/main.py` 的函数，针对 `N`。确定每个α的最优 `α = 0.5, 0.7, 0.85`。（提示：每次验证调用预期令牌数 = `N`。）
3. **困难。** 实现一个小型Medusa：取第14课的capstone GPT，添加3个额外的LM头，预测位置t+2, t+3, t+4。在tinyshakespeare上使用联合多头损失进行训练。将接受率与通过截断同一模型制作的原始草稿进行比较。
4. **困难。** 实现回滚：从一个10令牌的前缀KV缓存开始，输入5个草稿令牌，模拟在位置3拒绝。验证你的缓存读取在下次迭代中正确匹配“前缀 + 前2个接受的草稿”。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  草稿模型  |  “廉价的那个”  |  一个较小的模型，提出候选令牌；通常比验证器便宜10–50倍。  |
|  验证器  |  “大的那个”  |  保持其分布的目标模型；每个投机步骤运行一次。  |
|  接受率(α)  |  “草稿正确的频率”  |  验证器接受草稿的每个令牌概率。典型值0.7–0.9。  |
|  残差分布  |  “拒绝回退”  |  `(q - p)_+` 归一化；在拒绝时从中采样以保持验证器的分布。  |
|  奖励令牌  |  “免费的”  |  当所有N个草稿都被接受时，从验证器的下一步分布中再采样一个。  |
|  Medusa  |  “无草稿投机”  |  验证器上的多个LM头并行预测位置t+1..t+k。  |
|  EAGLE  |  “隐藏状态草稿”  |  以验证器最后一层隐藏状态为条件的微小Transformer草稿。  |
|  前瞻解码  |  “Jacobi迭代”  |  使用不动点迭代的自我投机；无草稿模型。  |
| 树注意力 | “一次性验证多个候选” | 一种分支验证方式，同时考虑多个草稿续写。 |
| KV回滚 | “撤销被拒绝的草稿” | 临时KV缓冲区；接受则提交，拒绝则丢弃。 |

## 延伸阅读

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 核心算法与等价定理。
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 并发引入；简洁的伯努利拒绝证明。
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — Medusa论文；树注意力验证。
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — EAGLE-1；隐状态条件草稿。
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — EAGLE-2；动态树深度。
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — EAGLE-3。
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 前瞻，无草稿方法。
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 标准生产参考，包含所有四种策略。
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — EAGLE-1/2/3的参考代码。
