# 推测性解码与EAGLE

> 一个前沿的大语言模型生成一个token需要对数十亿参数进行一次完整的前向传播。这一前向传播是过度配置的：大多数时候，一个更小的模型可以正确猜出接下来的3-5个token，而大模型只需要*验证*这个猜测。当猜测正确时，你就以一次计算的代价获得了5个token。推测性解码（Speculative Decoding, Leviathan等人, 2023）使这一点精确成立，而EAGLE-3（2025）将接受率提升到每个验证约4.5个token——在匹配输出分布的情况下实现了4-5倍的加速。

**类型：** 构建
**语言：** Python（使用numpy）
**前置条件：** 第10阶段第12课（推理优化），第10阶段第04课（预训练Mini-GPT）
**时间：** ~75分钟

## 问题

对于H100上的70B类模型，解码吞吐量通常为40-80 tokens/秒。每个token都需要一次完整的前向传播，从HBM中读取所有模型权重。你不能在不改变模型输出的情况下让模型变小。你无法将批次大小增加到超出内存限制。你陷入了困境——除非你能让模型每次前向传播输出多个token。

自回归生成看起来本质上是串行的：`x_{t+1} = sample(p(· | x_{1:t}))`。但这里存在并发机会。如果你有一个廉价的预测器，它说"接下来的4个token很可能是[a, b, c, d]"，那么你可以在大模型的**单次前向传播**中验证所有5个位置，并接受最长的匹配前缀。

Leviathan, Kalai, Matias (2023, "通过推测性解码实现Transformer的快速推理")通过一种巧妙的接受/拒绝规则使这一点精确成立，该规则保留了目标模型的采样分布。相同的输出分布，速度提升2-4倍。

## 核心概念

### 双模型设置

- **目标模型** `M_p`：你实际想要从中采样的大型、缓慢、高质量的模型。分布：`p(x)`。
- **草稿模型** `M_p`：小型、快速、低质量的模型。分布：`p(x)`。体积小5-30倍。

每一步：

1. 草稿模型自回归地提议`K`个token：`x_1, x_2, ..., x_K ~ q`。
2. 目标模型在所有`K`个位置上并行执行**一次**前向传播，为每个提议的token产生`x_1, x_2, ..., x_K ~ q`。
3. 通过下面的改进拒绝采样规则，从左到右接受/拒绝每个token。接受最长的匹配前缀。
4. 如果有任何token被拒绝，从修正后的分布中采样替换token并停止。否则从`K`中采样一个奖励token。

如果草稿与目标完美匹配，则每个目标前向传播获得K+1个token。如果草稿在位置1出错，则只获得1个token。

### 精确性规则

推测性解码在分布上**可证明等价于从p采样**。拒绝规则如下：

```
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

其中`(p - q)+`表示逐点差的正部。当草稿和目标一致时（`p ≈ q`），接受率接近1。当它们不一致时，构建残差分布使得整体样本仍然精确等于`p`。

**贪心情况。** 对于温度=0采样，只需检查`argmax(p) == x_t`。如果是，则接受；如果否，则输出`argmax(p)`并停止。

### 预期加速比

如果草稿模型的token级接受率为`α`，则每个目标前向传播预期的token数量为：

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = draft length, α in [0, 1]
```

当`α = 0.8, K = 4`时：每次前向传播`(1 - 0.8^5)/(1 - 0.8) = 3.36`个token。一次目标前向传播的成本约为`cost_q * K + cost_p`（K个草稿步骤加一次目标验证）。如果`cost_p >> cost_q * K`，则吞吐量的加速比为`3.36× / 1 = 3.36×`。

唯一的真实参数是`α`，它完全取决于草稿-目标的对齐程度。一个好的草稿就是一切。

### 训练草稿：蒸馏

一个随机的小模型是一个糟糕的草稿。标准配方是从目标模型进行蒸馏：

1. 选择一个小的架构（对于70B目标约1B，对于7B目标约500M）。
2. 在一个大型文本语料库上运行目标模型；存储其下一个token的分布。
3. 使用KL散度针对目标模型的分布（而非真实token）训练草稿模型。

结果：在编程任务上，`α`通常为0.6-0.8，在自然语言对话上为0.7-0.85。生产环境中加速2-3倍。

### EAGLE：树形草稿 + 特征重用

Li, Wei, Zhang, Zhang (2024, "EAGLE：推测性采样需要重新思考特征不确定性")观察到标准推测性解码中的两个低效之处：

1. 草稿执行K个串行步骤，每一步都完整堆栈。但草稿可以重用最近一次验证中目标的特征（隐藏状态）——目标已经计算了丰富的表示，而草稿却从头重新推导。
2. 草稿输出线性链。如果草稿能输出一个候选*树*（每个节点多个猜测），那么目标的单次前向传播可以通过树注意力掩码并行验证多个候选路径，并选择最长的接受分支。

EAGLE-1的改动：
- 草稿输入 = 目标在位置t的最终隐藏状态，而非原始token。
- 草稿架构 = 1个Transformer解码器层（而非独立的较小模型）。
- 输出 = 每层K=4-8个候选的树，深度为4-6。

EAGLE-2（2024）增加了动态树拓扑：在草稿不确定的地方树变宽，在确定的地方树变窄。提高了`α_effective`而不增加验证成本。

EAGLE-3（Li等人, 2025, "EAGLE-3：通过训练时测试扩展大语言模型的推理加速")去除了固定的顶层特征依赖，并使用新的"测试时模拟"损失函数训练草稿——草稿在符合目标测试时分布的输出上进行训练，而非教师强制训练分布。接受率从0.75（EAGLE-2）提升到0.82（EAGLE-3），平均每验证token数从3.0提升到4.5。

### 树注意力验证

当草稿输出树结构时，目标模型通过单次前向传递使用**树注意力掩码**进行验证——这是一种因果掩码，编码树拓扑而非纯线性结构。每个令牌仅关注其在树中的祖先。验证过程仍是一次前向、一次矩阵乘法；拓扑掩码仅增加少量KV条目。

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

如果`a, b`是竞争的首令牌候选，`c, d, e, f`是次令牌候选，则所有六个位置都在一次前向传递中验证完毕。输出是沿任何被接受路径的最长前缀。

### 优势与劣势

**优势：**
- 文本可预测的聊天/补全场景（代码、常见英语、结构化输出）。`α`较高。
- 解码阶段（内存受限阶段）有闲置GPU算力的场景。树草稿利用可用FLOPs。

**劣势/无优势：**
- 高度随机输出（高温下的创意写作）。`α`降至`1/|vocab|`。
- 高并发批处理服务——批处理已占满FLOPs，树验证空间小。
- 目标模型非常小，草稿与之差距不大。

生产环境通常报告：聊天场景2-3倍加速，代码生成3-5倍，创意写作接近零加速。

```figure
speculative-decoding
```

## 动手构建

`code/main.py`：

- 一个参考实现的`speculative_decode(target, draft, prompt, K, temperature)`，包含精确拒绝规则并验证其保持目标分布（经验KL散度<0.01 vs 普通目标采样）。
- 一个EAGLE风格树草稿器，构建深度为K、top-p分支的树。
- 一个树注意力掩码构建器，为验证器生成正确的因果模式。
- 一个接受率测试工具，在小型语言模型上运行（从GPT-2-medium蒸馏出GPT-2-small作为目标）。

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. Target computes p at every drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left-to-right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted → sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## 使用它

- **vLLM** 和 **SGLang** 原生支持推测解码。标志：`--speculative_model`, `--num_speculative_tokens`。通过`--spec_decoding_algorithm eagle`标志支持EAGLE-2/3。
- **NVIDIA TensorRT-LLM** 原生支持Medusa和EAGLE树。
- **参考草稿模型**：`--speculative_model`（为Qwen3-32B草稿），`--num_speculative_tokens`（为70B草稿）。
- **Medusa头**（Cai等人，2024年，“Medusa：具有多个解码头的简单LLM推理加速框架”）：不添加草稿模型，而是在目标模型上增加K个并行预测头。部署更简单，接受率略低于EAGLE。

## 发布

本课产出`outputs/skill-speculative-tuning.md`——一种技能，分析目标模型的工作负载并选择：草稿模型、K（草稿长度）、树宽度、温度，以及何时回退到普通解码。

## 练习

1. 实现精确拒绝规则并通过实验验证。通过`speculative_decode`和普通目标采样运行10K样本，计算两个输出分布之间的TV距离，应小于0.01。

2. 计算加速公式。给定固定的`α`和`K`，绘制每个目标前向的期望令牌数。找到α∈{0.5, 0.7, 0.9}时的最优K。

3. 训练小型草稿。以124M GPT-2为目标，用100M令牌以KL损失蒸馏出30M GPT-2草稿。在保留文本上测量`α`，期望值0.6-0.7。

4. 实现EAGLE风格树草稿。草稿在每层输出top-3分支而非链式结构。构建树注意力掩码。验证目标接受最长的正确分支。

5. 测量失败模式。在温度为1.5（高随机性）下运行推测解码。显示α崩溃且算法因草稿开销比普通解码更慢。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
|  目标模型  |  “大模型”  |  慢速高质量模型，从中采样（p分布）  |
|  草稿模型  |  “推测器”  |  小型快速预测器（q分布）；小5-30倍  |
|  K / 草稿长度  |  “前瞻步数”  |  每次验证推测的令牌数  |
|  α / 接受率  |  “命中率”  |  草稿提议被接受的每令牌概率  |
|  精确拒绝规则  |  “接受测试”  |  r < p/q 比较，保持目标分布  |
|  残差分布  |  “修正p-q”  |  (p - q)+ /  |  | (p - q)+ |  | _1，拒绝时采样的分布  |
|  树草稿  |  “分支推测”  |  草稿输出候选树，通过树结构注意力掩码一次验证  |
|  树注意力掩码  |  “拓扑掩码”  |  编码树拓扑的因果掩码，每个节点仅关注其祖先  |
|  Medusa头  |  “并行头”  |  目标模型上的K个额外预测头；无需独立草稿模型  |
|  EAGLE特征复用  |  “隐藏状态草稿”  |  草稿输入为目标模型最后隐藏状态而非原始令牌，缩小草稿  |
|  测试时模拟损失  |  “EAGLE-3训练”  |  在匹配目标测试时分布的输出上训练草稿，而非教师强制  |

## 延伸阅读

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — 精确拒绝规则与理论加速分析
- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — DeepMind并发推测采样论文
- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — 传统草稿模型替代方案：并行头
- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — 特征复用与树草稿
- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — 动态树拓扑
- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — 训练时-测试时匹配
- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — 雅可比/前瞻解码，无推测器替代方案
