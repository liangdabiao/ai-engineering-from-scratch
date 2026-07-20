# Proximal Policy Optimization (PPO)（近端策略优化）

> A2C在每次更新后丢弃整个轨迹(rollout)。PPO将策略梯度(policy gradient)包裹在一个裁剪后的重要性比率(clipped importance ratio)中，这样你可以在同一数据上执行10个以上的epoch而不会导致策略爆炸。Schulman等人(2017)。截至2026年，它仍然是默认的策略梯度算法。

**类型：** 构建
**语言：** Python
**先修条件：** 第9阶段·06 (REINFORCE)，第9阶段·07 (Actor-Critic)
**时间：** ~75分钟

## 问题

A2C（第07课）是在策略(on-policy)的：梯度`E_{π_θ}[A · ∇ log π_θ]`需要从*当前*`π_θ`采样的数据。进行一次更新后，`π_θ`发生变化；你使用的数据现在变成了离策略(off-policy)。重复使用它会导致梯度有偏。

轨迹(rollout)是很昂贵的。在Atari上，一次轨迹跨越8个环境×128步=1024个转移和十几秒的环境时间。在一次梯度更新后就丢弃它是浪费的。

信任区域策略优化(Trust Region Policy Optimization, TRPO, Schulman 2015)是第一个修复方案：约束每次更新，使得新旧策略之间的KL散度(KL divergence)保持在`δ`以下。理论上很干净，但每次更新都需要共轭梯度(conjugate-gradient)求解。到2026年没有人再运行TRPO。

PPO（Schulman等人2017）用一个简单的裁剪目标(clipped objective)替代了硬信任区域约束。多一行代码。每个轨迹十个epoch。无需共轭梯度。足够好的理论保证。九年后，它仍然是默认的策略梯度算法，从MuJoCo到RLHF都适用。

## 核心概念

![PPO clipped surrogate objective: ratio clipping at 1 ± ε](../assets/ppo.svg)

**重要性比率(The importance ratio)。**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

这是新策略相对于收集数据的策略的似然比(likelihood ratio)。`r_t = 1`表示无变化。`r_t = 2`表示新策略采取`a_t`的可能性是旧策略的两倍。

**裁剪后的代理目标(The clipped surrogate)。**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

两项：

- 如果优势(advantage)`A_t > 0`且比率试图增长超过`1 + ε`，则裁剪会扁平化梯度——不要将一个好的动作推高到超过旧概率的`+ε`。
- 如果优势(advantage)`A_t > 0`且比率试图增长超过`1 + ε`（意味着我们会使一个坏动作变得更可能，相比其裁剪后的减小），则裁剪会限制梯度——不要将一个坏动作推低到`+ε`以下。

`min`处理另一个方向：如果比率已朝着*有益*的方向移动，你仍然得到梯度（在可能伤害你的那一侧没有裁剪）。

典型的`ε = 0.2`。将目标函数绘制为`r_t`的函数：一个分段线性函数，在“好的一面”有一个平坦的屋顶，在“坏的一面”有一个平坦的地板。

**完整的PPO损失(The full PPO loss)。**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

与A2C相同的演员-评论家(actor-critic)结构。三个系数，通常`c_v = 0.5`, `c_e = 0.01`, `ε = 0.2`。

**训练循环(The training loop)。**

1. 在`N`个并行环境中收集`N × T`个转移，每个环境运行`T`步。
2. 计算优势(GAE)，将它们冻结为常数。
3. 冻结`N × T`作为当前`N`的快照。
4. 对于`N × T`个epoch，每个小批量(minibatch)大小为`N`：
   - 计算`N × T`。
   - 应用`N × T` + 价值损失(value loss) + 熵(entropy)。
   - 梯度步骤。
5. 丢弃该轨迹。返回到步骤1。

`K = 10`和小批量大小为64是标准的超参数集。PPO是鲁棒的：在±50%范围内，精确数字很少重要。

**KL惩罚变体(KL-penalty variant)。**原始论文提出了一个使用自适应KL惩罚的替代方案：`L = L^{PG} - β · KL(π_θ || π_old)`，其中`β`根据观察到的KL进行调整。裁剪版本成为主流；KL变体在RLHF中仍然存在（其中相对于参考策略的KL是你总想要的独立约束）。

## 动手构建

### 步骤1：在轨迹(rollout)时捕获`log π_old(a | s)`

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

快照只在轨迹时捕获一次。在更新epoch期间它不会改变。

### 步骤2：计算GAE优势（第07课）

与A2C相同。在整个批次中归一化。

### 步骤3：裁剪后的代理目标更新(clipped surrogate update)

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

“裁剪→零梯度”模式是PPO的核心。如果新策略已经在有益方向上漂移得太远，更新就停止。

### 步骤4：价值和熵(value and entropy)

向评论家目标添加标准MSE，并向演员添加熵奖励，与A2C相同。

### 步骤5：诊断(diagnostics)

每次更新要关注的三件事：

- **平均KL散度** `E[log π_old - log π_θ]`。应保持在 `[0, 0.02]` 内。如果超出 `0.1`，则减少 `K_EPOCHS` 或 `LR`。
- **裁剪比例** — 比率超出 `E[log π_old - log π_θ]` 的样本比例。应为 `[0, 0.02]`。如果 `0.1`，则裁剪从未触发 → 提高 `K_EPOCHS` 或 `LR`。如果 `[1-ε, 1+ε]`，则你在过拟合轨迹 → 降低它们。
- **解释方差** `E[log π_old - log π_θ]`。评论家质量指标。随着评论家学习应趋近于1。

## 陷阱

- **裁剪系数调校不当。** `ε = 0.2` 是事实标准。使用 `0.1` 会使更新过于保守；`0.3+` 则招致不稳定。
- **轮次过多。** `ε = 0.2` 通常会因策略偏离 `0.1` 过远而导致不稳定。限制轮次，尤其是对于大网络。
- **未进行奖励归一化。** 较大的奖励尺度会侵蚀裁剪范围。在计算优势之前对奖励进行归一化（运行标准差）。
- **忽略优势归一化。** 每批次零均值/单位标准差归一化是标准做法。跳过它会破坏大多数基准上的 PPO。
- **学习率未衰减。** PPO 受益于线性学习率衰减至零。恒定学习率通常更差。
- **重要性比率计算错误。** 为了数值稳定性，应始终使用 `ε = 0.2`，而不是 `0.1`。
- **梯度符号错误。** 最大化替代目标 = *最小化* `ε = 0.2`。符号反转是最常见的 PPO 错误。

## 使用它

PPO 是 2026 年众多领域中的默认强化学习算法：

|  使用场景  |  PPO 变体  |
|----------|-------------|
|  MuJoCo / 机器人控制  |  PPO with 高斯策略, GAE(0.95)  |
|  Atari / 离散游戏  |  PPO with 分类策略, 滚动128步轨迹  |
|  大语言模型的RLHF  |  PPO with 对参考模型的KL惩罚, 响应结束时从RM获得奖励  |
|  大规模游戏智能体  |  IMPALA + PPO (AlphaStar, OpenAI Five)  |
|  推理大语言模型  |  GRPO (第12课) — 无评论家的PPO变体  |
|  仅偏好数据  |  DPO — PPO+KL的闭式合并, 无在线采样  |

PPO *损失形态* — 裁剪替代目标 + 价值 + 熵 — 是 DPO、GRPO 以及几乎所有 RLHF 管道的脚手架。

## 发布

保存为 `outputs/skill-ppo-trainer.md`：

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## 练习

1. **简单.** 在4×4 GridWorld上运行PPO，使用 `ε=0.2, K=4`。在匹配的环境步数下，与A2C（每个轨迹一个轮次）比较样本效率。
2. **中等.** 扫描 `ε=0.2, K=4`。绘制回报与环境步数曲线，并跟踪每次更新的平均KL散度。在此任务中，KL 在什么 `K ∈ {1, 4, 10, 30}` 下爆炸？
3. **困难.** 将裁剪替代目标替换为自适应KL惩罚（如果 `K ∈ {1, 4, 10, 30}`，则 `ε=0.2, K=4` 加倍；如果 `K`，则减半）。比较最终回报、稳定性和无裁剪性。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  重要性比率  |  "r_t(θ)"  |  `π_θ(a\ | s) / π_old(a\ | s)`; 与收集数据的策略的偏差。  |
|  裁剪替代目标  |  "PPO的主要技巧"  |  `min(r·A, clip(r, 1-ε, 1+ε)·A)`; 在有利一侧超出裁剪后梯度平坦。  |
|  信任区域  |  "TRPO / PPO意图"  |  限制每次更新的KL以保证单调改善。  |
|  KL惩罚  |  "软信任区域"  |  替代PPO: `L - β · KL(π_θ \ | \ |  π_old)`. Adaptive `β`.  |
|  裁剪比例  |  "裁剪触发的频率"  |  诊断指标 — 应为 0.1-0.3；超出表示调校不当。  |
|  多轮次训练  |  "数据重用"  |  每个轨迹上K个轮次；方差代价换取样本效率。  |
|  近似在策略  |  "主要在策略"  |  PPO名义上是在策略，但K>1个轮次安全地使用了略微偏离策略的数据。  |
|  PPO-KL  |  "另一种PPO"  |  KL惩罚变体；用于RLHF中，其中对参考模型的KL已经是约束。  |

## 延伸阅读

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — 该论文。
- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — TRPO, PPO的前身。
- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — 每个PPO超参数被消融分析。
- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — InstructGPT; PPO应用于RLHF的配方。
- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — 使用PyTorch的清晰现代阐述。
- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — 许多论文引用的单文件PPO参考实现。
- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — 语言模型上PPO的生产配方; 与第09课(RLHF)一起阅读。
- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — "37个代码级优化"论文; 哪些PPO技巧是关键的,哪些是传说。
