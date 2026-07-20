# Actor-Critic — A2C与A3C

> REINFORCE噪声很大。添加一个学习`V̂(s)`的评论家，将其从回报中减去，你就能得到一个期望相同但方差低得多的优势。这就是actor-critic。A2C同步运行；A3C跨线程运行。两者都是所有现代深度强化学习方法的思维模型。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段9·04（时序差分学习），阶段9·06（REINFORCE）
**时间：** ~75分钟

## 问题

朴素REINFORCE有效，但其方差大得可怕。蒙特卡洛回报`G_t`在不同回合间可能波动一个数量级。将噪声乘以`∇ log π`并求平均，得到的梯度估计器需要数千个回合才能使策略移动与远少次DQN更新相同的距离。

方差源于使用原始回报。如果减去一个基线`b(s_t)`（任何状态的函数，包括学习到的值），期望不变而方差下降。最佳可解基线是`V̂(s_t)`。现在乘以`∇ log π`的量是*优势*：

`A(s, a) = G - V̂(s)`

如果某个动作产生高于平均的回报，它就是好的；低于平均则是差的。带有学习到的评论家的REINFORCE就是*actor-critic*。评论家为actor提供了一个低方差的教师。这是2015年之后所有深度策略方法（A2C、A3C、PPO、SAC、IMPALA）的基础。

## 核心概念

![Actor-critic: policy net plus value net, TD residual as advantage](../assets/actor-critic.svg)

**两个网络，共享损失：**

- **Actor** `π_θ(a | s)`：策略。用于采样行动。通过策略梯度训练。
- **Critic** `π_θ(a | s)`：估计从状态出发的期望回报。训练以最小化`V_φ(s)`。

**优势。** 两种标准形式：

- *MC优势：* `A_t = G_t - V_φ(s_t)`。无偏，方差较高。
- *TD优势：* `A_t = G_t - V_φ(s_t)`。有偏（使用`A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`），方差低得多。也称为*TD残差* `V_φ`。

**n步优势。** 在两者之间插值：

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1`是纯TD。`n = ∞`是MC。大多数实现中，Atari使用`n = 5`，MuJoCo上的PPO使用`n = 2048`。

**广义优势估计（GAE）。** Schulman等人（2016）提出了所有n步优势的指数加权平均：

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

其中`λ ∈ [0, 1]`。`λ = 0`是TD（低方差，高偏差）。`λ = 1`是MC（高方差，无偏）。`λ = 0.95`是2026年的默认值——调节直到偏差/方差度盘达到你想要的位置。

**A2C：同步优势actor-critic。** 跨`N`个并行环境收集`T`步。计算每一步的优势。在合并的批次上更新actor和critic。重复。A3C的更简单、更可扩展的兄弟。

**A3C：异步优势actor-critic。** Mnih等人（2016）。生成`N`个工作线程，每个运行一个环境。每个工作者在自身的轨迹上本地计算梯度，然后异步地将它们应用于共享参数服务器。无需经验回放缓冲区——工作者通过运行不同轨迹去相关。A3C证明了可以在CPU上大规模训练。2026年，基于GPU的A2C（批量并行环境）占主导地位，因为GPU需要大批次。

**组合损失。**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

三项：策略梯度损失、值回归、熵奖励。`c_v ~ 0.5`、`c_e ~ 0.01`是典型起点。

## 动手构建

### 第一步：评论家

线性评论家`V_φ(s) = w · features(s)`通过均方误差更新：

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

在表格环境中，评论家在几百回合内收敛。在Atari上，用共享CNN主干+值头替换线性评论家。

### 第二步：n步优势

给定长度为`T`的轨迹和自举的最终`V(s_T)`：

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns`是评论家目标。`advantages`是乘以`∇ log π`的量。

### 第三步：组合更新

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

在策略上，每次更新一个轨迹，actor和critic使用不同学习率。

### 第四步：并行化（A3C vs A2C）

- **A3C：** 启动`N`个线程。每个运行自己的环境和自己的一次前向传播。周期性地将梯度更新推送到共享主节点。主节点不加锁——竞争是可以的，只是增加噪声。
- **A2C：** 在单个进程中运行`N`个环境实例，将观测堆叠成`N`批次，批量前向传播，批量反向传播。更高的GPU利用率，确定性更强，更容易推理。2026年的默认选择。

我们的玩具代码为清晰起见是单线程的；重写为批量A2C只需三行numpy代码。

## 陷阱

- **评论家在actor梯度之前的偏差。** 如果评论家是随机的，其基线无信息量，你是在纯噪声上训练。在开启策略梯度之前，先预热评论家几百步，或者使用缓慢的actor学习率。
- **优势归一化。** 对每个批次的优势进行零均值/单位标准差归一化。以接近零的代价大幅稳定训练。
- **共享主干。** 对于图像输入，为actor和critic使用共享特征提取器。分离的输出头。共享特征在两种损失上都搭便车。
- **在策略契约。** A2C的数据仅用于一次更新。更多次使用会导致梯度有偏（重要性采样校正是PPO添加的内容）。
- **熵崩溃。** 如果没有`c_e > 0`，策略会在几百次更新后变得近乎确定性，停止探索。
- **奖励尺度。** 优势大小取决于奖励尺度。对不同任务归一化奖励（例如，使用运行标准差除法）以获得一致的梯度大小。

## 使用它

A2C/A3C在2026年很少是最终选择，但它们是后续所有改进所基于的架构：

|  方法  |  与A2C的关系 |
|--------|----------------|
|  PPO  |  A2C + 裁剪重要性比例用于多轮更新 |
|  IMPALA  |  A3C + V-trace离策略校正 |
|  SAC（Phase 9 · 07）  |  离策略A2C + 软值评论家（下一课） |
|  GRPO（Phase 9 · 12）  |  无评论家的A2C — 群体相对优势 |
|  DPO  |  将A2C压缩为偏好排序损失，无需采样 |
|  AlphaStar / OpenAI Five  |  A2C + 联赛训练 + 模仿预训练 |

如果你在2026年的论文中看到“优势”，就想到演员-评论家(actor-critic)。

## 发布

保存为 `outputs/skill-actor-critic-trainer.md`：

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
```

## 练习

1. **简单。** 在4×4 GridWorld上使用MC优势(`G_t - V(s_t)`)训练演员-评论家。与第06课中的带运行均值基线的REINFORCE比较样本效率。
2. **中等。**切换到TD残差优势(`G_t - V(s_t)`)。测量优势批次的方差。下降了多少？
3. **困难。**实现GAE(λ)。扫描`G_t - V(s_t)`。绘制最终回报与样本效率的关系图。此任务的偏差/方差最佳点在哪里？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  演员  |  “策略网络”  |  `π_θ(a\ | s)`，通过策略梯度更新。 |
|  评论家  |  “价值网络”  |  `V_φ(s)`，通过均方误差回归到回报/TD目标进行更新。 |
|  优势  |  “比平均好多少”  |  `A(s, a) = Q(s, a) - V(s)`或其估计量。`∇ log π`的乘数。 |
|  TD残差  |  “δ”  |  `δ_t = r + γ V(s') - V(s)`；单步优势估计。 |
|  GAE  |  “插值旋钮”  |  由`λ`参数化的n步优势的指数加权和。 |
|  A2C  |  “同步演员-评论家”  |  跨环境批处理；每次推出一个梯度步。 |
|  A3C  |  “异步演员-评论家”  |  工作线程将梯度推送到共享参数服务器。原始论文；在2026年不太常见。 |
|  自举  |  “在截止点使用V”  |  截断推出，添加`γ^n V(s_{t+n})`以闭合求和。 |

## 延伸阅读

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C，原始的异步演员-评论家论文。
- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — GAE。
- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — 基础；与第9章关于当评论家是神经网络时的函数近似结合。
- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — 具有V-trace离策略校正的可扩展分布式演员-评论家。
- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — 值得阅读的生产级A2C/PPO实现。
- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — 双时间尺度演员-评论家分解的基础收敛结果。
