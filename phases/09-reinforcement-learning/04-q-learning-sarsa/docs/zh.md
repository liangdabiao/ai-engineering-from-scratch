# 时序差分 — Q学习与SARSA

> 蒙特卡洛等待回合结束。时序差分通过自举下一步价值估计，在每一步后更新。Q学习是离策略且乐观的；SARSA是在策略且谨慎的。两者都是一行代码。两者都是这一阶段所有深度强化学习方法的基石。

**类型：** 构建
**语言：** Python
**前置要求：** 第9阶段·01（马尔可夫决策过程），第9阶段·02（动态规划），第9阶段·03（蒙特卡洛）
**时间：** 约75分钟

## 问题

蒙特卡洛有效但有两个昂贵的要求。它需要能终止的回合，并且仅在最终回报到达后才更新。如果你的回合有1000步，MC要等1000步才更新。它是高方差、低偏差的，实际中很慢。

动态规划具有相反的特点——零方差的自举备份——但需要已知的模型。

时序差分学习折中处理。从一个单一转移 `(s, a, r, s')`，形成一步目标 `r + γ V(s')` 并将 `V(s)` 向它推进。无需模型，无需完整回合。由于在右侧使用了近似的 `V` 而有偏差，但方差远低于MC，且从第一步开始在线更新。

这是所有现代强化学习——DQN、A2C、PPO、SAC——的枢轴。第9阶段的剩余部分是在你将在本课中编写的一步TD更新之上构建的函数近似层和技巧。

## 核心概念

![Q-learning vs SARSA: off-policy max vs on-policy Q(s', a')](../assets/td.svg)

**针对 V 的 TD(0) 更新：**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

括号内的量是TD误差 `δ = r + γ V(s') - V(s)`。它是MC中 `G_t - V(s_t)` 的在线对应物。收敛需要 `α`满足Robbins-Monro条件（`Σ α = ∞`, `Σ α² < ∞`），并且所有状态被无限次访问。

**Q学习。** 一种用于控制的离策略时序差分方法：

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max` 假设从 `s'` 开始将遵循*贪心*策略，无论智能体实际采取什么动作。这种解耦使得Q学习能够学习 `Q*`，同时智能体通过ε-贪心探索。Mnih等（2015）将其转化为Atari上的深度Q学习（第05课）。

**SARSA。** 一种在策略的时序差分方法：

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

名称是元组 `(s, a, r, s', a')`。SARSA使用智能体*实际*下一步采取的动作 `a'`，而不是贪心的 `argmax`。对于正在运行的任何ε-贪心 `π`，收敛到 `Q^π`，在极限情况下 `ε → 0` 变为 `Q*`。

**悬崖行走的差异。** 在经典悬崖行走任务（掉落悬崖 = 奖励 -100）中，Q学习学习沿悬崖边缘的最优路径，但在探索期间偶尔会受惩罚。SARSA学习离悬崖一步之遥的更安全路径，因为它将探索噪声纳入其Q值。经过训练，两者都在 `ε → 0` 时达到最优。在实践中这很重要：当部署时实际发生探索，SARSA的行为更加保守。

**期望SARSA。** 用 `Q(s', a')` 在 `π` 下的期望值替换它：

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

方差低于SARSA（无 `a'` 的样本），相同的在策略目标。在现代教科书中通常是默认选项。

**n步TD与TD(λ)。** 通过等待 `n` 步后再自举，在TD(0)和MC之间插值。`n=1` 是TD，`n=∞` 是MC。TD(λ) 以几何权重 `(1-λ)λ^{n-1}` 对所有 `n` 进行平均。大多数深度强化学习使用介于3和20之间的 `n`。

```figure
qlearning-gridworld
```

## 动手构建

### 步骤1：基于ε-贪心策略的SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

八行。与Q学习的*唯一*区别是目标行。

### 步骤2：Q学习

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` 将目标与行为解耦。这一个符号就是在策略与离策略之间的区别。

### 步骤3：学习曲线

追踪每100回合的平均回报。Q学习在简单的确定性网格世界中收敛更快；SARSA在悬崖行走上更保守。在 `code/main.py` 的4×4网格世界中，两者在约2,000回合后接近最优，`α=0.1, ε=0.1`。

### 步骤4：与动态规划真值比较

运行价值迭代（第02课）以获得 `Q*`。检查 `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`。一个健康的表格型TD智能体在10,000回合后落在4×4网格世界的 `~0.5` 内。

## 陷阱

- **初始Q值很重要。** 乐观初始化（在负奖励任务中`Q = 0`）鼓励探索。悲观初始化可能永远困住贪心策略。
- **α调度。** 对于非平稳问题，常数 `Q = 0` 即可。衰减的 `α` 在理论上能收敛，但实际中太慢——将 `α_n = 1/n` 固定于 `α` 并监控学习曲线。
- **ε调度。** 从高值开始（`Q = 0`），衰减到 `α`。"GLIE"（无限探索下的极限贪心）是收敛条件。
- **Q学习中的最大偏差。** 当 `α` 有噪声时，`Q = 0` 算子存在向上偏差。导致高估——Hasselt的双Q学习（第05课中DDQN使用）用两个Q表解决了这个问题。
- **非终止回合。** TD可以在没有终止状态的情况下学习，但你需要要么限制步数，要么在限制处正确处理自举。标准做法：将限制视为非终止，继续自举。
- **状态哈希。** 如果状态是元组/张量，使用可哈希的键（元组，不要列表；浮点元组要四舍五入，不要原始值）。

## 使用它

2026年时序差分概览：

|  任务  |  方法  |  原因  |
|------|--------|--------|
|  小型表格环境  |  Q学习  |  直接学习最优策略。  |
|  在策略的安全关键  |  SARSA / 期望SARSA  |  探索期间保守。  |
| 高维状态  |  DQN (第9阶段·05)  |  带经验回放和目标网络的神经网络Q函数。 |
| 连续动作  |  SAC / TD3 (第9阶段·07)  |  对Q网络进行TD更新；策略网络生成动作。 |
| LLM强化学习 (基于奖励模型)  |  PPO / GRPO (第9阶段·08, 12)  |  通过GAE计算TD风格优势的演员-评论家。 |
| 离线强化学习  |  CQL / IQL (第9阶段·08)  |  带保守正则化的Q学习。 |

你在2026年论文中读到的“RL”有百分之九十是Q学习或SARSA的某种变体。在深入阅读之前，请将表格更新了然于心。

## 发布

保存为 `outputs/skill-td-agent.md`：

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## 练习

1. **简单.** 在4×4网格世界上实现Q学习和SARSA。绘制2000个回合的学习曲线（每100回合的平均回报）。谁收敛得更快？
2. **中等.** 构建一个悬崖行走环境（4×12，最后一行是悬崖，奖励-100并重置到起点）。比较Q学习和SARSA的最终策略。截取各自所走的路径。哪个更靠近悬崖？
3. **困难.** 实现双Q学习。在一个有噪声奖励的网格世界上（每步奖励增加高斯噪声σ=5），展示Q学习会高估`V*(0,0)`一个显著的量，而双Q学习不会。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
| TD误差  |  “更新信号”  |  `δ = r + γ V(s') - V(s)`，自举残差。 |
| TD(0)  |  “单步TD”  |  每步转移后仅使用下一状态的估计进行更新。 |
| Q学习  |  “离策略强化学习入门”  |  对下一状态动作取`max`的TD更新；无论行为策略如何都学习`Q*`。 |
| SARSA  |  “在策略Q学习”  |  使用实际下一动作的TD更新；学习当前ε-贪婪策略π的`Q^π`。 |
| 期望SARSA  |  “低方差SARSA”  |  用其在策略π下的期望替换采样的`a'`。 |
| GLIE  |  “正确的探索调度”  |  无限探索下的极限贪心；Q学习收敛所需。 |
| 自举  |  “在目标中使用当前估计”  |  区分TD与MC的关键。偏差的来源，但大幅降低方差。 |
| 最大化偏差  |  “Q学习高估”  |  对噪声估计取`max`会产生向上偏差；通过双Q学习修正。 |

## 延伸阅读

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 原始论文及收敛性证明。
- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — TD(0)、SARSA、Q学习、期望SARSA。
- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 最大化偏差的修正。
- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 期望SARSA的动机。
- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 提出SARSA的论文（当时称为“修正联结主义Q学习”）。
- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 将TD(0)推广到TD(n)，从Q学习到资格迹，再到PPO中的GAE的路径。
