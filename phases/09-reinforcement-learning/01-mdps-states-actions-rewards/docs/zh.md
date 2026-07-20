# MDP、状态、动作与奖励

> 马尔可夫决策过程(Markov Decision Process)包含五个要素：状态、动作、转移、奖励、折扣因子。强化学习中的一切——Q学习、PPO、DPO、GRPO——都在此框架上优化。只需学习一次，即可免费理解后续的强化学习内容。

**类型：** 学习
**语言：** Python
**先修知识：** 阶段1·06（概率与分布）、阶段2·01（机器学习分类）
**时间：** 约45分钟

## 问题

你正在编写一个国际象棋机器人。或者一个库存规划器。或者一个交易代理。或者训练推理模型的PPO循环。四个不同的领域，一个惊人的事实：它们都归结为同一个数学对象。

监督学习给你`(x, y)`对，要求你拟合一个函数。强化学习没有标签——只有一连串的状态、你采取的动作以及一个标量奖励。这一步棋赢了吗？补货决策省钱了吗？交易盈利了吗？大语言模型刚刚生成的token是否让评判者给出了更高的奖励？

在形式化之前，你无法从这一连串数据中学习。“我看到了什么”、“我做了什么”、“接下来发生了什么”、“那有多好”——每个都必须成为你可以推理的对象。这种形式化就是马尔可夫决策过程(Markov Decision Process)。本阶段的所有强化学习算法，包括最后的RLHF和GRPO循环，都在此框架上优化。

## 核心概念

![Markov decision process: states, actions, transitions, rewards, discount](../assets/mdp.svg)

**五个对象。**

- **状态** `S`。代理决策所需的一切。在网格世界中，是单元格。在国际象棋中，是棋盘。在大语言模型中，是上下文窗口加上任何记忆。
- **动作** `S`。选择。上/下/左/右移动。走一步棋。生成一个token。
- **转移** `S`。给定状态`A`和动作`P(s' | s, a)`，下一状态的概率分布。国际象棋中确定，库存中随机，大语言模型解码中近似确定。
- **奖励** `S`。标量信号。赢=+1，输=-1。收入减去成本。GRPO中的对数似然比项。
- **折扣因子** `S`。未来奖励相对当前的重要程度。`A`提供约100步的视野；`P(s' | s, a)`提供约10步。

**马尔可夫性质** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`。未来只取决于当前状态。如果不满足，则状态表示不完整——这不是方法的失败，而是状态的失败。

**策略与回报。** 策略`π(a | s)`是从状态到动作分布的映射。回报`G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …`是未来奖励的折扣和。价值`V^π(s) = E[G_t | s_t = s]`是从`s`出发在策略`π`下的期望回报。Q值`Q^π(s, a) = E[G_t | s_t = s, a_t = a]`是从特定动作开始的期望回报。每个强化学习算法估计其中之一，然后相应地改进`π`。

**贝尔曼方程。** 本阶段所有内容都使用的固定点方程：

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

这些方程将期望回报拆分为“这一步的奖励”加上“当前位置的折扣价值”。递归。阶段9中的每个算法要么迭代此方程至收敛（动态规划），要么从中采样（蒙特卡洛），要么单步自举（时序差分）。

```figure
discount-horizon
```

## 动手构建

### 第1步：一个微小的确定性MDP

一个4×4网格世界。代理从左上角开始，终点在右下角，每步奖励-1，动作`{up, down, left, right}`。参见`code/main.py`。

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

五行代码。这就是整个环境。确定性转移，恒定的步长惩罚，吸收终止状态。

### 第2步：执行一个策略

策略是从状态到动作分布的函数。最简单的是均匀随机。

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

运行随机策略1000次。对这个4×4棋盘，平均回报约为-60到-80。最优回报为-6（直线下右路径）。缩小这个差距就是阶段9的全部内容。

### 第3步：通过贝尔曼方程精确计算`V^π`

对于小MDP，贝尔曼方程是一个线性系统。枚举状态，应用期望，迭代直到值不再变化。

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

这是迭代策略评估。它是Sutton & Barto中的第一个算法，也是后续所有强化学习方法的理论基础。

### 第4步：`γ`是具有物理意义的超参数

有效视野大约为`1 / (1 - γ)`。`γ = 0.9`→10步。`γ = 0.99`→100步。`γ = 0.999`→1000步。

太低会导致代理目光短浅。太高则信用分配变得嘈杂，因为许多早期步骤共同承担远期奖励的责任。大语言模型的RLHF通常使用`γ = 1`，因为回合短且有界。控制任务使用`0.95–0.99`。长视野策略游戏使用`0.999`。

## 陷阱

- **非马尔可夫状态。** 如果需要最后三个观测才能决策，那么“状态”不仅仅是当前观测。解决方法：堆叠帧（DQN在Atari上堆叠4帧）或使用循环状态（LSTM/GRU处理观测序列）。
- **稀疏奖励。** 仅在获胜时给予奖励使得在大状态空间中学习几乎不可能。塑造奖励（中间信号）或通过模仿引导（阶段9·09）。
- **奖励欺骗。** 优化代理奖励常常导致病态行为。OpenAI的赛艇代理原地打转收集能量道具，从不完成比赛。始终根据目标结果定义奖励，而不是代理。
- **折扣因子错误指定。** 在无限视野任务中设`γ = 1`会使所有价值无限大。始终用有限视野或`γ < 1`做上限。
- **奖励尺度。** {+100, -100}与{+1, -1}的奖励给出相同的最优策略，但梯度幅度差异巨大。在输入PPO/DQN前，归一化到`γ = 1`左右。

## 使用它

2026年的技术栈在触及代码之前将每个强化学习流程简化为一个MDP：

|  情景  |  状态  |  动作  |  奖励  |  γ  |
|-----------|-------|--------|--------|---|
|  控制（运动、操纵）  |  关节角度+速度  |  连续力矩  |  任务特定塑造  |  0.99  |
|  游戏（国际象棋、围棋、扑克）  |  棋盘+历史  |  合法移动  |  赢=+1 / 输=-1  |  1.0（有限）  |
|  库存/定价  |  库存+需求  |  订购数量  |  收入-成本  |  0.95  |
| 面向LLM的RLHF  |  上下文Token  |  下一个Token  |  末尾的奖励模型得分  |  1.0（回合约200个Token） |
| 用于推理的GRPO  |  提示+部分响应  |  下一个Token  |  末尾验证器0/1  |  1.0 |

在编写任何训练循环之前，先写出五个元组。大多数“RL不起作用”的bug报告都追溯到纸面上就有问题的MDP形式化。

## 发布

保存为 `outputs/skill-mdp-modeler.md`：

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## 练习

1. **简单.** 在`code/main.py`中实现4×4网格世界和随机策略rollout。运行10,000个回合。报告回报的均值和标准差。与最优回报(-6)进行比较。
2. **中等.** 使用`code/main.py`和`policy_evaluation`运行均匀随机策略。将`γ ∈ {0.5, 0.9, 0.99}`以4×4网格形式打印出来。解释为什么在较大的`V`下接近终止状态的state值增长更快。
3. **困难.** 将GridWorld变为随机：每个动作以概率`code/main.py`滑向相邻方向。重新评估均匀策略。`policy_evaluation`变好了还是变差了？为什么？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  MDP  |  “强化学习设置”  |  满足马尔可夫性质的元组`(S, A, P, R, γ)`。 |
|  状态  |  “智能体所见”  |  所选策略类下未来动态的充分统计量。 |
|  策略  |  “智能体的行为”  |  条件分布`π(a \ |  s)` or deterministic map `s → a`。 |
|  回报  |  “总奖励”  |  从当前步开始的折扣和`Σ γ^t r_t`。 |
|  值  |  “一个状态有多好”  |  在`π`下从`s`开始的期望回报。 |
|  Q值  |  “一个动作有多好”  |  在`π`下从`s`开始、第一个动作为`a`的期望回报。 |
|  贝尔曼方程  |  “动态规划递归”  |  值/Q的不动点分解，分解为单步奖励加上折扣后继值。 |
|  折扣`γ`  |  “未来vs现在”  |  对远未来奖励的几何权重；有效视界`~1/(1-γ)`。 |

## 延伸阅读

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书。第3章涵盖MDP和贝尔曼方程；第1章阐述了奖励假设，它是后续每一课的基础。
- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — 贝尔曼方程的起源。
- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — 从深度强化学习角度的简明MDP入门。
- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — 关于MDP和精确求解方法的运筹学参考。
- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — 将MDP作为动态规划特化的最清晰推导。
