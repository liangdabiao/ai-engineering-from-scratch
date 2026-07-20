# 蒙特卡洛方法——从完整幕中学习

> 动态规划需要模型。蒙特卡洛除了幕之外什么都不需要。运行策略，观察回报，求平均。这是强化学习中最简单的想法——也是解锁后续一切的关键。

**类型：** 构建
**语言：** Python
**前置知识：** 第9阶段·01（马尔可夫决策过程）、第9阶段·02（动态规划）
**时间：** 约75分钟

## 问题

动态规划很优雅，但它假设你可以对每个状态和动作查询`P(s' | s, a)`。现实世界中几乎没有事物如此工作。机器人无法解析地计算关节力矩后摄像头像素的分布。定价算法无法对所有可能的客户反应进行积分。大语言模型无法枚举一个token之后的所有可能延续。

你需要一种只需能够从环境中*采样*的方法。运行策略。获取轨迹`s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`。用它来估计价值。这就是蒙特卡洛。

从动态规划到蒙特卡洛的转变在哲学上很重要：我们从*已知模型+精确备份*转向*采样轨迹+平均回报*。方差增加了，但适用性爆炸性增长。此课之后的所有强化学习算法——时序差分、Q学习、REINFORCE、PPO、GRPO——本质上是蒙特卡洛估计器，有时在之上叠加了自助法。

## 核心概念

![Monte Carlo: rollout, compute returns, average; first-visit vs every-visit](../assets/monte-carlo.svg)

**核心思想，一句话：** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)`，其中`G^{(i)}(s)`是在策略`π`下访问`s`后观察到的回报。

**首次访问vs每次访问蒙特卡洛。** 给定一个多次访问状态`s`的幕，首次访问蒙特卡洛只计算第一次访问的回报；每次访问蒙特卡洛计算所有访问的回报。两者在极限中都是无偏的。首次访问更易于分析（独立同分布样本）。每次访问每幕使用更多数据，通常在实践中收敛更快。

**增量式平均。** 无需存储所有回报，更新运行平均值：

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

重新整理：`V_new = V_old + α · (target - V_old)`，其中`α = 1/n`。将`1/n`替换为恒定步长`α ∈ (0, 1)`，你就得到了一个非平稳的蒙特卡洛估计器，可以跟踪`π`的变化。这一步是从蒙特卡洛到时序差分再到所有现代强化学习算法的整个跳跃。

**探索现在是一个问题。** 动态规划通过枚举触及了每个状态。蒙特卡洛只看到策略访问的状态。如果`π`是确定性的，整个状态空间区域永远不会被采样，它们的价值估计永远停留在零。三种修复方法，按历史顺序：

1. **探索性起点。** 从随机（s，a）对开始每一幕。保证覆盖；实践中不现实（你不能将机器人“重置”到任意状态）。
2. **ε-贪婪。** 相对于当前Q贪婪行动，但以`ε`的概率随机选择动作。所有状态-动作对渐进地得到采样。
3. **离线策略蒙特卡洛。** 在行为策略`ε`下收集数据，通过重要性采样学习目标策略`μ`。方差高，但它是通往经验回放方法（如深度Q网络）的桥梁。

**蒙特卡洛控制。** 评估 → 改进 → 评估，就像策略迭代一样，但评估是基于采样的：

1. 运行`π`，获取一幕。
2. 根据观察到的回报更新`π`。
3. 使`π`相对于`Q(s, a)`成为ε-贪婪。
4. 重复。

在温和条件下（每个状态-动作对无限频繁访问，`α`满足Robbins-Monro），以概率1收敛到`Q*`和`π*`。

```figure
epsilon-greedy
```

## 动手构建

### 步骤1：轨迹展开 → (s, a, r)列表

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

没有模型，只有`env.reset()`和`env.step(s, a)`。与gym环境相同的接口，但更精简。

### 步骤2：计算回报（反向扫描）

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

一次遍历，`O(T)`。反向递推`G_t = r_{t+1} + γ G_{t+1}`避免了重新求和。

### 步骤3：首次访问蒙特卡洛评估

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

三行代码完成工作：首次访问时标记状态为已见，增加计数，更新运行均值。

### 步骤4：ε-贪婪蒙特卡洛控制（在策略）

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### 步骤5：与动态规划黄金标准比较

随着幕数→∞，你对`V^π`的蒙特卡洛估计应与第2课中的动态规划结果一致。实践中：在4×4网格世界上进行50,000幕，可在`~0.1`范围内接近动态规划答案。

## 陷阱

- **无限幕。** 蒙特卡洛要求幕*终止*。如果你的策略可以无限循环，则限制`max_steps`并将限制视为隐式失败。随机策略下的网格世界通常会超时——这很正常，只需确保正确计数。
- **方差。** 蒙特卡洛使用完整回报。在长幕中，方差巨大——末尾一次不幸的奖励会使`max_steps`同样变化。时序差分方法（第4课）通过自助法削减了这一点。
- **状态覆盖。** 在具有平局的Q上使用贪婪蒙特卡洛只会尝试一个动作。你*必须*探索（ε-贪婪、探索性起点、置信上界）。
- **非平稳策略。** 如果`max_steps`发生变化（如在蒙特卡洛控制中），旧回报来自不同策略。常数α蒙特卡洛可以处理；样本平均蒙特卡洛不能。
- **离线策略重要性采样。** 权重`max_steps`在轨迹中相乘。方差随长度爆炸。使用每决策加权重要性采样或切换到时序差分来限制。

## 使用它

蒙特卡洛方法在2026年的角色：

|  使用案例  |  为何选择蒙特卡洛  |
|----------|--------|
|  短视界游戏（二十一点、扑克）  |  幕自然终止；回报干净。  |
|  已记录策略的离线评估  |  对存储轨迹的折扣回报求平均。  |
|  蒙特卡洛树搜索（AlphaZero）  |  从树叶子出发的蒙特卡洛轨迹引导选择。  |
|  LLM强化学习评估  |  计算给定策略下对采样完成的平均奖励。  |
|  PPO中的基线估计  |  优势目标`A_t = G_t - V(s_t)`使用MC`G_t`。  |
|  教授强化学习  |  实际有效的最简算法——去掉自助法以观察核心。  |

现代深度强化学习算法（PPO、SAC）通过`n`步收益或GAE在纯MC（完整回报）和纯TD（单步自助法）之间插值。两个端点都是同一估计器的实例。

## 发布

保存为 `outputs/skill-mc-evaluator.md`：

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## 练习

1. **简单.** 在4×4的GridWorld上实现均匀随机策略的首访MC评估。运行10,000个回合。绘制`V(0,0)`作为回合数的函数，并与动态规划答案比较。
2. **中等.** 实现带`V(0,0)`的ε-贪婪MC控制。比较20,000个回合后的平均回报。曲线长什么样？偏差-方差权衡在哪里？
3. **困难.** 实现基于重要性采样的*离策略*MC：在均匀随机策略`V(0,0)`下收集数据，估算确定性最优策略`ε ∈ {0.01, 0.1, 0.3}`的`μ`。比较普通IS、每决策IS和加权IS。哪种方差最低？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  蒙特卡洛  |  "随机采样"  |  通过对分布中的独立同分布样本求平均来估计期望。  |
|  回报`G_t`  |  "未来奖励"  |  从步骤`t`到回合结束的折扣奖励之和：`Σ_{k≥0} γ^k r_{t+k+1}`。  |
|  首访MC  |  "每个状态只计数一次"  |  一个回合中仅有首次访问贡献于值估计。  |
|  每次访问MC  |  "使用所有访问"  |  每次访问都贡献；略微有偏但样本效率更高。  |
|  ε-贪婪  |  "探索噪声"  |  以概率`1-ε`选择贪婪动作；以概率`ε`选择随机动作。  |
|  重要性采样  |  "纠正从错误分布采样的偏差"  |  通过`π(a\ | s)/μ(a\ | s)` products to estimate `V^π` from `μ`数据对回报重新加权。  |
|  在策略  |  "从自己的数据学习"  |  目标策略=行为策略。标准MC、PPO、SARSA。  |
|  离策略  |  "从他人的数据学习"  |  目标策略≠行为策略。重要性采样MC、Q学习、DQN。  |

## 延伸阅读

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf)——标准处理方法。
- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf)——首访与每次访问分析。
- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf)——离策略MC与方差控制。
- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf)——现代低方差IS估计器。
- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf)——首次大规模实证证明MC/TD自博弈可收敛到超人水平；是本阶段后半部分每节课的概念前身。
