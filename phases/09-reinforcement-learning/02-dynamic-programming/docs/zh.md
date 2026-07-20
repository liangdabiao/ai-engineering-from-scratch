# 动态规划——策略迭代与值迭代

> 动态规划是带有作弊的强化学习。你已经知道转移函数和奖励函数；你只需要迭代贝尔曼方程，直到`V`或`π`停止变化。它是每一个基于采样的方法试图接近的基准。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段9·01（MDP）
**时间：** ~75分钟

## 问题

你有一个已知模型的MDP：你可以查询任何状态-动作对的`P(s' | s, a)`和`R(s, a, s')`。库存管理员知道需求分布。棋盘游戏具有确定性转移。网格世界（GridWorld）是四行Python代码。你拥有一个*模型*。

无模型强化学习（Q-learning、PPO、REINFORCE）是为你没有模型的情况而发明的——你只能从环境中采样。但当你确实拥有模型时，有更快、更好的方法：动态规划。贝尔曼在1957年设计了它们。它们至今仍定义着正确性：当人们说“这个MDP的最优策略”时，他们指的是动态规划会返回的那个策略。

你在2026年需要它们有三个原因。首先，强化学习研究中的每个表格型环境（GridWorld、FrozenLake、CliffWalking）都是用动态规划求解以产生黄金标准策略。其次，精确的值让你可以*调试*采样方法：如果Q-learning对`V*(s_0)`的估计与动态规划答案相差30%，你的Q-learning存在错误。第三，现代离线强化学习和规划方法（MCTS、AlphaZero的搜索、阶段9·10中的基于模型的强化学习）都在学习到的或给定的模型上迭代贝尔曼备份。

## 核心概念

![Policy iteration and value iteration, side by side](../assets/dp.svg)

**两种算法，都是贝尔曼方程的不动点迭代。**

**策略迭代。** 交替进行两个步骤，直到策略停止变化。

1. *策略评估：* 给定策略`π`，通过反复应用`V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`直到收敛来计算`V^π`。
2. *策略改进：* 给定`π`，使`V^π`相对于`V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`贪心：`V^π`。

收敛是有保证的，因为(a)每个改进步骤要么保持`π`不变，要么严格增加某些状态的`V^π`，(b)确定性策略的空间是有限的。即使对于大的状态空间，通常也在约5–20次外部迭代内收敛。

**值迭代。** 将评估和改进合并为一次扫描。应用贝尔曼*最优性*方程：

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

重复直到`max_s |V_{new}(s) - V(s)| < ε`。最后通过取贪心动作来提取策略。每次迭代严格更快——没有内部评估循环——但通常需要更多迭代才能收敛。

**广义策略迭代（GPI）。** 统一框架。值函数和策略被锁定在一个双向改进循环中；任何驱动两者走向相互一致的方法（异步值迭代、修改的策略迭代、Q-learning、演员-评论家、PPO）都是GPI的实例。

**为什么`γ < 1`很重要。** 贝尔曼算子在无穷范数下是一个`γ`-压缩映射：`||T V - T V'||_∞ ≤ γ ||V - V'||_∞`。压缩映射意味着唯一不动点和几何收敛。去掉`γ < 1`你就失去了保证——你需要一个有限时域或一个吸收终止状态。

```figure
value-iteration-gamma
```

## 动手构建

### 步骤1：构建GridWorld MDP模型

使用与第01课相同的4×4 GridWorld。我们添加一个随机变体：以概率`0.1`，智能体滑向一个随机的垂直方向。

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)`返回一个`(s', r, p)`列表。这就是整个模型。

### 步骤2：策略评估

给定策略`π(s) = {action: prob}`，迭代贝尔曼方程直到`V`停止变化：

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### 步骤3：策略改进

将`π`替换为相对于`V`的贪心策略。如果`π`没有变化，则返回——我们已达到最优。

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### 步骤4：将它们拼接在一起

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

4×4网格上的典型收敛：4–6次外部迭代。输出`V*(0,0) ≈ -6`和一个严格减少步数的策略。

### 步骤5：值迭代（单循环版本）

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

相同的不动点，更少的代码行。

## 陷阱

- **忘记处理终止状态。** 如果你对吸收状态应用贝尔曼方程，它仍然会选出一个什么也不改变的“最佳动作”。用`if s == terminal: V[s] = 0`保护。
- **无穷范数与L2收敛。** 使用`if s == terminal: V[s] = 0`，而不是平均。理论保证是基于无穷范数的。
- **原地更新与同步更新。** 原地更新`if s == terminal: V[s] = 0`（Gauss-Seidel）比单独的`max |V_new - V|`字典（Jacobi）收敛更快。生产代码使用原地更新。
- **策略平局。** 如果两个动作具有相同的Q值，`if s == terminal: V[s] = 0`可能每次迭代以不同方式打破平局，导致“策略稳定”检查振荡。使用稳定的平局打破方式（固定顺序中的第一个动作）。
- **状态空间爆炸。** 动态规划每次扫描是`if s == terminal: V[s] = 0`。适用于最多约10⁷个状态。超出此范围，你需要函数近似（阶段9·05及以后）。

## 使用它

在2026年，动态规划是正确性的基准和规划器的内部循环：

|  使用场景  |  方法  |
|----------|--------|
|  精确求解一个小型表格型MDP  |  值迭代（更简单）或策略迭代（更少外部步骤）  |
|  验证Q-learning/PPO实现  |  在玩具环境上与DP最优V*比较  |
| 基于模型的强化学习(Model-based RL) (Phase 9 · 10) | 基于学习到的转移模型的贝尔曼备份(Bellman backup on a learned transition model) |
| AlphaZero / MuZero 中的规划(Planning in AlphaZero / MuZero) | 蒙特卡洛树搜索(Monte Carlo Tree Search) = 异步贝尔曼备份(async Bellman backup) |
| 离线强化学习(Offline RL) (CQL, IQL) | 保守Q迭代(Conservative Q-iteration) — 对分布外(OOD)动作施加惩罚的动态规划(DP with a penalty on OOD actions) |

每当有人说"最优价值函数(the optimal value function)"时，他们指的是"动态规划的不动点(the DP fixed point)"。当你在论文中看到`V*`或`Q*`时，想象这个循环。

## 发布

保存为 `outputs/skill-dp-solver.md`：

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## 练习

1. **简单(Easy).** 在带有`γ ∈ {0.9, 0.99}`的4×4网格世界(GridWorld)上运行值迭代(Value Iteration)。需要多少次扫描(sweep)直到`max |ΔV| < 1e-6`？将`V*`打印为4×4网格。
2. **中等(Medium).** 在*随机*网格世界（滑倒概率`γ ∈ {0.9, 0.99}`）上比较策略迭代(Policy Iteration)与值迭代。统计：扫描次数、墙钟时间(wall-clock time)、最终`max |ΔV| < 1e-6`。哪个在迭代次数上收敛更快？在墙钟时间上呢？
3. **困难(Hard).** 构建改进的策略迭代(Modified Policy Iteration)：在评估步骤中只运行`γ ∈ {0.9, 0.99}`次扫描而不是直到收敛。针对`0.1`绘制`max |ΔV| < 1e-6`误差(error)与`V*`的关系图。该曲线告诉您关于评估/改进权衡的什么信息？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
| 策略迭代(Policy Iteration) | "DP算法(DP algorithm)" | 交替进行评估(`V^π`)和改进(相对于`V^π`的贪婪(greedy)`π`)直到策略停止变化。 |
| 值迭代(Value Iteration) | "更快的DP(Faster DP)" | 一次扫描中应用贝尔曼最优性备份(Bellman optimality backup)；几何收敛到`V*`。 |
| 贝尔曼算子(Bellman Operator) | "递归(The recursion)" | `(T V)(s) = max_a Σ P (r + γ V(s'))`；在sup范数(sup-norm)下的`γ`-压缩(Contraction)。 |
| 压缩(Contraction) | "为什么DP收敛(Why DP converges)" | 任何满足 `\\ | \\ | T x - T y\\ | \\ |  ≤ γ \\ | \\ | x - y\\ | \\ | ` 的算子@@SKIP0000@@都有唯一的不动点(fixed point)。 |
| 广义策略迭代(GPI) | "一切都是DP(Everything is DP)" | 广义策略迭代(Generalized Policy Iteration)：使`V`和`π`达到相互一致(mutual consistency)的任何方法。 |
| 同步更新(Synchronous Update) | "Jacobi风格(Jacobi-style)" | 在一次扫描中全部使用旧的`V`；可清晰分析但较慢。 |
| 就地更新(In-place Update) | "Gauss-Seidel风格(Gauss-Seidel-style)" | 使用正在更新的`V`；实践中收敛更快。 |

## 延伸阅读

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — 策略迭代和值迭代的经典表述。
- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — 压缩映射(Contraction Mapping)论证的严谨处理。
- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — 改进的策略迭代(Modified Policy Iteration)及其收敛分析。
- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — 原始的策略迭代论文。
- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — 从DP到近似DP/深度强化学习的桥梁，后续每一课都会用到。
