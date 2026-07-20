# 多智能体强化学习

> 单智能体强化学习假设环境是平稳的。当将两个学习智能体放入同一世界时，这一假设就不成立了：每个智能体都是另一个智能体环境的一部分，并且两者都在变化。多智能体强化学习是一组技巧，用于在马尔可夫假设不再成立时使学习收敛。

**类型:** 构建
**语言:** Python
**前提条件:** 阶段9·04 (Q学习), 阶段9·06 (REINFORCE), 阶段9·07 (演员-评论家)
**时间:** 约45分钟

## 问题

机器人学习导航房间是单智能体强化学习问题。足球队不是。AlphaStar对抗星际争霸对手不是。竞价代理的市场不是。两辆车在四向停车处协商不是。许多对许多的现实世界问题不是。

在每个多智能体环境中，从任何一个智能体的视角来看，其他智能体*是*环境的一部分。随着它们学习和改变行为，环境变得非平稳。马尔可夫性质——“下一个状态仅取决于当前状态和我的动作”——被违反，因为下一个状态还取决于*其他*智能体选择了什么，而它们的策略是移动的目标。

这破坏了表格收敛性证明（Q学习的保证假设环境是平稳的）。它也破坏了朴素深度强化学习：智能体互相追逐循环，从不收敛到稳定策略。你需要多智能体特定技术：集中训练/分散执行、反事实基线、联赛玩法、自我对弈。

2026年应用：机器人集群、交通路由、自动驾驶车队、市场模拟器、多智能体LLM系统（阶段16），以及任何具有多个智能玩家的游戏。

## 核心概念

![Four MARL regimes: indep, centralized critic, self-play, league](../assets/marl.svg)

**形式化：马尔可夫博弈。** MDP的推广：状态 `S`，联合动作 `a = (a_1, …, a_n)`，转移 `P(s' | s, a)`，以及每个智能体的奖励 `R_i(s, a, s')`。每个智能体 `i` 在自身策略 `π_i`下最大化自身回报。如果奖励相同，则为**完全合作**。如果零和，则为**对抗性**。如果混合，则为**一般和**。

**核心挑战：**

- **非平稳性。** 从智能体 `i` 的视角，`P(s' | s, a_i)` 取决于 `π_{-i}`，而后者正在变化。
- **信用分配。** 对于共享奖励，是哪个智能体导致的？
- **探索协调。** 智能体必须探索互补策略，而不是冗余探索相同状态。
- **可扩展性。** 联合动作空间随 `P(s' | s, a_i)` 指数增长。
- **部分可观测性。** 每个智能体仅看到自身观测；全局状态隐藏。

**四种主要范式：**

**1. 独立Q学习/独立PPO (IQL, IPPO)。** 每个智能体学习自己的Q或策略，将其他智能体视为环境的一部分。简单，有时有效（尤其是经验回放作为平滑智能体建模技巧）。理论收敛：无。实践中：适用于松散耦合任务，对紧密耦合任务效果较差。

**2. 集中训练，分散执行 (CTDE)。** 最常见的现代范式。每个智能体有自己的*策略* `π_i`，条件于局部观测 `o_i` —— 部署时标准分散执行。在*训练*期间，集中评论家 `Q(s, a_1, …, a_n)` 条件于完整全局状态和联合动作。示例：
- **MADDPG** (Lowe等人, 2017)：每个智能体的集中评论家的DDPG。
- **COMA** (Foerster等人, 2017)：反事实基线——问“如果我采取了动作 `a'` 而不是实际动作，我的奖励会是多少？”——隔离我的贡献。
- **MAPPO** / **IPPO** 与共享评论家 (Yu等人, 2022)：具有集中值函数的PPO。2026年合作MARL中的主导方法。
- **QMIX** (Rashid等人, 2018)：值分解——具有单调混合的 `a'`。

**3. 自我对弈。** 同一智能体的两个副本相互对弈。对手的策略*是*来自过去快照的我的策略。AlphaGo / AlphaZero / MuZero。OpenAI Five。最适用于零和博弈；训练信号对称。

**4. 联赛玩法。** 自我对弈扩展到一般和/对抗性环境：保留过去和当前策略的种群，从联赛中采样对手，对抗训练。加入利用者（专门击败当前最佳策略）和主利用者（专门击败利用者）。AlphaStar (星际争霸II)。当游戏存在“石头剪刀布”策略循环时需要。

**通信。** 允许智能体向彼此发送学习消息 `m_i`。在合作环境中有效。Foerster等人(2016)表明可微分的智能体间通信可以端到端训练。今天的基于LLM的多智能体系统（阶段16）本质上用自然语言通信。

## 动手构建

本课使用一个6×6 GridWorld，包含两个合作智能体。它们从对角开始，必须到达一个共享目标。共享奖励：任一智能体仍在移动时每步 `-1`，两者都到达时 `+10`。见 `code/main.py`。

### 步骤1：多智能体环境

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*联合*动作空间为 `|A|² = 16`。全局状态是两个位置。

### 步骤2：独立Q学习

每个智能体运行自己的Q表，以联合状态为键。每一步：两者选择ε-贪婪动作，收集联合转移，每个用共享奖励更新自己的Q。

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

在此任务上有效，因为奖励密集且一致。在紧密耦合任务上失败（例如，一个智能体必须*等待*另一个）。

### 步骤3：具有分解值更新的集中式Q

使用一个关于联合动作 `Q(s, a_1, a_2)` 的Q。从共享奖励更新。执行时通过边缘化分散：`π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`。用指数联合动作空间换取*正确*全局视图。

### 步骤4：简单自我对弈（对抗性双智能体）

同一智能体，两个角色。训练智能体A对抗智能体B；经过 `K` 个回合后，将A的权重复制到B。对称训练，一致进展。微缩版AlphaZero配方。

## 陷阱

- **非平稳回放。** 独立智能体的经验回放比单智能体更差，因为旧转移是由现已过时的对手生成的。修复：按近期性重新标注或加权。
- **信用分配模糊性。** 长时间回合后的共享奖励；无法清晰说明哪个智能体贡献。修复：反事实基线(COMA)，或每个智能体的奖励塑形。
- **策略漂移/追逐。** 每个智能体的最佳响应随其他智能体的更新而改变。修复：集中评论家、慢学习率，或一次冻结一个。
- **通过协调进行奖励攻击。** 智能体发现设计者未预见的协调利用。竞价代理收敛到出价为零。修复：精心设计奖励、行为约束。
- **探索冗余。** 两个智能体探索相同的状态-动作对。修复：每个智能体的熵奖励，或角色条件。
- **联赛循环。** 纯自我对弈可能陷入主导循环。修复：具有多样化对手的联赛玩法。
- **样本爆炸。** `n` 个智能体 × 状态空间 × 联合动作。使用函数逼近近似；分解动作空间（每个智能体一个策略输出头）。

## 使用它

2026年MARL应用图谱：

|  领域  |  方法  |  备注  |
|--------|--------|-------|
| 合作导航 / 操作 | MAPPO / QMIX | CTDE；共享评论家 + 分散执行者。 |
| 双人游戏（国际象棋、围棋、扑克）||| 使用MCTS的自对弈（AlphaZero）||| 零和；对称训练。 |  |  |
| 复杂多人游戏（Dota、星际争霸）||| 联赛玩法 + 模仿预训练 | OpenAI Five、AlphaStar。 |  |
| 自动驾驶车队 | 带有注意力机制的CTDE MAPPO / PPO | 部分观测；可变团队规模。 |
| 拍卖市场 | 博弈论均衡 + RL | 平均场强化学习当`n` → ∞时。 |
| 基于大型语言模型的多智能体系统（阶段16）||| 自然语言通信 + 角色条件 | 在智能体规划层的RL循环。 |  |

到2026年，MARL最大的增长领域是基于大型语言模型的：由语言模型智能体组成的群体进行协商、辩论和构建软件。RL表现为对*轨迹级别*输出的偏好优化，而非标记级别（阶段16·03）。

## 发布

保存为 `outputs/skill-marl-architect.md`：

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## 练习

1. **简单。** 在双智能体合作GridWorld上训练独立Q学习。需要多少回合直到平均回报>0？绘制联合学习曲线。
2. **中等。** 添加一个“协调”任务：目标仅当两个智能体在同一回合都踩到它时才达成。独立Q还能收敛吗？什么出了问题？
3. **困难。** 实现一个集中式评论家用于MAPPO风格训练，并在协调任务上比较与独立PPO的收敛速度。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
| 马尔可夫博弈 | “多智能体MDP” | `(S, A_1, …, A_n, P, R_1, …, R_n)`；每个智能体有自己的奖励。 |
| CTDE | “集中训练，分散执行” | 训练时联合评论家；每个智能体的策略仅使用局部观测。 |
| IPPO | “独立PPO” | 每个智能体独立运行PPO。简单基线；常被低估。 |
| MAPPO | “多智能体PPO” | 具有以全局状态为条件的集中值函数的PPO。 |
| QMIX | “单调值分解” | `Q_tot = f_monotone(Q_1, …, Q_n)`允许分散化argmax。 |
| COMA | “反事实多智能体” | 优势 = 我的Q减去对我动作边际化的期望Q。 |
| 自对弈 | “智能体对过去的自己” | 单个智能体，两个角色；零和游戏的标准。 |
| 联赛玩法 | “种群训练” | 缓存过去的策略，从池中采样对手；处理策略周期。 |

## 延伸阅读

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — 带有集中评论家的CTDE。
- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — 用于信用分配的反事实基线。
- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — 具有单调性的值分解。
- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — PPO对多智能体强化学习出奇地强大。
- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — 大规模联赛玩法。
- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — 零和博弈中的纯自对弈。
- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — 包含教材中对多智能体设置的简要处理以及CTDE旨在解决的非平稳性问题。
- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — 涵盖合作、竞争和混合多智能体强化学习的综述，附有收敛结果。
