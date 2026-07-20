# 深度Q网络（DQN）

> 2013年：Mnih在原始像素上训练了一个Q学习网络，在七个Atari游戏上击败了所有经典强化学习智能体。2015年：扩展到49个游戏，发表在《自然》杂志，引发了深度强化学习时代。DQN是Q学习加上三个使函数近似稳定的技巧。

**类型：** 实践
**语言：** Python
**前置知识：** 阶段3·03（反向传播），阶段9·04（Q学习，SARSA）
**时间：** ~75分钟

## 问题

表格型Q学习需要为每个（状态，动作）对单独存储一个Q值。一个棋盘约有10⁴³个状态。一个Atari画面有210×160×3=100,800个特征。表格型强化学习在数千个状态时就已失效，更不用说数十亿了。

事后来看，修复方法显而易见：用神经网络替换Q表格，`Q(s, a; θ)`。但这个“显而易见”却花了数十年。用Q学习进行简单的函数近似在“致命三要素”——函数近似+自举+离策略学习——下会发散。Mnih等人（2013，2015）确定了三个稳定学习的工程技巧：

1. **经验回放** 去相关转移。
2. **目标网络** 冻结自举目标。
3. **奖励裁剪** 归一化梯度幅度。

Atari上的DQN是首次用单一架构和单一超参数集从原始像素解决数十个控制问题。此后构建的一切“深度强化学习”——DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57——都基于这个三技巧基础。

## 核心概念

![DQN training loop: env, replay buffer, online net, target net, Bellman TD loss](../assets/dqn.svg)

**目标。** DQN最小化神经网络Q函数上的一步时序差分损失：

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = 在线网络，每一步通过梯度下降更新。`θ^-` = 目标网络，定期从`θ`复制（每~10,000步）。`D` = 过去转移的经验回放缓冲区。

**三个技巧，按重要性排序：**

**经验回放。** 一个包含`~10⁶`个转移的环形缓冲区。每个训练步骤均匀随机采样一个小批量。这打破了时间相关性（连续帧几乎相同），让网络从稀有奖励转移中多次学习，并去相关连续梯度更新。没有它，带神经网络的同策略时序差分在Atari上发散。

**目标网络。** 在贝尔曼方程两边使用同一个网络`Q(·; θ)`会使目标每次更新都移动——“追逐自己的尾巴”。修复方法：保留第二个网络`Q(·; θ^-)`，权重冻结。每`C`步，复制`θ → θ^-`。这使回归目标在数千个梯度步骤中保持稳定。软更新`θ^- ← τ θ + (1-τ) θ^-`（用于DDPG、SAC）是更平滑的变体。

**奖励裁剪。** Atari奖励幅度从1到1000以上不等。裁剪到`{-1, 0, +1}`可防止任何单个游戏主导梯度。当奖励幅度很重要时错误；对于只关心符号的Atari很好。

**双重DQN。** Hasselt（2016）修复了最大化偏差：用在线网络*选择*动作，用目标网络*评估*它。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

直接替换，始终更好。默认使用它。

**其他改进（Rainbow，2017）：** 优先回放（更多采样高TD误差转移）、决斗架构（分离`V(s)`和优势头）、噪声网络（学习探索）、n步回报、分布Q（C51/QR-DQN）、多步自举。每个增加几个百分点；收益大致可叠加。

## 动手构建

这里的代码仅使用标准库且无numpy——我们在一个小型连续网格世界上使用手动实现的单隐藏层MLP，因此每个训练步骤以微秒运行。算法与大规模Atari DQN相同。

### 步骤1：经验回放缓冲区

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari容量约50,000；我们的玩具环境5,000足够。

### 步骤2：一个小型Q网络（手动实现MLP）

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

前向传播：线性→ReLU→线性。这就是整个网络。

### 步骤3：DQN更新

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

其形式与第04课的Q学习相同，但有两个不同：（a）我们通过可微的`Q(·; θ)`进行反向传播，而不是索引表格；（b）目标使用`Q(·; θ^-)`。

### 步骤4：外部循环

对于每个回合，在`Q(·; θ)`上执行ε-贪婪，将转移推入缓冲区，采样一个小批量，执行一个梯度步骤，定期同步`θ^- ← θ`。模式如下：

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

在我们的具有16维独热状态的小型网格世界上，智能体在大约500个回合内学习到接近最优的策略。在Atari上，将其扩展到2亿帧并添加CNN特征提取器。

## 陷阱

- **致命三要素。** 函数近似+离策略+自举可能发散。DQN通过目标网络和回放缓解；不要移除任何一个。
- **探索。** ε必须衰减，通常在训练的前约10%从1.0衰减到0.01。没有足够的早期探索，Q网络会收敛到局部盆地。
- **过估计。** `max`对有噪声的Q值有向上偏差。在生产中始终使用双重DQN。
- **奖励尺度。** 裁剪或归一化奖励；梯度幅度与奖励幅度成正比。
- **回放缓冲区冷启动。** 在缓冲区有几千个转移之前不要训练。早期在约20个样本上的梯度过拟合。
- **目标同步频率。** 太频繁≈无目标网络；太不频繁≈目标过时。Atari DQN使用10,000个环境步骤。经验法则：每训练视界的约1/100同步一次。
- **观察预处理。** Atari DQN堆叠4帧使状态成为马尔可夫。任何具有速度信息的环境都需要帧堆叠或循环状态。

## 使用它

到2026年，DQN很少是最先进的，但仍然是离策略算法的参考基准：

|  任务  |  首选方法  |  为何不用DQN？  |
|------|------------------|--------------|
|  离散动作、类似Atari  |  Rainbow DQN或Muesli  |  相同框架，更多技巧。  |
| 连续控制 | SAC / TD3 (第9阶段·07) | DQN没有策略网络。 |
| 在线策略/高吞吐量 | PPO (第9阶段·08) | 无经验回放缓冲区；易于扩展。 |
| 离线强化学习 | CQL / IQL / Decision Transformer | 保守Q目标，无自举爆炸。 |
| 大型离散动作空间（推荐系统） | 带动作嵌入的DQN，或IMPALA | 可以；装饰重要。 |
| 大语言模型强化学习 | PPO / GRPO | 序列级别，非步骤级别；损失不同。 |

这些经验仍然适用。经验回放和目标网络出现在SAC、TD3、DDPG、SAC-X、AlphaZero的自对弈缓冲区以及每种离线强化学习方法中。奖励裁剪以优势归一化的形式存在于PPO中。架构就是蓝图。

## 发布

保存为 `outputs/skill-dqn-trainer.md`：

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## 练习

1. **简单。**运行`code/main.py`。绘制每幕回报曲线。运行均值超过-10需要多少幕？
2. **中等。**禁用目标网络（在贝尔曼目标两侧都使用在线网络）。测量训练不稳定性——回报是振荡还是发散？
3. **困难。**添加Double DQN：使用在线网络选择`code/main.py`，目标网络评估。在噪声奖励GridWorld上，有和没有Double DQN的情况下，经过1000幕后，比较`argmax a'`与真实`Q(s_0, best_a)`的偏差。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
| DQN | "深度Q学习" | 使用神经Q函数、经验回放和目标网络的Q学习。 |
| 经验回放 | "打乱转换" | 每一步梯度更新均匀采样的环型缓冲区；去相关数据。 |
| 目标网络 | "冻结自举" | 用于贝尔曼目标的Q的周期性副本；稳定训练。 |
| 致命三角 | "为什么RL发散" | 函数近似+自举+离策略=无收敛保证。 |
| Double DQN | "修复最大化偏差" | 在线网络选择动作，目标网络评估它。 |
| Dueling DQN | "V和A头" | 将Q分解为V + A - mean(A)；相同输出，更好的梯度流。 |
| Rainbow | "所有技巧" | DDQN + PER + dueling + n-step + noisy + distributional合一。 |
| PER | "优先经验回放" | 按TD误差大小比例采样转换。 |

## 延伸阅读

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)——2013年NeurIPS研讨会论文，开启了深度强化学习。
- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)——Nature论文，49游戏DQN。
- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)——DDQN。
- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)——dueling DQN。
- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)——堆叠技巧论文。
- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)——清晰的现代阐述。
- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)——教科书式处理“致命三角”（函数近似+自举+离策略），DQN的目标网络和经验回放正是为此而生。
- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)——用于消融研究的参考单文件DQN；建议与本课的从零实现版本一起阅读。
