# 用于游戏的强化学习(RL)——AlphaZero、MuZero与大语言模型(LLM)推理时代

> 1992年：TD-Gammon使用纯时序差分(TD)在双陆棋中击败人类冠军。2016年：AlphaGo击败李世石。2017年：AlphaZero从零开始统治了国际象棋、将棋和围棋。2024年：DeepSeek-R1证明了相同的配方（用GRPO替换PPO）在推理上有效。游戏是推动这一阶段每一次突破的基准。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段9·05（DQN），阶段9·08（PPO），阶段9·09（RLHF），阶段9·10（MARL）
**时间：** 约120分钟

## 问题

游戏拥有强化学习(RL)想要的一切：清晰的奖励（赢/输）、无限的回合（自我对弈重置）、完美的模拟（游戏*就是*模拟器）、离散或小规模连续动作空间、以及迫使产生对抗鲁棒性的多智能体结构。

而每一次重要的强化学习(RL)突破都是通过游戏进行测试的：TD-Gammon（双陆棋，1992）、Atari-DQN（2013）、AlphaGo（2016）、AlphaZero（2017）、OpenAI Five（Dota 2，2019）、AlphaStar（星际争霸II，2019）、MuZero（学习模型，2019）、AlphaTensor（矩阵乘法，2022）、AlphaDev（排序算法，2023）、DeepSeek-R1（数学推理，2025）——这是游戏RL技术应用于文本的最新证明。

本篇总结性文章通过一个统一的视角审视三个里程碑式的架构——AlphaZero、MuZero和GRPO：**自我对弈+搜索+策略改进**。每个架构都泛化了前者；特别是GRPO，是将AlphaZero的配方应用于大语言模型(LLM)推理，以词元(token)作为动作，数学验证作为获胜信号。

## 核心概念

![AlphaZero ↔ MuZero ↔ GRPO: same loop, different environments](../assets/rl-games.svg)

**统一循环。**

```
while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output
```

**AlphaZero（2017年）。** Silver等人。给定一个有已知规则的游戏（国际象棋、将棋、围棋）：

- 策略-价值网络：一个塔 `f_θ(s) → (p, v)`。`p` 是合法动作的先验概率。`v` 是期望的游戏结果。
- 蒙特卡洛树搜索(MCTS)：每一步，扩展一个可能后续步骤的树。使用 `f_θ(s) → (p, v)` 作为先验+引导。通过UCB（PUCT）选择节点：`p`。
- 自我对弈：智能体与智能体对弈。在第 `f_θ(s) → (p, v)` 步，MCTS访问分布 `p` 成为策略训练目标。
- 损失：`f_θ(s) → (p, v)`。`p` 是游戏结果（+1/0/-1）。

零人类知识。零手工启发式。一个单一的配方，在各自进行数千万次自我对弈后掌握了国际象棋、将棋和围棋。

**MuZero（2019年）。** Schrittwieser等人。移除了对规则已知的要求。

- 学习一个*潜在动力学模型* `(h, g, f)`，而不是固定环境：
  - `(h, g, f)`：将观测编码为潜在状态。
  - `(h, g, f)`：预测下一个潜在状态+奖励。
  - `(h, g, f)`：预测策略先验+价值。
- MCTS在*学习到的潜在空间*中运行。相同的搜索，相同的训练循环。
- 适用于围棋、国际象棋、将棋*以及*Atari——一个算法，无需规则知识。

**随机MuZero（2022年）。** 添加了随机动力学和机会节点；扩展至双陆棋类游戏。

**Muesli、Gumbel MuZero（2022-2024年）。** 在样本效率和确定性搜索上的改进。

**GRPO（2024-2025年）。** DeepSeek-R1的配方。与AlphaZero形状相同的循环，应用于语言模型推理：

- "游戏"：回答数学/编程/推理问题。"赢"=验证器（测试用例通过、数字答案匹配）返回1。
- 策略：大语言模型(LLM)。动作：词元(token)。状态：提示(prompt)+已生成回答。
- 无评论家(critic)（PPO风格的V_φ）。对于每个提示，从策略中采样 `G` 个完整回答。计算每个回答的奖励。使用**组相对优势(group-relative advantage)** `A_i = (r_i - mean_r) / std_r` 作为REINFORCE风格更新的信号。
- 对参考策略的KL惩罚以防止漂移（类似RLHF）。
- 完整损失：

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

无奖励模型，无评论家，无MCTS。组相对基线(group-relative baseline)取代了所有三者。在推理基准上的质量匹配或超过PPO-RLHF，而计算量仅为其一小部分。

**完整的R1配方。** DeepSeek-R1（DeepSeek 2025）在同一篇论文中包含了两个模型：

- **R1-Zero。** 从DeepSeek-V3基础模型开始。无监督微调(SFT)。直接应用GRPO，包含两个奖励组件：*准确性奖励*（基于规则——最终答案是否解析为正确数字/代码是否通过单元测试）和*格式奖励*（完整回答是否将其思维链(chain-of-thought)包裹在`<think>…</think>`标签中）。经过数千步，平均回答长度从约100词元增长到约10,000词元，数学基准分数攀升至接近o1-preview水平。模型从零开始学习推理。缺点：其思维链通常难以阅读、混用语言、缺乏风格润色。
- **R1。** 通过四阶段流水线修复R1-Zero的可读性问题：
  1. **冷启动SFT。** 收集数千个格式清晰的长思维链(Long CoT)示例。在其上监督微调基础模型。这提供了一个可读的起点。
  2. **面向推理的GRPO。** 应用GRPO，使用准确性+格式奖励以及*语言一致性*奖励以防止语言混用。
  3. **拒绝采样+第二轮SFT。** 从强化学习(RL)检查点采样约60万个推理轨迹，仅保留那些最终答案正确且思维链(CoT)可读的轨迹，并与约20万个非推理SFT示例（写作、问答、自我认知）结合。再次微调基础模型。
  4. **全频谱GRPO。** 再进行一轮强化学习(RL)，涵盖推理（基于规则的奖励）和一般对齐（有用性/无害性基于偏好的奖励）。

结果在AIME和MATH-500上以开放权重匹配o1，并且小到足以蒸馏。同一篇论文还通过在R1的推理轨迹上进行监督微调(SFT)发布了六个蒸馏的稠密模型（Qwen-1.5B到Llama-70B）——学生模型无需强化学习(RL)。强强化学习(RL)教师模型的蒸馏始终在学生规模上击败从头开始的强化学习(RL)。

**为什么推理中使用GRPO而不是PPO。** DeepSeekMath论文（2024年2月）中提出了三个原因：（1）无需训练价值网络，内存减半；（2）组基线自然处理推理任务产生的稀疏的轨迹结束奖励；（3）每个提示的归一化使优势在不同难度的问题之间具有可比性，而PPO的单一评论家无法做到这一点。

**无搜索与基于搜索。** 游戏已经分化：

- *具有长时域(horizons)的完美信息游戏*（围棋、国际象棋）：仍然基于搜索。AlphaZero/MuZero占主导。
- *大语言模型(LLM)推理*：生产中尚无MCTS；使用完整展开的GRPO，推理计算采用最佳N(Best-of-N)。过程奖励模型(PRMs)暗示逐步搜索将被重新加入。

## 动手构建

`code/main.py` 中的代码实现了**微型GRPO**——一个具有多组样本的赌博机(bandit)。算法与大语言模型(LLM)上的相同；只是策略和环境更简单。它教授了*损失*和*组相对优势(group-relative advantage)*，这是2025年的创新。

### 步骤1：一个微型验证器环境

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

在真实的GRPO中，验证器运行单元测试或检查数学等式。

### 步骤2：策略：对每个提示的K个答案词元(tokens)进行softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

等价于以大语言模型(LLM)在给定提示(prompt)条件下的最后一层输出。

### 步骤3：组采样和组相对优势(group-relative advantage)

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

组相对优势是2024年DeepSeek的技巧。无需评论家。"基线"是组均值，归一化使用组标准差。

### 第4步：与REINFORCE基线（无价值函数）比较

相同设置，相同计算量，普通REINFORCE。GRPO收敛更快且更稳定。

### 第5步：观察熵和KL散度

与RLHF相同的诊断指标：对参考模型的平均KL散度、策略熵、奖励随时间变化。一旦这些稳定，训练就完成了。

## 陷阱

- **通过验证器博弈进行奖励黑客攻击。** GRPO继承了RLHF的风险：如果验证器错误或可被利用，LLM会找到漏洞。鲁棒的验证器（多个测试用例、形式化证明）至关重要。
- **组大小太小。** 组基线的方差变化如`1/√G`。低于`G = 4`时，优势信号噪声大；标准选择是`G = 8`到`64`。
- **长度偏差。** 不同长度的LLM补全有不同的对数概率。按token数量归一化，或使用序列级对数概率，或截断到最大长度。
- **纯自我对弈循环。** AlphaZero式训练可能在一般和博弈中陷入支配循环。通过多样化的对手池缓解（联赛制，第10课）。
- **搜索-策略不匹配。** AlphaZero训练策略以模仿搜索输出。如果策略网络太小无法表示搜索的分布，训练会停滞。
- **计算门槛。** MuZero/AlphaZero需要大量计算。单个消融实验往往需要数百GPU小时。存在用于学习的小型演示（如Connect Four上的AlphaZero）。
- **验证器覆盖率。** 为有缺陷的解决方案通过的单元测试会强化缺陷。设计能捕获边缘案例的验证器。

## 使用它

2026年游戏强化学习领域概览（按领域分）：

|  领域  |  主导方法  |
|--------|-----------------|
|  双人零和棋盘游戏（围棋、国际象棋、将棋）  |  AlphaZero / MuZero / KataGo  |
|  不完美信息纸牌游戏（扑克）  |  CFR + 深度学习 (DeepStack, Libratus, Pluribus)  |
|  Atari / 像素游戏  |  Muesli / MuZero / IMPALA-PPO  |
|  大型多人策略游戏（Dota、星际争霸）  |  PPO + 自我对弈 + 联赛 (OpenAI Five, AlphaStar)  |
|  LLM数学/代码推理  |  GRPO (DeepSeek-R1, Qwen-RL, 开源复现)  |
|  LLM对齐  |  DPO / RLHF-PPO (不是GRPO；验证器是偏好而非可验证的)  |
|  机器人学  |  PPO + DR (不是游戏强化学习，但使用相同的策略梯度工具)  |
|  组合问题  |  AlphaZero变体 (AlphaTensor, AlphaDev)  |

这一*配方*——自我对弈、搜索增强改进、策略蒸馏——跨越文本、像素和物理控制。GRPO是最新的实例；更多即将到来。

## 发布

保存为 `outputs/skill-game-rl-designer.md`：

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## 练习

1. **简单。** 在`code/main.py`中实现GRPO赌博机。在2个提示×每个提示4个回答token上训练。在`G=8`下于<1000次更新内收敛。
2. **中等。** 接入PPO（裁剪版）和普通REINFORCE。在同一赌博机上比较样本效率和奖励方差与GRPO。
3. **困难。** 扩展为长度2的“推理链”：智能体发出两个token，验证器奖励这对。衡量GRPO如何处理跨两步序列的信用分配。（提示：计算每个*完整序列*的组优势，传播到两个token位置。）

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  MCTS  |  "带学习网络的树搜索"  |  蒙特卡洛树搜索；使用学习到的`(p, v)`先验进行UCB1/PUCT选择。 |
|  AlphaZero  |  "自我对弈 + MCTS"  |  训练策略价值网络以匹配MCTS访问次数和游戏结果。 |
|  MuZero  |  "学习模型AlphaZero"  |  相同循环但在通过学习的动力学实现的潜在空间中。 |
|  GRPO  |  "无评论家PPO"  |  组相对策略优化；带组均值基线+KL的REINFORCE。 |
|  PUCT  |  "AlphaZero的UCB"  |  `Q + c · p · √N / (1 + N_a)` — 平衡价值估计与先验。 |
|  自我对弈  |  "智能体与历史自身"  |  零和博弈的标准；对称训练信号。 |
|  联赛制  |  "基于群体的自我对弈"  |  历史+当前+利用者被采样为对手。 |
|  验证器奖励  |  "可验证强化学习"  |  奖励来自确定性检查器（测试通过，答案匹配）。 |
|  过程奖励  |  "PRM"  |  对每个推理步骤评分，而不仅仅是最终答案。 |

## 延伸阅读

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — 介绍GRPO和组相对基线的论文。
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — 完整的四阶段R1配方加上R1-Zero消融实验。
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — 大规模CFR+深度学习。
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — 开启一切的论文。
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — 使用自定义奖励函数应用GRPO的生产参考。
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — 多尺度R1配方的开源复现。
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — 教科书式的框架，用于自我对弈、搜索和R1在LLM规模上实例化的“设计奖励”。
