# 指令遵循作为对齐信号

> 后来对RLHF的每一条批评都针对这一管线。在研究优化压力如何扭曲代理(Proxy)之前，你必须先看到代理。InstructGPT (Ouyang et al., 2022) 定义了参考架构：对指令-响应对进行监督微调，在成对偏好排序上训练奖励模型，并使用带有KL惩罚的PPO对抗奖励模型以保持与SFT策略接近。一个1.3B的InstructGPT比175B的GPT-3更受青睐。这个单一结果就是2026年每个前沿实验室仍然配备基于RLHF的后训练管线的理由。

**类型:** 实践
**语言:** Python (stdlib, 玩具三阶段管线)
**前置要求:** 阶段10·06 (SFT), 阶段10·07 (RLHF), 阶段10·08 (DPO)
**时间:** ~45分钟

## 学习目标

- 说出InstructGPT管线的三个阶段以及每个阶段使用的损失函数。
- 解释为什么一个1.3B的指令微调模型在人类偏好评估上击败了原始的175B GPT-3。
- 说明第三阶段中KL惩罚保护的是什么，以及为什么去掉它会退化为模式寻求行为。
- 描述对齐税(Alignment Tax)以及Ouyang等人使用的PPO-ptx缓解措施。

## 问题

预训练语言模型补全文本。它们不回答问题。让GPT-3“写一个反转列表的Python函数”，你常常得到的是另一个提示，因为大部分训练分布是网络文本，而网络文本会继续产生更多网络文本。模型在执行它的任务——这个任务是错的。

每个严肃实验室用来修正这个问题的代理是人类偏好。两个补全呈现给评分者；评分者选出更好的；奖励模型学习评分者的偏好。然后一个RL循环将策略朝向奖励模型评分高的输出移动。这就是InstructGPT论文的三句话总结。论文的其余部分是工程细节。

## 核心概念

### 阶段1：监督微调(SFT)

收集提示-响应对，其中响应是一个善意的人类会写出的内容。Ouyang等人使用了来自标注者和OpenAI API的13k条提示。在此数据上使用标准交叉熵损失对基础模型进行微调。

SFT给你带来什么：模型现在回答问题而不是继续它们。它不给你带来什么：当多个回答都合理时，没有任何关于评分者偏好哪个回答的信号。

### 阶段2：奖励模型(RM)

对于每个提示，从SFT模型中采样K个补全。一个标注者对它们进行排序。训练一个奖励模型，对任何提示-响应对打分，使得对于`y_w`优于`y_l`的配对：

```
L_RM = -log sigmoid(r(x, y_w) - r(x, y_l))
```

这是Bradley-Terry成对偏好损失。RM通常从SFT模型初始化，将语言模型头替换为标量头。

奖励模型很小：6B对于175B的InstructGPT就足够了。它们也很脆弱——论文的第5节主要讨论了在小规模上出现的奖励破解行为。

### 阶段3：带有KL惩罚的PPO

定义目标：

```
J(pi) = E_{x~D, y~pi(.|x)} [ r(x, y) ] - beta * KL(pi(.|x) || pi_SFT(.|x))
```

使用PPO最大化。KL项使得`pi`不会偏离SFT策略太远。没有它，优化器会找到对抗样本——那些在RM下得分很高的字符串，因为RM从未见过它们，而不是因为人类真的偏好它们。

KL系数`beta`是唯一最重要的RLHF超参数。太低：奖励破解。太高：相比SFT没有改进。

### 对齐税

RLHF之后，模型更受人类偏好，但在标准基准（SQuAD, HellaSwag, DROP）上出现退化。Ouyang等人称之为对齐税，并使用PPO-ptx来修复：将预训练梯度混入RL目标，使模型不会忘记如何执行它从未因此获得奖励的下游任务。

```
J_ptx(pi) = J(pi) + gamma * E_{x~D_pretrain} [ log pi(x) ]
```

PPO-ptx成为标准。Anthropic、DeepMind和Meta都使用某种变体。

### 结果

一个1.3B的InstructGPT (SFT + RM + PPO-ptx) 在大约70%的情况下被标注者认为优于175B的基础GPT-3。在来自生产流量的隐藏测试提示上差距更大。从这个数字可以读出两件事：

1. 对齐是一个与能力不同的维度。175B模型有更强的能力；1.3B模型有更好的对齐；标注者偏好对齐的模型。
2. 能力下限由基础模型设定。你不能通过RLHF使基础模型知道它从未见过的事实。

### 为什么这是阶段18的参考点

之后每一课中的批评——奖励破解（第2课），DPO（第3课），谄媚（第4课），CAI（第5课），休眠智能体（第7课），对齐伪装（第9课）——都针对这个管线的某一部分。奖励破解攻击阶段2。DPO合并阶段2和阶段3。CAI取代人类标注者。谄媚表明标注者是一个有偏信号。对齐伪装表明策略可以完全绕过阶段3。如果脑中不先有这个管线，你就无法理解这些批评中的任何一个。

## 使用它

`code/main.py`在玩具偏好数据上模拟三个阶段。基础“策略”是一个在动作{A, B, C}上有偏的硬币。阶段1 SFT在200条提示上模仿标注者动作。阶段2从500个成对排序中拟合一个Bradley-Terry奖励模型。阶段3运行一个简化的PPO更新，带有对SFT策略的KL惩罚。你可以观察奖励上升、KL散度增长以及策略漂移——而且你可以关闭KL项，在50步更新内看到奖励破解出现。

需要关注的内容：

- 使用`beta = 0.1`对比`beta = 0.0`的奖励轨迹。
- KL(pi || pi_SFT)随训练步数的变化。
- 与标注者偏好相比的最终动作分布。

## 发布

本课产生`outputs/skill-instructgpt-explainer.md`。给定一个RLHF管线描述或论文摘要，它能识别正在修改的是三个阶段中的哪一个，每个阶段使用什么损失，以及是否存在KL惩罚或等效的正则化器。

## 练习

1. 运行`code/main.py`。设置`beta = 0.0`并在200个PPO步后报告动作分布。用一段话解释模式寻求行为。

2. 将奖励模型修改为对动作B有+0.5的偏置（模拟奖励漏洞）。使用`beta = 0.1`运行PPO。KL惩罚是否阻止策略利用该偏置？在哪个`beta`值时利用开始变得明显？

3. 阅读Ouyang等人 (arXiv:2203.02155) 图1。通过运行PPO 1、5、20、100步并测量相对于SFT模型的偏好，重现标注者偏好曲线。

4. 论文第4.3节报告，1.3B参数的InstructGPT在大约70%的情况下击败了175B参数的GPT-3。为什么在隐藏的生产提示上的比例高于标注者自己的提示？

5. 在相同的偏好数据上，用DPO（阶段10·08）替换PPO损失。比较最终策略漂移（相对SFT的KL散度）和最终奖励。在匹配奖励下，哪种方法漂移更远？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
|  SFT  |  "指令微调（instruction tuning）"  |  阶段1：在提示-响应对上进行交叉熵微调 |
|  奖励模型  |  "RM"  |  在(提示，响应)上的标量回归器，使用Bradley-Terry在成对标签上训练 |
|  Bradley-Terry  |  "成对偏好损失（pairwise preference loss）"  |  -log sigmoid(r_w - r_l)；将成对排序简化为二分类 |
|  KL惩罚  |  "正则化项（regularizer）"  |  `beta * KL(pi \ | \ |  pi_SFT)` — 保持RL策略接近SFT锚点 |
|  PPO-ptx  |  "混合预训练的PPO（PPO with pretraining mix）"  |  将一部分预训练的对数似然添加到PPO目标中，以抵消对齐税 |
|  对齐税  |  "RLHF回归（the RLHF regression）"  |  在RLHF未针对的标准基准上的RLHF后性能下降 |
|  标注者偏好  |  "真实标签（ground truth）"  |  人类排名的样本；RM是它的统计代理，而不是"人类价值观"的代理 |

## 延伸阅读

- [Ouyang et al. — Training language models to follow instructions with human feedback (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155) — InstructGPT论文，后续所有RLHF流程的基础
- [Ouyang et al. — Training language models to follow instructions with human feedback (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155) — 用于摘要的RLHF前身
- [Ouyang et al. — Training language models to follow instructions with human feedback (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155) — 原始的基于偏好的RL公式
- [Ouyang et al. — Training language models to follow instructions with human feedback (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155) — Anthropic的HH扩展，对InstructGPT流程的扩展
