# 直接偏好优化（DPO）家族

> Rafailov 等人 (2023) 证明了 RLHF 的最优解在偏好数据上具有封闭形式，因此可以跳过显式奖励模型，直接优化策略。这一洞见催生了一个家族——IPO、KTO、SimPO、ORPO、BPO——每个都修复了 DPO 的一种失败模式。到 2026 年，直接对齐算法在前沿后训练中的部署量将超过 PPO。但第二课中的过度优化曲线仍然适用：DAA 并不能逃脱古德哈特定律，它们只是改变了它咬人的地方。

**类型：** 学习
**语言：** Python (stdlib, six-variant preference-loss comparator)
**先决条件：** 阶段 18 · 01 (InstructGPT), 阶段 18 · 02 (Reward hacking), 阶段 10 · 08 (DPO基础)
**时间：** 约75分钟

## 学习目标

- 从带KL的RLHF最优解推导DPO的封闭形式。
- 说明IPO、KTO、SimPO、ORPO、BPO各自修复了DPO中的哪种失败模式。
- 区分“隐式奖励差距”与“偏好强度”，并解释为什么IPO的恒等映射很重要。
- 解释为什么Rafailov等人(NeurIPS 2024)证明DAA在没有显式奖励模型的情况下仍然会过度优化。

## 问题

RLHF目标函数（第1课）：

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

有一个已知的最优解：

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

因此奖励由最优策略与参考策略的比率隐式定义：

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

将此代入Bradley-Terry偏好似然函数，配分函数`Z(x)`被抵消，因为它仅依赖于`x`。剩下的只有策略参数的损失——无需奖励模型。这就是DPO。

棘手之处：推导假设最优解可达、偏好数据在分布内、参考策略是真实的模式锚点。这些假设没有一个完全成立。每个家族成员修复了一个不同的违反假设。

## 核心概念

### DPO (Rafailov 等人, 2023)

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

可能出现的问题：

- 隐式奖励差距`beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)`是无界的。一个微小的偏好可能产生任意大的差距。
- 损失函数将选定和被拒的log-prob推向相反方向。只要被拒的下降更快，它就能将选定的绝对log-prob拉低。这就是退化选定响应现象。
- 分布外偏好（稀有对 vs 稀有对）会产生任意的隐式奖励。

### IPO (Azar 等人, 2024)

恒等偏好优化将log-sigmoid替换为偏好概率上的恒等映射。损失函数变为有界目标上的平方误差：

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

间隔由`1/(2 beta)`界定。偏好强度与隐式奖励差距成正比。不会爆炸。

### KTO (Ethayarajh 等人, 2024)

卡尼曼-特沃斯基优化完全放弃了成对结构。给定单个标记输出和二元“可取”或“不可取”信号，它映射到前景理论效用：

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

对收益和损失赋予不同的权重（损失厌恶）。好处：可以使用未配对数据，这种数据丰富得多。

### SimPO (Meng 等人, 2024)

简单偏好优化使训练信号与生成对齐。完全移除参考策略，并按长度归一化log-likelihood：

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

使用间隔`gamma`来稳定。长度归一化消除了利用DPO的长度偏差失败模式的动机（更长的`y_w`天然会产生更大的log-prob差距）。

### ORPO (Hong 等人, 2024)

比率比值偏好优化在标准SFT负对数似然基础上增加了偏好项：

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

无参考策略——SFT项作为正则化器。从基础模型到对齐模型进行单阶段训练。无需单独的SFT检查点。

### BPO (ICLR 2026投稿, OpenReview id=b97EwMUWu7)

识别出退化选定响应问题：DPO保留了排名`y_w > y_l`但`y_w`的绝对log-prob可能下降。BPO添加了一行修正，惩罚选定响应上的下降动作。报告在Llama-3.1-8B-Instruct的数学推理上比DPO准确率高10.1%。

### 普遍结果：DAA仍然过度优化

Rafailov 等人《直接对齐算法中奖励模型过度优化的缩放定律》(NeurIPS 2024) 在多个数据集上使用DPO、IPO、SLiC训练策略，跨越KL预算。黄金奖励vs KL曲线具有与Gao等人相同的峰值-坍塌形状。隐式奖励在训练期间查询分布外样本；KL正则化无法稳定这一点。

DAA并不能逃脱古德哈特定律。它们将“咬人的表面”从“奖励模型过度优化”变为“参考策略比率过度优化”。通用的修复方法——更好的数据、集成、早停——对两者都适用。

### 如何选择 (2026)

- 如果你有大量成对偏好数据：使用保守beta的DPO，如果长度偏差明显则用SimPO。
- 如果你有未配对的二元反馈：使用KTO。
- 如果你想从基础模型进行单阶段流水线：使用ORPO。
- 如果你在DPO日志中看到退化选定log-prob：使用BPO。
- 如果偏好强度差异很大且DPO正在饱和：使用IPO。

每个实验室都在一组电池上运行所有五种方法，并按任务选出优胜者。没有理由认为数学推理和安全性上的最优解是相同的。

```figure
dpo-margin
```

## 使用它

`code/main.py`在一个人工偏好数据集上比较了六种损失函数（DPO、IPO、KTO、SimPO、ORPO、BPO），其中真实偏好强度随配对而变化。每种损失函数都针对同一个包含500个配对的样本进行优化，使用一个小型softmax策略。绘制了每种方法的最终胜率、选择日志概率漂移和隐式奖励分布。

## 发布

本课生成`outputs/skill-preference-loss-selector.md`。给定数据集统计信息（配对 vs 非配对、变化 vs 均匀偏好强度、长度分布）和目标（单阶段或SFT后偏好），推荐一种偏好损失并报告其防御的失败模式。

## 练习

1. 运行`code/main.py`。报告DPO和BPO的最终选择日志概率下降。BPO应保持更高的选择绝对概率——验证这一点。

2. 修改偏好数据，使所有配对具有相同的强度。六种方法中哪种最鲁棒？哪种会退化？在此解释IPO的优势。

3. 使被拒绝回复的平均长度是选择的2倍。在不改变其他条件的情况下，用数值展示DPO的长度利用和SimPO的修复。

4. Rafailov等人（NeurIPS 2024）声称DAA过度优化。重现一个单点版本：绘制选择减去拒绝的KL散度，并观察大beta下DPO的过度优化。

5. 阅读BPO论文摘要（OpenReview b97EwMUWu7）。写下BPO对DPO的单行修正。对照`code/main.py`中的实现进行确认。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
|  DPO  |  "无奖励模型的RLHF"  |  损失函数从闭式RLHF最优解导出；仅策略参数  |
|  隐式奖励  |  "对数比值"  |  `beta * log(pi(y\ | x) / pi_ref(y\ | x))` — DPO隐含的奖励  |
|  IPO  |  "有界DPO"  |  用恒等函数替换对数sigmoid；隐式奖励差距被`1/(2 beta)`限制  |
|  KTO  |  "非配对DPO"  |  基于前景理论对单个标签的效用，带有损失厌恶  |
|  SimPO  |  "无参考DPO"  |  长度归一化的对数似然加上边界；无参考策略  |
|  ORPO  |  "单阶段DPO"  |  负对数似然加优势比偏好项；从基础模型单次训练  |
|  BPO  |  "保留选择的DPO"  |  DPO加上一个惩罚项，用于防止选择回复的绝对对数概率下降  |
|  选择退化  |  "选择下降"  |  只要被拒绝的下降更快，DPO就会降低选择的日志概率  |
|  DAA  |  "直接对齐算法"  |  任何跳过显式奖励模型的偏好损失方法  |

## 延伸阅读

- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290) — IPO
- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
