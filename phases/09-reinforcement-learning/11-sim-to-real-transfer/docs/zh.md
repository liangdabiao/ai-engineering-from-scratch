# 仿真到实物的迁移

> 在仿真器中训练的策略如果在硬件上失败，说明该策略只是记住了仿真器。领域随机化、领域自适应和系统辨识是让学得的控制器跨越现实鸿沟的三种工具。

**类型：** 学习
**语言：** Python
**前置知识：** 第9阶段·08（PPO），第2阶段·10（偏差/方差）
**时间：** 约45分钟

## 问题

训练真实机器人缓慢、危险且昂贵。双足机器人需要数百万次训练回合才能学会行走；真实双足机器人哪怕摔倒一次就会损坏硬件。仿真提供了无限重置、确定性可重复性、并行环境以及无物理损坏。

但仿真器是不准确的。轴承的摩擦力比MuJoCo模型大；相机有镜头畸变而仿真器未包含；电机有延迟、回差和饱和，99%的仿真模型忽略了这些。风、灰尘和变化的光照会破坏在无菌渲染环境下训练的策略。**现实鸿沟**——仿真分布与真实分布之间的系统性差异——是机器人强化学习部署的核心问题。

你需要一个对仿真到现实分布偏移具有**鲁棒性**的策略。三种历史方法：随机化仿真器（领域随机化）、用少量真实数据调整策略（领域自适应/微调）、或识别真实系统参数并匹配（系统辨识）。到2026年，主流方案将三者结合，并使用大规模并行仿真（Isaac Sim、Isaac Lab、GPU上的Mujoco MJX）。

## 核心概念

![Three sim-to-real regimes: domain randomization, adaptation, system identification](../assets/sim-to-real.svg)

**领域随机化（DR）。** Tobin等人2017年，Peng等人2018年。训练时，随机化所有可能与真实机器人不同的仿真参数：质量、摩擦系数、电机PD增益、传感器噪声、相机位置、光照、纹理、接触模型。策略学习关于“当前处于哪个仿真”的条件分布，并泛化到整个范围。如果真实机器人落在训练范围内，策略就能工作。

- **优点：** 无需真实数据。一个方案，多个机器人。
- **缺点：** 过度随机化的训练产生“通用”但过于保守的策略。噪声过多≈正则化过多。

**系统辨识（SI）。** 在训练前将仿真器的参数拟合到真实世界数据。如果能测量真实机器人上臂关节的摩擦力，就将其代入仿真器。然后训练期望这些值的策略。需要访问真实系统，但直接缩小了现实鸿沟。

- **优点：** 精确、低噪声的训练目标。
- **缺点：** 残余模型误差对策略不可见；小的未辨识效应（如电机死区）仍会破坏部署。

**领域自适应。** 在仿真中训练，用少量真实数据微调。两种形式：

- **Real2Sim2Real：** 利用真实轨迹学习残差仿真器 `f(s, a, z) - f_sim(s, a)`，在修正后的仿真器中训练。用少量真实数据缩小鸿沟。
- **观测自适应：** 训练一个策略，通过学得的特征提取器（例如GAN像素到像素）将真实观测映射为类仿真观测。控制器仍停留在仿真中。

**特权学习/师生框架。** Miki等人2022年（ANYmal四足机器人）。在仿真中训练一个**教师**策略，它可以访问特权信息（地面真实摩擦力、地形高度、IMU漂移）。蒸馏出一个**学生**策略，它只能看到真实传感器观测。学生学会从历史中推断特权特征，从而对物理参数具有鲁棒性。

**大规模并行仿真。** 2024–2026年。Isaac Lab、Mujoco MJX、Brax均在单个GPU上运行数千个并行机器人。具有4096个并行人形机器人的PPO在数小时内就能收集数年的经验。随着训练分布变宽，“现实鸿沟”缩小；当每个环境都有不同的随机化参数时，DR几乎变得免费。

**2026年真实世界方案（四足行走示例）：**

1. 大规模并行仿真，带有领域随机化的重力、摩擦力、电机增益、负载。
2. 使用特权信息（地形图、身体速度真实值）训练的教师策略。
3. 从教师策略蒸馏出的学生策略，仅使用本体感觉（腿部关节编码器）。
4. 可选：通过真实IMU的自编码器进行观测自适应。
5. 部署。在10+个环境中零样本迁移。若失败，进行带有安全约束PPO的几分钟真实世界微调。

## 动手构建

本节课的代码是一个小型演示，在具有**噪声**转移的GridWorld上展示领域随机化。我们训练一个策略，在“仿真”中经历随机化的滑移概率，并在“真实”中评估一个训练期间从未见过的滑移水平。其形状直接映射到MuJoCo到硬件的迁移。

### 步骤1：参数化仿真

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` 是仿真器暴露的一个参数。在真实机器人中，它可能是摩擦力、质量、电机增益——任何在仿真和现实之间偏移的量。

### 步骤2：使用DR训练

在每个回合开始时，采样 `slip ~ Uniform[0.0, 0.4]`。训练PPO/Q学习/任何算法。进行多个回合。

### 步骤3：在“真实”滑移上零样本评估

在 `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}` 上评估。前四个在训练支持范围内；`0.5` 和 `0.7` 在范围外。经过DR训练的策略应在支持范围内保持接近最优，并在范围外优雅退化。固定滑移训练的策略在其训练滑移之外会变得脆弱。

### 步骤4：与狭窄训练比较

训练第二个策略，仅使用 `slip = 0.0`。在相同的 `slip` 扫描上评估。当真实滑移>0时，您会看到灾难性下降。

## 陷阱

- **过度随机化。** 在 `slip ∈ [0, 0.9]` 上训练，策略变得过于风险厌恶，从不尝试最优路径。匹配*期望的*真实世界分布，而非“任何情况都可能发生”。
- **随机化不足。** 在薄切片上训练，策略完全无法泛化。使用自适应课程（自动领域随机化）随着策略改进而拓宽分布。
- **参数空间误识别。** 随机化错误的东西（当现实鸿沟是电机延迟时随机化相机色调），DR无济于事。首先分析真实机器人。
- **特权信息泄漏。** 教师使用全局状态进行动作而非仅观测，可能导致学生无法跟上。确保教师的策略在给定观测历史下是学生可实现。
- **仿真到仿真迁移失败。** 如果策略对更难的仿真变体不鲁棒，那么对真实世界也不会鲁棒。在部署前始终在保留的仿真变体上测试。
- **缺乏真实世界安全保护。** 在没有低级安全防护的情况下，一个在仿真中有效且在“现实中有效”的策略仍可能损坏硬件。在非学习控制器中添加速率限制、力矩限制、关节限制。

## 使用它

2026年仿真到现实技术栈：

|  领域  |  技术栈  |
|--------|-------|
|  腿部运动（ANYmal、Spot、人形机器人）  |  Isaac Lab + DR + 特权教师/学生  |
|  操作（灵巧手、抓取放置）  |  Isaac Lab + DR + 视觉DR-GAN  |
|  自动驾驶  |  CARLA / NVIDIA DRIVE Sim + DR + 真实环境微调  |
|  无人机竞速  |  RotorS / Flightmare + DR + 在线自适应  |
|  手指/手内操作  |  OpenAI Dactyl（空前规模的域随机化）  |
|  工业机械臂  |  MuJoCo-Warp + 系统辨识 + 小规模真实环境微调  |

对于所有规模的控制，工作流程是一致的：尽可能拟合仿真，随机化无法拟合的部分，训练庞大的策略，蒸馏，部署时带安全护盾。

## 发布

保存为 `outputs/skill-sim2real-planner.md`：

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## 练习

1. **简单.** 在固定滑移的网格世界（滑移=0.0）上训练一个Q学习智能体。在滑移∈{0.0, 0.1, 0.3, 0.5}上评估。绘制回报与滑移的关系图。
2. **中等.** 训练一个采样`slip ~ Uniform[0, 0.3]`的域随机化Q学习智能体。评估相同的扫描。域随机化在滑移=0.5（分布外）时带来多少收益？
3. **困难.** 实现一个课程：从滑移=0.0开始，每当策略达到最优的90%时扩大域随机化范围。测量达到滑移=0.3零样本所需的总环境步数，并与固定域随机化基线进行比较。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  现实差距  |  "仿真到真实的差异"  |  训练与部署物理/感知之间的分布偏移。  |
|  域随机化（DR）  |  "在随机仿真中训练"  |  训练期间随机化仿真参数使策略泛化。  |
|  系统辨识（SI）  |  "测量真实并拟合仿真"  |  估计真实物理参数；设置仿真以匹配。  |
|  域自适应  |  "在真实数据上微调"  |  仿真训练后进行小规模真实世界微调；可能自适应观测或动力学。  |
|  特权信息  |  "教师的真实值"  |  只有仿真拥有的信息；学生必须从观测历史中推断。  |
|  教师/学生  |  "蒸馏特权信息为可观测信息"  |  教师使用捷径训练；学生学习在没有捷径的情况下模仿。  |
|  ADR  |  "自动域随机化"  |  随着策略改进而扩大域随机化范围的课程。  |
|  Real2Sim  |  "用真实数据弥合差距"  |  学习一个残差使仿真模仿真实轨迹。  |

## 延伸阅读

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — 原始域随机化论文（机器人视觉）。
- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — 用于动力学、四足运动的域随机化。
- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — Dactyl，大规模自动域随机化。
- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — 用于ANYmal的教师-学生方法。
- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — 驱动2025-2026部署的大规模并行仿真。
- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — ADR课程方法。
- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — Dyna框架（使用模型进行规划和轨迹生成），支撑现代仿真到现实流水线。
- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — 仿真到现实方法的分类学及基准测试结果。
