# 评估——FID、CLIP分数、人类偏好

> 每个生成模型排行榜都引用FID、CLIP分数和来自人类偏好竞技场的胜率。每个数字都有失败模式，一个坚定的研究者可以利用。如果你不知道这些失败模式，你就无法区分真正的改进和游戏运行。

**类型：** 构建
**语言：** Python
**先决条件：** 阶段8 · 01（分类学），阶段2 · 04（评估指标）
**时间：** ~45分钟

## 问题

生成模型根据*样本质量*和*条件遵循*来判断。两者都没有封闭形式的度量。你的模型必须渲染10,000张图像；必须有东西给它们分配数字；你必须信任这些数字跨模型家族、跨分辨率、跨架构。三个指标在2014-2026年的考验中幸存下来：

- **FID（Fréchet Inception距离）。** 两个分布——真实和生成——在Inception网络特征空间中的距离。越低越好。
- **CLIP分数。** 生成图像的CLIP图像嵌入与提示的CLIP文本嵌入之间的余弦相似度。越高越好。衡量提示遵循程度。
- **人类偏好。** 将两个模型在相同提示上直接对比，由人类（或GPT-4类模型）选择更好的一个，汇总为Elo分数。

你还会看到：IS（Inception分数，基本已淘汰）、KID、CMMD、ImageReward、PickScore、HPSv2、MJHQ-30k。每个都修正了前一个的一个失败。

## 核心概念

![FID, CLIP, and preference: three axes, different failure modes](../assets/evaluation.svg)

### FID——样本质量

Heusel等人(2017)。步骤：

1. 提取N张真实图像和N张生成图像的Inception-v3特征（2048维）。
2. 对每个池拟合高斯分布：计算均值`μ_r, μ_g`和协方差`Σ_r, Σ_g`。
3. FID = `μ_r, μ_g`。

解释：特征空间中两个多元高斯之间的Fréchet距离。越低=分布越相似。

失败模式：
- **对小N有偏。** FID是特征分布上的均方——小N低估协方差，给出虚假的低FID。始终使用N ≥ 10,000。
- **依赖Inception。** Inception-v3在ImageNet上训练。远离ImageNet的领域（人脸、艺术、文本图像）产生无意义的FID。使用领域特定的特征提取器。
- **游戏。** 对Inception先验的过拟合在视觉质量没有改进的情况下给出低FID。用CMMD（如下）击败它。

### CLIP分数——提示遵循

Radford等人(2021)。对于生成的图像+提示：

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

在30k张生成的图像上平均→一个可在模型间比较的标量。

失败模式：
- **CLIP自身的盲点。** CLIP的组成推理能力弱（"蓝色球体上的红色立方体"经常失败）。模型可以在不真正遵循复杂提示的情况下在CLIP分数上排名靠前。
- **短提示偏差。** 短提示在现实中有更多的CLIP-图像匹配。长提示机械地具有较低的CLIP分数。
- **提示游戏。** 在提示中包含"高质量、4k、杰作"会夸大CLIP分数而不改善图文绑定。

CMMD（Jayasumana等人，2024）修复了其中一些：使用CLIP特征而不是Inception，最大均值差异而不是Fréchet。更能检测微小的质量差异。

### 人类偏好——真实依据

选择一个提示池。用模型A和模型B生成。将成对结果展示给人类（或一个强LLM评判者）。将胜利汇总为Elo或Bradley-Terry分数。基准：

- **PartiPrompts（Google）：** 1,600个多样化提示，12个类别。
- **HPSv2：** 107k人类注释，广泛用作自动化代理。
- **ImageReward：** 137k提示-图像偏好对，MIT许可。
- **PickScore：** 在Pick-a-Pic的260万个偏好上训练。
- **Chatbot-Arena风格的图像竞技场：** https://imagearena.ai/等。

失败模式：
- **评判者差异。** 非专家与专家有不同的偏好。两者都用。
- **提示分布。** 精心挑选的提示偏向某一类。始终记录。
- **LLM评判者奖励黑客。** GPT-4评判者被漂亮但错误的输出欺骗。与人类三角验证。

## 一起使用

一份生产评估报告应包括：

1. 对10-30k样本相对于保留的真实分布计算FID（样本质量）。
2. 对相同样本相对于其提示计算CLIP分数/CMMD（遵循度）。
3. 在盲评测中与先前模型的胜率（总体偏好）。
4. 失败模式分析：随机抽样50个输出，标记已知问题（手部解剖、文本渲染、一致对象计数）。

任何单个指标都是谎言。三个佐证指标+定性审查才是主张。

## 动手构建

`code/main.py`实现了FID、类CLIP分数和Elo聚合，基于合成的“特征向量”（我们使用4维向量作为Inception特征的代替品）。你会看到：

- 对一个小N和一个大N的FID计算——偏差。
- 将“CLIP分数”作为特征池之间的余弦相似度。
- 来自合成偏好流的Elo更新规则。

### 步骤1：四行FID

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### 步骤2：CLIP风格的余弦相似度

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### 步骤3：Elo聚合

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## 陷阱

- **FID在N=1000时的表现。** 在N<10k时启发式方法不可靠。报告低N FID的论文是在钻空子。
- **跨分辨率比较FID。** Inception的299×299缩放改变了特征分布。仅在匹配分辨率下比较。
- **报告单个种子。** 至少运行3个种子。报告标准差。
- **通过负面提示膨胀CLIP分数。** 某些pipeline通过过拟合提示来提升CLIP分数。检查是否有视觉饱和。
- **提示重叠导致的Elo偏差。** 如果两个模型在训练中都见过基准提示，则Elo无意义。使用保留提示集。
- **人类评估的付费众包偏差。** Prolific、MTurk注释者偏向年轻/技术友好。混合招募的艺术/设计专家。

## 使用它

2026年生产评估方案：

|  支柱  |  最低要求  |  推荐做法  |
|--------|---------|-------------|
|  样本质量  |  在10k样本上计算FID与保留真实数据比较  |  + 在5k上计算CMMD + 按类别子集计算FID  |
|  提示遵循度  |  在30k样本上计算CLIP分数  |  + HPSv2 + ImageReward + VQA式问答  |
|  偏好  |  200个盲审对与基线比较  |  + 2000个人类配对 + LLM评判 + Chatbot Arena  |
|  故障分析  |  50个人工标记  |  500个人工标记 + 自动化安全分类器  |

四个支柱全部包含在一个报告中 = 声称。任何一个单独存在 = 营销。

## 发布

保存`outputs/skill-eval-report.md`。该技能接受一个新模型检查点+基线，输出完整的评估方案：样本量、指标、故障模式探测、签核标准。

## 练习

1. **简单。** 运行`code/main.py`。在相同合成分布上比较N=100与N=1000时的FID。报告偏差幅度。
2. **中等。** 从合成CLIP风格特征实现CMMD（公式见Jayasumana等，2024）。比较与FID相比对质量差异的敏感性。
3. **困难。** 复现HPSv2设置：从Pick-a-Pic子集中取1000个图像-提示对，基于偏好微调一个小型CLIP评分器，并测量其与保留集的一致性。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  FID  |  "Fréchet Inception Distance"  |  对真实和生成Inception特征拟合高斯分布后的Fréchet距离。  |
|  CLIP分数  |  "文本-图像相似度"  |  CLIP图像和文本嵌入之间的余弦相似度。  |
|  CMMD  |  "FID的替代品"  |  CLIP特征MMD；偏差更小，无高斯假设。  |
|  IS  |  "Inception分数"  |  Exp KL(p(y | x)  |  |  p(y))；在现代模型上相关性较差，已淘汰。  |
|  HPSv2 / ImageReward / PickScore  |  "学习的偏好代理"  |  在人类偏好上训练的小模型；用作自动评判者。  |
|  Elo  |  "国际象棋等级分"  |  配对获胜的Bradley-Terry聚合。  |
|  PartiPrompts  |  "基准提示集"  |  Google策划的1600个提示，涵盖12个类别。  |
|  FD-DINO  |  "自监督替代品"  |  使用DINOv2特征的FD；更适合非ImageNet领域。  |

## 生产备注：评估也是一种推理工作负载

在10k样本上运行FID意味着生成10k张图像。对于单个L4上50步SDXL基础模型在1024²分辨率下，这大约是11小时的单请求推理。评估预算真实存在，且场景正是离线推理场景（最大化吞吐量，忽略TTFT）：

- **硬批处理，忘记延迟。** 离线评估 = 在内存允许的最大静态批处理大小。在80GB H100上使用`pipe(...).images`和`num_images_per_prompt=8`运行，实际时间比单请求快4-6倍。
- **缓存真实特征。** 对真实参考集的Inception（FID）或CLIP（CLIP分数、CMMD）特征提取仅运行*一次*，存储为`pipe(...).images`。不要每次评估重新计算。

对于CI/回归门控：在每个PR的500样本子集上运行FID + CLIP分数（约30分钟）；每晚运行完整的10k FID + HPSv2 + Elo。

## 延伸阅读

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — FID论文。
- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — CMMD。
- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — CLIP。
- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — HPSv2。
- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — ImageReward。
- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — PartiPrompts。
- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — 故障模式调查。
