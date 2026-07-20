# StyleGAN

> 大多数生成器将 `z` 同时注入每一层。StyleGAN 将其分开：首先将 `z` 映射到一个中间 `w`，然后通过 AdaIN 在每个分辨率级别 *注入* `w`。这一改变解开了潜在空间的纠缠，使得照片级真实人脸在七年内成为已解决的问题。

**类型：** 构建
**语言：** Python
**先决条件：** 阶段 8 · 03（生成对抗网络），阶段 4 · 08（归一化），阶段 3 · 07（卷积神经网络）
**时间：** ~45分钟

## 问题

DCGAN 通过一系列转置卷积将 `z` 映射到图像。问题在于：`z` 控制所有内容——姿态、光照、身份、背景——纠缠在一起。沿着 `z` 的一个轴移动，所有四个都会变化。你无法要求模型“同一个人，不同姿态”，因为表示方式不支持这种分离。

Karras 等人（2019 年，NVIDIA）提出：停止将 `z` 直接输入卷积层。将一个常量 `4×4×512` 张量作为网络输入。学习一个 8 层 MLP 用于映射 `z ∈ Z → w ∈ W`。通过*自适应实例归一化*（AdaIN）在每个分辨率注入 `w`：归一化每个卷积特征图，然后通过 `w` 的仿射投影进行缩放和平移。为随机细节（毛孔、发丝）添加逐层噪声。

结果是：`W` 在“高层风格”（姿态、身份）与“精细风格”（光照、颜色）之间大致具有正交轴。你可以通过使用图像 A 的 `w` 用于低分辨率级别、图像 B 的 `w` 用于高分辨率级别来交换两张图像之间的风格。这开启了编辑、跨领域风格化以及整个“StyleGAN 反演”研究方向。

## 核心概念

![StyleGAN: mapping network + AdaIN + per-layer noise](../assets/stylegan.svg)

**映射网络。** `f: Z → W`，一个 8 层 MLP。`Z = N(0, I)^512`。`W` 不强制为高斯分布——它学习数据自适应的形状。

**合成网络。** 从一个学习的常量 `4×4×512` 开始。每个分辨率块：`upsample → conv → AdaIN(w_i) → noise → conv → AdaIN(w_i) → noise`。分辨率加倍：4, 8, 16, 32, 64, 128, 256, 512, 1024。

**AdaIN。**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

其中 `y_scale` 和 `y_bias` 来自 `w` 的仿射投影。对每个特征图进行归一化，然后重新风格化。这里的“风格”是特征图的一阶和二阶统计量。

**逐层噪声。** 单通道高斯噪声添加到每个特征图，由学习的每通道因子缩放。控制随机细节而不影响全局结构。

**截断技巧。** 在推理时，采样 `z`，计算 `w = mapping(z)`，然后 `w' = ŵ + ψ·(w - ŵ)`，其中 `ŵ` 是多个样本的均值 `w`。`ψ < 1` 以多样性换取质量。几乎每个 StyleGAN 演示都使用 `ψ ≈ 0.7`。

## StyleGAN 1 → 2 → 3

|  版本  |  年份  |  创新点  |
|---------|------|------------|
|  StyleGAN  |  2019  |  映射网络 + AdaIN + 噪声 + 渐进式生长。  |
|  StyleGAN2  |  2020  |  权重解调替代 AdaIN（修复液滴伪影）；跳跃/残差架构；路径长度正则化。  |
|  StyleGAN3  |  2021  |  无混叠卷积 + 等变核；消除纹理粘附像素网格。  |
|  StyleGAN-XL  |  2022  |  类别条件，1024²，ImageNet。  |
|  R3GAN  |  2024  |  使用更强的正则化重新命名；在 FFHQ-1024 上缩小与扩散模型的差距，参数减少 20 倍。  |

到 2026 年，StyleGAN3 仍然是以下场景的默认选择：(a) 高 FPS 下的窄领域照片级真实感，(b) 少样本领域适应（用 100 张图像训练新数据集，冻结映射），(c) 基于反演的编辑（找到重构真实照片的 `w`，然后编辑该 `w`）。对于开放领域的文本到图像，它不是合适的工具——扩散模型才是。

## 动手构建

`code/main.py` 实现了 1-D 的玩具“StyleGAN Lite”：一个映射 MLP，一个合成函数，它接受一个学习的常量向量并用 `w` 派生的尺度/偏置进行调制，以及逐层噪声。它表明通过仿射调制注入 `w` 匹配或优于将 `z` 连接到生成器输入。

### 步骤 1：映射网络

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### 步骤 2：自适应实例归一化

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

每个特征图的尺度和偏置来自 `w` 的线性投影。

### 步骤 3：逐层噪声

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

每通道的 sigma 是可学习的。

## 陷阱

- **液滴伪影。** StyleGAN 1 在特征图中产生斑状液滴，因为 AdaIN 将均值归零。StyleGAN 2 的权重解调通过缩放卷积权重而非特征图来修复。
- **纹理粘附。** StyleGAN 1 和 2 的纹理跟随像素坐标而非物体坐标（在插值时可见）。StyleGAN 3 的无混叠卷积通过窗口化 sinc 滤波器修复。
- **模式覆盖。** 截断 `ψ < 0.7` 看起来干净，但样本来自狭窄的锥体；如果需要多样性，请使用 `ψ = 1.0`。
- **反演是有损的。** 将真实照片反演为 `ψ < 0.7` 通常通过优化或编码器（e4e、ReStyle、HyperStyle）进行。结果会随着多次迭代而漂移。

## 使用它

|  用例  |  方法  |
|----------|----------|
|  照片级真实人脸（动漫、产品、窄领域）  |  StyleGAN3 FFHQ / 自定义微调  |
|  从照片进行人脸编辑  |  e4e 反演 + StyleSpace / InterFaceGAN 方向  |
| 换脸/重现  |  StyleGAN + 编码器 + 混合 |
| 虚拟形象管线  |  使用ADA的StyleGAN3用于低数据微调 |
| 从少量图像进行域适应  |  冻结映射网络，微调合成 |
| 多模态或文本条件生成  |  不要——使用扩散 |

在需要输出"人物脸部照片"的产品级演示中，StyleGAN在推理成本（单次前向传播，4090上<10ms）和相同质量下的清晰度方面优于扩散。

## 发布

保存`outputs/skill-stylegan-inversion.md`。技能接收一张真实照片并输出：反转方法（e4e / ReStyle / HyperStyle）、预期潜码损失、编辑预算（在`W`中可以移动多远而不产生伪影），以及已知的良好编辑方向列表（年龄、表情、姿态）。

## 练习

1. **容易。** 运行`code/main.py`，使用`adain_on=True`和`adain_on=False`。比较固定潜码与扰动潜码的输出分布。
2. **中等。** 实现混合正则化：对于一个训练批次，计算`code/main.py`、`adain_on=True`，并在合成的前半部分应用`adain_on=False`，后半部分应用`w_a`。解码器是否学习到解耦的风格？
3. **困难。** 使用预训练的StyleGAN3 FFHQ模型（ffhq-1024.pkl）。通过在标记样本上训练SVM，找到控制"微笑"的`code/main.py`方向；报告在身份漂移之前可以推动多远。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
| 映射网络  |  "MLP"  |  `f: Z → W`，8层，将潜空间几何与数据统计解耦。 |
| W空间  |  "风格空间"  |  映射网络的输出；大致解耦。 |
| AdaIN  |  "自适应实例归一化"  |  归一化特征图，然后通过`w`投影进行缩放和移位。 |
| 截断技巧  |  "Psi"  |  `w = mean + ψ·(w - mean)`，ψ<1用多样性换取质量。 |
| 路径长度正则化  |  "PL reg"  |  惩罚每单位`w`变化导致的图像大变化；使`W`更平滑。 |
| 权重解调  |  "StyleGAN2修复"  |  归一化卷积权重而非激活；消除液滴伪影。 |
| 无混叠  |  "StyleGAN3的技巧"  |  窗口化sinc滤波器；消除纹理粘附到像素网格。 |
| 反转  |  "为真实图像找到w"  |  优化或编码`x → w`使得`G(w) ≈ x`。 |

## 生产注意：为什么StyleGAN在2026年仍在发布

StyleGAN3在4090上生成1024² FFHQ脸部图像不到10毫秒——`num_steps = 1`，没有VAE解码，没有交叉注意力传递。在生产术语中，这是任何图像生成器的下限延迟。相同分辨率下，50步SDXL + VAE解码管线大约需要3秒。这是一个**300倍的差距**，对于窄域产品（虚拟形象服务、身份证件管线、库存人脸生成），它在总拥有成本（TCO）上胜出。

两个操作上的后果：

- **无调度器，无批处理器。** 在目标占用率下静态批处理是最优的。连续批处理（对LLM和扩散至关重要）没有任何好处，因为每个请求占用相同的FLOPs。
- **截断`ψ`是安全旋钮。** `ψ < 0.7`从映射网络范围的一个狭窄锥体内采样。这是服务层控制样本方差的唯一杠杆。在峰值负载时降低`ψ`，为高级用户提高它。

## 延伸阅读

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN。
- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN2。
- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN3。
- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — e4e反转。
- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN-XL。
- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — 现代最小GAN配方。
