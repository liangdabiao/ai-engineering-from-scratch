# 3D视觉 — 点云与NeRF

> 3D视觉有两种形式。点云是传感器的原始输出。NeRF是学习得到的体积场。两者都回答了“空间中有什么在哪里”。

**类型:** 学习+构建
**语言:** Python
**先修课程:** 阶段4第03课（CNN），阶段1第12课（张量操作）
**时间:** 约45分钟

## 学习目标

- 区分显式（点云、网格、体素）和隐式（符号距离场、NeRF）3D表示及其各自的应用场景
- 理解PointNet的对称函数技巧，使神经网络对无序点集具有置换不变性
- 追踪NeRF的前向传播过程：光线投射、体积渲染、位置编码、MLP密度+颜色头
- 使用`nerfstudio`或`instant-ngp`从少量带位姿的图像进行预训练3D重建

## 问题

相机生成2D图像。LIDAR生成一组无序的3D点。运动恢复结构（Structure-from-Motion）流水线生成稀疏的3D关键点云。NeRF从少量带位姿的图像重建整个3D场景。所有这些都属于“视觉”，但它们都不像CNN想要的那种密集张量。

3D视觉之所以重要，是因为几乎所有高价值机器人任务都在3D中运行：抓取、避障、导航、AR遮挡、3D内容捕捉。一个只懂2D图像的视觉工程师会被排除在这个领域增长最快的部分之外（AR/VR内容、机器人、自动驾驶堆栈、基于NeRF的房地产或建筑3D重建）。

这两种表示法因不同原因而占主导地位。点云是传感器免费提供的。NeRF及其后继者（3D高斯泼溅、神经SDF）是当你让神经网络学习一个场景时得到的。

## 核心概念

### 点云

点云是R^3中N个点的无序集合，每个点可选地带有特征（颜色、强度、法线）。

```
cloud = [
  (x1, y1, z1, r1, g1, b1),
  (x2, y2, z2, r2, g2, b2),
  ...
  (xN, yN, zN, rN, gN, bN),
]
```

没有网格，没有连接性。两个属性使得这对神经网络很困难：

- **置换不变性** — 输出不能依赖于点的顺序。
- **可变N** — 单个模型必须处理不同规模的点云。

PointNet（Qi等人，2017）用一个想法同时解决了这两个问题：对每个点应用共享MLP，然后用对称函数（最大池化）聚合。结果是一个固定大小的向量，不依赖于顺序。

```
f(P) = max_{p in P} MLP(p)
```

这就是PointNet的全部核心。更深的变体（PointNet++、Point Transformer）增加了分层采样和局部聚合，但对称函数技巧保持不变。

### PointNet架构

```mermaid
flowchart LR
    PTS["N points<br/>(x, y, z)"] --> MLP1["shared MLP<br/>(64, 64)"]
    MLP1 --> MLP2["shared MLP<br/>(64, 128, 1024)"]
    MLP2 --> MAX["max pool<br/>(symmetric)"]
    MAX --> FEAT["global feature<br/>(1024,)"]
    FEAT --> FC["MLP classifier"]
    FC --> CLS["class logits"]

    style MLP1 fill:#dbeafe,stroke:#2563eb
    style MAX fill:#fef3c7,stroke:#d97706
    style CLS fill:#dcfce7,stroke:#16a34a
```

"共享MLP"意味着同一个MLP独立地在每个点上运行。为了提高效率，实现为在点维度上的1x1卷积。

### 神经辐射场（NeRF）

NeRF（Mildenhall等人，2020）提出了问题“能否从N张照片重建3D场景？”，并用一个本身就是场景的神经网络给出了答案。该网络将`(x, y, z, viewing_direction)`映射到`(density, colour)`。渲染新视角是对该网络的光线投射循环。

```
NeRF MLP:  (x, y, z, theta, phi) -> (sigma, r, g, b)

To render a pixel (u, v) of a new view:
  1. Cast a ray from the camera through pixel (u, v)
  2. Sample points along the ray at distances t_1, t_2, ..., t_N
  3. Query the MLP at each point
  4. Composite the colours weighted by (1 - exp(-sigma * dt))
  5. The sum is the rendered pixel colour
```

损失函数将渲染像素与训练照片中的真实像素进行比较。通过渲染步骤的反向传播更新MLP。没有3D真实值，没有显式几何——场景存储在MLP权重中。

### NeRF中的位置编码

一个普通的MLP在`(x, y, z)`上无法表示高频细节，因为MLP在频谱上偏向低频。NeRF通过将每个坐标编码为MLP之前的傅里叶特征向量来解决这个问题：

```
gamma(p) = (sin(2^0 pi p), cos(2^0 pi p), sin(2^1 pi p), cos(2^1 pi p), ...)
```

最多L=10个频率级别。这与Transformer用于位置的技巧相同，并且在扩散时间条件化（第10课）中再次出现。没有它，NeRF看起来会模糊。

### 体积渲染

```
C(r) = sum_i T_i * (1 - exp(-sigma_i * delta_i)) * c_i

T_i  = exp(- sum_{j<i} sigma_j * delta_j)
delta_i = t_{i+1} - t_i
```

`T_i`是透射率——到达点i的光线剩余量。`(1 - exp(-sigma_i * delta_i))`是点i的不透明度。`c_i`是颜色。最终像素是沿光线方向的加权和。

### 什么取代了NeRF

纯NeRF训练慢（数小时），渲染慢（每张图像数秒）。此后的发展谱系：

- **Instant-NGP**（2022）— 哈希网格编码取代了MLP的位置输入；训练只需数秒。
- **Mip-NeRF 360** — 处理无界场景和抗锯齿。
- **3D高斯泼溅**（2023）— 用数百万个3D高斯体取代体积场；训练只需数分钟，实时渲染。当前生产环境默认方案。

2026年几乎所有真正的NeRF产品实际上都是3D高斯泼溅。但思维模型仍然是NeRF。

### 数据集与基准

- **ShapeNet** — 以点云形式对3D CAD模型进行分类和分割。
- **ScanNet** — 用于分割的真实室内扫描。
- **KITTI** — 用于自动驾驶的室外LIDAR点云。
- **NeRF Synthetic** / **Blended MVS** — 用于视角合成的带位姿图像数据集。
- **Mip-NeRF 360**数据集 — 无界真实场景。

## 动手构建

### 步骤1：PointNet分类器

```python
import torch
import torch.nn as nn

class PointNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),    nn.BatchNorm1d(64),   nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1),   nn.BatchNorm1d(64),   nn.ReLU(inplace=True),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),  nn.BatchNorm1d(128),  nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(1024, 512),   nn.BatchNorm1d(512),  nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),    nn.BatchNorm1d(256),  nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (N, 3, num_points) — transposed for Conv1d
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = torch.max(x, dim=-1)[0]       # (N, 1024)
        return self.head(x)

pts = torch.randn(4, 3, 1024)
net = PointNet(num_classes=10)
print(f"output: {net(pts).shape}")
print(f"params: {sum(p.numel() for p in net.parameters()):,}")
```

约1.6M参数。每点云运行于1,024个点。

### 步骤2：位置编码(Positional encoding)

```python
def positional_encoding(x, L=10):
    """
    x: (..., D) -> (..., D * 2 * L)
    """
    freqs = 2.0 ** torch.arange(L, dtype=x.dtype, device=x.device)
    args = x.unsqueeze(-1) * freqs * 3.141592653589793
    sinc = torch.cat([args.sin(), args.cos()], dim=-1)
    return sinc.reshape(*x.shape[:-1], -1)

x = torch.randn(5, 3)
y = positional_encoding(x, L=10)
print(f"input:  {x.shape}")
print(f"encoded: {y.shape}     # (5, 60)")
```

乘以`2^l * pi`会逐步产生更高的频率。

### 步骤3：Tiny NeRF MLP

```python
class TinyNeRF(nn.Module):
    def __init__(self, L_pos=10, L_dir=4, hidden=128):
        super().__init__()
        self.L_pos = L_pos
        self.L_dir = L_dir
        pos_dim = 3 * 2 * L_pos
        dir_dim = 3 * 2 * L_dir
        self.trunk = nn.Sequential(
            nn.Linear(pos_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),  nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),  nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),  nn.ReLU(inplace=True),
        )
        self.sigma = nn.Linear(hidden, 1)
        self.color = nn.Sequential(
            nn.Linear(hidden + dir_dim, hidden // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 3), nn.Sigmoid(),
        )

    def forward(self, x, d):
        x_enc = positional_encoding(x, self.L_pos)
        d_enc = positional_encoding(d, self.L_dir)
        h = self.trunk(x_enc)
        sigma = torch.relu(self.sigma(h)).squeeze(-1)
        rgb = self.color(torch.cat([h, d_enc], dim=-1))
        return sigma, rgb

nerf = TinyNeRF()
x = torch.randn(128, 3)
d = torch.randn(128, 3)
s, c = nerf(x, d)
print(f"sigma: {s.shape}   rgb: {c.shape}")
```

相比原始NeRF（具有2个深度为8的MLP主干），这是微小的。足以演示架构。

### 步骤4：沿射线体渲染(Volumetric rendering)

```python
def volumetric_render(sigma, rgb, t_vals):
    """
    sigma: (..., N_samples)
    rgb:   (..., N_samples, 3)
    t_vals: (N_samples,) distances along the ray
    """
    delta = torch.cat([t_vals[1:] - t_vals[:-1], torch.full_like(t_vals[:1], 1e10)])
    alpha = 1.0 - torch.exp(-sigma * delta)
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
    weights = alpha * trans
    rendered = (weights.unsqueeze(-1) * rgb).sum(dim=-2)
    depth = (weights * t_vals).sum(dim=-1)
    return rendered, depth, weights


N = 64
t_vals = torch.linspace(2.0, 6.0, N)
sigma = torch.rand(N) * 0.5
rgb = torch.rand(N, 3)
rendered, depth, weights = volumetric_render(sigma, rgb, t_vals)
print(f"rendered colour: {rendered.tolist()}")
print(f"depth:           {depth.item():.2f}")
```

一条射线，64个样本，合成单个RGB像素和一个深度。

## 使用它

实际工作中：

- `nerfstudio` (Tancik等人) — NeRF / Instant-NGP / 高斯泼洒(Gaussian Splatting)的当前参考库。命令行加上网络查看器。
- `nerfstudio` (Meta) — 可微渲染(differentiable rendering)、点云工具、网格操作。
- `nerfstudio` — 点云处理、配准、可视化。

在部署中，3D高斯泼洒(Gaussian splatting)已基本取代纯NeRF，因为其渲染速度快100倍。重建质量相当。

## 发布

本課(lesson)产出：

- `outputs/prompt-3d-task-router.md` — 一个提示，根据任务和输入数据路由到正确的3D表示（点云、网格、体素、NeRF、高斯泼洒）。
- `outputs/prompt-3d-task-router.md` — 一项技能，为.ply / .pcd / .xyz文件编写PyTorch `outputs/skill-point-cloud-loader.md`，具有正确的归一化、居中化和点采样。

## 练习

1. **(简单)** 证明PointNet是置换不变的：对同一个点云运行两次，一次点序随机打乱。验证输出在浮点噪声范围内相同。
2. **(中等)** 实现一个最小射线生成函数，给定相机内参和姿态，为H x W图像的每个像素生成射线起点和方向。
3. **(困难)** 在合成数据集（通过可微渲染或简单光线追踪器生成的彩色立方体渲染视图）上训练TinyNeRF。报告在第1、10、100个周期的渲染损失。模型在哪个周期产生可识别的视图？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|----------------------|
|  点云  |  "来自LIDAR的3D点"  |  无序的(x, y, z)集合 + 每点可选特征  |
|  PointNet  |  "首个处理点云的神经网络"  |  每点共享MLP + 对称（最大）池化；构造上具有置换不变性  |
|  NeRF  |  "本身就是场景的MLP"  |  将(x, y, z, dir)映射到(密度, 颜色)的网络；通过光线投射渲染  |
|  位置编码(Positional encoding)  |  "傅里叶特征(Fourier features)"  |  将每个坐标编码为多个频率的正弦/余弦，以克服MLP的低频偏差  |
|  体渲染(Volumetric rendering)  |  "射线积分(Ray integration)"  |  使用透射率和alpha将沿射线的样本合成为单个像素  |
|  Instant-NGP  |  "哈希网格NeRF"  |  用多分辨率哈希网格替换NeRF的坐标MLP；速度快100-1000倍  |
|  3D高斯泼洒(Gaussian splatting)  |  "数百万个高斯体"  |  场景=3D高斯集合；实时渲染，几分钟训练  |
|  SDF  |  "符号距离场(Signed distance field)"  |  返回到最近表面的带符号距离的函数；另一种隐式表示  |

## 延伸阅读

- [PointNet (Qi et al., 2017)](https://arxiv.org/abs/1612.00593) — 置换不变的分类器
- [PointNet (Qi et al., 2017)](https://arxiv.org/abs/1612.00593) — 将照片重建3D变为神经网络问题的论文
- [PointNet (Qi et al., 2017)](https://arxiv.org/abs/1612.00593) — 哈希网格，1000倍加速
- [PointNet (Qi et al., 2017)](https://arxiv.org/abs/1612.00593) — 在生产中取代NeRF的架构
