# 3D生成

> 3D是2D到3D迁移效果最强的模态。2023年的突破是3D高斯喷溅(3D Gaussian Splatting)。2024-2026年的生成式推进在此基础上叠加了多视角扩散(Multi-view Diffusion)和3D重建，从单个提示或照片生成物体和场景。

**类型:** 学习
**语言:** Python
**前置要求:** 阶段4 (视觉), 阶段8 · 07 (潜在扩散)
**时间:** ~45分钟

## 问题

3D内容令人痛苦：

- **表示法.** 网格、点云、体素网格、符号距离场(SDF)、神经辐射场(NeRF)、3D高斯。每种都有权衡。
- **数据稀缺.** ImageNet有1400万张图像。最大的干净3D数据集(Objaverse-XL, 2023)约有1000万个物体，多数质量低。
- **内存.** 一个512³体素网格有1.28亿个体素；一个有用的场景NeRF需要每射线100万个样本。生成比重建更难。
- **监督.** 对于2D图像，你有像素。对于3D，你通常只有少量2D视图，必须提升到3D。

2026年的技术栈将这两个问题分开。首先，使用扩散模型生成*2D多视角图像*。其次，将*3D表示*（通常是高斯喷溅）拟合到这些图像上。

## 核心概念

![3D generation: multi-view diffusion + 3D reconstruction](../assets/3d-generation.svg)

### 表示法：3D高斯喷溅(3D Gaussian Splatting) (Kerbl et al., 2023)

将一个场景表示为约100万个3D高斯的点云。每个高斯有59个参数：位置(3)、协方差(6，或四元数4+尺度3)、不透明度(1)、球谐颜色(3阶48，0阶3)。

渲染 = 投影 + alpha合成。快速（在4090上1080p约100fps）。可微分。通过梯度下降拟合真实照片。一个场景在消费级GPU上5-30分钟拟合。

在此基础上2023-2024年的两项创新：
- **生成式高斯喷溅.** LGM、LRM、InstantMesh等模型直接从一张或几张图像预测高斯云。
- **4D高斯喷溅.** 具有每帧偏移的高斯用于动态场景。

### 多视角扩散

微调预训练图像扩散模型，从文本提示或单张图像生成同一物体的多个一致视图。Zero123 (Liu et al., 2023)、MVDream (Shi et al., 2023)、SV3D (Stability, 2024)、CAT3D (Google, 2024)。通常输出物体周围4-16个视图，通过高斯喷溅或NeRF提升到3D。

### 文本到3D流水线

|  模型  |  输入  |  输出  |  时间  |
|-------|-------|--------|------|
|  DreamFusion (2022)  |  文本  |  NeRF via SDS  |  每个资产约1小时  |
|  Magic3D  |  文本  |  网格+纹理  |  约40分钟  |
|  Shap-E (OpenAI, 2023)  |  文本  |  隐式3D  |  约1分钟  |
|  SJC / ProlificDreamer  |  文本  |  NeRF / 网格  |  约30分钟  |
|  LRM (Meta, 2023)  |  图像  |  三平面  |  约5秒  |
|  InstantMesh (2024)  |  图像  |  网格  |  约10秒  |
|  SV3D (Stability, 2024)  |  图像  |  新视角  |  约2分钟  |
|  CAT3D (Google, 2024)  |  1-64张图像  |  3D NeRF  |  约1分钟  |
|  TripoSR (2024)  |  图像  |  网格  |  约1秒  |
|  Meshy 4 (2025)  |  文本+图像  |  PBR网格  |  约30秒  |
|  Rodin Gen-1.5 (2025)  |  文本+图像  |  PBR网格  |  约60秒  |
|  Tencent Hunyuan3D 2.0 (2025)  |  图像  |  网格  |  约30秒  |

2025-2026方向：直接文本到网格模型，具有适合游戏引擎的PBR材质。多视角扩散中间步骤仍然是通用物体表现最佳的方案。

### NeRF（背景）

神经辐射场(Neural Radiance Field) (Mildenhall et al., 2020)。一个小型MLP接收`(x, y, z, view direction)`并输出`(color, density)`。通过沿射线积分渲染。在质量上超越基于网格的新视角合成，但渲染速度慢100-1000倍。在大多数实时应用中被高斯喷溅取代，但在研究中仍占主导地位。

## 动手构建

`code/main.py` 实现了一个玩具2D“高斯泼溅”(Gaussian splatting)拟合：将合成目标图像（平滑渐变）表示为多个2D高斯泼溅之和。通过梯度下降优化位置、颜色和协方差以匹配目标。你会看到两个核心操作：前向渲染（泼溅+阿尔法合成）和通过梯度下降进行拟合。

### 步骤1: 2D高斯泼溅

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### 步骤2: 通过求和泼溅进行渲染

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

真实的3D高斯泼溅按深度排序高斯体并按顺序进行阿尔法合成。我们的2D玩具只是求和。

### 步骤3: 通过梯度下降进行拟合

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## 陷阱

- **视角不一致。** 如果独立生成4个视图，且它们对物体结构不一致，则3D拟合会模糊。解决方法：使用共享注意力的多视图扩散。
- **背面幻觉。** 单张图像→3D必须发明未看到的一面。质量差异很大。
- **高斯泼溅爆炸。** 无约束训练会导致泼溅数量增长到1000万个并过拟合。致密化+剪枝启发式（来自3D-GS原始论文）是必需的。
- **拓扑问题。** 来自隐式场（SDF）的网格通常有孔洞或自相交。在发布前运行重新网格化（例如Blender的体素重新网格化）。
- **训练数据许可。** Objaverse具有混合许可；商业用途因模型而异。

## 使用它

| 任务  |  2026年推荐 |
|------|-----------|
|  从照片进行场景重建  |  高斯泼溅 (3DGS, Gsplat, Scaniverse)  |
|  用于游戏的文本到3D物体  |  Meshy 4 或 Rodin Gen-1.5 (PBR输出)  |
|  图像到3D  |  Hunyuan3D 2.0, TripoSR, InstantMesh  |
|  从少量图像进行新视图合成  |  CAT3D, SV3D  |
|  动态场景重建  |  4D高斯泼溅  |
|  化身/穿着衣服的人  |  Gaussian Avatar, HUGS  |
|  研究/SOTA  |  上周发布的最新成果  |

对于在游戏或电商管线中交付生产级3D：Meshy 4 或 Rodin Gen-1.5 输出PBR网格，可直接导入Unity/Unreal。

## 发布

保存 `outputs/skill-3d-pipeline.md`。技能获取3D简要（输入：文本/单张图像/少量图像；输出：网格/泼溅/NeRF；用途：渲染/游戏/VR）并输出：管线（多视图扩散+拟合，或直接网格模型）、基础模型、迭代预算、拓扑后处理、所需材质通道。

## 练习

1. **简单。** 使用4、16、64个高斯体运行 `code/main.py`。报告最终MSE与目标对比。
2. **中等。** 扩展到彩色高斯体（RGB）。确认重建匹配目标颜色模式。
3. **困难。** 使用gsplat或Nerfstudio，从50张照片捕获重建真实物体。报告拟合时间和在保留视图上的最终SSIM。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  3D高斯泼溅  |  "3DGS"  |  场景表示为3D高斯点云；可微阿尔法合成渲染。  |
|  NeRF  |  "神经辐射场"  |  在3D点输出颜色+密度的MLP；通过光线积分渲染。  |
|  Triplane  |  "三个2D平面"  |  将3D分解为三个对齐坐标轴的2D特征网格；比体积表示更便宜。  |
|  SDS  |  "分数蒸馏采样"  |  通过将2D扩散分数用作伪梯度来训练3D模型。  |
|  多视图扩散  |  "一次多个视图"  |  输出一批一致相机视图的扩散模型。  |
|  PBR  |  "基于物理的渲染"  |  包含反照率、粗糙度、金属度、法线通道的材质。  |
|  致密化  |  "增长泼溅"  |  3DGS训练启发式：在高梯度区域分裂/克隆泼溅。  |

## 生产注意事项：3D尚未有共享基座

与图像（潜在扩散+DiT）和视频（时空DiT）不同，3D在2026年没有单一主导运行时。生产决策树在表示上分叉：

- **NeRF / triplane.** 推理是光线步进+每个样本一次MLP前向。512²渲染需要数百万次MLP前向。积极批处理光线样本；适用SDPA/xformers。
- **多视图扩散+LRM重建。** 两阶段管线。阶段1（多视图DiT）是与第07课类似的扩散服务器。阶段2（LRM transformer）是对视图的一次性前向传递。整体延迟特征是“扩散+一次性”——相应地选择每阶段服务原语。
- **SDS / DreamFusion.** 每个资产的优化，而非推理。构建任务，而非请求处理器。

对于大多数2026年产品，正确答案是“按需运行多视图扩散模型，异步重建为3DGS，提供实时查看的3DGS服务”。这将在GPU推理服务器（快速）和离线优化器（慢速）之间干净地分割工作负载。

## 延伸阅读

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF.
- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — 3DGS.
- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — SDS.
- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — Zero123.
- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — 多视图扩散.
- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — LRM.
- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — CAT3D.
- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — SV3D.
