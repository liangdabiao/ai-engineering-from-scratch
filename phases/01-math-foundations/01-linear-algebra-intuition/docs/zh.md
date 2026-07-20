# 线性代数直觉 (Linear Algebra Intuition)

> 每个 AI 模型都只是穿着华丽外衣的矩阵运算。

**类型：** 学习
**语言：** Python, Julia
**先修：** 第 0 阶段
**时间：** 约 60 分钟

## 学习目标

- 用 Python 从头实现向量和矩阵运算（加法、点积、矩阵乘法）
- 从几何角度解释点积 (dot product)、投影 (projection) 和格拉姆-施密特 (Gram-Schmidt) 过程的作用
- 使用行约简 (row reduction) 判断一组向量的线性无关性 (linear independence)、秩 (rank) 和基 (basis)
- 将线性代数概念与 AI 应用联系起来：嵌入 (embeddings)、注意力分数 (attention scores) 和 LoRA

## 问题

打开任何一篇 ML 论文。在第一页内，你就会看到向量、矩阵、点积和变换。如果没有线性代数直觉，它们只是符号。有了它，你就能看到神经网络实际上在做什么——在空间中移动点。

你不需要成为数学家。你需要理解这些运算的几何意义，然后亲自编写代码实现它们。

## 核心概念

### 向量是点（也是方向）

向量就是一串数字。但这些数字有意义——它们是空间中的坐标。

**2D向量[3, 2]：**

|  x  |  y  |  点  |
|---|---|-------|
|  3  |  2  |  向量从原点(0,0)指向平面上的(3, 2)  |

该向量的模为 sqrt(3^2 + 2^2) = sqrt(13)，指向右上方。

在AI中，向量代表一切：
- 一个词 → 一个768维的向量（其在嵌入空间中的“含义”）
- 一张图像 → 一个包含数百万像素值的向量
- 一个用户 → 一个偏好向量

### 矩阵即变换

矩阵将一个向量变换为另一个向量。它可以旋转、缩放、拉伸或投影。

```mermaid
graph LR
    subgraph Before
        A["Point A"]
        B["Point B"]
    end
    subgraph Matrix["Matrix Multiplication"]
        M["M (transformation)"]
    end
    subgraph After
        A2["Point A'"]
        B2["Point B'"]
    end
    A --> M
    B --> M
    M --> A2
    M --> B2
```

在AI中，矩阵即模型：
- 神经网络权重 → 将输入变换为输出的矩阵
- 注意力分数 → 决定关注什么的矩阵
- 嵌入 → 将词映射为向量的矩阵

### 点积衡量相似性

两个向量的点积告诉你它们有多相似。

```
a · b = a₁×b₁ + a₂×b₂ + ... + aₙ×bₙ

Same direction:      a · b > 0  (similar)
Perpendicular:       a · b = 0  (unrelated)
Opposite direction:  a · b < 0  (dissimilar)
```

这正是搜索引擎、推荐系统和RAG的工作原理——找到点积高的向量。

### 线性无关(Linear Independence)

如果集合中的任何一个向量都不能写成其他向量的组合，则这些向量线性无关。如果v1、v2、v3是线性无关的，它们张成一个三维空间。如果其中一个是其他向量的组合，则它们只张成一个平面。

它对人工智能的重要性：你的特征矩阵应具有线性无关的列。如果两个特征完全相关（线性相关），模型将无法区分它们的影响。这会导致回归中的多重共线性——权重矩阵变得不稳定，微小的输入变化会产生剧烈的输出波动。

**具体例子：**

```
v1 = [1, 0, 0]
v2 = [0, 1, 0]
v3 = [2, 1, 0]   # v3 = 2*v1 + v2
```

v1和v2是独立的——两者均不是对方的标量倍数或线性组合。但v3 = 2*v1 + v2，因此{v1, v2, v3}是一个线性相关组。这三个向量都位于xy平面内。无论你如何组合它们，都无法到达[0, 0, 1]。你拥有三个向量，却只有两个自由度。

在数据集中：如果feature_3 = 2*feature_1 + feature_2，添加feature_3不会给模型带来任何新信息。更糟糕的是，它会导致正规方程奇异——权重没有唯一解。

### 基与秩

基是张成整个空间的最小线性无关向量组。基向量的数量就是空间的维度。

三维空间的标准基是{[1,0,0], [0,1,0], [0,0,1]}。但三维空间中的任意三个独立向量都能构成一个有效基。基的选择就是坐标系的选择。

矩阵的秩 = 线性无关的列数 = 线性无关的行数。若秩 < min(行数, 列数)，则矩阵是秩亏缺的(rank-deficient)。这意味着：
- 系统有无穷多解（或无解）
- 变换中丢失了信息
- 矩阵不可求逆

|  情况  |  秩  |  对机器学习的意义  |
|-----------|------|---------------------|
|  满秩 (秩 = min(m, n))  |  最大值  |  存在唯一最小二乘解。模型是良态的。  |
|  秩亏缺 (秩 < min(m, n))  |  低于最大值  |  特征冗余。有无穷多个权重解。需要正则化。  |
|  秩为1  |  1  |  每一列都是一个向量的缩放副本。所有数据位于一条直线上。  |
| 接近秩亏损（小奇异值）  |  数值低  |  矩阵病态。微小的输入噪声会导致输出大幅变化。使用SVD截断或岭回归。 |

### 投影

将向量 **a** 投影到向量 **b** 上得到 **a** 在 **b** 方向上的分量：

```
proj_b(a) = (a dot b / b dot b) * b
```

残差 (a - proj_b(a)) 垂直于 b。这种正交分解是最小二乘拟合的基础。

投影在机器学习中无处不在：
- 线性回归最小化观测值到列空间的距离——解就是一个投影
- PCA 将数据投影到最大方差的方向上
- Transformer 中的注意力机制计算查询对键的投影

```mermaid
graph LR
    subgraph Projection["Projection of a onto b"]
        direction TB
        O["Origin"] --> |"b (direction)"| B["b"]
        O --> |"a (original)"| A["a"]
        O --> |"proj_b(a)"| P["projection"]
        A -.-> |"residual (perpendicular)"| P
    end
```

**示例:** a = [3, 4], b = [1, 0]

proj_b(a) = (3*1 + 4*0) / (1*1 + 0*0) * [1, 0] = 3 * [1, 0] = [3, 0]

投影去掉了 y 分量。这是最简形式的降维——舍弃你不关心的方向。

### Gram-Schmidt 过程

将任意一组线性无关的向量转化为标准正交基。标准正交意味着每个向量的模长为1，且两两垂直。

算法：
1. 先取第一个向量(vector)，将其归一化(normalize)；再取第二个向量，减去它在第一个上的投影(projection)，然后归一化；再取第三个向量，减去它在所有之前向量上的投影，然后归一化；对剩余向量重复此过程。
2. 
3. 
4. 

```
Input:  v1, v2, v3, ... (linearly independent)

u1 = v1 / |v1|

w2 = v2 - (v2 dot u1) * u1
u2 = w2 / |w2|

w3 = v3 - (v3 dot u1) * u1 - (v3 dot u2) * u2
u3 = w3 / |w3|

Output: u1, u2, u3, ... (orthonormal basis)
```

这就是QR分解(QR decomposition)的内部工作原理。Q是标准正交基(orthonormal basis)，R捕获了投影系数(projection coefficients)。QR分解用于：
- 求解线性系统(linear systems)（比高斯消元(Gaussian elimination)更稳定）
- 计算特征值(eigenvalues)（QR算法）
- 最小二乘回归(least-squares regression)（标准数值方法）

```figure
eigen-directions
```

## 动手构建

### 步骤1：从头实现向量(vectors)（Python）

```python
class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5

    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def __repr__(self):
        return f"Vector({self.components})"


a = Vector([1, 2, 3])
b = Vector([4, 5, 6])

print(f"a + b = {a + b}")
print(f"a · b = {a.dot(b)}")
print(f"|a| = {a.magnitude():.4f}")
print(f"cosine similarity = {a.cosine_similarity(b):.4f}")
```

### 步骤2：从头实现矩阵(matrices)（Python）

```python
class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(sum(
                    self.rows[i][k] * other.rows[k][j]
                    for k in range(self.shape[1])
                ))
            rows.append(row)
        return Matrix(rows)

    def transpose(self):
        return Matrix([
            [self.rows[j][i] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ])

    def __repr__(self):
        return f"Matrix({self.rows})"


rotation_90 = Matrix([[0, -1], [1, 0]])
point = Vector([3, 1])

rotated = rotation_90 @ point
print(f"Original: {point}")
print(f"Rotated 90°: {rotated}")
```

### 步骤3：为何这对人工智能(AI)至关重要

```python
import random

random.seed(42)
weights = Matrix([[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)])
input_vector = Vector([1.0, 0.5, -0.3])

output = weights @ input_vector
print(f"Input (3D): {input_vector}")
print(f"Output (2D): {output}")
print("This is what a neural network layer does -- matrix multiplication.")
```

### 第4步：Julia 版本

```julia
a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]

println("a + b = ", a + b)
println("a · b = ", a ⋅ b)       # Julia supports unicode operators
println("|a| = ", √(a ⋅ a))
println("cosine = ", (a ⋅ b) / (√(a ⋅ a) * √(b ⋅ b)))

# Matrix-vector multiplication
W = [0.1 -0.2 0.3; 0.4 0.5 -0.1]
x = [1.0, 0.5, -0.3]
println("Wx = ", W * x)
println("This is a neural network layer.")
```

### 第5步：线性无关性与投影从零开始实现（Python）

```python
def is_linearly_independent(vectors):
    n = len(vectors)
    dim = len(vectors[0].components)
    mat = Matrix([v.components[:] for v in vectors])
    rows = [row[:] for row in mat.rows]
    rank = 0
    for col in range(dim):
        pivot = None
        for row in range(rank, len(rows)):
            if abs(rows[row][col]) > 1e-10:
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col]
        rows[rank] = [x / scale for x in rows[rank]]
        for row in range(len(rows)):
            if row != rank and abs(rows[row][col]) > 1e-10:
                factor = rows[row][col]
                rows[row] = [rows[row][j] - factor * rows[rank][j] for j in range(dim)]
        rank += 1
    return rank == n


def project(a, b):
    scalar = a.dot(b) / b.dot(b)
    return Vector([scalar * x for x in b.components])


def gram_schmidt(vectors):
    orthonormal = []
    for v in vectors:
        w = v
        for u in orthonormal:
            proj = project(w, u)
            w = w - proj
        if w.magnitude() < 1e-10:
            continue
        orthonormal.append(w.normalize())
    return orthonormal


v1 = Vector([1, 0, 0])
v2 = Vector([1, 1, 0])
v3 = Vector([1, 1, 1])
basis = gram_schmidt([v1, v2, v3])
for i, u in enumerate(basis):
    print(f"u{i+1} = {u}")
    print(f"  |u{i+1}| = {u.magnitude():.6f}")

print(f"u1 · u2 = {basis[0].dot(basis[1]):.6f}")
print(f"u1 · u3 = {basis[0].dot(basis[2]):.6f}")
print(f"u2 · u3 = {basis[1].dot(basis[2]):.6f}")
```

## 使用它

现在用 NumPy 实现同样的操作——你在实践中实际会用到的方法：

```python
import numpy as np

a = np.array([1, 2, 3], dtype=float)
b = np.array([4, 5, 6], dtype=float)

print(f"a + b = {a + b}")
print(f"a · b = {np.dot(a, b)}")
print(f"|a| = {np.linalg.norm(a):.4f}")
print(f"cosine = {np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)):.4f}")

W = np.random.randn(2, 3) * 0.1
x = np.array([1.0, 0.5, -0.3])
print(f"Wx = {W @ x}")
```

### 用 NumPy 计算秩、投影和 QR 分解

```python
import numpy as np

A = np.array([[1, 2], [2, 4]])
print(f"Rank: {np.linalg.matrix_rank(A)}")

a = np.array([3, 4])
b = np.array([1, 0])
proj = (np.dot(a, b) / np.dot(b, b)) * b
print(f"Projection of {a} onto {b}: {proj}")

Q, R = np.linalg.qr(np.random.randn(3, 3))
print(f"Q is orthogonal: {np.allclose(Q @ Q.T, np.eye(3))}")
print(f"R is upper triangular: {np.allclose(R, np.triu(R))}")
```

### PyTorch —— 张量是具有自动微分的向量

```python
import torch

x = torch.randn(3, requires_grad=True)
y = torch.tensor([1.0, 0.0, 0.0])

similarity = torch.dot(x, y)
similarity.backward()

print(f"x = {x.data}")
print(f"y = {y.data}")
print(f"dot product = {similarity.item():.4f}")
print(f"d(dot)/dx = {x.grad}")
```

点积关于 x 的梯度就是 y。PyTorch 自动计算了这个梯度。神经网络中的每个操作都是由这样的操作构建的——矩阵乘法、点积、投影——而自动微分追踪所有这些操作的梯度。

你刚刚亲手实现了NumPy一行代码完成的功能。现在你知道了底层发生了什么。

## 发布

本課(lesson)产出：
- `outputs/prompt-linear-algebra-tutor.md` -- a prompt for AI assistants to teach linear algebra through geometric intuition

## 联系

本课中的所有内容都与现代AI的特定部分相关联：

|  概念  |  出现位置  |
|---------|------------------|
|  点积(Dot product)  |  变换器中的注意力分数，RAG中的余弦相似度  |
| 矩阵乘法 | 每个神经网络层，每个线性变换 |
| 线性无关 | 特征选择，避免多重共线性 |
| 秩 | 判断系统是否可解，LoRA（低秩适应） |
| 投影 | 线性回归（投影到列空间），PCA |
| 格拉姆-施密特/QR | 数值求解器，特征值计算 |
| 标准正交基 | 稳定的数值计算，白化变换 |

LoRA值得特别提及。它通过将权重更新分解为低秩矩阵来微调大型语言模型。与更新一个4096x4096的权重矩阵（1600万个参数）不同，LoRA更新两个大小分别为4096x16和16x4096的矩阵（13.1万个参数）。秩为16的约束意味着LoRA假设权重更新位于完整4096维空间的一个16维子空间中。这就是线性代数在真实场景中的应用。

## 练习

1. 实现 `Vector.angle_between(other)` 函数，返回两个向量之间的角度（以度为单位）
2. 创建一个二维缩放矩阵，将x坐标加倍，y坐标三倍，然后将其应用于向量 [1, 1]
3. 给定5个随机词向量（维度50），使用余弦相似度找出最相似的两个
4. 验证Gram-Schmidt输出是否真正正交归一：检查每一对向量的点积是否为0，每个向量的模是否为1
5. 创建一个3x3的秩为2的矩阵，使用 `Vector.angle_between(other)` 方法验证，然后解释列向量张成的几何对象
6. 将向量 [1, 2, 3] 投影到 [1, 1, 1] 上，结果在几何上表示什么？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|----------------------|
|  向量  |  "一个箭头"  |  表示n维空间中点或方向的数字列表  |
|  矩阵  |  "数字表格"  |  将向量从一个空间映射到另一个空间的变换  |
|  点积  |  "相乘求和"  |  衡量两个向量对齐程度的度量——相似性搜索的核心  |
|  嵌入  |  "某种AI魔法"  |  表示某物（单词、图像、用户）含义的向量  |
| 线性无关(Linear independence) | “它们不重叠” | 集合中没有向量可以写成其他向量的组合 |
| 秩(Rank) | “有多少个维度” | 矩阵中线性无关的列（或行）的数量 |
| 投影(Projection) | “影子” | 一个向量在另一个向量方向上的分量 |
| 基(Basis) | “坐标轴” | 一组能够张成空间的最小独立向量集 |
| 标准正交(Orthonormal) | “垂直的单位向量” | 相互垂直且长度均为1的向量 |
