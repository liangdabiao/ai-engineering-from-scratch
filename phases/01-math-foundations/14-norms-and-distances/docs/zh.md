# 范数与距离

> 你的距离函数定义了"相似"的含义。选错了，下游的所有东西都会出问题。

**类型:** 构建
**语言:** Python
**先决条件:** 阶段1，课程01（线性代数直觉），02（向量、矩阵与运算）
**时间:** 约90分钟

## 学习目标

- 从零实现L1、L2、余弦、马氏距离、杰卡德距离和编辑距离函数
- 为给定的机器学习任务选择合适的距离度量，并解释为什么其他替代方案失败
- 将L1和L2范数与LASSO和岭正则化及其几何约束区域联系起来
- 展示相同的数据集在不同度量下如何产生不同的最近邻

## 问题

你有两个向量。它们可能是词嵌入。可能是用户画像。可能是像素数组。你需要知道：它们有多接近？

答案完全取决于你选择哪个距离函数。两个数据点在一个度量下可能是最近邻，而在另一个度量下却相距甚远。你的KNN分类器、推荐引擎、向量数据库、聚类算法、损失函数——它们都依赖于这个选择。选错了，你的模型就会为错误的东西进行优化。

没有普遍最好的距离。L2适用于空间数据。余弦相似度主导自然语言处理。杰卡德距离处理集合。编辑距离处理字符串。马氏距离考虑相关性。Wasserstein距离移动概率质量。每一种都编码了对"相似"含义的不同假设。

本课从零构建每个主要距离函数，向你展示每个函数何时是合适的工具，并演示相同的数据如何根据你使用的度量产生完全不同的最近邻。

## 核心概念

### 范数：测量向量的大小

范数衡量向量的"大小"。两个向量之间的每个距离函数都可以写成它们差的范数：d(a, b) = ||a - b||。因此，理解范数就是理解距离。

### L1范数（曼哈顿距离）

L1范数求所有分量的绝对值之和。

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

它被称为曼哈顿距离，因为它测量你在只能沿坐标轴移动的城市网格上行走的距离。没有对角线。

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

何时使用L1：
- 高维稀疏数据（文本特征、独热编码）
- 当你希望对异常值具有鲁棒性时（单个巨大差异不会主导）
- 特征选择问题（L1正则化促进稀疏性）

与L1正则化（Lasso）的联系：在损失函数中添加||w||_1会惩罚绝对权重值的和。这会迫使小权重变为精确零，从而自动进行特征选择。L1惩罚在权重空间中创建菱形约束区域，菱形的角位于坐标轴上，其中一些权重为零。

与损失函数的联系：平均绝对误差(MAE)是预测值与目标值之间L1距离的平均值。它线性地惩罚所有误差，与MSE相比对异常值具有鲁棒性。

### L2范数（欧几里得距离）

L2范数是直线距离。分量平方和的平方根。

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

这就是你在几何课上学到的距离。n维空间中的勾股定理。

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

何时使用L2：
- 低到中维度的连续数据
- 当特征尺度可比较时
- 物理距离（空间数据、传感器读数）
- 像素级别的图像相似度

与L2正则化（Ridge）的联系：在损失函数中添加||w||_2^2会惩罚大权重。与L1不同，它不会将权重推到零。它按比例将所有权重向零收缩。L2惩罚创建圆形约束区域，因此坐标轴上没有角。权重变小但很少精确为零。

与损失函数的联系：均方误差(MSE)是L2距离平方的平均值。平方惩罚大误差比小误差更重。

```
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Lp范数：通用家族

L1和L2是Lp范数的特例：

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

不同的p值产生不同形状的"单位球"（所有到原点距离为1的点的集合）：

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L无穷范数（切比雪夫距离）

当p趋近于无穷大时，Lp范数收敛于最大绝对分量。

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

两点之间的距离由它们差异最大的单一维度决定。所有其他维度都被忽略。

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

何时使用 L-infinity 范数：
- 当单个维度上的最坏情况偏差至关重要时
- 游戏棋盘（国际象棋中的王走 L-infinity 范数：任意方向一步花费 1）
- 制造公差（每个维度都必须在规格范围内）

### 余弦相似度与余弦距离

余弦相似度衡量两个向量之间的角度，忽略其大小。

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

其取值范围从 -1（方向相反）到 +1（方向相同）。垂直向量的余弦相似度为 0。

余弦距离将其转换为距离：余弦距离 = 1 - 余弦相似度。取值范围从 0（方向相同）到 2（方向相反）。

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

为什么余弦相似度主导 NLP 和嵌入：在文本中，文档长度不应影响相似度。一篇关于猫的文档，如果长度是另一篇同类文档的两倍，仍应被认为是“相似”的。余弦相似度忽略大小（长度），只关心方向。两篇词分布相同但长度不同的文档指向同一方向，余弦相似度为 1.0。

何时使用余弦相似度：
- 文本相似度（TF-IDF 向量、词嵌入、句子嵌入）
- 任何大小是噪声、方向是信号的领域
- 推荐系统（用户偏好向量）
- 嵌入搜索（向量数据库几乎总是使用余弦或点积）

### 点积相似度与余弦相似度

两个向量的点积为：

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

余弦相似度是点积归一化后除以两个向量的大小。当两个向量都已单位归一化（大小 = 1）时，点积和余弦相似度相同。

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

它们的不同之处：点积包含大小信息。大小更大的向量获得更高的点积分值。这在某些检索系统中很重要，因为您希望“热门”项目排名更高。大小充当了隐式的质量或重要性信号。

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

实际应用中：
- 当您希望纯方向相似度时使用余弦相似度
- 当大小携带有意义信息时使用点积
- 许多向量数据库（Pinecone、Weaviate、Qdrant）允许您选择
- 如果您的嵌入是 L2 归一化的，则选择无关紧要

### 马氏距离 (Mahalanobis Distance)

欧氏距离 (Euclidean Distance) 平等对待所有维度。但如果特征相关或尺度不同，L2 范数会产生误导性结果。

马氏距离考虑了数据的协方差结构。

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

其中 S 是数据的协方差矩阵。

直观上：马氏距离首先对数据进行去相关和归一化（白化），然后在该变换空间中计算 L2 范数距离。如果 S 是单位矩阵（不相关、单位方差特征），马氏距离退化为欧氏距离。

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

何时使用马氏距离：
- 异常检测（距离均值马氏距离大的点是异常值）
- 当特征具有不同尺度和相关性时的分类
- 当您有足够的数据来估计可靠的协方差矩阵时
- 制造业质量控制（多变量过程监控）

### 杰卡德相似度 (Jaccard Similarity)（用于集合）

杰卡德相似度衡量两个集合的重叠程度。

```
J(A, B) = |A intersect B| / |A union B|
```

其取值范围从 0（无重叠）到 1（相同集合）。杰卡德距离 = 1 - 杰卡德相似度。

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

何时使用杰卡德相似度：
- 比较标签、类别或特征的集合
- 基于词出现（而非频率）的文档相似度
- 近似重复检测（MinHash 近似杰卡德）
- 比较二值特征向量（存在/不存在数据）
- 评估分割模型（交并比 = 杰卡德相似度）

### 编辑距离 (Levenshtein Distance)

编辑距离计算将一个字符串转换为另一个字符串所需的最小单字符操作数。操作包括：插入、删除或替换。

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

使用动态规划计算。填充一个矩阵，其中条目 (i, j) 是字符串 A 的前 i 个字符与字符串 B 的前 j 个字符之间的编辑距离。

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

何时使用编辑距离：
- 拼写检查和纠错
- DNA序列比对（带加权操作）
- 模糊字符串匹配
- 杂乱文本数据的去重

### KL散度（不是距离，但被当作距离使用）

KL散度衡量一个概率分布与另一个概率分布的差异。在第九课中会涉及，但它属于本讨论范围，因为人们常将其作为“距离”使用，尽管它并非真正的距离。

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

关键性质：KL散度不是对称的。

```
D_KL(P || Q) != D_KL(Q || P)
```

这意味着它不满足距离度量的基本要求。同时它也不满足三角不等式。它是一种散度，而非距离。

前向KL（D_KL(P || Q)）是“均值寻求”：Q试图覆盖P的所有模态。
反向KL（D_KL(Q || P)）是“模式寻求”：Q聚焦于P的单一模态。

当你看到KL散度时：
- 变分自编码器（ELBO中的KL项将潜在分布推向先验）
- 知识蒸馏（学生尝试匹配教师的分布）
- 基于人类反馈的强化学习（KL惩罚使微调模型接近基础模型）
- 策略梯度方法（约束策略更新）

### Wasserstein距离（推土机距离）

Wasserstein距离衡量将一个概率分布变换为另一个所需的最小“工作量”。可以这样理解：如果一个分布是一堆土，另一个是一个坑，你需要移动多少土以及多远？

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

对于一维分布，它简化为累积分布函数绝对差值的积分：

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

为什么Wasserstein重要：
- 它是真正的度量（对称，满足三角不等式）
- 即使分布不重叠，它也能提供梯度（KL散度会趋于无穷大）
- 这一特性使其成为Wasserstein生成对抗网络（WGAN）的核心，解决了原始GAN训练不稳定的问题

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

何时使用Wasserstein：
- 生成对抗网络训练（WGAN、WGAN-GP）
- 比较可能不重叠的分布
- 最优传输问题
- 图像检索（比较颜色直方图）

### 为什么不同任务需要不同距离

|  任务  |  最佳距离  |  原因  |
|------|--------------|-----|
|  文本相似度  |  余弦  |  量值是噪声，方向是意义  |
|  图像像素比较  |  L2  |  空间关系重要，特征尺度可比  |
|  稀疏高维特征  |  L1  |  鲁棒，不会放大罕见的大差异  |
|  集合重叠（标签、类别）  |  Jaccard  |  数据天然是集合值，非向量  |
|  字符串匹配  |  编辑距离  |  操作映射到人类编辑直觉  |
|  异常检测  |  马氏距离  |  考虑特征相关性和尺度  |
|  比较分布  |  KL散度  |  衡量用Q代替P所损失的信息  |
|  生成对抗网络训练  |  Wasserstein  |  即使分布不重叠也能提供梯度  |
|  嵌入（向量数据库）  |  余弦或点积  |  嵌入被训练为方向编码意义  |
|  推荐系统  |  点积  |  量值可编码流行度或置信度  |
|  DNA序列  |  加权编辑距离  |  替换成本因核苷酸对而异  |
|  制造质量控制  |  L-无穷  |  任何维度上的最坏偏差至关重要  |

### 与损失函数的联系

损失函数是应用于预测值与目标值的距离函数。

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### 与正则化的联系

正则化在损失函数上添加权重的范数惩罚项。

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

为什么L1产生稀疏性而L2不产生：想象二维权重空间中的约束区域。L1是一个菱形，L2是一个圆。损失函数的等高线（椭圆）最有可能与菱形在角点处相切，此时一个权重为零。它们与圆在平滑点处相切，此时两个权重都不为零。

### 最近邻搜索

每个距离函数都隐含一个最近邻搜索问题：给定一个查询点，在数据集中找到最近的点。

在具有n个点和d维的数据集中，精确最近邻搜索每次查询的时间复杂度为O(n * d)。对于大规模数据集，这太慢了。

近似最近邻(ANN)算法以少量的精度换取巨大的速度提升：

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW（分层可导航小世界）是现代向量数据库中的主导算法。它构建了一个多层图，每个节点与其近似最近邻相连。搜索从顶层（稀疏，长跳跃）开始，下降到底层（密集，短跳跃）。

```figure
norm-unit-balls
```

## 动手构建

### 第1步：所有范数和距离函数

参见`code/distances.py`获取完整实现。每个函数都仅使用基本的Python数学从头构建。

### 第2步：相同数据，不同距离，不同邻居

`distances.py`中的演示创建了一个数据集，选取一个查询点，并展示最近邻如何根据距离度量而变化。在L1下"最近"的点可能在L2或余弦下不是最近的。

### 第3步：嵌入相似性搜索

代码包含一个模拟的嵌入相似性搜索，使用余弦相似度与L2距离查找与查询最相似的"文档"，表明排名可能不同。

## 使用它

最常见的实际用途：在向量数据库中查找相似项。

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

当你调用`model.encode(text)`并搜索向量数据库时，后台会发生这些事情。嵌入模型将文本映射为向量。向量数据库计算查询向量与每个存储向量之间的余弦相似度（或点积），并使用ANN算法避免检查所有向量。

## 练习

1. 计算(1, 2, 3)和(4, 0, 6)之间的L1、L2和L-无穷距离。验证对于任意两个点，L-无穷 <= L2 <= L1始终成立。证明为什么这个顺序是保证的。

2. 创建两个向量，使得余弦相似度高（> 0.9）但L2距离大（> 10）。从几何角度解释发生了什么。然后创建两个向量，使得余弦相似度低（< 0.3）但L2距离小（< 0.5）。

3. 实现一个函数，接受数据集和查询点，返回L1、L2、余弦和马氏距离下的最近邻。找到一个数据集，使得所有四个距离在哪个点最近上意见不一致。

4. 使用CDF方法手动计算[0.5, 0.5, 0, 0]和[0, 0, 0.5, 0.5]之间的Wasserstein距离。然后计算[0.25, 0.25, 0.25, 0.25]和[0, 0, 0.5, 0.5]之间的。哪个更大，为什么？

5. 实现MinHash用于近似Jaccard相似度。生成100个随机集合，计算所有对的确切Jaccard值，并与使用50、100和200个哈希函数的MinHash近似进行比较。绘制近似误差图。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|----------------------|
|  范数  |  "向量的大小"  |  将向量映射为非负标量的函数，满足三角不等式、绝对齐次性，且仅零向量的范数为零  |
|  L1范数  |  "曼哈顿距离"  |  各分量绝对值之和。在优化中产生稀疏性。对异常值鲁棒  |
|  L2范数  |  "欧几里得距离"  |  各分量平方和的平方根。欧几里得空间中的直线距离  |
|  Lp范数  |  "广义范数"  |  各分量绝对值的p次幂之和的p次根。L1和L2是特例  |
|  L-无穷范数  |  "最大范数"或"切比雪夫距离"  |  各分量绝对值的最大值。Lp当p趋于无穷时的极限  |
|  余弦相似度  |  "向量之间的夹角"  |  点积除以两个模长。取值范围从-1到+1。忽略向量长度  |
|  余弦距离  |  "1减去余弦相似度"  |  将余弦相似度转换为距离。取值范围从0到2  |
| 点积(Dot product) | "未归一化的余弦" | 分量乘积之和。等于余弦相似度乘以两个向量模长。 |
| 马哈拉诺比斯距离(Mahalanobis distance) | "考虑相关性的距离" | 在使用数据协方差矩阵进行白化(去相关并归一化)后的空间中的L2距离。 |
| 杰卡德相似度(Jaccard similarity) | "集合重叠度" | 交集大小除以并集大小。适用于集合，不适用于向量。 |
| 编辑距离(Edit distance) | "莱文斯坦距离" | 将一个字符串转换为另一个字符串所需的最少插入、删除和替换操作数。 |
| KL散度(KL divergence) | "分布间的距离" | 并非真正的距离(不对称)。衡量使用Q编码P所需的额外比特数。 |
| 瓦瑟斯坦距离(Wasserstein distance) | "推土机距离" | 将一个分布的质量搬运到另一个分布所需的最小功。一种真正的度量。 |
| 近似最近邻(Approximate nearest neighbor) | "ANN搜索" | 比精确搜索更快地找到近似最近点的算法(HNSW, LSH, IVF)。 |
| HNSW | "向量数据库算法" | 分层可导航小世界图(Hierarchical Navigable Small World graph)。用于快速近似最近邻搜索的多层图。 |
| L1正则化(L1 regularization) | "套索(Lasso)" | 在损失中加入权重的L1范数。驱使权重变为零(稀疏性)。 |
| L2正则化(L2 regularization) | "岭回归(Ridge)或权重衰减(weight decay)" | 在损失中加入权重的平方L2范数。将权重向零收缩，但不产生稀疏性。 |
| 弹性网络(Elastic Net) | "L1+L2" | 结合L1和L2正则化。比单独使用其中一种能更好地处理相关特征组。 |

## 延伸阅读

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - Meta用于十亿级ANN搜索的库
- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - 将推土机距离引入GAN的论文
- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - 基础ANN算法
- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - Word2Vec，其中余弦相似度成为嵌入的默认度量
- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - scikit-learn中距离度量和邻近算法的实用指南
