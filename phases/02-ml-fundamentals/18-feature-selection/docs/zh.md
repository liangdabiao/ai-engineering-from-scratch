# 特征选择 (Feature Selection)

> 更多功能并不更好。正确的功能才是更好。

**类型：** 构建
**语言：** Python
**先决条件：** 阶段2，课程01-09，08（特征工程）
**时间：** ~75分钟

## 学习目标

- 从头实现过滤方法（方差阈值、互信息、卡方检验）和包装方法（递归特征消除、前向选择）
- 解释为何互信息能捕获相关性无法捕捉的非线性特征-目标关系
- 比较L1正则化（嵌入式选择）与递归特征消除（包装式选择），评估它们的计算权衡
- 构建一个结合多种方法的特征选择管道，并展示在留出数据上改进的泛化能力

## 问题

你拥有500个特征。模型训练缓慢，持续过拟合，且没人能解释它学到了什么。你添加更多特征希望提升性能。结果变得更糟。

这就是维数灾难的实际表现。随着特征数量增长，特征空间的体积爆炸式增长。数据点变得稀疏。点之间的距离趋同。模型需要指数级更多的数据才能找到真正的模式。噪声特征淹没了信号特征。过拟合成为常态。

特征选择是解药。剥离噪声。去除冗余。保留包含关于目标变量实际信息的特征。结果是：更快的训练，更好的泛化，以及你能够解释的模型。

目标不是使用所有可用信息，而是使用正确的信息。

## 核心概念

### 特征选择的三大类别

每个特征选择方法都属于以下三类之一：

```mermaid
flowchart TD
    A[Feature Selection Methods] --> B[Filter Methods]
    A --> C[Wrapper Methods]
    A --> D[Embedded Methods]

    B --> B1["Variance Threshold"]
    B --> B2["Mutual Information"]
    B --> B3["Chi-squared Test"]
    B --> B4["Correlation Filtering"]

    C --> C1["Recursive Feature Elimination"]
    C --> C2["Forward Selection"]
    C --> C3["Backward Elimination"]

    D --> D1["L1 / Lasso Regularization"]
    D --> D2["Tree-based Importance"]
    D --> D3["Elastic Net"]
```

**过滤方法** 使用统计指标独立评分每个特征。它们不使用模型。速度快，但忽略了特征间的交互。

**包装方法** 训练模型来评估特征子集。它们使用模型性能作为评分。结果更好，但代价高昂，因为需要多次重新训练模型。

**嵌入式方法** 在模型训练过程中选择特征。L1正则化将权重推向零。决策树在最有用的特征上分裂。选择发生在拟合过程中，而不是作为单独的步骤。

### 方差阈值

最简单的过滤方法。如果一个特征在样本间几乎不变，它几乎不携带任何信息。

考虑一个特征，在1000个样本中有999个为0.0。它的方差接近于零。任何模型都无法用它来区分类别。删除它。

```
variance(x) = mean((x - mean(x))^2)
```

设定一个阈值（例如0.01）。删除所有方差低于该阈值的特征。这在不考虑目标变量的情况下移除了常数或近似常数的特征。

何时使用：作为其他方法之前的预处理步骤。它以近乎零的成本捕获明显无用的特征。

局限性：一个特征可能具有高方差却仍然是纯噪声。方差阈值是必要的但不充分的。

### 互信息

互信息衡量知道特征X的值能将目标Y的不确定性减少多少。

```
I(X; Y) = sum_x sum_y p(x, y) * log(p(x, y) / (p(x) * p(y)))
```

如果X和Y独立，则p(x, y) = p(x) * p(y)，因此对数项为零，I(X; Y) = 0。X告诉你关于Y的信息越多，互信息就越高。

相对于相关性的关键优势：互信息捕获非线性关系。一个特征可能与目标变量的相关性为零，但由于关系是二次或周期性的，互信息很高。

对于连续特征，首先离散化为区间（基于直方图的估计）。区间的数量影响估计——区间太少丢失信息，区间太多引入噪声。常见选择：sqrt(n)个区间或斯特奇斯规则（1 + log2(n)）。

```mermaid
flowchart LR
    A[Feature X] --> B[Discretize into Bins]
    B --> C["Compute Joint Distribution p(x,y)"]
    C --> D["Compute MI = sum p(x,y) * log(p(x,y) / p(x)p(y))"]
    D --> E["Rank Features by MI Score"]
    E --> F[Select Top K]
```

### 递归特征消除（RFE）

RFE是一种包装方法。它使用模型自身的特征重要性来迭代剪枝：

1. 使用所有特征训练模型
2. 根据重要性对特征排序（线性模型用系数，树模型用不纯度减少）
3. 移除最不重要的特征
4. 重复直到剩余特征数量达到目标

```mermaid
flowchart TD
    A["Start: All N Features"] --> B["Train Model"]
    B --> C["Rank Feature Importances"]
    C --> D["Remove Least Important"]
    D --> E{"Features == Target Count?"}
    E -->|No| B
    E -->|Yes| F["Return Selected Features"]
```

RFE考虑了特征交互，因为模型同时看到所有剩余特征。移除一个特征会改变其他特征的重要性。这使得它比过滤方法更彻底。

代价：你训练模型的次数为N减去目标特征数。对于500个特征和目标10个，就是490次训练运行。对于昂贵的模型，这很慢。你可以通过每步移除多个特征来加速（例如每轮移除底部10%）。

### L1（Lasso）正则化

L1正则化在损失函数中添加权重的绝对值：

```
loss = prediction_error + alpha * sum(|w_i|)
```

alpha参数控制特征被修剪的激进程度。alpha值越大，意味着更多的权重精确变为零。

为什么精确为零？L1惩罚在权重空间中创建一个菱形约束区域。最优解往往落在这个菱形的角上，其中一个或多个权重为零。L2正则化(岭回归)创建圆形约束，权重会缩小但很少达到零。

这是嵌入式特征选择：模型在训练过程中学习忽略哪些特征。权重为零的特征被有效移除。

优点：单次训练运行，能处理相关特征（选择一个并将其他特征置零），内置于大多数线性模型实现中。

局限性：仅适用于线性模型。无法捕捉非线性特征重要性。

### 基于树的特征重要性

决策树及其集成方法（随机森林、梯度提升）自然地排序特征。每次分裂都会减少不纯度（分类用基尼系数或熵，回归用方差）。产生更大不纯度减少的特征更重要。

对于具有T棵树的随机森林：

```
importance(feature_j) = (1/T) * sum over all trees of
    sum over all nodes splitting on feature_j of
        (n_samples * impurity_decrease)
```

这为每个特征提供了一个归一化的重要性分数。它自动处理非线性关系和特征交互。

注意：基于树的重要性偏向于具有较多唯一值（高基数）的特征。一个随机ID列会因为完美分割每个样本而显得重要。使用排列重要性作为校验。

### 排列重要性

一种模型无关的方法：

1. 训练模型并记录验证数据上的基线性能
2. 对于每个特征：随机打乱其值，测量性能下降的程度
3. 下降越大，特征越重要

如果打乱某个特征不会损害性能，则模型不依赖它。如果性能崩溃，则该特征至关重要。

排列重要性避免了基于树的重要性的基数偏差。但速度较慢：每个特征需要一次完整评估，并且为稳定性需要重复多次。

### 对比表

|  方法  |  类型  |  速度  |  非线性  |  特征交互  |
|--------|------|-------|-----------|---------------------|
|  方差阈值  |  过滤法  |  非常快  |  否  |  否  |
|  互信息  |  过滤法  |  快  |  是  |  否  |
|  相关过滤  |  过滤法  |  快  |  否  |  否  |
|  RFE  |  包装法  |  慢  |  取决于模型  |  是  |
|  L1 / Lasso  |  嵌入法  |  快  |  否（线性）  |  否  |
|  树重要性  |  嵌入法  |  中  |  是  |  是  |
|  排列重要性  |  模型无关  |  慢  |  是  |  是  |

### 决策流程图

```mermaid
flowchart TD
    A[Start: Feature Selection] --> B{How many features?}
    B -->|"< 50"| C["Start with variance threshold + mutual information"]
    B -->|"50-500"| D["Variance threshold, then L1 or tree importance"]
    B -->|"> 500"| E["Variance threshold, then mutual info filter, then RFE on survivors"]

    C --> F{Using linear model?}
    D --> F
    E --> F

    F -->|Yes| G["L1 regularization for final selection"]
    F -->|No - trees| H["Tree importance + permutation importance"]
    F -->|No - other| I["RFE with your model"]

    G --> J[Validate: compare selected vs all features]
    H --> J
    I --> J

    J --> K{Performance improved?}
    K -->|Yes| L["Ship with selected features"]
    K -->|No| M["Try different method or keep all features"]
```

## 动手构建

### 步骤1：生成具有已知特征结构的合成数据

```python
import numpy as np


def make_feature_selection_data(n_samples=500, seed=42):
    rng = np.random.RandomState(seed)

    x1 = rng.randn(n_samples)
    x2 = rng.randn(n_samples)
    x3 = rng.randn(n_samples)
    x4 = x1 + 0.1 * rng.randn(n_samples)
    x5 = x2 + 0.1 * rng.randn(n_samples)

    informative = np.column_stack([x1, x2, x3, x4, x5])

    correlated = np.column_stack([
        x1 * 0.9 + 0.1 * rng.randn(n_samples),
        x2 * 0.8 + 0.2 * rng.randn(n_samples),
        x3 * 0.7 + 0.3 * rng.randn(n_samples),
        x1 * 0.5 + x2 * 0.5 + 0.1 * rng.randn(n_samples),
        x2 * 0.6 + x3 * 0.4 + 0.1 * rng.randn(n_samples),
    ])

    noise = rng.randn(n_samples, 10) * 0.5

    X = np.hstack([informative, correlated, noise])
    y = (2 * x1 - 1.5 * x2 + x3 + 0.5 * rng.randn(n_samples) > 0).astype(int)

    feature_names = (
        [f"info_{i}" for i in range(5)]
        + [f"corr_{i}" for i in range(5)]
        + [f"noise_{i}" for i in range(10)]
    )

    return X, y, feature_names
```

我们知道真实情况：特征0-4是信息性的（其中3和4是0和1的相关副本），特征5-9与信息性特征相关，特征10-19是纯噪声。一个好的选择方法应该将0-4排名最高，10-19排名最低。

### 步骤2：方差阈值

```python
def variance_threshold(X, threshold=0.01):
    variances = np.var(X, axis=0)
    mask = variances > threshold
    return mask, variances
```

### 步骤3：互信息（离散）

```python
def discretize(x, n_bins=10):
    min_val, max_val = x.min(), x.max()
    if max_val == min_val:
        return np.zeros_like(x, dtype=int)
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    binned = np.digitize(x, bin_edges[1:-1])
    return binned


def mutual_information(X, y, n_bins=10):
    n_samples, n_features = X.shape
    mi_scores = np.zeros(n_features)

    y_vals, y_counts = np.unique(y, return_counts=True)
    p_y = y_counts / n_samples

    for f in range(n_features):
        x_binned = discretize(X[:, f], n_bins)
        x_vals, x_counts = np.unique(x_binned, return_counts=True)
        p_x = dict(zip(x_vals, x_counts / n_samples))

        mi = 0.0
        for xv in x_vals:
            for yi, yv in enumerate(y_vals):
                joint_mask = (x_binned == xv) & (y == yv)
                p_xy = np.sum(joint_mask) / n_samples
                if p_xy > 0:
                    mi += p_xy * np.log(p_xy / (p_x[xv] * p_y[yi]))
        mi_scores[f] = mi

    return mi_scores
```

### 步骤4：递归特征消除

```python
def simple_logistic_importance(X, y, lr=0.1, epochs=100):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        error = pred - y
        w -= lr * (X.T @ error) / n_samples
        b -= lr * np.mean(error)

    return w, b


def rfe(X, y, n_features_to_select=5, lr=0.1, epochs=100):
    n_total = X.shape[1]
    remaining = list(range(n_total))
    rankings = np.ones(n_total, dtype=int)
    rank = n_total

    while len(remaining) > n_features_to_select:
        X_subset = X[:, remaining]
        w, _ = simple_logistic_importance(X_subset, y, lr, epochs)
        importances = np.abs(w)

        least_idx = np.argmin(importances)
        original_idx = remaining[least_idx]
        rankings[original_idx] = rank
        rank -= 1
        remaining.pop(least_idx)

    for idx in remaining:
        rankings[idx] = 1

    selected_mask = rankings == 1
    return selected_mask, rankings
```

### 步骤5：L1特征选择

```python
def soft_threshold(w, alpha):
    return np.sign(w) * np.maximum(np.abs(w) - alpha, 0)


def l1_feature_selection(X, y, alpha=0.1, lr=0.01, epochs=500):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        error = pred - y

        gradient_w = (X.T @ error) / n_samples
        gradient_b = np.mean(error)

        w -= lr * gradient_w
        w = soft_threshold(w, lr * alpha)
        b -= lr * gradient_b

    selected_mask = np.abs(w) > 1e-6
    return selected_mask, w
```

### 第6步：基于树的特征重要性（简单决策树）

```python
def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)


def best_split(X, y, feature_idx):
    values = np.unique(X[:, feature_idx])
    if len(values) <= 1:
        return None, -1.0

    best_threshold = None
    best_gain = -1.0
    parent_gini = gini_impurity(y)
    n = len(y)

    for i in range(len(values) - 1):
        threshold = (values[i] + values[i + 1]) / 2.0
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        if n_left == 0 or n_right == 0:
            continue

        gain = parent_gini - (n_left / n) * gini_impurity(y[left_mask]) - (n_right / n) * gini_impurity(y[right_mask])

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_threshold, best_gain


def tree_importance(X, y, n_trees=50, max_depth=5, seed=42):
    rng = np.random.RandomState(seed)
    n_samples, n_features = X.shape
    importances = np.zeros(n_features)

    for _ in range(n_trees):
        sample_idx = rng.choice(n_samples, size=n_samples, replace=True)
        feature_subset = rng.choice(n_features, size=max(1, int(np.sqrt(n_features))), replace=False)

        X_boot = X[sample_idx]
        y_boot = y[sample_idx]

        tree_imp = _build_tree_importance(X_boot, y_boot, feature_subset, max_depth)
        importances += tree_imp

    total = importances.sum()
    if total > 0:
        importances /= total

    return importances


def _build_tree_importance(X, y, feature_subset, max_depth, depth=0):
    n_features = X.shape[1]
    importances = np.zeros(n_features)

    if depth >= max_depth or len(np.unique(y)) <= 1 or len(y) < 4:
        return importances

    best_feature = None
    best_threshold = None
    best_gain = -1.0

    for f in feature_subset:
        threshold, gain = best_split(X, y, f)
        if gain > best_gain:
            best_gain = gain
            best_feature = f
            best_threshold = threshold

    if best_feature is None or best_gain <= 0:
        return importances

    importances[best_feature] += best_gain * len(y)

    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask

    importances += _build_tree_importance(X[left_mask], y[left_mask], feature_subset, max_depth, depth + 1)
    importances += _build_tree_importance(X[right_mask], y[right_mask], feature_subset, max_depth, depth + 1)

    return importances
```

### 第7步：运行所有方法并比较

代码文件在同一合成数据集上运行所有五种方法，并输出一个对比表格，显示每种方法选择了哪些特征。

## 使用它

使用 scikit-learn，特征选择直接集成在管道中：

```python
from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif,
    RFE,
    SelectFromModel,
)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

vt = VarianceThreshold(threshold=0.01)
X_filtered = vt.fit_transform(X)

mi_scores = mutual_info_classif(X, y)
top_k = np.argsort(mi_scores)[-10:]

rfe_selector = RFE(LogisticRegression(), n_features_to_select=10)
rfe_selector.fit(X, y)
X_rfe = rfe_selector.transform(X)

lasso_selector = SelectFromModel(Lasso(alpha=0.01))
lasso_selector.fit(X, y)
X_lasso = lasso_selector.transform(X)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)
importances = rf.feature_importances_
```

从头实现的代码展示了每种方法内部的具体操作。方差阈值只是计算`var(X, axis=0)`并应用掩码。互信息是在列联表中统计联合频率和边际频率。RFE 是一个循环：训练、排序、剪枝。L1 是带软阈值步骤的梯度下降。树重要性则是在分裂过程中累积不纯度减少量。没有魔法——只有统计和循环。

sklearn 版本增加了鲁棒性（例如，mutual_info_classif 使用 k-NN 密度估计而非分箱）、速度（C 实现）和管道集成。

## 发布

本課(lesson)产出：
- `outputs/skill-feature-selector.md` —— 用于选择合适特征选择方法的快速参考决策树

## 练习

1. **前向选择**：实现 RFE 的反向过程。从零个特征开始，每一步添加对模型性能提升最大的特征。当添加特征不再有帮助时停止。将所选特征与 RFE 的结果进行比较。哪种更快？哪种结果更好？

2. **稳定性选择**：运行 L1 特征选择 50 次，每次使用数据的随机 80% 子样本，并略微不同的 alpha 值。统计每个特征被选中的次数。在超过 80% 的运行中被选中的特征是“稳定的”。将稳定特征与单次 L1 选择的结果进行比较。哪种更可靠？

3. **多重共线性检测**：计算所有特征的相关系数矩阵。实现一个函数，给定一个相关系数阈值（例如 0.9），从每个高度相关的特征对中移除一个特征（保留与目标互信息更高的那个）。在合成数据集上测试，并验证它移除了冗余的相关特征。

4. **特征选择管道**：将方差阈值、互信息过滤和 RFE 连接成单个管道。首先移除接近零方差的特征，然后保留互信息排名前 50% 的特征，再对剩余特征运行 RFE。将此管道与对所有特征单独运行 RFE 进行比较。管道更快吗？准确度是否相同？

5. **从头实现置换重要性**：实现置换重要性。对每个特征，将其值打乱 10 次，测量 F1 分数的平均下降。将排名与基于树的特征重要性进行比较。找出它们不一致的情况并解释原因（提示：相关特征）。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|----------------------|
|  过滤法  |  "独立评分特征"  |  一种特征选择方法，使用统计量对特征进行排序，无需训练模型，独立评估每个特征  |
|  包装法  |  "用模型挑选特征"  |  一种特征选择方法，通过训练模型并以其性能作为选择标准来评估特征子集  |
|  嵌入法  |  "模型在训练过程中选择特征"  |  特征选择作为模型拟合的一部分进行，例如 L1 正则化将权重驱至零  |
|  互信息  |  "一个变量告诉你关于另一个变量的信息量"  |  衡量在已知 X 的情况下 Y 的不确定性减少量，捕捉线性和非线性依赖关系  |
|  递归特征消除  |  "训练、排序、剪枝、重复"  |  一种迭代包装法，训练模型，移除最不重要的特征，并重复直至达到目标数量  |
|  L1 / Lasso 正则化  |  "惩罚使特征消失"  |  将权重绝对值的总和添加到损失函数，使得不重要特征的权重恰好为零  |
|  方差阈值  |  "移除常数特征"  |  剔除样本间方差低于指定阈值的特征，过滤掉不携带信息的特征  |
|  特征重要性  |  "哪些特征最重要"  |  表示每个特征对模型预测贡献程度的分数，来自分裂增益（树）或系数幅度（线性模型）  |
|  置换重要性  |  "打乱并衡量损失"  |  通过随机打乱每个特征的值并测量模型性能下降来评估特征重要性  |
|  维度灾难  |  "特征太多，数据不足"  |  添加特征会使特征空间体积呈指数增长，导致数据稀疏且距离失去意义的现象  |

## 延伸阅读

- [An Introduction to Variable and Feature Selection (Guyon & Elisseeff, 2003)](https://jmlr.org/papers/v3/guyon03a.html) —— 关于特征选择方法的基础性综述，仍被广泛引用
- [An Introduction to Variable and Feature Selection (Guyon & Elisseeff, 2003)](https://jmlr.org/papers/v3/guyon03a.html) —— 过滤法、包装法和嵌入法的实用参考，附代码示例
- [An Introduction to Variable and Feature Selection (Guyon & Elisseeff, 2003)](https://jmlr.org/papers/v3/guyon03a.html) —— 将子采样与特征选择相结合，以获得稳健、可重复的结果
- [An Introduction to Variable and Feature Selection (Guyon & Elisseeff, 2003)](https://jmlr.org/papers/v3/guyon03a.html) —— 展示了树基特征重要性的基数偏差，并提出条件重要性作为替代方案
