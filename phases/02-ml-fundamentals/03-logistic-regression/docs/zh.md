# 逻辑回归（Logistic Regression）

> 逻辑回归将一条直线弯曲成S形曲线，用于回答是/否问题，并给出概率。

**类型：** 构建
**语言：** Python
**前置知识：** 第二阶段第1-2课（什么是机器学习、线性回归）
**时间：** 约90分钟

## 学习目标

- 使用Sigmoid函数和二元交叉熵损失从零实现逻辑回归
- 计算并解释精确率（Precision）、召回率（Recall）、F1分数（F1 Score）和二元分类的混淆矩阵（Confusion Matrix）
- 解释为什么均方误差（MSE）不适合分类，而二元交叉熵能产生凸的代价曲面
- 构建用于多类分类的Softmax回归模型，并评估阈值调优的权衡

## 问题

你想根据肿瘤的大小预测它是恶性还是良性。你尝试使用线性回归，它输出像0.3、1.7或-0.5这样的数字。这些数字意味着什么？1.7是“非常恶性”吗？-0.5是“非常良性”吗？线性回归输出无界的数字。分类需要0到1之间的有界概率，并做出明确的决策：是或否。

逻辑回归解决了这个问题。它采用相同的线性组合（wx + b），并通过Sigmoid函数，将任何数值压缩到(0, 1)范围内。输出是一个概率。你设置一个阈值（通常为0.5）来做出决策。

这是实践中最广泛使用的算法之一。尽管名字带有“回归”，但逻辑回归是一种分类算法，而非回归算法。其名称来源于它使用的逻辑（Sigmoid）函数。

## 核心概念

### 为什么线性回归不适合分类

想象一下根据学习小时数预测通过/未通过（1/0）。线性回归会在数据上拟合一条直线：

```
hours:  1   2   3   4   5   6   7   8   9   10
actual: 0   0   0   0   1   1   1   1   1   1
```

线性拟合可能会在第1小时输出-0.2，在第10小时输出1.3。这些值不是概率，它们低于0或高于1。更糟糕的是，一个离群点（学习了50小时的人）会拖拽整条直线，从而改变每个人的预测。

分类需要一个函数，该函数：
- 输出0到1之间的值（概率）
- 产生一个锐利的过渡（决策边界）
- 不会被远离边界的离群点扭曲

### Sigmoid函数

Sigmoid函数正是这样做的：

```
sigmoid(z) = 1 / (1 + e^(-z))
```

性质：
- 当z很大且为正时，sigmoid(z)趋近于1
- 当z很大且为负时，sigmoid(z)趋近于0
- 当z = 0时，sigmoid(z) = 0.5
- 输出始终在0到1之间
- 函数处处光滑且可导

导数具有简洁形式：sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))，这使得梯度计算高效。

### 逻辑回归 = 线性模型 + Sigmoid

模型计算z = wx + b（和线性回归相同），然后应用sigmoid：

```mermaid
flowchart LR
    X[Input features x] --> L["Linear: z = wx + b"]
    L --> S["Sigmoid: p = 1/(1+e^-z)"]
    S --> D{"p >= 0.5?"}
    D -->|Yes| P[Predict 1]
    D -->|No| N[Predict 0]
```

输出p被解释为P(y=1 | x)，即输入属于类别1的概率。决策边界位于wx + b = 0处，此时sigmoid输出恰好为0.5。

### 二元交叉熵损失（Binary Cross-Entropy Loss）

你不能在逻辑回归中使用均方误差（MSE）。带Sigmoid的MSE会产生非凸的代价曲面，具有多个局部最小值。相反，应使用二元交叉熵（对数损失）：

```
Loss = -(1/n) * sum(y * log(p) + (1-y) * log(1-p))
```

为何有效：
- 当y=1且p接近1时：log(1) = 0，损失接近0（正确，低代价）
- 当y=1且p接近0时：log(0)趋近负无穷，损失巨大（错误，高代价）
- 当y=0且p接近0时：log(1) = 0，损失接近0（正确，低代价）
- 当y=0且p接近1时：log(0)趋近负无穷，损失巨大（错误，高代价）

该损失函数对于逻辑回归是凸的，保证了唯一的全局最小值。

### 逻辑回归的梯度下降（Gradient Descent）

带Sigmoid的二元交叉熵的梯度具有简洁形式：

```
dL/dw = (1/n) * sum((p - y) * x)
dL/db = (1/n) * sum(p - y)
```

这些看起来与线性回归的梯度相同。区别在于p = sigmoid(wx + b)而非p = wx + b。Sigmoid引入了非线性，但梯度更新规则保持不变。

```mermaid
flowchart TD
    A[Initialize w=0, b=0] --> B[Forward pass: z = wx+b, p = sigmoid z]
    B --> C[Compute loss: binary cross-entropy]
    C --> D["Compute gradients: dw = (1/n) * sum((p-y)*x)"]
    D --> E[Update: w = w - lr*dw, b = b - lr*db]
    E --> F{Converged?}
    F -->|No| B
    F -->|Yes| G[Model trained]
```

### 决策边界（Decision Boundary）

对于二维输入（两个特征），决策边界是满足以下条件的直线：

```
w1*x1 + w2*x2 + b = 0
```

一侧的点被分类为1，另一侧的点被分类为0。逻辑回归(Logistic Regression)总是产生线性决策边界。如果需要弯曲的边界，可以添加多项式特征(Polynomial Features)或使用非线性模型(Nonlinear Model)。

### 基于Softmax的多类分类

二元逻辑回归处理两个类别。对于k个类别，使用softmax函数：

```
softmax(z_i) = e^(z_i) / sum(e^(z_j) for all j)
```

每个类别有自己的权重向量。模型为每个类别计算得分z_i，然后softmax将得分转换为概率，概率之和为1。预测类别是概率最高的那个。

损失函数变为分类交叉熵(Categorical Cross-Entropy)：

```
Loss = -(1/n) * sum(sum(y_k * log(p_k)))
```

其中y_k对于真实类别为1，对于其他类别为0（独热编码(One-Hot Encoding)）。

### 评估指标

仅靠准确率(Accuracy)是不够的。对于包含95%负类和5%正类的数据集，一个总是预测负类的模型能达到95%的准确率，但却毫无用处。

**混淆矩阵(Confusion Matrix)**：

|   |  预测正类  |  预测负类  |
|---|---|---|
|  实际正类  |  真正例(True Positive, TP)  |  假负例(False Negative, FN)  |
|  实际负类  |  假正例(False Positive, FP)  |  真负例(True Negative, TN)  |

**精确率(Precision)**：在所有预测为正类的样本中，实际为正类的比例。
```
Precision = TP / (TP + FP)
```

**召回率(Recall)**（灵敏度(Sensitivity)）：在所有实际为正类的样本中，我们捕获了多少。
```
Recall = TP / (TP + FN)
```

**F1分数(F1 Score)**：精确率和召回率的调和平均数。平衡两个指标。
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

何时优先考虑：
- **精确率**：当假正例代价高昂时（垃圾邮件过滤器，不希望拦截合法邮件）
- **召回率**：当假负例代价高昂时（癌症筛查，不希望漏掉肿瘤）
- **F1**：当你需要一个单一的平衡指标时

```figure
logistic-sigmoid
```

## 动手构建

### 第1步：Sigmoid函数和数据生成

```python
import random
import math

def sigmoid(z):
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))


random.seed(42)
N = 200
X = []
y = []

for _ in range(N // 2):
    X.append([random.gauss(2, 1), random.gauss(2, 1)])
    y.append(0)

for _ in range(N // 2):
    X.append([random.gauss(5, 1), random.gauss(5, 1)])
    y.append(1)

combined = list(zip(X, y))
random.shuffle(combined)
X, y = zip(*combined)
X = list(X)
y = list(y)

print(f"Generated {N} samples (2 classes, 2 features)")
print(f"Class 0 center: (2, 2), Class 1 center: (5, 5)")
print(f"First 5 samples:")
for i in range(5):
    print(f"  Features: [{X[i][0]:.2f}, {X[i][1]:.2f}], Label: {y[i]}")
```

### 第2步：从零实现逻辑回归

```python
class LogisticRegression:
    def __init__(self, n_features, learning_rate=0.01):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.loss_history = []

    def predict_proba(self, x):
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return sigmoid(z)

    def predict(self, x, threshold=0.5):
        return 1 if self.predict_proba(x) >= threshold else 0

    def compute_loss(self, X, y):
        n = len(y)
        total = 0.0
        for i in range(n):
            p = self.predict_proba(X[i])
            p = max(1e-15, min(1 - 1e-15, p))
            total += y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p)
        return -total / n

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            dw = [0.0] * n_features
            db = 0.0
            for i in range(n):
                p = self.predict_proba(X[i])
                error = p - y[i]
                for j in range(n_features):
                    dw[j] += error * X[i][j]
                db += error
            for j in range(n_features):
                self.weights[j] -= self.lr * (dw[j] / n)
            self.bias -= self.lr * (db / n)
            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Loss: {loss:.4f} | w: [{self.weights[0]:.3f}, {self.weights[1]:.3f}] | b: {self.bias:.3f}")
        return self

    def accuracy(self, X, y):
        correct = sum(1 for i in range(len(y)) if self.predict(X[i]) == y[i])
        return correct / len(y)


split = int(0.8 * N)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("\n=== Training Logistic Regression ===")
model = LogisticRegression(n_features=2, learning_rate=0.1)
model.fit(X_train, y_train, epochs=1000, print_every=200)

print(f"\nTrain accuracy: {model.accuracy(X_train, y_train):.4f}")
print(f"Test accuracy:  {model.accuracy(X_test, y_test):.4f}")
print(f"Weights: [{model.weights[0]:.4f}, {model.weights[1]:.4f}]")
print(f"Bias: {model.bias:.4f}")
```

### 第3步：从零实现混淆矩阵和指标

```python
class ClassificationMetrics:
    def __init__(self, y_true, y_pred):
        self.tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        self.tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        self.fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        self.fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    def accuracy(self):
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0

    def precision(self):
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0

    def recall(self):
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0

    def f1(self):
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    def print_confusion_matrix(self):
        print(f"\n  Confusion Matrix:")
        print(f"                  Predicted")
        print(f"                  Pos   Neg")
        print(f"  Actual Pos     {self.tp:4d}  {self.fn:4d}")
        print(f"  Actual Neg     {self.fp:4d}  {self.tn:4d}")

    def print_report(self):
        self.print_confusion_matrix()
        print(f"\n  Accuracy:  {self.accuracy():.4f}")
        print(f"  Precision: {self.precision():.4f}")
        print(f"  Recall:    {self.recall():.4f}")
        print(f"  F1 Score:  {self.f1():.4f}")


y_pred_test = [model.predict(x) for x in X_test]
print("\n=== Classification Report (Test Set) ===")
metrics = ClassificationMetrics(y_test, y_pred_test)
metrics.print_report()
```

### 第4步：决策边界分析

```python
print("\n=== Decision Boundary ===")
w1, w2 = model.weights
b = model.bias
print(f"Decision boundary: {w1:.4f}*x1 + {w2:.4f}*x2 + {b:.4f} = 0")
if abs(w2) > 1e-10:
    print(f"Solved for x2:     x2 = {-w1/w2:.4f}*x1 + {-b/w2:.4f}")

print("\nSample predictions near the boundary:")
test_points = [
    [3.0, 3.0],
    [3.5, 3.5],
    [4.0, 4.0],
    [2.5, 2.5],
    [5.0, 5.0],
]
for point in test_points:
    prob = model.predict_proba(point)
    pred = model.predict(point)
    print(f"  [{point[0]}, {point[1]}] -> prob={prob:.4f}, class={pred}")
```

### 第5步：使用softmax的多类分类

```python
class SoftmaxRegression:
    def __init__(self, n_features, n_classes, learning_rate=0.01):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = learning_rate
        self.weights = [[0.0] * n_features for _ in range(n_classes)]
        self.biases = [0.0] * n_classes

    def softmax(self, scores):
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        return [e / total for e in exp_scores]

    def predict_proba(self, x):
        scores = [
            sum(self.weights[k][j] * x[j] for j in range(self.n_features)) + self.biases[k]
            for k in range(self.n_classes)
        ]
        return self.softmax(scores)

    def predict(self, x):
        probs = self.predict_proba(x)
        return probs.index(max(probs))

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        for epoch in range(epochs):
            grad_w = [[0.0] * self.n_features for _ in range(self.n_classes)]
            grad_b = [0.0] * self.n_classes
            total_loss = 0.0
            for i in range(n):
                probs = self.predict_proba(X[i])
                for k in range(self.n_classes):
                    target = 1.0 if y[i] == k else 0.0
                    error = probs[k] - target
                    for j in range(self.n_features):
                        grad_w[k][j] += error * X[i][j]
                    grad_b[k] += error
                true_prob = max(probs[y[i]], 1e-15)
                total_loss -= math.log(true_prob)
            for k in range(self.n_classes):
                for j in range(self.n_features):
                    self.weights[k][j] -= self.lr * (grad_w[k][j] / n)
                self.biases[k] -= self.lr * (grad_b[k] / n)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Loss: {total_loss / n:.4f}")
        return self

    def accuracy(self, X, y):
        correct = sum(1 for i in range(len(y)) if self.predict(X[i]) == y[i])
        return correct / len(y)


random.seed(42)
X_3class = []
y_3class = []

centers = [(1, 1), (5, 1), (3, 5)]
for label, (cx, cy) in enumerate(centers):
    for _ in range(50):
        X_3class.append([random.gauss(cx, 0.8), random.gauss(cy, 0.8)])
        y_3class.append(label)

combined = list(zip(X_3class, y_3class))
random.shuffle(combined)
X_3class, y_3class = zip(*combined)
X_3class = list(X_3class)
y_3class = list(y_3class)

split_3 = int(0.8 * len(X_3class))
X_train_3 = X_3class[:split_3]
y_train_3 = y_3class[:split_3]
X_test_3 = X_3class[split_3:]
y_test_3 = y_3class[split_3:]

print("\n=== Multi-class Softmax Regression (3 classes) ===")
softmax_model = SoftmaxRegression(n_features=2, n_classes=3, learning_rate=0.1)
softmax_model.fit(X_train_3, y_train_3, epochs=1000, print_every=200)
print(f"\nTrain accuracy: {softmax_model.accuracy(X_train_3, y_train_3):.4f}")
print(f"Test accuracy:  {softmax_model.accuracy(X_test_3, y_test_3):.4f}")

print("\nSample predictions:")
for i in range(5):
    probs = softmax_model.predict_proba(X_test_3[i])
    pred = softmax_model.predict(X_test_3[i])
    print(f"  True: {y_test_3[i]}, Predicted: {pred}, Probs: [{', '.join(f'{p:.3f}' for p in probs)}]")
```

### 第6步：阈值调优

```python
print("\n=== Threshold Tuning ===")
print("Default threshold: 0.5. Adjusting the threshold trades precision for recall.\n")

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print(f"{'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 52)

for t in thresholds:
    y_pred_t = [1 if model.predict_proba(x) >= t else 0 for x in X_test]
    m = ClassificationMetrics(y_test, y_pred_t)
    print(f"{t:>10.1f} {m.accuracy():>10.4f} {m.precision():>10.4f} {m.recall():>10.4f} {m.f1():>10.4f}")
```

## 使用它

现在使用scikit-learn实现相同的功能。

```python
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

np.random.seed(42)
X_0 = np.random.randn(100, 2) + [2, 2]
X_1 = np.random.randn(100, 2) + [5, 5]
X_sk = np.vstack([X_0, X_1])
y_sk = np.array([0] * 100 + [1] * 100)

X_tr, X_te, y_tr, y_te = train_test_split(X_sk, y_sk, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

lr = SklearnLR()
lr.fit(X_tr_sc, y_tr)
y_pred = lr.predict(X_te_sc)

print("=== Scikit-learn Logistic Regression ===")
print(f"Accuracy:  {accuracy_score(y_te, y_pred):.4f}")
print(f"Precision: {precision_score(y_te, y_pred):.4f}")
print(f"Recall:    {recall_score(y_te, y_pred):.4f}")
print(f"F1:        {f1_score(y_te, y_pred):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_te, y_pred)}")
print(f"\nClassification Report:\n{classification_report(y_te, y_pred)}")
```

你的从零实现产生了相同的决策边界和指标。scikit-learn增加了求解器选项（liblinear、lbfgs、saga）、自动正则化(Regularization)、多类策略（一对余(One-vs-Rest)、多项(Multinomial)）以及数值稳定性优化。

## 发布

本課(lesson)产出：
- `code/logistic_regression.py` - 从零实现的逻辑回归及其指标

## 练习

1. 生成一个不是线性可分(Linearly Separable)的数据集（例如两个同心圆）。训练逻辑回归并观察其失败。然后添加多项式特征（x1^2、x2^2、x1*x2）并再次训练。展示准确率提升。
2. 为3类softmax模型实现多类混淆矩阵。计算每类的精确率和召回率。哪个类别最难分类？
3. 从零构建ROC曲线。对于从0到1的100个阈值，计算真正例率和假正例率。使用梯形法则计算AUC（曲线下面积）。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|----------------------|
|  逻辑回归  |  "用于分类的回归"  |  一个线性模型后接sigmoid函数，输出类别概率  |
|  Sigmoid函数  |  "S曲线"  |  函数1/(1+e^(-z))，将任意实数映射到(0, 1)区间  |
|  二元交叉熵  |  "对数损失"  |  损失函数-[y*log(p) + (1-y)*log(1-p)]，对自信的错误预测施加严重惩罚  |
| Decision boundary (决策边界) | "分界线" | 模型输出概率等于0.5的曲面，用于分隔预测类别 |
| Softmax (Softmax函数) | "多类别sigmoid" | 将一个得分向量转换为概率之和为1的函数 |
| Precision (精确率) | "选出的有多少是相关的" | TP / (TP + FP)，即预测为正的样本中实际为正的比例 |
| Recall (召回率) | "相关的有多少被选出" | TP / (TP + FN)，即实际为正的样本中被模型正确识别的比例 |
| F1 score (F1分数) | "平衡准确率" | 精确率和召回率的调和平均数：2*P*R / (P+R) |
| Confusion matrix (混淆矩阵) | "错误分解" | 显示每个类别对的TP、TN、FP、FN计数的表格 |
| Threshold (阈值) | "截止点" | 模型预测为类别1的概率临界值（默认0.5，可调） |
| One-hot encoding (独热编码) | "类别的二进制列" | 将类别k表示为一个除第k位为1外其余为0的向量 |
| Categorical cross-entropy (分类交叉熵) | "多类别对数损失" | 使用独热编码标签将二元交叉熵扩展到k个类别 |
