# 概率与分布

> 概率是人工智能用来表达不确定性的语言。

**类型：** 学习
**语言：** Python
**前置条件：** 阶段1，课程01-04
**时间：** ~75分钟

## 学习目标

- 从头实现伯努利分布、类别分布、泊松分布、均匀分布和正态分布的PMF和PDF
- 计算期望值、方差，并使用中心极限定理解释为什么高斯分布占主导地位
- 构建带有数值稳定性技巧（减去最大logit）的softmax和log-softmax函数
- 根据logits计算交叉熵损失，并将其与负对数似然联系起来

## 问题

一个分类器输出`[0.03, 0.91, 0.06]`。一个语言模型从5万个候选中选择下一个词。一个扩散模型通过从学习的分布中采样来生成图像。这些都是概率的实际应用。

模型做出的每个预测都是一个概率分布。每个损失函数衡量预测分布与真实分布之间的差距。每个训练步骤调整参数以使一个分布更接近另一个分布。没有概率，你无法阅读任何一篇机器学习论文，调试任何一个模型，或者理解为什么你的训练损失是NaN。

## 核心概念

### 事件、样本空间与概率

样本空间$S$是所有可能结果的集合。事件是样本空间的子集。概率将事件映射到0到1之间的数字。

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

三个公理定义了所有概率：
1. $P(A) \geq 0$ 对任意事件$A$成立
2. $P(S) = 1$（某事总是发生）
3. 当$A$和$B$不能同时发生时，$P(A \text{ 或 } B) = P(A) + P(B)$

其他所有内容（贝叶斯定理、期望、分布）都源自这三条规则。

### 条件概率与独立性

$P(A|B)$ 是给定$B$发生时$A$的概率。

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

当知道一个事件对另一个事件一无所知时，这两个事件是独立的：

```
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

抛硬币是独立的。无放回抽牌则不是。

### 概率质量函数 vs 概率密度函数

离散随机变量有一个概率质量函数(PMF)。每个结果都有一个可以直接读出的特定概率。

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

连续随机变量有一个概率密度函数(PDF)。单点的密度不是概率。概率来自对区间上的密度进行积分。

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

这个区别在机器学习中很重要。分类输出是PMFs（离散选择）。VAE潜在空间使用PDFs（连续）。

### 常见分布

**伯努利分布：**一次试验，两个结果。用于建模二元分类。

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**类别分布：**一次试验，k个结果。用于建模多类分类（softmax输出）。

```
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**均匀分布：**所有结果等可能。用于随机初始化。

```
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**正态分布（高斯分布）：**钟形曲线。由均值(mu)和方差(sigma^2)参数化。

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**泊松分布：**在固定区间内稀有事件的计数。用于建模事件发生率。

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### 期望值和方差

期望值是加权平均结果。

```
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

方差衡量围绕均值的离散程度。

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

在机器学习中，期望值作为损失函数出现（数据分布上的平均损失）。方差告诉你模型的稳定性。梯度的方差大意味着训练噪声大。

### 联合分布和边缘分布

联合分布 P(X, Y) 描述了两个随机变量共同的行为。

联合概率质量函数示例 (X = 天气, Y = 雨伞):

|   |  Y=0 (无雨伞)  |  Y=1 (有雨伞)  |  边缘 P(X)  |
|---|---|---|---|
|  X=0 (晴天)  |  0.40  |  0.10  |  P(X=0) = 0.50  |
|  X=1 (雨天)  |  0.05  |  0.45  |  P(X=1) = 0.50  |
|  **边缘 P(Y)**  |  P(Y=0) = 0.45  |  P(Y=1) = 0.55  |  1.00  |

边缘分布通过对另一个变量求和得到：

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

上表中行和列的总和即为边缘分布。

### 为什么正态分布无处不在

中心极限定理：大量独立随机变量的和（或平均值）趋近于正态分布，无论原始分布如何。

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

原因如下：
- 测量误差近似正态（许多小的独立来源）
- 神经网络中的权重初始化使用正态分布
- SGD中的梯度噪声近似正态（许多样本梯度的和）
- 正态分布是在给定均值和方差下的最大熵分布

### 对数概率

原始概率会导致数值问题。将许多小概率相乘会很快下溢为零。

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

对数概率解决了这个问题。乘法变成了加法。

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

规则：
- log(a * b) = log(a) + log(b)
- 对数概率总是 <= 0（因为 0 < P <= 1）
- 越负表示可能性越小
- 交叉熵损失是正确类别的负对数概率

### Softmax 作为概率分布

神经网络输出原始分数（logits）。Softmax 将它们转换为有效的概率分布。

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

softmax 技巧：在指数运算前减去最大 logit 以防止溢出。

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

对数 softmax 结合了 softmax 和对数以实现数值稳定性。PyTorch 在内部使用它来计算交叉熵损失。

### 采样

采样意味着从分布中随机抽取值。在机器学习中：
- Dropout 随机采样要置零的神经元
- 数据增强随机采样变换
- 语言模型从预测分布中采样下一个 token
- 扩散模型采样噪声并逐步去噪

从任意分布中采样需要诸如逆变换采样、拒绝采样或重参数化技巧（用于 VAE）等技术。

```figure
gaussian-pdf
```

## 动手构建

### 第 1 步：概率基础

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### 第 2 步：从零开始理解 PMF 和 PDF

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### 第 3 步：期望值和方差

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### 第 4 步：从分布中采样

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### 第 5 步：Softmax 和对数概率

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### 第6步：中心极限定理演示

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 第7步：可视化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

所有可视化的完整实现位于 `code/probability.py` 中。

## 使用它

使用 NumPy 和 SciPy，上述所有操作都是一行代码：

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

你是从零开始构建这些的。现在你知道了库调用背后的原理。

## 练习

1. 为指数分布实现逆变换采样。通过抽取10,000个样本并将直方图与真实概率密度函数(PDF)进行比较来验证。

2. 为两个非均匀骰子构建联合分布表。计算边缘分布并检查骰子是否独立。

3. 计算一个5类分类器的交叉熵损失，该分类器输出logits `[2.0, 0.5, -1.0, 3.0, 0.1]`，正确类别索引为3。然后使用PyTorch的`nn.CrossEntropyLoss`验证你的答案。

4. 编写一个函数，接收对数概率列表，返回最可能的序列、总对数概率以及等效的原始概率。用一个由50个单词组成的句子来测试，其中每个单词的概率为0.01。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|----------------------|
|  样本空间  |  "所有可能性"  |  集合S，包含实验的所有可能结果  |
|  概率质量函数(PMF)  |  "概率函数"  |  一个函数，给出每个离散结果的精确概率，总和为1  |
|  概率密度函数(PDF)  |  "概率曲线"  |  连续变量的密度函数。在区间上积分得到概率。 |
|  条件概率  |  "给定某条件下的概率"  |  P(A\ | B) = P(A与B同时发生) / P(B)。贝叶斯思维和贝叶斯定理的基础。 |
|  独立性  |  "互不影响"  |  P(A与B同时发生) = P(A) * P(B)。知道一个事件对另一个事件没有提供任何信息。 |
|  期望值  |  "平均值"  |  所有结果的概率加权和。损失函数是一种期望值。 |
|  方差  |  "分布有多分散"  |  与均值的期望平方偏差。高方差=噪声大、不稳定的估计 |
|  正态分布(Normal distribution)  |  "钟形曲线"  |  f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2))。由于中心极限定理(CLT)无处不在 |
|  中心极限定理(Central Limit Theorem)  |  "平均值趋于正态"  |  大量独立样本的均值收敛到正态分布，无论原始分布是什么 |
|  联合分布(Joint distribution)  |  "两个变量一起"  |  P(X, Y) 描述了 X 和 Y 所有组合结果的概率 |
|  边缘分布(Marginal distribution)  |  "对另一个变量求和"  |  P(X) = sum_y P(X, Y)。从联合分布中恢复单个变量的分布 |
|  对数概率(Log probability)  |  "概率的对数"  |  log P(x)。将乘积转化为和，防止长序列中的数值下溢 |
|  Softmax  |  "将分数转为概率"  |  softmax(z_i) = exp(z_i) / sum(exp(z_j))。将实值逻辑值映射为有效的概率分布 |
|  交叉熵(Cross-entropy)  |  "损失函数"  |  -sum(p_true * log(p_predicted))。衡量两个分布的差异。越低越好 |
|  逻辑值(Logits)  |  "原始模型输出"  |  softmax 之前的未归一化分数。得名于逻辑函数 |
|  采样(Sampling)  |  "抽取随机值"  |  根据概率分布生成值。模型生成输出的方式 |

## 延伸阅读

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 平均值为何变为正态的直观证明
- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 涵盖此处所有内容及更多的简洁参考
- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 数值稳定性为何重要以及如何实现
