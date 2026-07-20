# 机器学习统计学

> 统计学能让你知道你的模型是真的有效，还是只是运气好。

**类型：** 构建
**语言：** Python
**先修条件：** 阶段1，第06课（概率与分布），第07课（贝叶斯定理）
**时间：** 约120分钟

## 学习目标

- 从零开始计算描述性统计量、皮尔逊/斯皮尔曼相关性和协方差矩阵
- 正确执行假设检验（t检验、卡方检验）并解释p值和置信区间
- 使用自助法重抽样为任何指标构建置信区间，无需分布假设
- 使用效应量指标区分统计显著性与实际显著性

## 问题

你训练了两个模型。模型A在测试集上得分0.87。模型B得分0.89。你部署了模型B。三周后，生产环境的指标比以前更差了。发生了什么？

模型B实际上并没有优于模型A。0.02的差异是噪声。你的测试集太小，或者方差太高，或者两者都有。你把伪装成改进的随机性部署上线了。

这种情况经常发生。Kaggle排行榜波动。无法复现的论文。基于几百个样本就宣布胜利者的A/B测试。根本原因总是一样的：有人跳过了统计学。

统计学为你提供了区分信号和噪声的工具。它告诉你一个差异是否真实，你应该有多自信，以及在你可以信任结果之前需要多少数据。每个机器学习流水线、每个模型比较、每个实验都需要统计学。没有它，你就是在猜测。

## 核心概念

### 描述性统计：总结你的数据

在对任何东西建模之前，你需要知道你的数据是什么样的。描述性统计将数据集压缩成几个数字，捕捉其形态。

**集中趋势的度量**回答“中间在哪里？”

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

均值是平衡点。中位数是中间值。当它们偏离时，你的分布是偏斜的。收入分布中均值 >> 中位数（亿万富翁导致的右偏）。训练期间的损失分布通常均值 << 中位数（易样本导致的左偏）。

**离散程度的度量**回答“数据有多分散？”

```
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**百分位数**将排序后的数据分成100个相等的部分。第25百分位数（Q1）意味着25%的值低于这个点。第50百分位数是中位数。第75百分位数是Q3。

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

在机器学习中，你关心推理延迟、预测置信度分布以及理解误差分布的百分位数。一个平均误差低但P99误差高的模型可能在安全关键应用中毫无用处。

**样本与总体统计量。** 当从样本计算方差时，除以(n-1)而不是n。这是贝塞尔校正。它补偿了样本均值不是真正的总体均值这一事实。分母为n时，你会系统性地低估真实方差。分母为(n-1)时，估计是无偏的。

```
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

在实践中：如果n很大（数千个样本），差异可忽略。如果n很小（几十个样本），则很重要。

### 相关性：变量如何一起变化

相关性衡量两个变量之间线性关系的强度和方向。

**皮尔逊相关系数**衡量线性关联：

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

皮尔逊相关系数假设关系是线性的，并且两个变量大致服从正态分布。它对异常值敏感。一个极端的点可以将r从0.1拉到0.9。

**斯皮尔曼秩相关系数**衡量单调关联：

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**何时使用哪种：**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**黄金法则：** 相关性不代表因果关系。冰淇淋销量和溺水死亡人数相关，因为两者都在夏季增加。你的模型准确率和参数数量相关，但增加参数不会自动提高准确率（参见：过拟合）。

### 协方差矩阵

两个变量之间的协方差衡量它们如何一起变化：

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

对于d个特征，协方差矩阵C是一个d×d矩阵，其中C[i][j] = Cov(特征_i, 特征_j)。对角线条目C[i][i]是每个特征的方差。

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**与PCA的联系。** PCA对协方差矩阵进行特征分解。特征向量是主成分（最大方差的方向）。特征值告诉你每个成分捕捉了多少方差。这正是第10课的内容，但现在你明白为什么协方差矩阵是正确的分解对象：它编码了数据中所有成对的线性关系。

**与相关性的联系。** 相关矩阵是标准化变量（每个除以自己的标准差）的协方差矩阵。相关性对协方差进行归一化，使得所有值落在[-1, 1]之间。

### 假设检验

假设检验是在不确定性下做出决策的框架。你从某个主张开始，收集数据，然后判断数据是否与该主张一致。

**设置：**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**p值**是假设H0为真的情况下，观察到与你所观测数据一样极端的数据的概率。它不是H0为真的概率。这是统计学中最常见的误解。

```
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**置信区间**给出了参数的合理取值范围：

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

置信区间的宽度反映了精度。宽区间意味着高不确定性。窄区间意味着你的估计精确（但如果数据有偏，则不一定准确）。

### t检验

t检验用于比较均值。有几种不同的类型。

**单样本t检验：**总体均值是否不同于假设值？

```
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**独立双样本t检验：**两个组的均值是否不同？

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**配对t检验：**当测量值成对出现时（例如，在同一数据划分上评估相同模型）：

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

在机器学习中，配对t检验很常见：你在相同的10个交叉验证折上运行两个模型，并成对比较它们的得分。

### 卡方检验

卡方检验检查观测频数是否与期望频数一致。适用于分类数据。

```
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### 机器学习模型的A/B测试

机器学习中的A/B测试与网页A/B测试不同。模型比较面临特定的挑战：

```
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**流程：**

```
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### 统计显著性 vs 实际显著性

一个结果可能在统计上显著，但在实际中毫无意义。当数据足够多时，即使是微小的差异也会变得统计显著。

```
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**效应量**量化了差异的大小，与样本量无关：

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

始终同时报告p值和效应量。p值告诉你差异是否真实存在。效应量告诉你差异是否重要。

### 多重比较问题

当你检验多个假设时，有些会偶然变得“显著”。如果你在α=0.05的水平上检验20个假设，即使所有假设都不成立，你也会预期得到1个假阳性。

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**邦费罗尼校正：**将α除以检验次数。

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

在机器学习中，当你跨多个指标比较模型、测试许多超参数配置或在多个数据集上评估时，这一点很重要。

### 自助法

自助法通过有放回地重抽样数据来估计统计量的抽样分布。不需要对底层分布作任何假设。

**算法：**

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**自助法置信区间（百分位法）：**

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**为什么自助法对机器学习很重要：**

```
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**用于模型比较的自助法：**

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

这比配对t检验更稳健，因为无需分布假设。

### 参数检验与非参数检验

**参数检验(Parametric tests)**假设特定分布（通常是正态分布）：

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**非参数检验(Non-parametric tests)**无分布假设：

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**何时使用非参数检验：**

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**何时使用参数检验：**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

在机器学习实验中，通常样本量较小（5或10折交叉验证），因此Wilcoxon符号秩检验等非参数检验往往比t检验更合适。

### 中心极限定理(Central Limit Theorem)：实际意义

中心极限定理指出，随着样本量n增大，样本均值的分布趋近于正态分布，无论总体分布如何。

```
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

**这对机器学习的重要性：**

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**中心极限定理不能做什么：**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### 机器学习论文中常见的统计错误

1. **在训练集上测试。**这必然导致过拟合。始终保留模型在训练过程中从未见过的数据。

2. **无置信区间。**报告一个单一的准确率数值而不包含不确定性，导致结果无法复现和验证。

3. **忽略多重比较。**测试50种配置并报告最好结果而不进行校正，会提高假阳性率。

4. **混淆统计显著性与实际显著性。**准确率提升0.01%时p值为0.001并无实际意义。

5. **在不平衡数据上使用准确率。**在负类占99%的数据集上达到99%的准确率，意味着模型什么都没学到。应使用精确率、召回率、F1或AUC。

6. **挑选有利指标。**只报告模型表现好的指标。诚实的评估应报告所有相关指标。

7. **在训练/测试划分中泄露信息。**在划分前进行归一化，或用未来数据预测过去。

8. **小测试集且无方差估计。**在100个样本上评估并声称提升了2%，这属于噪声而非信号。

9. **数据不独立时假设独立性。**来自同一患者的医学图像，同一文档的多个句子。组内观测值存在相关性。

10. **P值操纵(P-hacking)。**尝试不同的检验、子集或排除标准，直到获得p<0.05。结果是搜索的产物。

## 动手构建

你将实现：

1. **从头实现描述性统计**（均值、中位数、众数、标准差、百分位数、四分位距）
2. **相关函数**（皮尔逊相关系数和斯皮尔曼相关系数，含协方差矩阵）
3. **假设检验**（单样本t检验、双样本t检验、卡方检验）
4. **自助法置信区间**（适用于任何统计量，无需假设）
5. **A/B测试模拟器**（生成数据、检验、检查第一类错误和第二类错误）
6. **统计显著性与实际显著性演示**（展示大样本量使一切“显著”）

全部从头实现，仅使用`math`和`random`。不使用numpy或scipy。

## 关键术语

|  术语  |  定义  |
|---|---|
|  均值(Mean)  |  值之和除以个数。对异常值敏感。  |
|  中位数(Median)  |  排序数据的中间值。对异常值稳健。  |
|  标准差(Standard deviation)  |  方差的平方根。以原始单位衡量离散程度。  |
| 百分位数(Percentile)  |  给定百分比之下的数据值。 |
| 四分位距(IQR)  |  四分位距。Q3减Q1。中间50%的散布范围。 |
| 皮尔逊相关系数(Pearson correlation)  |  度量两个变量之间的线性关联。范围[-1,1]。 |
| 斯皮尔曼相关系数(Spearman correlation)  |  使用秩度量单调关联。 |
| 协方差矩阵(Covariance matrix)  |  所有特征之间成对协方差的矩阵。 |
| 零假设(Null hypothesis)  |  没有效应或没有差异的默认假设。 |
| p值(p-value)  |  在零假设为真的条件下，观察到如此极端数据的概率。 |
| 置信区间(Confidence interval)  |  在给定置信水平下，参数的可能取值区间。 |
| t检验(t-test)  |  检验均值是否显著不同。使用t分布。 |
| 卡方检验(Chi-squared test)  |  检验观测频数是否与期望频数有差异。 |
| 效应量(Effect size)  |  差异的幅度，独立于样本量。常见的有Cohen's d。 |
| 邦费罗尼校正(Bonferroni correction)  |  将显著性阈值除以检验次数以控制假阳性。 |
| 自助法(Bootstrap)  |  有放回地重抽样以估计抽样分布。 |
| 第一类错误(Type I error)  |  假阳性。拒绝真实的H0。 |
| 第二类错误(Type II error)  |  假阴性。未拒绝错误的H0。 |
| 统计功效(Statistical power)  |  正确拒绝错误的H0的概率。功效 = 1 - 第二类错误率。 |
| 中心极限定理(Central limit theorem)  |  随着样本量增大，样本均值收敛于正态分布。 |
| 参数检验(Parametric test)  |  假设数据服从特定分布（通常为正态分布）。 |
| 非参数检验(Non-parametric test)  |  不假设分布。基于秩或符号。 |
