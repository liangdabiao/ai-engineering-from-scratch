# 时间序列基础

> 过去的表现确实能预测未来的结果——前提是先检查平稳性。

**类型：** 构建
**语言：** Python
**先修知识：** 第二阶段，第01-09课
**时间：** 约90分钟

## 学习目标

- 将时间序列分解为趋势、季节性和残差分量并检验平稳性
- 实现滞后特征和滚动统计，将时间序列转化为监督学习问题
- 构建一个前向验证框架，防止未来数据泄露到训练中
- 解释为什么随机训练/测试划分对时间序列无效，并展示与正确时间划分之间的性能差距

## 问题

你拥有按时间排序的数据：每日销售额、每小时温度、每分钟CPU使用率、每周股票价格。你想预测下一个值、下一周、下一季度。

你拿起标准的机器学习工具包：随机训练/测试划分、交叉验证、特征矩阵输入、预测输出。每一步都是错的。

时间序列打破了标准机器学习所依赖的假设。样本不是独立的——今天的气温取决于昨天。随机划分会将未来信息泄露到过去。在回测中表现良好的特征在生产中失效，因为它们依赖于随时间变化的模式。

一个在随机交叉验证中达到95%准确率的模型，在正确的时间评估下可能只有55%。这种差异并非技术细节。它区分了纸上可用的模型和实际生产中可用的模型。

本课涵盖基础知识：时间数据的特殊性、如何诚实地评估模型，以及如何将时间序列转化为标准机器学习模型可以使用的特征。

## 核心概念

### 时间序列的特殊性

标准机器学习假设独立同分布——独立同分布。每个样本都从同一分布中独立抽取。时间序列违反了这两点：

- **非独立。** 今天的股票价格取决于昨天。本周的销售额与上周相关。
- **非同分布。** 分布随时间变化。12月的销售额与3月的销售额看起来不同。

这些违反并非微不足道。它们改变了你构建特征、评估模型以及选择可用算法的方式。

```mermaid
flowchart LR
    subgraph IID["Standard ML (i.i.d.)"]
        direction TB
        S1[Sample 1] ~~~ S2[Sample 2]
        S2 ~~~ S3[Sample 3]
    end
    subgraph TS["Time Series (not i.i.d.)"]
        direction LR
        T1[t=1] --> T2[t=2]
        T2 --> T3[t=3]
        T3 --> T4[t=4]
    end

    style S1 fill:#dfd
    style S2 fill:#dfd
    style S3 fill:#dfd
    style T1 fill:#ffd
    style T2 fill:#ffd
    style T3 fill:#ffd
    style T4 fill:#ffd
```

在标准机器学习中，样本是可互换的。打乱它们不会改变任何东西。在时间序列中，顺序就是一切。打乱会破坏信号。

### 时间序列的组成部分

每个时间序列由以下几部分组成：

```mermaid
flowchart TD
    A[Observed Time Series] --> B[Trend]
    A --> C[Seasonality]
    A --> D[Residual/Noise]

    B --> E[Long-term direction: up, down, flat]
    C --> F[Repeating patterns: daily, weekly, yearly]
    D --> G[Random variation after removing trend and seasonality]
```

- **趋势：** 长期方向。收入每年增长10%。全球气温上升。
- **季节性：** 固定间隔的重复模式。12月零售额激增。7月空调使用量达到峰值。
- **残差：** 去除趋势和季节性后剩下的部分。如果残差看起来像白噪声，则分解捕获了信号。

### 平稳性

如果时间序列的统计特性（均值、方差、自相关）不随时间变化，则该序列是平稳的。大多数预测方法假设平稳性。

**为什么重要：** 非平稳序列的均值会漂移。在1月数据上训练的模型学到了与2月数据不同的均值。系统性地出错。

**如何检验：** 计算窗口上的滚动均值和滚动标准差。如果它们漂移，则序列是非平稳的。

**如何修正：** 差分。不对原始值建模，而是对相邻值之间的变化建模：

```
diff[t] = value[t] - value[t-1]
```

如果一轮差分不能使序列平稳，则再次应用（二阶差分）。大多数实际序列最多需要两轮。

**示例：**

原始序列：[100, 102, 106, 112, 120]
一阶差分：[2, 4, 6, 8]（仍在上升趋势）
二阶差分：[2, 2, 2]（常数——平稳）

原始序列具有二次趋势。一阶差分将其转化为线性趋势。二阶差分使其平坦。在实践中，很少需要超过两轮差分。

**正式检验：** 增广迪基-富勒检验是平稳性的标准统计检验。原假设是“序列是非平稳的”。p值低于0.05意味着你可以拒绝原假设并得出平稳性结论。我们不从头实现ADF（它需要渐近分布表），但我们代码中的滚动统计方法提供了实用的视觉检查。

### 自相关

自相关衡量时间t的值与时间t-k（过去的k步）的值之间的相关程度。自相关函数绘制了每个滞后k的相关系数。

**ACF告诉你：**
- 序列记忆多远。若ACF在滞后5后降至零，则5步前的值无关紧要。
- 是否存在季节性。若ACF在滞后12（月度数据）处有尖峰，则存在年度季节性。
- 创建多少滞后特征。使用直到ACF变得可忽略的滞后。

**PACF（偏自相关函数）**去除间接相关性。如果今天与3天前相关仅仅是因为两者都与昨天相关，那么滞后3的PACF将为0，而滞后3的ACF则不会。

### 滞后特征：将时间序列转化为监督学习

标准机器学习模型需要特征矩阵X和目标值y。时间序列只提供一个单列数值。桥梁就是滞后特征。

以序列[10, 12, 14, 13, 15]为例，创建滞后1和滞后2特征：

|  lag_2  |  lag_1  |  target  |
|-------|-------|--------|
|  10     |  12     |  14      |
|  12     |  14     |  13      |
|  14     |  13     |  15      |

现在你有了一个标准的回归问题。任何机器学习模型（线性回归、随机森林、梯度提升）都可以从滞后特征预测目标值。

你可以构建的其他特征：
- **滚动统计量：** 最近k个值的均值、标准差、最小值、最大值
- **日历特征：** 星期几、月份、是否节假日、是否周末
- **差分值：** 相对于前一步的变化
- **扩展统计量：** 累积均值、累积和
- **比率特征：** 当前值/滚动均值（离近期平均值的距离）
- **交互特征：** lag_1 * day_of_week（星期几对动量效应的影响）

**需要多少滞后？** 使用自相关函数。若ACF在滞后10内显著，则使用至少10个滞后。若存在周季节性，则包含滞后7（可能还有14）。更多滞后给模型提供更多历史信息，但也带来更多特征需要拟合，增加过拟合风险。

**目标对齐陷阱。** 创建滞后特征时，目标必须是时间t的值，所有特征必须使用时间t-1或更早的值。如果不小心将时间t的值作为特征纳入，你就会得到一个完美的预测器——但模型完全无用。这是时间序列特征工程中最常见的错误。

### 前向验证

这是本课最重要的概念。标准的k折交叉验证随机将样本分配到训练集和测试集。对于时间序列，这会导致未来信息泄露。

```mermaid
flowchart TD
    subgraph WRONG["Random Split (WRONG)"]
        direction LR
        W1[Jan] --> W2[Mar]
        W2 --> W3[Feb]
        W3 --> W4[May]
        W4 --> W5[Apr]
        style W1 fill:#fdd
        style W3 fill:#fdd
        style W5 fill:#fdd
        style W2 fill:#dfd
        style W4 fill:#dfd
    end

    subgraph RIGHT["Walk-Forward (CORRECT)"]
        direction LR
        R1["Train: Jan-Mar"] --> R2["Test: Apr"]
        R3["Train: Jan-Apr"] --> R4["Test: May"]
        R5["Train: Jan-May"] --> R6["Test: Jun"]
        style R1 fill:#dfd
        style R2 fill:#fdd
        style R3 fill:#dfd
        style R4 fill:#fdd
        style R5 fill:#dfd
        style R6 fill:#fdd
    end
```

前向验证方法：
1. 在时间t之前的数据上训练
2. 预测时间t+1（或多步时为t+1到t+k）
3. 向前滑动窗口
4. 重复

每个测试折只包含所有训练数据之后的数据。没有未来泄露。这能让你对模型部署后的表现有一个诚实的估计。

**扩展窗口**使用所有历史数据进行训练（窗口增长）。**滑动窗口**使用固定大小的训练窗口（窗口滑动）。当你认为旧数据仍然相关时使用扩展窗口。当世界变化、旧数据有负面影响时使用滑动窗口。

### ARIMA直觉

ARIMA是经典的时间序列模型。它由三个部分组成：

- **AR（自回归）：** 根据过去的值进行预测。AR(p)使用最近p个值。
- **I（整合）：** 通过差分使序列平稳。I(d)进行d轮差分。
- **MA（移动平均）：** 根据过去的预测误差进行预测。MA(q)使用最近q个误差。

ARIMA(p, d, q)将三者结合。你可以基于ACF/PACF分析或自动搜索（auto-ARIMA）选择p、d、q。

我们不会从头实现ARIMA——它需要数值优化，超出了本课的范围。关键是要理解每个组件的作用，以便解释ARIMA结果并知道何时使用它。

### 何时使用何种方法

|  方法  |  最佳适用场景  |  处理季节性  |  处理外部特征  |
|----------|---------|-------------------|------------------------|
|  滞后特征+机器学习  |  表格数据，许多外部特征  |  结合日历特征  |  是  |
|  ARIMA  |  单变量序列，短期  |  通过SARIMA变体  |  否（ARIMAX有限支持）  |
|  指数平滑  |  简单趋势+季节性  |  是（Holt-Winters）  |  否  |
| 先知(Prophet) | 业务预测, 节假日 | 是(傅里叶项) | 有限 |
| 神经网络(LSTM, Transformer) | 长序列, 多序列 | 学习得到 | 是 |

对于大多数实际问题，滞后特征+梯度提升是最强的起点。它自然地处理外部特征，不需要平稳性，且易于调试。

### 预测范围与策略

单步预测预测一个时间步。多步预测预测多个步。有三种策略：

**递归(迭代式):** 预测一步，将该预测用作下一步的输入。简单但误差累积——每个预测都使用前一个预测，因此错误会叠加。

**直接式:** 为每个预测范围训练一个单独的模型。模型-1预测t+1，模型-5预测t+5。没有误差累积，但每个模型的训练样本更少，且不共享信息。

**多输出式:** 训练一个模型同时输出所有预测范围。跨预测范围共享信息，但需要支持多输出的模型（或自定义损失函数）。

对于大多数实际问题，短范围（1-5步）从递归式开始，长范围则使用直接式。

### 时间序列常见错误

| 错误 | 原因 | 解决方法 |
|---------|---------------|-----------|
| 随机训练/测试划分 | 标准机器学习的习惯 | 使用向前滚动或时间划分 |
| 使用未来特征 | 时间t的特征被错误包含 | 检查每个特征的时间对齐 |
| 对季节性过拟合 | 模型记忆了日历模式 | 在测试集中留出一个完整的季节周期 |
| 忽略尺度变化 | 收入翻倍但模式不变 | 建模百分比变化而非绝对值 |
| 过多滞后特征 | “历史越多越好” | 使用ACF确定相关滞后期 |
| 不进行差分 | “模型会自己搞定” | 树模型处理趋势；线性模型需要平稳性 |

## 动手构建

`code/time_series.py`中的代码从头实现了核心构建模块。

### 滞后特征创建器

```python
def make_lag_features(series, n_lags):
    n = len(series)
    X = np.full((n, n_lags), np.nan)
    for lag in range(1, n_lags + 1):
        X[lag:, lag - 1] = series[:-lag]
    valid = ~np.isnan(X).any(axis=1)
    return X[valid], series[valid]
```

将一维序列转换为特征矩阵，其中每一行以最近`n_lags`个值作为特征，当前值作为目标。

### 向前滚动交叉验证

```python
def walk_forward_split(n_samples, n_splits=5, min_train=50):
    assert min_train < n_samples, "min_train must be less than n_samples"
    step = max(1, (n_samples - min_train) // n_splits)
    for i in range(n_splits):
        train_end = min_train + i * step
        test_end = min(train_end + step, n_samples)
        if train_end >= n_samples:
            break
        yield slice(0, train_end), slice(train_end, test_end)
```

每次划分确保训练数据严格在测试数据之前。训练窗口随着每一折扩大。

### 简单自回归模型

纯AR模型只是对滞后特征进行线性回归：

```python
class SimpleAR:
    def __init__(self, n_lags=5):
        self.n_lags = n_lags
        self.weights = None
        self.bias = None

    def fit(self, series):
        X, y = make_lag_features(series, self.n_lags)
        # Solve via normal equations
        X_b = np.column_stack([np.ones(len(X)), X])
        theta = np.linalg.lstsq(X_b, y, rcond=None)[0]
        self.bias = theta[0]
        self.weights = theta[1:]
        return self
```

这概念上与第二课中的线性回归相同，但应用于同一变量的时间滞后版本。

### 平稳性检验

代码计算滚动统计量来从视觉和数值上评估平稳性：

```python
def check_stationarity(series, window=50):
    rolling_mean = np.array([
        series[max(0, i - window):i].mean()
        for i in range(1, len(series) + 1)
    ])
    rolling_std = np.array([
        series[max(0, i - window):i].std()
        for i in range(1, len(series) + 1)
    ])
    return rolling_mean, rolling_std
```

如果滚动均值漂移或滚动标准差变化，则序列非平稳。应用差分并再次检查。

代码还通过比较序列的前半部分和后半部分来检查平稳性。如果均值差异超过半个标准差或方差比超过2倍，则标记序列为非平稳。

### 自相关

```python
def autocorrelation(series, max_lag=20):
    n = len(series)
    mean = series.mean()
    var = series.var()
    acf = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        cov = np.mean((series[:n-k] - mean) * (series[k:] - mean))
        acf[k] = cov / var if var > 0 else 0
    return acf
```

## 使用它

使用sklearn，您可以直接将滞后特征与任何回归器一起使用：

```python
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

X, y = make_lag_features(series, n_lags=10)

for train_idx, test_idx in walk_forward_split(len(X)):
    model = Ridge(alpha=1.0)
    model.fit(X[train_idx], y[train_idx])
    predictions = model.predict(X[test_idx])
```

对于ARIMA，使用statsmodels：

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train_series, order=(5, 1, 2))
fitted = model.fit()
forecast = fitted.forecast(steps=30)
```

`time_series.py`中的代码演示了两种方法，并通过滚动向前验证(walk-forward validation)进行比较。

### sklearn的TimeSeriesSplit

sklearn提供了实现滚动向前验证的`TimeSeriesSplit`：

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
```

这等同于我们从头实现的`walk_forward_split`，但融入到了sklearn的交叉验证框架中。你可以将其与`cross_val_score`一起使用：

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=5))
print(f"Mean score: {scores.mean():.4f} +/- {scores.std():.4f}")
```

### 评估指标

时间序列预测使用回归指标，但需考虑时间上下文：

- **MAE（平均绝对误差）：** |y_true - y_pred|的平均值。易于以原始单位解释。"平均而言，预测偏差3.2度。"
- **RMSE（均方根误差）：** 均方误差的平方根。比MAE更惩罚大误差。当大误差比多个小误差更严重时使用。
- **MAPE（平均绝对百分比误差）：** |误差 / 真实值| * 100的平均值。无量纲，适用于不同序列间的比较。但当真实值为零时未定义。
- **朴素基线比较：** 始终与简单基线比较。季节性朴素基线预测上一个周期（昨天、上周）的值。如果你的模型无法击败朴素基线，那么一定有问题。

### 滚动特征

代码演示了向滞后特征添加滚动统计量（7天和14天窗口的均值、标准差、最小值、最大值）。这些为模型提供了滞后特征单独无法捕捉的近期趋势和波动信息。

例如，如果滚动均值上升，表明存在上升趋势。如果滚动标准差增加，表明波动性增大。这些是树模型可以学习但线性模型无法学习的模式。

## 发布

本課(lesson)产出：
- `outputs/prompt-time-series-advisor.md` -- 用于框架时间序列问题的提示
- `outputs/prompt-time-series-advisor.md` -- 滞后特征、滚动向前验证、AR模型、平稳性检验

### 你必须击败的基线

在构建任何模型之前，先建立基线：

1. **最后值（持久性）。** 预测明天与今天相同。对于许多序列，这出人意料地难以击败。
2. **季节性朴素。** 预测今天与上周同一天（或去年同一天）相同。如果你的模型无法击败这个，它就没有学到除季节性之外的任何有用模式。
3. **移动平均。** 预测最近k个值的平均值。平滑噪声，但无法捕捉突变。

如果你花哨的机器学习模型输给了季节性朴素基线，那你就有问题。最常见的是：特征中的未来泄露、错误的评估方法，或者序列确实是随机的且不可预测。

### 实用技巧

1. **从绘图开始。** 在进行任何建模之前，先绘制原始序列。寻找趋势、季节性、异常值、结构性突变（行为的突然变化）。30秒的目视检查通常比一小时的自动分析更能说明问题。

2. **先差分，后建模。** 如果序列有明显趋势，在创建滞后特征之前先进行差分。树模型可以处理趋势，但线性模型不能，而差分无害。

3. **至少保留一个完整的季节性周期。** 如果你有周季节性，你的测试集需要至少一整周。如果是月季节性，至少一整月。否则你将无法评估模型是否捕捉到了季节性模式。

4. **在生产中监控。** 时间序列模型会随着世界的变化而随时间退化。滚动跟踪预测误差。当误差开始增加时，在最新数据上重新训练模型。

5. **警惕制度变化。** 在疫情前数据上训练的模型无法预测疫情后的行为。将已知制度变化的指示器作为特征，或使用遗忘旧数据的滑动窗口。

6. **对偏态序列进行对数变换。** 收入、价格和计数通常右偏。取对数可以稳定方差，并将乘法模式变为加法模式，线性模型可以处理。在对数空间进行预测，然后取指数恢复原始单位。

## 练习

1. **平稳性实验。** 生成一个带有线性趋势的序列。用滚动统计量检查平稳性。应用一阶差分。再次检查。对于二次趋势需要多少轮差分？

2. **滞后选择。** 在季节性序列（周期=7）上计算ACF。哪些滞后具有最高的自相关性？仅使用这些滞后（而不是连续滞后）创建滞后特征。与使用滞后1到7相比，精度是否提高？

3. **滚动向前 vs 随机分割。** 在滞后特征上训练岭回归。使用随机80/20分割和滚动向前验证进行评估。随机分割高估了多少性能？

4. **特征工程。** 向滞后特征添加滚动均值（窗口=7）、滚动标准差（窗口=7）和星期几特征。使用滚动向前验证比较有和没有这些额外特征时的精度。

5. **多步预测。** 修改AR模型以预测未来5步而不是1步。比较两种策略：(a) 预测一步，将预测值作为下一步的输入（递归），(b) 为每个预测视界分别训练模型（直接）。哪个更准确？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|----------------------|
|  平稳性(Stationarity)  |  "统计量不随时间变化"  |  均值、方差和自相关结构随时间恒定的序列  |
|  差分(Differencing)  |  "减去连续值"  |  计算y[t] - y[t-1]以去除趋势并达到平稳性  |
|  自相关(ACF)  |  "序列与自身的相关性"  |  时间序列与其滞后副本之间的相关性，作为滞后的函数  |
|  偏自相关(PACF)  |  "仅直接相关"  |  在去除所有较短滞后效应后的滞后k处的自相关  |
|  滞后特征(Lag features)  |  "将过去值作为输入"  |  使用 y[t-1], y[t-2], ..., y[t-k] 作为特征来预测 y[t]  |
|  前向验证(Walk-forward validation)  |  "尊重时间的交叉验证"  |  评估时训练数据始终按时间顺序先于测试数据  |
|  ARIMA  |  "经典时间序列模型"  |  自回归整合移动平均：结合过去值(AR)、差分(I)和过去误差(MA)  |
|  季节性(Seasonality)  |  "重复的日历模式"  |  时间序列中与日历周期（日、周、年）相关的规律可预测的循环  |
|  趋势(Trend)  |  "长期方向"  |  序列水平随时间持续增加或减少  |
|  扩展窗口(Expanding window)  |  "使用所有历史"  |  前向验证中训练集随每次折叠增长  |
|  滑动窗口(Sliding window)  |  "固定大小历史"  |  前向验证中训练集是固定长度窗口向前滑动  |

## 延伸阅读

- [Hyndman and Athanasopoulos, Forecasting: Principles and Practice (3rd ed.)](https://otexts.com/fpp3/) -- 最好的免费时间序列预测教科书
- [Hyndman and Athanasopoulos, Forecasting: Principles and Practice (3rd ed.)](https://otexts.com/fpp3/) -- sklearn的前向验证分割器
- [Hyndman and Athanasopoulos, Forecasting: Principles and Practice (3rd ed.)](https://otexts.com/fpp3/) -- 带诊断的ARIMA实现
- [Hyndman and Athanasopoulos, Forecasting: Principles and Practice (3rd ed.)](https://otexts.com/fpp3/) -- 展示机器学习方法与统计方法的大规模预测竞赛
