# 朴素贝叶斯

> "天真的"假设是错误的，但它仍然有效。这正是它的美妙之处。

**类型：** 构建
**语言：** Python
**先决条件：** 阶段2，第1-7课（分类，贝叶斯定理）
**时间：** 约75分钟

## 学习目标

- 从头实现带拉普拉斯平滑的多项朴素贝叶斯用于文本分类
- 解释为什么朴素独立性假设在数学上是错误的，但在实践中能产生正确的类别排序
- 比较多项朴素贝叶斯、伯努利朴素贝叶斯和高斯朴素贝叶斯的变体，并为给定特征类型选择合适的一种
- 在高维稀疏数据上评估朴素贝叶斯与逻辑回归的性能，并解释其中的偏差-方差权衡

## 问题

你需要对文本进行分类。邮件分为垃圾邮件和非垃圾邮件。客户评论分为正面或负面。支持工单分为不同类别。你有数千个特征（每个词一个特征）且训练数据有限。

大多数分类器在这里会失效。逻辑回归需要足够多的样本来可靠地估计数千个权重。决策树一次只在一个词上分裂，容易严重过拟合。在10000维空间中KNN没有意义，因为每个点与其他所有点距离相等。

朴素贝叶斯可以处理这种情况。它做出一个在数学上错误的假设（给定类别下每个特征与其他所有特征独立），但在文本分类中，它在小训练集上的表现仍然优于"更聪明"的模型。它只需一次数据遍历即可完成训练，可扩展到数百万个特征，并能产生概率估计（尽管由于独立性假设，概率校准通常较差）。

理解为什么一个错误的假设能带来好的预测，能让你学到机器学习中的一个基本原理：最好的模型不是最正确的模型，而是对你的数据拥有最佳偏差-方差权衡的模型。

## 核心概念

### 贝叶斯定理（快速回顾）

贝叶斯定理翻转了条件概率：

```
P(class | features) = P(features | class) * P(class) / P(features)
```

我们想要 `P(class | features)`——给定文档中的词语，文档属于某个类别的概率。我们可以从以下计算得到：
- `P(features | class)`——在该类别文档中看到这些词语的似然
- `P(features | class)`——类别的先验概率（垃圾邮件通常有多常见？）
- `P(features | class)`——证据，所有类别相同，因此在比较时可以忽略

拥有最高 `P(class | features)` 的类别获胜。

### 朴素独立性假设

精确计算 `P(features | class)` 需要估计所有特征联合分布。如果有10000个词汇，则需要估计 2^10000 种组合的分布。这是不可能的。

朴素假设：给定类别，每个特征条件独立。

```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

我们不再估计一个不可能的联合分布，而是估计 n 个简单的单特征分布。每个分布只需要一个计数。

这个假设显然错误。词语"machine"和"learning"在任何文档中都不是独立的。但分类器不需要正确的概率估计。它需要正确的排序——哪个类别的概率最高。独立性假设引入系统性误差，但这些误差对所有类别影响相似，因此排序保持正确。

### 为什么仍然有效

三个原因：

1. **排序重于校准。** 分类只需要排名最高的类别正确。即使 P(spam) = 0.99999 而真实概率为0.7，分类器仍然正确选择了垃圾邮件。我们不需要正确的概率。我们需要正确的胜者。

2. **高偏差，低方差。** 独立性假设是一个强先验。它强烈约束模型，防止过拟合。在有限训练数据下，一个略有错误但稳定的模型胜过理论上正确但极不稳定的模型。这就是偏差-方差权衡的实际体现。

3. **特征冗余相互抵消。** 相关特征提供冗余证据。分类器会重复计算这些证据，但也会为正确类别重复计算。如果"machine"和"learning"总是同时出现，两者都为"技术"类别提供证据。朴素贝叶斯将其计数两次，但为正确类别计数了两次。

第四个实际原因是：朴素贝叶斯速度极快。训练只需一次数据遍历来统计频次。预测是矩阵乘法。你可以在几秒内训练百万文档的模型。这种速度意味着你可以更快迭代、尝试更多特征组合、比慢速模型运行更多实验。

### 逐步数学推导

让我们通过具体例子来追踪。假设有两个类别：垃圾邮件和非垃圾邮件。词汇表有三个词："free"、"money"、"meeting"。

训练数据：
- 垃圾邮件中"free"出现80次，"money"出现60次，"meeting"出现10次（总共150词）
- 非垃圾邮件中"free"出现5次，"money"出现10次，"meeting"出现100次（总共115词）
- 40%的邮件是垃圾邮件，60%是非垃圾邮件

使用拉普拉斯平滑（alpha=1）：

```
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

新邮件包含："free"（2次），"money"（1次），"meeting"（0次）。

```
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

垃圾邮件以很大优势胜出。"free"出现两次是垃圾邮件的强有力证据。注意"meeting"没有出现对两个对数总和贡献为零（0 * log(P)）——在多项朴素贝叶斯中，缺失的词没有影响。显式建模词缺失的是伯努利朴素贝叶斯。

### 三种变体

朴素贝叶斯有三种变体。每种变体对`P(feature | class)`的建模方式不同。

#### 多项式朴素贝叶斯

将每个特征建模为计数。最适合特征为词频或TF-IDF值的文本数据。

```
P(word_i | class) = (count of word_i in class + alpha) / (total words in class + alpha * vocab_size)
```

其`alpha`是拉普拉斯平滑（如下所述）。该变体是文本分类的主力。

#### 高斯朴素贝叶斯

将每个特征建模为正态分布。最适合连续特征。

```
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

每个类别对每个特征有自己的均值和方差。当特征在每个类别内确实服从钟形曲线时效果良好。

#### 伯努利朴素贝叶斯

将每个特征建模为二元变量（存在或不存在）。最适合短文本或二元特征向量。

```
P(word_i | class) = (docs in class containing word_i + alpha) / (total docs in class + 2 * alpha)
```

与多项式不同，伯努利明确惩罚词语的缺失。如果“免费”通常出现在垃圾邮件中但该邮件中没有，伯努利会将其作为反对垃圾邮件的证据。

### 何时使用每种变体

|  变体  |  特征类型  |  最佳用途  |  示例  |
|---------|-------------|----------|---------|
|  多项式  |  计数或频率  |  文本分类，词袋模型  |  邮件垃圾邮件，主题分类  |
|  高斯  |  连续值  |  具有近似正态特征的表格数据  |  鸢尾花分类，传感器数据  |
|  伯努利  |  二元（0/1）  |  短文本，二元特征向量  |  短信垃圾邮件，存在/缺失特征  |

### 拉普拉斯平滑

如果测试数据中出现一个词，但某个类别的训练数据中从未出现过该词，会发生什么？

没有平滑：`P(word | class) = 0/N = 0`。一个零乘遍整个乘积会使`P(class | features) = 0`为零，无论其他证据如何。一个未见过的词会破坏整个预测，无论其他证据多么支持。

拉普拉斯平滑为每个特征计数添加一个小计数`alpha`（通常为1）：

```
P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```

当alpha=1时，每个词至少有一个微小的概率。测试邮件中出现单词“discombobulate”不再使垃圾邮件概率为零。平滑具有贝叶斯解释：相当于在词分布上放置了均匀的狄利克雷先验。

更高的alpha意味着更强的平滑（更均匀的分布）。更低的alpha意味着模型更信任数据。Alpha是你可以调节的超参数。

Alpha的影响：

|  Alpha  |  影响  |  何时使用  |
|-------|--------|-------------|
|  0.001  |  几乎没有平滑，信任数据  |  非常大的训练集，预期未见特征很少  |
|  0.1  |  轻度平滑  |  大型训练集  |
|  1.0  |  标准拉普拉斯平滑  |  默认起点  |
|  10.0  |  强平滑，使分布平坦  |  非常小的训练集，预期有很多未见特征  |

### 对数空间计算

将数百个概率（每个小于1）相乘会导致浮点数下溢。乘积在浮点数中变为零，而实际值是一个非常小的正数。

解决方案：在对数空间中计算。不乘概率，而是加其对数：

```
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

这就将预测转化为了一个点积。

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

矩阵乘法。这就是朴素贝叶斯预测如此快速的原因——它与单层线性模型的操作相同。

### 朴素贝叶斯与逻辑回归

两者都是文本的线性分类器。区别在于它们所建模的对象。

|  方面  |  朴素贝叶斯  |  逻辑回归  |
|--------|------------|-------------------|
|  类型  |  生成式（建模 P(X\ | Y)）  |  判别式（建模 P(Y\ | X)）  |
|  训练  |  计数频率  |  优化损失函数  |
|  小数据  |  更好（强先验有帮助）  |  更差（不足以估计权重）  |
|  大数据  |  更差（错误假设有害）  |  更好（决策边界灵活）  |
|  特征  |  假设独立性  |  处理相关性  |
|  速度  |  单次遍历，非常快  |  迭代优化  |
|  校准  |  概率质量较差  |  概率质量更好  |

经验法则：从朴素贝叶斯开始。如果数据足够多且朴素贝叶斯性能停滞，则切换到逻辑回归。

### 分类流水线

```mermaid
flowchart LR
    A[Raw Text] --> B[Tokenize]
    B --> C[Build Vocabulary]
    C --> D[Count Word Frequencies]
    D --> E[Apply Smoothing]
    E --> F[Compute Log Probabilities]
    F --> G[Predict: argmax P class given words]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

实践中，我们在对数空间操作以避免浮点数下溢。我们不直接乘许多小概率，而是相加它们的对数：

```
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

```figure
naive-bayes
```

## 动手构建

`code/naive_bayes.py`中的代码从零实现了多项式朴素贝叶斯和高斯朴素贝叶斯。

### 多项式朴素贝叶斯

从零实现的代码：

1. **fit(X, y)**：对于每个类别，计算每个特征的频率。添加拉普拉斯平滑。计算对数概率。存储类别先验（类别频率的对数）。

2. **predict_log_proba(X)**：对于每个样本，计算全类别的 log P(类别) + log P(特征_i | 类别) 的和。这是一个矩阵乘法：X @ log_probs.T + log_priors。

3. **predict(X)**：返回对数概率最高的类别。

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

关键洞察：拟合后，预测就是矩阵乘法加偏置。这就是朴素贝叶斯如此快速的原因。

### 高斯朴素贝叶斯

对于连续特征，我们估计每个类别每个特征的均值和方差：

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

预测使用每个特征的高斯概率密度函数，跨特征相乘（在对数空间相加）。

### 演示：文本分类

代码生成模拟两个类别（科技文章 vs 体育文章）的合成词袋数据。每个类别具有不同的词频分布。多项式朴素贝叶斯使用词计数对其进行分类。

合成数据的工作原理如下：我们创建了200个“词”（特征列）。词0-39在科技文章中频率高，在体育文章中低。词80-119在体育文章中频率高，在科技文章中低。词40-79在两个类别中都是中等频率。这就创造了一个现实场景，其中一些词是强类别指示器，而另一些是噪声。

### 演示：连续特征

代码生成类似鸢尾花的数据（3个类别，4个特征，高斯簇）。高斯朴素贝叶斯使用每个类别的均值和方差进行分类。每个类别具有不同的中心（均值向量）和不同的离散程度（方差），模拟了不同类别间测量值系统性差异的现实世界数据。

代码还演示了：
- **平滑度比较：** 使用不同的alpha值训练MultinomialNB，以显示平滑强度对准确率的影响。
- **训练规模实验：** 随着训练数据从20个样本增加到1600个，朴素贝叶斯(Naive Bayes)准确率如何提升。即使样本很少，朴素贝叶斯也能达到不错的准确率——这是它的主要优势。
- **混淆矩阵：** 每个类别的精确率、召回率和F1分数，以显示朴素贝叶斯在哪些类别上出错。

### 预测速度

朴素贝叶斯的预测是一个矩阵乘法。对于n个样本、d个特征和k个类别：
- MultinomialNB：一次矩阵乘法 (n x d) @ (d x k) = O(n * d * k)
- GaussianNB：n * k次高斯概率密度函数(Gaussian PDF)计算，每次在d个特征上进行 = O(n * d * k)

两者在每个维度上都是线性的。相比之下，KNN（需要对所有训练点进行距离计算）或带RBF核的SVM（需要对所有支持向量进行核评估）则慢得多。朴素贝叶斯在预测时快了几个数量级。

## 使用它

在sklearn中，两种变体都是一行代码：

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

对于使用sklearn的文本分类：

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

在`naive_bayes.py`中的代码将从头实现的版本与sklearn在相同数据上进行对比，以验证正确性。

### TF-IDF结合朴素贝叶斯

原始词频统计给每个单词的每次出现赋予相同权重。但像“the”和“is”这样的常见词在每个类别中频繁出现——它们不携带信息。TF-IDF（词频-逆文档频率）降低常见词权重，提高稀有、有区分力的词权重。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

TF-IDF值是非负的，因此它们适用于MultinomialNB。TF-IDF与MultinomialNB的组合是文本分类中最强的基线之一。在训练样本少于10,000个的数据集上，它常常击败更复杂的模型。

### 短文本的BernoulliNB

对于短文本（推文、短信、聊天消息），BernoulliNB可能比MultinomialNB表现更好。短文本的词数少，因此MultinomialNB依赖的频率信息噪声大。BernoulliNB只关心词语出现与否，这更可靠。

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

CountVectorizer中的`binary=True`标志将所有计数转换为0/1。没有它，BernoulliNB仍然有效，但它看到的是它并非为计数而设计的。

### 校准朴素贝叶斯概率

朴素贝叶斯的概率校准不好。当朴素贝叶斯说P(垃圾邮件)=0.95时，真实概率可能只有0.7。如果你需要可靠的概率估计（例如，设置阈值或与其他模型结合），请使用sklearn的CalibratedClassifierCV：

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

这在朴素贝叶斯的原始分数之上通过交叉验证拟合一个逻辑回归。得到的概率更接近真实的类别频率。

### 常见陷阱

1. **负特征值。** MultinomialNB要求非负特征。如果你有负值（如某些设置下的TF-IDF或标准化后的特征），请改用GaussianNB，或将特征平移为正。

2. **零方差特征。** GaussianNB除以方差。如果某个特征对于某个类别方差为零（所有值相同），则概率计算会崩溃。代码对所有方差添加了一个小的平滑项（1e-9）来防止这种情况。

3. **类别不平衡。** 如果99%的邮件不是垃圾邮件，先验P(非垃圾邮件)=0.99非常强，以至于压倒了似然证据。你可以手动设置类别先验或使用sklearn中的class_prior参数。

4. **特征缩放。** MultinomialNB不需要缩放（它基于计数工作）。GaussianNB也不需要缩放（它估计每个特征的统计量）。这是相对于逻辑回归和SVM的一个优势，后者对特征尺度敏感。

## 发布

本課(lesson)产出：
- `outputs/skill-naive-bayes-chooser.md` —— 选择合适朴素贝叶斯变体的决策技能
- `outputs/skill-naive-bayes-chooser.md` —— 从头实现MultinomialNB和GaussianNB，并与sklearn比较

### 当朴素贝叶斯失效时

当独立性假设导致排名错误（不仅仅是概率错误）时，朴素贝叶斯会失败。这发生在以下情况：

1. **强特征交互。** 如果类别取决于两个特征的组合但单独看任何一个都不起作用（类似异或(XOR)的模式），朴素贝叶斯将完全错过它。每个特征单独提供不了证据，而朴素贝叶斯无法非线性地组合它们。

2. **具有相反证据的高度相关特征。** 如果特征A说“垃圾邮件”，特征B说“非垃圾邮件”，但A和B完全相关（它们实际上总是一致），朴素贝叶斯会看到本不存在的相互矛盾的证据。

3. **非常大的训练集。** 有足够数据时，像逻辑回归这样的判别模型能学到真正的决策边界，并超越朴素贝叶斯。在小数据上有帮助的独立性假设现在成了模型进步的障碍。

在实践中，这些失败模式在文本分类中很少见。文本特征数量多、单独弱，且独立性假设的误差倾向于相互抵消。对于特征少且强相关的表格数据，请优先考虑逻辑回归或基于树的模型。

## 练习

1. **平滑实验。** 使用alpha值为0.01、0.1、1.0、10.0和100.0对文本数据训练MultinomialNB。绘制准确率与alpha的关系图。性能在何处达到峰值？为什么非常高的alpha会损害性能？

2. **特征独立性检验。** 取一个真实文本数据集。选择两个明显相关的词（“机器”和“学习”）。计算P(word1 | class) * P(word2 | class)并与P(word1 AND word2 | class)进行比较。独立性假设的错误程度如何？它是否影响分类准确率？

3. **伯努利实现。** 扩展代码，添加BernoulliNB类。将词袋表示转换为二进制（存在/不存在），并在文本数据上与MultinomialNB比较准确率。伯努利何时胜出？

4. **朴素贝叶斯与逻辑回归对比。** 两者都在文本数据上训练。从100个训练样本开始，增加到10,000个。绘制两者的准确率与训练集大小的关系图。逻辑回归在什么点上超越朴素贝叶斯？

5. **垃圾邮件过滤器。** 构建一个完整的垃圾邮件分类器：对原始电子邮件文本进行分词、构建词汇表、创建词袋特征、训练MultinomialNB、使用精确率和召回率进行评估（而不仅仅是准确率——为什么？）。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|----------------------|
|  朴素贝叶斯  |  "简单概率分类器"  |  一种应用贝叶斯定理的分类器，假设特征在给定类别的条件下相互独立  |
|  条件独立性  |  "特征之间互不影响"  |  P(A, B \ |  C) = P(A \ |  C) * P(B \ |  C) —— 知道C后，知道B不会给你关于A的新信息  |
|  拉普拉斯平滑  |  "加一平滑"  |  为每个特征添加一个小计数，以防止零概率主导预测  |
|  先验  |  "看到数据前你所相信的"  |  P(class) —— 在观察到任何特征之前每个类别的概率  |
|  似然  |  "数据拟合得如何"  |  P(features \ |  class) —— 如果类别已知，观察到这些特征的概率  |
|  后验  |  "看到数据后你所相信的"  |  P(class \ |  features) —— 观察到特征后类别的更新概率  |
|  生成模型  |  "模拟数据生成方式"  |  一种学习P(X \ |  Y)和P(Y)，然后使用贝叶斯定理得到P(Y \ |  X)的模型  |
|  判别模型  |  "模拟决策边界"  |  一种直接学习P(Y \ |  X)而不模拟X如何生成的模型  |
|  对数概率  |  "避免下溢"  |  使用log P代替P，以防止许多小数的乘积在浮点运算中变为零  |

## 延伸阅读

- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html) —— 所有三种变体的数学细节
- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html) —— 文本中Multinomial与Bernoulli的经典比较
- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html) —— 文本中朴素贝叶斯的改进
- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html) —— 证明朴素贝叶斯在较少数据下比逻辑回归收敛更快
