# 采样方法

> 采样是AI探索可能性空间的方式。

**类型:** 构建
**语言:** Python
**前置条件:** 阶段1，第06-07课（概率，贝叶斯定理）
**时间:** ~120分钟

## 学习目标

- 仅使用均匀随机数从头实现逆CDF、拒绝采样和重要性采样
- 为语言模型token生成构建温度、top-k和top-p（核）采样
- 解释重参数化技巧及其如何在VAE中通过采样实现反向传播
- 运行Metropolis-Hastings MCMC从未归一化的目标分布中采样

## 问题

语言模型处理完你的提示后，会生成一个包含50,000个逻辑值的向量。每个值对应词汇表中的一个token。现在它必须选择一个。如何选择？

如果它总是选择概率最高的token，那么每个响应都相同。确定的。无聊的。如果它均匀随机选择，则输出是乱码。答案存在于这两个极端之间，而这个“之间”由采样控制。

采样不仅限于文本生成。强化学习通过采样轨迹来估计策略梯度。VAE通过从学习到的分布中采样并透过随机性反向传播来学习潜在表示。扩散模型通过采样噪声并迭代去噪来生成图像。蒙特卡洛方法估计没有闭式解的积分。MCMC算法探索无法枚举的高维后验分布。

每个生成式AI系统都是一个采样系统。采样策略决定了输出的质量、多样性和可控性。本课从头构建每一种主要的采样方法，从均匀随机数开始，到驱动现代LLM和生成模型的技术结束。

## 核心概念

### 为什么采样至关重要

采样在AI和机器学习中扮演四个基本角色：

**生成。** 语言模型、扩散模型和GAN都通过采样产生输出。采样算法直接控制创造力、连贯性和多样性。温度、top-k和核采样是工程师日常调整的旋钮。

**训练。** 随机梯度下降采样小批量。Dropout采样要停用的神经元。数据增强采样随机变换。重要性采样重新加权样本以减少强化学习（PPO、TRPO）中的梯度方差。

**估计。** 机器学习中的许多量没有闭式解。数据分布上的期望损失、基于能量的模型的配分函数、贝叶斯推断中的证据。蒙特卡洛估计通过对样本取平均来近似所有这些量。

**探索。** MCMC算法探索贝叶斯推断中的后验分布。进化策略采样参数扰动。汤普森采样在赌博机中平衡探索与利用。

核心挑战：你只能直接从简单分布（均匀、正态）中采样。对于其他所有情况，你需要一种方法将简单样本转换为来自目标分布的样本。

### 均匀随机采样

每种采样方法都从这里开始。一个均匀随机数生成器产生[0, 1)内的值，其中每个等长子区间具有相等的概率。

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

为了从包含n个项的离散集合中均匀采样，生成U并返回floor(n * U)。为了从连续范围[a, b]中采样，计算a + (b - a) * U。

关键洞察：一个均匀随机数恰好包含足够随机性，可以产生来自任何分布的一个样本。诀窍在于找到正确的变换。

### 逆CDF方法（逆变换采样）

累积分布函数(CDF)将值映射为概率：

```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

逆CDF将概率映射回值。若U ~ Uniform(0, 1)，则X = F_inverse(U)服从目标分布。

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**指数分布示例：**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

当你能写出F_inverse的闭式表达式时，这种方法完美工作。对于正态分布，没有闭式逆CDF，所以我们使用其他方法（Box-Muller，或数值近似）。

**离散版本：** 对于离散分布，将CDF构建为累积和，生成U，然后找到累积和首次超过U的索引。这就是第06课中`sample_categorical`的工作原理。

### 拒绝采样

当你无法求逆CDF但可以评估目标PDF（最多差一个常数）时，拒绝采样可行。

```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```

边界M越紧，接受率越高。在低维（1-3）中，拒绝采样效果良好。在高维中，接受率指数下降，因为大部分提议体积被拒绝。这就是拒绝采样的维度灾难。

**示例：从截断正态分布中采样。** 在截断区间内使用均匀提议。包络M是该区间内正态PDF的最大值。

**示例：从半圆中采样。** 在边界矩形内均匀提议。如果点落在半圆内则接受。这就是蒙特卡洛计算pi的方式：接受率等于面积比pi/4。

### 重要性采样

有时你并不需要从目标分布 p(x) 中采样。你需要估计 p(x) 下的期望，而你已经从另一个分布 q(x) 中得到了样本。

```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

这在强化学习中至关重要。在 PPO（近端策略优化）中，你在旧策略 pi_old 下收集轨迹，但想要优化新策略 pi_new。重要性权重是 pi_new(a|s) / pi_old(a|s)。PPO 对这些权重进行裁剪，以防止新策略偏离旧策略太远。

重要性采样估计器的方差取决于 q 与 p 的相似程度。如果 q 与 p 差异很大，少数样本会获得巨大权重并主导估计。自归一化重要性采样通过除以权重的和来减轻这个问题：

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### 蒙特卡洛估计

蒙特卡洛估计通过对随机样本取平均来近似积分。大数定律保证了收敛性。

```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

误差率与维度无关。这就是为什么在高维度下（基于网格的积分不可行），蒙特卡洛方法占据主导地位的原因。

**估计 π：**

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**估计期望：**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### 马尔可夫链蒙特卡洛（MCMC）：Metropolis-Hastings

MCMC 构建一个马尔可夫链，其平稳分布是目标分布 p(x)。经过足够多的步骤后，链中的样本（近似）是来自 p(x) 的样本。

```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

对于对称提议分布（q(x'|x) = q(x|x')），比值简化为 p(x')/p(x)。这就是原始的 Metropolis 算法。

**原理。** 接受规则确保了细致平衡：处于 x 并移动到 x' 的概率等于处于 x' 并移动到 x 的概率。细致平衡意味着 p(x) 是链的平稳分布。

**实践考虑：**
- 预热（Burn-in）：在链达到平衡之前丢弃早期样本
- 稀疏化（Thinning）：每隔 k 个样本保留一个以减少自相关
- 提议尺度（Proposal scale）：太小则链移动缓慢（接受率高，探索慢）；太大则大多数提议被拒绝（接受率低，卡住不动）
- 高维高斯提议的最优接受率约为 0.234

### 吉布斯采样

吉布斯采样是 MCMC 在多变量分布下的一个特例。它不是同时对所有维度提出移动，而是每次从一个变量的条件分布中更新该变量。

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

吉布斯采样要求你能从每个条件分布 p(x_i | x_{-i}) 中采样。这对许多模型来说很直接：
- 贝叶斯网络：条件分布由图结构决定
- 高斯混合模型：条件分布是高斯分布
- 伊辛模型：每个自旋的条件分布仅取决于其邻居

接受率始终为 1（每个提议都被接受），因为从精确条件分布中采样自动满足细致平衡。

**局限性。** 当变量高度相关时，吉布斯采样混合缓慢，因为一次只更新一个变量无法在分布中做出大的对角移动。

### 温度采样（用于大语言模型）

语言模型对词汇表中的每个 token 输出 logits z_1, ..., z_V。Softmax 将其转换为概率。温度在 softmax 之前重新缩放 logits：

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**原理。** 将 logits 除以 T < 1 会放大 logits 之间的差异。如果 z_1 = 2 且 z_2 = 1，除以 T = 0.5 得到 z_1/T = 4 和 z_2/T = 2，拉大了差距。经过 softmax 后，最高 logit 的 token 获得了更大的份额。

**实践中：**
- T = 0.0：贪心解码，最适合事实问答
- T = 0.3-0.7：略有创造性，适合代码生成
- T = 0.7-1.0：平衡，适合一般对话
- T = 1.0-1.5：创意写作、头脑风暴
- T > 1.5：越来越随机，很少有用

温度不会改变哪些 token 是可能的。它改变了分配给每个 token 的概率质量。

### Top-k 采样

Top-k 采样将候选集限制为概率最高的 k 个 token，然后重新归一化并从该限制集中采样。

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

Top-k 防止模型选择词汇分布长尾中极不可能的 token（拼写错误、无意义内容）。问题是：k 是固定的，与上下文无关。当模型自信时（一个 token 有 95% 的概率），k = 40 仍然允许 39 个替代选项。当模型不确定时（概率分布在 1000 个 token 上），k = 40 会截断合理的选项。

### Top-p（核）采样

Top-p采样动态调整候选集大小。它不是保持固定数量的token，而是保持累积概率超过p的最小token集合。

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

当模型置信度高时，核采样只保留少数token（可能2-3个）。当模型不确定时，它会保留很多（可能200个）。这种自适应行为使得核采样通常比top-k生成更好的文本。

**常见组合：**
- 温度0.7 + top-p 0.9：良好的通用设置
- 温度0.0（贪婪）：最适合确定性任务
- 温度1.0 + top-k 50：Fan等人（2018）原始论文设置

Top-k和top-p可以结合使用。先应用top-k，然后在剩余集合上应用top-p。

### 重参数化技巧（用于VAE）

变分自编码器（VAE）通过将输入编码为潜在空间中的分布、从该分布采样、然后将样本解码回来进行学习。问题在于：无法通过采样操作进行反向传播。

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

重参数化技巧将随机性与参数分离：

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

这是因为N(mu, sigma^2)与mu + sigma * N(0, 1)具有相同的分布。关键洞察：将随机性移到一个无参数来源（epsilon），然后将样本表示为参数的可微变换。

**在VAE训练循环中：**
1. 编码器为每个输入输出mu和log(sigma^2)
2. 采样epsilon ~ N(0, 1)
3. 计算z = mu + sigma * epsilon
4. 解码z以重建输入
5. 通过步骤4、3、2、1反向传播（可能因为步骤3是可微的）

没有重参数化技巧，VAE就无法用标准反向传播进行训练。这一单一洞察使得VAE变得实用。

### Gumbel-Softmax（可微类别采样）

重参数化技巧适用于连续分布（高斯）。对于离散类别分布，我们需要另一种方法。Gumbel-Softmax提供了类别采样的可微近似。

**Gumbel-Max技巧（不可微）：**

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax（可微近似）：**

```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax产生离散样本的连续松弛。输出是一个概率向量（软独热）而不是硬独热。梯度通过softmax流动。在训练的前向传播中，可以使用“直通”估计器：前向传播使用硬argmax，但反向传播使用软Gumbel-Softmax梯度。

**应用：**
- VAE中的离散潜在变量
- 神经架构搜索（选择离散操作）
- 硬注意力机制
- 具有离散动作的强化学习

### 分层采样

标准蒙特卡洛采样可能偶然在样本空间中留下间隙。分层采样通过将空间划分为层并从每一层采样来强制均匀覆盖。

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

与标准蒙特卡洛相比，分层采样总是具有更低或相等的方差：

```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**应用：**
- 数值积分（拟蒙特卡洛）
- 训练数据划分（确保每个折中的类别平衡）
- 带分层的重要性采样（结合两种技术）
- NeRF（神经辐射场）沿相机射线使用分层采样

### 与扩散模型的联系

扩散模型通过采样过程生成图像。前向过程在T步中向图像添加高斯噪声，直到成为纯噪声。反向过程学习去噪，逐步恢复原始图像。

```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

与本课方法的联系：
- 每个去噪步骤使用重参数化技巧（采样噪声，应用确定性变换）
- 噪声调度{alpha_t}控制一种形式的温度退火
- 训练使用蒙特卡洛估计来近似ELBO（证据下界）
- 扩散模型中的祖先采样是一个马尔可夫链（每一步仅依赖于当前状态）

整个图像生成过程是迭代采样：从噪声开始，每一步都基于学到的去噪模型采样一个噪声稍小的版本。

```figure
monte-carlo-pi
```

## 动手构建

### 步骤1：均匀和逆CDF采样

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

生成10,000个指数分布样本，并验证其均值为1/lambda。

### 步骤2：拒绝采样(Rejection sampling)

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

使用拒绝采样从截断正态分布中进行抽样。通过直方图展示样本形状以验证结果。

### 步骤3：重要性采样(Importance sampling)

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

使用均匀提议分布估计正态分布下E[X^2]。与已知结果（mu^2 + sigma^2）进行比较。

### 步骤4：蒙特卡洛(Monte Carlo)估计π

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### 步骤5：Metropolis-Hastings马尔可夫链蒙特卡洛(MCMC)

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

从双峰分布（两个高斯分布的混合）中采样。可视化链的轨迹。

### 步骤6：吉布斯采样(Gibbs sampling)

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### 步骤7：温度采样(Temperature sampling)

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

展示温度如何改变一组词元logits的输出分布。

### 步骤8：Top-k和Top-p采样

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### 步骤9：重参数化技巧(Reparameterization trick)

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

证明梯度可以通过重参数化样本传播，而无法通过直接采样传播。

### 步骤10：Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

展示降低温度如何使输出趋近于one-hot向量。

所有可视化的完整实现位于 `code/sampling.py` 中。

## 使用它

使用NumPy和SciPy的生产版本：

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

对于大规模MCMC，请使用专用库：
- PyMC：使用NUTS（自适应HMC）的完整贝叶斯建模
- emcee：集成MCMC采样器
- NumPyro/JAX：GPU加速的MCMC

你是从零开始构建这些的。现在你知道了库调用背后的原理。

## 练习

1. 实现柯西分布的逆CDF采样。CDF为F(x) = 0.5 + arctan(x)/pi。生成10,000个样本并绘制直方图与真实PDF对比。注意重尾现象（远离中心的极值）。

2. 使用拒绝采样从Beta(2, 5)分布中生成样本，提议分布为Uniform(0, 1)。绘制接受样本与真实Beta PDF的对比。理论接受率是多少？

3. 使用蒙特卡洛(Monte Carlo)方法估计sin(x)从0到π的积分，分别使用1,000、10,000和100,000个样本。比较每个样本量下的误差。验证误差以O(1/sqrt(N))的规模变化。

4. 实现Metropolis-Hastings对二维分布p(x, y) ∝ exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2)进行采样。绘制样本和链轨迹。尝试不同的提议标准差。

5. 构建一个完整的文本生成演示：给定包含10个词的词表及其logits，使用(a)贪心、(b)温度=0.7、(c) top-k=3、(d) top-p=0.9生成20个词元的序列。比较5次运行中输出的多样性。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|----------------------|
| 抽 样  |  "随机取值"  |  根据概率分布生成数值。所有生成式AI背后的机制 |
| 均匀分布  |  "所有值等可能"  |  [a, b]内的每个值具有相等的概率密度1/(b-a)。所有采样方法的起点 |
| 逆CDF  |  "概率变换"  |  F_inverse(U)将均匀样本转换为来自任何已知CDF分布的样本。精确且高效 |
| 拒绝采样  |  "提议并接受/拒绝"  |  从简单提议分布生成，以与目标/提议比例成正比的概率接受。精确但浪费样本 |
| 重要性采样  |  "重新加权样本"  |  使用来自q(x)的样本通过加权每个样本乘以p(x)/q(x)来估计p(x)下的期望。是强化学习中PPO的核心 |
| 蒙特卡洛  |  "随机样本平均"  |  将积分近似为样本均值。误差O(1/sqrt(N))，与维度无关 |
|  MCMC  |  "收敛的随机游走"  |  构造一个马尔可夫链，其平稳分布为目标分布。Metropolis-Hastings是基础算法  |
|  Metropolis-Hastings  |  "接受上坡，有时下坡"  |  提议移动，根据密度比接受。详细平衡确保收敛到目标分布  |
|  吉布斯采样（Gibbs sampling）  |  "一次一个变量"  |  每次从一个变量的条件分布中更新它，固定其他变量。100%接受率  |
|  温度  |  "置信度旋钮"  |  在softmax之前用T除以logits。T<1使分布更尖锐（更自信），T>1使分布更平坦（更多样化）  |
|  Top-k采样  |  "保留k个最好的"  |  将除了概率最高的k个token之外的所有token置零，重新归一化，然后采样。固定候选集大小  |
|  核采样（top-p）  |  "保留那些可能的"  |  保留累积概率超过p的最小token集合。自适应候选集大小  |
|  重参数化技巧  |  "将随机性移出"  |  写成 z = mu + sigma * epsilon，其中 epsilon ~ N(0,1)。使采样可微。对VAE训练至关重要  |
|  Gumbel-Softmax  |  "软分类采样"  |  使用Gumbel噪声和带温度的softmax对分类采样进行可微近似  |
|  分层采样  |  "强制覆盖"  |  将样本空间划分为层，从每层中采样。方差总是低于朴素蒙特卡洛  |
|  预烧期  |  "预热期"  |  在链到达其平稳分布之前丢弃的初始MCMC样本  |
|  详细平衡  |  "可逆性条件"  |  p(x) * T(x->y) = p(y) * T(y->x)。p成为马尔可夫链平稳分布的充分条件  |
|  扩散采样  |  "迭代去噪"  |  从噪声开始，应用学到的去噪步骤生成数据。每一步都是一个条件采样操作  |

## 延伸阅读

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - MCMC基础详细教程
- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - 原始Gumbel-Softmax论文
- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - 核采样（top-p）论文
- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - 介绍重参数化技巧的VAE论文
- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - DDPM将采样与图像生成联系起来
