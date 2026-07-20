# 数值稳定性

> 浮点是一种有漏洞的抽象。在训练过程中它会咬你一口，而你却浑然不觉。

**类型：** 构建
**语言：** Python
**先决条件：** 阶段 1，第 01-04 课
**时长：** 约 120 分钟

## 学习目标

- 使用最大值减去技巧实现数值稳定的 softmax 和 log-sum-exp
- 识别浮点计算中的上溢、下溢和灾难性抵消
- 使用中心有限差分法将解析梯度与数值梯度进行验证
- 解释为什么在训练中 bfloat16 比 float16 更受欢迎，以及损失缩放如何防止梯度下溢

## 问题

你的模型训练了三小时，然后损失变成 NaN。你加了一条打印语句。在步骤 9,000 时 logits 还是正常的。到步骤 9,001 时它们变成了 `inf`。到步骤 9,002 时每个梯度都是 `nan`，训练彻底失败。

或者：你的模型训练完成，但准确率比论文声称的低 2%。你检查了所有东西。架构匹配。超参数匹配。数据匹配。问题在于论文使用了 float32，而你使用了没有正确缩放的 float16。32 位的累积舍入误差悄悄地吞噬了你的准确率。

或者：你从头实现了交叉熵损失。它在小 logits 上正常工作。当 logits 超过 100 时，它返回 `inf`。softmax 溢出了，因为 `exp(100)` 大于 float32 能表示的范围。每个机器学习框架都用两行代码的技巧处理了这个问题。而你却不知道这个技巧的存在。

数值稳定性并非理论问题。它决定了训练是成功还是悄然失败。你将会调试的每个严重的机器学习 bug，归根结底都与浮点有关。

## 核心概念

### IEEE 754：计算机如何存储实数

计算机按照 IEEE 754 标准将实数存储为浮点值。一个浮点数有三部分：符号位、指数和尾数（有效数字）。

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

尾数决定精度（多少位有效数字）。指数决定范围（数字可以有多大或多小）。

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32 提供大约 7 位十进制精度。这意味着它能区分 1.0000001 和 1.0000002，但无法区分 1.00000001 和 1.00000002。7 位之后，一切都成了舍入噪声。

float16 大约提供 3 位精度。它能表示的最大数是 65,504。对于机器学习而言，这个数小得惊人，因为 logits、梯度和激活值经常超过这个范围。

bfloat16 是 Google 针对 float16 范围问题的解决方案。它具有与 float32 相同的 8 位指数（相同范围，最大 3.4e38），但只有 7 位尾数（精度低于 float16）。对于训练神经网络，范围比精度更重要，因此 bfloat16 通常更胜一筹。

### 为什么 0.1 + 0.2 != 0.3

数字0.1无法在二进制浮点数中精确表示。在二进制中，它是一个循环小数：

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

Float32将其截断为23位尾数。存储的值约为0.100000001490116。类似地，0.2存储为约0.200000002980232。它们的和是0.300000004470348，而不是0.3。

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这对机器学习很重要，因为：

1. 像`if loss < threshold`这样的损失比较可能给出错误答案
2. 累加许多小值（数千步的梯度更新）会偏离真实和
3. 如果使用`if loss < threshold`比较浮点数，校验和与可重现性测试会失败

解决方法：永远不要使用`==`比较浮点数。使用`abs(a - b) < epsilon`或`math.isclose()`。

### 灾难性抵消

当减去两个几乎相等的浮点数时，有效数字抵消，只剩下舍入噪声提升到前导数字。

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

那是单次减法产生的19%相对误差。在机器学习中，每当出现以下情况时就会发生：

- 计算具有大均值的数据的方差：当E[x]大时使用`E[x^2] - E[x]^2`
- 减去几乎相等的对数概率
- 使用过小的epsilon计算有限差分梯度

解决方法：重新排列公式以避免减去大的近似相等的数。对于方差，使用Welford算法或先对数据中心化。对于对数概率，始终在对数空间中计算。

### 溢出和下溢

当结果太大而无法表示时发生溢出。当结果太小（比最小可表示正数更接近零）时发生下溢。

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

`exp()`函数是机器学习中溢出的主要来源：

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

`log()`函数则走向另一个方向：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

在机器学习中，`exp()`出现在softmax、sigmoid和概率计算中。`log()`出现在交叉熵、对数似然和KL散度中。组合`log(exp(x))`在缺乏正确技巧时是一个雷区。

### Log-Sum-Exp技巧

直接计算`log(sum(exp(x_i)))`在数值上很危险。如果任一个`x_i`很大，`exp(x_i)`会溢出。如果所有`x_i`都非常负，每个`exp(x_i)`都会下溢为零，且`log(0)`为`-inf`。

技巧：在指数化之前减去最大值。

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

为什么有效：减去`max(x)`后，最大指数为`exp(0) = 1`。不可能发生溢出。和中至少有一项为1，因此和至少为1，且`log(1) = 0`。不可能下溢到`-inf`。

证明：

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

设`c = max(x)`，溢出被消除。

这个技巧在机器学习中随处可见：
- Softmax归一化
- 交叉熵损失计算
- 序列模型中的对数概率求和
- 高斯混合模型
- 变分推断

### 为什么Softmax需要最大值减法技巧

Softmax将logits转换为概率：

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

没有该技巧，logits [100, 101, 102]会导致溢出：

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

使用技巧，减去max(x)=102：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

概率是相同的。计算是安全的。这不是优化，而是正确性的要求。

### NaN与Inf：检测与预防

`nan`（非数值）和`inf`（无穷大）会在计算中像病毒一样传播。梯度更新中的一个`nan`会使权重变成`nan`，进而使所有后续输出变成`nan`。训练在一步内就死了。

如何出现`inf`：
- `exp()` 的大正数
- 除以零：`exp()`
- `exp()` 在累加中溢出

如何出现`nan`：
- `0.0 / 0.0`
- `0.0 / 0.0`
- `0.0 / 0.0`
- `0.0 / 0.0` 的负数
- `0.0 / 0.0` 的负数
- 任何涉及已有`0.0 / 0.0`的算术

检测：

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

预防策略：

1. 将输入夹紧到`exp()`：`exp(clamp(x, -80, 80))`
2. 在分母中添加 epsilon：`exp()`
3. 在`exp()`内部添加 epsilon：`exp(clamp(x, -80, 80))`
4. 使用稳定实现（log-sum-exp、稳定 softmax）
5. 梯度裁剪以防止权重爆炸
6. 在调试过程中每次前向传播后检查`exp()`/`exp(clamp(x, -80, 80))`

### 数值梯度检查

分析梯度（来自反向传播）可能包含错误。数值梯度检查通过有限差分计算梯度来验证它们。

中心差分公式：

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

这是 O(h^2) 精度的，远优于仅 O(h) 的前向差分`(f(x+h) - f(x)) / h`。

选择 h：太大则近似错误。太小则灾难性抵消破坏答案。通常选择 `h = 1e-5` 到 `1e-7`。

检查：计算分析梯度和数值梯度之间的相对差异。

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验法则：
- relative_error < 1e-7：完美，梯度正确
- relative_error < 1e-5：可接受，可能正确
- relative_error > 1e-3：有问题
- relative_error > 1：梯度完全错误

在实现新层或损失函数时始终检查梯度。PyTorch 为此提供了`torch.autograd.gradcheck()`。

### 混合精度训练

现代GPU拥有专门的硬件（张量核心），其计算float16矩阵乘法的速度比float32快2-8倍。混合精度训练利用了这一特性：

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

纯float16训练的问题：梯度通常非常小（1e-8或更小）。Float16将任何低于约6e-8的值下溢为零。你的模型会停止学习，因为所有梯度更新都为零。

解决办法是损失缩放：

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

动态损失缩放自动调整缩放因子。从一个较大的值（65536）开始。如果梯度上溢到`inf`，则将其减半。如果N步内没有发生上溢，则将其加倍。

### bfloat16 vs float16：为什么bfloat16在训练中更胜一筹

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16精度更高（尾数位10位对7位），但范围有限（最大约65,504）。bfloat16精度较低，但范围与float32相同（最大约3.4e38）。

对于训练神经网络：

- 在训练尖峰期间，激活值和logits经常超过65,504。float16会溢出；bfloat16则能处理。
- float16需要损失缩放，而bfloat16通常不需要，因为它的范围覆盖了梯度幅值谱。
- bfloat16是float32的简单截断：丢弃尾数的低16位。转换很简单，且指数部分无损。

在推理中，由于数值范围有限且精度更重要，更倾向于使用float16。在训练中，由于范围更重要，更倾向于使用bfloat16。这就是TPU和现代NVIDIA GPU（A100、H100）原生支持bfloat16的原因。

### 梯度裁剪

梯度爆炸发生在梯度通过多层指数级增长时（常见于RNN、深度网络和Transformer）。单个大梯度可以在一步内破坏所有权重。

两种裁剪类型：

**按值裁剪：** 独立地限制每个梯度元素。

```
grad = clamp(grad, -max_val, max_val)
```

简单但可能改变梯度向量的方向。

**按范数裁剪：** 缩放整个梯度向量，使其范数不超过阈值。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

保持梯度的方向。这正是`torch.nn.utils.clip_grad_norm_()`所做的。这是标准选择。

典型值：对于Transformer为`max_norm=1.0`，对于强化学习为`max_norm=0.5`，对于简单网络为`max_norm=5.0`。

梯度裁剪并非权宜之计，而是一种安全机制。若没有它，单个异常批次就可能产生足以毁掉数周训练的梯度。

### 归一化层作为数值稳定器

批归一化、层归一化和RMS归一化通常被描述为有助于训练收敛的正则化器。它们同时也是数值稳定器。

没有归一化，激活值会在层间呈指数级增长或衰减：

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

归一化在每一层对激活值进行重新居中与缩放：

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

当所有激活值相同时，`epsilon`（通常设为1e-5）可防止分母为零。可学习参数`gamma`和`beta`让网络能够恢复所需的任何尺度。

这使得数值在整个网络中始终保持在安全范围内，既防止了前向传播中的上溢，也避免了反向传播中的梯度爆炸。

### 常见的机器学习数值错误

**错误：训练几轮后损失变为NaN。**
原因：logits值过大导致softmax溢出；或学习率过高导致权重发散。
修复：使用稳定的softmax（减去最大值），降低学习率，添加梯度裁剪。

**错误：损失卡在log(类别数)。**
原因：模型输出概率接近均匀分布，通常意味着梯度消失或模型完全没有学习。
修复：检查数据标签是否正确，验证损失函数，检查是否存在死亡ReLU。

**错误：验证精度比预期低1-3%。**
原因：混合精度训练时未正确缩放损失，梯度下溢导致小更新被静默清零。
修复：启用动态损失缩放，或改用bfloat16。

**错误：某些层的梯度范数为0.0。**
原因：死亡ReLU神经元（所有输入为负），或float16下溢。
修复：使用LeakyReLU或GELU，使用梯度缩放，检查权重初始化。

**错误：模型在一个GPU上表现正常，但在另一个GPU上结果不同。**
原因：浮点运算的累加顺序具有非确定性。不同硬件上GPU并行规约的求和顺序不同，而浮点加法不满足结合律。
修复：接受微小差异（1e-6），或设置`torch.use_deterministic_algorithms(True)`并接受速度惩罚。

**Bug：在损失计算中`exp()`返回`inf`。**
原因：原始logits传递给`exp()`时未使用减去最大值的技巧。
修复：使用内部实现了log-sum-exp的`torch.nn.functional.log_softmax()`。

**Bug：从float32切换到float16后训练发散。**
原因：float16无法表示低于6e-8的梯度大小或高于65,504的激活值。
修复：使用带损失缩放的混合精度（AMP），或改用bfloat16。

```figure
logsumexp-stability
```

## 动手构建

### 步骤1：展示浮点数精度限制

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### 步骤2：实现朴素softmax与稳定softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### 步骤3：实现稳定的log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### 步骤4：实现稳定的交叉熵

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### 步骤5：梯度检查

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 使用它

### 混合精度模拟

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 梯度裁剪

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf检测

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

参见`code/numerical.py`获取展示所有边缘情况的完整实现。

## 发布

本課(lesson)产出：
- `code/numerical.py`包含稳定softmax、log-sum-exp、交叉熵、梯度检查和混合精度模拟
- `code/numerical.py`用于诊断训练中的NaN/Inf和数值问题

这些稳定实现在阶段3构建训练循环以及阶段4实现注意力机制时会再次出现。

## 练习

1. **灾难性抵消。** 使用朴素公式`E[x^2] - E[x]^2`在float32下计算[1000000.0, 1000001.0, 1000002.0]的方差。然后使用Welford在线算法计算。比较误差与真实方差（0.6667）。

2. **精度探索。** 找到最小的正float32值`x`，使得在Python中`1.0 + x == 1.0`。这就是机器精度。验证其是否与`numpy.finfo(numpy.float32).eps`一致。

3. **Log-sum-exp 边缘情况。** 用以下情况测试你的 `logsumexp_stable` 函数：(a) 所有值相等，(b) 一个值远大于其他值，(c) 所有值非常负（-1000）。验证其在朴素版本失败时能得到正确结果。

4. **梯度检查一个神经网络层。** 实现一个单线性层 `y = Wx + b` 及其解析反向传播。用 `numerical_gradient` 验证一个 3x2 权重矩阵的正确性。

5. **损失缩放实验。** 模拟使用 float16 训练：在范围 [1e-9, 1e-3] 内创建随机梯度，转换为 float16，测量变为零的比例。然后应用损失缩放（乘以 1024），转换为 float16，再缩放回来，再次测量零的比例。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|----------------------|
|  IEEE 754  |  "浮点标准"  |  定义了二进制浮点格式、舍入规则和特殊值（inf, nan）的国际标准。每个现代 CPU 和 GPU 都实现它。 |
|  机器精度  |  "精度极限"  |  在给定浮点格式下，使得 1.0 + e != 1.0 的最小值 e。对于 float32，约为 1.19e-7。 |
|  灾难性抵消  |  "减法导致的精度损失"  |  当减去两个接近的浮点数时，有效数字抵消，舍入噪声主导结果。 |
|  上溢  |  "数字太大"  |  结果超过最大可表示值变为 inf。exp(89) 会使 float32 上溢。 |
|  下溢  |  "数字太小"  |  结果比最小正可表示数更接近零，变为 0.0。exp(-104) 会使 float32 下溢。 |
|  Log-sum-exp 技巧  |  "先减最大值"  |  通过提取 exp(max(x)) 来计算 log(sum(exp(x)))，防止上溢和下溢。用于 softmax、交叉熵和对数概率计算。 |
|  稳定 Softmax  |  "不会爆炸的 Softmax"  |  在指数化之前减去 max(logits)。数值结果相同，不可能发生上溢。 |
|  梯度检查  |  "验证你的反向传播"  |  通过有限差分将反向传播的解析梯度与数值梯度进行比较，以发现实现错误。 |
|  混合精度  |  "Float16 前向，float32 反向"  |  对速度关键的操作使用较低精度浮点数，对数值敏感的操作使用较高精度浮点数。典型加速为 2-3 倍。 |
|  损失缩放  |  "防止梯度下溢"  |  在反向传播前将损失乘以一个大的常数，使梯度保持在 float16 的可表示范围内，然后在权重更新前除以相同的常数。 |
|  bfloat16  |  "脑浮点"  |  谷歌的 16 位格式，具有 8 位指数（与 float32 范围相同）和 7 位尾数（精度低于 float16）。训练中更受欢迎。 |
|  梯度裁剪  |  "限制梯度范数"  |  缩放梯度向量，使其范数不超过阈值。防止梯度爆炸破坏权重。 |
|  NaN  |  "Not a Number"  |  意为“非数字”，是未定义操作（如0/0、inf-inf、sqrt(-1)）产生的特殊浮点数值。会传播到所有后续算术运算中。  |
|  Inf  |  "Infinity"  |  意为“无穷大”，是溢出或除零操作产生的特殊浮点数值。组合运算可能产生NaN（如inf - inf、inf * 0）。  |
|  数值梯度  |  "暴力导数"  |  通过计算f(x+h)和f(x-h)并除以2h来近似导数。速度慢，但验证可靠。  |

## 延伸阅读

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- 权威参考，内容密集但完整
- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- NVIDIA介绍loss scaling用于float16训练的论文
- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- PyTorch混合精度的实用指南
- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- Google为何为TPU选择此格式
- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- 减少浮点求和舍入误差的算法
