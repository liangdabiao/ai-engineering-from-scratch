# JAX简介

> PyTorch 会就地修改张量。TensorFlow 构建计算图。JAX 编译纯函数。最后这一点改变了你对深度学习的思考方式。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段03 第01-10课，基础NumPy
**时长：** ~90分钟

## 学习目标

- 使用JAX的函数式API（jax.numpy、jax.grad、jax.jit、jax.vmap）编写纯函数神经网络代码
- 解释PyTorch的即时修改与JAX的函数式编译模型之间的关键设计差异
- 应用jit编译和vmap向量化来加速训练循环，与朴素Python相比
- 在JAX中训练一个简单网络，并将其显式状态管理与PyTorch的面向对象方法进行对比

## 问题

你知道如何在PyTorch中构建神经网络。你定义一个`nn.Module`，调用`.backward()`，让优化器前进。这很有效。数百万人都在使用它。

但PyTorch在其基因中内置了一个限制：它在Python中逐个地即时追踪操作。每个`tensor + tensor`都是一个独立的内核启动。每个训练步骤都会重新解释相同的Python代码。这种方法在需要跨2048个TPU训练一个5400亿参数模型之前都工作良好——然后开销会置你于死地。

Google DeepMind在JAX上训练了Gemini。Anthropic在JAX上训练了Claude。这些可不是小规模操作——它们是地球上最大的神经网络训练任务。他们选择JAX，是因为它将你的训练循环视为一个可编译的程序，而非一系列Python调用。

JAX是NumPy加上三个超能力：自动微分、JIT编译到XLA、以及自动向量化。你编写一个处理单个样本的函数。JAX给你一个处理批量数据、计算梯度、编译为机器码并在多设备上运行的函数。这一切都无需修改原始函数。

## 核心概念

### JAX哲学

JAX是一个函数式框架。没有类，没有可变状态，没有`.backward()`方法。取而代之的是：

|  PyTorch  |  JAX  |
|---------|-----|
|  `nn.Module` 带状态的类  |  纯函数：`f(params, x) -> y`  |
|  `loss.backward()`  |  `jax.grad(loss_fn)(params, x, y)`  |
|  即时执行  |  通过XLA进行JIT编译  |
|  `for x in batch:` 手动循环  |  `jax.vmap(f)` 自动向量化  |
|  `DataParallel` / `FSDP`  |  `jax.pmap(f)` 自动并行化  |
|  可变的`model.parameters()`  |  不可变的数组pytree  |

这并非风格偏好，而是编译器约束。JIT编译要求纯函数——相同的输入始终产生相同的输出，无副作用。正是这个限制让100倍加速成为可能。

### jax.numpy：熟悉的界面

JAX在加速器上重新实现了NumPy API：

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

相同的函数名。相同的广播规则。相同的切片语义。但数组驻留在GPU/TPU上，并且每个操作都可被编译器追踪。

一个关键区别：JAX数组是不可变的。没有`a[0] = 5`。取而代之的是：`a = a.at[0].set(5)`。这在一周内会感觉别扭，然后你会豁然开朗——不可变性正是使得`grad`、`jit`和`vmap`等变换可组合的原因。

### jax.grad：函数式自动微分

PyTorch将梯度附加到张量（`.grad`）。JAX将梯度附加到函数。

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad`接收一个函数并返回一个计算梯度的新函数。没有`.backward()`调用。没有存储在张量上的计算图。梯度只是另一个你可以调用、组合或JIT编译的函数。

这种方式可以任意组合：

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

二阶导数。三阶导数。雅可比矩阵。海森矩阵。全部通过组合`grad`实现。PyTorch也可以做到（`torch.autograd.functional.hessian`），但那是后加的功能。而在JAX中，这是基石。

约束：`grad`只作用于纯函数。内部不能有打印语句（它们会在追踪期间运行，而不是执行期间）。不能修改外部状态。不能在没有显式密钥管理的情况下生成随机数。

### jit：编译到XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

首次调用时，JAX会追踪该函数——记录下发生了哪些操作，但不执行它们。然后将追踪结果交给XLA（加速线性代数），这是Google面向TPU和GPU的编译器。XLA会融合操作、消除冗余内存拷贝，并生成优化的机器码。

后续调用完全跳过Python。编译后的代码以C++的速度在加速器上运行。

JIT何时有帮助：
- 训练步骤（相同计算重复数千次）
- 推理（相同模型，不同输入）
- 任何被调用超过一次且输入形状相似的函数

JIT何时有损害：
- 函数内部含有依赖于值的Python控制流（如`if x > 0`，其中x是跟踪数组）
- 一次性计算（编译开销超过运行时间）
- 调试（跟踪隐藏了实际执行）

控制流限制是真实存在的。`jax.lax.cond`取代了`if/else`。`jax.lax.scan`取代了`for`循环。这些不是可选的——它们是编译的代价。

### vmap：自动向量化

你编写一个处理单个样本的函数：

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap`将其提升以处理一个批次：

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)`的意思是：不对`params`进行批处理（共享的），对`x`的第0轴进行批处理。无需手动编写`for`循环。无需重塑。无需批次维度线程化。JAX会自动找出批次维度并对整个计算进行向量化。

这不是语法糖。`vmap`生成融合向量化代码，运行速度比Python循环快10-100倍。并且它可以与`jit`和`grad`组合使用：

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

每个样本的梯度。一行代码。这在PyTorch中几乎不可能实现，除非使用技巧。

### pmap：跨设备的数据并行

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap`将所有可用设备（GPU/TPU）上的函数复制并分割批次。在函数内部，`jax.lax.pmean`和`jax.lax.psum`跨设备同步梯度。

Google使用`pmap`（及其后继者`shard_map`）在数千个TPU v5e芯片上训练Gemini。编程模型：编写单设备版本，用`pmap`包装，完成。

### Pytrees：通用数据结构

JAX操作于“pytrees”——列表、元组、字典和数组的嵌套组合。你的模型参数就是一个pytree：

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

每个JAX转换——`grad`、`jit`、`vmap`——都知道如何遍历pytrees。`jax.tree.map(f, tree)`对每个叶子应用`f`。这就是优化器一次性更新所有参数的方式：

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

没有`.parameters()`方法。没有参数注册。树结构就是模型。

### 函数式 vs 面向对象

PyTorch在对象内部存储状态：

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX使用带有显式状态的纯函数：

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

参数被传入。没有存储。没有修改。这使得每个函数都可测试、可组合、可编译。这也意味着你需要自己管理参数——或者使用像Flax或Equinox这样的库。

### JAX生态系统

JAX提供原语。库提供人体工程学：

|  库  |  作用  |  风格  |
|---------|------|-------|
|  **Flax** (Google)  |  神经网络层  |  带有显式状态的`nn.Module`  |
|  **Equinox** (Patrick Kidger)  |  神经网络层  |  基于Pytree，Python风格  |
|  **Optax** (DeepMind)  |  优化器 + 学习率调度  |  可组合的梯度变换  |
|  **Orbax** (Google)  |  检查点  |  保存/恢复pytrees  |
|  **CLU**（Google）|||  指标 + 日志  |  训练循环工具 |  |

Optax 是标准的优化器库。它将梯度变换（Adam、SGD、裁剪）与参数更新分离，使得组合变得非常简单：

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### 何时使用 JAX 与 PyTorch

|  因素  |  JAX  |  PyTorch  |
|--------|-----|---------|
|  TPU 支持  |  一流（Google 同时构建了两者）|||  社区维护（torch_xla） |  |
|  GPU 支持  |  良好（通过 XLA 的 CUDA）|||  最佳（原生 CUDA） |  |
|  调试  |  困难（追踪 + 编译）|||  简单（即时执行，逐行进行） |  |
|  生态系统  |  研究导向（Flax、Equinox）|||  庞大（HuggingFace、torchvision 等） |  |
|  招聘  |  小众（Google/DeepMind/Anthropic）|||  主流（无处不在） |  |
|  大规模训练  |  优越（XLA、pmap、mesh）|||  良好（FSDP、DeepSpeed） |  |
|  原型开发速度  |  较慢（函数式开销）|||  较快（可变操作，立即执行） |  |
|  生产推理  |  TensorFlow Serving、Vertex AI  |  TorchServe、Triton、ONNX  |
|  使用者  |  DeepMind（Gemini）、Anthropic（Claude）|||  Meta（Llama）、OpenAI（GPT）、Stability AI  |  |

说实话：除非你有特定的理由使用 JAX，否则使用 PyTorch。这些理由是——需要访问 TPU、需要逐样本梯度、大规模多设备训练，或者在 Google/DeepMind/Anthropic 工作。

### JAX 中的随机数

JAX 没有全局随机状态。每次随机操作都需要一个明确的 PRNG 密钥：

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

一开始这很烦人。但它保证了跨设备和编译的可重复性——这是 PyTorch 的 `torch.manual_seed` 在多 GPU 设置下无法保证的特性。

```figure
batchnorm-effect
```

## 动手构建

### 步骤 1：设置和数据

我们将使用 JAX 和 Optax 在 MNIST 上训练一个 3 层 MLP。784 个输入，两个隐藏层分别有 256 和 128 个神经元，10 个输出类别。

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### 步骤 2：初始化参数

没有类。只是一个返回 pytree 的函数：

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

He 初始化，手动完成。从一个种子分裂出三个 PRNG 密钥。每个权重都是嵌套字典中的一个不可变数组。

### 步骤 3：前向传播

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

纯函数。参数输入，预测输出。没有 `self`，没有存储状态。`loss_fn` 从头计算交叉熵——softmax、log、负均值。

### 步骤 4：JIT 编译的训练步骤

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` 在一次传递中返回损失值和梯度。`@jax.jit` 装饰器将两个函数编译为 XLA。第一次调用后，每个训练步骤都不再触及 Python。

### 步骤 5：训练循环

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 个 epoch。测试准确率约 97%。第一个 epoch 很慢（JIT 编译）。第 2 到第 10 个 epoch 很快。

注意缺少了什么：没有 `.zero_grad()`，没有 `.backward()`，没有 `.step()`。整个更新是一个组合的函数调用。梯度被计算、由 Adam 变换并应用于参数——所有这些都在 `train_step` 内部完成。

## 使用它

### Flax：Google 的标准

Flax 是最常见的 JAX 神经网络库。它重新引入了`nn.Module`，但带有显式的状态管理：

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

与 PyTorch 结构相同，但`params`与模型分离。`model.init()`创建参数。`model.apply(params, x)`执行前向传播。模型对象没有状态。

### Equinox：Python 风格的替代方案

Equinox（作者 Patrick Kidger）将模型表示为 pytree：

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

模型本身就是一个 pytree。无需`.apply()`。参数只是模型的叶子节点。这更符合 JAX 的思维方式。

### Optax：可组合的优化器

Optax 将梯度转换与更新解耦：

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

梯度裁剪、学习率预热、权重衰减——所有操作都组合成一个转换链。每个转换都会看到梯度、修改它们并将它们传递给下一个。没有单一的优化器类。

## 发布

**安装：**

```bash
pip install jax jaxlib optax flax
```

对于 GPU 支持：

```bash
pip install jax[cuda12]
```

对于 TPU（Google Cloud）：

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**性能注意事项：**

- 首次 JIT 调用较慢（编译）。在基准测试前先预热。
- 避免在 JIT 内部对 JAX 数组使用 Python 循环。使用`jax.lax.scan`或`jax.lax.fori_loop`。
- `jax.lax.scan`在 JIT 内部有效。常规的`jax.lax.fori_loop`无效。
- 使用`jax.lax.scan`或 TensorBoard 进行分析。XLA 编译可能会隐藏瓶颈。
- JAX 默认预分配 75% 的 GPU 内存。设置`jax.lax.scan`可禁用。

**检查点：**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**本课程产出：**
- `outputs/prompt-jax-optimizer.md` —— 选择正确 JAX 优化器配置的提示
- `outputs/prompt-jax-optimizer.md` —— 涵盖 JAX 函数式模式的技能

## 练习

1. 向 MLP 添加 Dropout。在 JAX 中，Dropout 需要 PRNG 密钥——在前向传播中传递密钥，并为每个 Dropout 层拆分密钥。比较有无 Dropout 时的测试准确率。

2. 使用`jax.vmap`计算一批 32 张 MNIST 图像的逐样本梯度。计算每个样本的梯度范数。哪些样本的梯度最大？为什么？

3. 将手动前向函数替换为通用的`mlp_forward(params, x)`，使其适用于任意数量的层。使用`jax.tree.leaves`自动确定深度。

4. 对有无`@jax.jit`的训练步骤进行基准测试。分别计时 100 步。在您的硬件上加速比有多大？首次调用的编译开销是多少？

5. 通过组合`optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`实现梯度裁剪。分别训练有无裁剪的情况。绘制训练过程中的梯度范数以观察效果。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|----------------------|
|  XLA  |  "让 JAX 变快的东西"  |  加速线性代数——一种编译器，它融合运算并从计算图生成优化的 GPU/TPU 内核  |
|  JIT  |  "即时编译"  |  JAX 在首次调用时追踪函数，编译为 XLA，然后在后续调用中运行编译后的版本  |
|  纯函数  |  "无副作用"  |  输出仅依赖于输入的函数——没有全局状态、没有突变、没有无显式键的随机性  |
|  vmap  |  "自动批处理"  |  将处理单个样本的函数转换为处理批处理的函数，无需重写  |
|  pmap  |  "自动并行"  |  在多设备上复制函数并分割输入批次  |
|  Pytree  |  "嵌套的数组字典"  |  JAX 可以遍历和转换的任何嵌套结构，包括列表、元组、字典和数组  |
|  追踪  |  "记录计算过程"  |  JAX 使用抽象值执行函数以构建计算图，而不计算实际结果  |
|  函数式自动微分  |  "函数的梯度"  |  通过变换函数计算导数，而不是将梯度存储附加到张量上  |
|  Optax  |  "JAX 的优化器库"  |  一个可组合的梯度变换库——Adam、SGD、裁剪、调度——它们可以链接在一起  |
|  Flax  |  "JAX的nn.Module"  |  Google公司的JAX神经网络库，在保持状态显式的同时添加层抽象  |

## 延伸阅读

- JAX文档：https://jax.readthedocs.io/ -- 官方文档，包含关于grad、jit和vmap的出色教程
- "JAX: composable transformations of Python+NumPy programs" (Bradbury et al., 2018) -- 解释设计哲学的原始论文
- Flax文档：https://jax.readthedocs.io/ -- Google公司的JAX神经网络库
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations" (2021) -- Flax的Pythonic替代方案
- DeepMind, "Optax: composable gradient transformation and optimisation" -- 标准优化器库
- "You Don't Know JAX" (Colin Raffel, 2020) -- 关于JAX陷阱和模式的实用指南，来自T5作者之一
