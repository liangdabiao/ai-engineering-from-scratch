# 构建你自己的迷你框架

> 你已经构建了神经元、层、网络、反向传播、激活函数、损失函数、优化器、正则化、初始化以及学习率调度。所有这些都是独立的部件。现在将这些部件连接成一个框架。不是PyTorch。不是TensorFlow。是你自己的。

**类型：** 构建
**语言：** Python
**前置条件：** 第三阶段全部（第01-09课）
**时间：** 约120分钟

## 学习目标

- 构建一个完整的深度学习框架（约500行），包含Module、Linear、ReLU、Sigmoid、Dropout、BatchNorm、Sequential、损失函数、优化器和DataLoader
- 解释Module抽象（前向传播、反向传播、参数）以及为什么需要训练/评估模式切换
- 将所有组件连接成一个可运行训练循环，用于在圆形分类任务上训练一个4层网络
- 将框架的每个组件映射到其PyTorch等价物（nn.Module、nn.Sequential、optim.Adam、DataLoader）

## 问题

你已经在不同的文件中散布了十节课的构建模块。一个`Value`类在这里，一个训练循环在那里，权重初始化在另一个文件中，学习率调度在又一个文件中。要训练一个网络，你需要从五个不同的课程中复制粘贴，并手动将它们连接起来。

这就是框架要解决的问题。PyTorch提供了`nn.Module`、`nn.Sequential`、`optim.Adam`、`DataLoader`以及一个将它们结合在一起的训练循环模式。TensorFlow提供了`keras.Layer`、`keras.Sequential`、`keras.optimizers.Adam`。这些不是魔法。它们是一种组织模式，使得定义、训练和评估网络成为可能，而无需每次都重新发明轮子。

你将用大约500行Python代码构建同样的东西。没有NumPy。没有外部依赖。一个能够定义任何前馈网络、使用SGD或Adam进行训练、批量处理数据、应用dropout和批量归一化、使用任意激活函数以及调度学习率的框架。

完成后，你将确切理解在PyTorch中写`model = nn.Sequential(...)`时发生了什么。你将理解为什么存在`model.train()`和`model.eval()`。你将理解为什么`optimizer.zero_grad()`是一个单独的调用。你将理解这一切，因为你亲手构建了这一切。

## 核心概念

### Module抽象

PyTorch中的每个层都继承自`nn.Module`。Module有三个职责：

1. **forward()** —— 根据输入计算输出
2. **parameters()** —— 返回所有可训练权重
3. **backward()** —— 计算梯度（在PyTorch中由autograd处理，在我们这里是显式的）

Linear层是一个Module。ReLU激活函数是一个Module。dropout层是一个Module。批量归一化层是一个Module。它们都具有相同的接口。

### Sequential容器

`nn.Sequential`将Modules串联起来。前向传播：数据经过Module 1，然后Module 2，然后Module 3。反向传播：反向串联。容器本身就是一个Module——它拥有forward()、parameters()和backward()。这是组合模式：一个Modules序列本身就是一个Module。

### 训练模式与评估模式

Dropout在训练期间随机将神经元置零，但在评估期间通过所有信号。批量归一化在训练期间使用批量统计量，但在评估期间使用运行平均值。`train()`和`eval()`方法切换此行为。每个Module都有一个`training`标志。

### 优化器

优化器使用梯度更新参数。SGD：`param -= lr * grad`。Adam：维护动量和方差估计，然后更新。优化器不知道网络架构——它只看到参数的扁平列表及其梯度。

### DataLoader

批处理很重要有两个原因。首先，对于大型问题，无法将整个数据集放入内存。其次，小批量梯度下降提供了有助于跳出局部极小值的噪声。DataLoader将数据分割成批次，并可以选择在每个epoch之间打乱。

### 框架架构

```mermaid
graph TD
    subgraph "Modules"
        Linear["Linear<br/>W*x + b"]
        ReLU["ReLU<br/>max(0, x)"]
        Sigmoid["Sigmoid<br/>1/(1+e^-x)"]
        Dropout["Dropout<br/>random zero mask"]
        BatchNorm["BatchNorm<br/>normalize activations"]
    end

    subgraph "Containers"
        Sequential["Sequential<br/>chains modules"]
    end

    subgraph "Loss Functions"
        MSE["MSELoss<br/>(pred - target)^2"]
        BCE["BCELoss<br/>binary cross-entropy"]
    end

    subgraph "Optimizers"
        SGD["SGD<br/>param -= lr * grad"]
        Adam["Adam<br/>adaptive moments"]
    end

    subgraph "Data"
        DataLoader["DataLoader<br/>batching + shuffle"]
    end

    Sequential --> |"contains"| Linear
    Sequential --> |"contains"| ReLU
    Sequential --> |"forward/backward"| MSE
    SGD --> |"updates"| Sequential
    DataLoader --> |"feeds"| Sequential
```

### 训练循环

```mermaid
sequenceDiagram
    participant DL as DataLoader
    participant M as Model
    participant L as Loss
    participant O as Optimizer

    loop Each Epoch
        DL->>M: batch of inputs
        M->>M: forward pass (layer by layer)
        M->>L: predictions
        L->>L: compute loss
        L->>M: backward pass (gradients)
        M->>O: parameters + gradients
        O->>M: updated parameters
        O->>O: zero gradients
    end
```

### Module层级结构

```mermaid
classDiagram
    class Module {
        +forward(x)
        +backward(grad)
        +parameters()
        +train()
        +eval()
    }

    class Linear {
        -weights
        -biases
        +forward(x)
        +backward(grad)
    }

    class ReLU {
        +forward(x)
        +backward(grad)
    }

    class Sequential {
        -modules[]
        +forward(x)
        +backward(grad)
        +parameters()
    }

    Module <|-- Linear
    Module <|-- ReLU
    Module <|-- Sequential
    Sequential *-- Module
```

```figure
gradient-clipping
```

## 动手构建

### 步骤1：Module基类

每个层实现的抽象接口。

```python
class Module:
    def __init__(self):
        self.training = True

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def parameters(self):
        return []

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
```

### 步骤2：Linear层

基本的构建模块。存储权重和偏置，计算前向传播Wx + b，以及反向传播的权重/输入梯度。

```python
import math
import random


class Linear(Module):
    def __init__(self, fan_in, fan_out):
        super().__init__()
        std = math.sqrt(2.0 / fan_in)
        self.weights = [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]
        self.biases = [0.0] * fan_out
        self.weight_grads = [[0.0] * fan_in for _ in range(fan_out)]
        self.bias_grads = [0.0] * fan_out
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.input = None

    def forward(self, x):
        self.input = x
        output = []
        for i in range(self.fan_out):
            val = self.biases[i]
            for j in range(self.fan_in):
                val += self.weights[i][j] * x[j]
            output.append(val)
        return output

    def backward(self, grad):
        input_grad = [0.0] * self.fan_in
        for i in range(self.fan_out):
            self.bias_grads[i] += grad[i]
            for j in range(self.fan_in):
                self.weight_grads[i][j] += grad[i] * self.input[j]
                input_grad[j] += grad[i] * self.weights[i][j]
        return input_grad

    def parameters(self):
        params = []
        for i in range(self.fan_out):
            for j in range(self.fan_in):
                params.append((self.weights, i, j, self.weight_grads))
            params.append((self.biases, i, None, self.bias_grads))
        return params
```

### 步骤3：激活函数Modules

将ReLU、Sigmoid和Tanh作为Modules实现。每个都缓存反向传播所需的内容。

```python
class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        self.mask = [1.0 if v > 0 else 0.0 for v in x]
        return [max(0.0, v) for v in x]

    def backward(self, grad):
        return [g * m for g, m in zip(grad, self.mask)]


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        self.output = []
        for v in x:
            v = max(-500, min(500, v))
            self.output.append(1.0 / (1.0 + math.exp(-v)))
        return self.output

    def backward(self, grad):
        return [g * o * (1 - o) for g, o in zip(grad, self.output)]


class Tanh(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        self.output = [math.tanh(v) for v in x]
        return self.output

    def backward(self, grad):
        return [g * (1 - o * o) for g, o in zip(grad, self.output)]
```

### 步骤4：Dropout Module

在训练期间随机将元素置零。将剩余元素缩放至 1/(1-p)，以保持期望值不变。评估期间不执行任何操作。

```python
class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        if not self.training:
            return x
        self.mask = [0.0 if random.random() < self.p else 1.0 / (1 - self.p) for _ in x]
        return [v * m for v, m in zip(x, self.mask)]

    def backward(self, grad):
        if self.mask is None:
            return grad
        return [g * m for g, m in zip(grad, self.mask)]
```

### 第 5 步：批归一化模块

将激活值在每个特征上跨批次归一化为零均值和单位方差。为评估模式维护运行统计量。

```python
class BatchNorm(Module):
    def __init__(self, size, momentum=0.1, eps=1e-5):
        super().__init__()
        self.size = size
        self.gamma = [1.0] * size
        self.beta = [0.0] * size
        self.gamma_grads = [0.0] * size
        self.beta_grads = [0.0] * size
        self.running_mean = [0.0] * size
        self.running_var = [1.0] * size
        self.momentum = momentum
        self.eps = eps
        self.x_norm = None
        self.std_inv = None
        self.batch_input = None

    def forward_batch(self, batch):
        batch_size = len(batch)
        output_batch = []

        if self.training:
            mean = [0.0] * self.size
            for sample in batch:
                for j in range(self.size):
                    mean[j] += sample[j]
            mean = [m / batch_size for m in mean]

            var = [0.0] * self.size
            for sample in batch:
                for j in range(self.size):
                    var[j] += (sample[j] - mean[j]) ** 2
            var = [v / batch_size for v in var]

            self.std_inv = [1.0 / math.sqrt(v + self.eps) for v in var]

            self.x_norm = []
            self.batch_input = batch
            for sample in batch:
                normed = [(sample[j] - mean[j]) * self.std_inv[j] for j in range(self.size)]
                self.x_norm.append(normed)
                output = [self.gamma[j] * normed[j] + self.beta[j] for j in range(self.size)]
                output_batch.append(output)

            for j in range(self.size):
                self.running_mean[j] = (1 - self.momentum) * self.running_mean[j] + self.momentum * mean[j]
                self.running_var[j] = (1 - self.momentum) * self.running_var[j] + self.momentum * var[j]
        else:
            std_inv = [1.0 / math.sqrt(v + self.eps) for v in self.running_var]
            for sample in batch:
                normed = [(sample[j] - self.running_mean[j]) * std_inv[j] for j in range(self.size)]
                output = [self.gamma[j] * normed[j] + self.beta[j] for j in range(self.size)]
                output_batch.append(output)

        return output_batch

    def forward(self, x):
        result = self.forward_batch([x])
        return result[0]

    def backward(self, grad):
        if self.x_norm is None:
            return grad
        for j in range(self.size):
            self.gamma_grads[j] += self.x_norm[0][j] * grad[j]
            self.beta_grads[j] += grad[j]
        return [grad[j] * self.gamma[j] * self.std_inv[j] for j in range(self.size)]

    def parameters(self):
        params = []
        for j in range(self.size):
            params.append((self.gamma, j, None, self.gamma_grads))
            params.append((self.beta, j, None, self.beta_grads))
        return params
```

### 第 6 步：顺序容器

将模块串联起来。前向传播从左到右，反向传播从右到左。

```python
class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)

    def forward(self, x):
        for module in self.modules:
            x = module.forward(x)
        return x

    def backward(self, grad):
        for module in reversed(self.modules):
            grad = module.backward(grad)
        return grad

    def parameters(self):
        params = []
        for module in self.modules:
            params.extend(module.parameters())
        return params

    def train(self):
        self.training = True
        for module in self.modules:
            module.train()

    def eval(self):
        self.training = False
        for module in self.modules:
            module.eval()
```

### 第 7 步：损失函数

均方误差和二分类交叉熵。每个都返回损失值，并提供返回梯度的 backward() 方法。

```python
class MSELoss:
    def __call__(self, predicted, target):
        self.predicted = predicted
        self.target = target
        n = len(predicted)
        self.loss = sum((p - t) ** 2 for p, t in zip(predicted, target)) / n
        return self.loss

    def backward(self):
        n = len(self.predicted)
        return [2 * (p - t) / n for p, t in zip(self.predicted, self.target)]


class BCELoss:
    def __call__(self, predicted, target):
        self.predicted = predicted
        self.target = target
        eps = 1e-7
        n = len(predicted)
        self.loss = 0
        for p, t in zip(predicted, target):
            p = max(eps, min(1 - eps, p))
            self.loss += -(t * math.log(p) + (1 - t) * math.log(1 - p))
        self.loss /= n
        return self.loss

    def backward(self):
        eps = 1e-7
        n = len(self.predicted)
        grads = []
        for p, t in zip(self.predicted, self.target):
            p = max(eps, min(1 - eps, p))
            grads.append((-t / p + (1 - t) / (1 - p)) / n)
        return grads
```

### 第 8 步：SGD 和 Adam 优化器

两者都接受参数列表并使用梯度更新权重。

```python
class SGD:
    def __init__(self, parameters, lr=0.01):
        self.params = parameters
        self.lr = lr

    def step(self):
        for container, i, j, grad_container in self.params:
            if j is not None:
                container[i][j] -= self.lr * grad_container[i][j]
            else:
                container[i] -= self.lr * grad_container[i]

    def zero_grad(self):
        for container, i, j, grad_container in self.params:
            if j is not None:
                grad_container[i][j] = 0.0
            else:
                grad_container[i] = 0.0


class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [0.0] * len(parameters)
        self.v = [0.0] * len(parameters)

    def step(self):
        self.t += 1
        for idx, (container, i, j, grad_container) in enumerate(self.params):
            if j is not None:
                g = grad_container[i][j]
            else:
                g = grad_container[i]

            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * g
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * g * g

            m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
            v_hat = self.v[idx] / (1 - self.beta2 ** self.t)

            update = self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

            if j is not None:
                container[i][j] -= update
            else:
                container[i] -= update

    def zero_grad(self):
        for container, i, j, grad_container in self.params:
            if j is not None:
                grad_container[i][j] = 0.0
            else:
                grad_container[i] = 0.0
```

### 第 9 步：数据加载器

将数据分成批次，可选地在每个 epoch 进行洗牌。

```python
class DataLoader:
    def __init__(self, data, batch_size=32, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch = [self.data[i] for i in batch_indices]
            inputs = [item[0] for item in batch]
            targets = [item[1] for item in batch]
            yield inputs, targets

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size
```

### 第 10 步：在圆分类任务上训练一个 4 层网络

将所有组件连接起来。定义模型，选择损失函数，选择优化器，运行训练循环。

```python
def make_circle_data(n=500, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], [label]))
    return data


def train():
    random.seed(42)

    model = Sequential(
        Linear(2, 16),
        ReLU(),
        Linear(16, 16),
        ReLU(),
        Linear(16, 8),
        ReLU(),
        Linear(8, 1),
        Sigmoid(),
    )

    criterion = BCELoss()
    optimizer = Adam(model.parameters(), lr=0.01)

    data = make_circle_data(500)
    split = int(len(data) * 0.8)
    train_data = data[:split]
    test_data = data[split:]

    loader = DataLoader(train_data, batch_size=16, shuffle=True)

    model.train()

    for epoch in range(100):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_inputs, batch_targets in loader:
            batch_loss = 0
            for x, t in zip(batch_inputs, batch_targets):
                pred = model.forward(x)
                loss = criterion(pred, t)
                batch_loss += loss

                optimizer.zero_grad()
                grad = criterion.backward()
                model.backward(grad)
                optimizer.step()

                predicted_class = 1.0 if pred[0] >= 0.5 else 0.0
                if predicted_class == t[0]:
                    total_correct += 1
                total_samples += 1

            total_loss += batch_loss

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples * 100

        if epoch % 10 == 0 or epoch == 99:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f} | Train Accuracy: {accuracy:.1f}%")

    model.eval()
    correct = 0
    for x, t in test_data:
        pred = model.forward(x)
        predicted_class = 1.0 if pred[0] >= 0.5 else 0.0
        if predicted_class == t[0]:
            correct += 1
    test_accuracy = correct / len(test_data) * 100
    print(f"\nTest Accuracy: {test_accuracy:.1f}% ({correct}/{len(test_data)})")

    return model, test_accuracy
```

## 使用它

以下是你刚刚构建的 PyTorch 等价版本：

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_predictions = model(test_inputs)
```

结构完全相同。`Sequential`、`Linear`、`ReLU`、`Sigmoid`、`BCELoss`、`Adam`、`zero_grad`、`backward`、`step`、`train`、`eval`。每个概念一一对应。区别在于 PyTorch 自动处理自动微分（无需在每个模块中实现 backward()），可以在 GPU 上运行，并且经过多年优化。但核心骨架是相同的。

现在当你看到 PyTorch 代码时，你确切地知道每一行发生了什么。这种理解正是关键所在。

## 发布

本課(lesson)产出：
- `outputs/prompt-framework-architect.md` -- 使用框架抽象设计神经网络架构的提示

## 练习

1. 添加一个 `SoftmaxCrossEntropyLoss` 类用于多分类。对预测结果进行 Softmax，计算交叉熵损失，并处理组合的反向传播。在一个 3 类螺旋数据集上进行测试。

2. 在优化器中实现学习率调度：添加一个 `set_lr()` 方法并接入第 09 课的余弦调度。使用 warmup + cosine 训练圆分类器，并与恒定学习率进行比较。

3. 为 Sequential 添加 `save()` 和 `load()` 方法，将所有权重序列化到 JSON 文件并加载回来。验证加载后的模型产生与原始模型相同的预测。

4. 在 Adam 优化器中实现权重衰减（L2 正则化）。添加一个 `weight_decay` 参数，每步将权重向零收缩。比较 decay=0 和 decay=0.01 的训练效果。

5. 将逐样本训练循环替换为合适的小批量梯度累积：累积一个批次中所有样本的梯度，然后除以批次大小并执行一次优化器步骤。测量这是否改变收敛速度。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|----------------------|
|  Module  |  "层"  |  框架中的基础抽象——任何具有 forward()、backward() 和 parameters() 方法的对象 |
|  Sequential  |  "按顺序堆叠层"  |  一个串联模块的容器，前向传播按顺序应用模块，反向传播按相反顺序应用 |
|  Forward pass  |  "运行网络"  |  通过按顺序将输入传递给每个模块来计算输出 |
|  Backward pass  |  "计算梯度"  |  反向传播损失梯度通过每个模块以计算参数梯度 |
|  Parameters  |  "可训练的权重"  |  网络中优化器可以更新的所有值——权重和偏置 |
|  Optimizer  |  "更新权重的对象"  |  使用梯度更新参数的算法，实现 SGD、Adam 或其他规则 |
|  DataLoader  |  "提供数据的对象"  |  一个迭代器，将数据集分成批次，可选地在各 epoch 之间进行洗牌 |
|  Training mode  |  "model.train()"  |  一个标志，启用随机行为，如 dropout 和基于批次统计的批归一化 |
| 评估模式 | "model.eval()" | 一个标志，用于禁用dropout并使用运行统计量进行批归一化 |
| 零梯度 | "清除梯度" | 在计算下一批次的梯度之前将所有参数梯度重置为零 |

## 延伸阅读

- Paszke 等人，《PyTorch: An Imperative Style, High-Performance Deep Learning Library》(2019) —— 描述 PyTorch 设计决策的论文
- Chollet，《Deep Learning with Python, Second Edition》(2021) —— 第3章使用相同的模块/层抽象覆盖了 Keras 内部实现
- Johnson，《Tiny-DNN》(https://github.com/tiny-dnn/tiny-dnn) —— 一个用于理解框架内部实现的仅头文件的 C++ 深度学习框架
