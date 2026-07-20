# 调试与性能分析

> 最糟糕的AI错误不会导致崩溃——它们会在垃圾数据上默默训练，然后汇报出一条漂亮的损失曲线。

**类型：** 构建
**语言：** Python
**前置条件：** 第1课（开发环境）、PyTorch基本熟悉
**时长：** 约60分钟

## 学习目标

- 使用条件语句 `breakpoint()` 和 `debug_print` 在训练过程中检查张量的形状、数据类型和NaN值
- 使用 `breakpoint()`、`debug_print` 和 `cProfile` 分析训练循环以发现瓶颈
- 检测常见AI错误：形状不匹配、NaN损失、数据泄露和错误设备上的张量
- 设置TensorBoard以可视化损失曲线、权重直方图和梯度分布

## 问题

AI代码的错误表现与常规代码不同。一个网络应用崩溃时会输出堆栈跟踪，而一个配置错误的训练循环会运行8小时、消耗200美元GPU费用，最终产出一个预测每个输入均值的模型。代码从未报错。错误可能是一个张量放在了错误的设备上、一个被遗忘的 `.detach()`，或者标签泄漏到了特征中。

你需要能够在你浪费时间和算力之前捕捉这些静默错误的调试工具。

## 核心概念

AI调试在三个层面进行：

```mermaid
graph TD
    L3["3. Training Dynamics<br/>Loss curves, gradient norms, activations"] --> L2
    L2["2. Tensor Operations<br/>Shapes, dtypes, devices, NaN/Inf values"] --> L1
    L1["1. Standard Python<br/>Breakpoints, logging, profiling, memory"]
```

大多数人会直接跳到第3层（盯着TensorBoard看），但80%的AI错误存在于第1层和第2层。

## 动手构建

### 第一部分：打印调试（没错，它有效）

打印调试常被轻视，但不应如此。对于张量代码，有针对性的打印语句比单步调试更有效，因为你需要一次性看到形状、数据类型和数值范围。

```python
def debug_print(name, tensor):
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
          f"device={tensor.device}, "
          f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
          f"mean={tensor.mean().item():.4f}, "
          f"has_nan={tensor.isnan().any().item()}")
```

在每个可疑操作后调用此方法。找到错误后，移除打印语句。就这么简单。

### 第二部分：Python调试器（pdb和breakpoint）

内置调试器在AI工作中被低估了。将 `breakpoint()` 放入训练循环中，交互式地检查张量。

```python
def training_step(model, batch, criterion, optimizer):
    inputs, labels = batch
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    if loss.item() > 100 or torch.isnan(loss):
        breakpoint()

    loss.backward()
    optimizer.step()
```

当调试器进入后，有用的命令：

- `p outputs.shape` 用于检查形状
- `p outputs.shape` 用于查看损失值
- `p outputs.shape` 用于统计NaN数量
- `p outputs.shape` 用于检查梯度
- `p outputs.shape` 继续执行，`p loss.item()` 退出

这是条件调试。只有当某些看起来不对时你才会停下来。对于10000步的训练运行，这很重要。

### 第3部分：Python日志记录

当你的调试超出快速检查的范围时，用日志记录替换print语句。

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting training: lr=%.4f, batch_size=%d", lr, batch_size)
logger.warning("Loss spike detected: %.4f at step %d", loss.item(), step)
logger.error("NaN loss at step %d, stopping", step)
```

日志记录会提供时间戳、严重级别和文件输出。当训练在凌晨3点失败时，你需要的是日志文件，而不是已经滚出屏幕的终端输出。

### 第4部分：代码段计时

了解时间消耗在哪里是优化的第一步。

```python
import time

class Timer:
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        print(f"[{self.name}] {elapsed:.4f}s")

with Timer("data loading"):
    batch = next(dataloader_iter)

with Timer("forward pass"):
    outputs = model(batch)

with Timer("backward pass"):
    loss.backward()
```

常见发现：数据加载占用60%的训练时间。解决方法是在DataLoader中使用`num_workers > 0`，而不是换更快的GPU。

### 第5部分：cProfile和line_profiler

当你需要比手动计时器更多的东西时：

```bash
python -m cProfile -s cumtime train.py
```

这会显示按累积时间排序的每个函数调用。对于逐行分析：

```bash
pip install line_profiler
```

```python
@profile
def train_step(model, data, target):
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    return loss

# Run with: kernprof -l -v train.py
```

### 第6部分：内存分析

#### 使用tracemalloc分析CPU内存

```python
import tracemalloc

tracemalloc.start()

# your code here
model = build_model()
data = load_dataset()

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics("lineno")
for stat in top_stats[:10]:
    print(stat)
```

#### 使用memory_profiler分析CPU内存

```bash
pip install memory_profiler
```

```python
from memory_profiler import profile

@profile
def load_data():
    raw = read_csv("data.csv")       # watch memory jump here
    processed = preprocess(raw)       # and here
    return processed
```

使用`python -m memory_profiler your_script.py`运行以查看逐行内存使用情况。

#### 使用PyTorch分析GPU内存

```python
import torch

if torch.cuda.is_available():
    print(torch.cuda.memory_summary())

    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

当遇到OOM（内存溢出）时：

1. 减小批大小（始终首先尝试）
2. 使用`torch.cuda.empty_cache()`释放缓存内存
3. 对于大型中间变量，先使用`torch.cuda.empty_cache()`再使用`del tensor`
4. 使用混合精度（`torch.cuda.empty_cache()`）将内存使用减半
5. 对非常深的模型使用梯度检查点

### 第7部分：常见AI错误及其捕捉方法

#### 形状不匹配

最常见的错误。张量的形状为`[batch, features]`，而模型期望`[batch, channels, height, width]`。

```python
def check_shapes(model, sample_input):
    print(f"Input: {sample_input.shape}")
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            in_shape = inp[0].shape if isinstance(inp, tuple) else inp.shape
            out_shape = out.shape if hasattr(out, "shape") else type(out)
            print(f"  {name}: {in_shape} -> {out_shape}")
        return hook

    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(sample_input)

    for h in hooks:
        h.remove()
```

使用一个样本批次运行一次。它映射了模型中的每个形状变换。

#### NaN损失

NaN损失意味着某些东西爆炸了。常见原因：

- 学习率过高
- 自定义损失中的除以零
- 对零或负数取对数
- RNN中的梯度爆炸

```python
def detect_nan(model, loss, step):
    if torch.isnan(loss):
        print(f"NaN loss at step {step}")
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"  NaN gradient in {name}")
                if torch.isinf(param.grad).any():
                    print(f"  Inf gradient in {name}")
        return True
    return False
```

#### 数据泄露

模型在测试集上达到99%的准确率。听起来很棒。但这其实是一个错误。

```python
def check_data_leakage(train_set, test_set, id_column="id"):
    train_ids = set(train_set[id_column].tolist())
    test_ids = set(test_set[id_column].tolist())
    overlap = train_ids & test_ids
    if overlap:
        print(f"DATA LEAKAGE: {len(overlap)} samples in both train and test")
        return True
    return False
```

还要检查时间泄漏：使用未来数据预测过去。在分割之前按时间戳排序。

#### 设备错误

不同设备（CPU与GPU）上的张量会导致运行时错误。但有时一个张量会静默地留在CPU上，而其他所有东西都在GPU上，训练只是运行缓慢。

```python
def check_devices(model, *tensors):
    model_device = next(model.parameters()).device
    print(f"Model device: {model_device}")
    for i, t in enumerate(tensors):
        if t.device != model_device:
            print(f"  WARNING: tensor {i} on {t.device}, model on {model_device}")
```

### 第8部分：TensorBoard基础

TensorBoard 向你展示训练过程中的内部情况。

```bash
pip install tensorboard
```

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/experiment_1")

for step in range(num_steps):
    loss = train_step(model, batch)

    writer.add_scalar("loss/train", loss.item(), step)
    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], step)

    if step % 100 == 0:
        for name, param in model.named_parameters():
            writer.add_histogram(f"weights/{name}", param, step)
            if param.grad is not None:
                writer.add_histogram(f"grads/{name}", param.grad, step)

writer.close()
```

启动它：

```bash
tensorboard --logdir=runs
```

需要注意什么：

- **损失不下降**：学习率过低，或模型架构问题
- **损失剧烈震荡**：学习率过高
- **损失变为 NaN**：数值不稳定（见上文 NaN 部分）
- **训练损失下降，验证损失上升**：过拟合
- **权重直方图塌缩到零**：梯度消失
- **梯度直方图爆炸**：需要梯度裁剪

### 第 9 部分：VS Code 调试器

要进行交互式调试，使用 `launch.json` 配置 VS Code：

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Training",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}
```

通过点击行号设置断点。使用变量面板检查张量属性。调试控制台允许你在执行过程中运行任意 Python 表达式。

适用于逐步执行数据预处理流水线，希望查看每个转换步骤的场景。

## 使用它

以下是可以捕获大多数 AI 错误的调试工作流：

1. **训练前**：使用一个样本批次运行 `check_shapes`。验证输入和输出维度是否符合预期。
2. **前 10 步**：对损失、输出和梯度使用 `check_shapes`。确认没有 NaN，且数值在合理范围内。
3. **训练期间**：记录损失、学习率和梯度范数。使用 TensorBoard 进行可视化。
4. **出现问题时**：在失败点放置 `check_shapes`。交互式检查张量。
5. **性能方面**：计时数据加载、前向传播和反向传播。如果接近内存不足，则分析内存使用。

## 发布

运行调试工具包脚本：

```bash
python phases/00-setup-and-tooling/12-debugging-and-profiling/code/debug_tools.py
```

参见 `outputs/prompt-debug-ai-code.md` 获取帮助诊断 AI 特定错误的提示。

## 练习

1. 运行 `debug_tools.py` 并阅读每个部分的输出。修改虚拟模型以引入 NaN（提示：在前向传播中除以零），观察检测器如何捕获它。
2. 使用 `debug_tools.py` 分析训练循环并找出最慢的函数。
3. 使用 `debug_tools.py` 找出数据加载流水线中哪一行分配了最多内存。
4. 为简单训练运行设置 TensorBoard，识别模型是否过拟合。
5. 在训练循环中使用 `debug_tools.py`。练习从调试器提示符检查张量形状、设备和梯度值。
