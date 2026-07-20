# 调试神经网络

> 你的网络编译通过了。它运行了。它产生了一个数值。这个数值是错误的，但没有崩溃。欢迎来到最困难的一类调试——没有错误消息的那种。

**类型：** 构建
**语言：** Python, PyTorch
**先修课程：** 阶段03 第01-10课（特别是反向传播、损失函数、优化器）
**时间：** 约90分钟

## 学习目标

- 使用系统化的调试策略诊断常见的神经网络故障（NaN损失、平坦损失曲线、过拟合、振荡）
- 应用“过拟合一个批次”技术来验证模型架构和训练循环是否正确
- 检查梯度大小、激活分布和权重范数以识别梯度消失/爆炸问题
- 构建一个涵盖数据管道、模型架构、损失函数、优化器和学习率问题的调试清单

## 问题

传统软件在出问题时会崩溃。空指针会引发异常。类型不匹配在编译时失败。差一错误会产生明显错误的结果。

神经网络却不给你这种便利。

一个出错的神经网络会运行到结束，打印一个损失值，并输出预测。损失可能会下降。预测可能看起来合理。但模型在无声地出错——学习捷径、记忆噪声、或收敛到无用的局部最小值。谷歌研究人员估计，60-70%的机器学习调试时间花费在“无声”的bug上，这些bug不产生错误但降低模型质量。

一个工作模型和一个故障模型之间的差异往往只是一行代码放错了位置：缺少`zero_grad()`、维度转置、学习率差了10倍。经典文章《训练神经网络的秘诀》（2019）开头写道：“最常见的神经网络错误是不导致崩溃的bug。”

本节课教你如何找到这些bug。

## 核心概念

### 调试心态

忘掉“打印并祈祷”式的调试。神经网络调试需要系统化的方法，因为反馈循环很慢（每次训练运行需要几分钟到几小时），而且症状模糊（糟糕的损失值可能意味着20种不同的问题）。

黄金法则：**从简单开始，一次只添加一个复杂度，并独立验证每个部分。**

```mermaid
flowchart TD
    A["Loss not decreasing"] --> B{"Check learning rate"}
    B -->|"Too high"| C["Loss oscillates or explodes"]
    B -->|"Too low"| D["Loss barely moves"]
    B -->|"Reasonable"| E{"Check gradients"}
    E -->|"All zeros"| F["Dead ReLUs or vanishing gradients"]
    E -->|"NaN/Inf"| G["Exploding gradients"]
    E -->|"Normal"| H{"Check data pipeline"}
    H -->|"Labels shuffled"| I["Random-chance accuracy"]
    H -->|"Preprocessing bug"| J["Model learns noise"]
    H -->|"Data is fine"| K{"Check architecture"}
    K -->|"Too small"| L["Underfitting"]
    K -->|"Too deep"| M["Optimization difficulty"]
```

### 症状1：损失不下降

这是最常见的抱怨。训练循环运行，轮次流逝，损失保持平坦或剧烈振荡。

**错误的学习率。** 过高：损失振荡或跳到NaN。过低：损失下降极慢，看起来平坦。对于Adam，从1e-3开始。对于SGD，从1e-1或1e-2开始。在得出其他环节有问题的结论之前，总是尝试三个跨度10倍的学习率（例如1e-2, 1e-3, 1e-4）。

**死亡ReLU。** 如果ReLU神经元收到很大的负输入，它输出0并且梯度为0。它再也无法激活。如果足够多的神经元死亡，网络无法学习。检查：打印每个ReLU层后激活值恰好为0的比例。如果超过50%是死的，切换为LeakyReLU或降低学习率。

**梯度消失。** 在具有sigmoid或tanh激活函数的深度网络中，梯度在反向传播时呈指数级缩小。当它们到达第一层时，几乎为0。第一层停止学习。修复：使用ReLU/GELU、添加残差连接或使用批归一化。

**梯度爆炸。** 相反的问题——梯度呈指数级增长。常见于RNN和极深网络。损失跳到NaN。修复：梯度裁剪（`torch.nn.utils.clip_grad_norm_`）、降低学习率或添加归一化。

### 症状2：损失下降但模型效果差

损失下降了。训练准确率达到99%。但测试准确率只有55%。或者模型在真实数据上产生无意义的输出。

**过拟合。** 模型记忆训练数据而不是学习模式。训练损失和验证损失之间的差距随时间增大。修复：更多数据、dropout、权重衰减、早停、数据增强。

**数据泄露。** 测试数据泄露到训练中。准确率高得可疑。常见原因：在划分前进行shuffle、使用全数据集统计量进行预处理、不同划分间存在重复样本。修复：先划分、再预处理、检查重复。

**标签错误。** 大多数真实数据集中5-10%的标签是错误的（Northcutt等，2021——《测试集中普遍存在的标签错误》）。模型学习了噪声。修复：使用置信学习来发现和修正错误标注的样本，或使用损失截断忽略高损失样本。

### 症状3：损失中出现NaN或Inf

损失值变为`nan`或`inf`。训练中断。

**学习率过高。** 梯度更新超调太远，导致权重爆炸。修复：降低10倍。

**log(0)或log(负数)。** 交叉熵损失计算`log(p)`。如果模型输出恰好为0或负概率，对数爆炸。修复：将预测值裁剪到`[eps, 1-eps]`之间，其中`eps=1e-7`。

**除以零。** 批归一化除以标准差。一个批次的值为常数时，std=0。修复：在分母中添加epsilon（PyTorch默认这样做，但自定义实现可能没有）。

**数值溢出。** 大的激活值输入`exp()`产生Inf。Softmax尤其容易。修复：在指数运算前减去最大值（log-sum-exp技巧）。

### 技术1：梯度检查

将你的解析梯度（来自反向传播）与数值梯度（来自有限差分）进行比较。如果它们不一致，你的反向传播存在错误。

参数 `w` 的数值梯度：

```
grad_numerical = (loss(w + eps) - loss(w - eps)) / (2 * eps)
```

一致性度量（相对差异）：

```
rel_diff = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

如果 `rel_diff < 1e-5`：正确。如果 `rel_diff > 1e-3`：几乎肯定是错误。

```mermaid
flowchart LR
    A["Parameter w"] --> B["w + eps"]
    A --> C["w - eps"]
    B --> D["Forward pass"]
    C --> E["Forward pass"]
    D --> F["loss+"]
    E --> G["loss-"]
    F --> H["(loss+ - loss-) / 2eps"]
    G --> H
    H --> I["Compare to backprop gradient"]
```

### 技术2：激活统计

在训练过程中监控每层之后激活的均值和标准差。健康的网络保持激活的均值接近0，标准差接近1（经过归一化后），或者至少是有界的。

|  健康指标  |  均值  |  标准差  |  诊断  |
|-----------------|------|-----|-----------|
|  健康  |  ~0  |  ~1  |  网络正常学习  |
|  饱和  |  >>0或<<0  |  ~0  |  激活值卡在极端值  |
|  死亡  |  0  |  0  |  神经元死亡（全零）  |
|  爆炸  |  >>10  |  >>10  |  激活值无界增长  |

### 技术3：梯度流可视化

绘制每层的平均梯度幅值。在健康的网络中，各层的梯度幅值应大致相似。如果早期层的梯度比后期层小1000倍，则出现了梯度消失。

```mermaid
graph LR
    subgraph "Healthy Gradient Flow"
        L1["Layer 1<br/>grad: 0.05"] --- L2["Layer 2<br/>grad: 0.04"] --- L3["Layer 3<br/>grad: 0.06"] --- L4["Layer 4<br/>grad: 0.05"]
    end
```

```mermaid
graph LR
    subgraph "Vanishing Gradient Flow"
        V1["Layer 1<br/>grad: 0.0001"] --- V2["Layer 2<br/>grad: 0.003"] --- V3["Layer 3<br/>grad: 0.02"] --- V4["Layer 4<br/>grad: 0.08"]
    end
```

### 技术4：过拟合单批次测试

深度学习中最重要的一种调试技术。

取一个小批次（8-32个样本）。在其上训练100次以上迭代。损失应降至接近零，训练准确率应达到100%。如果不是这样，你的模型或训练循环存在根本性错误——不要继续进行完整训练。

这个测试能发现：
- 损坏的损失函数
- 损坏的反向传播
- 架构太小无法表示数据
- 优化器未连接到模型参数
- 数据与标签不对齐

运行只需要30秒，却可以节省数小时的完整训练调试时间。

### 技术5：学习率寻找器

Leslie Smith (2017) 提出在一个epoch内将学习率从非常小（1e-7）扫描到非常大（10），同时记录损失。绘制损失与学习率的关系图。最优学习率大约比损失开始最快下降的学习率小10倍。

```mermaid
graph TD
    subgraph "LR Finder Plot"
        direction LR
        A["1e-7: loss=2.3"] --> B["1e-5: loss=2.3"]
        B --> C["1e-3: loss=1.8"]
        C --> D["1e-2: loss=0.9 -- steepest"]
        D --> E["1e-1: loss=0.5"]
        E --> F["1.0: loss=NaN -- too high"]
    end
```

本例中的最佳学习率：~1e-3（在最陡点之前的一个数量级）。

### 常见PyTorch错误

这些是PyTorch社区中浪费集体时间最多的错误：

|  错误  |  症状  |  修复  |
|-----|---------|-----|
|  忘记 `optimizer.zero_grad()`  |  梯度在各个批次间累积，损失振荡  |  在 `loss.backward()` 之前添加 `optimizer.zero_grad()`  |
|  在测试时忘记 `model.eval()`  |  Dropout和批归一化行为不同，测试准确率在不同运行间变化  |  添加 `model.eval()` 和 `torch.no_grad()`  |
|  张量形状错误  |  静默广播产生错误结果，没有报错  |  在调试时每个操作后打印形状  |
|  CPU/GPU不匹配  |  `RuntimeError: expected CUDA tensor`  |  在模型和数据上都使用 `.to(device)`  |
|  未分离张量  |  计算图无限增长，内存溢出  |  使用 `.detach()` 或 `with torch.no_grad()`  |
| 原地操作破坏自动求导 | `RuntimeError: modified by in-place operation` | 将 `x += 1` 替换为 `x = x + 1` |
| 数据未归一化 | 损失卡在随机水平 | 将输入归一化至均值=0，标准差=1 |
| 标签数据类型错误 | 交叉熵期望 `Long`，但得到 `Float` | 转换标签：`labels.long()` |

### 主调试表

| 症状 | 可能原因 | 首选尝试 |
|---------|-------------|-------------------|
| 损失卡在 -log(1/类别数) | 模型预测均匀分布 | 检查数据管道，确保标签与输入匹配 |
| 几步后损失变为NaN | 学习率过高 | 将学习率降低10倍 |
| 损失立即NaN | log(0) 或除以零 | 在log/除法运算中添加epsilon |
| 损失剧烈震荡 | 学习率过高或批次大小过小 | 降低学习率，增大批次大小 |
| 损失下降后停滞 | 学习率对微调阶段过高 | 添加学习率调度（余弦或阶梯衰减） |
| 训练准确率高，测试准确率低 | 过拟合 | 添加dropout、权重衰减、更多数据 |
| 训练准确率 = 测试准确率 = 随机水平 | 模型未学习到任何内容 | 执行单批次过拟合测试 |
| 训练准确率 = 测试准确率但两者都低 | 欠拟合 | 更大的模型、更多层、更多特征 |
| 梯度全为零 | ReLU死亡或计算图断开 | 切换为LeakyReLU，检查 `.requires_grad` |
| 训练时内存不足 | 批次过大或计算图未释放 | 减小批次大小，评估时使用 `torch.no_grad()` |

```figure
learning-curves
```

## 动手构建

一个监控激活值、梯度和损失曲线的诊断工具包。你将故意破坏网络，并使用该工具包诊断每个问题。

### 第1步：NetworkDebugger类

钩入PyTorch模型，记录每层的激活值和梯度统计信息。

```python
import torch
import torch.nn as nn
import math


class NetworkDebugger:
    def __init__(self, model):
        self.model = model
        self.activation_stats = {}
        self.gradient_stats = {}
        self.loss_history = []
        self.lr_losses = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.LeakyReLU)):
                hook = module.register_forward_hook(self._make_activation_hook(name))
                self.hooks.append(hook)
                hook = module.register_full_backward_hook(self._make_gradient_hook(name))
                self.hooks.append(hook)

    def _make_activation_hook(self, name):
        def hook(module, input, output):
            with torch.no_grad():
                out = output.detach().float()
                self.activation_stats[name] = {
                    "mean": out.mean().item(),
                    "std": out.std().item(),
                    "fraction_zero": (out == 0).float().mean().item(),
                    "min": out.min().item(),
                    "max": out.max().item(),
                }
        return hook

    def _make_gradient_hook(self, name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                with torch.no_grad():
                    grad = grad_output[0].detach().float()
                    self.gradient_stats[name] = {
                        "mean": grad.mean().item(),
                        "std": grad.std().item(),
                        "abs_mean": grad.abs().mean().item(),
                        "max": grad.abs().max().item(),
                    }
        return hook

    def record_loss(self, loss_value):
        self.loss_history.append(loss_value)

    def check_loss_health(self):
        if len(self.loss_history) < 2:
            return "NOT_ENOUGH_DATA"
        recent = self.loss_history[-10:]
        if any(math.isnan(v) or math.isinf(v) for v in recent):
            return "NAN_OR_INF"
        if len(self.loss_history) >= 20:
            first_half = sum(self.loss_history[:10]) / 10
            second_half = sum(self.loss_history[-10:]) / 10
            if second_half >= first_half * 0.99:
                return "NOT_DECREASING"
        if len(recent) >= 5:
            diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
            if max(diffs) - min(diffs) > 2 * abs(sum(diffs) / len(diffs)):
                return "OSCILLATING"
        return "HEALTHY"

    def check_activations(self):
        issues = []
        for name, stats in self.activation_stats.items():
            if stats["fraction_zero"] > 0.5:
                issues.append(f"DEAD_NEURONS: {name} has {stats['fraction_zero']:.0%} zero activations")
            if abs(stats["mean"]) > 10:
                issues.append(f"EXPLODING_ACTIVATIONS: {name} mean={stats['mean']:.2f}")
            if stats["std"] < 1e-6:
                issues.append(f"COLLAPSED_ACTIVATIONS: {name} std={stats['std']:.2e}")
        return issues if issues else ["HEALTHY"]

    def check_gradients(self):
        issues = []
        grad_magnitudes = []
        for name, stats in self.gradient_stats.items():
            grad_magnitudes.append((name, stats["abs_mean"]))
            if stats["abs_mean"] < 1e-7:
                issues.append(f"VANISHING_GRADIENT: {name} abs_mean={stats['abs_mean']:.2e}")
            if stats["abs_mean"] > 100:
                issues.append(f"EXPLODING_GRADIENT: {name} abs_mean={stats['abs_mean']:.2e}")
        if len(grad_magnitudes) >= 2:
            first_mag = grad_magnitudes[0][1]
            last_mag = grad_magnitudes[-1][1]
            if last_mag > 0 and first_mag / last_mag > 100:
                issues.append(f"GRADIENT_RATIO: first/last = {first_mag/last_mag:.0f}x (vanishing)")
        return issues if issues else ["HEALTHY"]

    def print_report(self):
        print("\n=== NETWORK DEBUGGER REPORT ===")
        print(f"\nLoss health: {self.check_loss_health()}")
        if self.loss_history:
            print(f"  Last 5 losses: {[f'{v:.4f}' for v in self.loss_history[-5:]]}")
        print("\nActivation diagnostics:")
        for item in self.check_activations():
            print(f"  {item}")
        print("\nGradient diagnostics:")
        for item in self.check_gradients():
            print(f"  {item}")
        print("\nPer-layer activation stats:")
        for name, stats in self.activation_stats.items():
            print(f"  {name}: mean={stats['mean']:.4f} std={stats['std']:.4f} zero={stats['fraction_zero']:.1%}")
        print("\nPer-layer gradient stats:")
        for name, stats in self.gradient_stats.items():
            print(f"  {name}: abs_mean={stats['abs_mean']:.2e} max={stats['max']:.2e}")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
```

### 第2步：单批次过拟合测试

```python
def overfit_one_batch(model, x_batch, y_batch, criterion, lr=0.01, steps=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    print("\n=== OVERFIT ONE BATCH TEST ===")
    print(f"Batch size: {x_batch.shape[0]}, Steps: {steps}")

    for step in range(steps):
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                preds = (output > 0).float() if output.shape[-1] == 1 else output.argmax(dim=1)
                targets = y_batch if y_batch.dim() == 1 else y_batch.squeeze()
                acc = (preds.squeeze() == targets).float().mean().item()
            print(f"  Step {step:3d} | Loss: {loss.item():.6f} | Accuracy: {acc:.1%}")

    final_loss = loss.item()
    if final_loss > 0.1:
        print(f"\n  FAIL: Loss did not converge ({final_loss:.4f}). Model or training loop is broken.")
        return False
    print(f"\n  PASS: Loss converged to {final_loss:.6f}")
    return True
```

### 第3步：学习率查找器

```python
def find_learning_rate(model, x_data, y_data, criterion, start_lr=1e-7, end_lr=10, steps=100):
    import copy
    original_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr)
    lr_mult = (end_lr / start_lr) ** (1 / steps)

    model.train()
    results = []
    best_loss = float("inf")
    current_lr = start_lr

    print("\n=== LEARNING RATE FINDER ===")

    for step in range(steps):
        optimizer.zero_grad()
        output = model(x_data)
        loss = criterion(output, y_data)

        if math.isnan(loss.item()) or loss.item() > best_loss * 10:
            break

        best_loss = min(best_loss, loss.item())
        results.append((current_lr, loss.item()))

        loss.backward()
        optimizer.step()

        current_lr *= lr_mult
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

    model.load_state_dict(original_state)

    if len(results) < 10:
        print("  Could not complete LR sweep -- loss diverged too quickly")
        return results

    min_loss_idx = min(range(len(results)), key=lambda i: results[i][1])
    suggested_lr = results[max(0, min_loss_idx - 10)][0]

    print(f"  Swept {len(results)} steps from {start_lr:.0e} to {results[-1][0]:.0e}")
    print(f"  Minimum loss {results[min_loss_idx][1]:.4f} at lr={results[min_loss_idx][0]:.2e}")
    print(f"  Suggested learning rate: {suggested_lr:.2e}")

    return results
```

### 第4步：梯度检查器

```python
def _flat_to_multi_index(flat_idx, shape):
    multi_idx = []
    remaining = flat_idx
    for dim in reversed(shape):
        multi_idx.insert(0, remaining % dim)
        remaining //= dim
    return tuple(multi_idx)


def gradient_check(model, x, y, criterion, eps=1e-4):
    model.train()
    x_double = x.double()
    y_double = y.double()
    model_double = model.double()

    print("\n=== GRADIENT CHECK ===")
    overall_max_diff = 0
    checked = 0

    for name, param in model_double.named_parameters():
        if not param.requires_grad:
            continue

        layer_max_diff = 0

        model_double.zero_grad()
        output = model_double(x_double)
        loss = criterion(output, y_double)
        loss.backward()
        analytical_grad = param.grad.clone()

        num_checks = min(5, param.numel())
        for i in range(num_checks):
            idx = _flat_to_multi_index(i, param.shape)
            original = param.data[idx].item()

            param.data[idx] = original + eps
            with torch.no_grad():
                loss_plus = criterion(model_double(x_double), y_double).item()

            param.data[idx] = original - eps
            with torch.no_grad():
                loss_minus = criterion(model_double(x_double), y_double).item()

            param.data[idx] = original

            numerical = (loss_plus - loss_minus) / (2 * eps)
            analytical = analytical_grad[idx].item()

            denom = max(abs(numerical), abs(analytical), 1e-8)
            rel_diff = abs(numerical - analytical) / denom

            layer_max_diff = max(layer_max_diff, rel_diff)
            checked += 1

        overall_max_diff = max(overall_max_diff, layer_max_diff)
        status = "OK" if layer_max_diff < 1e-5 else "MISMATCH"
        print(f"  {name}: max_rel_diff={layer_max_diff:.2e} [{status}]")

    model.float()

    print(f"\n  Checked {checked} parameters")
    if overall_max_diff < 1e-5:
        print("  PASS: Gradients match (rel_diff < 1e-5)")
    elif overall_max_diff < 1e-3:
        print("  WARN: Small differences (1e-5 < rel_diff < 1e-3)")
    else:
        print("  FAIL: Gradient mismatch detected (rel_diff > 1e-3)")
    return overall_max_diff
```

### 第5步：故意破坏的网络

现在将工具包应用于被破坏的网络，并诊断每个网络。

```python
def demo_broken_networks():
    torch.manual_seed(42)
    x = torch.randn(64, 10)
    y = (x[:, 0] > 0).long()

    print("\n" + "=" * 60)
    print("BUG 1: Learning rate too high (lr=10)")
    print("=" * 60)
    model1 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    debugger1 = NetworkDebugger(model1)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=10.0)
    criterion = nn.CrossEntropyLoss()
    for step in range(20):
        optimizer1.zero_grad()
        out = model1(x)
        loss = criterion(out, y)
        debugger1.record_loss(loss.item())
        loss.backward()
        optimizer1.step()
    debugger1.print_report()
    debugger1.remove_hooks()

    print("\n" + "=" * 60)
    print("BUG 2: Dead ReLUs from bad initialization")
    print("=" * 60)
    model2 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
    with torch.no_grad():
        for m in model2.modules():
            if isinstance(m, nn.Linear):
                m.weight.fill_(-1.0)
                m.bias.fill_(-5.0)
    debugger2 = NetworkDebugger(model2)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    for step in range(50):
        optimizer2.zero_grad()
        out = model2(x)
        loss = criterion(out, y)
        debugger2.record_loss(loss.item())
        loss.backward()
        optimizer2.step()
    debugger2.print_report()
    debugger2.remove_hooks()

    print("\n" + "=" * 60)
    print("BUG 3: Missing zero_grad (gradients accumulate)")
    print("=" * 60)
    model3 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    debugger3 = NetworkDebugger(model3)
    optimizer3 = torch.optim.SGD(model3.parameters(), lr=0.01)
    for step in range(50):
        out = model3(x)
        loss = criterion(out, y)
        debugger3.record_loss(loss.item())
        loss.backward()
        optimizer3.step()
    debugger3.print_report()
    debugger3.remove_hooks()

    print("\n" + "=" * 60)
    print("HEALTHY NETWORK: Correct setup for comparison")
    print("=" * 60)
    model_good = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    debugger_good = NetworkDebugger(model_good)
    optimizer_good = torch.optim.Adam(model_good.parameters(), lr=1e-3)
    for step in range(50):
        optimizer_good.zero_grad()
        out = model_good(x)
        loss = criterion(out, y)
        debugger_good.record_loss(loss.item())
        loss.backward()
        optimizer_good.step()
    debugger_good.print_report()
    debugger_good.remove_hooks()

    print("\n" + "=" * 60)
    print("OVERFIT-ONE-BATCH TEST (healthy model)")
    print("=" * 60)
    model_test = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    overfit_one_batch(model_test, x[:8], y[:8], criterion)

    print("\n" + "=" * 60)
    print("LEARNING RATE FINDER")
    print("=" * 60)
    model_lr = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    find_learning_rate(model_lr, x, y, criterion)

    print("\n" + "=" * 60)
    print("GRADIENT CHECK")
    print("=" * 60)
    model_grad = nn.Sequential(nn.Linear(10, 8), nn.ReLU(), nn.Linear(8, 2))
    gradient_check(model_grad, x[:4], y[:4], criterion)
```

## 使用它

### PyTorch内置工具

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

with torch.autograd.detect_anomaly():
    output = model(input_tensor)
    loss = criterion(output, target)
    loss.backward()

for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_mean={param.grad.abs().mean():.2e}")
```

### Weights & Biases集成

```python
import wandb

wandb.init(project="debug-training")

for epoch in range(100):
    loss = train_one_epoch()
    wandb.log({
        "loss": loss,
        "lr": optimizer.param_groups[0]["lr"],
        "grad_norm": torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")),
    })

    for name, param in model.named_parameters():
        if param.grad is not None:
            wandb.log({f"grad/{name}": wandb.Histogram(param.grad.cpu().numpy())})
```

### TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/debug_experiment")

for epoch in range(100):
    loss = train_one_epoch()
    writer.add_scalar("Loss/train", loss, epoch)

    for name, param in model.named_parameters():
        writer.add_histogram(f"weights/{name}", param, epoch)
        if param.grad is not None:
            writer.add_histogram(f"gradients/{name}", param.grad, epoch)
```

### 调试检查清单（在完整训练前）

1. 执行单批次过拟合测试。如果失败，停止。
2. 打印模型摘要——验证参数数量合理。
3. 使用随机数据执行一次前向传播——检查输出形状。
4. 训练5个epoch——验证损失下降。
5. 检查激活统计信息——无死亡层、无爆炸。
6. 检查梯度流——无消失、无爆炸。
7. 验证数据管道——打印5个随机样本及其标签。

## 发布

本課(lesson)产出：
- `outputs/prompt-nn-debugger.md` —— 诊断神经网络训练失败的提示
- `outputs/prompt-nn-debugger.md` —— 调试训练问题的决策树检查清单

调试的关键部署模式：
- 向生产训练脚本添加监控钩子
- 每N步将激活值和梯度统计信息记录到W&B或TensorBoard
- 实现NaN损失、死亡神经元（>80%为零）或梯度爆炸的自动告警
- 在更改架构或数据流水线时，始终执行过拟合单批次测试

## 练习

1. **添加梯度爆炸检测器。** 修改`NetworkDebugger`以检测梯度何时超过阈值，并自动建议梯度裁剪值。在一个没有归一化的20层网络上进行测试。

2. **构建死亡神经元复活器。** 编写一个函数，识别死亡的ReLU神经元（始终输出0），并使用Kaiming初始化重新初始化其输入权重。展示这可以恢复一个超过70%神经元死亡的网络。

3. **实现带绘图的学利率查找器。** 扩展`find_learning_rate`将结果保存为CSV，并编写一个单独的脚本读取CSV并使用matplotlib显示学习率对损失曲线。确定ResNet-18在CIFAR-10上的最优学习率。

4. **创建数据流水线验证器。** 编写一个函数检查：训练/测试拆分中的重复样本、标签分布不平衡（>10:1比率）、输入归一化（均值接近0，标准差接近1），以及数据中的NaN/Inf值。在一个故意损坏的数据集上运行它。

5. **调试一个真实故障。** 获取第10课的小框架，引入一个微妙的错误（例如在反向传播中转置权重矩阵），并使用梯度检查精确定位哪个参数有错误的梯度。记录调试过程。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|----------------------|
|  静默错误  |  "能运行但结果差"  |  一种不产生错误但降低模型质量的错误——机器学习中的主要故障模式  |
|  死亡ReLU  |  "神经元死了"  |  输入始终为负的ReLU神经元，因此输出0并永久接收0梯度  |
|  梯度消失  |  "早期层停止学习"  |  梯度在层间指数级缩小，导致早期层的权重实际上被冻结  |
|  梯度爆炸  |  "损失变为NaN"  |  梯度在层间指数级增长，导致权重更新过大而溢出  |
|  梯度检查  |  "验证反向传播正确"  |  将反向传播的分析梯度与有限差分的数值梯度进行比较  |
|  过拟合单批次  |  "最重要的调试测试"  |  在单个小批次上训练以验证模型能够学习——如果不能，则存在根本性问题  |
|  学习率查找器  |  "扫描以找到合适的学习率"  |  在一个epoch内指数级增加学习率，并选择损失发散前的学习率  |
|  数据泄露  |  "测试数据泄露到训练中"  |  当测试集的信息污染训练时，产生人为的高准确率  |
|  激活统计  |  "监测层健康状况"  |  跟踪每层输出的均值、标准差和零分数，以检测死亡、饱和或爆炸的神经元  |
|  梯度裁剪  |  "限制梯度幅度"  |  当梯度范数超过阈值时按比例缩小梯度，防止梯度爆炸更新  |

## 延伸阅读

- Smith, "Cyclical Learning Rates for Training Neural Networks" (2017) —— 提出学习率范围测试（学习率查找器）的论文
- Northcutt et al., "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks" (2021) —— 表明ImageNet、CIFAR-10等主要基准中有3-6%的标签是错误的
- Zhang et al., "Understanding Deep Learning Requires Rethinking Generalization" (2017) —— 展示神经网络可以记住随机标签的论文，这就是过拟合单批次测试有效的原因
- PyTorch关于`torch.autograd.detect_anomaly`和`torch.autograd.set_detect_anomaly`的文档，用于内置NaN/Inf检测
