# 梯度检查点(Gradient Checkpointing)与激活重计算(Activation Recomputation)

> 反向传播会保留每个中间激活值。对于70B参数和128K上下文，每个rank的激活值高达3 TB。检查点以FLOPs换内存：重计算而不是保存。问题在于丢弃哪些片段，答案并非"全部丢弃"。

**类型：** 构建
**语言：** Python (使用numpy, 可选torch)
**前置条件：** 阶段10课程04 (预训练迷你GPT), 阶段10课程05 (扩展与分布式)
**时间：** 约70分钟

## 问题

训练Transformer时，对于每一层，会存储反向传播中每个需要微分的操作的输入：注意力输入、Q/K/V投影、softmax输出、FFN输入、归一化输出和残差流。对于隐藏大小为`d`、序列长度为`L`、批大小为`B`的层，每层约有`12 * B * L * d`个浮点数。

对于`d=8192, L=8192, B=1`，在BF16下每层为800 MB。64层模型激活值达51 GB——这还没乘以微批大小，还没加上注意力softmax中间值（每头`L^2`），还没考虑张量并行的部分副本。

双重账单：BF16权重加上优化器状态可能适合80GB，但激活值会超出。梯度检查点（又称激活重计算）是标准解决方案。丢弃大部分激活值；在反向传播期间重新执行前向传播以恢复它们。代价：额外FLOPs。收益：内存减少比例取决于检查点段数与总层数之比。

朴素地做检查点，每一步大约增加33%的前向传播FLOPs。做得好——根据Korthikanti等人的"智能选择"进行选择性检查点——在不到5%的FLOP开销下节省5倍内存。而使用FP8矩阵乘法、FSDP卸载和专家并行MoE，这一点至关重要：你既承受不起内存开销，也承受不起计算浪费。

## 核心概念

### 反向传播实际需要什么

`output = layer(input)`。反向传播需要`grad_input`和`grad_params`。要计算它们需要：

- `input`（用于计算线性层的`grad_params = input.T @ grad_output`）
- 一些激活值的导数中间值（ReLU/GELU/softmax的导数依赖于激活值）

前向传播会在自动求导图中自动存储这些。每个`tensor.retain_grad()`和每个需要其输入的操作都会保留一个引用。

### 朴素完整检查点

将网络分成`N`个段。在前向传播时，只存储每个段的*输入*。当反向传播需要中间值时，重新运行该段的前向传播以生成它们，然后进行微分。

示例：32层Transformer分成32个段，每段1层。

- 内存：32个层输入（小） vs 32 * (每层激活体积) (巨大)。
- 额外计算：每段多一次前向传播，即总共约33%更多的前向FLOPs（因为反向传播是前向的2倍，完整步骤变为1+1+2=4个单位，而不是1+2=3）。

这是Chen等人2016年的原始方案：每`sqrt(L)`层设置一个检查点以平衡内存和计算。对于L=64，那就是8个检查点。

### 选择性检查点 (Korthikanti 2022)

并非所有激活值成本相同。注意力softmax输出是`B*L*L*heads`，且随序列长度*二次方*增长。FFN隐藏激活值是`B*L*4d`，线性增长。对于长序列，softmax占主导。

选择性检查点保留存储成本低的激活值（线性投影、残差），只重计算成本高的激活值（注意力）。你付出最小的FLOPs来重计算，但节省了O(L^2)的内存。

Megatron-Core将其实现为"选择性"激活重计算。用于大多数2024+前沿训练运行。

### 卸载(Offload)

重计算的替代方案：在前向和反向之间将激活值传输到CPU RAM。需要PCIe带宽；当空闲带宽超过重计算成本时有利。混合策略常见：对某些层检查点，卸载其他层。

FSDP2将卸载作为一等选项提供。当GPU内存是瓶颈但CPU-GPU传输有余量时，卸载表现出色。

### 重计算成本模型

采用朴素检查点每`k`层（共`L`层）时每步的FLOPs：

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

使用选择性检查点时，只重计算注意力核，而不是整个层：

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 内存节省模型

每层激活体积：`A`。对于`L`层，总激活内存：`L * A`。

完整检查点（段大小为1）：只存储`L * input_volume`（对于标准Transformer约为`L * 1/10 A`）。节省约`9 * L * A * 1/10`。

每`k`层设置检查点：存储`L/k * A`加上活动段内`k-1`层的值。

在`k = sqrt(L)`处，内存和重计算成本均随`sqrt(L)`缩放——这是均匀成本层的最优权衡。

### 何时不应使用检查点

- 管道阶段最内层的层已经在执行中。它们无论如何都必须完成。
- 如果第一层和最后一层主导了阶段的计算（在Transformer中很少见）。
- 已经在使用FlashAttention的注意力核——Flash已经快速重计算了softmax，因此额外的层级别检查点带来的收益很小。

### 实现模式

1. **函数包装器：** 将一段包装在`torch.utils.checkpoint.checkpoint(fn, input)`中。PyTorch仅存储`input`，在反向传播时重计算所有其他内容。

2. **装饰器方法：** 将层标记为可检查点；训练器在配置时决定哪些段被包装。

3. **手动显式重计算：** 自己编写反向传播，调用自定义的`recompute_forward`，使用存储的输入复制前向过程。

三者产生相同的功能结果。包装器是标准用法。

### 与张量并行/管道并行/FP8的交互

- **张量并行：** 检查点输入必须在重计算时收集或重新分发；处理通信开销。
- **管道并行：** 典型模式是检查每个管道阶段的前向过程，以便反向顺序的微批可以重用激活内存。
- **FP8重计算：** 重计算期间更新的amax历史必须与原始前向匹配，否则FP8缩放会漂移。大多数框架会快照缩放。

## 动手构建

### 第1步：带段的玩具模型

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 第2步：需要所有激活的朴素反向传播

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### 第3步：每k个检查点的内存

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### 第4步：成本模型

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### 第5步：内存估计器

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 第6步：最优段大小

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### 第7步：选择性检查点决策

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 使用它

- **torch.utils.checkpoint**：`from torch.utils.checkpoint import checkpoint`——PyTorch中的标准包装器。包装函数，仅存储输入，反向传播时重计算。
- **Megatron-Core激活重计算**：支持`from torch.utils.checkpoint import checkpoint`、`selective`和`full`模式。在2024+前沿训练中是标准配置。
- **FSDP2卸载**：`from torch.utils.checkpoint import checkpoint`与FSDP2中的`selective`将激活分片到CPU，而不是重计算。
- **DeepSpeed ZeRO-Offload**：将优化器状态和激活卸载到CPU，补充检查点。

## 发布

本课生成`outputs/prompt-activation-recompute-policy.md`——一个提示，接受你的模型配置（层数、隐藏维度、序列长度、批量大小）和可用GPU内存，并输出逐层重计算策略（无/选择性/完全/卸载）。

## 练习

1. 验证正确性。运行`model_forward` + `model_backward`（全激活）与`model_forward_checkpointed` + `model_backward_checkpointed`（段）进行比较。参数梯度必须与机器精度一致。

2. 扫描段大小`k`从1到`L`。绘制FLOP开销和内存图。找到曲线的拐点。

3. 实现选择性检查点：存储注意力模块的输入，但不存储其中间结果。在序列长度8192的32层模型上测量FLOP开销与全层检查点的对比。

4. 添加卸载。将段输入保存到模拟的“CPU缓冲区”（一个单独的列表）。以字节/时间测量“PCIe带宽”，并找到卸载和重计算之间的平衡点。

5. 对有和没有`torch.utils.checkpoint`的真实PyTorch Transformer进行基准测试。测量内存（通过`torch.cuda.max_memory_allocated`）和步进时间。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|----------------------|
|  梯度检查点  |  "通过重做前向来节省内存"  |  仅存储段输入；在反向传播期间重计算中间结果以获取梯度支持张量  |
|  激活重计算  |  "与检查点相同"  |  同一技术的HPC风格名称  |
|  段大小 (k)  |  "每个检查点包含多少层"  |  其中间结果被丢弃并一起重新物化的层数  |
|  选择性检查点  |  "Korthikanti的技巧"  |  仅重计算存储成本高的激活（注意力softmax）；保留成本低的  |
|  完全检查点  |  "朴素版本"  |  在每个段中重计算每一层的中间结果  |
|  块检查点  |  "粗粒度"  |  检查整个Transformer块；最大粒度  |
|  FLOP开销  |  "计算税"  |  每步额外FLOPs = (重计算FLOPs) / (前向+后向FLOPs); 朴素33%, 选择性5%  |
|  激活卸载  |  "卸载到CPU"  |  在前向->后向过程中将激活移至CPU RAM; 重计算的替代方案  |
|  sqrt-L规则  |  "经典最优"  |  对于均匀成本层，最优检查点间距为sqrt(L)层  |
|  注意力-softmax体积  |  "O(L^2)问题"  |  L^2 * 头数 * 批次浮点数; 在长上下文下主导激活内存  |

## 延伸阅读

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- 规范化梯度检查点的原始论文
- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- 选择性激活重计算及形式化成本分析
- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- 通过反向模式重物化的恒定内存替代方法
- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- 大规模的激活卸载
- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- 标准API
- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- 选择性、完整和分块模式
