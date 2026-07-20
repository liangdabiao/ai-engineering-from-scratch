# GPU设置与云

> 在CPU上训练适合学习。真正的训练需要GPU。

**类型:** 构建
**语言:** Python
**先修条件:** 阶段0，第01课
**时间:** ~45分钟

## 学习目标

- 使用`nvidia-smi`和PyTorch的CUDA API验证本地GPU可用性
- 配置Google Colab并使用T4 GPU进行免费的云端实验
- 对比CPU和GPU上的矩阵乘法性能，测量加速比
- 使用fp16经验法则估算适合你显存的最大模型

## 问题

阶段1-3的大多数课程在CPU上运行良好。但一旦开始训练CNN、Transformer或LLM（阶段4以后），就需要GPU加速。在CPU上需要8小时的训练，在GPU上仅需10分钟。

你有三个选择：本地GPU、云端GPU或Google Colab（免费）。

## 核心概念

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

## 动手构建

### 选项1：本地NVIDIA GPU

检查你是否拥有：

```bash
nvidia-smi
```

安装带CUDA的PyTorch：

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 选项2：Google Colab

1. 前往[colab.research.google.com](https://colab.research.google.com)
2. 运行环境 > 更改运行时类型 > T4 GPU
3. 运行[colab.research.google.com](https://colab.research.google.com)以验证

将本课程的笔记本直接上传到Colab。

### 选项3：云GPU

对于Lambda Labs、RunPod或Vast.ai：

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 没有GPU？没问题。

大多数课程在CPU上即可运行。需要GPU的课程会特别说明，并附上Colab链接。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## 动手实践：GPU vs CPU基准测试

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## 练习

1. 运行上述基准测试，比较CPU与GPU的时间
2. 如果没有GPU，请在Google Colab上运行并比较
3. 检查你的GPU内存大小，估算可容纳的最大模型（经验法则：fp16每个参数占2字节）

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|----------------------|
|  CUDA  |  "GPU programming"  |  NVIDIA的并行计算平台，让你能在GPU上运行代码  |
|  VRAM  |  "GPU memory"  |  显存（Video RAM），位于GPU上，与系统内存分离。限制模型大小。  |
|  fp16  |  "Half precision"  |  16位浮点数，使用fp32一半的内存，精度损失最小  |
|  Tensor Core  |  "Fast matrix hardware"  |  专用于矩阵乘法的GPU核心，比普通核心快4-8倍  |
