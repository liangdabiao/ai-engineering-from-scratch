# KV缓存、Flash Attention与推理优化

> 训练是并行的且受FLOP限制。推理是串行的且受内存限制。不同的瓶颈，不同的技巧。

**类型:** 构建
**语言:** Python
**先修知识:** 阶段7 · 02（自注意力），阶段7 · 05（完整Transformer），阶段7 · 07（GPT）
**时间:** ~75分钟

## 问题

一个朴素的自回归解码器需要做`O(N²)`次工作来生成`N`个token：每一步它都会重新计算整个前缀的注意力。对于一个4K token的响应，这是1600万次注意力操作，其中大部分是冗余的。前缀token的每个隐藏状态一旦计算出来就是确定的——你只需要用新token的查询对之前所有内容的缓存键和值进行运算。

此外，注意力本身会移动大量数据。标准注意力会实例化一个N×N的分数矩阵、N×d的softmax输出、N×d的最终输出——对HBM的读写太多。当N≥2K时，注意力在达到FLOP限制之前就已经受内存限制了。经典的注意力内核在4-10倍的性能差距下未能充分利用现代GPU。

两项优化（均来自Dao等人）将前沿推理从"慢"推向了"快"：

1. **KV缓存。** 存储每个前缀token的K和V向量。每个新token的注意力只是一个对缓存键的查询。推理从`O(N²)`减少到`O(N)`每步生成。
2. **Flash Attention。** 分块计算注意力，使得完整的N×N矩阵永远不会位于HBM中。所有的softmax和矩阵乘法都在SRAM中完成。在A100上加速2-4倍；在H100上使用FP8加速5-10倍。

到2026年，这两项技术都将普及。每个生产推理栈（vLLM、TensorRT-LLM、SGLang、llama.cpp）都假设采用它们。每个前沿模型都启用了Flash Attention。

## 核心概念

![KV cache growth and Flash Attention tiling](../assets/kv-cache-flash-attn.svg)

### KV缓存数学

每个解码器层、每个token、每个头：

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K and V
```

对于一个7B模型，32层，32个头，d_head=128，fp16：

```
per token per layer = 2 * 128 * 2 = 512 bytes
per token (32 layers) = 16 KB
per 32K context = 512 MB
```

对于Llama 3 70B（80层，d_head=128，GQA有8个KV头）：

```
per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KB)
per 32K context = 10.4 GB
```

这10 GB就是为什么Llama 3 70B在128K上下文下，批大小为1时，仅KV缓存就需要大部分40 GB的A100内存。

**GQA是KV缓存的胜利。** 64个头的MHA将是32 GB。MLA进一步压缩。

拖动维度，观察缓存大小的变化。增加序列长度或批量大小，看看它如何迅速超过单个GPU的限制：

```figure
kv-cache-sizer
```

### Flash Attention —— 分块技巧

标准注意力：

```
S = Q @ K^T          (HBM read, N×N, HBM write)
P = softmax(S)       (HBM read, HBM write)
O = P @ V            (HBM read, HBM write)
```

三次HBM往返。在H100上，HBM带宽为3 TB/s；SRAM为30 TB/s。每次HBM往返比将所有数据保留在片上慢一个数量级。

Flash Attention：

```
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

每个分块一次HBM往返。总内存占用从`O(N²)`减少到`O(N)`。反向传播从正向传播中重计算一些值，而不是存储它们——又一个内存优势。

**数值技巧。** 运行softmax在分块之间维护`(max, sum)`，因此最终归一化是精确的。不是近似——Flash Attention计算与标准注意力比特相同（除了fp16非结合性）。

**版本演进：**

|  版本  |  年份  |  关键变化  |  在参考硬件上的加速比  |
|---------|------|-----------|-------------------------------|
|  Flash 1  |  2022  |  分块SRAM内核  |  在A100上2倍  |
|  Flash 2  |  2023  |  更好的并行性，因果优先排序  |  在A100上3倍  |
|  Flash 3  |  2024  |  Hopper异步性，FP8  |  在H100上1.5-2倍（约740 TFLOPs FP16）  |
|  Flash 4  |  2026  |  Blackwell 5级流水线，软件exp2  |  推理优先（初始仅前向）  |

Flash 4在发布时仅支持前向传播。训练仍使用Flash 3。Flash 4的GQA和varlen支持待定（2026年中）。

### 推测性解码 —— 另一个延迟优势

廉价模型提出N个token。大型模型并行验证所有N个token。如果验证接受了k个token，那么你为k个生成支付了一次大型模型前向传播。在代码和散文上典型的k=3-5。

2026年默认值：
- **EAGLE 2 / Medusa。** 集成草稿头，共享验证器的隐藏状态。速度提升2-3倍，质量无损。
- **基于草稿模型的推测解码。** 在消费级硬件上速度提升2-4倍。
- **先行解码。** 雅可比迭代，无需草稿模型。小众但免费。

### 持续批处理

经典批推理：等待最慢的序列完成，然后启动新批次。短响应提前完成时会浪费GPU。

持续批处理（首次在Orca中推出，现用于vLLM、TensorRT-LLM、SGLang）：一旦旧请求完成，立即将新请求交换到批次中。对于典型聊天工作负载，吞吐量提升5-10倍。

### PagedAttention——KV缓存作为虚拟内存

vLLM的旗舰特性。KV缓存以16令牌块分配；页表将逻辑位置映射到物理块。允许在并行采样（束搜索、并行采样）之间共享KV，为提示缓存热交换前缀，以及整理内存碎片。相比朴素连续分配，吞吐量提升4倍。

```figure
flash-attention-memory
```

## 动手构建

参见`code/main.py`。我们实现了：

1. 一个朴素的`O(N²)`增量解码器。
2. 一个`O(N²)` KV缓存解码器。
3. 一个模拟Flash Attention运行最大值算法的分块softmax。

### 步骤1：KV缓存

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

简单：在每层、每头的列表中不断增长每个令牌的K、V向量。

### 步骤2：分块softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

一次性输出与`softmax(qK) V`逐位相同，但任何时候工作集都是一个`tile × d_head`块，而不是完整的`N × d_head`。

### 步骤3：比较朴素解码与缓存解码在生成100个令牌时的表现

统计注意力操作数。朴素：`O(N²)` = 5050。缓存：`O(N)` = 100。代码会打印两者。

## 使用它

```python
# HuggingFace transformers auto-enables KV cache on decoder-only generate().
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # use FA3 if Hopper
    torch_dtype="bfloat16",
)
# generate() uses KV cache automatically
```

vLLM生产环境：

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

跨请求的前缀缓存是2026年的一大胜利——相同的系统提示、少样本示例或长上下文文档在多次调用间复用KV。对于重复工具提示的智能体工作负载，前缀缓存通常带来5倍吞吐量提升。

## 发布

参见`outputs/skill-inference-optimizer.md`。该技能为新的推理部署选择注意力实现、KV缓存策略、量化和推测解码。

## 练习

1. **简单。** 运行`code/main.py`。确认朴素解码器和缓存解码器产生相同输出；注意操作数差异。
2. **中等。** 实现前缀缓存：给定提示P和多个补全，对P进行一次前向传播填充KV缓存，然后按补全分支。测量与为每个补全重新编码P相比的速度提升。
3. **困难。** 实现一个玩具版PagedAttention：固定16令牌块的KV缓存，带空闲列表。当一个序列完成时，将其块返回到池中。模拟1000个不同长度的聊天补全。比较内存碎片与连续分配。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  KV缓存  |  "让解码变快的技巧"  |  存储每个前缀令牌的K和V；新查询关注它们而不是重新计算。 |
|  HBM  |  "GPU主内存"  |  高带宽内存；H100上80 GB，B200上192 GB。约3 TB/s带宽。 |
|  SRAM  |  "片上内存"  |  每个SM的快速内存，H100上每个SM约256 KB。约30 TB/s带宽。 |
|  Flash Attention  |  "分块注意力核"  |  计算注意力时不将N×N矩阵具体化到HBM。 |
|  持续批处理  |  "无等待批处理"  |  交换完成的序列出去，新的进来，无需清空批次。 |
|  PagedAttention  |  "vLLM的旗舰特性"  |  使用页表在固定块中分配KV缓存；消除碎片。 |
|  前缀缓存  |  "复用长提示"  |  为跨请求的共享前缀缓存KV；大幅降低智能体成本。 |
|  推测解码  |  "草稿+验证"  |  廉价的草稿模型提出令牌；大模型一次性验证k个。 |

## 延伸阅读

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1。
- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 2。
- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 3。
- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Blackwell 5阶段流水线和software-exp2技巧；阅读仓库README了解本课提及的仅前向启动注意事项。
- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — vLLM论文。
- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — spec解码。
- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — EAGLE-1/2论文，关于本课引用的集成草稿方法。
- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — 与EAGLE一同引用的Medusa方法。
- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — 关于16令牌块和页表设计的权威深入分析。
