# vLLM 服务内部机制：PagedAttention、连续批处理、分块预填充

> vLLM在2026年的主导地位依赖于三个复合默认配置，而非单一技巧。PagedAttention始终开启。连续批处理(Continuous batching)在解码迭代之间将新请求注入活跃批次。分块预填充(Chunked prefill)将长提示切分成切片，确保解码令牌永不饥饿。同时开启这三项，Llama 3.3 70B FP8在单个H100 SXM5上，128并发时可达2200-2400 tok/s——比vLLM自身默认配置高出约25%，是朴素PyTorch循环的3-4倍。本课程从可图示的层面解读调度器和注意力内核，并以一个用`code/main.py`实现的玩具连续批处理器结束，该处理器按照vLLM的方式调度预填充和解码。

**类型：** 学习
**语言：** Python（标准库，玩具连续批处理调度器）
**前置要求：** 阶段17·01（模型服务），阶段11（大语言模型工程）
**时间：** 约75分钟

## 学习目标

- 解释PagedAttention作为KV缓存分配器：块、块表，以及为什么在生产负载下碎片率保持在4%以下。
- 在迭代级别上绘制连续批处理图：已完成序列如何离开批次，新序列如何加入而无需排空。
- 用一句话描述分块预填充，并指出它保护了哪个延迟指标（提示：是TTFT尾部，而非平均吞吐量）。
- 指出2026年vLLM v0.18.0中团队同时启用所有优化时容易踩的坑。

## 问题

一个朴素的PyTorch服务循环一次处理一个请求：分词、预填充、解码直到EOS，返回。一个用户时没问题。一百个用户时，就是耐心的排队。明显的修复——静态批处理——将每个请求填充到窗口中最长提示的长度，将每个解码填充到最长预期输出的长度，并且整个批次会等待最慢的序列。你为从未使用的填充付费，快请求等待慢请求。

vLLM同时解决三个问题。PagedAttention阻止KV缓存碎片像传统连续分配那样消耗60-80%的GPU内存。连续批处理允许请求在每个解码迭代之间加入和离开批次，这样批次始终充满实际工作。分块预填充将32k令牌的提示分解为约512令牌的切片，与解码交错，这样长提示不会冻结GPU上的每个解码令牌。

2026年的生产默认配置是三者全开。你需要理解每个配置的作用，因为失败模式都在调度器上，而非模型上。

## 核心概念

### PagedAttention作为虚拟内存系统

每个序列的KV缓存是`num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element`。对于Llama 3.3 70B，8192令牌时，每个序列在BF16下大约需要1.25 GB。如果你为每个请求预分配8192个槽位，但平均请求只使用1500个令牌，那么你浪费了大约82%的预留HBM。传统批处理承担这种浪费。

PagedAttention借鉴了操作系统虚拟内存的思想。KV缓存不再是每个序列连续分配。它被分配为固定大小的块（默认16个令牌）。每个序列有一个块表，将其逻辑令牌位置映射到物理块ID。当序列增长超过已分配块时，再添加一个块。当序列结束时，其块归还到池中。

碎片率从60-80%（传统）降至4%以下（PagedAttention）。你不需要通过标志启用PagedAttention——它是vLLM唯一附带的分配器。可调参数是`--gpu-memory-utilization`（默认0.9），它告诉vLLM在加载权重和激活后为KV块保留多少HBM。

### 迭代级别的连续批处理

旧的“动态批处理”等待一个窗口（例如10毫秒）来填满批次，然后运行预填充+解码+解码+解码直到每个序列完成。快序列提前离开并闲置，而GPU完成慢序列。

连续批处理在每个解码步骤之间操作。将正在运行的序列集称为`RUNNING`列表。每次迭代：

1. 任何刚刚达到EOS或max_tokens的`RUNNING`中的序列被移除。
2. 调度器查看等待队列。如果有空闲KV块，它接纳新序列（预填充或恢复）。
3. 前向传播在当前`RUNNING`中的任何序列上运行，每个序列产生一个新令牌。

批次大小从不填充到固定数量。不同输出位置的序列共享一个融合的前向传播。在2026年的vLLM中，这被称为`V1 scheduler`。关键不变性：调度器每次解码迭代运行一次，而不是每个请求一次。

### 分块预填充保护TTFT尾部

预填充是计算密集的。Llama 3.3 70B上32k令牌的提示在单个H100上需要大约800毫秒的纯预填充。预填充运行时，批次中其他每个序列的解码令牌等待。在服务循环中，一个长提示的首令牌延迟(TTFT)成为其他几十个用户的令牌间延迟(ITL)毛刺。

分块预填充将预填充拆分为固定大小的块（默认512令牌），并将每个块作为一个单元调度。块之间，调度器可以推进解码序列一个令牌。你以小的绝对预填充延迟损失（每个块几毫秒）换取低得多的解码时抖动。在已发布基准测试中，混合负载下的P99 ITL从约50毫秒降至约15毫秒。

### 三个默认配置的交互

所有三个特性相互依赖。PagedAttention为调度器提供了精细的KV资源进行权衡。连续批处理需要这种精细资源，以便接纳新序列不会强制全局重排。分块预填充是调度器在同一个`RUNNING`列表上做出的决策——它只是另一个调度策略，而非独立系统。

你不需要知道每个标志。你需要知道调度器优化什么：在KV块预算下，受分块预填充切片约束的有效吞吐量。

### 2026年v0.18.0的坑

在vLLM v0.18.0中，你不能将`--enable-chunked-prefill`与草稿模型推测解码(`--speculative-model`)结合使用。文档说明的例外是V1调度器中的N-gram GPU推测解码。没有阅读发布说明就打开所有标志的团队会在启动时遇到运行时错误，而非软退化。如果你的推测收益值得启用分块预填充，请重新审视选择——2026年的正确答案通常是无分块预填充的EAGLE-3，而不是无法编译的草稿模型加分块预填充。

### 你应该记住的数字

- Llama 3.3 70B FP8，H100 SXM5，128并发，三者全开：2200-2400 tok/s。
- 相同模型，默认vLLM（无分块预填充）：约1800 tok/s。
- 相同模型，朴素PyTorch前向循环：约600 tok/s。
- 生产负载下PagedAttention的KV碎片浪费：<4%。
- 混合负载下P99 ITL：分块预填充时约15 ms，无分块预填充时约50 ms。

### 调度器看起来什么样

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # schedule prefill chunks + decode in one batch
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # e.g. 512 tokens
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # one fused GPU call
```

`code/main.py`正是这个使用标准库Python的循环，带有假令牌计数和假前向延迟。运行它显示分块预填充如何在长预填充期间保持解码序列存活。

```figure
tensor-parallel
```

## 使用它

`code/main.py`模拟了具有可切换特性的vLLM风格调度器。运行它可以看到：

- `NAIVE`模式：一次一个请求，无批处理。
- `NAIVE`模式：填充并等待，传统批处理。
- `NAIVE`模式：迭代级别接纳和释放。
- `NAIVE`模式：预填充切片与解码交错。

输出显示总吞吐量（每虚拟秒令牌数）、TTFT均值和P99 ITL。`CONTINUOUS + CHUNKED`行应在混合流量下占优。

## 发布

本课程产生`outputs/skill-vllm-scheduler-reader.md`。给定服务配置（批次大小、KV内存利用率、分块预填充大小、推测配置），它生成一个调度器诊断，指出三个默认配置中哪个是瓶颈以及如何调整。

## 练习

1. 运行 `code/main.py`。在混合短请求和长请求的工作负载下，将 `STATIC` 与 `CONTINUOUS` 进行比较。吞吐量差距来自哪里——预填充效率、解码效率还是尾部延迟？
2. 修改玩具调度器以添加 `code/main.py`。对于运行 Llama 3.3 70B FP8 的 H100，正确的值是多少？（提示：它是 KV 块大小和空闲块数量的函数，而不是原始 HBM。）
3. 重新阅读 vLLM v0.18.0 的发布说明。哪些标志组合是互斥的？列出它们。
4. 计算 1000 个请求（平均 1500 个输出 token，标准差 600 token）的 KV 缓存碎片浪费，在 (a) 连续按请求分配，最大 8192 和 (b) 使用 16 token 块的 PagedAttention 下。
5. 用一段解释为什么分块预填充独立有助于 P99 ITL 但不会提高吞吐量。在实践中，吞吐量的提升来自哪里？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  PagedAttention  |  "KV 技巧"  |  固定大小的 KV 缓存块分配器；碎片 <4%  |
|  块表  |  "页表"  |  从逻辑 token 位置到物理 KV 块的逐序列映射  |
|  连续批处理  |  "动态批处理，但正确"  |  每次解码迭代中做出接纳/释放决策  |
|  分块预填充  |  "预填充拆分"  |  将长预填充分解为 512 token 的切片，与解码交错执行  |
|  TTFT  |  "首个 token 时间"  |  预填充 + 队列 + 网络；在长提示下主要由预填充主导  |
|  ITL  |  "token 间延迟"  |  连续解码 token 之间的时间；主要由批大小主导  |
|  Goodput  |  "满足 SLO 的吞吐量"  |  每秒 token 数，且每个请求仍能达到 TTFT 和 ITL 目标  |
|  V1 调度器  |  "新调度器"  |  vLLM 的 2026 年调度器；N-gram 推测解码是与分块预填充兼容的路径  |
|  `--gpu-memory-utilization`  |  "内存旋钮"  |  在权重和激活之后为 KV 块预留的 HBM 比例  |

## 延伸阅读

- [vLLM documentation — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) — 关于分块预填充和推测解码兼容性的官方来源。
- [vLLM documentation — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) — 2026 年发布节奏和版本特定行为。
- [vLLM documentation — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) — 仍然定义如何思考分配器的原始文章。
- [vLLM documentation — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) — 碎片分析及调度器设计。
- [vLLM documentation — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) — 带有火焰图的详细 V1 调度器演练。
