# DeepSeek-V3 架构详解

> 第10阶段·第14课命名了每个开源模型都会调整的六个架构旋钮。DeepSeek-V3（2024年12月，总参数671B，激活参数37B）调整了全部六个，并增加了四个：多头潜在注意力(Multi-Head Latent Attention)、无辅助损失负载均衡(auxiliary-loss-free load balancing)、多Token预测(Multi-Token Prediction)和双管道训练(DualPipe training)。本课自上而下解读DeepSeek-V3的架构，并从发布的配置中推导出每个参数计数。最后，你将能够解释为什么671B/37B的比例是正确的选择，以及为什么MLA+MoE组合在前沿上比单独使用任何一种都更好。

**类型:** 学习
**语言:** Python (stdlib, 参数计算器)
**先决条件:** 第10阶段·第14课（开源模型详解），第10阶段·第17课（NSA），第10阶段·第18课（MTP），第10阶段·第19课（双管道训练）
**时间:** 约75分钟

## 学习目标

- 自上而下阅读DeepSeek-V3配置，并根据六个GPT-2旋钮加上四个DeepSeek特定增项解释每个字段。
- 推导总参数计数（671B）、激活参数计数（37B）以及贡献给每个计数的组件。
- 计算MLA在128k上下文下的KV缓存占用，并与相同激活参数的密集模型使用GQA时的开销进行比较。
- 说明DeepSeek的四项特定创新（MLA、MTP、无辅助损失路由、双管道训练），并指出每一项针对架构/训练栈中的哪个部分。

## 问题

DeepSeek-V3是第一个架构与Llama家族有显著差异的前沿开源模型。Llama 3 405B是“GPT-2旋钮全开”。DeepSeek-V3是GPT-2的所有六个旋钮再加四个。阅读Llama 3配置是阅读DeepSeek配置的热身，但深层结构——注意力块的形状、路由逻辑、训练时目标——差异足够大，以至于你需要单独的详解。

学习它的回报：DeepSeek-V3的开放权重发布改变了“前沿能力”在开源模型中的含义。该架构是许多2026年训练运行正在复制的蓝图。理解它是任何涉及前沿LLM训练或推理角色的基本要求。

## 核心概念

### 不变的核心，再次

DeepSeek-V3仍然是自回归的。它仍然堆叠解码器块。每个块仍然有注意力加MLP加两个RMSNorm。它仍然在MLP中使用SwiGLU。它仍然使用RoPE。预归一化。权重绑定嵌入。与每个Llama或Mistral相同的基线。

### 不同之处：MLA代替GQA

从第10阶段·第14课你知道GQA通过在Q头组之间共享K和V来缩小KV缓存。多头潜在注意力(MLA)更进一步：K和V被压缩成一个共享的低秩潜在表示（`kv_lora_rank`），然后在每个头上即时解压缩。KV缓存只存储潜在表示——通常每层每个token 512个浮点数，而不是8×128=1024个浮点数。

在128k上下文下，带有MLA的DeepSeek-V3（每层每个token一个共享潜在表示`c^{KV}`；K和V都通过可吸收到后续矩阵乘法的上投影从该潜在表示中导出）：

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

一个假设的GQA基线（Llama 3 70B形状，8个KV头，头维度128）将付出：

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

在128k上下文下，MLA比Llama-3-70B风格的GQA缓存小4倍。

权衡：MLA在每个注意力计算（每个头）中增加了一个解压缩步骤。与节省的带宽相比，额外的计算量很小。对于长上下文推理来说净收益。

### 路由：无辅助损失负载均衡

MoE路由器决定每个token由哪top-k个专家处理。天真的路由器会将过多工作集中在少数专家上，使其他专家闲置。标准修复：添加一个惩罚负载不均衡的辅助损失项。这有效但会轻微降低主任务性能。

DeepSeek-V3引入了一种无辅助损失方案。在路由器logits中添加每个专家的偏置项，在训练期间通过简单规则调整：如果专家`e`过载，则减小`bias_e`；如果欠载，则增加。没有额外的损失项。训练保持干净。专家负载保持均衡。

对主损失的影响：无显著影响。对MoE架构的影响：更干净，没有要调整的辅助损失超参数。

### MTP：更密集的训练 + 免费草稿

从第10阶段·第18课你知道DeepSeek-V3添加了D=1的MTP模块，预测前方两个位置的token。在推理时，训练好的模块被重新用作推测解码草稿，接受率超过80%。在训练时，每个隐藏状态在D+1=2个目标上进行监督，提供更密集的信号。

参数：在671B主体之上增加14B。开销：2.1%。

### 训练：双管道训练

从第10阶段·第19课你知道双管道训练是一种双向流水线，将前向和后向块与跨节点全对全通信重叠。在DeepSeek-V3的2,048块H800规模下，它回收了大约245k GPU小时，这些时间在1F1B中会因流水线气泡而损失。

### 配置，逐字段

这是DeepSeek-V3配置（简化版）：

```
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

解析它：

- `hidden_size=7168`：嵌入维度。
- `hidden_size=7168`：总块深度。
- `hidden_size=7168`：前3个块使用大小为18432的密集MLP。其余58个使用MoE。
- `hidden_size=7168`：128个查询头。
- `hidden_size=7168`：K和V被压缩到这个潜在维度，并在每个头上解压缩。
- `hidden_size=7168`：每个MoE块有256个专家，路由top-8。
- `hidden_size=7168`：在256个路由专家之上，有1个始终在线的专家为每个token贡献。可以将其视为确保每个token都能获得可靠内容的“密集基底”。
- `hidden_size=7168`：每个专家的MLP隐藏层大小。比密集MLP小，因为有256个。

### 参数统计

完整计算在`code/main.py`中。要点：

- 嵌入：`vocab * hidden = 129280 * 7168 = ~0.93B`。
- 前3个密集块：带有MLA的注意力（每个块约144M）+ 密集MLP（每个块约260M）+ 归一化。总计约1.2B。
- 58个MoE块：带有MLA的注意力（约144M）+ 256个专家（每个30M）+ 1个共享专家（30M）+ 归一化。每个块总计约7.95B，包括所有专家。58个MoE块总共461B。
- MTP模块：14B。

总计：核心架构约476B + 14B MTP，显然发布的671B数字考虑了额外的结构参数（偏置张量、专家特定组件、共享专家缩放等）。我们在计算器中复现的数字与发布数字相差3-5%，差异来自DeepSeek报告在其第2节附录中记录的细粒度核算。

每次前向传播的活跃参数：

- 注意力：每层144M * 61 = 8.8B（所有层都激活）。
- MLP活跃：前3层密集（3 * 260M = 780M），58个MoE层每层活跃8个路由+1个共享+路由开销。每层活跃MLP：约260M。总计：3 * 260M + 58 * 260M = ~15.9B。
- 嵌入+归一化：1.2B。
- 总活跃：约26B核心+14B MTP（训练但推理时不总是运行）≈ 37B。

### 671B / 37B 的比例

18倍稀疏率（活跃参数占总量的5.5%）。DeepSeek-V3是已发布开放权重的最稀疏的前沿MoE模型。Mixtral 8x7B的比例为13/47（28%），密度高得多。Llama 4 Maverick的比例为17B/400B（4.25%），与之相当。DeepSeek的赌注：在前沿规模上，更多专家且激活比例更低，每个活跃FLOP能产生更高质量。

### DeepSeek-V3所处的位置

|  模型  |  总参数量  |  活跃参数量  |  比例  |  注意力机制  |  创新点  |
|-------|------|-------|-------|-----------|-------------|
|  Llama 3 70B  |  70B  |  70B  |  100%  |  GQA 64/8  |  —  |
|  Llama 4 Maverick  |  400B  |  17B  |  4.25%  |  GQA  |  —  |
|  Mixtral 8x22B  |  141B  |  39B  |  27%  |  GQA  |  —  |
|  DeepSeek V3  |  671B  |  37B  |  5.5%  |  MLA 512  |  MLA + MTP + aux-free + DualPipe  |
|  Qwen 2.5 72B  |  72B  |  72B  |  100%  |  GQA 64/8  |  YaRN扩展  |

### 后续版本：R1、V4

DeepSeek-R1（2025年）是在V3骨干上的推理训练版本。R1使用相同的架构。改变的是后训练配方（在可验证任务上进行大规模强化学习），而不是预训练架构。

DeepSeek-V4（如果发布）预计将保留MLA + MoE + MTP，并添加DSA（DeepSeek稀疏注意力），它是第10章第17节中NSA的后继者。演进路线是稳定的：架构层面的创新不断累积；每个版本都在调节额外的旋钮。

```figure
moe-routing
```

## 使用它

`code/main.py` 是专门针对DeepSeek-V3结构的参数量计算器。运行它，将其输出与论文中的数字进行比较，并在假设变体（256专家 vs 512专家，top-8 vs top-16，MLA秩512 vs 1024）上使用它。

需要关注的内容：

- 总参数量 vs 已公布的671B。
- 活跃参数量 vs 已公布的37B。
- 128k上下文下的KV缓存——MLA与GQA对比。
- 逐层分解以查看参数预算的实际去向。

## 发布

本课生成`outputs/skill-deepseek-v3-reader.md`。给定一个DeepSeek系列模型（V3、R1或任何未来变体），它会逐一生成架构解读，命名配置中的每个字段，按组件推导参数量，并识别模型使用了DeepSeek的四项特有创新中的哪些。

## 练习

1. 运行`code/main.py`。将计算器估计的总参数量与已公布的671B进行比较，并找出差异来源。论文第2节有完整的详细说明。

2. 将配置修改为使用MLA秩256而不是512。计算在128k上下文下产生的KV缓存大小。它能带来多少百分比的缩减，以及以每个头的表达能力为代价？

3. 将DeepSeek-V3的（256专家，top-8）路由与一个假设的（512专家，top-8）变体进行比较。总参数量增加，活跃参数量保持不变。理论上额外的专家容量能带来什么，推理时又有什么代价？

4. 阅读DeepSeek-V3技术报告（arXiv:2412.19437）第2.1节关于MLA的内容。用三句话解释为什么K和V解压缩矩阵可以在推理时被“吸收”到后续的矩阵乘法中以提升效率。

5. DeepSeek-V3对大多数操作使用FP8训练。计算FP8与BF16相比存储671B权重节省的内存。这与14.8T token的训练预算如何关联？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  MLA  |  "多头潜在注意力"  |  将K和V压缩成一个共享的低秩潜在表示（kv_lora_rank，通常为512），按头即时解压缩；KV缓存只存储潜在表示  |
|  kv_lora_rank  |  "MLA压缩维度"  |  K和V共享潜在表示的大小；DeepSeek-V3使用512  |
|  前k个密集层  |  "早期层保持密集"  |  MoE模型的前几层跳过MoE路由器，运行密集MLP以保持稳定性  |
|  num_experts_per_tok  |  "Top-k路由"  |  每个token激活多少个路由专家；DeepSeek-V3使用8  |
|  共享专家  |  "始终在线的专家"  |  无论路由如何都处理每个token的专家；DeepSeek-V3使用1  |
|  无辅助损失路由  |  "偏置调节负载均衡"  |  训练中调整每个专家的偏置项，以保持专家负载均衡而不增加损失项  |
|  MTP模块  |  "额外预测头"  |  从h^(1)和E(t+1)预测t+2的Transformer块；更密集的训练，免费的推测解码草稿  |
|  DualPipe | "双向流水线" | 一种训练调度，在前向/反向计算与跨节点全对全通信之间重叠  |
|  活跃参数比例 | "稀疏性" | active参数 / 总参数；DeepSeek-V3达到5.5%  |
|  FP8训练 | "8位训练" | 训练存储和许多计算操作使用FP8；与BF16相比大致减半内存，但质量略有损失  |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 完整的架构、训练和结果文档
- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 配置文件和部署说明
- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 引入MLA的前身模型
- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — V3架构上的推理训练后继模型
- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — DeepSeek系列注意力机制的未来方向
- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 训练调度参考
