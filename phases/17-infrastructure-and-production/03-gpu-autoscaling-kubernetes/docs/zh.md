# Kubernetes 上的 GPU 自动扩缩容 — Karpenter、KAI Scheduler、Gang Scheduling

> 三层，而非一层。Karpenter 动态配置节点（一分钟内完成，比 Cluster Autoscaler 快 40%）。KAI Scheduler 处理群体调度(Gang Scheduling)、拓扑感知和分层队列——它避免了 7-of-8 部分分配陷阱：七个节点等待一个缺失的 GPU 而空转。应用级自动缩放器（NVIDIA Dynamo Planner、llm-d 工作负载变体自动缩放器）基于推理特定信号（队列深度、KV 缓存利用率）进行缩放，而非 CPU/DCGM 占空比。经典的 HPA 陷阱在于 `DCGM_FI_DEV_GPU_UTIL` 是占空比测量：100% 可能对应 10 个请求或 100 个。vLLM 预分配 KV 缓存内存，因此内存永远不会触发缩容。本课程教你组合这三个层次，并避免默认的 Karpenter `WhenEmptyOrUnderutilized` 策略（该策略会在推理过程中终止正在运行的 GPU 任务）。

**类型：** 学习
**语言：** Python（标准库，简易队列深度自动缩放器模拟器）
**先修课程：** 第 17 阶段·02（推理平台经济学）、第 17 阶段·04（vLLM 服务内部原理）
**时间：** 约 75 分钟

## 学习目标

- 绘制三个自动缩放层次（节点配置、群体调度、应用级）并命名每层使用的工具。
- 解释为什么 `DCGM_FI_DEV_GPU_UTIL` 是 vLLM 错误的 HPA 信号，并给出两个替代信号（队列深度、KV 缓存利用率）。
- 描述群体调度以及 KAI Scheduler 防止的部分分配失败模式（8 个 GPU 中 7 个空闲）。
- 命名会终止正在运行的 GPU 任务的 Karpenter 整合策略(`DCGM_FI_DEV_GPU_UTIL`)，并指出 2026 年的安全替代方案。

## 问题

你的团队在 Kubernetes 上部署了一个 LLM 推理服务。你设置了以 `DCGM_FI_DEV_GPU_UTIL` 为信号的 HPA。服务在业务时间内维持在 100% 利用率。HPA 从未扩容——它认为已经满载。你手动添加了一个副本；TTFT 下降了。HPA 仍未扩容。信号在欺骗你。

另外，你使用 Cluster Autoscaler 进行节点管理。一个 100 万 token 的提示在凌晨 2 点到达；集群花费 3 分钟配置一个节点，请求超时。

另外一次，你部署了一个需要 8 个 GPU（跨 2 个节点）的 70B 模型。集群有 7 个空闲 GPU，另外 1 个分散在 3 个节点上。Cluster Autoscaler 为缺失的那个 GPU 配置了一个节点。七个节点等待了 4 分钟，空转烧钱，而 Kubernetes 才把最后一个 GPU 启动好。

三层，三种不同的失败模式。2026 年的 GPU 感知自动缩放不是“开启 HPA”。而是组合节点配置、群体调度和应用信号自动缩放。

## 核心概念

### 第 1 层——节点配置（Karpenter）

Karpenter 监控待处理 Pod，并在约 45-60 秒内配置节点（Cluster Autoscaler 通常需要 90-120 秒配置 GPU 节点）。它根据 `NodePool` 约束动态选择实例类型——如果你的 Pod 需要 8 个 H100 且集群没有匹配的节点，Karpenter 会直接配置一个节点，而非扩展现有节点组。

**整合陷阱**：Karpenter 的默认 `consolidationPolicy: WhenEmptyOrUnderutilized` 对 GPU 池很危险。它会终止一个正在运行的 GPU 节点，以将 Pod 迁移到更便宜的合适实例。对于推理工作负载，这意味着驱逐正在运行的请求并在新节点上重新加载 70B 模型。损失是数分钟的容量加上请求失败。

GPU 池的安全设置：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

允许 Karpenter 在一小时后整合真正空的节点，但绝不驱逐正在运行的任务。

### 第 2 层——群体调度（KAI Scheduler）

KAI Scheduler（项目原名为“Karp”，后更名）处理默认 kube-scheduler 无法处理的任务：

**群体调度**——全有或全无调度。一个需要 8 个 GPU 的分布式推理 Pod，要么全部 8 个一起启动，要么一个都不启动。没有这个功能，就会出现部分分配陷阱：7 个 Pod 启动，无限期等待，空转烧钱。

**拓扑感知**——知道哪些 GPU 共享 NVLink，哪些位于同一机架，哪些之间有 InfiniBand。据此放置 Pod。一个 DeepSeek-V3 67B 张量并行工作负载必须位于同一个 NVLink 域内；KAI Scheduler 遵守这一点。

**分层队列**——多个团队竞争同一 GPU 池，具有优先级和配额。团队 A 的生产性需求只有在优先级规则允许时才会被团队 B 的训练任务抢占。

KAI 作为辅助调度器与 kube-scheduler 一起部署；你通过注解让工作负载使用它。Ray 和 vLLM 生产堆栈都已集成。

### 第 3 层——应用级信号

**HPA 陷阱**：`DCGM_FI_DEV_GPU_UTIL` 是一个占空比指标——它测量 GPU 在每个采样间隔内是否在做工作。100% 利用率可能意味着 10 个并发请求或 100 个；GPU 无论如何都是忙的。基于占空比进行缩放是盲目缩放。

更糟的是，vLLM 和类似引擎预分配 KV 缓存内存（高达 `--gpu-memory-utilization`）。即使只有一个请求，内存使用也接近 90%。基于内存的 HPA 永远不会缩容。

**2026 年替代信号**：

- 队列深度（等待前缀填充的请求数）。
- KV 缓存利用率（分配给活跃序列的块比例）。
- 每个副本的 P99 TTFT（你的 SLA 信号）。
- 好吞吐量(Goodput)（每秒满足所有 SLO 的请求数）。

NVIDIA Dynamo Planner 和 llm-d 工作负载变体自动缩放器消耗这些信号并缩放副本。它们完全取代了 LLM 推理的 HPA。

### 何时使用什么

|  缩放决策  |  工具  |
|----------------|------|
|  添加/移除节点  |  Karpenter  |
|  调度多 GPU 任务  |  KAI Scheduler  |
|  添加/移除副本  |  Dynamo Planner / llm-d WVA（或基于队列深度的自定义 HPA）  |
|  选择 GPU 类型  |  Karpenter NodePool  |
| 抢占低优先级  |  KAI 调度器队列 |

### 分离式预填充/解码使一切复杂化

如果运行分离式预填充/解码（第17阶段·17），会有两个具有不同扩缩触发器的Pod类别：预填充Pod根据队列深度扩缩，解码Pod根据KV缓存压力扩缩。llm-d通过每角色HPA将这些暴露为单独的`Services`。不要试图在两者前放置单个HPA。

### 冷启动在此处也很重要

冷启动缓解（第17阶段·10）是指节点配置时间变得对用户可见。Karpenter的45-60秒预热加上20GB模型加载和引擎初始化意味着从零开始的请求需要2-5分钟。对于SLO关键路径，保持一个热池（`min_workers=1`），或在应用层使用Modal风格的检查点。

### 你应该记住的数字

- Karpenter节点配置：约45-60秒 vs 集群自动扩缩器约90-120秒（GPU节点）。
- KAI调度器防止部分分配浪费——7/8陷阱。
- `DCGM_FI_DEV_GPU_UTIL`作为HPA信号：不可靠；使用队列深度或KV利用率。
- Karpenter `DCGM_FI_DEV_GPU_UTIL`：终止正在运行的GPU任务。使用`WhenEmptyOrUnderutilized`进行推理。

```figure
autoscaling
```

## 使用它

`code/main.py`模拟突发GPU工作负载上的三层自动扩缩器。比较朴素HPA（占空比）、队列深度HPA和KAI组调度扩缩。报告未满足的请求、空闲GPU分钟数和综合评分。

## 发布

本课产生`outputs/skill-gpu-autoscaler-plan.md`。给定集群拓扑、工作负载形状和SLO，设计一个三层自动扩缩计划。

## 练习

1. 运行`code/main.py`。在突发工作负载下，朴素占空比HPA丢弃的请求中有多少被队列深度HPA捕获？差异从何而来？
2. 为服务于Llama 3.3 70B FP8（H100 SXM5）的集群设计一个Karpenter NodePool。指定`code/main.py`、`capacity-type`、`disruption.consolidationPolicy`以及一个保持非GPU工作负载不运行在这些节点上的污点。
3. 你的团队报告部署卡在Pending状态，因为“GPU可用但Pod无法调度”。诊断——是Karpenter、kube-scheduler还是KAI调度器？哪些指标可以确认？
4. 选择用于自动扩缩分离式预填充Pod的信号和用于解码Pod的不同信号。证明两者合理性。
5. 计算在平均每天60次请求丢弃事件（P99 TTFT > 10s）的24x7生产服务上，`code/main.py`整合陷阱的成本。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  Karpenter  |  "节点配置器"  |  Kubernetes节点自动扩缩器；亚分钟级配置  |
|  集群自动扩缩器  |  "旧扩缩器"  |  Kubernetes节点自动扩缩器前身；较慢，基于组  |
|  KAI 调度器  |  "GPU调度器"  |  用于组调度+拓扑+队列的二次调度器  |
|  组调度  |  "全有或全无"  |  原子调度N个Pod，否则全部延迟  |
|  拓扑感知  |  "机架感知"  |  基于NVLink/IB/机架放置Pod  |
|  `DCGM_FI_DEV_GPU_UTIL`  |  "GPU利用率"  |  占空比指标；不是LLM的扩缩信号  |
|  队列深度  |  "等待请求"  |  适用于预填充扩缩的正确HPA信号  |
|  KV缓存利用率  |  "内存压力"  |  适用于解码扩缩的正确HPA信号  |
|  整合  |  "Karpenter整合"  |  节点终止以换成更便宜实例类型  |
|  `WhenEmpty + 1h`  |  "安全整合"  |  不驱逐正在运行的GPU任务的策略  |

## 延伸阅读

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler)——设计文档和配置示例。
- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler)——整合策略语义和GPU安全默认值。
- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler)——Dynamo Planner扩缩信号。
- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler)——Ray集成模式。
- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler)——托管Kubernetes特定指南。
- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler)——工作负载变体自动扩缩器设计。
