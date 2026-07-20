# 自托管服务选型 — llama.cpp、Ollama、TGI、vLLM、SGLang

> 2026年自托管推理领域由四个引擎主导。根据硬件、规模及生态系统选择。**llama.cpp** 在CPU上最快——模型支持最广，对量化和线程有完全控制。**Ollama** 是开发者笔记本电脑的一条命令安装，比llama.cpp慢约15-30%（Go + CGo + HTTP序列化），在类似生产负载下吞吐量差距达3倍。**TGI于2025年12月11日进入维护模式**——仅修复bug，原始吞吐量比vLLM慢约10%，但历史上拥有顶尖的可观测性和HuggingFace生态系统集成。维护状态使其长期风险较高——对于新项目，SGLang或vLLM是更安全的默认选择。**vLLM** 是通用生产默认选择——v0.15.1（2026年2月）增加了PyTorch 2.10、RTX Blackwell SM120、H200优化。**SGLang** 是代理人多轮/前缀密集型场景的专家——生产环境中超过40万块GPU（xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS）。硬件约束：纯CPU → 仅llama.cpp。AMD/非NVIDIA → 仅vLLM（TRT-LLM受NVIDIA锁定）。2026年流程模式：开发 = Ollama，预发 = llama.cpp，生产 = vLLM或SGLang。全程使用相同的GGUF/HF权重。

**类型：** 学习
**语言：** Python（标准库，引擎决策树遍历）
**前置要求：** 所有第17阶段关于引擎的课程（04, 06, 07, 09, 18）
**时间：** 约45分钟

## 学习目标

- 给定硬件（CPU / AMD / NVIDIA Hopper / Blackwell）、规模（1用户 / 100 / 10,000）和工作负载（通用聊天 / 代理人 / 长上下文），选择一个引擎。
- 说明2026年TGI维护模式状态（2025年12月11日），以及为何它使新项目偏向vLLM或SGLang。
- 描述使用相同GGUF或HF权重贯穿开发/预发/生产流程。
- 解释为何“纯CPU”强制选择llama.cpp，以及“AMD”排除TRT-LLM。

## 问题

你的团队启动了一个新的自托管LLM项目。一位工程师说用Ollama，另一位说用vLLM，第三位说“TGI不是开箱即用吗？”三种说法在不同语境下都是对的。但没有一种在所有情况下都正确。

2026年，选择树很重要：先硬件，再规模，后工作负载。并且2025年的一件具体事件——TGI于12月11日进入维护模式——改变了新项目的默认选择。

## 核心概念

### 五个引擎

|  引擎 | 最适合 | 备注  |
|--------|----------|-------|
|  **llama.cpp** | CPU / 边缘设备 / 最小依赖 / 模型支持最广 | CPU上最快，完全控制  |
|  **Ollama** | 开发笔记本电脑，单用户，一条命令安装 | 比llama.cpp慢15-30%；生产吞吐量差距3倍  |
|  **TGI** | HuggingFace生态系统，受监管行业 | **2025年12月11日进入维护模式**  |
|  **vLLM** | 通用生产环境，100+用户 | 广泛的生产默认选择；v0.15.1 2026年2月  |
|  **SGLang** | 代理人多轮，前缀密集型工作负载 | 生产环境中超过40万块GPU  |

### 硬件优先决策

**纯CPU** → llama.cpp。Ollama也可用但较慢。其他引擎在CPU上不具备竞争力。

**AMD GPU** → vLLM（支持AMD ROCm）。SGLang也可用。TRT-LLM受NVIDIA锁定，因此排除。

**NVIDIA Hopper（H100 / H200）** → vLLM或SGLang或TRT-LLM。三者均为顶级。

**NVIDIA Blackwell（B200 / GB200）** → TRT-LLM吞吐量领先（第17阶段·07）。vLLM和SGLang紧随其后。

**Apple Silicon（M系列）** → llama.cpp（Metal）。Ollama封装了它。

### 规模第二决策

**1用户 / 本地开发** → Ollama。一条命令，数秒内得到第一个词。

**10-100用户 / 小团队** → vLLM单GPU。

**100-10k用户 / 生产环境** → vLLM生产堆栈（第17阶段·18）或SGLang。

**10k+用户 / 企业级** → vLLM生产堆栈 + 分离式架构（第17阶段·17）+ LMCache（第17阶段·18）。

### 工作负载第三决策

**通用聊天 / 问答** → vLLM在广泛默认场景中胜出。

**代理人多轮（工具、规划、记忆）** → SGLang的RadixAttention（第17阶段·06）占主导。

**高前缀复用的RAG** → SGLang。

**代码生成** → vLLM良好；SGLang在缓存方面稍优。

**长上下文（128K+）** → vLLM + 分块预填充；SGLang + 分级KV。

### TGI维护陷阱

Hugging Face TGI 于 2025 年 12 月 11 日进入维护模式——此后仅进行错误修复。历史表现：顶级可观测性、一流的 HF 生态系统集成（模型卡、安全工具），原始吞吐量略低于 vLLM。

对于 2026 年的新项目：默认不采用 TGI。现有 TGI 部署可以继续，但最终应迁移。SGLang 和 vLLM 是更安全的默认选择。

### 流水线模式

开发（Ollama）→ 预发（llama.cpp）→ 生产（vLLM）。全程使用相同的 GGUF 或 HF 权重。工程师在笔记本电脑上快速迭代；预发环境镜像生产量化；生产环境是服务目标。

### Ollama 注意事项

Ollama 适合开发，但不适合共享生产环境：Go HTTP 序列化增加了开销，并发管理比 vLLM 简单，OpenTelemetry 支持滞后。在 Ollama 擅长的场景下使用它——单用户，单命令——并在共享环境中切换到 vLLM。

### 自托管与托管是独立的决策

Phase 17 · 01（托管超大规模云商）、· 02（推理平台）涵盖托管。本课假设您已决定自托管。自托管的理由：数据驻留、自定义微调、规模化总拥有成本、托管环境未提供的领域模型。

### 你应该记住的数字

- TGI 维护模式：2025 年 12 月 11 日。
- vLLM v0.15.1：2026 年 2 月；PyTorch 2.10；支持 Blackwell SM120。
- SGLang 生产足迹：400,000+ 块 GPU。
- Ollama 与 llama.cpp 的吞吐量差距：慢 15-30%；生产负载下慢 3 倍。

```figure
data-parallel
```

## 使用它

`code/main.py` 是一个决策树遍历器：给定硬件、规模和工作负载，选择一个引擎并解释原因。

## 发布

本课产出 `outputs/skill-engine-picker.md`。在给定约束条件下，选择一个引擎并编写迁移计划。

## 练习

1. 使用您的硬件/规模/工作负载运行 `code/main.py`。输出是否符合您的直觉？
2. 您的基础设施是 12 块 H100 和 8 块 MI300X AMD。选择哪个引擎？为什么 TRT-LLM 不可用？
3. 一个团队想在 2026 年使用 TGI，理由是“我们用惯了”。请论证迁移的必要性。
4. 从 Ollama 开发环境到 vLLM 生产环境：量化、配置和可观测性方面需要哪些变化？
5. RAG 产品，P99 前缀长度为 8K，且跨租户高度复用。选择一个引擎，并与 Phase 17 · 11 + 18 堆叠使用。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  llama.cpp  |  “CPU 版本”  |  最广泛的模型支持，CPU 上最快  |
|  Ollama  |  “笔记本版本”  |  一键安装，开发级吞吐量  |
|  TGI  |  “HF 的服务”  |  自 2025 年 12 月起进入维护模式  |
|  vLLM  |  “默认选择”  |  2026 年广泛的生产基线  |
|  SGLang  |  “代理型”  |  前缀密集型，RadixAttention  |
|  TRT-LLM  |  “NVIDIA 锁定”  |  Blackwell 吞吐量领导者，仅限 NVIDIA  |
|  GGUF  |  “llama.cpp 格式”  |  捆绑的 K-quant 变体  |
|  生产栈  |  “vLLM K8s”  |  Phase 17 · 18 参考部署  |
|  流水线模式  |  “开发→预发→生产”  |  相同权重上的 Ollama → llama.cpp → vLLM  |

## 延伸阅读

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/) — 发布说明。
- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
