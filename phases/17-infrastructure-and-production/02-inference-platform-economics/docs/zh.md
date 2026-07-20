# 推理平台经济学 —— Fireworks、Together、Baseten、Modal、Replicate、Anyscale

> 2026年推理市场不再是GPU时间租赁。它分化为定制芯片（Groq、Cerebras、SambaNova）、GPU平台（Baseten、Together、Fireworks、Modal）和API优先市场（Replicate、DeepInfra）。Fireworks以每天10T+ tokens的规模获得了$1/hr per GPU on May 1, 2026, and $40亿估值，说明规模化驱动模式行得通。Baseten在2026年1月完成了$300M Series E at $50亿美元融资。竞争定位规则很简单：Fireworks优化延迟，Together优化目录广度，Baseten优化企业级精致度，Modal优化Python原生开发者体验，Replicate优化多模态覆盖，Anyscale优化分布式Python。这节课会提供一个你可以交给创始人的矩阵。

**类型：** 学习
**语言：** Python（标准库，简单的每次调用经济学比较器）
**前提条件：** 第17阶段·01（托管LLM平台），第17阶段·04（vLLM服务内部原理）
**时间：** 约60分钟

## 学习目标

- 列举三个细分市场（定制芯片、GPU平台、API优先）并将每个供应商对应到细分市场。
- 解释为什么“按token”API定价模型会压缩到服务引擎的成本曲线，而不是硬件的成本曲线。
- 计算至少三个供应商的每次请求有效成本，并解释按分钟计费（Baseten、Modal）何时优于按token计费。
- 识别对于给定工作负载（无服务器突发型、稳定高吞吐、微调变体、多模态）哪个平台是合适的默认选项。

## 问题

你已经评估了托管超大规模平台。你决定需要一个更窄、更快的供应商——Fireworks用于低延迟，Together用于广度，Baseten用于微调的自定义模型。现在你有六个真实选择，但定价页面并不对齐。Fireworks显示$/M tokens; Baseten shows $/分钟；Modal显示$/second; Replicate shows $/次预测。不对工作负载建模就无法直接比较。

更糟的是，每个定价页面背后的商业模式不同。Fireworks在共享GPU上运行自己的自定义引擎（FireAttention）；每token费率反映了其利用率曲线。Baseten提供Truss + 专属GPU；按分钟计费反映了独占性。Modal是真正的Python无服务器——按秒计费，冷启动亚秒级。同样的输出（LLM响应），三种不同的成本函数。

这节课对六个平台建模，并告诉你每个平台何时胜出。

## 核心概念

### 三个细分市场

**定制芯片** — Groq（LPU）、Cerebras（WSE）、SambaNova（RDU）。通常在同一模型上解码速度比基于GPU的集群快5-10倍。单token价格更高（2025年底Groq在Llama-70B上约为$0.99/M），但在延迟敏感型用例中无与伦比。Groq是语音助手和实时翻译的生产选择。

**GPU平台** — Baseten、Together、Fireworks、Modal、Anyscale。运行在NVIDIA（H100、H200、B200，2026年）或有时AMD上。介于“裸GPU租赁”（RunPod、Lambda）和“超大规模托管服务”（Bedrock）之间的经济层。

**API优先市场** — Replicate、DeepInfra、OpenRouter、Fal。目录广泛，按预测或按秒付费，强调首次调用时间。

### Fireworks——延迟优化的GPU平台

- FireAttention引擎（自定义）；宣称同等配置下延迟比vLLM低4倍。
- 非交互工作负载的批处理层级价格约为无服务器层级的50%。
- 微调模型以与基础模型相同的速率提供服务——与对LoRA收取溢价的供应商相比是真正的差异化优势。
- 2026年中期：按需GPU租赁有效价格$1/小时（2026年5月1日起）。大规模时价格可协商。
- 财务信号：40亿估值，每天处理10T+ tokens。

### Together——广度优化

- 200+个模型，包括上游发布后几天内的开源版本。
- 同等LLM模型上比Replicate便宜50-70%——“AI原生云”定位的是规模和目录。
- 推理+微调+训练统一API。

### Baseten——企业级精致度优化

- Truss框架：模型打包，包含依赖、密钥、服务配置在一个清单中。
- GPU范围从T4到B200。按分钟计费，冷启动缓解合理。
- SOC 2 Type II、HIPAA兼容。常见的金融科技和医疗保健选择。
- $5B valuation, January 2026 Series E ($3亿美元（来自CapitalG、IVP、NVIDIA）。

### Modal——Python原生优化

- 纯Python的基础设施即代码。用`@modal.function(gpu="A100")`装饰函数，一条命令部署。
- 按秒计费。冷启动2-4秒，预热后；小模型<1秒。
- $87M Series B at $11亿美元估值（2025年）。在独立调查中开发者体验评分最高。

### Replicate——多模态广度

- 按预测付费。图像、视频和音频模型的默认平台。
- 集成生态系统（Zapier、Vercel、CMS插件）。
- 在LLM每token费率上竞争力较弱，但在多模态多样性上胜出。

### Anyscale——Ray原生

- 基于Ray构建；RayTurbo是Anyscale的专有推理引擎（与vLLM竞争）。
- 最适合分布式Python工作负载，其中推理步骤是更大图中的一个节点。
- 托管Ray集群；与Ray AIR和Ray Serve紧密集成。

### 按token vs 按分钟——各自何时胜出

按token在工作负载对延迟不敏感且突发性时合理——你只为使用付费。按分钟在利用率高且可预测时合理——一旦GPU饱和，你就超过了按token计费。

粗略规则：对于持续利用率超过约30%的专用GPU工作负载，按分钟计费（Baseten、Modal）开始优于按token计费（Fireworks、Together）。低于此，按token胜出，因为你避免了为空闲付费。

### 自定义引擎才是真正的护城河

每个vLLM和SGLang之上的平台都宣称有自定义引擎。FireAttention、RayTurbo、Baseten的推理栈。自定义引擎的说法带有营销色彩——诚实的说法是vLLM + SGLang大约占生产开源推理的80%，平台层的差异化在于开发者体验、归因和服务等级协议。

### 你应该记住的数字

- Fireworks GPU租赁：$1/小时（2026年5月1日起生效）。
- Fireworks宣称：同等配置下延迟比vLLM低4倍。
- Together：在LLM上比Replicate便宜50-70%。
- Baseten估值：$5B (Series E, Jan 2026, $3亿美元轮次）。
- Modal估值：$11亿（B轮，2025年）。
- 持续利用率超过约30%时按分钟优于按token。

```figure
cost-per-token
```

## 使用它

`code/main.py`在合成工作负载上比较六个供应商的定价模型。报告$/day and effective $/M tokens。运行它来找到按token和按分钟之间的盈亏平衡点。

## 发布

这节课产生`outputs/skill-inference-platform-picker.md`。给定工作负载概况、服务等级协议和预算，选择主推理平台并命名亚军。

## 练习

1. 运行`code/main.py`。对于70B模型在单张H100上，Baseten（按分钟计费）在多大的持续利用率下优于Fireworks（按token计费）？自行推导交叉点并与经验法则进行比较。
2. 你的产品支持图像生成、聊天和语音转文本。为每种模态选择平台，并命名统一它们的网关模式。
3. Fireworks将你的主要模型价格提高了1美元/小时。如果40%的流量转移到批处理层（50%折扣），建模混合成本影响。
4. 一个受监管的客户需要SOC 2 Type II + HIPAA + 专用GPU。哪三个平台是可行的，哪个在FinOps上胜出？
5. 比较Llama 3.1 70B在Fireworks无服务器、Together按需、Baseten专用和Replicate API上每1000次预测的成本。在10次预测/天时哪个最便宜？在10,000次时呢？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
| 定制芯片  |  "非GPU芯片"  |  Groq LPU、Cerebras WSE、SambaNova RDU — 针对解码优化 |
| FireAttention  |  "Fireworks引擎"  |  自定义注意力内核；声称比vLLM延迟低4倍 |
| Truss  |  "Baseten的格式"  |  模型打包清单；依赖项+密钥+服务配置 |
| 按token计费  |  "API定价"  |  按消耗的token收费；无空闲时不付费 |
| 按分钟计费  |  "专用定价"  |  按挂钟GPU时间收费；在高利用率时胜出 |
| 按预测计费  |  "Replicate定价"  |  按模型调用收费；常用于图像/视频 |
| RayTurbo  |  "Anyscale引擎"  |  基于Ray的专有推理；与Ray集群上的vLLM竞争 |
| 批处理层  |  "50%折扣"  |  以降低价格的非交互式队列；常见于Fireworks、OpenAI |
| 按基础费率微调  |  "Fireworks LoRA"  |  对LoRA服务的请求按基础模型费率收费（差异化因素） |

## 延伸阅读

- [Fireworks Pricing](https://fireworks.ai/pricing) — 按token费率、批处理层、GPU租赁。
- [Fireworks Pricing](https://fireworks.ai/pricing) — 按分钟费率、承诺容量、企业层。
- [Fireworks Pricing](https://fireworks.ai/pricing) — 按秒GPU费率和免费层。
- [Fireworks Pricing](https://fireworks.ai/pricing) — 模型目录和按token费率。
- [Fireworks Pricing](https://fireworks.ai/pricing) — RayTurbo和托管Ray定价。
- [Fireworks Pricing](https://fireworks.ai/pricing) — 比较评估。
- [Fireworks Pricing](https://fireworks.ai/pricing) — 供应商格局。
