# 托管 LLM 平台 — Bedrock、Vertex AI、Azure OpenAI

> 三大超大规模云服务商，三种截然不同的策略。AWS Bedrock 是一个模型市场 — Claude、Llama、Titan、Stability、Cohere 统一到一个 API 后面。Azure OpenAI 是 Open AI 的独家合作伙伴关系，并提供预置吞吐量单位（Provisioned Throughput Units, PTUs）用于专用容量。Vertex AI 以 Gemini 为先，拥有最好的长上下文和多模态能力。2026 年 Artificial Analysis 测量到 Azure OpenAI 的中位延迟约为 50 毫秒，而 Bedrock 在等效的 Llama 3.1 405B 模型上约为 75 毫秒 — PTUs 解释了这一差距，因为专用容量优于共享按需。决策规则不是“哪个最快”，而是“哪个模型目录和 FinOps 层面与我的产品匹配”。本课程教你根据书面权衡而非直觉进行选择。

**类型：** 学习
**语言：** Python（标准库，玩具式成本与延迟比较器）
**前提条件：** 阶段 11（LLM 工程），阶段 13（工具与协议）
**时间：** 约 60 分钟

## 学习目标

- 列举三种平台策略（市场 vs 独家 vs 以 Gemini 为先），并将每种策略匹配到一个产品用例。
- 解释预置吞吐量单位（PTUs）在 Azure OpenAI 中的用途，以及为什么按需 Bedrock 在 405B 规模下通常慢约 25 毫秒。
- 描绘每个平台的 FinOps 归因层面（Bedrock Application Inference Profiles vs Vertex 每个团队一个项目 vs Azure 作用域 + PTU 预留）。
- 写下“双提供商最低”策略，并解释为什么单一供应商锁定是 2026 年昂贵的错误。

## 问题

你为产品选择了 Claude 3.7 Sonnet。现在你需要提供服务。你可以直接调用 Anthropic API，或通过 AWS Bedrock 调用，或通过网关调用。直接 API 最简单；Bedrock 增加了 BAAs、VPC 端点、IAM 和 CloudWatch 归因。网关增加了跨提供商的故障转移、统一计费和速率限制。

更深层的问题是目录。如果你需要在同一产品中使用 Claude、Llama 和 Gemini，你无法从一个地方全部购买，除非那个地方同时是 Bedrock、Vertex 和 Azure OpenAI。超大规模云服务商不是可互换的 — 它们各自对谁拥有模型层做出了不同的押注。

本课程描绘了这三个押注、延迟差距、FinOps 差距和锁定风险。

## 核心概念

### 三大策略

**AWS Bedrock** — 市场。Claude（Anthropic）、Llama（Meta）、Titan（AWS 第一方）、Stability（图像）、Cohere（嵌入）、Mistral，以及图像和嵌入子目录。一个 API，一个 IAM 层，一个 CloudWatch 导出。Bedrock 的押注是客户更想要可选性而非单一模型。

**Azure OpenAI** — 独家合作关系。你可以在 Azure 数据中心获得 GPT-4 / 4o / 5 / o 系列、DALL·E、Whisper 和 OpenAI 模型的微调。在“Azure OpenAI 服务”目录中没有非 OpenAI 模型 — 那些模型归入 Azure AI Foundry（独立产品）。Azure 的押注是 OpenAI 保持前沿，并且客户希望对该特定关系拥有企业控制权。

**Vertex AI** — Gemini 优先，其他其次。Gemini 1.5 / 2.0 / 2.5 Flash 和 Pro，外加 Model Garden（第三方）。Vertex 的押注是多模态长上下文 — 100 万 token 的 Gemini 上下文是差异化因素。

### 规模化下的延迟差距

Artificial Analysis 运行持续基准测试。在等效的 Llama 3.1 405B 部署（共享按需）上，Azure OpenAI 的中位首 token 延迟约为 50 毫秒；Bedrock 约为 75 毫秒。差距并非 AWS 的失败 — 而是容量模型的不同。Azure 销售 PTUs（预置吞吐量单位），为你的租户预留 GPU 容量。Bedrock 的等效产品（预置吞吐量）存在，但每单位起价约 21 美元/小时，大多数客户仍使用共享按需。

共享按需容量与所有其他客户的流量竞争。专用容量没有竞争。如果你的产品 SLA 要求 TTFT < 100 毫秒（P99），你可以选择购买 Azure 的 PTUs、购买 Bedrock 的预置吞吐量，或者接受默认方差。

### 预置吞吐量经济学

Azure PTUs：预留的推理计算块。对于可预测的工作负载，相比按需节省高达约 70%。无论流量如何，每小时费用固定 — 即使空闲也要支付预留费用。盈亏平衡点通常在持续利用率为 40-60% 左右。

Bedrock 预置吞吐量：$21-$50 每小时，取决于模型和区域。类似的计算 — 盈亏平衡点约为峰值利用率的一半。需要月度承诺。

Vertex 的预置容量按 Gemini SKU 出售；定价因模型和区域而异，且较少公开宣传。

### FinOps 层面 — 真正的差异化因素

**Bedrock Application Inference Profiles** 是市场上最干净的归因方式。使用 `team`、`product`、`feature` 标记配置；将所有模型调用路由通过它；CloudWatch 无需后处理即可按配置分解成本。于 2025 年添加，仍然是超大规模云原生中最细粒度的。

**Vertex** 的归因方式是每个团队一个项目加上无处不在的标签。你将每个团队建模为一个 GCP 项目，在每个资源上贴上标签，并使用 BigQuery Billing Export + DataStudio 进行汇总。工作量更大，但 BigQuery 允许对成本数据执行任意 SQL。

**Azure** 依赖于订阅/资源组作用域加标签，并将 PTU 预留作为一等成本对象。标签从资源组继承，而非请求，因此按请求归因需要 Application Insights 自定义指标或一个添加请求头的网关。

模式：Bedrock 原生最干净，Vertex 通过 BigQuery 最灵活，Azure 最不透明，除非你进行检测。

### 锁定是 2026 年的风险

当单一模型占主导时，单一超大规模云服务商承诺是可以的。2026 年，前沿每月变化 — 这个季度是 Claude 3.7，下个季度是 Gemini 2.5，再下个季度是 GPT-5。锁定到一个平台将使你失去三分之二的前沿。

实际团队采用的模式：对任何产品关键的 LLM 调用采用双提供商最低。Bedrock 加 Azure OpenAI 是常见的配对 — Claude 来自一个，GPT 来自另一个，它们之间进行故障转移，使用同一网关。成本增加微不足道，因为网关路由最优；中断期间的可用性提升（例如 Azure OpenAI 2025 年 1 月事件、AWS us-east-1 中断）是决定性的。

### 数据驻留、BAAs 与受监管行业

Bedrock：大多数区域有 BAAs；VPC 端点；防护措施。常见的金融科技默认选择。
Azure OpenAI：HIPAA、SOC 2、ISO 27001；欧盟数据驻留；企业受监管默认选择。
Vertex：HIPAA、GDPR、按区域的数据驻留；Google Cloud 的合规栈。

三者都满足基本要求。差异在于数据保留策略、日志处理方式以及滥用监控是否读取你的流量（多数默认选择加入；企业可选择退出）。

### 你应该记住的数字

- Azure OpenAI 在 Llama 3.1 405B 等效模型上的 TTFT 中位数：约 50 毫秒（使用 PTU）。
- Bedrock 按需 TTFT 中位数：约 75 毫秒。
- Bedrock 预置吞吐量：每小时每单元 $21-$50。
- Azure PTU 盈亏平衡点：持续利用率约 40-60%。
- 高利用率下 PTU 相比按需节省：高达 70%。

## 使用它

`code/main.py` 在合成工作负载上比较三个平台——它模拟按需与 PTU 的经济性、TTFT 方差以及成本归因精度。运行它以查看 PTU 在何处值得投入，以及市场模型广度在何处能够弥补 TTFT 差距。

## 发布

本课生成 `outputs/skill-managed-platform-picker.md`。给定工作负载概况（所需模型、TTFT SLA、日处理量、合规要求），它会推荐主平台、备用平台以及 FinOps 工具计划。

## 练习

1. 运行 `code/main.py`。对于 70B 类模型，在什么持续利用率下 Azure PTU 比按需更优？计算盈亏平衡点，并与宣传的 40-60% 区间进行比较。
2. 您的产品需要 Claude 3.7 Sonnet 和 GPT-4o。设计一个双供应商部署——哪个模型放在哪个超大规模云上，前端使用什么网关，故障转移策略是什么？
3. 一家受监管的医疗客户要求 BAA、美国东部数据驻留以及低于 100 毫秒的 P99 TTFT。选择一个平台，并用三个具体特性证明其合理性。
4. 您发现本月 Bedrock 账单在没有流量变化的情况下上涨了 4 倍。如果没有应用推理配置文件，您如何找到原因？有配置文件的情况下，需要多长时间？
5. 阅读 Azure OpenAI 和 Bedrock 定价页面。对于每月 1 亿 token 的 Claude 工作负载，哪个更便宜——直接使用 Anthropic API、Bedrock 按需还是 Bedrock 预置吞吐量？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  Bedrock  |  "AWS LLM 服务"  |  模型市场，涵盖 Claude、Llama、Titan、Mistral、Cohere  |
|  Azure OpenAI  |  "Azure 的 ChatGPT"  |  Azure 数据中心内的专属 OpenAI 模型，提供企业管控  |
|  Vertex AI  |  "Google 的 LLM"  |  以 Gemini 为先的平台，包含用于第三方模型的 Model Garden  |
|  PTU  |  "专用容量"  |  预置吞吐量单元——保留的推理 GPU，按小时定价  |
|  应用推理配置文件  |  "Bedrock 标记"  |  带标签的按产品成本和用量配置文件，原生集成 CloudWatch  |
|  Model Garden  |  "Vertex 目录"  |  Vertex AI 的第三方模型分区，与 Gemini 分开  |
|  双供应商最小化  |  "LLM 冗余"  |  策略：每个关键 LLM 路径至少运行在两家超大规模云上  |
|  BAA  |  "HIPAA 文书"  |  业务伙伴协议；PHI 所必需；三家供应商均提供  |
|  滥用监控  |  "日志观察者"  |  供应商端对提示/输出的安全扫描；企业版可选退出  |

## 延伸阅读

- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)——官方费率卡和预置吞吐量定价。
- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)——PTU 经济性和费率卡。
- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)——Gemini 分层价格和 Model Garden 附加费。
- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)——各供应商的连续延迟和吞吐量基准。
- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)——企业决策框架。
- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)——归因机制对比。
