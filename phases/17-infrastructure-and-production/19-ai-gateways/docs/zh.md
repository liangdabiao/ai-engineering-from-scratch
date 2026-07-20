# AI 网关 — LiteLLM、Portkey、Kong AI Gateway、Bifrost

> 网关位于你的应用和模型提供商之间。核心功能包括：提供商路由、故障转移、重试、速率限制、密钥引用、可观测性、护栏。2026年市场划分：**LiteLLM**是MIT开源许可证，支持100+提供商，兼容OpenAI，但在约2000 RPS（8 GB内存，公开基准测试中出现级联故障）时崩溃；最适合Python，<500 RPS，开发/原型设计。**Portkey**定位于控制平面（护栏、PII脱敏、越狱检测、审计追踪），于2026年3月改为Apache 2.0开源，延迟增加20-40毫秒，定价$100/模型/月（Plus层最多5个）；如果你已在用Kong，则适合企业场景。**Bifrost**（Maxim AI）——自动重试并支持可配置退避，在OpenAI 429时回退到Anthropic。**Cloudflare / Vercel AI Gateways**——托管，零运维，基本重试。数据驻留决定了自托管的选择；Portkey和Kong位于中间，提供开源和可选托管。

**类型：** 学习
**语言：** Python（标准库，玩具级网关路由模拟器）
**前置条件：** 阶段17·01（托管LLM平台），阶段17·16（模型路由）
**时间：** ~60分钟

## 学习目标

- 列举六个核心网关功能（路由、故障转移、重试、速率限制、密钥、可观测性、护栏）。
- 将2026年的四个网关（LiteLLM、Portkey、Kong AI、Bifrost）映射到规模上限和用例。
- 引用Kong的基准测试（比Portkey快228%，比LiteLLM快859%）并解释为何这对>500 RPS场景重要。
- 根据数据驻留和运维预算选择自托管还是托管。

## 问题

你的产品调用OpenAI、Anthropic和自托管的Llama。每个提供商有不同的SDK、错误模型、速率限制和认证方案。你需要故障转移（如果OpenAI 429，则尝试Anthropic）、单一凭证存储、统一可观测性以及每个租户的速率限制。

在应用层重新实现这一点会将每个服务与每个提供商耦合起来。网关层将其整合到一个进程中，并通过一个API（通常兼容OpenAI）分发到各个提供商。

## 核心概念

### 六个核心功能

1. **提供商路由** —— OpenAI、Anthropic、Gemini、自托管等，统一在一个API后面。
2. **故障转移** —— 在429、5xx或质量失败时，在其他地方重试。
3. **重试** —— 指数退避，有限尝试次数。
4. **速率限制** —— 按租户、按密钥、按模型。
5. **密钥引用** —— 运行时从保险库拉取凭证（永远不会出现在应用中）。
6. **可观测性** —— OTel + GenAI属性（阶段17·13）+ 成本归属。
7. **护栏** —— PII脱敏、越狱检测、允许主题过滤器。

### LiteLLM — MIT开源，Python

- 100+提供商，兼容OpenAI，路由器配置，故障转移，基本可观测性。
- 在Kong的基准测试中约2000 RPS时崩溃；8 GB内存占用，持续负载下出现级联故障。
- 最佳适用场景：Python应用，<500 RPS，开发/测试网关，实验性路由。
- 成本：开源免费；云免费层存在。

### Portkey — 控制平面定位

- 2026年3月起成为Apache 2.0开源。护栏、PII脱敏、越狱检测、审计追踪。
- 每个请求增加20-40毫秒延迟。
- 生产层（含数据保留和SLA）每月$49。
- 最佳适用场景：需要护栏和可观测性捆绑的受监管行业。

### Kong AI Gateway — 扩展方案

- 基于Kong Gateway（成熟的API网关产品，lua+OpenResty）。
- Kong自身的基准测试（12 CPU等效）：比Portkey快228%，比LiteLLM快859%。
- 定价：$100/模型/月，Plus层最多5个。
- 最佳适用场景：已在用Kong；>1000 RPS；愿意付费。

### Bifrost (Maxim AI)

- 自动重试并支持可配置退避。
- OpenAI 429时回退到Anthropic是标准方案。
- 较新的进入者；商用。

### Cloudflare AI Gateway / Vercel AI Gateway

- 托管，零运维。基本重试和可观测性。
- 最佳适用场景：运行在Cloudflare/Vercel上的边缘JavaScript应用。
- 在护栏和速率限制方面不如Kong/Portkey全面。

### 自托管 vs 托管

数据驻留是驱动因素。医疗和金融默认自托管（LiteLLM或Portkey开源或Kong）。消费产品默认托管（Cloudflare AI Gateway）或中间层（Portkey托管）。混合方案：受监管租户用自托管，其他用托管。

### 延迟预算

- LiteLLM：典型延迟增加5-15毫秒。
- Portkey：延迟增加20-40毫秒。
- Kong：延迟增加3-8毫秒。
- Cloudflare/Vercel：边缘延迟增加1-3毫秒。

网关延迟直接增加TTFT。对于TTFT P99 < 100毫秒的SLA，选择Kong或Cloudflare。对于P99 < 500毫秒，任意。

### 速率限制语义很重要

简单的令牌桶适用于中等规模。多租户需要滑动窗口 + 突发许可 + 按租户分层。LiteLLM提供令牌桶；Kong提供滑动窗口；Portkey提供分层。

### 网关 + 可观测性 + 路由组合

阶段17·13（可观测性）+ 16（模型路由）+ 19（网关）在生产中是同一层。选择一个覆盖所有功能的工具，或仔细集成它们：2026年的大多数部署将Helicone（可观测性）或Portkey（护栏）与Kong（扩展）结合以实现角色分离。

### 你应该记住的数字

- LiteLLM：约2000 RPS时崩溃，8 GB内存。
- Portkey：延迟增加20-40毫秒；2026年3月起Apache 2.0。
- Kong：比Portkey快228%，比LiteLLM快859%。
- Kong定价：$100/模型/月，Plus层最多5个。
- Cloudflare/Vercel：边缘延迟1-3毫秒。

## 使用它

`code/main.py`模拟网关路由，带故障转移，跨3个提供商，注入429/5xx错误。报告延迟、重试率和故障转移命中率。

## 发布

本课生成`outputs/skill-gateway-picker.md`。根据规模、运维姿态、合规性、延迟预算，选择网关。

## 练习

1. 运行`code/main.py`。配置从OpenAI→Anthropic→自托管的故障转移。在5%的提供商错误率下，预期命中率是多少？
2. 你的SLA要求TTFT P99 < 200毫秒，基线为300毫秒。哪些网关在预算内？
3. 一个医疗客户要求自托管 + PII脱敏 + 审计。选择Portkey开源或Kong。
4. 比较LiteLLM与Kong：团队应该在什么RPS上限时迁移？
5. 为一个多租户SaaS设计速率限制策略：免费层、试用层、付费层。令牌桶还是滑动窗口？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  Gateway  |  "API broker"  |  位于应用和提供商之间的进程  |
|  LiteLLM  |  "MIT许可证的那个"  |  Python开源，100+供应商，在2K RPS时崩溃  |
|  Portkey  |  "护栏网关"  |  控制平面+可观测性，Apache 2.0  |
|  Kong AI Gateway  |  "可扩展的那个"  |  基于Kong Gateway构建，基准测试领先  |
|  Bifrost  |  "Maxim的网关"  |  重试+Anthropic回退方案  |
|  Cloudflare AI Gateway  |  "边缘托管"  |  边缘部署的托管网关，零运维  |
|  PII脱敏  |  "数据清洗"  |  发送到模型前的正则+NER掩码  |
|  越狱检测  |  "提示注入防护"  |  对用户输入进行分类器检测  |
|  审计追踪  |  "监管日志"  |  每次LLM调用的不可变记录  |
|  令牌桶  |  "简单限流"  |  基于补充的速率限制器  |
|  滑动窗口  |  "精确限流"  |  时间窗口速率限制器；更好的公平性  |

## 延伸阅读

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)
