# 安全性——密钥、API密钥轮换、审计日志、护栏

> 通过集中式密钥存储（HashiCorp Vault、AWS Secrets Manager、Azure Key Vault）消除密钥蔓延。切勿将凭证存储在配置文件、版本控制系统中的环境变量文件或电子表格中。使用IAM角色代替静态密钥；CI/CD使用OIDC。AI网关模式是2026年的解决方案：应用→网关→模型提供者，网关在运行时从密钥存储中拉取凭证。在密钥存储中轮换后，所有应用在几分钟内更新——无需重新部署，无需在Slack中询问“谁有新密钥”。轮换策略≤90天；每次提交使用TruffleHog/GitGuardian/Gitleaks扫描。零信任：多因子认证、单点登录、基于角色的访问控制/基于属性的访问控制、短期令牌、设备合规性。PII清洗使用实体识别在转发前对受保护健康信息/个人身份信息进行掩码；一致性令牌化（Mesh方法）将敏感值映射到稳定占位符，使大语言模型保留代码/关系语义。网络出站：大语言模型服务在专用虚拟私有云/虚拟网络子网中，仅白名单`api.openai.com`、`api.anthropic.com`等；阻止所有其他出站流量。2026年事件驱动：Vercel供应链攻击通过被泄露的CI/CD凭证，从数千个客户部署中窃取环境变量。

**类型：** 学习
**语言：** Python（标准库，简单PII清洗器+审计日志写入器）
**前置条件：** 阶段17·19（AI网关）、阶段17·13（可观测性）
**时间：** 约60分钟

## 学习目标

- 列举四种密钥管理反模式（版本控制系统中的配置文件、硬编码环境变量、电子表格、静态密钥）并说明其替代方案。
- 解释2026年生产标准的AI网关从密钥存储拉取模式。
- 实现一个具有一致性令牌化（相同值→相同占位符，使语义保留）的PII清洗器。
- 说明2026年Vercel供应链事件及其对CI/CD凭证管理的教训。

## 问题

一名实习生提交了带有API密钥的`.env`。他们很快删除了它。但密钥已经在git历史中——GitGuardian扫描捕获到它，你的轮换流程是“在Slack通知团队，更新40个配置文件，重新部署所有服务。”8小时后，一半服务已上线，一半在等待部署窗口。

另外，用户提示中包含“我的社会安全号码是123-45-6789。”提示发送给OpenAI。你有业务伙伴协议，但内部政策是在转发前对PII进行掩码。你没有执行。

另外，你的EKS集群的大语言模型Pod可以访问任何互联网主机。有人通过向攻击者控制的域进行DNS查找窃取了数据。没有任何东西阻止它。

大语言模型服务的安全性必须解决所有三个向量：基于密钥存储的凭证、PII清洗、网络出站过滤、审计日志。

## 核心概念

### 集中式密钥存储+IAM角色拉取

**密钥存储：** HashiCorp Vault、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager。单一事实来源。

**IAM角色：** 应用/网关通过其IAM身份进行身份验证，而不是静态密钥。密钥存储返回令牌生命周期内的密钥。

**AI网关模式：** 网关在请求时从密钥存储拉取`OPENAI_API_KEY`。在密钥存储中轮换；下一个请求获取新密钥。无需重新部署。

### 轮换策略≤90天

所有API密钥、密钥存储根令牌、CI/CD凭证。尽可能自动轮换。手动轮换需记录并跟踪。

### 密钥扫描

- **TruffleHog** — 在提交上使用正则表达式+熵检测。
- **GitGuardian** — 商业产品，高精度。
- **Gitleaks** — 开源，在CI中运行。

每次提交运行。如果检测到新密钥则阻止PR。

### 零信任姿态

- 所有账户需要多因子认证。
- 通过安全断言标记语言/OIDC实现单点登录。
- 基于角色的访问控制或基于属性的访问控制实现细粒度访问。
- 短期令牌（小时级别，而非天级别）。
- 设备合规性——仅允许带有磁盘加密的公司设备。

### PII/PHI清洗

在提示离开你的基础设施之前：

1. 实体识别（spaCy命名实体识别、Presidio、商业产品）。
2. 对匹配的实体进行掩码：`"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`。
3. 一致性令牌化（Mesh方法）：相同值映射到相同占位符，使大语言模型保留关系。
4. 可选的对大语言模型响应的反向映射。

静态正则过滤器捕获基本模式；命名实体识别捕获更多。两者都使用。

### 输入+输出护栏

输入：阻止已知越狱、禁止主题；按用户进行速率限制。

输出：正则表达式清洗泄露的密钥（API密钥模式、拒绝上下文的电子邮件模式），分类器检测策略违规。

### 网络出站白名单

大语言模型服务在专用子网中：
- 白名单：`api.openai.com`、`api.anthropic.com`、向量数据库端点、密钥存储端点。
- 其他所有流量：丢弃。
- DNS通过仅允许列表的解析器（避免DNS隧道窃取）。

### 审计日志

每次LLM调用的不可变日志，包含：
- 时间戳。
- 用户/租户。
- 提示词哈希（为隐私保护不记录原始提示词）。
- 模型+版本。
- 令牌数。
- 成本。
- 响应哈希。
- 任何触发防护栏的记录。

根据合规要求保留（SOC 2为1年，HIPAA为6年）。

### 2026年Vercel事件

供应链攻击：受损的CI/CD凭证泄露了数千个客户部署中的环境变量。教训：CI/CD凭证等同于生产环境凭证。应存储在保险库中。严格限定范围。频繁轮换。

### 你应该记住的数字

- 轮换策略：≤ 90天。
- 每次提交时扫描：TruffleHog / GitGuardian / Gitleaks。
- Vercel 2026：CI/CD凭证受损 → 数千个客户环境变量泄露。
- 审计日志保留：SOC 2 = 1年，HIPAA = 6年。

## 使用它

`code/main.py`实现了一个玩具级PII擦洗器，具有一致的标记化和仅追加审计日志。

## 发布

本节课生成`outputs/skill-llm-security-plan.md`。根据监管范围和当前状态，规划保险库迁移、擦洗器、出口、审计日志。

## 练习

1. 运行`code/main.py`。发送两个引用同一SSN的提示词。确认两者得到相同的占位符。
2. 设计vLLM-on-EKS部署的网络出口策略，该部署调用OpenAI + Anthropic + Weaviate。
3. 你在Git历史中发现了一个密钥（2年前）。正确的响应是什么——轮换密钥、擦除历史记录，还是两者都做？请论证。
4. 你的审计日志每天增长10 GB。设计保留层级（热数据30天，温数据12个月，冷数据6年）。
5. 讨论反向标记化（将真实值替换回LLM响应）是否值得增加复杂性，而不是保持占位符可见。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  Vault  |  "secrets store"  |  集中式凭证管理服务  |
|  IAM role  |  "identity-based auth"  |  由应用假设的角色；返回短期凭证  |
|  OIDC for CI/CD  |  "cloud-issued tokens"  |  CI中没有静态密钥——通过OIDC进行身份验证  |
|  TruffleHog / GitGuardian / Gitleaks  |  "secret scanners"  |  提交时的秘密检测  |
|  RBAC / ABAC  |  "access control"  |  基于角色的访问控制 vs 基于属性的访问控制  |
|  PII scrubbing  |  "data masking"  |  移除或标记化敏感实体  |
|  Consistent tokenization  |  "stable placeholders"  |  相同值 → 每次相同的令牌  |
|  Mesh approach  |  "Mesh tokenization"  |  语义保持的标记化模式  |
|  Egress whitelist  |  "outbound allowlist"  |  仅允许可达的域名  |
|  Audit log  |  "immutable history"  |  仅追加的合规记录  |

## 延伸阅读

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security) — PII检测和匿名化。
- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
