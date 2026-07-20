# 顶点项目 06 — 面向 Kubernetes 的 DevOps 故障排查智能体

> AWS的DevOps Agent已正式上线，Resolve AI发布了其K8s playbooks，NeuBird演示了语义监控，Metoro将AI SRE与每服务SLO关联起来。生产形态已确定：告警webhook触发，代理读取遥测数据，遍历K8s对象图，对根因假设进行排序，并在Slack中发布带有审批按钮的简报。默认只读。所有修复操作均需人工审批。本毕业设计就是实现这样一个代理，在20个合成事件上进行评估，并与AWS的Agent在三个共享案例上进行比较。

**类型：** 毕业设计
**语言：** Python（代理）、TypeScript（Slack集成）
**前置条件：** 阶段11（LLM工程）、阶段13（工具与MCP）、阶段14（代理）、阶段15（自主）、阶段17（基础设施）、阶段18（安全）
**涉及阶段：** P11 · P13 · P14 · P15 · P17 · P18
**时间：** 30小时

## 问题

2025-2026年的SRE叙述变成：“AI代理负责事件分类，人工审批修复操作。”AWS DevOps Agent、Resolve AI、NeuBird、Metoro、PagerDuty AIOps都在生产中采用这种形态。代理读取Prometheus指标、Loki日志、Tempo追踪、kube-state-metrics以及K8s对象的知识图谱。它在五分钟内生成带有遥测引证的排序根因假设。未经Slack上的人工明确批准，绝不执行破坏性命令。

大部分难点在于范围界定和安全，而非推理。代理需要默认只读的RBAC面、加固的MCP工具服务器，以及所有考虑过和执行过的命令的审计日志。它需要知道自己何时超出能力范围并上报。而且运行成本必须足够低，以免OOM-kill级联产生5000美元的代理费用。

## 概念

代理基于知识图谱运行。节点是K8s对象（Pod、Deployment、Service、Node、HPA、PVC）以及遥测源（Prometheus序列、Loki流、Tempo追踪）。边编码了归属关系（Pod -> ReplicaSet -> Deployment）、调度关系（Pod -> Node）和观测关系（Pod -> Prometheus序列）。图谱通过kube-state-metrics同步保持最新，并在每次告警时重新采样。

当告警触发时，代理从受影响的对象出发进行根因分析。它遍历边，拉取相关的遥测切片（最近15分钟），并草拟假设。假设根据证据排序：支持的遥测引证数量、引证的新近度、特异性。排名前三的假设连同图谱路径可视化和修复操作审批按钮一起发送到Slack。

修复操作被门控。允许的默认操作为只读。破坏性操作（缩容、回滚、删除Pod）需要Slack审批；ArgoCD回滚钩子需要代理从未持有的身份验证令牌。审计日志记录代理*考虑过*的每个命令——而不仅仅是执行过的——因此审查过程能捕获未遂事件。

## 架构

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI receiver
           |
           v
   LangGraph root-cause agent
           |
           +---- read-only MCP tools ----+
           |                             |
           v                             v
   K8s knowledge graph              telemetry slices
     (Neo4j / kuzu)              Prometheus, Loki, Tempo
   ownership + scheduling          last 15m, scoped
           |
           v
   hypothesis ranking (evidence weight)
           |
           v
   Slack brief + approval buttons
           |
           v (approved)
   ArgoCD rollback hook / PagerDuty escalate
           |
           v
   audit log: considered vs executed, every command
```

## 技术栈

- 可观测性来源：Prometheus、Loki、Tempo、kube-state-metrics
- 知识图谱：Neo4j（托管）或kuzu（嵌入式）存储K8s对象+遥测边
- 代理：LangGraph，每个工具带有允许列表，默认只读
- 工具传输：通过StreamableHTTP的FastMCP；破坏性工具单独服务器，位于审批门之后
- 模型：Claude Sonnet 4.7用于根因推理，Gemini 2.5 Flash用于日志摘要
- 修复：ArgoCD回滚webhook、PagerDuty升级、Slack审批卡片
- 审计：仅追加的结构化日志（考虑、执行、批准、结果）
- 部署：K8s部署，使用自身狭义的RBAC角色；独立命名空间

## 动手构建

1. **图谱摄取。** 每30秒将kube-state-metrics同步到Neo4j/kuzu。节点：Pod、Deployment、Node、Service、PVC、HPA。边：OWNED_BY、SCHEDULED_ON、EXPOSES、MOUNTS、SCALES。遥测覆盖边：OBSERVED_BY（Pod被Prometheus序列观测）。

2. **告警接收器。** FastAPI端点，接受PagerDuty或Alertmanager webhook。提取受影响的对象和SLO违规。

3. **只读工具面。** 通过FastMCP封装kubectl、Prometheus查询、Loki logql、Tempo traceql。每个工具都有狭义的RBAC动词（"get"、"list"、"describe"）。默认服务器中没有"delete"、"exec"、"scale"。

4. **根因代理。** LangGraph包含三个节点：`sample`拉取最近15分钟的遥测切片，`walk`查询邻近对象的图谱，`hypothesize`草拟带有遥测引证的排序根因候选。

5. **证据评分。** 每个假设的分数 = 新近度 * 特异性 * 图谱路径长度倒数 * 引证数量。返回前三名。

6. **Slack简报。** 发布附件，包含假设、图谱路径可视化（服务端渲染的子图图像），以及最多一个修复操作的审批按钮。

7. **修复门控。** 破坏性工具（缩容、回滚、删除）位于第二个MCP服务器上，位于审批令牌之后。只有在Slack卡片被人类批准后，代理才能调用它们。

8. **审计日志。** 仅追加的JSONL：对于每个候选命令，记录是否被考虑、是否被执行、由谁批准。每日发送到S3。

9. **合成事件套件。** 构建20个场景：OOMKill级联、DNS抖动、HPA震荡、PVC填满、噪声邻居、有问题的Sidecar、错误的ConfigMap发布、证书轮换、镜像拉取回退等。根据根因准确性和假设生成时间对代理评分。

## 使用它

```
webhook: alert.pagerduty.com -> checkout-api SLO breach, error rate 14%
[graph]   affected: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    neighbors: ReplicaSet checkout-api-abc, Service checkout-api,
           recent rollout 14m ago
[sample]  prometheus error_rate 14%, up-trend; loki 500s on /api/v2/pay
[hypo]    #1 bad rollout: latest image checkout-api:v2.41 fails /healthz
          citations: deploy.yaml (rev 42), prometheus errorRate, loki 500 stack
[slack]   [ROLL BACK to v2.40]  [ESCALATE]  [IGNORE]
          (approval required; agent does not roll back unilaterally)
```

## 发布

`outputs/skill-devops-agent.md`是交付物。给定K8s集群和告警源，代理生成排序的根因假设和Slack门控的修复流程。

|  权重  |  标准  |  衡量方式  |
|:-:|---|---|
|  25  |  场景套件上的RCA准确率  |  在20个合成事件中≥80%的根因正确 |
|  20  |  安全性  |  审计日志中破坏性操作守卫在没有Slack审批时从不触发 |
|  20  |  假设生成时间  |  从告警到Slack简报的中位数时间低于5分钟 |
|  20  |  可解释性  |  每个假设都有图谱路径和遥测引证 |
|  15  |  集成完整性  |  PagerDuty、Slack、ArgoCD、Prometheus端到端工作 |
|  **100**  |   |   |

## 练习

1. 在与AWS DevOps Agent演示相同的三个事件上运行你的代理。发布对比结果。报告代理在哪些方面存在分歧。

2. 添加一个“未遂”审计，标记代理*考虑过*的、但未经批准就会具有破坏性的任何命令。测量一周内的未遂率。

3. 将假设模型从Claude Sonnet 4.7替换为自托管的Llama 3.3 70B。测量RCA准确率差值以及每个事件的美元成本。

4. 构建一个因果过滤器：区分相关的遥测尖峰与真正的根因。在20个场景标签上训练一个小型分类器。

5. 添加回滚预演：针对具有相同清单的预发集群进行ArgoCD回滚。在Slack审批按钮之前验证实时集群中的回滚计划。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
|  K8s知识图谱  |  "集群图"  |  节点 = K8s对象 + 遥测序列；边 = 归属、调度、观测 |
| 默认只读  |  "作用域RBAC"  |  代理的服务账户仅具有 get/list/describe 动词；破坏性动词位于需要审批的独立服务器中 |
| 审计日志  |  "考虑与执行"  |  每个候选命令的仅追加记录，包括是否执行、由谁审批 |
| 假设排序  |  "证据分数"  |  时效性 × 特异性 × 图路径长度倒数 × 引用次数 |
| Slack审批卡片  |  "人在回路关口"  |  带有修复按钮的交互式Slack消息；代理必须等待人工点击后才能继续 |
| 遥测引用  |  "证据指针"  |  支持某个主张的Prometheus查询、Loki选择器或Tempo跟踪URL |
| MTTR  |  "修复时间"  |  从告警触发到SLO恢复的挂墙时间 |

## 延伸阅读

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 2026年规范参考
- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 竞争对手参考
- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 语义图方法
- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 以SLO为先的生产框架
- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 集群状态来源
- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 参考代理编排器
- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — Python MCP服务器框架
- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 有闸门的修复目标
