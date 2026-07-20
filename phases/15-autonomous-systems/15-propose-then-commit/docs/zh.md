# 人在回路中：提议-然后-提交

> 2026年关于HITL的共识是具体的。它并非“代理询问，用户点击批准”。而是提议-然后-提交：提议的动作持久化到具有幂等键(Idempotency Key)的持久存储中；向审核者展示，包含意图、数据沿袭(Data Lineage)、触达的权限、爆炸半径(Blast Radius)和回滚计划；仅在收到肯定确认后提交；执行后验证以确认副作用确实发生。LangGraph的`interrupt()`配合PostgreSQL检查点、Microsoft Agent Framework的`RequestInfoEvent`以及Cloudflare的`waitForApproval()`都实现了相同的模式。典型的失败模式是橡皮图章式批准：未审核就点击“批准”。文档中记载的缓解措施是带有明确检查清单的挑战-应答(Challenge-and-Response)。

**类型：** 学习
**语言：** Python (stdlib, 提议-然后-提交状态机，带幂等性)
**前置条件：** 第15阶段·12（持久执行），第15阶段·14（绊线）
**时间：** 约60分钟

## 问题

代理执行一个动作。用户必须决定：批准还是不批准。如果决策是瞬间完成的，那很可能不是真正的审核。如果决策是有结构的，虽然慢但可靠。工程问题是如何让结构化审核成为阻力最小的路径。

2023年时代的HITL模式是一个同步提示：“代理想要向X发送正文为Y的邮件——批准吗？”用户点击批准。每个人都觉得系统安全。实际上，这种表面形式被严重橡皮图章化：用户批准很快，批准预测很少，当代理出错时，审计追踪显示用户无法回忆的长期批准历史。

2026年的模式——提议-然后-提交——将HITL迁移到持久基座上，附加结构化元数据，并要求积极提交。每个托管代理SDK都提供了一个版本：LangGraph `interrupt()`、Microsoft Agent Framework `RequestInfoEvent`、Cloudflare `waitForApproval()`。API名称不同，但形状相同。

## 核心概念

### 提议-然后-提交状态机

1. **提议。** 代理生成提议动作。持久化到持久存储（PostgreSQL、Redis、Durable Object）。包括：
   - 意图（代理为何执行此动作）
   - 数据沿袭（什么数据源导致了此提议）
   - 触达的权限（哪些范围/文件/端点）
   - 爆炸半径（最坏情况是什么）
   - 回滚计划（如果提交了，如何撤销）
   - 幂等键（每个提议唯一；重新提交返回相同记录）
2. **展示。** 审核者看到带有所有元数据的提议。审核者是一个人（而非代理自我审核）。
3. **提交。** 肯定确认。动作执行。
4. **验证。** 执行后，读取并确认副作用。如果验证步骤失败，系统处于已知的坏状态，警报触发。

### 幂等键

没有幂等键，临时故障后的重试可能导致已批准的动作被执行两次。具体例子：用户批准“从A转账100美元到B”。网络波动。工作流重试。用户已批准一次，但转账执行了两次。幂等键将批准绑定到一个唯一的副作用上；第二次执行是空操作。

这与Stripe和AWS API使用的幂等模式相同。将其用于代理批准在Microsoft Agent Framework文档中明确说明。

### 持久性：为什么批准比进程更持久

批准等待室是代理不拥有的状态。工作流暂停（第12课）。当批准到达时，工作流从该点精确恢复。这就是为什么LangGraph将`interrupt()`与PostgreSQL检查点配对，而不仅仅是内存状态——两天后的批准仍然能找到完好的工作流。

### 橡皮图章式批准与挑战-应答缓解措施

HITL的默认UI（“批准”/“拒绝”按钮）产生快速批准，没有真正的审核。文档中记载的缓解措施：一个挑战-应答检查清单，要求在启用批准按钮之前对特定问题给出肯定答案。具体形式：

- "你理解这个动作会触达什么资源吗？[ ]"
- "你已确认爆炸半径可接受吗？[ ]"
- "如果失败，你有回滚计划吗？[ ]"

不是为了官僚主义而官僚主义——而是一个强制函数。无法打勾的审核者要么要求澄清（升级），要么拒绝（安全默认）。Anthropic的代理安全研究明确将清单驱动的HITL列为橡皮图章式批准模式的缓解措施。

### 什么算作重大后果

并非每个动作都需要提议-然后-提交。2026年的指导：

- **重大后果动作**（始终HITL）：不可逆写入、金融交易、对外通信、生产数据库变更、破坏性文件系统操作。
- **可逆动作**（有时HITL）：对本地文件的编辑、暂存环境变更、具有明确回滚的可逆写入。
- **读取和检查**（从不HITL）：读取文件、列出资源、调用只读API。

### 执行后验证

“提交已运行”不等于“副作用发生了”。网络分区和竞态条件可能导致工作流认为成功，而后端并未持久化。验证步骤在提交后重新读取目标资源以确认。这与带有`RETURNING`子句的数据库事务或AWS `GetObject`(在`PutObject`之后)的模式相同。

### 欧盟AI法案第14条

第14条要求对欧盟高风险AI系统进行有效的人类监督。“有效”不是装饰性的。监管语言明确排除了橡皮图章模式。带有挑战-应答的提议-然后-提交是在Microsoft Agent治理工具包合规文档中能经受第14条审查的形式。

## 使用它

`code/main.py`用stdlib Python实现了提议-然后-提交状态机。持久存储是一个JSON文件。幂等键是(thread_id, action_signature)的哈希。驱动程序模拟了三种情况：干净的批准流程、临时故障后的重试（不能双重执行）、以及橡皮图章默认与挑战-应答流程。

## 发布

`outputs/skill-hitl-design.md`审查提议的HITL工作流是否符合提议-然后-提交形状，并标记缺失的元数据、幂等性、验证或挑战-应答层。

## 练习

1. 运行`code/main.py`。确认已批准提议的重试使用持久记录且不重新执行。然后更改幂等键以包含时间戳，并展示重试导致双重执行。

2. 扩展提议记录，添加一个`rollback`字段。模拟一个执行后验证步骤失败的情况。展示回滚自动触发。

3. 阅读Microsoft Agent Framework的`RequestInfoEvent`文档。识别API包含但玩具引擎缺失的一个元数据字段。添加它并解释它所防范的内容。

4. 为特定动作（例如“发布到公共Twitter账户”）设计一个挑战-应答检查清单。审核者必须回答哪三个问题？为什么是这三个？

5. 选择一个同步的“批准？”提示就足够的情况（无需持久化存储）。解释原因，并说明你所接受的风险类别。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|---|---|---|
| 提议-然后-提交  |  “两阶段批准”  |  持久化的提议 + 正向提交 + 验证 |
| 幂等键  |  “重试安全令牌”  |  每个提议唯一；第二次执行无操作 |
| 数据谱系  |  “来源追溯”  |  导致该提议的特定源内容 |
| 爆炸半径  |  “最坏情况”  |  操作出错时的影响范围 |
| 橡皮图章  |  “快速批准”  |  未经认真审查就点击“批准” |
| 挑战与响应  |  “强制检查清单”  |  审阅者必须积极确认特定问题 |
| RequestInfoEvent  |  “MS Agent 框架原语”  |  带有结构化元数据的持久化人工介入请求 |
| `interrupt()` / `waitForApproval()`  |  “框架原语”  |  相同形态的 LangGraph / Cloudflare 等价物 |

## 延伸阅读

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent`，持久化批准。
- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent` 和 Durable Objects。
- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — 将人工介入作为长期风险的缓解措施。
- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — 高风险系统的监管基线。
- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — 围绕监督的宪法框架。
