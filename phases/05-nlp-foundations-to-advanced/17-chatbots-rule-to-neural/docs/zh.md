# 聊天机器人——从基于规则到神经网络到LLM智能体

> ELIZA通过模式匹配回复。DialogFlow映射意图。GPT从权重中回答。Claude运行工具并验证。每个时代解决了前一个时代最严重的失败。

**类型：** 学习
**语言：** Python
**先修课程：** 阶段5 · 13（问答），阶段5 · 14（信息检索）
**时间：** 约75分钟

## 问题

用户说“我想更改我的航班。”系统必须弄清楚他们想要什么，缺少什么信息，如何获取这些信息，以及如何完成操作。然后用户说“等等，如果我取消呢？”系统必须记住上下文，切换任务，并保持状态。

对话对于机器学习系统来说是困难的。输入是开放式的。输出必须在多轮对话中保持连贯。系统可能需要对外部世界采取行动（更改航班、扣款）。每一步错误都对用户可见。

聊天机器人架构经历了四种范式，每种范式都是因为前一种的失败太明显而被引入的。本课按顺序逐一讲解。2026年的生产环境是后两种的混合体。

## 核心概念

![Chatbot evolution: rule-based → retrieval → neural → agent](../assets/chatbot.svg)

**基于规则（ELIZA, AIML, DialogFlow）。** 手工编写的模式匹配用户输入并生成回复。意图分类器路由到预定义流程。槽填充状态机收集所需信息。在其设计的狭窄范围内工作得非常出色。一超出范围就立即失败。仍然用于安全性要求极高的领域（银行认证、机票预订），在这些领域幻觉不被容忍。

**基于检索。** 一种FAQ式系统。编码每一对（话语，回复）。运行时，编码用户消息并检索最接近的存储回复。可以看作是Zendesk经典的“相似文章”功能。处理释义比规则更好。没有生成，因此没有幻觉。

**神经网络（seq2seq）。** 在对话日志上训练的编码器-解码器。从头生成回复。流畅但容易生成通用输出（“我不知道”）和事实偏离。从来不能可靠地保持在主题上。这是谷歌、Facebook和微软在2016-2019年都出现令人失望的聊天机器人的原因。

**LLM智能体。** 一个语言模型被封装在一个循环中，进行规划、调用工具和验证结果。不是带有长提示的聊天机器人。一个智能体循环：规划→调用工具→观察结果→决定下一步。以检索为先的接地（RAG）防止幻觉。工具调用使其能够实际执行操作。这就是2026年的架构。

这四种范式不是依次替换的关系。2026年的生产聊天机器人会路由通过所有四种：基于规则用于身份验证和破坏性操作，检索用于FAQ，神经网络生成用于自然措辞，LLM智能体用于模糊的开放式查询。

## 动手构建

### 第1步：基于规则的模式匹配

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

20行代码的ELIZA。反射技巧（“我感到难过”→“你为什么感到难过”）是Weizenbaum 1966年的经典心理治疗师演示。仍然具有启发性。

### 第2步：基于检索（FAQ）

这个说明性代码片段需要`pip install sentence-transformers`（它会引入torch）。本课的可运行`code/main.py`使用标准库的Jaccard相似度，因此本课无需外部依赖即可运行。

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

基于阈值的拒绝是关键设计选择。如果最佳匹配不够接近，返回`None`并让系统升级。

### 第3步：神经网络生成（基线）

使用一个小的指令微调的编码器-解码器（FLAN-T5）或一个微调的对话模型。2026年单独使用时无法投入生产（矛盾、偏离主题、事实性胡说），但作为混合系统的一部分用于自然措辞。DialoGPT风格的仅解码器模型需要明确的分轮符和EOS处理才能产生连贯的回复；FLAN-T5的文本到文本流水线开箱即用，适合教学示例。

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### 第4步：LLM智能体循环

2026年的生产形态：

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

需要命名三个东西。工具是LLM可以调用的可调用函数。当LLM返回最终答案而不是工具调用时，循环终止。步骤预算防止在模糊任务上无限循环。

实际生产环境还会添加：以检索为先的接地（在每次LLM调用前注入相关文档），护栏（拒绝未经确认的破坏性操作），可观测性（记录每一步），以及评估（自动化检查智能体行为是否符合规范）。

### 第5步：混合路由

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

模式：对于任何破坏性操作使用确定性规则，对于现成FAQ使用检索，对于其他所有情况使用LLM智能体。这就是2026年客户支持系统的形态。

## 使用它

2026年技术栈：

|  用例  |  架构  |
|---------|---------------|
|  预订、支付、认证  |  基于规则的状态机 + 槽填充  |
|  客户支持FAQ  |  对精心策划的答案进行检索  |
|  开放式帮助聊天  |  带有RAG+工具调用的LLM智能体  |
|  内部工具/IDE助手  |  带有工具调用（搜索、读取、写入）的LLM智能体  |
|  陪伴/角色聊天机器人  |  具有角色系统提示和知识检索的微调LLM  |

在生产环境中始终使用混合路由。没有任何单一架构能够妥善处理所有请求。路由层本身通常是一个小型意图分类器。

## 仍会发布的故障模式

- **虚假执行。** 大语言模型(LLM)智能体声称完成了它并未执行的动作。缓解措施：验证结果，记录工具调用，绝不允许LLM在未成功返回工具的情况下声称已完成某操作。
- **提示注入。** 用户插入覆盖系统提示的文本。位列OWASP 2025年LLM应用十大安全风险之首。两种形式：直接注入（粘贴到聊天中）和间接注入（隐藏在智能体读取的文档、邮件或工具输出中）。

  攻击成功率因场景而异。在通用工具使用和编程基准测试中，前沿模型的实测成功率约为0.5%至8.5%。特定高风险配置（针对AI编程智能体的自适应攻击、脆弱编排）可达约84%。生产环境中的常见漏洞与暴露(CVE)包括 EchoLeak（CVE-2025-32711，CVSS 9.3）——微软365 Copilot中由攻击者控制的邮件触发的零点击数据泄露漏洞。

  缓解措施：在整个循环中将用户输入视为不可信；在工具调用前进行清洗；将工具输出与主提示隔离；采用计划-验证-执行(PVE)模式，智能体先规划，然后在执行前针对计划验证每个动作（这可以防止工具结果注入新的未计划动作）；对破坏性操作要求用户确认；对工具作用域应用最小权限原则。

  任何程度的提示工程都无法完全消除这一风险。需要外部运行时防御层（如LLM Guard、白名单验证、语义异常检测）。
- **范围蔓延。** 智能体因工具调用返回了边缘相关信息而偏离任务。缓解措施：缩小工具合同；保持系统提示聚焦；增加对离任务率的评估。
- **无限循环。** 智能体持续调用同一工具。缓解措施：步骤预算、工具调用去重、设置LLM裁判判断“是否在取得进展”。
- **上下文窗口耗尽。** 长对话将较早轮次挤出上下文。缓解措施：摘要较早轮次，通过相似度检索相关过往轮次，或使用长上下文模型。

## 发布

保存为 `outputs/skill-chatbot-architect.md`：

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## 练习

1. **简单。** 实现上述基于规则的响应，为一个咖啡店点单机器人提供10种模式。测试边界情况：重复订单、修改、取消、歧义意图。
2. **中等。** 构建一个混合FAQ+LLM降级方案。为一个SaaS产品准备50条预置FAQ条目，LLM降级时检索文档站点。在100个真实支持问题上测量拒绝率和准确率。
3. **困难。** 实现上述包含三个工具（搜索、读取用户数据、发送邮件）的智能体循环。用50个测试场景（包括提示注入尝试）进行评估。报告离任务率、任务失败率以及任何注入成功情况。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  意图 | 用户想要什么 | 类别标签（预订航班、重置密码）。路由到处理器。 |
|  槽位 | 一条信息 | 机器人所需的参数（日期、目的地）。槽位填充是连续提问的序列。 |
|  RAG | 检索加生成 | 检索相关文档，然后基于此生成LLM的回复。 |
|  工具调用 | 函数调用 | LLM发出带有名称和参数的结构化调用。运行时执行，返回结果。 |
|  智能体循环 | 计划、行动、验证 | 控制器，交替运行LLM调用和工具调用，直至任务完成。 |
|  提示注入 | 用户攻击提示 | 试图覆盖系统提示的恶意输入。 |

## 延伸阅读

- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 最初的基于规则的聊天机器人论文。
- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — Google的晚期神经聊天机器人论文，就在LLM智能体接管之前。
- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 命名智能体循环模式的论文。
- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 2024年生产指南，2026年仍适用。
- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 提示注入论文。
- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 将提示注入列为最高安全威胁的排名。
- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 实际的编排层防御措施，包括计划-验证-执行和用户确认流程。
- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 来自间接提示注入的典型零点击数据泄露CVE。为何写权限智能体需要运行时防御的参考案例。
