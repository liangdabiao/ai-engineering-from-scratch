# 提示缓存(Prompt Caching)与上下文缓存(Context Caching)

> 你的系统提示有4,000个词元(token)。你的RAG上下文有20,000个词元。你每次请求都发送这两者。你也每次都支付——每次。提示缓存(Prompt Caching)让提供商在你这一侧保持前缀的热度，并在复用时按正常费率的10%收费。正确使用的话，它能将推理成本降低50–90%，并将首个词元的延迟降低40–85%。

**类型：** 构建
**语言：** Python
**前置条件：** 第11阶段 · 01（提示工程），第11阶段 · 05（上下文工程），第11阶段 · 11（缓存与成本）
**时间：** 约60分钟

## 问题

一个编码代理在每次对话轮次中向Claude发送相同的15,000词元系统提示。二十次轮次仅输入成本就达$3/M input tokens is $0.90——还未包括用户的任何实际消息。乘以每天10,000次对话，账单将达到9,000美元/天，而这些文字从未改变。

你不能在不影响质量的情况下缩小提示。你无法避免发送它——模型在每次轮次中都需要它。唯一的办法是停止为提供商已经见过的前缀支付全价。

这个办法就是提示缓存(Prompt Caching)。Anthropic于2024年8月推出了它（2025年推出了1小时扩展TTL变体），OpenAI在同年晚些时候将其自动化，Google在Gemini 1.5发布时推出了显式上下文缓存(Explicit Context Caching)，现在这三家公司都将其作为前沿模型的一级特性提供。

## 核心概念

![Prompt caching: write once, read cheap](../assets/prompt-caching.svg)

**机制。** 当请求的前缀与最近某个请求的前缀匹配时，提供商提供前一次运行的KV缓存(KV-cache)，而不是重新编码词元。你第一次支付少量写入溢价，之后每次享受大幅读取折扣。

**2026年的三种提供商风格。**

|  提供商 | API风格 | 命中折扣 | 写入溢价 | 默认TTL | 最小可缓存  |
|---------|-----------|--------------|---------------|-------------|---------------|
|  Anthropic | 内容块上的显式`cache_control`标记 | 输入减免90% | 25%附加费 | 5分钟（可延长至1小时） | 1,024词元（Sonnet/Opus），2,048（Haiku）  |
|  OpenAI | 自动前缀检测 | 输入减免50% | 无 | 最长1小时（尽力而为） | 1,024词元  |
|  Google (Gemini) | 显式`CachedContent` API | 按存储计费；读取约为正常的25% | 每词元·小时的存储费 | 用户设置（默认1小时） | 4,096词元（Flash），32,768（Pro）  |

**不变性。** 三者都只缓存前缀。如果请求之间的任何词元不同，则从第一个不同词元之后的所有内容都会未命中。将*稳定*部分放在顶部，*可变*部分放在底部。

### 缓存友好的布局

```
[system prompt]          <-- cache this
[tool definitions]       <-- cache this
[few-shot examples]      <-- cache this
[retrieved documents]    <-- cache if reused, else don't
[conversation history]   <-- cache up to last turn
[current user message]   <-- never cache (different every time)
```

违反顺序——将用户消息放在系统提示之上，在少样本之间插入动态检索——那么缓存永远不会命中。

### 盈亏平衡计算

Anthropic的25%写入溢价意味着缓存块必须至少被读取两次才能实现净节省。1次写入+1次读取平均每次请求成本0.675倍（节省32%）；1次写入+10次读取平均0.205倍（节省80%）。经验法则：缓存任何你预计在TTL内至少重复使用3次的内容。

## 动手构建

### 第1步：使用显式标记的Anthropic提示缓存

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control`标记告诉Anthropic将块存储5分钟。在该窗口内重复使用会命中；过期后重复使用则会再次写入。

**响应使用字段：**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # paid at 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # paid at 0.1x
```

在CI中检查这两个字段——如果`cache_read_input_tokens`在请求间保持为零，则你的缓存键正在漂移。

### 第2步：一小时扩展TTL

对于长时间运行的批处理作业，5分钟默认值会在作业之间过期。设置`ttl`：

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1小时TTL的写入溢价成本是2倍（比基线高50%而不是25%），但在任何重复使用前缀超过5次的批处理中能快速回本。

### 第3步：OpenAI自动缓存

OpenAI无需你进行任何配置。任何超过1,024词元且与最近请求匹配的前缀都会自动获得50%折扣。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # long and stable
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # the discounted portion
```

同样的缓存友好布局规则适用。有两件事会破坏OpenAI的缓存但不会破坏Anthropic的：更改`user`字段（用作缓存键组件）和重新排序工具。

### 第4步：Gemini显式上下文缓存

Gemini将缓存视为一个你可以创建并命名的一级对象：

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini对缓存的存续时间按每词元·小时收取存储费，读取速率约为正常输入速率的25%。当你连续数天在多个会话中重复使用同一个巨型提示时，这是正确的模式。

### 第5步：在生产环境中测量缓存命中率

请参阅`code/main.py`，了解一个模拟的三提供商记账器，用于跟踪写入/读取/未命中次数，并计算每1000次请求的综合成本。Gate根据目标命中率进行部署——在预热后，大多数生产环境Anthropic设置的读取占比应超过80%。

## 2026年仍存在的陷阱

- **动态时间戳位于顶部。** 在系统提示词的顶部放置`"Current time: 2026-04-22 15:30:02"`。每个请求都会未命中。将时间戳移至缓存断点下方。
- **工具重排序。** 以稳定顺序序列化工具——部署之间的字典重排会破坏每个命中。
- **自由文本近似重复。** “You are helpful.” 与 “You are a helpful assistant.” ——一个字节的差异即导致完全未命中。
- **过小的块。** Anthropic强制执行1024个令牌的下限（Haiku为2048个）。较小的块会静默不缓存。
- **成本仪表盘不透明。** 将“输入令牌”拆分为已缓存和未缓存。否则流量下降看起来像是缓存胜利。

## 使用它

2026年的缓存技术栈：

|  情况  |  选择  |
|-----------|------|
|  具有稳定10k+系统提示词的Agent，多次轮次  |  Anthropic的`cache_control`，5分钟TTL  |
|  重复使用前缀超过30分钟的批处理任务  |  Anthropic的`ttl: "1h"`  |
|  在GPT-5上的无服务器端点，无需自定义基础设施  |  OpenAI自动缓存（只需使前缀稳定且长）  |
|  大规模代码/文档语料库的多日复用  |  Gemini显式的`CachedContent`  |
|  跨提供商回退  | 保持可缓存前缀布局在各提供商间相同，以便任何命中都有效  |

与语义缓存（第11章第11节）结合用于用户消息层：提示词缓存(Prompt Caching)处理*令牌相同*的复用，语义缓存(Semantic Caching)处理*含义相同*的复用。

## 发布

保存 `outputs/skill-prompt-caching-planner.md`：

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## 练习

1. **简单。** 取一个10轮对话，带有5000个令牌的系统提示词，针对Claude。在不使用`cache_control`的情况下运行，然后使用。报告每种情况的输入令牌计费。
2. **中等。** 编写一个测试工具，给定提示词模板和请求日志，计算每个提供商的预期命中率和美元节省（Anthropic 5分钟，Anthropic 1小时，OpenAI自动，Gemini显式）。
3. **困难。** 构建一个布局优化器：给定一个提示词和一个标记为`cache_control`的字段列表，重写提示词，在最大缓存友好位置放置单个缓存断点，同时不丢失信息。在真实的Anthropic端点上验证。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  提示词缓存(Prompt Caching)  |  "让长提示词变便宜"  |  重复使用提供商端的KV缓存以匹配前缀；对重复的输入令牌享受50-90%折扣。  |
|  `cache_control`  |  "Anthropic标记"  |  内容块属性，声明“到此为止的所有内容都是可缓存的”；`{"type": "ephemeral"}`。  |
|  缓存写入(Cache Write)  |  "支付溢价"  |  填充缓存的第一个请求；在Anthropic上按输入速率约1.25倍计费，OpenAI免费。  |
|  缓存读取(Cache Read)  |  "折扣"  |  匹配前缀的后续请求；按10%（Anthropic）、50%（OpenAI）、约25%（Gemini）计费。  |
|  TTL  |  "存活时间"  |  缓存保持热度的秒数；Anthropic默认5分钟（可延长至1小时），OpenAI尽力而为最长1小时，Gemini用户设置。  |
|  延长TTL(Extended TTL)  |  "1小时Anthropic缓存"  |  `{"type": "ephemeral", "ttl": "1h"}`；写入溢价2倍，但对于批量复用是值得的。  |
|  前缀匹配(Prefix Match)  |  "为什么我的缓存未命中"  |  仅当从开始到断点的每个令牌字节相同时缓存才命中。  |
|  上下文缓存(Context Caching) (Gemini)  |  "显式的那种"  |  Google命名的、按存储计费的缓存对象；最适合多日复用大型语料库。  |

## 延伸阅读

- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`，1小时TTL，盈亏平衡表。
- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 自动前缀匹配。
- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control` API和存储定价。
- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 原始发布文章，含延迟数据。
- 第11章第05节（上下文工程）—— 如何切分提示词以使缓存生效。
- 第11章第11节（缓存与成本）—— 将提示词缓存与用户消息上的语义缓存配对。
- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 提示词缓存暴露给用户的KV缓存内存模型；解释了为什么缓存前缀的重新读取比重新计算便宜约10倍。
- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 预填充(Prefill)是提示词缓存加速的阶段；本文解释了为什么缓存命中时TTFT急剧下降而TPOT不受影响。
- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 提示词缓存与投机解码(Speculative Decoding)、Flash Attention和MQA/GQA并列为弯曲推理成本曲线的杠杆；阅读本文了解其他三种。
