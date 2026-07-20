# 缓存、速率限制与成本优化

> 大多数AI初创公司并非死于糟糕的模型，而是死于糟糕的单位经济。单次GPT-4o调用仅耗费零点几美分。一万个用户每天调用十次，仅输入token成本就高达250美元——而你还没向用户收取一分钱。那些存活下来的公司，将每一次API调用视为金融交易，而非函数调用。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段11 第09课（函数调用）
**时间：** 约45分钟
**相关内容：** 阶段11 · 15（提示缓存）——本课涵盖应用层缓存（语义缓存、精确哈希缓存、模型路由）。第15课涵盖供应商层提示缓存（Anthropic的cache_control、OpenAI的自动缓存、Gemini的CachedContent）。两者结合可实现50%-95%的成本降低。

## 学习目标

- 实现语义缓存，从缓存中提供重复或相似的查询，而不是发起新的API调用
- 计算各供应商的每次请求成本，并实现token感知的速率限制和预算警报
- 构建成本优化层，包含提示压缩、模型路由（贵 vs 便宜）和响应缓存
- 设计分层缓存策略，针对不同查询类型使用精确匹配、语义相似度和前缀缓存

## 问题

你构建了一个RAG聊天机器人。它工作得很完美。用户很喜欢它。

然后，账单来了。

GPT-5每百万输出token花费$5 per million input tokens and $15美元。Claude Opus 4.7每百万输出token花费$15 input / $75美元。Gemini 3 Pro每百万输出token花费$1.25 input / $5美元。GPT-5-mini为$0.25/$2美元。以下价格为示例；请始终查看供应商的最新定价页面。

以下是扼杀初创公司的数学：

- 10,000名日活跃用户
- 每名用户每天10次查询
- 每次查询1,000个输入token（系统提示词+上下文+用户消息）
- 每次回复500个输出token

**每日输入成本：** 10,000 x 10 x 1,000 / 1,000,000 x $2.50 = **$250/天**
**每日输出成本：** 10,000 x 10 x 500 / 1,000,000 x $10.00 = **$500/天**
**月度总计：** **$22,500/月**

这仅仅是LLM的费用。再加上嵌入、向量数据库托管、基础设施。你面对的是一个聊天机器人每月$30,000的成本。

残酷之处在于：这些查询中有40%-60%是近似重复的。用户用稍微不同的词问相同的问题。你的系统提示词——每次请求都一样——却每次都收费。RAG检索到的上下文文档在不同询问相同主题的用户间重复出现。

你正在为冗余计算支付全价。

## 核心概念

### LLM调用的成本剖析

每次API调用有五个成本组成部分。

```mermaid
graph LR
    A[User Query] --> B[System Prompt<br/>500-2000 tokens]
    A --> C[Retrieved Context<br/>500-4000 tokens]
    A --> D[User Message<br/>50-500 tokens]
    B --> E[Input Cost<br/>$2.50/1M tokens]
    C --> E
    D --> E
    E --> F[Model Processing]
    F --> G[Output Cost<br/>$10.00/1M tokens]
```

系统提示词是无声的杀手。一个1,500 token的系统提示词随每次请求发送，成本为$3.75 per million requests just for that prefix. At 100K requests per day, that is $375/天——$11,250/月——而这段文本从未改变。

### 供应商缓存：内置折扣

三大主要供应商在2026年都提供供应商端提示缓存，但机制不同。详情请参见阶段11·第15课。

|  供应商  |  机制  |  折扣  |  最小长度  |  缓存持续时间  |
|----------|-----------|----------|---------|----------------|
|  Anthropic  |  显式cache_control标记  |  缓存命中90%折扣（写入时多付25%）  |  1,024 tokens (Sonnet/Opus), 2,048 (Haiku)  |  默认5分钟；1小时延长（2倍写入溢价）  |
|  OpenAI  |  自动前缀匹配  |  缓存命中50%折扣  |  1,024 tokens  |  尽力而为，最长1小时  |
|  Google Gemini  |  显式CachedContent API  |  ~75%减少（加上存储）  |  4,096 (Flash) / 32,768 (Pro)  |  用户可配置TTL  |

**Anthropic的方法**是显式的。你用`cache_control: {"type": "ephemeral"}`标记提示词的各个部分。第一个请求支付25%的写入溢价。后续具有相同前缀的请求获得90%折扣。一个2,000 token的系统提示词，在缓存命中时成本为$0.005 normally costs $0.000625。超过100K次请求，每天节省$437.50。

**OpenAI的方法**是自动的。任何与先前请求匹配的提示词前缀可获得50%折扣。无需标记。权衡：折扣较少，控制较少，但实现工作量零。

### 语义缓存：你的自定义层

供应商缓存仅适用于完全相同的前缀。语义缓存处理更困难的情况：不同查询具有相同含义。

"退货政策是什么？"和"如何退货？"是不同的字符串，但意图相同。语义缓存将两个查询嵌入，计算余弦相似度，如果相似度超过阈值（通常0.92-0.95），则返回缓存的响应。

```mermaid
flowchart TD
    A[User Query] --> B[Embed Query]
    B --> C{Similar query<br/>in cache?}
    C -->|sim > 0.95| D[Return Cached Response]
    C -->|sim < 0.95| E[Call LLM API]
    E --> F[Cache Response<br/>with Embedding]
    F --> G[Return Response]
    D --> G
```

嵌入成本可以忽略不计。OpenAI的text-embedding-3-small每百万token花费$0.02。与一次完整的LLM调用相比，检查缓存的成本几乎为零。

### 精确缓存：哈希与匹配

对于确定性调用（temperature=0，相同模型，相同提示词），精确缓存更简单、更快。对完整提示词进行哈希，检查缓存，如果找到则返回。

这完美适用于：
- 系统提示词 + 固定上下文 + 相同的用户查询
- 相同工具定义的函数调用
- 同一文档被多次处理的批量处理

### 速率限制：保护你的预算

速率限制不仅关乎公平，更关乎生存。

**令牌桶算法：**每个用户得到一个容量为N个令牌的桶，以每秒R个令牌的速率补充。请求从桶中消耗令牌。如果桶为空，则请求被拒绝。这允许突发流量（一次性使用整个桶），同时强制执行平均速率。

**每用户配额：**按用户层级设置每日/每月的令牌限制。

| 层级  |  每日令牌限制  |  最大请求/分钟  |  模型访问 |
|------|------------------|------------------|-------------|
|  免费  |  50,000  |  10  |  仅GPT-4o-mini  |
|  专业  |  500,000  |  60  |  GPT-4o, Claude Sonnet  |
|  企业  |  5,000,000  |  300  |  全部模型  |

### 模型路由：将合适的模型用于合适的任务

并非每个查询都需要GPT-4o。

"商店几点关门？"不需要一个$10/M-output model. GPT-4o-mini at $0.60/M的输出就能完美处理。Claude Haiku以每百万输出令牌$1.25的价格也可以处理。一个简单的分类器将便宜的查询路由到便宜的模型，复杂的查询路由到昂贵的模型。

```mermaid
flowchart TD
    A[User Query] --> B[Complexity Classifier]
    B -->|Simple: lookup, FAQ| C[GPT-4o-mini<br/>$0.15/$0.60 per 1M]
    B -->|Medium: analysis, summary| D[Claude Sonnet<br/>$3.00/$15.00 per 1M]
    B -->|Complex: reasoning, code| E[GPT-4o / Claude Opus<br/>$2.50/$10.00+]
```

一个调优良好的路由器仅模型成本就能节省40-70%。

### 成本追踪：了解资金去向

你无法优化无法衡量的东西。记录每次API调用，包含以下信息：

- 时间戳
- 模型名称
- 输入令牌
- 输出令牌
- 延迟（毫秒）
- 计算成本（美元）
- 用户ID
- 缓存命中/未命中
- 请求类别

这些数据揭示了哪些功能成本高，哪些用户是重度消费者，以及缓存在哪里影响最大。

### 批处理：批量折扣

OpenAI的批处理API以50%的折扣异步处理请求。您可以提交最多50,000个请求的批次，结果在24小时内返回。

批处理适用于：
- 夜间文档处理
- 批量分类
- 评估运行
- 数据增强管道

不适用于：实时的面向用户的查询（延迟很重要）。

### 预算警报和断路器

断路器在达到限制时停止支出。如果没有它，一个错误或滥用行为可能在几小时内耗尽你的月度预算。

设置三个阈值：
1. **警告**（预算的70%）：发送警报
2. **限制**（预算的85%）：仅切换到更便宜的模型
3. **停止**（预算的95%）：拒绝新请求，仅返回缓存的响应

### 优化堆栈

按顺序应用这些技术。每一层都在前一层的基础上叠加效果。

| 层  |  技术  |  典型节省  |  实施工作量 |
|-------|-----------|----------------|----------------------|
| 1  |  提供者提示缓存  |  30-50%  |  低（添加缓存标记） |
| 2  |  精确缓存  |  10-20%  |  低（哈希+字典） |
| 3  |  语义缓存  |  15-30%  |  中（嵌入+相似度） |
| 4  |  模型路由  |  40-70%  |  中（分类器） |
| 5  |  速率限制  |  预算保护  |  低（令牌桶） |
| 6  |  提示压缩  |  10-30%  |  中（重写提示） |
| 7  |  批处理  |  符合条件的50%  |  低（批处理API） |

一个应用了第1-5层优化的RAG应用通常能将成本从$22,500/month to $4,000-6,000美元/月降低。这就是烧钱和建立业务之间的区别。

### 实际节省：优化前后对比

以下是一个为10,000日活跃用户提供服务的RAG聊天机器人的实际成本分解。

| 指标  |  优化前  |  优化后  |  节省 |
|--------|--------------------|--------------------|---------|
| 每月LLM成本  |  $22,500  |  $5,200  |  77% |
| 每次查询平均成本  |  $0.0075  |  $0.0017  |  77% |
| 缓存命中率  |  0%  |  52%  |  -- |
| 路由到小型模型的查询  |  0%  |  65%  |  -- |
| P95延迟  |  2,800ms  |  900ms（缓存命中：50ms）  |  68% |
| 每月嵌入成本  |  $0  |  $180  |  （新增成本） |
| 每月总成本  |  $22,500  |  $5,380  |  76% |

语义缓存的嵌入成本（180美元/月）在缓存命中的第一个小时内就能收回成本。

## 动手构建

### 步骤1：成本计算器

构建一个令牌成本计算器，了解主要模型的当前定价。

```python
import hashlib
import time
import json
import math
from dataclasses import dataclass, field


MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00, "cached_input": 1.25},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached_input": 0.075},
    "gpt-4.1": {"input": 2.00, "output": 8.00, "cached_input": 0.50},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "cached_input": 0.10},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40, "cached_input": 0.025},
    "o3": {"input": 2.00, "output": 8.00, "cached_input": 0.50},
    "o3-mini": {"input": 1.10, "output": 4.40, "cached_input": 0.55},
    "o4-mini": {"input": 1.10, "output": 4.40, "cached_input": 0.275},
    "claude-opus-4": {"input": 15.00, "output": 75.00, "cached_input": 1.50},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    "claude-haiku-3.5": {"input": 0.80, "output": 4.00, "cached_input": 0.08},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00, "cached_input": 0.3125},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60, "cached_input": 0.0375},
}


def calculate_cost(model, input_tokens, output_tokens, cached_input_tokens=0):
    if model not in MODEL_PRICING:
        return {"error": f"Unknown model: {model}"}
    pricing = MODEL_PRICING[model]
    non_cached = input_tokens - cached_input_tokens
    input_cost = (non_cached / 1_000_000) * pricing["input"]
    cached_cost = (cached_input_tokens / 1_000_000) * pricing["cached_input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total = input_cost + cached_cost + output_cost
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "input_cost": round(input_cost, 6),
        "cached_input_cost": round(cached_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total, 6),
    }
```

### 步骤2：精确缓存

对完整提示进行哈希处理，为相同请求返回缓存响应。

```python
class ExactCache:
    def __init__(self, max_size=1000, ttl_seconds=3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _hash(self, model, messages, temperature):
        key_data = json.dumps({"model": model, "messages": messages, "temperature": temperature}, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(self, model, messages, temperature=0.0):
        if temperature > 0:
            self.misses += 1
            return None
        key = self._hash(model, messages, temperature)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                self.hits += 1
                entry["access_count"] += 1
                return entry["response"]
            del self.cache[key]
        self.misses += 1
        return None

    def put(self, model, messages, temperature, response):
        if temperature > 0:
            return
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache, key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
        key = self._hash(model, messages, temperature)
        self.cache[key] = {
            "response": response,
            "timestamp": time.time(),
            "access_count": 1,
        }

    def stats(self):
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0,
            "cache_size": len(self.cache),
        }
```

### 步骤3：语义缓存

嵌入查询，当相似度超过阈值时返回缓存响应。

```python
def simple_embed(text):
    words = text.lower().split()
    vocab = {}
    for w in words:
        vocab[w] = vocab.get(w, 0) + 1
    norm = math.sqrt(sum(v * v for v in vocab.values()))
    if norm == 0:
        return {}
    return {k: v / norm for k, v in vocab.items()}


def cosine_similarity(a, b):
    if not a or not b:
        return 0.0
    all_keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in all_keys)
    return dot


class SemanticCache:
    def __init__(self, similarity_threshold=0.85, max_size=500, ttl_seconds=3600):
        self.entries = []
        self.threshold = similarity_threshold
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def get(self, query):
        query_embedding = simple_embed(query)
        now = time.time()
        best_match = None
        best_sim = 0.0
        for entry in self.entries:
            if now - entry["timestamp"] > self.ttl:
                continue
            sim = cosine_similarity(query_embedding, entry["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_match = entry
        if best_match and best_sim >= self.threshold:
            self.hits += 1
            best_match["access_count"] += 1
            return {"response": best_match["response"], "similarity": round(best_sim, 4), "original_query": best_match["query"]}
        self.misses += 1
        return None

    def put(self, query, response):
        if len(self.entries) >= self.max_size:
            self.entries.sort(key=lambda e: e["timestamp"])
            self.entries.pop(0)
        self.entries.append({
            "query": query,
            "embedding": simple_embed(query),
            "response": response,
            "timestamp": time.time(),
            "access_count": 1,
        })

    def stats(self):
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0,
            "cache_size": len(self.entries),
        }
```

### 步骤4：速率限制器

使用每用户配额的令牌桶速率限制器。

```python
class TokenBucketRateLimiter:
    def __init__(self):
        self.buckets = {}
        self.tiers = {
            "free": {"capacity": 50_000, "refill_rate": 500, "max_requests_per_min": 10},
            "pro": {"capacity": 500_000, "refill_rate": 5_000, "max_requests_per_min": 60},
            "enterprise": {"capacity": 5_000_000, "refill_rate": 50_000, "max_requests_per_min": 300},
        }

    def _get_bucket(self, user_id, tier="free"):
        if user_id not in self.buckets:
            tier_config = self.tiers.get(tier, self.tiers["free"])
            self.buckets[user_id] = {
                "tokens": tier_config["capacity"],
                "capacity": tier_config["capacity"],
                "refill_rate": tier_config["refill_rate"],
                "last_refill": time.time(),
                "request_timestamps": [],
                "max_rpm": tier_config["max_requests_per_min"],
                "tier": tier,
                "total_tokens_used": 0,
            }
        return self.buckets[user_id]

    def _refill(self, bucket):
        now = time.time()
        elapsed = now - bucket["last_refill"]
        refill = int(elapsed * bucket["refill_rate"])
        if refill > 0:
            bucket["tokens"] = min(bucket["capacity"], bucket["tokens"] + refill)
            bucket["last_refill"] = now

    def check(self, user_id, tokens_needed, tier="free"):
        bucket = self._get_bucket(user_id, tier)
        self._refill(bucket)
        now = time.time()
        bucket["request_timestamps"] = [t for t in bucket["request_timestamps"] if now - t < 60]
        if len(bucket["request_timestamps"]) >= bucket["max_rpm"]:
            return {"allowed": False, "reason": "rate_limit", "retry_after_seconds": 60 - (now - bucket["request_timestamps"][0])}
        if bucket["tokens"] < tokens_needed:
            deficit = tokens_needed - bucket["tokens"]
            wait = deficit / bucket["refill_rate"]
            return {"allowed": False, "reason": "token_limit", "tokens_available": bucket["tokens"], "retry_after_seconds": round(wait, 1)}
        return {"allowed": True, "tokens_available": bucket["tokens"]}

    def consume(self, user_id, tokens_used, tier="free"):
        bucket = self._get_bucket(user_id, tier)
        bucket["tokens"] -= tokens_used
        bucket["request_timestamps"].append(time.time())
        bucket["total_tokens_used"] += tokens_used

    def get_usage(self, user_id):
        if user_id not in self.buckets:
            return {"error": "User not found"}
        b = self.buckets[user_id]
        return {
            "user_id": user_id,
            "tier": b["tier"],
            "tokens_remaining": b["tokens"],
            "capacity": b["capacity"],
            "total_tokens_used": b["total_tokens_used"],
            "utilization": round(b["total_tokens_used"] / b["capacity"], 4) if b["capacity"] else 0,
        }
```

### 步骤5：成本追踪器

记录每次调用并计算运行总计。

```python
class CostTracker:
    def __init__(self, monthly_budget=1000.0):
        self.logs = []
        self.monthly_budget = monthly_budget
        self.alerts = []

    def log_call(self, model, input_tokens, output_tokens, cached_input_tokens=0, latency_ms=0, user_id="anonymous", cache_status="miss"):
        cost = calculate_cost(model, input_tokens, output_tokens, cached_input_tokens)
        entry = {
            "timestamp": time.time(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached_input_tokens,
            "latency_ms": latency_ms,
            "cost": cost["total_cost"],
            "user_id": user_id,
            "cache_status": cache_status,
        }
        self.logs.append(entry)
        self._check_budget()
        return entry

    def _check_budget(self):
        total = self.total_cost()
        pct = total / self.monthly_budget if self.monthly_budget > 0 else 0
        if pct >= 0.95 and not any(a["level"] == "stop" for a in self.alerts):
            self.alerts.append({"level": "stop", "message": f"Budget 95% consumed: ${total:.2f}/${self.monthly_budget:.2f}", "timestamp": time.time()})
        elif pct >= 0.85 and not any(a["level"] == "throttle" for a in self.alerts):
            self.alerts.append({"level": "throttle", "message": f"Budget 85% consumed: ${total:.2f}/${self.monthly_budget:.2f}", "timestamp": time.time()})
        elif pct >= 0.70 and not any(a["level"] == "warning" for a in self.alerts):
            self.alerts.append({"level": "warning", "message": f"Budget 70% consumed: ${total:.2f}/${self.monthly_budget:.2f}", "timestamp": time.time()})

    def total_cost(self):
        return round(sum(e["cost"] for e in self.logs), 6)

    def cost_by_model(self):
        by_model = {}
        for e in self.logs:
            m = e["model"]
            if m not in by_model:
                by_model[m] = {"calls": 0, "cost": 0, "input_tokens": 0, "output_tokens": 0}
            by_model[m]["calls"] += 1
            by_model[m]["cost"] = round(by_model[m]["cost"] + e["cost"], 6)
            by_model[m]["input_tokens"] += e["input_tokens"]
            by_model[m]["output_tokens"] += e["output_tokens"]
        return by_model

    def cache_savings(self):
        cache_hits = [e for e in self.logs if e["cache_status"] == "hit"]
        if not cache_hits:
            return {"saved": 0, "cache_hits": 0}
        saved = 0
        for e in cache_hits:
            full_cost = calculate_cost(e["model"], e["input_tokens"], e["output_tokens"])
            saved += full_cost["total_cost"]
        return {"saved": round(saved, 4), "cache_hits": len(cache_hits)}

    def summary(self):
        if not self.logs:
            return {"total_calls": 0, "total_cost": 0}
        total_latency = sum(e["latency_ms"] for e in self.logs)
        cache_hits = sum(1 for e in self.logs if e["cache_status"] == "hit")
        return {
            "total_calls": len(self.logs),
            "total_cost": self.total_cost(),
            "avg_cost_per_call": round(self.total_cost() / len(self.logs), 6),
            "avg_latency_ms": round(total_latency / len(self.logs), 1),
            "cache_hit_rate": round(cache_hits / len(self.logs), 4),
            "cost_by_model": self.cost_by_model(),
            "cache_savings": self.cache_savings(),
            "budget_remaining": round(self.monthly_budget - self.total_cost(), 2),
            "budget_utilization": round(self.total_cost() / self.monthly_budget, 4) if self.monthly_budget > 0 else 0,
            "alerts": self.alerts,
        }
```

### 步骤6：模型路由器

将查询路由到能处理它们的最便宜模型。

```python
SIMPLE_KEYWORDS = ["what time", "hours", "address", "phone", "price", "return policy", "hello", "hi", "thanks", "yes", "no"]
COMPLEX_KEYWORDS = ["analyze", "compare", "explain why", "write code", "debug", "architect", "design", "trade-off", "evaluate"]


def classify_complexity(query):
    q = query.lower()
    if len(q.split()) <= 5 or any(kw in q for kw in SIMPLE_KEYWORDS):
        return "simple"
    if any(kw in q for kw in COMPLEX_KEYWORDS):
        return "complex"
    return "medium"


def route_model(query, tier="pro"):
    complexity = classify_complexity(query)
    routing_table = {
        "simple": {"free": "gpt-4.1-nano", "pro": "gpt-4o-mini", "enterprise": "gpt-4o-mini"},
        "medium": {"free": "gpt-4o-mini", "pro": "claude-sonnet-4", "enterprise": "claude-sonnet-4"},
        "complex": {"free": "gpt-4o-mini", "pro": "gpt-4o", "enterprise": "claude-opus-4"},
    }
    model = routing_table[complexity].get(tier, "gpt-4o-mini")
    return {"query": query, "complexity": complexity, "model": model, "tier": tier}
```

### 第7步：运行演示

```python
def simulate_llm_call(model, query):
    input_tokens = len(query.split()) * 4 + 500
    output_tokens = 150 + (len(query.split()) * 2)
    latency = 200 + (output_tokens * 2)
    return {
        "model": model,
        "response": f"[Simulated {model} response to: {query[:50]}...]",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency,
    }


def run_demo():
    print("=" * 60)
    print("  Caching, Rate Limiting & Cost Optimization Demo")
    print("=" * 60)

    print("\n--- Model Pricing ---")
    for model, pricing in list(MODEL_PRICING.items())[:6]:
        cost_1k = calculate_cost(model, 1000, 500)
        print(f"  {model}: ${cost_1k['total_cost']:.6f} per 1K in + 500 out")

    print("\n--- Cost Comparison: 100K Requests ---")
    for model in ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4", "claude-haiku-3.5"]:
        cost = calculate_cost(model, 1000 * 100_000, 500 * 100_000)
        print(f"  {model}: ${cost['total_cost']:.2f}")

    print("\n--- Anthropic Cache Savings ---")
    no_cache = calculate_cost("claude-sonnet-4", 2000, 500, 0)
    with_cache = calculate_cost("claude-sonnet-4", 2000, 500, 1500)
    saving = no_cache["total_cost"] - with_cache["total_cost"]
    print(f"  Without cache: ${no_cache['total_cost']:.6f}")
    print(f"  With 1500 cached tokens: ${with_cache['total_cost']:.6f}")
    print(f"  Savings per call: ${saving:.6f} ({saving/no_cache['total_cost']*100:.1f}%)")

    exact_cache = ExactCache(max_size=100, ttl_seconds=300)
    semantic_cache = SemanticCache(similarity_threshold=0.75, max_size=100)
    rate_limiter = TokenBucketRateLimiter()
    tracker = CostTracker(monthly_budget=100.0)

    print("\n--- Exact Cache ---")
    messages_1 = [{"role": "user", "content": "What is the return policy?"}]
    result = exact_cache.get("gpt-4o-mini", messages_1, 0.0)
    print(f"  First lookup: {'HIT' if result else 'MISS'}")
    exact_cache.put("gpt-4o-mini", messages_1, 0.0, "You can return items within 30 days.")
    result = exact_cache.get("gpt-4o-mini", messages_1, 0.0)
    print(f"  Second lookup: {'HIT' if result else 'MISS'} -> {result}")
    result = exact_cache.get("gpt-4o-mini", messages_1, 0.7)
    print(f"  With temp=0.7: {'HIT' if result else 'MISS (non-deterministic, skip cache)'}")
    print(f"  Stats: {exact_cache.stats()}")

    print("\n--- Semantic Cache ---")
    test_queries = [
        ("What is the return policy?", "Items can be returned within 30 days with receipt."),
        ("How do I return an item?", None),
        ("What are your store hours?", "We are open 9am-9pm Monday through Saturday."),
        ("When does the store open?", None),
        ("Tell me about quantum computing", "Quantum computers use qubits..."),
        ("Explain quantum mechanics", None),
    ]
    for query, response in test_queries:
        cached = semantic_cache.get(query)
        if cached:
            print(f"  '{query[:40]}' -> CACHE HIT (sim={cached['similarity']}, original='{cached['original_query'][:40]}')")
        elif response:
            semantic_cache.put(query, response)
            print(f"  '{query[:40]}' -> MISS (stored)")
        else:
            print(f"  '{query[:40]}' -> MISS (no match)")
    print(f"  Stats: {semantic_cache.stats()}")

    print("\n--- Rate Limiting ---")
    for i in range(12):
        check = rate_limiter.check("user_1", 1000, "free")
        if check["allowed"]:
            rate_limiter.consume("user_1", 1000, "free")
        status = "OK" if check["allowed"] else f"BLOCKED ({check['reason']})"
        if i < 5 or not check["allowed"]:
            print(f"  Request {i+1}: {status}")
    print(f"  Usage: {rate_limiter.get_usage('user_1')}")

    print("\n--- Model Routing ---")
    routing_queries = [
        "What time do you close?",
        "Summarize this quarterly earnings report",
        "Analyze the trade-offs between microservices and monoliths",
        "Hello",
        "Write code for a binary search tree with deletion",
    ]
    for q in routing_queries:
        route = route_model(q, "pro")
        print(f"  '{q[:50]}' -> {route['model']} ({route['complexity']})")

    print("\n--- Full Pipeline: Before vs After Optimization ---")
    queries = [
        "What is the return policy?",
        "How do I return something?",
        "What are your hours?",
        "When do you open?",
        "Explain the difference between TCP and UDP",
        "Compare TCP vs UDP protocols",
        "Hello",
        "What is your phone number?",
        "Write a Python function to sort a list",
        "Analyze the pros and cons of serverless architecture",
    ]

    print("\n  [Before: no caching, single model (gpt-4o)]")
    tracker_before = CostTracker(monthly_budget=1000.0)
    for q in queries:
        result = simulate_llm_call("gpt-4o", q)
        tracker_before.log_call("gpt-4o", result["input_tokens"], result["output_tokens"], latency_ms=result["latency_ms"], cache_status="miss")
    before = tracker_before.summary()
    print(f"  Total cost: ${before['total_cost']:.6f}")
    print(f"  Avg cost/call: ${before['avg_cost_per_call']:.6f}")
    print(f"  Avg latency: {before['avg_latency_ms']}ms")

    print("\n  [After: caching + routing + rate limiting]")
    exact_c = ExactCache()
    semantic_c = SemanticCache(similarity_threshold=0.75)
    tracker_after = CostTracker(monthly_budget=1000.0)

    for q in queries:
        messages = [{"role": "user", "content": q}]
        cached = exact_c.get("gpt-4o", messages, 0.0)
        if cached:
            tracker_after.log_call("gpt-4o-mini", 0, 0, latency_ms=5, cache_status="hit")
            continue
        sem_cached = semantic_c.get(q)
        if sem_cached:
            tracker_after.log_call("gpt-4o-mini", 0, 0, latency_ms=15, cache_status="hit")
            continue
        route = route_model(q)
        result = simulate_llm_call(route["model"], q)
        tracker_after.log_call(route["model"], result["input_tokens"], result["output_tokens"], latency_ms=result["latency_ms"], cache_status="miss")
        exact_c.put(route["model"], messages, 0.0, result["response"])
        semantic_c.put(q, result["response"])

    after = tracker_after.summary()
    print(f"  Total cost: ${after['total_cost']:.6f}")
    print(f"  Avg cost/call: ${after['avg_cost_per_call']:.6f}")
    print(f"  Avg latency: {after['avg_latency_ms']}ms")
    print(f"  Cache hit rate: {after['cache_hit_rate']:.0%}")

    if before["total_cost"] > 0:
        savings_pct = (1 - after["total_cost"] / before["total_cost"]) * 100
        print(f"\n  SAVINGS: {savings_pct:.1f}% cost reduction")
        print(f"  Latency improvement: {(1 - after['avg_latency_ms'] / before['avg_latency_ms']) * 100:.1f}% faster")

    print("\n--- Budget Alerts Demo ---")
    alert_tracker = CostTracker(monthly_budget=0.01)
    for i in range(5):
        alert_tracker.log_call("gpt-4o", 5000, 2000, latency_ms=500)
    print(f"  Total spent: ${alert_tracker.total_cost():.6f} / ${alert_tracker.monthly_budget}")
    for alert in alert_tracker.alerts:
        print(f"  ALERT [{alert['level'].upper()}]: {alert['message']}")

    print("\n--- Cost Breakdown by Model ---")
    multi_tracker = CostTracker(monthly_budget=500.0)
    for _ in range(50):
        multi_tracker.log_call("gpt-4o-mini", 800, 200, latency_ms=150)
    for _ in range(30):
        multi_tracker.log_call("claude-sonnet-4", 1500, 500, latency_ms=400)
    for _ in range(10):
        multi_tracker.log_call("gpt-4o", 2000, 800, latency_ms=600)
    for _ in range(10):
        multi_tracker.log_call("claude-opus-4", 3000, 1000, latency_ms=1200)
    breakdown = multi_tracker.cost_by_model()
    for model, data in sorted(breakdown.items(), key=lambda x: x[1]["cost"], reverse=True):
        print(f"  {model}: {data['calls']} calls, ${data['cost']:.6f}, {data['input_tokens']:,} in / {data['output_tokens']:,} out")
    print(f"  Total: ${multi_tracker.total_cost():.6f}")

    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
```

## 使用它

### Anthropic 提示缓存 (Prompt Caching)

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-sonnet-4-20250514",
#     max_tokens=1024,
#     system=[
#         {
#             "type": "text",
#             "text": "You are a helpful customer support agent for Acme Corp...",
#             "cache_control": {"type": "ephemeral"},
#         }
#     ],
#     messages=[{"role": "user", "content": "What is the return policy?"}],
# )
#
# print(f"Input tokens: {response.usage.input_tokens}")
# print(f"Cache creation tokens: {response.usage.cache_creation_input_tokens}")
# print(f"Cache read tokens: {response.usage.cache_read_input_tokens}")
```

首次调用写入缓存（加价25%）。后续每次使用相同系统提示前缀的调用从缓存读取（折扣90%）。缓存持续5分钟，每次命中时重置计时器。

### OpenAI 自动缓存 (Automatic Caching)

```python
# from openai import OpenAI
#
# client = OpenAI()
#
# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "system", "content": "You are a helpful customer support agent..."},
#         {"role": "user", "content": "What is the return policy?"},
#     ],
# )
#
# print(f"Prompt tokens: {response.usage.prompt_tokens}")
# print(f"Cached tokens: {response.usage.prompt_tokens_details.cached_tokens}")
# print(f"Completion tokens: {response.usage.completion_tokens}")
```

OpenAI 自动缓存。任何与最近请求匹配的1024个以上令牌的提示前缀可获得50%折扣。无需更改代码——只需检查响应中的`prompt_tokens_details.cached_tokens`即可验证是否生效。

### OpenAI 批量 API (Batch API)

```python
# import json
# from openai import OpenAI
#
# client = OpenAI()
#
# requests = []
# for i, query in enumerate(queries):
#     requests.append({
#         "custom_id": f"request-{i}",
#         "method": "POST",
#         "url": "/v1/chat/completions",
#         "body": {
#             "model": "gpt-4o-mini",
#             "messages": [{"role": "user", "content": query}],
#         },
#     })
#
# with open("batch_input.jsonl", "w") as f:
#     for r in requests:
#         f.write(json.dumps(r) + "\n")
#
# batch_file = client.files.create(file=open("batch_input.jsonl", "rb"), purpose="batch")
# batch = client.batches.create(input_file_id=batch_file.id, endpoint="/v1/chat/completions", completion_window="24h")
# print(f"Batch ID: {batch.id}, Status: {batch.status}")
```

批量 API 对所有令牌提供统一50%折扣。结果在24小时内返回。非常适合非实时工作负载：评估、数据标注、批量摘要。

### 生产环境语义缓存（Redis）

```python
# import redis
# import numpy as np
# from openai import OpenAI
#
# r = redis.Redis()
# client = OpenAI()
#
# def get_embedding(text):
#     response = client.embeddings.create(model="text-embedding-3-small", input=text)
#     return response.data[0].embedding
#
# def semantic_cache_lookup(query, threshold=0.95):
#     query_emb = np.array(get_embedding(query))
#     keys = r.keys("cache:emb:*")
#     best_sim, best_key = 0, None
#     for key in keys:
#         stored_emb = np.frombuffer(r.get(key), dtype=np.float32)
#         sim = np.dot(query_emb, stored_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(stored_emb))
#         if sim > best_sim:
#             best_sim, best_key = sim, key
#     if best_sim >= threshold and best_key:
#         response_key = best_key.decode().replace("cache:emb:", "cache:resp:")
#         return r.get(response_key).decode()
#     return None
```

在生产环境中，用向量索引（Redis Vector Search、Pinecone 或 pgvector）替换线性扫描。线性扫描适用于少于1000条条目。超过此数量，使用近似最近邻 (ANN) 实现O(log n)查找。

## 发布

本课产出`outputs/prompt-cost-optimizer.md` ——一个可复用的提示词，用于分析你的LLM应用并推荐具体的成本优化方案及预计节省金额。

同时还产出`outputs/skill-cost-patterns.md` ——一个决策框架，用于为你的用例选择合适的缓存策略、速率限制配置和模型路由规则。

## 练习

1. **为语义缓存实现最近最少使用(LRU)驱逐策略。** 将最早优先驱逐替换为最近最少使用。记录每条条目的最后访问时间，当缓存满时驱逐访问时间最早的条目。比较两种策略在100次查询中的命中率。

2. **构建成本预测工具。** 给定API调用日志（CostTracker日志），基于过去7天的滚动平均预测月度成本。考虑工作日/周末模式。如果预测月度成本超出预算20%以上，触发警报。

3. **实现分层语义缓存。** 使用两个相似度阈值：0.98用于高置信度命中（立即返回），0.90用于中等置信度命中（附带免责声明返回：“基于之前类似的问题...”）。记录每次命中来自哪个层级，并测量用户满意度差异。

4. **构建模型路由分类器。** 将基于关键字的分类器替换为基于嵌入的分类器。嵌入50个标记查询（简单/中等/复杂），然后通过查找最近的标记示例对新查询进行分类。使用20个查询的测试集衡量分类准确率。

5. **实现具有降级级别的断路器。** 预算消耗70%时，记录警告。85%时，自动将所有路由切换到最便宜模型（gpt-4o-mini）。95%时，仅提供缓存的响应并拒绝新查询。通过模拟1.00美元预算下的1000次请求进行测试，验证每个阈值是否正确触发。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|----------------------|
|  提示缓存 (Prompt caching)  |  "缓存系统提示"  |  提供方级别的缓存，重复的提示前缀可享受折扣（Anthropic 90%，OpenAI 50%）——OpenAI无需代码更改，Anthropic需显式标记  |
|  语义缓存 (Semantic caching)  |  "智能缓存"  |  嵌入查询，计算与历史查询的相似度，如果相似度超过阈值则返回缓存的响应——捕获精确匹配遗漏的改写  |
|  精确缓存 (Exact caching)  |  "哈希缓存"  |  对完整提示（模型+消息+温度）进行哈希，对完全相同的输入返回缓存的响应——仅适用于temperature=0的确定性调用  |
|  令牌桶 (Token bucket)  |  "速率限制器"  |  一种算法，每个用户拥有一个容量为N的令牌桶，以每秒R个令牌的速率补充——允许最多N的突发流量，同时确保平均速率为R  |
|  模型路由 (Model routing)  |  "吝啬路由"  |  使用分类器将简单查询发送到廉价模型（GPT-4o-mini、Haiku），复杂查询发送到昂贵模型（GPT-4o、Opus）——节省40-70%的模型成本  |
|  成本跟踪 (Cost tracking)  |  "计量"  |  记录每次API调用的模型、令牌、延迟、成本和用户ID，以便准确了解资金去向及哪些功能成本高昂  |
|  断路器 (Circuit breaker)  |  "紧急开关"  |  当支出接近预算限制时，自动降级服务（使用更便宜的模型、仅缓存）或完全停止请求  |
|  批量 API (Batch API)  |  "批量折扣"  |  OpenAI的异步处理，享受50%折扣——最多提交50,000个请求，24小时内获得结果  |
|  提示压缩 (Prompt compression)  |  "令牌减量"  |  重写系统提示和上下文，使用更少的令牌同时保留含义——更短的提示成本更低，通常表现更好  |
|  缓存命中率 (Cache hit rate)  |  "缓存效率"  |  从缓存而不是调用LLM处理的请求百分比——生产环境聊天机器人通常为40-60%，按比例节省成本  |

## 延伸阅读

- [Anthropic Prompt Caching Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) —— Anthropic 显式 cache_control 标记、定价和缓存生命周期行为的官方文档
- [Anthropic Prompt Caching Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) —— OpenAI 自动缓存、如何通过 usage 字段验证缓存命中以及最小前缀长度
- [Anthropic Prompt Caching Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) —— 异步处理享受50%折扣、JSONL 格式、24小时完成窗口和50K请求限制
- [Anthropic Prompt Caching Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) —— 开源语义缓存库，支持多种嵌入后端、向量存储和驱逐策略
- [Anthropic Prompt Caching Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) —— 生产环境模型路由，自动选择能够处理每个查询的最便宜模型
- [Anthropic Prompt Caching Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) —— 基于机器学习的模型路由器，从您的流量模式中学习，以优化跨提供商的成本/质量权衡
- [Anthropic Prompt Caching Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) —— LLM 可观测性平台，提供成本跟踪、缓存、速率限制和预算警报，作为代理层
- [Anthropic Prompt Caching Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) —— 延迟、吞吐量、TTFT/TPOT 百分位数和对冲请求；“选择仍满足 P95 的最便宜模型”背后的成本模型
- [Anthropic Prompt Caching Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) —— vLLM 论文；为什么分页 KV 缓存 + 连续批处理在吞吐量上比朴素服务器高出24倍，“缓存与成本”下的基础设施层
- [Anthropic Prompt Caching Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) —— 与提示缓存正交的内核级成本降低；与推测解码和 GQA 一起阅读以获得完整的成本曲线图。
