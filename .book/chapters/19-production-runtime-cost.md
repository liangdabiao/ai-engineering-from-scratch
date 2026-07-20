## §19 生产运行时：选错形状，钱和稳定性一起崩

2025 年一个定时 Agent 在机器重启后丢了半夜进度。我才认真看运行时形状。

### 19.1 一句判断：先选运行时，再选框架

多数 AI 创业公司不是死在模型差。

是死在单位经济差。

一次 GPT 调用几分钱。一万用户每天十次，光输入 token 就 250 美元一天。

我见过的账单：一个 RAG 客服，裸模型 2.25 万美元/月。

加了缓存和路由，降到 5000 多。省下的就是 runway。

> **标叔的经验**：那张 2.2 万刀的账单
>
> 2025 年帮一个团队看成本。系统提示每次都重发 1500 token。10 万请求/天，光这段不变文字就 375 刀/天。加一行 cache_control，当天省 337 刀。

### 19.2 六种运行时形状

生产 Agent 跑在六种外壳上。形状决定哪些故障可活。

- **请求-响应**：同步 HTTP，用户等。只适合 <30 秒的短任务。
- **流式**：SSE/WebSocket 渐进输出。语音经 WebRTC（§18）。
- **持久执行**：每步存盘，失败自动续。长任务必备。
- **队列/后台**：任务进队列，worker 取，结果回传。长程 Agent 靠它。
- **事件驱动**：订阅触发（新邮件、PR 开、cron 响）。Claude 托管 Agent 自带。
- **定时**：cron 周期跑。配合持久执行，夜里挂了下 tick 续。

框架落点：Agno 走 stateless FastAPI；Mastra 走 server adapter；CrewAI Flow 管事件驱动；Pipecat/LiveKit 管语音；Claude Managed Agents 管长程异步。

### 19.3 可观测性是承重墙

没有 OpenTelemetry 的 GenAI span，你调不了第 40 步挂掉的 Agent。

Langfuse / Phoenix / Opik 不是锦上添花。是地基。

没有它，你只有两条路：要么快速复现，要么从头重跑加日志。

| 形状 | 代表栈 | 致命坑 | 标叔的结论 |
|------|--------|--------|-----------|
| 请求-响应 | Agno + FastAPI | 5 分钟任务用户挂 | 超 30 秒别用 |
| 流式 | SSE/WS | 背压没做 | 客户端慢要暂停 |
| 持久 | LangGraph | 不做就重跑 | 长任务必须做 |
| 队列 | Celery/BullMQ | 无 DLQ 任务消失 | 死信队列必建 |
| 事件 | CrewAI Flow | 触发源不清 | 埋 trace |
| 定时 | K8s CronJob | 重启即丢 | 配持久执行 |

### 19.4 Provider 缓存：白送的折扣

三家 2026 都给 prompt 缓存，机制不同：

| Provider | 机制 | 折扣 | 最低长度 |
|----------|------|------|---------|
| Anthropic | 显式 cache_control | 90% 命中 | 1024 token |
| OpenAI | 自动前缀匹配 | 50% 命中 | 1024 token |
| Gemini | 显式 CachedContent | ~75% | 4096 (Flash) |

Anthropic 写法：给不变前缀打 `cache_control: {"type": "ephemeral"}`。

首次写贵 25%，之后命中省 90%。系统提示从此几乎免费。

```python
# Anthropic 提示缓存：打标记即可
client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=[{
        "type": "text",
        "text": "你是 Acme 客服...",        # 不变的长提示
        "cache_control": {"type": "ephemeral"}  # 命中省 90%
    }],
    messages=[{"role": "user", "content": "退货政策？"}],
)
```

### 19.5 应用层缓存：精确与语义

Provider 缓存只认相同前缀。语义缓存管"同义不同字"。

"退货政策？" 和 "怎么退货？" 不是同一串，意思一样。

```python
# 精确缓存：相同输入直接命中
class ExactCache:
    def _hash(self, model, messages, temperature):
        key = json.dumps({"model": model, "messages": messages,
                          "temperature": temperature}, sort_keys=True)
        return hashlib.sha256(key.encode()).hexdigest()  # 哈希全提示

    def get(self, model, messages, temperature=0.0):
        if temperature > 0:
            return None                     # 非确定，跳过
        return self.cache.get(self._hash(model, messages, temperature))
```

```python
# 语义缓存：embedding 后比余弦相似度
def semantic_cache_get(query, entries, threshold=0.92):
    q = embed(query)
    best, sim = None, 0.0
    for e in entries:
        s = cosine(q, e["embedding"])      # 0.92+ 视为同义
        if s > sim:
            best, sim = e, s
    return best["response"] if best and sim >= threshold else None
```

### 19.6 限速与熔断：活下去的开关

限速不是讲公平，是讲生存。

令牌桶：每用户一桶 N 个 token，按 R/秒 补。空了就拒。允许突发。

熔断：到预算就降级。三档：

- 70%：告警。
- 85%：只走便宜模型。
- 95%：拒新请求，只回缓存。

```python
# 令牌桶限速器（每用户）
class TokenBucket:
    def __init__(self, cap=50_000, refill=500):
        self.cap, self.refill = cap, refill
        self.tokens = cap
        self.last = time.time()

    def check(self, need):
        now = time.time()
        self.tokens = min(self.cap, self.tokens + (now - self.last) * self.refill)
        self.last = now
        return self.tokens >= need          # 不够就拒
```

### 19.7 模型路由：对的事给对的价

不是每句都要贵模型。

"几点关门？" 用 GPT-4o-mini 足够。复杂推理才上 Opus。

一个简单的分类器，就能省 40–70% 模型费。

```python
# 复杂度分类 → 路由模型
SIMPLE = ["几点", "价格", "退货", "你好"]
COMPLEX = ["分析", "对比", "写代码", "架构"]

def route(query, tier="pro"):
    q = query.lower()
    if len(q.split()) <= 5 or any(k in q for k in SIMPLE):
        return "gpt-4o-mini"                # 便宜模型
    if any(k in q for k in COMPLEX):
        return "claude-opus-4"              # 贵模型
    return "claude-sonnet-4"
```

### 19.8 优化栈：一层叠一层

| 层 | 技术 | 典型省 | 难度 |
|----|------|--------|------|
| 1 | Provider 缓存 | 30–50% | 低 |
| 2 | 精确缓存 | 10–20% | 低 |
| 3 | 语义缓存 | 15–30% | 中 |
| 4 | 模型路由 | 40–70% | 中 |
| 5 | 限速熔断 | 保预算 | 低 |
| 6 | 批量 API | 50% | 低 |

RAG 应用叠 1–5 层，通常从 2.25 万/月 降到 4000–6000/月。

这就是烧 runway 和做业务的差别。

> **核心建议**：先接成本追踪
>
> 每笔调用记 model、token、延迟、花费、用户、命中。不量就无从优化。先装 CostTracker，再谈省。

### 19.9 先给结论

运行时形状定生死。缓存和路由定成本。

两者都做，才叫生产级。

[向前桥接] 最后一步。把前面所有能力，装进一个能跑真仓库的工作台。
