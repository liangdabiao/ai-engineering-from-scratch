## §04 第一个真实 Agent：stdlib 写个能查能算的助手

### 04.1 我们要做成什么

一个命令行助手：你问"巴黎人口多少，乘 2 是多少"，它查、算、答。

### 04.2 接一个"查"工具

```python
import urllib.request, json

def wiki_search(q: str) -> str:
    # 教学：真实环境请用官方 API + 限流
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + q
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.load(r)["extract"][:300]
    except Exception as e:
        return f"查不到: {e}"
```

预期结果：返回摘要前 300 字，喂回模型当 observation。

### 04.3 拼装并跑

```python
reg.register("wiki", wiki_search)
reg.register("calc", calculator)
loop = AgentLoop(RealLLM(), reg, max_turns=8)
print(loop.run("查巴黎人口，再乘 2"))
```

预期结果：模型先 wiki，再 calc，最后给数字。

> **标叔的经验**：先小后大
>
> 我第一版直接塞 10 个工具。模型乱调。降到 3 个，反而全对。工具多≠强。

### 04.4 回顾

纯 stdlib，零框架。一个能查能算的 Agent 站起来了。

[向前桥接] 但这是玩具模型。下一章，换真模型，讲原生推理。
