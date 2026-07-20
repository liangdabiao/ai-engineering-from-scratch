## §02 ReAct 循环：Observe→Think→Act

### 02.1 三句话讲清 ReAct

2022 年，Yao 等人（arXiv:2210.03629）提出 Reason + Act。

每轮输出三段：Thought（想）、Action（动）、Observation（看）。

你看这段原始样例：

```text
Thought: 我要查法国首都。
Action: search("capital of France")
Observation: 巴黎是法国首都。
Thought: 答案是巴黎。
Action: finish("巴黎")
```

就这。一个循环包住这三段，Agent 就成了。

### 02.2 为什么三段缺一不可

原文给了硬数据。不是我吹：

- ALFWorld：只加 1–2 个示例，成功率 +34 点。
- WebShop：比模仿学习高 10 点。
- HotpotQA：靠检索 grounding，能从幻觉里爬出来。

> **标叔的经验**：先想后动，是为了能纠错
>
> 我做过对比。只让模型"动"不让"想"，遇到意外返回直接崩。加了 Thought，它能说"这结果不对，换条路"。

### 02.3 五个零件，少一个都不是 Agent

每个循环要五样东西：

1. **消息缓冲**：user→assistant→tool→assistant…一条 growing 的列表。
2. **工具注册表**：模型按名字调，schema 进、执行、结果字符串出。
3. **停止条件**：说 finish、或没工具调用、或超轮次、或护栏触发。
4. **轮次预算**：防死循环。2026 年 Agent 跑 40–400 步很正常。
5. **观测格式化**：把工具输出变成模型能读的字。每个 400 错误都得变成 observation，不能崩。

### 02.4 亲手写一个（<200 行，纯 stdlib）

```python
# agent_loop.py —— 标叔手写的最小 Agent 循环
class ToolRegistry:
    def __init__(self):
        self.tools = {}
    def register(self, name, fn):
        self.tools[name] = fn          # 名字 → 可调用
    def call(self, name, **kw):
        return self.tools[name](**kw)  # 执行，返回字符串

class AgentLoop:
    def __init__(self, llm, registry, max_turns=20):
        self.llm = llm
        self.registry = registry
        self.max_turns = max_turns     # 轮次预算，防死循环
    def run(self, user_msg):
        buf = [("user", user_msg)]
        for _ in range(self.max_turns):
            out = self.llm(buf)        # 模型想一步
            if out.action is None:     # 停止条件：没工具调用
                return out.text
            obs = self.registry.call(out.action, **out.args)
            buf.append(("tool", obs))  # 观测进缓冲，下轮再想
        return "超轮次，强制结束"
```

跑起来长这样：

```python
reg = ToolRegistry()
reg.register("calc", lambda expr: str(eval(expr)))  # 示例工具
loop = AgentLoop(ToyLLM(), reg)
print(loop.run("算 12*8 再加 3"))  # → 99
```

> **注意**：ToyLLM 只是教学替身
> 换真模型时，把 `ToyLLM` 换成 Responses API 调用即可。循环本身一行不用改。

### 02.5 2026 的转向：原生推理

`Thought:` 是 2022 的权宜之计。2025–2026 的 Responses API 改了：

模型把"想"放到独立通道。跨轮传递，生产环境加密。

变的只是想的载体。循环没变。Observe→Think→Act，纹丝不动。

[向前桥接] 循环写完了。但光有循环，模型还是够不着真实世界。下一章，接工具。
