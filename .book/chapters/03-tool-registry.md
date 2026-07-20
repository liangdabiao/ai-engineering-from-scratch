## §03 工具注册表与停止条件：把 LLM 接上真实世界

### 03.1 你需要什么

- Python 3.11+，纯 stdlib，无需联网。
- 预计 30 分钟。

### 03.2 我们要做成什么

给 Agent 接两个工具：计算器、键值库。让它真能存、真能算。

### 03.3 写工具注册表

**第一步**：定义工具。

```python
def calculator(expr: str) -> str:
    try:
        return str(eval(expr))        # 教学用，生产勿直接 eval
    except Exception as e:
        return f"错误: {e}"           # 错误也变 observation
```

预期结果：输入 `"12*8"` 返回 `"96"`。

**第二步**：注册。

```python
reg = ToolRegistry()
reg.register("calculator", calculator)
reg.register("kv_set", lambda k, v: store.set(k, v))
```

预期结果：两个工具进表，模型可按名调用。

> **标叔的经验**：错误也是 observation
>
> 我早期让异常直接抛。Agent 一碰坏输入就崩。后来所有错误都包成字符串喂回去。稳了。

### 03.4 停止条件怎么定

关键看三件事：

- 显式 `finish`：模型说完了。
- 无工具调用：模型这轮只说话，默认结束。
- 超预算：max_turns 触发，强停。

> **核心建议**：别只靠 finish
>
> 模型会忘写 finish。加"无工具调用即停"兜底。少这层，常早退。

### 03.5 回顾

我们接了两个工具，定了三道停止线。Agent 第一次够到了外部状态。

[向前桥接] 零件齐了。下一章，拼一个能查资料、能算数的真 Agent。
