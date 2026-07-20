## §05 从 Toy 到真实模型：原生推理来了

### 05.1 现象：Thought 正在消失

2026 年，你很少再看到 `Thought:` 打印在终端。

不是循环变了。是"想"搬了家。

### 05.2 核心维度：旧 vs 新

| 维度 | 2022 提示式 | 2026 原生推理 | 标叔的结论 |
|------|-----------|-------------|-----------|
| 想的载体 | 打印 Thought 字符串 | 独立 reasoning 通道 | 通道更干净 |
| 跨轮传递 | 拼回 prompt | 加密随会话走 | 不易被注入篡改 |
| 可观测 | 全透明 | 部分隐藏 | 调试靠 trace |
| 框架支持 | 全手写 | SDK 内置 | 直接用车 |

### 05.3 换模型的代价

把 `ToyLLM` 换成真调用：

```python
class RealLLM:
    def __init__(self, client):
        self.client = client
    def __call__(self, buf):
        # 调用 Responses API，reasoning 走独立字段
        return self.client.respond(messages=buf)
```

注意：reasoning 字段别写回用户可见 prompt。那会污染上下文。

> **注意**：reasoning 泄露是坑
>
> 某次我把推理串回 prompt，模型开始"读自己心思"循环。隔离后正常。

### 05.4 先给结论

循环不变。变的只是"想"存在哪。你懂循环，就懂所有 2026 框架。

[向前桥接] 起步结束。下面进 Part 2，拆 Agent 的五脏六腑：记忆、规划、反思、失败。
