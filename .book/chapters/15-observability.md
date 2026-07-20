## §15 可观测性：看不见的 Agent 最危险

一次线上 Agent 在第 40 步挂了。我翻三小时聊天记录才定位。从此装上 trace。

### 15.1 三个开源平台

| 平台 | 协议 | 最强 | 标叔的结论 |
|------|------|------|-----------|
| Langfuse | MIT | 全链路 + prompt 版本 | 要一体化选它 |
| Phoenix | Elastic 2.0 | RAG 相关性、漂移 | 深研 RAG 选它 |
| Opik | Apache 2.0 | 自动优化、护栏 | 要实验闭环选它 |

数据：2026 年 89% 的组织已上 agent 可观测。质量问题是头号生产拦路虎（32% 提及）。

> **标叔的经验**：只追 trace 不评，是贵日志
>
> 我早期就囤 span。后来加 LLM-judge，才真发现问题。

### 15.2 没有评估，trace 是摆设

- 评估策略要先定。
- LLM-judge 也要接地（CRITIC）：评委需外部工具验事实。
- prompt 版本要绑 trace。回归了才能 bisect 到那版 prompt。

### 15.3 OTel 是底座

三家都吃 OpenTelemetry GenAI 语义约定。span 跨厂商可导出到 Datadog、New Relic。

手写极简评估管线：

```python
def judge(trace):
    score = rubric_eval(trace)    # LLM-judge 按评分标准
    return tag_failures(score)    # 标失败原因
```

[向前桥接] 能看见了。但看见之后，还要防被黑。下一章，提示注入。
