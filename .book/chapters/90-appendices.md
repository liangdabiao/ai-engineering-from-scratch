## 附录 A 核心概念速查表

> 复用课程 `glossary/terms.md` 的"人话 vs 真相"表。挑了本书高频词。

| 概念 | 人话 | 真相 |
|------|------|------|
| Agent | 自主思考行动的 AI | 一个 while 循环：LLM 决定调哪个工具，执行，看结果，重复 |
| LLM | AI / 大脑 | 预测下一个 token 的 transformer，数十亿参数 |
| Context Window | AI 能记多少 | 单次 API 调用能装的最大 token 数，不是记忆，每调用重置 |
| Token | 一个词 | 子词单元，英文约 3–4 字符；"unbelievable" 可能是 3 个 token |
| Transformer | 现代 AI 架构 | 用自注意力处理序列，不靠循环，可大规模并行 |
| Attention | AI 专注重点 | 每个 token 对所有 token 算加权求和，权重由 query·key 决定 |
| RAG | 会搜索的 AI | 检索相关文档（embedding 相似度），塞进提示，让 LLM 据此答 |
| Function Calling | AI 能用工具 | 用 JSON Schema 定义工具，模型输出结构化调用，你执行，结果回传 |
| MCP | AI 连工具的标准 | 开放协议（JSON-RPC），统一 AI 应用接外部数据源和工具 |
| Prompt Injection | 用话黑 AI | 输入里的恶意文本覆盖系统提示；间接注入藏在检索内容里，无完美解 |
| Hallucination | AI 在撒谎 | 生成听着合理但不基于训练或上下文的内容；是补全，不是检索 |
| Embedding | 把词变数字 | 把离散项映射到向量空间，相似的近，用于语义搜索 |
| Cosine Similarity | 两向量多像 | 夹角余弦：dot(a,b)/(‖a‖·‖b‖)，-1 到 1，只看方向不看大小 |
| Guardrails | AI 安全滤网 | 输入输出校验层，拦有害内容、注入、PII、跑题 |
| Streaming | 逐字蹦答案 | 边生成边发 token，用 SSE/WebSocket，首 token 延迟低 |
| Temperature | 创意档 | 除 logits 再 softmax；高=随机，0=最确定（贪心） |
| System Prompt | AI 的指令 | 对话开头的特殊消息，定行为、人设、约束，开发者设 |
| Quantization | 把模型变小 | 权重从 float32 降到 int8/int4，精度微损，内存省 4–8 倍 |
| LoRA | 高效微调 | 插小低秩矩阵只训它们，内存省 10–100 倍 |
| Fine-tuning | 用你的数据训 | 从预训练权重接着训小数据集，只更新已有权重 |

## 附录 B 常见错误与解法

> 每章"注意"框的汇总。踩过的坑，别再踩。

| 章节 | 错误 | 解法 |
|------|------|------|
| §02 | 无限循环不收敛 | 设轮次预算 + 停止条件 |
| §05 | reasoning 泄露到用户面 | 内部思考走字段，只显动作/最终答 |
| §06 | 把存档当长期记忆全塞 | 分层：工作/情景/语义，按时清 |
| §08 | 单模型自省陷入群体思维 | 换多实例交叉批驳（§17） |
| §09 | 长上下文中段信息丢失 | 重要放头尾，压缩中段，RAG 即上下文工程 |
| §10 | 90% 陷阱：随机动作 | 写 trace 检测器，定位首错步 |
| §13 | 过度用 handoff 互踢 | 加跳数计数器，超 3 跳拒绝 |
| §14 | 把 CUA 当完全自主 | 不可信输入契约 + 逐步安全评估 |
| §16 | 只靠系统提示防注入 | 护栏 + PVE 验证器前置 |
| §17 | 拓扑优先，不问为何 | 先单后多，Anthropic 决策顺序 |
| §18 | 不打断（barge-in） | UPSTREAM 取消帧停 TTS |
| §19 | 选请求-响应跑 5 分钟任务 | 超 30 秒用队列/持久执行 |
| §19 | 不装成本追踪 | 先接 CostTracker，再谈省 |
| §20 | Agent 自己判完工 | 确定性验收闸门 + 签名 override |

## 附录 C 阅读指南（按天分组）

> 标叔建议你这么读。每天 1–2 小时，六天走完。

**Day 1 · 跑通**：§01 → §05。亲手写循环、工具、第一个真 Agent。目标：能在终端跑起来。

**Day 2–3 · 核心**：§06 记忆 → §10 失败模式。搞懂 Agent 的五脏六腑，为什么崩。

**Day 4–5 · 框架**：§11 横评 → §16 安全。挑一个框架实战，接可观测，补安全。

**Day 6 · 进阶**：§17 多智能体 → §20 Capstone。按场景上多体、语音、成本，装出工作台。

**随时翻**：附录 A 查词，附录 B 避坑，附录 D 追源。

## 附录 D 参考文献

> 书中引用的论文与来源。课程内论文为主，附关键官方文档。

**论文（arXiv）**

- ReAct: Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2022, arXiv:2210.03629)
- MemGPT: Packer et al., *MemGPT: Towards LLMs as Operating Systems* (arXiv:2310.08560)
- Mem0: Mem0 Team, *Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory* (arXiv:2504.19413)
- ReWOO: Xu et al., *ReWOO: Decoupling Reasoning from Observations* (arXiv:2305.18323)
- ToT: Yao et al., *Tree of Thoughts* (arXiv:2305.10601)
- LATS: Zhou et al., *Language Agent Tree Search* (arXiv:2310.04406)
- Reflexion: Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement* (arXiv:2303.11366)
- Self-Refine: Madaan et al., *Self-Refine* (arXiv:2303.17651)
- CRITIC: Gou et al., *CRITIC: Large Language Models Can Self-Correct* (arXiv:2305.11738)
- Lost in the Middle: Liu et al. (arXiv:2307.03172)
- MASFT: "Mitigating Agentic System Failure Taxonomy" (Berkeley, arXiv:2503.13657)
- Society of Minds: Du et al. (ICML 2024, arXiv:2305.14325)
- Sparse Topology: arXiv:2406.11776
- Indirect Prompt Injection: Greshake et al. (AISec 2023, arXiv:2302.12173)
- AlphaEvolve: arXiv:2506.13131

**官方文档与来源**

- Anthropic, *Building Effective Agents*（五模式 + agent vs workflow）
- Anthropic, *Prompt Caching* 指南（cache_control，90% 折扣）
- OpenAI, *Agents SDK*（Handoffs / Guardrails / Sessions / Tracing）
- OpenAI, *Prompt Caching*（自动前缀匹配，50% 折扣）
- LangGraph 文档（supervisor / swarm / hierarchical）
- CrewAI 文档（Crew vs Flow）
- Pipecat 文档（帧管线、processor、transport）
- LiveKit Agents 文档（WebRTC + 语音原语）
- Cloudflare, *Orchestrating AI Code Review at Scale*（13 万次/30 天）
- agents.md — 开放规范（Cursor / Codex / Claude Code / Copilot 共用）
- 课程仓库：ai-engineering-from-scratch（Phase 11 · 14 · 13）
