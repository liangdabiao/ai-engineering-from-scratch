# 顶点项目 08 — 面向受监管垂直行业的生产级 RAG 聊天机器人

> Harvey、Glean、Mendable和LlamaCloud在2026年都运行相同的生产形态。使用docling或Unstructured进行摄取，结合ColPali处理视觉内容。混合搜索。使用bge-reranker-v2-gemma进行重排序。使用Claude Sonnet 4.7进行合成，并利用提示缓存( Prompt Caching )实现60-80%的命中率。使用Llama Guard 4和NeMo Guardrails进行防护。使用Langfuse和Phoenix进行监控。基于200题黄金测试集使用RAGAS进行评分。在受监管领域（法律、临床、保险）构建一个系统，最终成果是通过黄金测试集、红队测试和漂移仪表盘。

**类型：** 结业项目
**语言：** Python（流水线 + API），TypeScript（聊天UI）
**先决条件：** 阶段5（NLP）、阶段7（Transformer）、阶段11（LLM工程）、阶段12（多模态）、阶段17（基础设施）、阶段18（安全性）
**涉及阶段：** P5 · P7 · P11 · P12 · P17 · P18
**时长：** 30小时

## 问题

受监管领域的RAG（法律合同、临床试验方案、保险政策）是2026年最常部署的生产形态，因为其投资回报率显而易见且风险具体。Harvey（Allen & Overy）为其法律场景构建了该系统。Mendable提供了开发者文档版本。Glean覆盖了企业搜索。其模式是：高保真摄入、混合检索与重排序、带引用强制和提示缓存的合成、多层安全防护，以及持续漂移监控。

难点不在于模型本身。而在于：具有管辖意识合规性（HIPAA、GDPR、SOC2）、引用级别可审计性、成本控制（当缓存命中率高时，提示缓存可获得60-90%的折扣）、通过RAGAS忠实度进行幻觉检测，以及在源文档更新而索引未同步时进行漂移检测。本结业项目要求您在包含200题黄金测试集和一套红队测试的基础上实现所有这些功能。

## 概念

流水线分为两部分。**摄入**：docling或Unstructured解析结构化文档；ColPali处理视觉丰富的文档；分块生成摘要、标签和基于角色的访问标签。向量存入pgvector + pgvectorscale（5000万以下向量）或Qdrant Cloud；稀疏BM25并行运行。**对话**：LangGraph处理记忆和多轮对话；每次查询执行混合检索，使用bge-reranker-v2-gemma-2b重排序，使用Claude Sonnet 4.7（提示缓存）合成，输出经过Llama Guard 4和NeMo Guardrails过滤，并生成带引用的响应。

评估栈有四层。**黄金测试集**（200个带标签的问答对和引用）用于正确性评估。**红队测试**（越狱、PII提取尝试、领域外问题）用于安全性评估。**RAGAS**自动评估每轮的忠实度/答案相关性/上下文精确度。**漂移仪表盘**（Arize Phoenix）每周监控检索质量和幻觉分数。

提示缓存是成本杠杆。Claude 4.5+和GPT-5+支持缓存系统提示和检索到的上下文。在60-80%命中率下，每次查询成本降低3-5倍。流水线必须设计为稳定的前缀（系统提示+重排序上下文优先）以实现高缓存命中率。

## 架构

```
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## 技术栈

- 摄入：Unstructured.io或docling用于结构化文档；ColPali用于视觉丰富的PDF
- 向量数据库：5000万以下向量使用pgvector + pgvectorscale；否则使用Qdrant Cloud
- 稀疏检索：Tantivy BM25，带字段权重
- 编排：LlamaIndex Workflows（摄入）+ LangGraph（对话）
- 重排序器：自托管bge-reranker-v2-gemma-2b或托管Voyage rerank-2
- LLM：带提示缓存的Claude Sonnet 4.7；备用自托管Llama 3.3 70B
- 评估：RAGAS 0.2在线，DeepEval用于幻觉和越狱测试套件
- 可观测性：自托管Langfuse带标注队列；Arize Phoenix用于漂移
- 护栏：Llama Guard 4输入/输出分类器，NeMo Guardrails v0.12策略，Presidio PII清除
- 合规性：分块上的基于角色访问标签；用于GDPR/HIPAA的管辖标签

```figure
canary-rollout
```

## 动手构建

1. **摄入。** 使用Unstructured或docling解析您的语料库（严肃构建需要1000-10000个文档）。对于扫描/视觉密集页面，通过ColPali路由。生成带有摘要、角色标签和管辖标签的分块。

2. **索引。** 将稠密嵌入（Voyage-3或Nomic-embed-v2）存入pgvector + pgvectorscale。通过Tantivy构建BM25侧索引。角色和管辖过滤器作为负载信息。

3. **混合检索。** 首先按角色+管辖过滤；然后并行稠密+BM25；使用倒数排序融合( Reciprocal Rank Fusion )合并；前20个送入重排序器；前5个送入合成。

4. **使用提示缓存合成。** 系统提示和静态策略放入缓存头；重排序上下文作为缓存扩展；用户问题作为非缓存后缀。目标稳态达到60-80%缓存命中率。

5. **护栏。** 输入上的Llama Guard 4；NeMo Guardrails规则阻止领域外问题或策略禁止的主题；Presidio清除输出中意外的PII；引用强制后过滤。

6. **黄金测试集。** 由领域专家标注的200个问答对（答案、引用）。基于精确引用匹配、答案正确性、忠实度（RAGAS）对智能体评分。

7. **红队测试。** 50个对抗性提示：越狱（PAIR、TAP）、PII窃取尝试、领域外、跨管辖泄露。使用通过/失败和严重程度评分。

8. **漂移仪表盘。** Arize Phoenix每周跟踪检索质量（nDCG、引用忠实度）。在下降5%时发出警报。

9. **成本报告。** Langfuse：提示缓存命中率、每次查询令牌数、各阶段$/查询分解。

## 使用它

```
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## 发布

`outputs/skill-production-rag.md` 描述了交付物。一个部署了合规标签、通过了评分标准、并受到实时漂移监控的受监管领域聊天机器人。

|  权重  |  标准  |  衡量方式  |
|:-:|---|---|
|  25  |  RAGAS忠实度 + 答案相关性  |  黄金测试集（200问答）上的在线分数  |
|  20  |  引用正确性  |  带有可验证源锚点的答案比例  |
|  20  |  护栏覆盖率  |  Llama Guard 4通过率 + 越狱测试套件结果  |
|  20  |  成本/延迟工程  |  提示缓存命中率、p95延迟、$/查询  |
|  15  |  漂移监控仪表盘  |  Phoenix实时仪表盘，显示每周检索质量趋势  |
|  **100**  |   |   |

## 练习

1. 在另一个管辖权限下构建第二个语料库切片（例如，在GDPR基础上增加HIPAA）。通过一个20问题跨管辖探测，展示角色+管辖过滤防止交叉泄露。

2. 在一周的生产流量中测量提示缓存命中率。识别哪些查询破坏了缓存前缀。进行重构。

3. 添加多轮对话记忆，使用1万令牌摘要缓冲区。测量随着对话增长，忠实度是否下降。

4. 将Claude Sonnet 4.7替换为自托管的Llama 3.3 70B。测量$/查询和忠实度差值。

5. 添加“不确定”模式：如果前几个重排序分数低于阈值，智能体回答“我没有信心提供引用”而不是给出答案。测量错误自信的减少。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
|  提示缓存  |  "缓存的系统+上下文"  |  Claude/OpenAI功能：命中时缓存前缀令牌折扣60-90%  |
| RAGAS | "RAG评估器" | 忠实度、答案相关性和上下文精度的自动评分 |
| 黄金测试集 | "带标注的评估" | 200多个专家标注的问答及引用；真实基准 |
| 管辖标签 | "合规标签" | 附加到文本块上的GDPR/HIPAA/SOC2范围；由检索过滤器强制执行 |
| 引用忠实度 | "依据回答率" | 可由可检索源片段支持的声明比例 |
| 漂移 | "检索质量衰减" | nDCG或引用分数的每周变化；告警阈值5% |
| 红队 | "对抗性评估" | 发布前的越狱测试、PII提取、域外探针 |

## 延伸阅读

- [Harvey AI](https://www.harvey.ai) — 参考法律生产栈
- [Harvey AI](https://www.harvey.ai) — 企业级RAG参考
- [Harvey AI](https://www.harvey.ai) — 开发者文档RAG参考
- [Harvey AI](https://www.harvey.ai) — 托管采集
- [Harvey AI](https://www.harvey.ai) — 成本杠杆参考
- [Harvey AI](https://www.harvey.ai) — 标准RAG评估框架
- [Harvey AI](https://www.harvey.ai) — 参考漂移可观测性
- [Harvey AI](https://www.harvey.ai) — 2026安全分类器
- [Harvey AI](https://www.harvey.ai) — 策略护栏框架
