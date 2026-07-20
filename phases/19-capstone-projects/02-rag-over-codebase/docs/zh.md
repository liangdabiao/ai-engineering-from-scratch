# 顶点项目 02 — 基于代码库检索增强生成（跨仓库语义搜索）

> 每个严肃的工程组织在2026年都会运行一个理解含义而不仅仅是字符串的内部代码搜索。Sourcegraph Amp、Cursor 的代码库答案、Augment 的企业图、Aider 的仓库地图、Pinterest 的 MCP 内部版——同样的形态。摄取多个仓库，用 tree-sitter 解析，嵌入函数级和类级代码块，混合搜索，重新排序，带引用给出答案。本毕业设计要求你构建一个能处理 10 个仓库中 200 万行代码，并在每次 git 推送时进行增量重新索引的系统。

**类型：** 毕业设计
**语言：** Python（摄取），TypeScript（API + UI）
**前置条件：** 阶段 5（NLP 基础），阶段 7（变换器），阶段 11（LLM 工程），阶段 13（工具），阶段 17（基础设施）
**涉及阶段：** P5 · P7 · P11 · P13 · P17
**时间：** 30 小时

## 问题

到2026年，每个前沿编码代理都会配备代码库检索层，因为仅靠上下文窗口无法解决跨仓库问题。Claude 的 100 万 token 上下文有所帮助；但它并不能消除对排序检索的需求。对原始代码块进行简单的余弦搜索会污染结果，尤其是在生成代码、单仓库重复以及很少导入的符号长尾上。生产级答案是针对 AST 感知块进行混合（稠密 + BM25）搜索，结合重新排序器，并以符号引用图为支撑。

你通过索引一个真实的仓库群（而不是一个教程仓库）并测量 MRR@10、引用忠实度和增量新鲜度来学习这一点。故障模式是基础设施层面的：一个包含 10 万文件的单仓库，一个修改了一半文件的推送，一个需要跨越四个仓库才能正确回答的查询。

## 概念

AST 感知的摄取管道使用 tree-sitter 解析每个文件，提取函数和类节点，并在节点边界处切块，而不是固定的 token 窗口。每个代码块获得三种表示：稠密嵌入（Voyage-code-3 或 nomic-embed-code）、稀疏 BM25 词项和一段简短的自然语言摘要。该摘要增加了第三种可检索的模态——用户询问“X 如何被授权”，而摘要提到“authz”，即使代码中只有 `check_permission`。

检索是混合式的。查询同时触发稠密搜索和 BM25 搜索，合并 top-k，并将并集交给交叉编码器重新排序器（Cohere rerank-3 或 bge-reranker-v2-gemma-2b）。重新排序后的列表进入长上下文合成器（带有提示缓存的 Claude Sonnet 4.7，或自托管的 Llama 3.3 70B），并附带指令要求每个声明都按文件和行范围进行引用。没有引用的答案会被后置过滤器拒绝。

增量新鲜度是基础设施问题。Git 推送触发差异比较：哪些文件发生变化，哪些符号发生变化。只有受影响的代码块重新嵌入。受影响的跨文件符号边（导入、方法调用）被重新计算。索引保持一致，无需每次提交都重新处理 200 万行代码。

## 架构

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## 技术栈

- 解析：tree-sitter 支持 17 种语言语法（Python、TS、Rust、Go、Java、C++ 等）
- 稠密嵌入：Voyage-code-3（托管）或 nomic-embed-code-v1.5（自托管），回退到 bge-code-v1
- 稀疏索引：Tantivy（Rust）使用 BM25F，按符号名与正文进行字段加权
- 向量数据库：支持混合搜索的 Qdrant 1.12，或针对少于 5000 万向量团队使用的 pgvector + pgvectorscale
- 代码块摘要模型：带有提示缓存的 Claude Haiku 4.5 或 Gemini 2.5 Flash
- 重新排序器：Cohere rerank-3 或自托管的 bge-reranker-v2-gemma-2b
- 协调：用于摄取的 LlamaIndex Workflows，用于查询代理的 LangGraph
- 合成器：带有提示缓存的 Claude Sonnet 4.7（100 万上下文）
- 符号图：用于导入和调用边的 Neo4j（托管）或 kuzu（嵌入式）
- 可观测性：每个检索和合成步骤的 Langfuse spans

## 动手构建

1. **摄取遍历器。** 在每个推送钩子上迭代 git 历史。收集变更的文件。对于每个文件，使用 tree-sitter 解析，提取函数和类节点及其完整源代码跨度。发出代码块记录 `{repo, path, start_line, end_line, symbol, body}`。

2. **代码块摘要器。** 将代码块批量处理为对 Haiku 4.5 的调用，对系统前导进行提示缓存。提示：“用一句话总结此函数，说明其公共合约和副作用。”将摘要与代码块一起存储。

3. **嵌入池。** 两个并行队列：稠密（Voyage-code-3 批量 128）和摘要（相同模型，但在摘要字符串上）。将向量写入 Qdrant，附带载荷 `{repo, path, start_line, end_line, symbol, kind}`。

4. **BM25 索引。** 字段加权的 Tantivy 索引：符号名权重 4，符号主体权重 1，摘要权重 2。支持“查找名为 X 的函数”以及“查找做 X 的函数”这类查询。

5. **符号图。** 对于每个代码块，记录边：导入（此文件使用仓库 Z 中的符号 Y）、调用（此函数调用类 C 上的方法 M）、继承。存储在 kuzu 中。在查询时用于跨仓库边界扩展检索。

6. **查询代理。** 包含三个节点的 LangGraph。`retrieve` 并行触发稠密搜索和 BM25 搜索，按（仓库、路径、符号）去重。`rerank` 对前 50 个结果运行交叉编码器，保留前 10 个。`synth` 使用上下文中重新排序后的代码块调用 Claude Sonnet 4.7，缓存系统提示，要求文件:行引用。

7. **引用强制执行。** 解析模型输出；任何没有 `(repo/path:start-end)` 锚点的声明都会被标记为重新提问或丢弃。仅返回带引用的答案给用户。

8. **增量重新索引。** 在每个 webhook 上，计算符号级的差异。仅重新嵌入文本发生变化的代码块。对于导入发生变化的代码块，重新计算符号边。衡量：对于 200 万行代码的仓库群，一个包含 50 个文件的推送在 60 秒内完成重新索引。

9. **评估。** 对 100 个跨仓库问题标注黄金文件:行答案。测量 MRR@10、nDCG@10、引用忠实度（具有可验证锚点的声明比例）以及 p50/p99 延迟。

## 使用它

```
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## 发布

可交付技能 `outputs/skill-codebase-rag.md`。给定一个仓库语料库，它搭建起摄取管道、混合索引和查询代理，并针对任何跨仓库问题返回带有引用的答案。评分标准：

|  权重  |  标准  |  衡量方式  |
|:-:|---|---|
|  25  |  检索质量  |  在 100 个问题的预留集上的 MRR@10 和 nDCG@10  |
|  20  |  引用忠实度  |  具有可验证文件:行锚点的答案声明比例  |
|  20  |  延迟与规模  |  在索引语料库大小上，10k QPS 下的 p95 查询延迟  |
|  20  |  增量索引正确性  |  从 git 推送到 50 个文件提交变为可搜索的时间  |
|  15  |  用户体验与答案格式  |  引用可点击性、摘要预览、跟进提示  |
|  **100**  |   |   |

## 练习

1. 将 Voyage-code-3 替换为自托管的 nomic-embed-code。测量 MRR@10 的变化量。报告在启用重新排序时差距是否缩小。

2. 向语料库注入 20% 的生成代码（LLM 产生的样板代码）并重新评估。观察检索污染。向载荷添加“generated”标志并降低这些命中结果的权重。

3. 在你的语料库规模上对 Qdrant 混合搜索与 pgvector + pgvectorscale 进行基准测试。报告批次大小为 1 时的 p99。

4. 添加基于采样的漂移检查：每周重新运行 100 个问题的评估。当 MRR@10 下降超过 5% 时发出警报。

5. 扩展到跨语言符号解析：一个调用通过 gRPC 的 Go 服务的 Python 函数。使用符号图将它们链接起来。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
|  AST 感知分块  |  “函数级拆分”  |  在 tree-sitter 节点边界处切分代码，而不是固定 token 窗口  |
| 混合搜索(Hybrid search)  |  "稠密+稀疏(Dense + sparse)"  |  并行运行BM25和向量搜索，合并top-k结果，重新排序 |
| 交叉编码器重排序(Cross-encoder rerank)  |  "第二阶段排序(Second-stage rank)"  |  对每个(查询, 候选)对进行联合评分的模型，比余弦相似度更准确 |
| 提示缓存(Prompt caching)  |  "缓存系统提示(Cached system prompt)"  |  2026年Claude/OpenAI的功能，对重复前缀token最多优惠90% |
| 符号图(Symbol graph)  |  "代码图(Code graph)"  |  表示跨文件和仓库的导入、调用、继承关系的边 |
| 引用忠实度(Citation faithfulness)  |  "可验证答案率(Grounded answer rate)"  |  用户通过点击锚点并阅读引用片段能够验证的声明比例 |
| 增量重建索引(Incremental re-index)  |  "推送至可搜索时间(Push-to-search time)"  |  从git push到变更符号可查询的挂钟时间 |

## 延伸阅读

- [Sourcegraph Amp](https://ampcode.com) — 生产环境的跨仓库代码智能
- [Sourcegraph Amp](https://ampcode.com) — 本顶点项目的参考深度解读
- [Sourcegraph Amp](https://ampcode.com) — tree-sitter排序的仓库视图
- [Sourcegraph Amp](https://ampcode.com) — 商业符号图RAG
- [Sourcegraph Amp](https://ampcode.com) — 参考实现
- [Sourcegraph Amp](https://ampcode.com) — Voyage-code-3详情
- [Sourcegraph Amp](https://ampcode.com) — 交叉编码器参考
- [Sourcegraph Amp](https://ampcode.com) — 内部平台参考
