# 顶点项目 05 — 自主研究智能体（AI-Scientist 级别）

> Sakana的AI科学家v2发表了完整的论文。Agent实验室进行了实验。Allen AI分享了轨迹。2026年的形态是：在实验上进行计划-执行-验证树搜索，预算成本，沙箱化代码执行，一个视觉反馈的LaTeX写作器，以及一个自动化的NeurIPS风格审稿人集成。顶点项目是构建一个这样的系统，每篇论文端到端运行成本低于30美元，并经受住Sakana记录的那些沙箱逃逸红队测试。

**类型：** 顶点项目
**语言：** Python（智能体+沙箱），LaTeX（输出）
**先修课程：** 第二阶段（机器学习），第三阶段（深度学习），第七阶段（Transformer），第十阶段（从头构建LLM），第十四阶段（智能体），第十五阶段（自主系统），第十六阶段（多智能体），第十八阶段（安全）
**涉及的阶段：** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18
**时间：** 40小时

## 问题

自主研究智能体在2026年跨过了一个门槛。Sakana AI的AI科学家v2发表在《自然》杂志上，其生成的论文通过了研讨会同行评审。ShinkaEvolve（ICLR 2026）将这一思路扩展到进化假说。AMD的智能体实验室发布了可复现的轨迹。这些智能体并非魔法——它们是一个在候选实验树上运行的计划-执行-验证循环，带有成本上限、种子绑定的沙箱和自动评审。其技巧在于循环本身、预算和安全策略。

你通过在一个狭窄领域内针对一个种子想法实现这个循环来学习它（例如，在一个1亿参数的Transformer上做注意力稀疏性消融实验）。其价值并不在于第一次运行就能发现新东西，而在于基础设施：树搜索、实验沙箱、写作-审稿循环、红队报告。Sakana团队记录了沙箱逃逸的失败案例；你的智能体必须通过相同的红队测试。

## 概念

智能体是一种最佳优先树搜索。节点是实验描述：（假说、配置、代码、预期结果）。扩展步骤通过小的修改（更换优化器、调整批量大小、消融一个组件）来生成子节点。每个子节点在一个全新的沙箱中运行，并带有严格的资源上限。结果反馈给评分函数，该函数根据（新颖性 × 质量 × 剩余预算）对节点进行排序。树不断增长直到预算耗尽，然后最佳分支被撰写成文。

写作器是多模态的。它生成LaTeX草稿，编译它，渲染图形，并将渲染后的PDF反馈给Claude Opus 4.7的视觉模式，以获得关于布局、图形可读性和主张-证据一致性的批评。一个由五个LLM评委组成的审稿人集成输出NeurIPS风格的分数（新颖性、严谨性、清晰性、可复现性、影响力）；如果平均分低于阈值，论文将连同批评意见返回给写作器。

安全是承重结构。每个实验都在E2B或Daytona沙箱中运行，没有网络出口，有壁钟时间限制和固定的资源限制。智能体的代码生成步骤通过一个策略层，该层阻止逃逸沙箱的系统调用。红队报告再现了Sakana记录的受攻击面（fork炸弹、文件系统逃逸、LLM编写的网络调用）。

## 架构

```
seed idea + domain
      |
      v
  literature search (Semantic Scholar + OpenAlex + FAISS cache)
      |
      v
  LangGraph plan-execute-verify tree
      |
      v
  +--- expand node ----+      per-node sandbox
  |                    |      (E2B / Daytona)
  v                    v      resource caps
  child_1           child_k   no network egress
  |                    |      deterministic seeds
  v                    v
  run experiment       run experiment
  |                    |
  v                    v
  score nodes by (novelty, quality, budget)
      |
      v
  best branch -> LaTeX writer
      |
      v
  compile + vision critique (Opus 4.7 vision)
      |
      v
  reviewer ensemble (5 LLM judges, NeurIPS rubric)
      |
      v
  paper.pdf + review.md + trace.json
```

## 技术栈

- 编排：LangGraph（带检查点和人工批准门控）
- 树搜索：自定义最佳优先搜索实验节点（类似Sakana v2的AB-MCTS风格）
- 沙箱：每个实验使用E2B，备用方案为Docker-in-Docker；通过cgroups设置资源上限
- 文献：Semantic Scholar图API + OpenAlex + 本地FAISS摘要缓存
- 写作器：LaTeX模板 + Claude Opus 4.7（视觉模式）用于图形批评和布局
- 审稿人：5个评委集成（Opus 4.7, GPT-5.4, Gemini 3 Pro, DeepSeek R1, Qwen3-Max），加权聚合
- 实验框架：PyTorch 2.5用于物理实验，W&B用于日志记录
- 可观测性：Langfuse用于智能体轨迹，每篇论文硬预算30美元

## 动手构建

1. **种子和领域范围界定。** 选择一个种子想法（例如，“研究低于1B参数的Transformer注意力图中的稀疏性模式”）。定义搜索空间：模型、数据集、计算预算。

2. **文献检索。** 查询Semantic Scholar + OpenAlex，获取50篇最相关的引用论文；本地缓存摘要；生成1页的领域摘要。

3. **树框架。** 用种子假说初始化根节点。实现`expand(node) -> children`，使用做小修改的提议（每个子节点一个配置变化）。实现`score(node)`作为加权的新颖性×质量×预算项。

4. **沙箱封装。** 每个实验运行`docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only`（或等效的E2B策略）。种子写入沙箱；输出以只读方式挂载出来。

5. **计划-执行-验证循环。** `plan`提议子节点。`execute`运行沙箱，捕获日志和指标。`verify`对指标进行单元检查（损失是否下降？消融是否隔离了效果？）。失败的节点在树上存储失败原因。

6. **写作器。** 预算用完后，选择最佳分支。用matplotlib渲染图形。通过Claude Opus 4.7生成LaTeX草稿，并附上分支轨迹的上下文。编译。将编译后的PDF反馈给Opus 4.7视觉模式进行批评。迭代。

7. **审稿人集成。** 五个评委根据NeurIPS风格的评分标准对草稿进行评分（新颖性、严谨性、清晰性、可复现性、影响力）。如果平均分<4.0/5，则连同批评意见返回写作器。最多重写3次后硬停止。

8. **红队。** 构建或集成一组针对沙箱的对抗性任务：fork炸弹、网络外泄尝试、文件系统逃逸、LLM编写的shell元字符。确认所有攻击都被阻止。记录发现。

9. **可复现性。** 每篇论文附带其树搜索轨迹JSON、种子、W&B运行链接、沙箱配置以及端到端复现的README。

## 使用它

```
$ ai-scientist run --seed "attention sparsity in sub-1B transformers" --budget 30
[lit]    50 papers, digest in 12s
[tree]   expanded 8 nodes, budget 12/30
[exec]   node #3 sparsity=top-8, loss=2.83 (best so far)
[exec]   node #6 sparsity=top-4, loss=3.12 (worse)
[exec]   ...
[tree]   chose branch rooted at node #3 (novelty 0.62, quality 0.81)
[write]  LaTeX draft v1 complete
[vision] critique: figure 2 legend too small, claim-evidence ok
[write]  draft v2 after 3 edits
[review] mean 4.2/5 (novelty 3.9, rigor 4.3, clarity 4.1, repro 4.5, impact 4.2)
[done]   paper.pdf + review.md + trace.json     $28.40 spent
```

## 发布

`outputs/skill-ai-scientist.md`是交付物。给定一个种子想法+一个领域+30美元预算，它运行完整流程并输出一篇可审阅的论文以及一个可复现性包。

|  权重  |  标准  |  衡量方式  |
|:-:|---|---|
|  25  |  论文质量  |  针对已发表的研讨会论文进行盲审评分  |
|  20  |  实验严谨性  |  基线、种子、消融；每个主张都有结果表中的一个单元格支持  |
|  20  |  成本与计算纪律  |  严格执行每篇论文30美元上限，Langfuse追踪  |
|  20  |  安全性  |  沙箱红队通过；网络策略和终止开关已验证  |
|  15  |  可复现性  |  使用相同种子一键重新运行可复现论文  |
|  **100**  |   |   |

## 练习

1. 针对同一领域内的三个不同种子想法运行流程。比较树搜索的哪些部分重叠。识别重复浪费的计算。

2. 在实验执行前添加一个人工在环门控，用于预估成本超过5美元的节点。衡量总成本降低了多少。

3. 将审稿人集成替换为单个评委。在已知质量差的论文保留集上衡量误接受率。

4. 引入一个网络外泄红队测试：智能体编写代码尝试`curl`一个外部地址。确认`--network=none`策略阻止了该行为。记录尝试。

5. 将你的树搜索与平面随机基线（相同预算，无扩展策略）进行比较。报告新颖性×质量的提升。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
|  树搜索  |  "AB-MCTS风格扩展"  |  使用新颖性×质量×预算分数对实验节点进行最佳优先探索  |
|  沙箱(Sandbox)  |  "实验隔离(Experiment isolation)"  |  无网络、受CPU/内存限制、固定种子、只读输入的容器  |
|  视觉批判(Vision critique)  |  "先渲染后读取(Render-then-read)"  |  将论文编译为PDF，将PDF反馈给视觉语言模型(VLM)进行布局和主张-证据批判  |
|  评审者集成(Reviewer ensemble)  |  "自动同行评审(Automated peer review)"  |  多个大语言模型(LLM)评审者使用NeurIPS评分标准对论文评分；加权聚合控制流水线  |
|  新颖性分数(Novelty score)  |  "这是新的吗？"  |  一种启发式方法，对接近50篇论文文献缓存的内容进行惩罚  |
|  成本上限(Cost ceiling)  |  "$预算"  |  每篇论文总花费的硬上限；Langfuse计数器 + 运行前预估  |
|  红队(Red team)  |  "沙箱逃逸审计(Sandbox-escape audit)"  |  如果策略错误则逃逸沙箱的对抗性任务  |

## 延伸阅读

- [Sakana AI-Scientist-v2 repository](https://github.com/SakanaAI/AI-Scientist-v2) — 参考生产研究代理(Reference production research agent)
- [Sakana AI-Scientist-v2 repository](https://github.com/SakanaAI/AI-Scientist-v2) — 原始方法论(Original methodology)
- [Sakana AI-Scientist-v2 repository](https://github.com/SakanaAI/AI-Scientist-v2) — 进化扩展(Evolutionary extension)
- [Sakana AI-Scientist-v2 repository](https://github.com/SakanaAI/AI-Scientist-v2) — 多角色研究实验室框架(Multi-role research-lab framework)
- [Sakana AI-Scientist-v2 repository](https://github.com/SakanaAI/AI-Scientist-v2) — 参考编排层(Reference orchestration layer)
- [Sakana AI-Scientist-v2 repository](https://github.com/SakanaAI/AI-Scientist-v2) — 文献搜索(Literature search)
- [Sakana AI-Scientist-v2 repository](https://github.com/SakanaAI/AI-Scientist-v2) — 参考实验隔离(Reference experiment isolation)
- [Sakana AI-Scientist-v2 repository](https://github.com/SakanaAI/AI-Scientist-v2) — 评审者集成编码的评分标准(The rubric the reviewer ensemble encodes)
