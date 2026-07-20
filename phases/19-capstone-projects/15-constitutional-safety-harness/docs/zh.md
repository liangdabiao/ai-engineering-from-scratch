# 顶点项目 15 — 宪法级安全约束 + 红队靶场

> Anthropic的Constitutional Classifiers(宪法分类器)、Meta的Llama Guard 4(守卫4)、Google的ShieldGemma-2(盾牌Gemma-2)、NVIDIA的Nemotron 3内容安全(Nemotron 3 Content Safety)以及用于多语言覆盖的X-Guard(X守卫)定义了2026年的安全分类器堆栈。garak、PyRIT、NVIDIA Aegis和promptfoo成为标准对抗性评估工具。NeMo Guardrails v0.12将它们整合到生产管线中。本综合项目将所有内容连接在一起：围绕目标应用的分层安全防护、运行6类以上攻击的自主红队代理以及产生可测量无害性增量(Delta)的宪法自我批判运行。

**类型：** 综合项目
**语言：** Python（安全管线、红队）、YAML（策略配置）
**前置条件：** 第10阶段（从头构建LLM）、第11阶段（LLM工程）、第13阶段（工具）、第14阶段（智能体）、第18阶段（伦理、安全、对齐）
**实践阶段：** P10 · P11 · P13 · P14 · P18
**时间：** 25小时

## 问题

2026年LLM安全的前沿不在于分类器是否有效（它们大致有效），而在于如何围绕生产应用正确组合它们，既不过度拒绝也不留下明显漏洞。Llama Guard 4处理英文策略违规。X-Guard（132种语言）处理多语言越狱。ShieldGemma-2捕捉基于图像的提示注入。NVIDIA Nemotron 3内容安全覆盖企业类别。Anthropic的Constitutional Classifiers是一种独立方法，在训练期间而非服务期间使用。

攻击方式演变也很重要。PAIR和TAP自动发现越狱。GCG运行基于梯度的后缀攻击。多轮和代码转换(Code-Switch)攻击利用智能体记忆。任何部署的LLM都需要一个红队范围——garak和PyRIT是标准驱动——以及记录的缓解措施和CVSS评分发现。

你将加固一个目标应用（一个8B指令微调模型或其他综合项目中的RAG聊天机器人之一），对其运行6类以上攻击家族，并测量前后无害性。

## 概念

安全管线分为五层。**输入清洁**：去除零宽字符、解码base64/rot13、标准化Unicode。**策略层**：NeMo Guardrails v0.12护栏（域外、毒性、PII提取）。**分类器门**：输入上的Llama Guard 4、非英语上的X-Guard、图像输入上的ShieldGemma-2。**模型**：目标LLM。**输出过滤器**：输出上的Llama Guard 4、Presidio PII清理、适用时的引文执行。**人工介入层**：被标记为高风险输出进入Slack队列。

红队范围按调度程序运行。PAIR和TAP自主发现越狱。GCG运行基于梯度的后缀攻击。ASCII/base64/rot13编码攻击。多轮攻击（角色采纳、记忆利用）。代码转换攻击（混合英语与斯瓦希里语或泰语）。每次运行生成包含CVSS评分和披露时间线的结构化发现文件。

宪法自我批判运行是一种训练时干预。取1k个有害尝试提示，让模型草拟响应，根据书面宪法（不伤害规则）进行批判，并在批判循环上重新训练。在保留评估集上测量前后无害性增量(Delta)。

## 架构

```
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## 技术栈

- 安全分类器：Llama Guard 4、ShieldGemma-2、NVIDIA Nemotron 3 Content Safety、X-Guard
- 护栏框架：NeMo Guardrails v0.12 + OPA
- 红队驱动：garak (NVIDIA)、PyRIT (Microsoft Azure)、NVIDIA Aegis、promptfoo
- 越狱代理：PAIR (Chao et al., 2023)、Tree-of-Attacks (TAP)、GCG后缀
- 宪法训练：Anthropic式的自我批判循环 + 基于批判的SFT
- PII清理：Presidio
- 目标：一个8B指令微调模型或其他综合项目中的RAG聊天机器人

## 动手构建

1. **目标设置。** 在vLLM上搭建一个8B指令微调模型（或重用其他综合项目的RAG聊天机器人）。这是被测应用。

2. **安全管线包裹。** 在目标周围连接五层管线。验证每一层在Langfuse中单独可观测。

3. **分类器覆盖。** 加载Llama Guard 4、X-Guard（多语言）、ShieldGemma-2（图像）。在每个小标注集上运行以建立基线。

4. **红队调度器。** 调度garak、PyRIT、一个PAIR代理、一个TAP代理、一个GCG运行器、一个多轮攻击者以及一个代码转换攻击者。每个在独立队列上运行。

5. **攻击组合。** 六类攻击家族：(1) PAIR自动化越狱，(2) TAP树攻击，(3) GCG梯度后缀，(4) ASCII/base64/rot13编码，(5) 多轮角色，(6) 多语言代码转换。报告每类的成功率。

6. **宪法自我批判。** 整理1k个有害尝试提示。对每个提示，目标草拟一个响应。一个批判LLM根据书面宪法（“不伤害”、“引用证据”、“拒绝非法请求”）评分。批判者反对的提示会被重写；目标在批判改进的对上进行微调。在保留评估集上测量前后无害性。

7. **过度拒绝测量。** 在良性提示集（例如XSTest）上跟踪假阳性率。目标必须在良性问题上保持有用。

8. **CVSS评分。** 对每次成功的越狱，按CVSS 4.0评分（攻击向量、复杂度、影响）。生成披露时间线和缓解计划。

9. **范围自动化。** 以上所有内容在cron上运行；发现写入队列；过度拒绝回归警报发送到Slack。

## 使用它

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## 发布

`outputs/skill-safety-harness.md` 是可交付成果。一个生产级分层安全管线加上可复现的红队范围，带有前后无害性增量。

|  权重  |  标准  |  衡量方式  |
|:-:|---|---|
|  25  |  攻击面覆盖  |  6类以上攻击家族实施，2种以上语言  |
|  20  |  真阳性/假阳性权衡  |  攻击阻断率 vs XSTest良性通过率  |
|  20  |  自我批判增量  |  保留评估集上的前后无害性  |
|  20  |  文档与披露  |  带时间线的CVSS评分发现  |
|  15  |  自动化与可重复性  |  所有内容在cron上运行并带有警报  |
|  **100**  |   |   |

## 练习

1. 运行garak的提示注入插件于RAG聊天机器人，并比较有无输出过滤器层时的攻击成功率。

2. 添加第七类攻击家族：通过检索文档的间接提示注入。测量所需的额外防御。

3. 实现“带帮助的拒绝”模式：当护栏阻止时，目标提供更安全的相关答案而非直接拒绝。测量XSTest增量。

4. 多语言覆盖缺口：找到一种X-Guard表现不佳的语言。提议针对该语言微调的数据集。

5. 在30B模型上运行宪法自我批判，并测量增量是否扩展。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
| 分层安全 | "纵深防御" | 在输入、门控、输出、人工介入处设置多重防护栏 |
| Llama Guard 4 | "Meta的安全分类器" | 2026年参考输入/输出内容分类器 |
| PAIR | "越狱代理" | 论文(Chao等人)关于LLM驱动的越狱发现 |
| TAP | "攻击树" | PAIR的树搜索变体 |
| GCG | "贪心坐标梯度" | 基于梯度的对抗后缀攻击 |
| 宪法自批评 | "Anthropic式训练" | 目标草稿→评判者评分→重写→重新训练 |
| XSTest | "良性探测集" | 过度拒绝回归的基准测试 |
| CVSS 4.0 | "严重性评分" | 安全发现的标准漏洞评分 |

## 延伸阅读

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — 训练时参考
- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — 2026年输入/输出分类器
- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — 图像+多模态安全
- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — 企业参考
- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — 132语言多语言安全
- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — NVIDIA红队工具包
- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — Microsoft红队框架
- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — 护栏框架
- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — 越狱代理论文
