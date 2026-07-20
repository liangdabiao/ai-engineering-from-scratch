# ASCII艺术与视觉越狱

> Jiang, Xu, Niu, Xiang, Ramasubramanian, Li, Poovendran, "ArtPrompt: 基于ASCII艺术的对齐LLM越狱攻击" (ACL 2024, arXiv:2402.11753)。将有危害请求中的安全相关令牌(Token)遮蔽，替换为相同字母的ASCII艺术渲染，然后发送隐藏后的提示。GPT-3.5、GPT-4、Gemini、Claude、Llama-2都无法稳健识别ASCII艺术令牌。该攻击绕过了PPL（困惑度过滤器）、释义防御和重新分词。相关：ViTC基准测试衡量对非语义视觉提示的识别；StructuralSleight推广到不常见文本编码结构（树、图、嵌套JSON）作为一类编码攻击。

**类型:** 构建
**语言:** Python (stdlib, ArtPrompt令牌遮蔽工具)
**前提条件:** 阶段18 · 12 (PAIR), 阶段18 · 13 (MSJ)
**时间:** 约60分钟

## 学习目标

- 描述ArtPrompt攻击：单词识别步骤、ASCII艺术替换、最终隐藏提示。
- 解释标准防御（PPL、释义、重新分词）为何在ArtPrompt上失效。
- 定义ViTC并描述其衡量内容。
- 描述StructuralSleight作为对任意不常见文本编码结构的推广。

## 问题

通过释义和角色扮演的攻击（第12课）以及通过长上下文的攻击（第13课）在文本层面模式上操作。ArtPrompt在识别层面操作：模型不解析被禁止的令牌。它解析一个用字符渲染的图像。安全过滤器看到无害的标点符号。模型看到一个单词。

## 核心概念

### ArtPrompt，两步

步骤1. 单词识别。给定一个有危害的请求，攻击者使用LLM识别出与安全相关的单词（例如，“如何制作炸弹”中的“炸弹”）。

步骤2. 隐藏提示生成。将每个识别出的单词替换为其ASCII艺术渲染（一个7x5或7x7的字符块，形成字母形状）。模型收到一个由标点符号和空格组成的网格，足够强大的模型可以识别为单词；安全过滤器只看到网格。

结果：GPT-4、Gemini、Claude、Llama-2、GPT-3.5全部失败。在其基准子集上攻击成功率超过75%。

### 为什么标准防御失效

- **PPL（困惑度过滤器）。** ASCII艺术具有高困惑度——但所有新颖输入也是如此。阻止ArtPrompt的阈值选择也会阻止合法的结构化输入。
- **释义。** 对提示进行释义会破坏ASCII艺术。在实践中，释义LLM通常会保留或重建艺术。
- **重新分词。** 不同地切分令牌并不能改变模型的视觉正在识别字母形状这一事实。

根本问题在于安全过滤器是令牌级或语义级的；ArtPrompt在视觉识别级别操作。

### ViTC基准测试

对非语义视觉提示的识别。衡量模型阅读ASCII艺术、Webdings和其他非文本语义视觉内容的能力。ArtPrompt的有效性与ViTC准确性相关：模型读取视觉文本的能力越强，ArtPrompt在其上的效果越好。这是一种能力-安全权衡。

### StructuralSleight

推广了ArtPrompt：不常见文本编码结构（UTES）。树、图、嵌套JSON、JSON中的CSV、差异风格代码块。如果一个结构在训练安全数据中罕见但模型可解析，它可以隐藏有害内容。

防御启示：安全性必须推广到模型可以解析的各种结构化表示。这个集合很大且不断增长。

### 图像模态类比

视觉LLM（GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）扩展了攻击面。实际图像的ArtPrompt风格攻击比ASCII艺术类比更强，因为图像编码器产生更丰富的信号。

### 这在阶段18中的位置

第12-14课描述了三个正交攻击向量：迭代细化（PAIR）、上下文长度（MSJ）和编码（ArtPrompt/StructuralSleight）。第15课从模型中心攻击转移到系统边界攻击（间接提示注入）。第16课描述了防御工具响应。

## 使用它

`code/main.py` 构建了一个玩具ArtPrompt。你可以用ASCII艺术字形隐藏有害查询中的特定单词，验证隐藏字符串能通过关键词过滤器，并（可选）使用简单识别器将隐藏字符串解码回来。

## 发布

本课生成 `outputs/skill-encoding-audit.md`。给定一份越狱防御报告，它列举了所涵盖的编码攻击家族（ASCII艺术、base64、leet语、UTF-8同形字、UTES）以及捕获每种攻击的防御层。

## 练习

1. 运行 `code/main.py`。验证隐藏字符串能通过简单的关键词过滤器。报告所需的字符级变化。

2. 实现第二种编码：对同一目标单词使用base64。比较绕过过滤器率与ArtPrompt以及恢复难度。

3. 阅读Jiang等人2024年论文第4.3节（五个模型的结果）。提出一个理由，说明为什么Claude在相同基准测试中的ArtPrompt抵抗性高于Gemini。

4. 设计一种预生成防御，检测提示中的ASCII艺术形状区域。测量对合法代码、表格和数学符号的假阳性率。

5. StructuralSleight列出了10种编码结构。设计一种能够处理所有10种结构的通用防御，并估计每个受防御提示的计算成本。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
|  ArtPrompt  |  "ASCII艺术攻击"  |  两步越狱，用ASCII艺术渲染隐藏安全相关单词  |
|  Cloaking  |  "隐藏单词"  |  将禁止令牌替换为模型能读取但过滤器不能读取的视觉表示  |
|  UTES  |  "不常见结构"  |  不常见文本编码结构——树、图、嵌套JSON等，用于走私内容  |
|  ViTC  |  "visual-text capability"  |  衡量模型读取非语义视觉编码能力的基准测试  |
|  Perplexity filter  |  "PPL defense"  |  拒绝高困惑度的提示；但因合法结构化输入同样得分较高而失效  |
|  Retokenization  |  "tokenizer shift defense"  |  使用不同的分词器预处理提示；但因识别是视觉性的而失效  |
|  Homoglyph  |  "lookalike characters"  |  与拉丁字母外观相同的Unicode字符；绕过子串检查  |

## 延伸阅读

- [Jiang et al. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — the ASCII-art jailbreak paper
- [Jiang et al. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — UTES generalization
- [Jiang et al. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — complementary iterative attack
- [Jiang et al. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — complementary length attack
