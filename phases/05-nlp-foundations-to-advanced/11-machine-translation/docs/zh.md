# 机器翻译

> 翻译是支撑了NLP研究三十年并仍在持续的任务。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段5·10（注意力机制），阶段5·04（GloVe、FastText、子词）
**时间：** 约75分钟

## 问题

模型读取一种语言的句子，生成另一种语言的句子。长度可变，词序可变。某些源词映射到多个目标词，反之亦然。习语拒绝一一映射。法语的“I miss you”是“tu me manques”——字面意思是“你对我缺失”。没有词级别的对齐能通过这个测试。

机器翻译是迫使NLP发明编码器-解码器、注意力机制、Transformer，最终是整个LLM范式的任务。每一步进展都是因为翻译质量可衡量，且人与机器之间的差距顽固不化。

本课跳过历史，直接讲授2026年的工作流程：预训练多语言编码器-解码器（NLLB-200或mBART）、子词分词、束搜索、BLEU和chrF评估，以及那些仍未在生产中被捕获的少数失败模式。

## 核心概念

![MT pipeline: tokenize → encode → decode with attention → detokenize](../assets/mt-pipeline.svg)

现代MT是在平行文本上训练的Transformer编码器-解码器。编码器以其语言的分词读取源文本。解码器通过交叉注意力（第10课）利用编码器的输出，一次生成一个子词。解码使用束搜索以避免贪婪解码陷阱。输出经过去分词、去大小写还原，并与参考文本评分。

三个操作选择决定了实际MT质量。

- **分词器。** 在多语言语料库上训练的SentencePiece BPE。跨语言共享词汇使得NLLB中的零样本对成为可能。
- **模型大小。** NLLB-200蒸馏版600M可在笔记本电脑上运行。NLLB-200 3.3B是已发布的生产默认版本。54.5B是研究上限。
- **解码。** 一般内容束宽4-5。长度惩罚以避免输出过短。需要术语一致性时使用约束解码。

```figure
seq2seq-alignment
```

## 动手构建

### 步骤1：调用预训练MT

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

这里有三点重要。`src_lang`告诉分词器使用哪种文字和切分。`forced_bos_token_id`告诉解码器生成哪种语言。两者都是NLLB特定的技巧；mBART和M2M-100使用各自的约定，不可互换。

### 步骤2：BLEU和chrF

BLEU衡量输出与参考之间的n-gram重叠。四个参考n-gram大小（1-4），精确率的几何平均值，对过短输出的简短惩罚。分数范围为[0, 100]。常用但解释困难：30 BLEU是“可用”；40是“良好”；50是“优秀”；1 BLEU以下的差异是噪声。

chrF衡量字符级别的F值。对形态丰富的语言更敏感，因为BLEU在这些语言中低估了匹配。通常与BLEU一起报告。

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

始终使用`sacrebleu`。它标准化分词，使得分数在论文间可比。自行计算BLEU正是产生误导性基准的原因。

### 三级评估体系（2026）

现代MT评估使用三个互补的指标系列。至少使用两个。

- **启发式**（BLEU, chrF）。快速、基于参考、可解释，但对释义不敏感。用于遗留比较和回归检测。
- **学习型**（COMET, BLEURT, BERTScore）。基于人类判断训练的神经模型；比较翻译与源文和参考的语义相似性。COMET自2023年以来与MT研究关联最高，是2026年质量关键场景的生产默认选择。
- **LLM作为评判**（无参考）。提示大模型根据流畅性、充分性、语气、文化适宜性对翻译评分。当评分准则设计良好时，GPT-4作为评判与人类一致性达到约80%。用于不存在参考的开放式内容。

2026年实用堆栈：`sacrebleu`用于BLEU和chrF，`unbabel-comet`用于COMET，以及一个提示的LLM用于最终面向人类的信号。在信任生产数据之前，先用50-100个人工标注样本校准每个指标。

无参考指标（COMET-QE, BLEURT-QE, LLM-as-judge）允许你在没有参考的情况下评估翻译，这对于不存在参考翻译的长尾语言对很重要。

### 步骤3：生产中出什么问题

上述工作流程在80%的时间里会流畅翻译，但在剩下的20%中会静默失败。已命名的失败模式：

- **幻觉。** 模型编造源文中不存在的内容。在不熟悉的领域词汇中常见。症状：输出流畅但声称了源文未提及的事实。缓解：对领域术语使用约束解码，对受监管内容进行人工审核，监控输出是否远长于输入。
- **非目标语言生成。** 模型翻译成错误语言。NLLB在稀有语言对上出奇地容易发生。缓解：验证`forced_bos_token_id`，并始终使用语言识别模型检查输出。
- **术语漂移。** “Sign up”在文档1中变成“s'inscrire”，在文档2中变成“créer un compte”。对于UI文本和面向用户的字符串，一致性比原始质量更重要。缓解：使用词汇表约束解码或后期编辑字典。
- **礼貌级别不匹配。** 法语的“tu”与“vous”，日语的礼貌级别。模型会选择训练中更常见的形式。对于面向客户的内容，这通常是错误的。缓解：如果模型支持，在提示前缀中添加礼貌级别标记，或仅在正式语料上微调小模型。
- **短输入长度爆炸。** 非常短的输入句子常常产生过长的翻译，因为长度惩罚在源词少于约5个时急剧下降。缓解：设置与源长度成比例的最大长度上限。

### 步骤4：为特定领域微调

预训练模型是通才。法律、医学或游戏对话翻译通过领域平行数据微调可显著受益。配方并不奇特：

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

几千个高质量平行示例胜过几十万个有噪声的网页抓取示例。训练数据的质量是生产中最核心的杠杆。

## 使用它

2026年MT生产堆栈：

|  用例  |  推荐起点  |
|---------|---------------------------|
|  任意到任意，200种语言  |  `facebook/nllb-200-distilled-600M`（笔记本电脑）或`nllb-200-3.3B`（生产）  |
|  以英语为中心，高质量，50种语言  |  `facebook/mbart-large-50-many-to-many-mmt`  |
|  短文本、低开销推理、英法/德/西语  |  Helsinki-NLP / Marian 模型  |
|  对延迟敏感的浏览器端  |  ONNX量化后的Marian（约50 MB）  |
|  最高质量，愿意付费  |  GPT-4 / Claude / Gemini 配合翻译提示  |

截至2026年，LLM已在多个语言对上超越专门的机器翻译模型，尤其是在习语内容和长上下文方面。代价是每个token的成本和延迟。当上下文长度、风格一致性或通过提示进行领域适配比吞吐量更重要时，选择LLM。

## 发布

保存为 `outputs/skill-mt-evaluator.md`：

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## 练习

1. **简单.** 将一段含5个句子的英文段落翻译成法语，再使用`nllb-200-distilled-600M`译回英语。测量往返结果与原文的接近程度。你应该看到语义保留但用词有所漂移。
2. **中等.** 使用`nllb-200-distilled-600M`或`fasttext lid.176`实现对翻译输出的语言ID检查。集成到机器翻译调用中，以便在返回前捕获偏离目标语言的生成结果。
3. **困难.** 在一个你选择的5000对领域语料库上，对`nllb-200-distilled-600M`进行微调。在微调前后分别测量保留集上的BLEU值。报告哪些类型的句子有所改进，哪些出现了退化。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  BLEU  |  翻译评分  |  带简短惩罚的N-gram精确率。[0, 100]。  |
|  chrF  |  字符F值  |  字符级F值。对形态丰富的语言更敏感。  |
|  NMT  |  神经机器翻译  |  基于并行文本训练的Transformer编码器-解码器。2017年后的默认方案。  |
|  NLLB  |  不落下任何语言  |  Meta的200种语言机器翻译模型系列。  |
|  受限解码  |  受控输出  |  强制指定token或n-gram在输出中出现或不出现。  |
|  幻觉  |  虚构内容  |  模型输出中未经源文本支持的内容。  |

## 延伸阅读

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672)——NLLB论文。
- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672)——为什么[Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/)是报告BLEU的唯一正确方式。
- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672)——chrF论文。
- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672)——实用的微调教程。
