# BERT — 掩码语言建模(Masked Language Modeling)

> GPT预测下一个词。BERT预测一个缺失的词。一句话的差异——以及半个时代的一切以嵌入为形状的事物。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段7·05（完整Transformer），阶段5·02（文本表示）
**时间：** 约45分钟

## 问题

2018年，每个NLP任务——情感分析、命名实体识别、问答、蕴涵——都在各自的标注数据上从头训练自己的模型。没有可以微调的预训练“理解英语”的检查点。ELMo（2018）展示了可以用双向LSTM预训练上下文嵌入；它有帮助，但未能泛化。

BERT（Devlin等人，2018）提出：如果我们采用一个Transformer编码器，在互联网上的每个句子上训练它，并迫使它从双向上下文中预测缺失的词，会怎样？然后在下游任务上微调一个头部。参数效率是一场启示。

结果：在18个月内，BERT及其变体（RoBERTa、ALBERT、ELECTRA）主导了所有存在的NLP排行榜。到2020年，地球上的每个搜索引擎、内容审核管道和语义搜索系统内部都有一个BERT。

到2026年，仅编码器(Encoder-only)模型仍然是分类、检索和结构化抽取的正确工具——它们每词元的运行速度比解码器快5–10倍，且它们的嵌入是现代所有检索堆栈的支柱。ModernBERT（2024年12月）将架构推至8K上下文，采用了Flash注意力 + RoPE + GeGLU。

## 核心概念

![Masked language modeling: pick tokens, mask them, predict originals](../assets/bert-mlm.svg)

### 训练信号

取一个句子：`the quick brown fox jumps over the lazy dog`。

随机掩码15%的词元：

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

训练模型在掩码位置预测原始词元。因为编码器是双向的，在位置1预测`[MASK]`可以使用位置2及之后的`brown fox jumps`。这是GPT做不到的事情。

### BERT掩码规则

在用于预测的15%的词元中：

- 80%被替换为`[MASK]`。
- 10%被替换为一个随机词元。
- 10%保持不变。

为什么不是始终使用`[MASK]`？因为`[MASK]`在推理时从不出现。训练模型期望100%的掩码位置为`[MASK]`会在预训练和微调之间产生分布偏移。10%随机+10%保持不变让模型保持诚实。

### 下一句预测(Next Sentence Prediction, NSP)——以及为何被抛弃

原始BERT还训练了NSP：给定两个句子A和B，预测B是否跟在A之后。RoBERTa（2019）通过消融实验证明NSP有害无益。现代编码器已跳过它。

### 2026年的变化：ModernBERT

2024年的ModernBERT论文用2026年的原语重建了模块：

|  组件 | 原始BERT (2018) | ModernBERT (2024)  |
|-----------|----------------------|-------------------|
|  位置编码 | 学习的绝对位置编码 | RoPE  |
|  激活函数 | GELU | GeGLU  |
|  归一化 | LayerNorm | 前归一化RMSNorm  |
|  注意力 | 全密集注意力 | 交替局部（128）+全局注意力  |
|  上下文长度 | 512 | 8192  |
|  分词器 | WordPiece | BPE  |

与2018年的堆栈不同，它是原生支持Flash注意力的。在序列长度8K下，推理速度比DeBERTa-v3快2–3倍，且GLUE得分更高。

### 2026年仍然选择编码器的用例

|  任务 | 为什么编码器优于解码器  |
|------|---------------------------|
|  检索/语义搜索嵌入 | 双向上下文 = 每个词元更好的嵌入质量  |
| 分类（情感、意图、有害性） | 一次前向传播；无生成开销 |
| 命名实体识别/Token标注 | 逐位置输出，原生双向 |
| 零样本蕴含（NLI） | 编码器顶部的分类头 |
| RAG的重排序器 | 交叉编码器评分，比LLM重排序器快10倍 |

```figure
transformer-residual
```

## 动手构建

### 第一步：掩码逻辑

参见 `code/main.py`。函数 `create_mlm_batch` 接受一个Token ID列表、词汇表大小和掩码概率。返回输入ID（已应用掩码）和标签（仅在掩码位置，其余为-100——PyTorch的忽略索引约定）。

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### 第二步：在小语料库上运行MLM预测

在一个包含20个单词、200个句子的词汇表上训练一个2层编码器+MLM头。无梯度——进行前向传播检查。完整训练需要PyTorch。

### 第三步：比较掩码类型

展示三路规则如何使模型在没有 `[MASK]` 的情况下保持可用。对未掩码句子和掩码句子进行预测。两者都应产生合理的Token分布，因为模型在训练中看到了两种模式。

### 第四步：微调头部

在一个玩具情感数据集上用分类头替换MLM头。仅头部训练；编码器冻结。这是每个BERT应用遵循的模式。

## 使用它

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**嵌入模型是微调后的BERT。**  `sentence-transformers` 模型（如 `all-MiniLM-L6-v2`）是使用对比损失训练的BERT。编码器相同。损失函数变了。

**交叉编码器重排序器也是微调后的BERT。** 在 `[CLS] query [SEP] doc [SEP]` 上进行配对分类。查询和文档之间的双向注意力正是交叉编码器优于双编码器的质量优势所在。

**2026年何时不选择BERT。** 任何生成式任务。编码器没有合理的方式来自回归生成Token。另外：任何参数少于10亿的小型解码器可以在相同质量下提供更多灵活性（如Phi-3-Mini、Qwen2-1.5B）。

## 发布

参见 `outputs/skill-bert-finetuner.md`。该技能为一个新的分类或提取任务定义了BERT微调的范围（骨干网络选择、头部规范、数据、评估、停止）。

## 练习

1. **简单。** 运行 `code/main.py` 并打印10000个Token上的掩码分布。确认约15%被选中，其中约80%变为 `[MASK]`。
2. **中等。** 实现整词掩码：如果一个单词被分词为子词，则要么全部掩码，要么都不掩码。测量这是否能提高500句语料库上的MLM准确率。
3. **困难。** 在一个包含10000个句子的公共数据集上训练一个小型（2层，d=64）BERT。微调 `code/main.py` Token用于SST-2情感分析。与参数匹配的解码器基线相比，哪一个更好？

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
| MLM | "掩码语言建模" | 训练信号：随机将15%的Token替换为 `[MASK]`，预测原始Token。 |
| 双向 | "双向查看" | 编码器注意力无因果掩码——每个位置都能看到所有其他位置。 |
| `[CLS]` | "汇聚Token" | 一个特殊Token，附加到每个序列开头；其最终嵌入用作句子级表示。 |
| `[SEP]` | "片段分隔符" | 分隔配对序列（例如查询/文档，句子A/B）。 |
| NSP | "下一句预测" | BERT的第二个预训练任务；在RoBERTa中被证明无用，2019年后被弃用。 |
| 微调 | "适应任务" | 保持编码器基本冻结；在顶部训练一个小型头部用于下游任务。 |
| 交叉编码器 | "重排序器" | 一种BERT，将查询和文档都作为输入，输出相关性分数。 |
| ModernBERT | "2024年刷新" | 使用RoPE、RMSNorm、GeGLU、交替局部/全局注意力、8K上下文重新构建的编码器。 |

## 延伸阅读

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — 原始论文。
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — 如何正确训练BERT；弃用NSP。
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — 在相同计算量下，替换Token检测优于MLM。
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — ModernBERT论文。
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — 经典编码器参考资料。
