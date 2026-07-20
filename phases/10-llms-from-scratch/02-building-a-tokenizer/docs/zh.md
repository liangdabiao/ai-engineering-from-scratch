# 构建分词器从零开始

> 第01课给了你一个玩具。本课给你一件武器。

**类型：** 构建
**语言：** Python
**前置条件：** 第10阶段，第01课（分词器：BPE、WordPiece、SentencePiece）
**时间：** 约90分钟

## 学习目标

- 构建一个处理Unicode、空白标准化和特殊标记的生产级BPE分词器
- 实现字节级回退，使分词器能够编码任何输入（包括表情符号、中日韩文字和代码）而不会有未知标记
- 添加预分词正则表达式模式，在应用BPE合并之前按词边界分割文本
- 在语料库上训练自定义分词器，并在多语言文本上评估其相对于tiktoken的压缩比

## 问题

你在第01课中的BPE分词器可以处理英文文本。现在用日语试试。或者表情符号。或者混合了制表符和空格的Python代码。

它会失效。

不是因为BPE错了——而是因为实现不完整。一个生产级分词器处理任何编码中的原始字节，在分割之前对Unicode进行标准化，管理从不合并的特殊标记，将预分词与子词拆分链接起来，并且所有这些操作都要足够快，以免成为处理15万亿标记的训练管道的瓶颈。

GPT-2的分词器有50,257个标记。Llama 3有128,256个。GPT-4大约有100,000个。这些不是玩具数字。那些词汇背后的合并表是在数百GB的文本上训练的，而周围的机制——标准化、预分词、特殊标记注入、聊天模板格式化——正是区分一个处理“hello world”的分词器和一个处理整个互联网的分词器的关键。

你将构建这个机制。

## 核心概念

### 完整管道

一个生产级分词器不是一个算法。它是一个由五个阶段组成的管道，每个阶段解决不同的问题。

```mermaid
graph LR
    A[Raw Text] --> B[Normalize]
    B --> C[Pre-Tokenize]
    C --> D[BPE Merge]
    D --> E[Special Tokens]
    E --> F[Token IDs]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

每个阶段都有特定的工作：

|  阶段  |  作用  |  重要性  |
|-------|-------------|----------------|
|  标准化  |  NFKC Unicode，可选小写化，可选去重音  |  “fi”连字（U+FB01）变为“fi”（两个字符）。没有这个，相同的词会得到不同的标记。  |
|  预分词  |  在BPE之前将文本分割成块  |  防止BPE跨词边界合并。“the cat”绝不应产生标记“e c”。  |
|  BPE合并  |  将学到的合并规则应用于字节序列  |  核心压缩。将原始字节转换为子词标记。  |
|  特殊标记  |  注入[BOS]、[EOS]、[PAD]、聊天模板标记  |  这些标记有固定的ID。它们从不参与BPE合并。模型需要它们来构建结构。  |
|  ID映射  |  将标记字符串转换为整数ID  |  模型看到的是整数，而不是字符串。  |

### 字节级BPE

第01课的分词器操作在UTF-8字节上。那是正确的选择。但我们跳过了重要的一点：当这些字节不是有效的UTF-8时会发生什么？

字节级BPE通过将每一个可能的字节值（0-255）视为有效标记来解决这个问题。你的基础词汇正好有256个条目。任何文件——文本、二进制、损坏的——都可以被标记化而不会产生未知标记。

GPT-2添加了一个技巧：将每个字节映射到一个可打印的Unicode字符，以便词汇表对人类可读。字节0x20（空格）在他们的映射中变成了字符“G”。这纯粹是装饰性的。算法并不关心。

真正的力量：字节级BPE处理地球上的每一种语言。汉字每个是3个UTF-8字节。日语可以是3-4个字节。阿拉伯语、天城文、表情符号——都只是字节序列。BPE算法在这些字节序列中寻找模式的方式与在英文ASCII字节中寻找模式的方式完全相同。

### 预分词

在BPE触及你的文本之前，你需要将其分割成块。这可以防止合并算法创建跨越词边界的标记。

GPT-2使用一个正则表达式模式来分割文本：

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

该模式分割缩略词（“don't”变成“don”+“'t”）、带有可选前导空格的词、数字、标点符号和空白。前导空格保留在单词上——所以“the cat”变成[“ the”, “ cat”]，而不是[“the”, “ ”, “cat”]。

Llama使用SentencePiece，它完全跳过了正则表达式。它将原始字节流视为一个长序列，让BPE算法自己找出边界。这更简单，但给BPE更多自由来创建跨词标记。

这个选择很重要。GPT-2的正则表达式防止分词器学习到一个词末尾的“the”和下一个词开头的“the”应该合并。SentencePiece允许这样做，这有时会产生更高效的压缩，但标记的可解释性较差。

### 特殊标记

每个生产级分词器都会为结构标记保留词元 ID：

|  词元  |  用途  |  使用方  |
|-------|---------|---------|
|  `[BOS]` / `<s>`  |  序列开始  |  Llama 3, GPT  |
|  `[EOS]` / `</s>`  |  序列结束  |  所有模型  |
|  `[PAD]`  |  批处理对齐填充  |  BERT, T5  |
|  `[UNK]`  |  未知词元（字节级BPE消除了此需求）  |  BERT, WordPiece  |
|  `<\ | im_start\ | >`  |  聊天消息边界开始  |  ChatGPT, Qwen  |
|  `<\ | im_end\ | >`  |  聊天消息边界结束  |  ChatGPT, Qwen  |
|  `<\ | user\ | >`  |  用户轮次标记  |  Llama 3  |
|  `<\ | assistant\ | >`  |  助手轮次标记  |  Llama 3  |

特殊词元永远不会被BPE拆分。它们在合并算法运行前被精确匹配，替换为固定ID，然后正常对周围文本进行分词。

### 聊天模板（Chat Templates）

这是大多数人感到困惑且大多数实现出错的地方。

当你向聊天模型发送消息时，API会接受一个消息列表：

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

模型看到的不是JSON，而是一个平坦的词元序列。聊天模板使用特殊词元将消息转换为该平坦序列。每个模型的做法都不同：

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

模板用错，模型就会输出垃圾数据。模型是在一种精确格式上训练的。任何偏差——缺少换行、词元顺序错误、多余空格——都会使输入落在训练分布之外。

### 速度

Python对于生产级分词来说太慢了。

tiktoken（OpenAI）使用Rust编写并带有Python绑定。HuggingFace分词器同样使用Rust。SentencePiece使用C++。这些实现相比纯Python达到了10-100倍的速度提升。

从数据上看：为Llama 3预训练处理15万亿个词元，以每秒100万个词元的速度（快速Python）需要174天。而以每秒1亿个词元的速度（Rust）只需1.7天。

你使用Python构建是为了理解算法。在生产环境中，你会使用编译实现，仅通过Python封装进行调用。

```figure
weight-tying
```

## 动手构建

### 第一步：字节级编码（Byte-Level Encoding）

基础步骤。将任意字符串转换为字节序列，将每个字节映射为可打印字符以便显示，并逆转该过程。

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

在多语言文本上测试以查看字节数：

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"hello" 占5字节。"你好" 占6字节（每个字符3字节）。火焰表情符号占4字节。字节级分词器不关心是什么语言。字节就是字节。

### 第二步：使用正则表达式进行预分词（Pre-Tokenizer）

使用GPT-2正则表达式模式将文本分割成块。每个块由BPE独立分词。

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex` 模块支持Unicode属性转义（`\p{L}` 表示字母，`\p{N}` 表示数字）。标准库 `re` 模块不支持，因此我们回退到ASCII字符类。对于生产级的多语言分词器，请安装 `regex`。

尝试一下：

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

前导空格保持附着在单词上。缩写词在撇号处分开。标点符号成为独立块。BPE永远不会跨越这些边界合并词元。

### 第三步：对字节序列进行BPE（BPE on Byte Sequences）

来自第01课的核心算法，但现在独立地对预分词块进行操作。

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### 第4步：特殊令牌处理

特殊令牌需要精确匹配和固定ID。它们完全绕过BPE。

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### 第5步：完整的分词器类

将所有步骤串联起来：标准化、按特殊令牌分割、预分词、BPE合并、映射到ID。

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### 第6步：多语言测试

真正的测试。用英语、中文、表情符号和代码来测试它。

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

每个中文字符产生3个字节。表情符号产生4个字节。这些都不会导致分词器崩溃。也不会产生未知令牌。这就是字节级BPE的强大之处。

## 使用它

### 比较真实的分词器

加载来自Llama 3、GPT-4和Mistral的实际分词器。查看每个分词器如何处理相同的多语言段落。

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

你会看到相同文本的不同令牌计数。拥有128K词汇量的Llama 3在合并常见模式上更为激进。拥有100K词汇量的GPT-4处于中间位置。拥有32K词汇量的Mistral产生更多令牌，但嵌入层更小。

权衡始终相同：更大的词汇量意味着更短的序列但更多的参数。

## 发布

本节提供了一个用于构建和调试生产级分词器的提示。参见`outputs/prompt-tokenizer-builder.md`。

## 练习

1. **简单:** 添加一个`get_token_bytes(id)`方法，显示任何令牌ID的原始字节。用它来检查你最常用的合并令牌实际上代表什么。
2. **中等:** 实现Llama风格的预分词器，根据空白和数字进行分割，但保留前导空格。在相同语料库上将其词汇表与GPT-2正则表达式方法进行比较。
3. **困难:** 添加一个聊天模板方法，该方法接受`get_token_bytes(id)`消息列表，并生成Llama 3聊天格式的正确令牌序列。将其与HuggingFace实现进行测试。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|----------------------|
|  字节级BPE  |  "基于字节工作的分词器"  |  基础词汇表为256个字节值的BPE——处理任何输入都不会出现未知令牌  |
|  预分词  |  "在BPE之前分割"  |  基于正则表达式或规则的分割，防止BPE跨越词边界合并  |
|  NFKC标准化  |  "Unicode清理"  |  规范分解后跟兼容组合——"fi"连字变为"fi"，全角"A"变为"A"  |
|  聊天模板  |  "消息如何变为令牌"  |  将角色/内容消息列表转换为扁平令牌序列的确切格式——特定于模型，必须与训练格式匹配  |
|  特殊令牌  |  "控制令牌"  |  绕过BPE的保留令牌ID——[BOS]、[EOS]、[PAD]、聊天标记——在合并前精确匹配  |
|  繁殖率  |  "每个词的令牌数"  |  输出令牌与输入词的比例——GPT-4中英语为1.3，韩语为2-3，更高意味着上下文浪费  |
|  tiktoken  |  "OpenAI分词器"  |  带Python绑定的Rust BPE实现——比纯Python快10-100倍  |
|  合并表  |  "词汇表"  |  训练期间学习到的字节对合并的有序列表——这就是分词器学到的知识  |

## 延伸阅读

- [OpenAI tiktoken source](https://github.com/openai/tiktoken) -- GPT-3.5/4使用的Rust BPE实现
- [OpenAI tiktoken source](https://github.com/openai/tiktoken) -- 支持BPE、WordPiece、Unigram的Rust分词器库
- [OpenAI tiktoken source](https://github.com/openai/tiktoken) -- 关于128K词汇表和分词器训练的详细信息
- [OpenAI tiktoken source](https://github.com/openai/tiktoken) -- 语言无关的分词
- [OpenAI tiktoken source](https://github.com/openai/tiktoken) -- 原始的字节到Unicode映射
