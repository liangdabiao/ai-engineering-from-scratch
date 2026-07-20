# Whisper — 架构与微调

> Whisper 是一个30秒窗口的Transformer编码器-解码器，在68万小时的多语言弱监督音频-文本对上训练。单一架构，多任务，在99种语言上鲁棒。2026年的参考ASR。

**类型:** 构建
**语言:** Python
**前置条件:** 阶段6 · 04 (ASR), 阶段5 · 10 (注意力), 阶段7 · 05 (完整Transformer)
**时间:** ~75分钟

## 问题

Whisper，由OpenAI于2022年9月发布，是第一款作为商品提供的ASR模型：粘贴音频，获取文本，支持99种语言，对噪声鲁棒，可在笔记本上运行。到2024年，OpenAI已发布了Large-v3和Turbo变体；到2026年，Whisper成为从播客转录到语音助手再到YouTube字幕等所有任务的默认基线。

但Whisper不是一个可以永远当作黑盒的流水线。领域迁移会使其失效——技术术语、说话人口音、专有名词、短片段、静音。你需要了解：

1. 它内部到底是什么。
2. 如何正确处理分块、流式或长形式音频。
3. 何时微调以及如何微调。

## 核心概念

![Whisper encoder-decoder, tasks, chunked inference, fine-tune](../assets/whisper.svg)

**架构。** 标准Transformer编码器-解码器。

- 输入：30秒对数梅尔频谱图，80个梅尔滤波器，10毫秒跳步 → 3000帧。较短的片段补零，较长的片段分块。
- 编码器：卷积下采样（步长2）+ `N` 个Transformer块。对于Large-v3：32层，1280维，20个头。
- 解码器：`N` 个Transformer块，带有因果自注意力和与编码器输出的交叉注意力。与编码器大小相同。
- 输出：基于51865个token词表的BPE token。

Large-v3有15.5亿参数。Turbo使用4层解码器（从32层减少），将延迟降低8倍，而WER损失小于1%。

**提示格式。** Whisper是一个多任务模型，由解码器提示中的特殊token引导：

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` — 语言标签；强制翻译与转录行为。
- `<|en|>` 或 `<|transcribe|>` — 从任意语言输入翻译为英语输出，或逐字转录。
- `<|en|>` — 跳过词级时间戳（更快）。

提示使得一个模型能够完成多个任务。将 `<|en|>` 改为 `<|fr|>`，它就会转录法语。

**30秒窗口。** 一切都固定在30秒。较长的片段需要分块；较短的片段补零。窗口并非原生流式——这就是WhisperX、Whisper-Streaming和faster-whisper存在的原因。

**对数梅尔归一化。** `(log_mel - mean) / std` 其中统计数据来自Whisper自己的训练语料。你*必须*使用Whisper的预处理（`whisper.audio.log_mel_spectrogram`），而不是 `librosa.feature.melspectrogram`。

### 2026年的变体

|  变体  |  参数  |  延迟（A100）  |  WER（LibriSpeech-clean）  |
|---------|--------|----------------|------------------------|
|  Tiny  |  39M  |  1×实时  |  5.4%  |
|  Base  |  74M  |  1×  |  4.1%  |
|  Small  |  244M  |  1×  |  3.0%  |
|  Medium  |  769M  |  1×  |  2.7%  |
|  Large-v3  |  1.55B  |  2×  |  1.8%  |
|  Large-v3-turbo  |  809M  |  8×  |  1.58%  |
|  Whisper-Streaming（2024）  |  1.55B  |  流式  |  2.0%  |

### 微调

2026年的标准工作流：

1. 收集10–100小时目标领域音频及其对齐的转录文本。
2. 使用 `transformers.Seq2SeqTrainer` 和 `generate_with_loss` 回调运行。
3. 参数高效：对注意力层的 `transformers.Seq2SeqTrainer`、`generate_with_loss`、`q_proj` 应用LoRA，可将GPU内存减少4倍，WER损失低于0.3。
4. 如果少于10小时数据，冻结编码器。仅调整解码器。
5. 使用Whisper自己的分词器和提示格式；切勿更换分词器。

社区结果：在20小时医学听写数据上微调Medium，使医学词汇的WER从12%降至4.5%。在4小时冰岛语数据上微调Turbo，使WER从18%降至6%。

## 动手构建

### 第一步：开箱即用Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # prevents runaway repetition
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

你应该始终覆盖的关键默认值：`temperature=0.0`（采样默认从0.0 → 0.2 → 0.4 … 回退链），`condition_on_previous_text=False`（防止级联幻觉问题），以及 `no_speech_threshold=0.6`（静音检测）。

### 第二步：分块长形式

```python
# whisperx is the 2026 reference for long-form with word-level timestamps
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX 增加了 (1) Silero VAD 门控、(2) 通过 wav2vec 2.0 进行的词级对齐、(3) 通过 `pyannote.audio` 进行的说话人分离。这是 2026 年生产环境转录的主力工具。

### 步骤 3：使用 LoRA 进行微调

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M trainable / 809M total
```

然后使用标准的 Trainer 循环。每 1000 步保存检查点，在保留集上评估 WER。

### 步骤 4：检查每一层学到什么

```python
# Grab cross-attention weights during decode to see what the decoder attends to.
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

用热力图可视化——你会看到解码器步骤扫描编码器帧时出现对角线对齐。这条对角线就是 Whisper 对词时间戳的理解。

## 使用它

2026年技术栈：

|  情况  |  选择  |
|-----------|------|
|  通用英语，离线  |  通过 `whisperx` 使用 Large-v3-turbo  |
|  移动端 / 边缘设备  |  Whisper-Tiny 量化版 (int8) 或 Moonshine  |
|  多语言长音频  |  通过 `whisperx` 使用 Large-v3 并加上说话人分离  |
|  低资源语言  |  使用 LoRA 微调 Medium 或 Turbo  |
|  流式处理（2 秒延迟） |  Whisper-Streaming 或 Parakeet-TDT  |
|  词级时间戳  |  WhisperX（通过 wav2vec 2.0 强制对齐） |

`faster-whisper`（CTranslate2 后端）是 2026 年最快的 CPU+GPU 推理运行时——比原始版本快 4 倍，输出相同。

## 2026年仍存在的陷阱

- **静音片段出现幻觉文本。** Whisper 在字幕上训练，会产生“感谢观看！”、“订阅！”、歌词等内容。调用前务必使用 VAD 门控。
- **`condition_on_previous_text` 级联问题。** 一次幻觉会污染后续窗口。除非需要跨片段流畅性，否则设置 `False`。
- **短片段填充。** 一个 2 秒的片段填充到 30 秒后，可能在尾部的静音中产生幻觉。使用 `condition_on_previous_text` 或 VAD 门控。
- **错误的梅尔频谱统计。** 使用 librosa 的梅尔频谱而非 Whisper 的，会产生近乎随机的输出。使用 `condition_on_previous_text`。

## 发布

保存为 `outputs/skill-whisper-tuner.md`。为给定领域设计 Whisper 微调或推理流程。

## 练习

1. **简单版。** 运行 `code/main.py`。它会对 Whisper 风格的提示进行分词，计算解码后的形状预算，并打印出一个 10 分钟片段的分块调度。
2. **中等版。** 安装 `code/main.py`，转录一个 10 分钟的播客，并与人工转录的 WER 进行比较。尝试 `faster-whisper` 对比强制 `language="auto"`。
3. **困难版。** 使用 Hugging Face 的 `code/main.py`，选择一种 Whisper 处理不佳的语言（如乌尔都语），使用 LoRA 对 Medium 进行 2 小时的 2 轮微调，并报告 WER 差异。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|-----------------------|
|  30 秒窗口  |  Whisper 的限制  |  硬性输入上限；将长音频分块。  |
|  SOT  |  转录起始  |  `<\ | startoftranscript\ | >` 启动解码器提示。  |
|  时间戳令牌  |  时间对齐  |  每 0.02 秒偏移是 51k 词汇表中的一个特殊令牌。  |
|  Turbo  |  快速变体  |  4 层解码器，速度快 8 倍，WER 退化 <1%。  |
|  WhisperX  |  长音频封装  |  VAD + Whisper + wav2vec 对齐 + 说话人分离。  |
|  LoRA 微调  |  高效调优  |  向注意力机制添加低秩适配器；训练约 0.3% 的参数。  |
|  幻觉  |  无声错误  |  Whisper 从噪声/静默中生成流利英语。  |

## 延伸阅读

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356)——原始架构和训练方案。
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363)——4层解码器，8倍加速。
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747)——长音频、词级对齐、说话人分离。
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper)——基于CTranslate2，快4倍。
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper)——标准LoRA/全微调指南。
