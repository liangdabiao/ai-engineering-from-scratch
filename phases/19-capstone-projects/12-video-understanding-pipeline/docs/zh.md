# 顶点项目 12 —— 视频理解流水线（场景、问答、搜索）

> Twelve Labs 将 Marengo + Pegasus 产品化。VideoDB 推出了面向视频的 CRUD API。AI2 的 Molmo 2 发布了开源 VLM 检查点。Gemini 长上下文原生支持处理数小时的视频。TimeLens-100K 在大规模上定义了时间定位(Temporal Grounding)。2026 年的流程已经确定：场景分割、每场景描述(Per-scene Caption) + 嵌入(Embedding)、转录对齐(Transcript Alignment)、多向量索引(Multi-vector Index)，以及返回 (开始, 结束) 时间戳加帧预览(Frame Previews)的查询。顶点项目是摄入 100 小时视频、在公共基准上测试、并测量在计数和动作类问题上的幻觉(Hallucination)。

**类型：** 顶点项目
**语言：** Python（管道）、TypeScript（UI）
**先决条件：** 阶段 4（计算机视觉）、阶段 6（语音）、阶段 7（Transformer）、阶段 11（大语言模型工程）、阶段 12（多模态）、阶段 17（基础设施）
**涉及阶段：** P4 · P6 · P7 · P11 · P12 · P17
**时间：** 30 小时

## 问题

长视频问答(Long-form video QA)是 2026 年规模下最消耗带宽的多模态问题。Gemini 2.5 Pro 可以原生读取 2 小时的视频，但将 100 小时视频摄入为可查询的语料库仍然需要场景级索引。生产形态结合了场景分割(Scene Segmentation)（TransNetV2 或 PySceneDetect）、使用 VLM 进行每场景描述(Per-scene Captioning)（Gemini 2.5、Qwen3-VL-Max 或 Molmo 2）、转录对齐(Transcript Alignment)（带有单词时间戳的 Whisper-v3-turbo），以及存储描述、帧嵌入(Frame Embedding)和转录并排的多向量索引(Multi-vector Index)。查询管道返回 (开始, 结束) 时间戳加帧预览。

基准是公开的（ActivityNet-QA、NeXT-GQA）加上自建的 100 查询定制集。计数和动作类型问题上的幻觉是已知的困难失败类别；顶点项目明确测量它。

## 概念

摄入时三个管道并行运行。**场景分割**将视频分割成场景。**VLM 描述**为每个场景生成描述和关键帧的帧嵌入。**ASR 对齐**产生单词级别的时间戳。三个流通过 (场景 ID, 时间范围) 连接。每个场景在多向量索引中获得三种向量类型（Qdrant）：描述嵌入、关键帧嵌入、转录嵌入。

查询时，自然语言问题对所有三个向量发起查询；结果通过 RRF 合并；一个时间定位适配器（TimeLens 风格）在顶级场景内细化 (开始, 结束) 窗口。VLM 合成器（Gemini 2.5 Pro 或 Qwen3-VL-Max）接收查询 + 顶级场景 + 裁剪帧，并以带引用的时间戳和帧预览回答。

幻觉测量很重要。计数（“多少人进入房间？”）和动作类型（“厨师在搅拌前倒水吗？”）问题众所周知不可靠。与描述性问题分开报告准确率。

## 架构

```
video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations
```

## 技术栈

- 场景分割：TransNetV2（2024-26 年最先进）或 PySceneDetect
- ASR：通过 faster-whisper 使用单词时间戳的 Whisper-v3-turbo
- VLM 描述器 + 回答器：Gemini 2.5 Pro 或 Qwen3-VL-Max 或 Molmo 2
- 时间定位：TimeLens-100K 训练的适配器或 VideoITG
- 索引：支持多向量（描述/帧/转录）的 Qdrant
- UI：Next.js 15 配合 HTML5 视频播放器和场景缩略图
- 评估：ActivityNet-QA、NeXT-GQA、自定义 100 问题手工标注集
- 幻觉基准：手工标注的计数和动作类型子集

## 动手构建

1. **摄入遍历器。** 接受 YouTube URL 或本地 MP4。必要时降采样到 720p。持久化 `{video_id, file_path}`。

2. **场景分割。** 运行 TransNetV2 或 PySceneDetect 生成 `[{scene_id, start_ms, end_ms, keyframe_path}]`。目标 100 小时：约 6000-8000 个场景。

3. **ASR 通道。** 在音频上运行 Whisper-v3-turbo；导出单词级时间戳；拆分为每场景转录片段。

4. **VLM 描述。** 每个场景，使用关键帧和简短描述模板调用 Gemini 2.5 Pro（或 Qwen3-VL-Max）。生成描述 + 帧嵌入。

5. **多向量索引。** 具有三个命名向量的 Qdrant 集合。载荷：`{video_id, scene_id, start_ms, end_ms, keyframe_url}`。

6. **查询。** 自然语言问题发起三个密集查询；通过倒数排名融合合并；top-k=5 个场景。

7. **时间定位。** 在顶级场景上运行 TimeLens 风格适配器，以在场景内细化 (开始, 结束) 窗口。

8. **VLM 合成。** 使用查询 + 前 3 场景片段（作为图像或短片）+ 转录调用 Gemini 2.5 Pro。要求 `(video_id, start_ms, end_ms)` 引用。

9. **评估。** 运行 ActivityNet-QA 和 NeXT-GQA。构建 100 查询定制集。报告总体准确率 + 每个类别（计数、动作、描述）的细分。

## 使用它

```
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## 发布

`outputs/skill-video-qa.md` 是可交付成果。给定 YouTube URL 或上传的视频，管道索引场景并回答带有时间戳引用的问题。

|  权重  |  标准  |  衡量方式  |
|:-:|---|---|
|  25  |  时间定位 IoU  |  保留测试集上的交并比  |
|  20  |  QA 准确率  |  NeXT-GQA 和定制 100 查询  |
|  20  |  摄入吞吐量  |  每美元处理的视频小时数  |
|  20  |  UI 和引用用户体验  |  时间戳链接、缩略图条、跳转到帧  |
|  15  |  幻觉率  |  计数和动作类型准确率分开  |
|  **100**  |   |   |

## 练习

1. 在描述通道中用 Qwen3-VL-Max 替换 Gemini 2.5 Pro。在人工评分的 50 场景样本上报告描述质量差异。

2. 将每场景帧嵌入减少为一个池化向量而不是多向量。测量检索性能下降。

3. 构建一个“严格计数”模式：合成器提取每个计数实例及其时间戳，用户点击验证。测量用户验证是否减少幻觉。

4. 基准摄入成本：三种 VLM 选择下每美元的视频小时数。选择最佳点。

5. 添加说话人分离转录：在音频上运行 pyannote 说话人分离并嵌入每说话人转录。演示“Alice 对 X 说了什么？”查询。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
|  场景分割  |  "镜头检测"  |  在镜头边界处将视频切割为场景  |
| 多向量索引(Multi-vector index)  |  "字幕+关键帧+转录"  |  每个表示使用命名向量的Qdrant集合(Qdrant collection with named vectors per representation)  |
| 时间定位(Temporal grounding)  |  "确切发生时间"  |  精细化查询答案的(开始, 结束)窗口  |
| 关键帧嵌入(Frame embedding)  |  "视觉表示"  |  关键帧的向量嵌入；用于场景视觉相似度  |
| RRF融合(RRF fusion)  |  "倒数排名融合"  |  跨多个排序列表的合并策略；经典的混合检索技巧  |
| 计数幻觉(Counting hallucination)  |  "数错"  |  VLM在"多少个X"问题上的已知失败模式  |
| ActivityNet-QA  |  "视频问答基准"  |  长视频问答准确率基准  |

## 延伸阅读

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 开源VLM检查点
- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 大规模时间定位
- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 托管参考
- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 视频CRUD API参考
- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 商业参考
- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 场景分割模型
- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 经典开源替代方案
- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 参考评估基准
