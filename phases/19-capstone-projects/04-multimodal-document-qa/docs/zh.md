# 顶点项目 04 — 多模态文档问答（视觉优先 PDF、表格、图表）

> 2026年的文档问答前沿已从“先OCR再文本”转向“视觉优先的后期交互”。ColPali、ColQwen2.5和ColQwen3-omni将每个PDF页面视为图像，使用多向量后期交互进行嵌入，并让查询直接关注图像块(patch)。在金融10-K文件、科学论文和手写笔记上，这种模式大幅优于先OCR的方法。在10,000页上构建端到端流水线，并与先OCR再文本的方法进行并排对比发布。

**类型：** 顶点项目
**语言：** Python（流水线）、TypeScript（查看器UI）
**前置条件：** 阶段4（计算机视觉）、阶段5（自然语言处理）、阶段7（Transformer）、阶段11（大语言模型工程）、阶段12（多模态）、阶段17（基础设施）
**涉及阶段：** P4 · P5 · P7 · P11 · P12 · P17
**时间：** 30小时

## 问题

企业积压着大量PDF，OCR流水线常常处理不佳：扫描的10-K文件带有旋转表格、公式密集的科学论文、只有作为图像才有意义的图表、手写注释。将其视为文本优先意味着丢失一半的信号。2026年的答案是：对原始页面图像进行后期交互多向量检索。ColPali（Illuin Tech）引入了这一方法，ColQwen2.5-v0.2和ColQwen3-omni提升了准确性。在ViDoRe v3上，视觉优先的检索以显著的优势超越先OCR再文本——在图表、表格和手写内容上差距更大。

权衡在于存储和延迟。ColQwen的嵌入是每页约2048个图像块向量，而不是单个1024维向量。原始存储会急剧膨胀。DocPruner（2026年）实现了50%的剪枝而不带来可衡量的精度损失。你将索引10,000页，测量ViDoRe v3的nDCG@5，在2秒内提供答案，并直接与先OCR再文本的基线进行比较。

## 概念

后期交互意味着每个查询令牌(token)与每个图像块令牌进行评分，并对每个查询令牌的最大得分求和。你可以在不需要单个池化向量的情况下获得细粒度的匹配。多向量索引（Vespa、Qdrant multi-vector或AstraDB）存储每个图像块的嵌入，并在检索时运行MaxSim。

回答器是一个视觉语言模型，它接收查询和检索到的前k个页面图像，并写出带有证据区域（边界框或页面引用）的答案。Qwen3-VL-30B、Gemini 2.5 Pro和InternVL3是2026年前沿的选择。对于方程和科学符号，可以接入OCR后备（Nougat、dots.ocr）作为可选的文本通道。

评估是一个二维矩阵。一个轴：内容类型（纯文本段落、密集表格、柱状/折线图、手写笔记、方程）。另一个轴：检索方法（视觉优先的后期交互 vs 先OCR再文本 vs 混合）。每个单元格获得nDCG@5和答案准确率。报告是交付物。

## 架构

```
PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page
```

## 技术栈

- 页面渲染：PyMuPDF (fitz)，180 DPI，纵向标准化
- 后期交互模型：ColQwen2.5-v0.2或ColQwen3-omni（Hugging Face上的vidore团队）
- 索引：具有多向量字段的Vespa，或Qdrant多向量，或使用MaxSim的AstraDB
- 剪枝：DocPruner 2026策略（保留高方差图像块，50%压缩，精度损失< 0.5%）
- OCR后备（方程/密集表格）：dots.ocr或Nougat
- VLM回答器：自托管的Qwen3-VL-30B或托管的Gemini 2.5 Pro；后备使用InternVL3
- 评估：ViDoRe v3基准，M3DocVQA用于多页推理
- 查看器UI：Next.js 15，带有用于证据区域的画布覆盖层

## 动手构建

1. **摄入。** 遍历包含10,000页PDF的语料库，涵盖10-K文件、科学论文和扫描文档。将每个页面渲染为1536x2048的PNG。持久化 `{doc_id, page_num, image_path}`。

2. **嵌入。** 对每个页面图像运行ColQwen2.5-v0.2。输出形状约为2048个维度为128的图像块嵌入。应用DocPruner保留信号最强的一半。写入Vespa的多向量字段或Qdrant的多向量。

3. **查询。** 对于每个传入的查询，使用查询塔（令牌级嵌入）进行嵌入。针对索引运行MaxSim：对于每个查询令牌，取页面图像块嵌入上的最大点积，求和。返回前k个页面。

4. **综合。** 调用Qwen3-VL-30B，传入查询和前5个页面图像。提示语：“仅使用提供的页面作答。每个论断引用(doc_id, page)并指明区域（图、表、段落）。”

5. **证据区域。** 对答案进行后处理以提取引用的区域。如果VLM输出了边界框（Qwen3-VL会这样做），则在查看器中将其渲染为覆盖层。

6. **OCR后备。** 对于被识别为方程密集的页面（基于图像方差的启发式方法），运行Nougat或dots.ocr，并将OCR文本作为额外通道与图像一起传递。

7. **评估。** 运行ViDoRe v3（检索nDCG@5）和M3DocVQA（多页问答准确率）。同时在同一语料库上使用相同的综合器运行先OCR再文本的流水线。生成一个内容类型×方法的矩阵。

8. **用户界面。** 先做Streamlit原型；然后Next.js 15生产环境查看器，带逐页证据区域覆盖。

## 使用它

```
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4
```

## 发布

`outputs/skill-doc-qa.md` 描述了交付物：一个视觉优先的多模态文档问答系统，针对特定语料库进行调优，并在ViDoRe v3上与先OCR再文本的基线进行评估。

|  权重  |  标准  |  衡量方式  |
|:-:|---|---|
|  25  |  ViDoRe v3 / M3DocVQA准确率  |  与OCR文本基线和已发布排行榜的基准数字  |
|  20  |  证据区域定位  |  被引用区域中实际包含答案范围的比率  |
|  20  |  存储和延迟工程  |  DocPruner压缩率、索引p95、答案p95  |
|  20  |  多页推理  |  在人工标注的100题多页数据集上的准确率  |
|  15  |  源检查用户体验  |  查看器清晰度、覆盖层保真度、并排比较工具  |
|  **100**  |   |   |

## 练习

1. 在同一语料库上测量ColQwen2.5-v0.2与ColQwen3-omni。哪些页面其中一个正确而另一个错误？在索引中添加“内容类别”标签以按类型路由。

2. 激进地剪枝嵌入（75%、90%）。找到压缩悬崖：即ViDoRe nDCG@5低于OCR基线的点。

3. 构建混合方法：并行运行先OCR再文本和ColQwen，使用RRF融合，再用交叉编码器重排序。混合方法是否胜过单独任何一种？它在哪些方面帮助最大？

4. 将Qwen3-VL-30B替换为更小的VLM（Qwen2.5-VL-7B）。测量每美元准确率曲线。

5. 添加手写笔记支持。渲染手写语料库，用ColQwen嵌入，测量检索。与手写OCR流水线进行比较。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|-----------------|------------------------|
|  后期交互  |  “ColPali风格检索”  |  查询令牌独立地对页面图像块评分；MaxSim聚合  |
|  多向量  |  “每图像块嵌入”  |  每个文档有多个向量，而不是一个池化向量  |
| MaxSim  |  “延迟交互评分”  |  对每个查询词元，取与文档向量的最大相似度并求和 |
| DocPruner  |  “块压缩”  |  2026年提出的剪枝方法，保留50%的块且精度损失可忽略 |
| ViDoRe v3  |  “文档检索基准”  |  2026年用于衡量视觉文档检索的标准 |
| 证据区域  |  “引用的边界框”  |  源页面上定位答案范围的边界框 |
| OCR后备  |  “公式通道”  |  与视觉联合用于公式或表格密集页面的文本管线 |

## 延伸阅读

-   [ColPali (Illuin Tech) repository](https://github.com/illuin-tech/colpali) — 参考延迟交互文档检索
- [ColPali (Illuin Tech) repository](https://github.com/illuin-tech/colpali) — 基础方法论文
- [ColPali (Illuin Tech) repository](https://github.com/illuin-tech/colpali) — 生产就绪的检查点
- [ColPali (Illuin Tech) repository](https://github.com/illuin-tech/colpali) — 多页多模态RAG基线
- [ColPali (Illuin Tech) repository](https://github.com/illuin-tech/colpali) — 参考服务栈
- [ColPali (Illuin Tech) repository](https://github.com/illuin-tech/colpali) — 替代索引
- [ColPali (Illuin Tech) repository](https://github.com/illuin-tech/colpali) — 替代托管索引
- [ColPali (Illuin Tech) repository](https://github.com/illuin-tech/colpali) — 支持公式的OCR后备
