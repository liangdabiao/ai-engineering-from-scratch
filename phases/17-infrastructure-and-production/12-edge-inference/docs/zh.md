# 边缘推理 — Apple Neural Engine、Qualcomm Hexagon、WebGPU/WebLLM、Jetson

> 边缘推理的核心约束是内存带宽，而非计算能力。移动DRAM带宽为50-90 GB/s；数据中心HBM3可达2-3 TB/s——存在30-50倍的差距。解码是内存受限的，因此这一差距具有决定性。到2026年，局面将分化为四种方向。Apple M4/A18神经引擎峰值达38 TOPS，采用统一内存（无需CPU↔NPU拷贝）。Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon达到45 TOPS。WebGPU + WebLLM在M3 Max上以约41 tok/s运行Llama 3.1 8B（Q4），约为原生性能的70-80%；GitHub星标17.6k，兼容OpenAI API，移动端覆盖率约70-75%。NVIDIA Jetson Orin Nano Super（8GB）可运行Llama 3.2 3B / Phi-3；AGX Orin通过vLLM以约40 tok/s运行gpt-oss-20b；Jetson T4000（JetPack 7.1）性能为AGX Orin的2倍。TensorRT Edge-LLM支持EAGLE-3、NVFP4、分块预填充——在CES 2026上由Bosch、ThunderSoft、MediaTek展示。

**类型:** 学习
**语言:** Python (stdlib, 玩具带宽受限解码模拟器)
**前置条件:** 阶段 17 · 04 (vLLM服务内部), 阶段 17 · 09 (生产量化)
**时间:** 约60分钟

## 学习目标

- 解释为什么移动端LLM推理是内存带宽受限的，而计算是次要的。
- 列举四个边缘目标（Apple ANE, Qualcomm Hexagon, WebGPU/WebLLM, NVIDIA Jetson），并将每个目标匹配到一个用例。
- 指出2026年WebGPU覆盖的差距（Firefox Android正在追赶）以及Safari iOS 26的落地。
- 为每个目标选择一种量化格式（ANE使用Core ML INT4 + FP16, Hexagon使用QNN INT8/INT4, 浏览器使用WebGPU Q4, Jetson Thor使用NVFP4）。

## 问题

一位客户想要一个设备端聊天机器人：语音优先、默认隐私、离线工作。在MacBook Pro M3 Max上，Llama 3.1 8B Q4以约55 tok/s运行——很好。在iPhone 16 Pro上，同一个模型以3 tok/s运行——不好。在搭载Snapdragon 8 Gen 3的中端Android设备上，7 tok/s。在Chrome Android v121+浏览器中通过WebGPU，根据设备不同为4-8 tok/s。

吞吐量的差异并非移植问题。它是带宽差距乘以量化格式再乘以NPU是否可从用户空间访问的结果。2026年的边缘推理是四个不同的问题，对应四种不同的解决方案。

## 核心概念

### 带宽是真正的天花板

解码需要读取每个token的全部权重。一个7B模型采用Q4量化后为3.5 GB。以50 GB/s读取3.5 GB需要70毫秒——理论天花板约为14 tok/s。在90 GB/s（高端移动DRAM）下，天花板提升至约25 tok/s。低于这个数字，任何计算都无济于事。

数据中心HBM3以3 TB/s速度在1.2毫秒内读取同样的3.5 GB——天花板为830 tok/s。同样的模型，同样的权重。不同的内存子系统。

### Apple神经引擎 (M4 / A18)

- 高达38 TOPS。统一内存（CPU和ANE共享同一内存池）——无拷贝开销。
- 通过Core ML + `.mlmodel`编译模型访问，或通过PyTorch使用Metal Performance Shaders (MPS)。
- Llama.cpp Metal后端使用MPS，而非直接使用ANE；原生ANE需要Core ML转换。
- 2026年iOS应用的最佳实践路径：Core ML使用INT4权重 + FP16激活。

### Qualcomm Hexagon (Snapdragon X Elite / 8 Gen 4)

- 高达45 TOPS。与SoC中的CPU和GPU集成，但内存域独立。
- QNN (Qualcomm Neural Network) SDK和AI Hub提供从PyTorch/ONNX的转换。
- 聊天模板、Llama 3.2、Phi-3均作为一流构件在AI Hub上发布。

### Intel / AMD NPU (Lunar Lake, Ryzen AI 300)

- 40-50 TOPS。软件落后于Apple/Qualcomm；OpenVINO正在改进但较为小众。
- 最适合Windows ARM copilot应用；在AMD/Intel桌面平台上原生支持本地优先。

### WebGPU + WebLLM

- 通过WebGPU计算着色器在浏览器中运行模型；无需安装。
- Llama 3.1 8B Q4在M3 Max上约41 tok/s——约为同一后端原生性能的70-80%。
- WebLLM在GitHub上拥有17.6k星标；兼容OpenAI的JS API；Apache 2.0许可。
- 2026年覆盖率：Chrome Android v121+、Safari iOS 26正式版、Firefox Android仍在追赶。总体移动端覆盖率约70-75%。

### NVIDIA Jetson系列

- Orin Nano Super (8GB)：可运行Llama 3.2 3B、Phi-3，token/s表现良好。
- AGX Orin：通过vLLM以约40 tok/s运行gpt-oss-20b。
- Thor / T4000 (JetPack 7.1)：性能为AGX Orin的2倍，支持EAGLE-3和NVFP4。
- TensorRT Edge-LLM (2026)支持EAGLE-3推测解码、NVFP4权重、分块预填充——数据中心优化已移植到边缘。

### 每个目标的量化选择

|  目标  |  格式  |  备注  |
|--------|--------|-------|
|  Apple ANE  |  INT4权重 + FP16激活  |  Core ML转换路径  |
|  Qualcomm Hexagon  |  QNN INT8 / INT4  |  AI Hub转换器  |
|  WebGPU / WebLLM  |  Q4 MLC (q4f16_1)  |  使用 `mlc_llm convert_weight` + 编译的 `.wasm`；不支持GGUF  |
|  Jetson Orin Nano  |  Q4 GGUF 或 TRT-LLM INT4  |  内存受限  |
|  Jetson AGX / Thor  |  NVFP4 + FP8 KV  |  Edge-LLM路径  |

### 边缘的长上下文陷阱

Llama 3.1的128K上下文是数据中心特性。在8GB RAM的手机上，4GB模型 + 2GB KV缓存（32K tokens）+ 操作系统开销 = 内存不足。边缘部署将上下文保持在4K-8K，除非接受激进的KV量化（Q4 KV）。

### 语音是杀手级应用

语音代理对延迟敏感（首token < 500毫秒）。本地推理完全消除网络延迟。结合语音转文字（Whisper Turbo变体在边缘运行），边缘推理成为生产级语音回路。

### 你应该记住的数字

- Apple M4 / A18 ANE: 38 TOPS。
- Qualcomm Hexagon SD X Elite: 45 TOPS。
- WebLLM M3 Max: Llama 3.1 8B Q4约41 tok/s。
- AGX Orin: 通过vLLM运行gpt-oss-20b约40 tok/s。
- 数据中心-边缘带宽差距: 30-50倍。
- WebGPU移动端覆盖率: 约70-75%（Firefox Android滞后）。

## 使用它

`code/main.py` 计算跨边缘目标由带宽限制的数学理论解码吞吐上限。与观察到的基准测试对比，并指出带宽（而非计算）是瓶颈的位置。

## 发布

本课生成 `outputs/skill-edge-target-picker.md`。给定平台（iOS/Android/浏览器/Jetson）、模型以及延迟/内存预算，选择量化格式和转换流水线。

## 练习

1. 运行 `code/main.py`。对于骁龙8 Gen 3（约77 GB/s带宽）上Q4量化的7B模型，计算解码上限。与观察到的6-8 tok/s比较——运行时效率如何？
2. Android上的WebGPU需要Chrome v121+。为旧浏览器设计一个回退方案——通过同一OpenAI兼容API的服务端实现。
3. 你的iOS应用需要4K上下文流式传输。在iPhone 16上，哪种模型/格式组合能让你保持在4 GB活跃内存以下？
4. Jetson AGX Orin以40 tok/s运行gpt-oss-20b。Jetson Nano只能容纳3B模型。如果你的产品同时针对两者，如何统一推理栈？
5. 论证“WebLLM在2026年是否可用于生产环境”。引用覆盖率、性能以及Firefox Android的差距。

## 关键术语

|  术语  |  人们的说法  |  实际含义  |
|------|----------------|------------------------|
|  ANE  |  "苹果神经网络引擎"  |  M系列和A系列中的设备端NPU；统一内存 |
|  Hexagon  |  "高通NPU"  |  骁龙NPU；通过QNN SDK访问 |
|  WebGPU  |  "浏览器GPU"  |  W3C标准化的浏览器GPU API；Chrome/Safari 2026 |
|  WebLLM  |  "浏览器LLM运行时"  |  MLC-LLM项目；Apache 2.0；与OpenAI兼容的JS |
|  Jetson  |  "英伟达边缘设备"  |  Orin Nano / AGX / Thor / T4000系列 |
|  TRT Edge-LLM  |  "边缘TensorRT"  |  2026年TensorRT-LLM的边缘移植版；EAGLE-3 + NVFP4 |
|  统一内存  |  "共享池"  |  CPU和NPU看到同一RAM；无拷贝开销 |
|  带宽受限  |  "内存限制"  |  解码受限于读取权重的字节/秒 |
|  Core ML  |  "苹果转换"  |  用于ANE原生模型的苹果框架 |
|  QNN  |  "高通栈"  |  高通神经网络SDK |

## 延伸阅读

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — 全景与基准测试。
- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — Orin / AGX / Thor。
- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — 2026年边缘移植版发布。
- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — 设计与基准测试。
- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — ANE原生转换。
- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — 面向Hexagon的预转换模型。
