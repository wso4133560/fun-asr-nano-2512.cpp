## Why

现有项目依赖混合 ONNX Runtime + llama.cpp 管线运行 Fun-ASR-Nano-2512，引入两套异构依赖、分离的 CUDA 调度器，维护成本高且无法统一优化。改用纯 GGML C++ CUDA 后端可消除所有外部依赖，实现单一后端统一调度、量化推理、零拷贝 KV cache，代码结构完全对齐已验证的 voxtral.cpp-cuda 参考实现。

## What Changes

- **BREAKING**: 删除全部现有代码（ONNX + llama.cpp 管线），以纯 GGML C++ 实现替换
- 新增 `include/fun_asr.h`：公开 API（model/context 结构体 + 推理函数）
- 新增 `src/fun_asr.cpp`：核心实现（GGUF 加载、WavFrontend、SenseVoice 编码器、MLP 投影、Adaptor Transformer、Qwen3 解码器、BPE tokenizer）
- 新增 `src/main.cpp`：CLI 入口（`--model-dir`, `--wav`, `--device`）
- 新增 `CMakeLists.txt`：C++17, add_subdirectory(ggml), GGML_CUDA option
- 复制 `ggml/` 子目录（来自 `/home/tanglin/workspace2/voxtral.cpp-cuda/ggml/`）
- 模型文件已存在于 `./models/`，无需下载

## Capabilities

### New Capabilities

- `gguf-loader`: 从两个 GGUF 文件加载所有权重到 GGML CUDA backend buffer
- `wav-frontend`: WAV → 80-mel Hamming Fbank → LFR(m=7,n=6) → [T',560] 特征提取
- `sensevoice-encoder`: SenseVoiceEncoderSmall SANM 编码器前向传播（71块，CUDA 执行）
- `audio-projector`: MLP projector + 2层 Adaptor Transformer，512→1024 投影
- `qwen3-decoder`: Qwen3-0.6B 自回归解码（GQA + QK-Norm + RoPE + SwiGLU + KV cache）
- `bpe-tokenizer`: GPT-2 BPE tokenizer 解码（从 GGUF metadata 加载词表）

### Modified Capabilities

<!-- 无现有 spec，全部为新增 -->

## Impact

- 完全替换 `/home/tanglin/workspace2/fun-asr-nano-2512.cpp/` 下所有源码
- 依赖从 ONNX Runtime + llama.cpp 变为纯 ggml（来自 `/home/tanglin/workspace2/voxtral.cpp-cuda/ggml/`）
- 构建系统从 CMake FetchContent(llama.cpp) 变为 add_subdirectory(ggml)
- 运行时依赖：libcuda, libcublas（通过 GGML CUDA 后端）
