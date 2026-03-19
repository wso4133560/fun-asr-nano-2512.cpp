## Context

当前项目 `/home/tanglin/workspace2/fun-asr-nano-2512.cpp` 使用 ONNX Runtime + llama.cpp 混合管线。目标是完全替换为纯 GGML C++ CUDA 实现，参照 `/home/tanglin/workspace2/voxtral.cpp-cuda` 已验证的代码结构。

两个 GGUF 模型已就绪：
- `./models/Fun-ASR-Nano-2512-llm-Q5_K_M.gguf`（311 tensors, Qwen3-0.6B）
- `./models/Fun-ASR-Nano-2512-audio-mmproj-mtmd-F16.gguf`（950 tensors, SenseVoice + Adaptor）

## Goals / Non-Goals

**Goals:**
- 单一 GGML CUDA 后端，无外部 ML 依赖
- 完全对齐 voxtral.cpp-cuda 代码结构（include/, src/, ggml/, CMakeLists.txt）
- 正确实现 SenseVoice SANM + Qwen3 GQA + QK-Norm 推理
- 构建命令：`cmake -DGGML_CUDA=ON && cmake --build build -j$(nproc)`

**Non-Goals:**
- 流式推理、实时转写（可后续扩展）
- 量化转换工具（模型已量化）
- 多 GPU 支持
- Python 绑定

## Decisions

### D1: 文件布局完全对齐 voxtral.cpp-cuda
- `include/fun_asr.h`：公开 API（`fun_asr_model`, `fun_asr_ctx` 结构体 + `fun_asr_init/transcribe/free`）
- `src/fun_asr.cpp`：核心实现（GGUF 加载、Fbank、编码器、投影器、解码器）
- `src/main.cpp`：CLI（`--model-dir ./models --wav input.wav [--device cuda|cpu]`）
- `ggml/`：从 voxtral.cpp-cuda 复制（`cp -r /home/tanglin/workspace2/voxtral.cpp-cuda/ggml ./`）

**理由**：复用已验证的 CMake 配置、backend init 模式和 GGUF 加载代码，降低初始实现风险。

### D2: 单 ggml_context 持久分配策略
在 `fun_asr_ctx` 初始化时一次性分配所有持久 buffer：
- LLM 权重 buffer（device）
- 音频编码器权重 buffer（device）
- KV cache `[1024, 2048, 28]` × 2（K+V，device）
- encoder_output `[512, MAX_ENC_SEQ]`（device）
- audio_tokens `[1024, MAX_ENC_SEQ]`（device）

推理时只分配临时 `ggml_context`（`no_alloc=true`）用于图构建，`ggml_backend_sched` 管理调度。

**理由**：避免推理时动态分配、防止 CUDA OOM、对齐 voxtral.cpp-cuda 模式。

### D3: 分离 LLM 和 mmproj 的 GGUF 加载
- `load_llm_weights()` → 填充 `fun_asr_llm` 结构体（311 tensors）
- `load_mmproj_weights()` → 填充 `fun_asr_audio` 结构体（950 tensors）
- 两个文件可使用同一个 `ggml_backend`，共享一个 CUDA 设备

**张量名称精确对应（无宏生成）**：直接用字符串常量 `"a.blk.0.self_attn.linear_qkv.weight"` 等查找，避免运行时命名错误。

### D4: 编码器图构建策略
编码器 71 个块一次性构建完整 GGML 图（静态图）：
- 仅对变长输入（T'）在 `MAX_ENC_SEQ` 内重建图（每次推理）
- 不缓存编码器图（音频长度变化频繁，缓存收益低）
- 使用单一 `ggml_backend_sched`（encoder + decoder 共用）

### D5: KV cache 使用 shift-left eviction（非 ring buffer）
- shape：`[1024, 2048, 28]`（kv_dim × window × layers），K 和 V 各一份
- 写入：`ggml_set_rows(k_cache, k_row, kv_used)`，然后 `kv_used++`
- 溢出：`memmove`（CPU 侧执行）rows `[1..2047]` → `[0..2046]`，写第 2047 行
- 读取：`ggml_view_2d`（slice at row=0 to kv_used）传给 `ggml_flash_attn_ext`

**理由**：对齐 voxtral.cpp-cuda 的 `kv_cache_shift_left` 模式，避免 ring buffer 引入的 attention mask 复杂性。

### D6: Qwen3 QK-Norm 实现方式
```
q = ggml_mul_mat(attn_q_w, h_norm)         // [2048, T]
q = ggml_reshape_3d(q, 128, 16, T)          // per-head
q = ggml_rms_norm(q, 1e-6)                  // norm over dim-0 (128)
q = ggml_mul(q, attn_q_norm_w)             // scale [128]
q = ggml_reshape_2d(q, 2048, T)            // flatten back
```
对 K 同理（8 heads，[1024, T]）。之后再应用 RoPE。

### D7: FSMN 在 SANM 中的精确位置
```
x_norm = layer_norm(x, norm1_w, norm1_b)
qkv    = linear_qkv_w * x_norm + linear_qkv_b  // [1536, T]
// split Q, K, V
mha    = flash_attn(Q, K, V)                    // [512, T]
fsmn_m = ggml_conv_1d(x_norm, fsmn_w, 1, 5)    // [512, T] 对 norm 后的 x 做 depthwise conv
out    = linear_out_w * (mha + fsmn_m) + linear_out_b
h_attn = x + out                                 // 第一个残差
```

### D8: Prompt 模板（字符串常量，硬编码）
```cpp
constexpr const char* PROMPT_SYSTEM  = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
constexpr const char* PROMPT_PREFIX  = "<|im_start|>user\n<|startofspeech|>";
constexpr const char* PROMPT_SUFFIX  = "<|endofspeech|>\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
```
三段分别 tokenize，拼接 embedding 序列，audio tokens 直接插入中间（不走 token_embd 查表）。

## Risks / Trade-offs

| 风险 | 缓解 |
|------|------|
| FSMN 公式未从 FunASR 原始代码验证 | 实现后与 Python FunASR 参考输出逐帧对比；若偏差大则调整 FSMN 组合位置 |
| MLP projector 激活函数（SiLU，用户推荐验证）未 100% 确认 | 编译时宏 `FUN_ASR_PROJ_ACT_RELU`/`SILU` 切换；默认 SiLU |
| ggml_flash_attn_ext 对 GQA 的支持（16Q/8KV）取决于 ggml 版本 | 使用从 voxtral.cpp-cuda 复制的 ggml 子目录，已验证 GQA 支持 |
| QK-Norm reshape 在 CUDA 后端的内存布局 | 必须在 reshape 前后插入 `ggml_cont()` 确保连续内存 |
| 音频编码器 GGML 图节点数量大（71 块 × ~10 ops/块 = ~710 节点） | 使用 `ggml_new_graph_custom(gctx, 4096, false)` 预分配足够图容量 |

## Migration Plan

1. 复制 ggml 子目录：`cp -r /home/tanglin/workspace2/voxtral.cpp-cuda/ggml ./`
2. 实现 `include/fun_asr.h` + `src/fun_asr.cpp`（按 spec 顺序：加载→Fbank→编码器→投影→解码）
3. 实现 `src/main.cpp`（CLI）
4. 编写 `CMakeLists.txt`
5. 构建：`cmake -B build -DGGML_CUDA=ON && cmake --build build -j$(nproc)`
6. 测试：`./build/fun-asr-nano --model-dir ./models --wav test.wav`
7. 参考对比：与 FunASR Python 参考输出（同一 WAV）比较转写文本

## Open Questions

所有主要歧义已在 research 和 plan 阶段解决。唯一待运行时验证的项目：
- MLP projector 激活函数（SiLU 已选定，对比验证后如不符可切换为 ReLU）
- FSMN 公式（`norm1(x)` 作为 conv 输入，输出与 MHA 相加后过 linear_out）
