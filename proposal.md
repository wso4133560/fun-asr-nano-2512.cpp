# Proposal: Fun-ASR-Nano-2512 Pure GGML/CUDA C++ Runtime

## 1. 需求陈述

将 `/home/tanglin/workspace2/fun-asr-nano-2512.cpp` 项目从现有的混合 ONNX + llama.cpp 管线**完全替换**为纯粹的 C++ GGML CUDA 后端实现。

**目标模型文件（来自 HuggingFace wso4133560freewind/Fun-ASR-Nano-2512-gguf）：**
- `Fun-ASR-Nano-2512-llm-Q5_K_M.gguf` — LLM 解码器（量化权重）
- `Fun-ASR-Nano-2512-audio-mmproj-mtmd-F16.gguf` — 音频编码器 + 多模态投影层（F16 权重）

**代码结构参考：** `/home/tanglin/workspace2/voxtral.cpp-cuda`（纯 GGML 实现的 ASR runtime，已在 CUDA 上验证）

---

## 2. 约束集合（Constraint Sets）

### 2.1 硬约束（不可违反）

| 约束 | 来源 |
|------|------|
| 仅使用 GGML C++ API，**禁止**引入 llama.cpp 高层 API 或 ONNX Runtime | 需求 |
| CUDA 后端通过 `ggml_backend_cuda_init(0)` 初始化，fallback 到 CPU | voxtral.cpp 模式 |
| GGUF 加载使用 `gguf_init_from_file` + 手动 `fseek/fread` 逐张量加载 | voxtral.cpp 模式 |
| `ggml_context` 内存在推理前一次性分配，不允许动态扩展 | ggml API |
| GPU 结果读取前必须调用 `ggml_backend_synchronize()` | ggml API |
| 每次新图前必须调用 `ggml_backend_sched_reset()` | ggml API |
| LLM 架构、音频编码器维度**必须从 GGUF 元数据动态读取**，禁止硬编码（模型文件尚未下载，架构待确认） | 用户确认 |
| 项目结构对齐 voxtral.cpp-cuda：`include/`, `src/`, `ggml/`（submodule）, `CMakeLists.txt` | 用户确认 |
| 完全替换现有代码，不保留任何 ONNX/llama.cpp 依赖 | 用户确认 |

### 2.2 软约束（代码规范）

- C++17，零冗余注释（非必要不写）
- 精简高效：不引入用不到的抽象层
- 张量命名方案对齐 GGUF 元数据中的实际键名（由 `gguf_dump` 决定）
- 编译选项：`-DGGML_CUDA=ON`，支持 `cmake -DGGML_CUDA=ON`

### 2.3 开放约束（待 GGUF 元数据确认后填写）

- LLM backbone 架构（Qwen2.5? MiniCPM? → 决定 decoder layer 实现）
- 音频编码器架构（Whisper-style? Paraformer Conformer? → 决定 encoder forward pass）
- 是否保留 CTC hint 机制（影响 prompt 构建逻辑）
- 词表大小、层数、注意力头数、KV 头数等超参数

---

## 3. 实现分三阶段（因架构未知）

### Phase 0：模型下载 + 元数据探查（前置条件）

**目标**：在编写任何推理代码前，先获得 GGUF 元数据以确定架构参数。

**步骤 0.1**：下载模型
```bash
# 使用 huggingface-cli 或 wget
huggingface-cli download wso4133560freewind/Fun-ASR-Nano-2512-gguf \
  Fun-ASR-Nano-2512-llm-Q5_K_M.gguf \
  Fun-ASR-Nano-2512-audio-mmproj-mtmd-F16.gguf \
  --local-dir ./models/
```

**步骤 0.2**：使用 `gguf-dump`（或自写 C 工具）读取元数据
```bash
# 若 gguf-dump 可用：
python3 -c "
import struct, sys
# 解析 GGUF header + kv metadata + tensor list
# 输出：architecture, n_layer, n_head, n_kv_head, d_model, vocab_size
" ./models/Fun-ASR-Nano-2512-llm-Q5_K_M.gguf

# 对 mmproj 文件同样操作：
# 确认：audio encoder 层数/维度, projector MLP 结构, 张量名称列表
```

**交付物**：填满所有「开放约束」，形成确定的架构参数表（见附录 A）。

---

### Phase 1：项目骨架（参照 voxtral.cpp-cuda 结构）

**文件结构：**
```
fun-asr-nano-2512.cpp/
├── CMakeLists.txt              # C++17, GGML_CUDA=ON, add_subdirectory(ggml)
├── ggml/                       # ggml 库（git submodule 或 vendored copy from voxtral.cpp-cuda）
├── include/
│   └── fun_asr.h               # 公开 API：model/context 结构体声明 + 推理函数签名
├── src/
│   ├── fun_asr.cpp             # 核心实现：GGUF 加载、编码器、解码器、推理循环
│   └── main.cpp                # CLI 入口：--model-dir --wav --device
├── models/                     # （gitignore）存放 .gguf 文件
└── scripts/
    └── download_model.sh       # HuggingFace 下载脚本
```

**`CMakeLists.txt` 模式：**（直接复用 voxtral.cpp-cuda 的 CMake 骨架）
- `cmake_minimum_required(VERSION 3.16)`
- `GGML_CUDA` option → `add_subdirectory(ggml)`
- `add_library(fun_asr_lib src/fun_asr.cpp)`
- `add_executable(fun-asr-nano src/main.cpp)`

---

### Phase 2：核心推理实现（架构确认后执行）

**模块划分：**

#### 2.1 GGUF 加载模块（`fun_asr.cpp`）

参照 voxtral.cpp-cuda 的加载模式：
```
gguf_init_from_file(path, {.no_alloc=true, .ctx=&ctx_meta})
→ 遍历张量列表，snprintf 拼接张量名
→ ggml_backend_alloc_ctx_tensors(ctx_meta, backend)
→ fseek/fread 逐张量填充数据
→ ggml_backend_tensor_set()
```

分两次加载：
- `load_llm(path)` → 填充 LLM decoder 权重（token_embd, 每层 attn/ffn 权重）
- `load_mmproj(path)` → 填充音频编码器权重（encoder layers, projector MLP）

#### 2.2 音频预处理（Fbank 特征提取）

从 voxtral.cpp-cuda 直接复用（或按 Fun-ASR 参数调整）：
- 采样率：16kHz，梅尔滤波器组：80 bins（待元数据确认）
- 帧长：25ms/10ms，FFT：512 点，预加重：0.97
- 对齐 Fun-ASR-Nano 训练时的参数（从 mmproj GGUF 元数据读取）

#### 2.3 音频编码器前向传播

**架构确认后填写具体实现。**预期两种情况：
- **Whisper-style**：Conv1D stem (2层) + Transformer encoder layers + final norm
- **Paraformer Conformer**：Conformer blocks + CTC head（有 subsampling）

通用 GGML 图构建模式（参照 voxtral.cpp-cuda encoder 实现）：
```
ggml_new_graph_custom(gctx, ..., false)
→ 构建 encoder 算子链（conv/attention/norm/ffn）
→ ggml_build_forward_expand(gf, output)
→ ggml_backend_sched_alloc_graph(sched, gf)
→ ggml_backend_tensor_set(input, fbank_data, ...)
→ ggml_backend_sched_graph_compute(sched, gf)
→ ggml_backend_tensor_get(encoder_out, ...)
```

#### 2.4 多模态投影层（mmproj）

将音频编码器输出投影到 LLM embedding 空间：
- 通常为 MLP：`Linear → GELU → Linear` 或 `Linear → SiLU → Linear`
- 输出：audio token embeddings，与 LLM text embeddings 拼接

#### 2.5 LLM 解码器自回归循环

参照 voxtral.cpp-cuda decoder 实现：
- Prefill：将音频 embeddings + 前缀文本 embeddings 一次性输入
- Decode loop：逐 token 生成，KV cache 缓存历史
- 终止条件：EOS token 或 max_tokens 达到
- **具体超参数（n_layers, n_heads, rope_theta 等）从 GGUF 元数据读取**

#### 2.6 Tokenizer

从 LLM GGUF 元数据中提取词表：
- 读取 `tokenizer.ggml.tokens` 字符串数组
- 读取 `tokenizer.ggml.scores`（BPE 合并优先级）
- 解码：BPE/SentencePiece 风格 token 拼接

---

### Phase 3：构建验证与测试

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j$(nproc)

# 成功判据：
./build/fun-asr-nano --model-dir ./models --wav test.wav
# 输出：正确的中文转写文本
# CUDA 日志：显示 GPU 内存分配成功
```

---

## 4. 可验证的成功判据

| 判据 | 验证方法 |
|------|----------|
| 编译成功，链接 GGML CUDA | `cmake --build` 零错误 |
| GGUF 加载无报错 | 运行时打印所有张量 shape 与 GGUF 元数据匹配 |
| 音频编码器输出 shape 正确 | 打印 encoder_out 的 `ne[0..3]` |
| LLM 解码产生有效 token | 输出非空、非乱码的 UTF-8 中文文本 |
| CUDA 全程运行 | `ggml_backend_cuda_init` 无 fallback 警告 |
| 无 ONNX/llama.cpp 依赖 | `ldd ./build/fun-asr-nano` 不含 libonnxruntime/libllama |

---

## 附录 A：架构参数表（待 Phase 0 填写）

| 参数 | LLM GGUF 值 | mmproj GGUF 值 |
|------|------------|----------------|
| `llm.architecture` | TBD | N/A |
| `llm.block_count` | TBD | N/A |
| `llm.attention.head_count` | TBD | N/A |
| `llm.attention.head_count_kv` | TBD | N/A |
| `llm.embedding_length` | TBD | N/A |
| `llm.feed_forward_length` | TBD | N/A |
| `tokenizer.ggml.model` | TBD | N/A |
| `general.architecture` (mmproj) | N/A | TBD |
| audio encoder n_layers | N/A | TBD |
| audio encoder d_model | N/A | TBD |
| audio encoder n_mel | N/A | TBD |
| projector MLP structure | N/A | TBD |

---

## 附录 B：参考文件路径索引

| 参考对象 | 本地路径 |
|----------|---------|
| 参考项目（完整实现） | `/home/tanglin/workspace2/voxtral.cpp-cuda/src/voxtral.cpp` |
| 参考项目头文件 | `/home/tanglin/workspace2/voxtral.cpp-cuda/include/voxtral.h` |
| 参考 CMakeLists | `/home/tanglin/workspace2/voxtral.cpp-cuda/CMakeLists.txt` |
| GGML 核心 API | `/home/tanglin/workspace2/voxtral.cpp-cuda/ggml/include/ggml.h` |
| GGML Backend API | `/home/tanglin/workspace2/voxtral.cpp-cuda/ggml/include/ggml-backend.h` |
| GGUF 格式 API | `/home/tanglin/workspace2/voxtral.cpp-cuda/ggml/include/gguf.h` |
| CUDA Backend API | `/home/tanglin/workspace2/voxtral.cpp-cuda/ggml/include/ggml-cuda.h` |
| 目标 LLM 模型 | `./models/Fun-ASR-Nano-2512-llm-Q5_K_M.gguf`（待下载） |
| 目标音频投影器 | `./models/Fun-ASR-Nano-2512-audio-mmproj-mtmd-F16.gguf`（待下载） |
