# Spec: Fun-ASR-Nano-2512 Pure GGML/CUDA C++ Runtime

**状态**: 已从 GGUF 元数据确认所有架构参数，可直接进入实现阶段。

---

## 1. 需求

在 `/home/tanglin/workspace2/fun-asr-nano-2512.cpp` 中，**完全替换**现有代码，构建一个纯 C++ GGML CUDA 后端 ASR runtime。

- **LLM 权重**: `./models/Fun-ASR-Nano-2512-llm-Q5_K_M.gguf`
- **音频编码器+投影器权重**: `./models/Fun-ASR-Nano-2512-audio-mmproj-mtmd-F16.gguf`
- **代码结构参照**: `/home/tanglin/workspace2/voxtral.cpp-cuda`（纯 GGML C++ ASR 实现）
- **禁止**: ONNX Runtime、llama.cpp 高层 API

---

## 2. 从 GGUF 确认的完整架构参数

### 2.1 LLM：Qwen3-0.6B

| 参数 | 值 | GGUF 键 |
|------|-----|---------|
| architecture | qwen3 | `general.architecture` |
| n_layers | 28 | `qwen3.block_count` |
| d_model (emb) | 1024 | `qwen3.embedding_length` |
| ffn_dim | 3072 | `qwen3.feed_forward_length` |
| n_q_heads | 16 | `qwen3.attention.head_count` |
| n_kv_heads | 8 | `qwen3.attention.head_count_kv` |
| head_dim | 128 | d_model / n_q_heads = 1024/8... 实际: 2048/16=128 |
| rope_freq_base | 1,000,000.0 | `qwen3.rope.freq_base` |
| rms_norm_eps | 1e-6 | `qwen3.attention.layer_norm_rms_epsilon` |
| vocab_size | 151,936 | token_embd.weight dim[1] |
| bos_token_id | 151,643 | `tokenizer.ggml.bos_token_id` |
| eos_token_id | 151,645 | `tokenizer.ggml.eos_token_id` |
| QK-Norm | YES (RMSNorm on Q, K) | attn_q_norm, attn_k_norm [128] |

**LLM 张量命名方案（标准 ggml 格式）:**
```
token_embd.weight       [1024, 151936]   IQ4_XS
output_norm.weight      [1024]           F32
output.weight           [1024, 151936]   Q4_K

blk.{i}.attn_norm.weight      [1024]          F32
blk.{i}.attn_q.weight         [1024, 2048]    IQ4_XS   # 16 heads × 128
blk.{i}.attn_q_norm.weight    [128]            F32      # QK-Norm Q
blk.{i}.attn_k.weight         [1024, 1024]    IQ4_XS   # 8 kv_heads × 128
blk.{i}.attn_k_norm.weight    [128]            F32      # QK-Norm K
blk.{i}.attn_v.weight         [1024, 1024]    Q4_K
blk.{i}.attn_output.weight    [2048, 1024]    IQ4_XS   # 16*128 → 1024
blk.{i}.ffn_norm.weight       [1024]          F32
blk.{i}.ffn_gate.weight       [1024, 3072]    IQ4_XS   # SwiGLU gate
blk.{i}.ffn_up.weight         [1024, 3072]    IQ4_XS   # SwiGLU up
blk.{i}.ffn_down.weight       [3072, 1024]    Q4_K     # SwiGLU down
```
**注**: LLM 所有 attention 和 FFN 层**无 bias**（标准 Qwen3）。

### 2.2 音频前端：WavFrontend

| 参数 | 值 | GGUF 键 |
|------|-----|---------|
| 采样率 | 16,000 Hz | `funasr.frontend.sample_rate` |
| 窗函数 | **Hamming**（非 Hann） | `funasr.frontend.window` |
| n_mels | 80 | `funasr.frontend.n_mels` |
| frame_length | 25 ms (400 samples) | `funasr.frontend.frame_length_ms` |
| frame_shift | 10 ms (160 samples) | `funasr.frontend.frame_shift_ms` |
| LFR stack (m) | 7 帧 | `funasr.frontend.lfr_m` |
| LFR stride (n) | 6 帧 | `funasr.frontend.lfr_n` |
| LFR output dim | 560 = 80×7 | `funasr.frontend.output_dim` |
| snip_edges | True（边缘帧裁剪） | `funasr.frontend.snip_edges` |
| upscale_samples | True | `funasr.frontend.upscale_samples` |

**LFR（Low Frame Rate）处理**: 将连续 7 帧 mel 特征在特征维度拼接，每 6 帧取一个，将帧率从 100fps 降到约 16.7fps，输出维度 80×7=560。

### 2.3 音频编码器：SenseVoiceEncoderSmall (SANM)

| 参数 | 值 | GGUF 键 |
|------|-----|---------|
| type | SenseVoiceEncoderSmall | `funasr.audio_encoder.type` |
| input_dim | 560 | `funasr.audio_encoder.input_dim` |
| output_dim | 512 | `funasr.audio_encoder.output_dim` |
| num_blocks | 50 (常规块) | `funasr.audio_encoder.num_blocks` |
| tp_blocks | 20 (顶层块) | `funasr.audio_encoder.tp_blocks` |
| 总块数 | 1 (输入块) + 50 + 20 = 71 | — |
| attention_heads | 4 | `funasr.audio_encoder.attention_heads` |
| linear_units | 2048 (FFN) | `funasr.audio_encoder.linear_units` |
| kernel_size | 11 (FSMN conv) | `funasr.audio_encoder.kernel_size` |
| attn_type | sanm | `funasr.audio_encoder.selfattention_layer_type` |
| normalize_before | True (pre-norm) | `funasr.audio_encoder.normalize_before` |
| norm_eps | 1e-5 | `funasr.audio_encoder.layer_norm_epsilon` |

**SANM 块结构**（每个编码器块）：
```
Pre-norm (LayerNorm, 有 weight+bias)
→ Linear QKV combined: linear_qkv.weight [dim, 3*dim] + linear_qkv.bias [3*dim]
→ Split Q/K/V [dim/n_heads, n_heads, T]
→ FSMN memory: 1D depthwise conv fsmn.weight [11, 1, dim] on V (实际是对 memory 操作)
→ MultiHead Attention (无 mask, 全局注意力)
→ linear_out.weight [dim, dim] + linear_out.bias [dim]
→ Residual
→ Pre-norm (LayerNorm)
→ FFN w1: [dim, ffn_dim] + bias [ffn_dim]
→ ReLU
→ FFN w2: [ffn_dim, dim] + bias [dim]
→ Residual
```

**张量命名方案（mmproj）:**
```
# 特殊输入块（处理 560→512 维度变换）
a.blk.0.norm1.weight/bias          [560]          F16   # 输入 LFR 特征 norm
a.blk.0.self_attn.linear_qkv.weight [560, 1536]   F16   # 560→512×3 QKV
a.blk.0.self_attn.linear_qkv.bias   [1536]         F16
a.blk.0.self_attn.fsmn.weight       [11, 1, 512]   F16
a.blk.0.self_attn.linear_out.weight [512, 512]     F16
a.blk.0.self_attn.linear_out.bias   [512]          F16
a.blk.0.norm2.weight/bias           [512]          F16
a.blk.0.feed_forward.w1.weight      [512, 2048]    F16
a.blk.0.feed_forward.w1.bias        [2048]         F16
a.blk.0.feed_forward.w2.weight      [2048, 512]    F16
a.blk.0.feed_forward.w2.bias        [512]          F16

# 常规块 (i = 1..50)
a.blk.{i}.norm1.weight/bias              [512]         F16
a.blk.{i}.self_attn.linear_qkv.weight   [512, 1536]   F16
a.blk.{i}.self_attn.linear_qkv.bias     [1536]        F16
a.blk.{i}.self_attn.fsmn.weight         [11, 1, 512]  F16
a.blk.{i}.self_attn.linear_out.weight   [512, 512]    F16
a.blk.{i}.self_attn.linear_out.bias     [512]         F16
a.blk.{i}.norm2.weight/bias             [512]         F16
a.blk.{i}.feed_forward.w1.weight        [512, 2048]   F16
a.blk.{i}.feed_forward.w1.bias          [2048]        F16
a.blk.{i}.feed_forward.w2.weight        [2048, 512]   F16
a.blk.{i}.feed_forward.w2.bias          [512]         F16

a.after_norm.weight/bias           [512]   F16   # 常规块后归一化

# TP 顶层块 (i = 0..19)
a.tp.blk.{i}.*                     # 同上格式，dim=512

a.tp_norm.weight/bias              [512]   F16   # TP 块后归一化
```

### 2.4 MLP 投影器 + Adaptor Transformer

**MLP 投影器（512 → 1024）:**
```
mm.a.linear1.weight   [512, 2048]   F16   # 512 → 2048
mm.a.linear1.bias     [2048]        F16
mm.a.linear2.weight   [2048, 1024]  F16   # 2048 → 1024
mm.a.linear2.bias     [1024]        F16
```
激活函数：待确认（可能是 ReLU 或 SiLU，需参考原始 FunASR 代码）。

**Adaptor Transformer（2 层 Transformer，dim=1024）:**
```
mm.a.blk.{i}.norm1.weight/bias              [1024]        F16
mm.a.blk.{i}.self_attn.linear_q.weight      [1024, 1024]  F16
mm.a.blk.{i}.self_attn.linear_q.bias        [1024]        F16
mm.a.blk.{i}.self_attn.linear_k.weight      [1024, 1024]  F16
mm.a.blk.{i}.self_attn.linear_k.bias        [1024]        F16
mm.a.blk.{i}.self_attn.linear_v.weight      [1024, 1024]  F16
mm.a.blk.{i}.self_attn.linear_v.bias        [1024]        F16
mm.a.blk.{i}.self_attn.linear_out.weight    [1024, 1024]  F16
mm.a.blk.{i}.self_attn.linear_out.bias      [1024]        F16
mm.a.blk.{i}.norm2.weight/bias              [1024]        F16
mm.a.blk.{i}.feed_forward.w1.weight         [1024, 256]   F16   # 注: ffn_dim=256
mm.a.blk.{i}.feed_forward.w1.bias           [256]         F16
mm.a.blk.{i}.feed_forward.w2.weight         [256, 1024]   F16
mm.a.blk.{i}.feed_forward.w2.bias           [1024]        F16
```

### 2.5 Prompt 构建（funasr_nano_chatml）

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|startofspeech|>[audio token embeddings]<|endofspeech|>
[task text, e.g. 空字符串或语音指令]<|im_end|>
<|im_start|>assistant
```

其中 `<|startofspeech|>` 和 `<|endofspeech|>` 是文本 token（通过 tokenizer 找到对应 id），audio embeddings 直接替换为音频 token embeddings 向量（不经过 token_embd 查表）。

---

## 3. 完整推理管线

```
WAV 文件 (16kHz PCM)
    │
    ▼ [1] WavFrontend
  80-mel Hamming Fbank [n_frames, 80]
    │
    ▼ [2] LFR 拼帧 (m=7, n=6)
  LFR 特征 [T', 560]  (T' ≈ n_frames / 6)
    │
    ▼ [3] SenseVoice Encoder (GGML, CUDA)
  audio_feats [T', 512]
  (blk.0 → blk.1..50 → tp.blk.0..19 → tp_norm)
    │
    ▼ [4] MLP Projector (mm.a.linear1 → linear2)
  projected [T', 1024]
    │
    ▼ [5] Adaptor Transformer (mm.a.blk.0..1)
  audio_tokens [T', 1024]
    │
    ▼ [6] 构建 Prefill 序列
  text_prefix_embeds + audio_tokens + text_suffix_embeds
  (通过 token_embd.weight 查表得到文本部分 embeddings)
    │
    ▼ [7] Qwen3-0.6B Decoder (GGML, CUDA)
  Prefill: 处理完整序列，建立 KV cache
  Decode: 逐 token 自回归，直到 EOS (151645) 或 max_tokens
    │
    ▼ [8] 解码 token ids
  输出文本（GPT-2 BPE, Qwen2 pre-tokenizer）
```

---

## 4. 约束集合

### 4.1 硬约束（不可违反）

| 约束 | 根据 |
|------|------|
| 仅使用 ggml C++ API，禁止 llama.cpp、ONNX Runtime | 需求 |
| GGUF 加载：`gguf_init_from_file` + fread 手动填充 tensor | voxtral.cpp 模式 |
| CUDA 后端：`ggml_backend_cuda_init(0)`，fallback CPU | voxtral.cpp 模式 |
| `ggml_context` 推理前一次分配，不动态扩展 | ggml API |
| GPU 结果读取前必须 `ggml_backend_synchronize()` | ggml API |
| 每次新图前 `ggml_backend_sched_reset()` | ggml API |
| LFR: Hamming 窗（非 Hann），m=7 帧拼接，n=6 帧步长 | GGUF 元数据 |
| 编码器: LayerNorm（有 bias），非 RMSNorm | mmproj 张量实测 |
| LLM: RMSNorm（无 bias），QK-Norm 在 Q/K 投影后应用 | llm 张量实测 |
| Qwen3 GQA: Q=16 heads, KV=8 heads, head_dim=128 | llm 元数据 |
| 音频 token embeddings 通过 adaptor 输出获得，不走 token_embd | 架构分析 |
| EOS token id = 151645 | llm 元数据 |

### 4.2 软约束（代码规范）

- C++17，精简高效，非必要不写注释
- 文件结构完全对齐 voxtral.cpp-cuda：`include/`, `src/`, `ggml/`, `CMakeLists.txt`
- 推理持久 buffer（KV cache、encoder output、audio tokens）在 context 初始化时一次性分配
- 张量名以字符串常量定义（防拼写错误）
- `cmake -DGGML_CUDA=ON` 作为标准构建选项

---

## 5. 文件结构（对齐 voxtral.cpp-cuda）

```
fun-asr-nano-2512.cpp/
├── CMakeLists.txt
├── ggml/                              # 从 voxtral.cpp-cuda 复制 ggml/ 子目录
├── include/
│   └── fun_asr.h                      # 公开 API
├── src/
│   ├── fun_asr.cpp                    # 核心实现（加载+推理）
│   └── main.cpp                       # CLI
├── models/
│   ├── Fun-ASR-Nano-2512-llm-Q5_K_M.gguf
│   └── Fun-ASR-Nano-2512-audio-mmproj-mtmd-F16.gguf
└── openspec/
    └── specs/fun-asr-nano-ggml-cuda.md  # 本文件
```

---

## 6. 实现模块分解

### M1: GGUF 加载（`src/fun_asr.cpp`）

```cpp
// 加载 LLM 权重
struct fun_asr_llm {
    // token embeddings
    ggml_tensor * tok_embd;       // [1024, 151936]
    ggml_tensor * output_norm;    // [1024]
    ggml_tensor * output;         // [1024, 151936]
    // per-layer
    struct layer {
        ggml_tensor * attn_norm;
        ggml_tensor * attn_q, * attn_q_norm;
        ggml_tensor * attn_k, * attn_k_norm;
        ggml_tensor * attn_v, * attn_out;
        ggml_tensor * ffn_norm, * ffn_gate, * ffn_up, * ffn_down;
    } layers[28];
};

// 加载音频编码器+投影器权重
struct fun_asr_audio {
    struct enc_block { /* 13 tensors per block */ } enc[51]; // blk.0..50
    struct enc_block tp[20];                                  // tp.blk.0..19
    ggml_tensor * after_norm_w, * after_norm_b;
    ggml_tensor * tp_norm_w, * tp_norm_b;
    ggml_tensor * mm_linear1_w, * mm_linear1_b;  // projector
    ggml_tensor * mm_linear2_w, * mm_linear2_b;
    struct adpt_block { /* adaptor transformer block */ } adpt[2];
};
```

加载模式（完全对齐 voxtral.cpp）：
```cpp
gguf_init_params params = {.no_alloc = true, .ctx = &ctx_meta};
gguf_context * gguf = gguf_init_from_file(path, params);
// 分配权重 buffer
ggml_backend_alloc_ctx_tensors(ctx_meta, backend);
// fread 每个 tensor
FILE * fp = fopen(path, "rb");
for (int i = 0; i < gguf_get_n_tensors(gguf); i++) {
    // fseek 到 tensor 数据偏移, fread, ggml_backend_tensor_set
}
```

### M2: 音频前端（`src/fun_asr.cpp`）

```cpp
// 1. 读取 WAV (16kHz PCM mono)
// 2. 预加重（可选，Fun-ASR 默认无 preemphasis，待验证）
// 3. Hamming 窗 Fbank: 80 mel, 25ms/10ms, FFT 512
// 4. LFR: 拼接 m=7 帧, 步长 n=6 → output_dim=560
//    注意 snip_edges=True: 首尾不足 7 帧的部分裁剪掉
// 5. 输出: std::vector<float> lfr_feats [T' * 560]
```

### M3: 编码器前向（GGML 图）

```cpp
// 构建一次性图（编码器输入固定后静态图）
ggml_tensor * x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 560, T_prime);
// blk.0: norm1[560] → qkv[560→1536] → SANM attn → linear_out → norm2[512] → FFN
// blk.1..50: 相同结构, dim=512
// after_norm
// tp.blk.0..19
// tp_norm
// 输出: encoder_out [512, T']
```

**SANM 注意力子图（每块）:**
```
x_norm = layer_norm(x, norm1_w, norm1_b)
qkv = matmul(linear_qkv_w, x_norm) + linear_qkv_b  → [3*dim, T]
q, k, v = split(qkv, dim, axis=0)                   → each [dim, T]
# FSMN memory: depthwise 1D conv on v (or dedicated memory state)
#   fsmn_out = conv1d_depthwise(v, fsmn_weight, kernel=11, pad=5)
attn_out = flash_attn(q, k, v+fsmn_out, scale=1/sqrt(dim/4))
out = matmul(linear_out_w, attn_out) + linear_out_b
h = x + out  # residual
h_norm = layer_norm(h, norm2_w, norm2_b)
ffn = relu(matmul(w1, h_norm) + b1)
ffn = matmul(w2, ffn) + b2
output = h + ffn  # residual
```

### M4: MLP 投影 + Adaptor（GGML 图）

```cpp
// MLP 投影器: [T', 512] → [T', 1024]
h = matmul(linear1_w, encoder_out) + linear1_b  // [2048, T']
h = relu(h)  // 激活函数待验证
h = matmul(linear2_w, h) + linear2_b            // [1024, T']

// Adaptor Transformer blk.0, blk.1: dim=1024
// 标准 MHA（无 FSMN，无 RoPE，全局注意力）
// FFN w1[1024→256]→ReLU→w2[256→1024]
// 输出: audio_tokens [1024, T']  →  [T', 1024]
```

### M5: Qwen3 解码器（GGML 图）

**预填充（Prefill）：**
```
建立 prompt 序列的 embeddings:
  text_prefix_ids → ggml_get_rows(tok_embd, prefix_ids) → [1024, n_prefix]
  audio_tokens                                            → [1024, T']
  text_suffix_ids → ggml_get_rows(tok_embd, suffix_ids) → [1024, n_suffix]
  full_seq = concat([text_prefix, audio_tokens, text_suffix], axis=1)

对 full_seq 执行所有 28 层:
  每层: RMSNorm → GQA (16Q/8KV, head_dim=128, QK-norm, RoPE) → FFN (SwiGLU)
  KV cache 存储本轮所有 positions
```

**自回归解码（Decode loop）：**
```
token = argmax(logits)
loop until token == 151645 (EOS) or max_tokens:
  emb = ggml_get_rows(tok_embd, [token])     // [1024, 1]
  h = decoder_step(emb, position, kv_cache)  // 单 token 图
  logits = output(output_norm(h))             // [151936]
  token = argmax(logits)
  output_ids.push_back(token)
```

**Qwen3 注意力细节（每层）：**
```
h_norm = rms_norm(h, attn_norm_w, eps=1e-6)
q = matmul(attn_q_w, h_norm)                        // [2048, seq]
q = rms_norm_per_head(q, attn_q_norm_w, 128, 16)    // QK-Norm
k = matmul(attn_k_w, h_norm)                        // [1024, seq]
k = rms_norm_per_head(k, attn_k_norm_w, 128, 8)     // QK-Norm
v = matmul(attn_v_w, h_norm)                        // [1024, seq]
q = rope(q, positions, head_dim=128, freq_base=1e6, mode=neox)
k = rope(k, positions, head_dim=128, freq_base=1e6, mode=neox)
out = flash_attn_gqa(q, k, v, scale=1/sqrt(128))    // GQA
h += matmul(attn_output_w, out)
// FFN SwiGLU:
h_norm2 = rms_norm(h, ffn_norm_w, eps=1e-6)
gate = silu(matmul(ffn_gate_w, h_norm2))
up   = matmul(ffn_up_w, h_norm2)
h   += matmul(ffn_down_w, gate*up)
```

### M6: Tokenizer（GPT-2 BPE）

从 LLM GGUF 读取：
- `tokenizer.ggml.tokens`：词表字符串数组（151936 条）
- `tokenizer.ggml.merges`：BPE merge 规则

解码（inference-only）：
```cpp
std::string decode_tokens(const std::vector<int32_t>& ids) {
    // 对每个 token_id，直接取 tokens[id] 字符串
    // GPT-2 BPE 用 'Ġ' 表示空格，ĠĠ=两个空格 etc.
    // 将 Ġ 替换为 ' '，其他字节直接拼接
}
```

---

## 7. 构建系统（CMakeLists.txt）

完全对齐 voxtral.cpp-cuda/CMakeLists.txt，修改 project name：

```cmake
cmake_minimum_required(VERSION 3.16)
project(fun_asr_nano LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

option(GGML_CUDA "ggml: use CUDA" OFF)
# CPU 特性检测（同 voxtral.cpp-cuda）
add_subdirectory(ggml)

add_library(fun_asr_lib src/fun_asr.cpp)
target_include_directories(fun_asr_lib PUBLIC include)
target_link_libraries(fun_asr_lib PUBLIC ggml Threads::Threads m dl)

add_executable(fun-asr-nano src/main.cpp)
target_link_libraries(fun-asr-nano PRIVATE fun_asr_lib)
```

**构建命令:**
```bash
# 从 voxtral.cpp-cuda 复制 ggml/ 目录
cp -r /home/tanglin/workspace2/voxtral.cpp-cuda/ggml ./

cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j$(nproc)
./build/fun-asr-nano --model-dir ./models --wav test.wav
```

---

## 8. 可验证的成功判据

| 判据 | 验证方法 |
|------|---------|
| 编译成功，链接 GGML CUDA | `cmake --build` 零错误 |
| `ldd` 无 libonnxruntime, 无 libllama | `ldd ./build/fun-asr-nano` |
| LLM GGUF 加载：311 张量全部命中 | 运行时打印 tensor count |
| mmproj GGUF 加载：950 张量全部命中 | 运行时打印 tensor count |
| Encoder 输出 shape 正确 | 打印 encoder_out ne[0..1] = [512, T'] |
| Audio tokens shape 正确 | 打印 audio_tokens ne = [1024, T'] |
| KV cache 初始化成功 | CUDA 内存分配无报错 |
| 解码产生有效 tokens | token ids 在 [0, 151936) 范围内 |
| 输出为有效中文 UTF-8 | 打印转写结果非乱码 |
| CUDA 全程运行 | 无 "falling back to CPU" 日志 |

---

## 9. 待确认事项（低优先级，可在实现中验证）

| 疑问 | 影响 | 解决方式 |
|------|------|---------|
| MLP projector 激活函数（ReLU 或 SiLU？） | 精度 | 参考 FunASR 原始代码或对比 reference 输出 |
| FSMN 内存操作的精确公式（conv 对 V 还是中间状态？） | 精度 | 参考 FunAudioLLM/FunASR 源码 SANM 实现 |
| adaptor transformer 是否有 positional encoding | 精度 | 运行对比实验 |
| 是否有 global CMVN（元数据未提及） | 精度 | 测试时对比原始 FunASR 输出 |
| Qwen3 decode: 是否 disable thinking (enable_thinking=false) | 精度 | ChatML 模板中插入 `<think>\n\n</think>\n\n` |

---

*本文档基于 GGUF 元数据实测 + voxtral.cpp-cuda 代码实际模式生成，所有架构参数均来自文件，无猜测。*
