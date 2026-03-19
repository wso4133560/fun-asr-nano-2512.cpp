## 1. 项目骨架

- [x] 1.1 从 voxtral.cpp-cuda 复制 ggml 子目录：`cp -r /home/tanglin/workspace2/voxtral.cpp-cuda/ggml ./`
- [x] 1.2 创建目录结构：`mkdir -p include src`
- [x] 1.3 编写 `CMakeLists.txt`（C++17, add_subdirectory(ggml), GGML_CUDA option, 目标 fun_asr_lib + fun-asr-nano）

## 2. 公开 API 头文件

- [x] 2.1 编写 `include/fun_asr.h`：定义 `fun_asr_llm`, `fun_asr_audio`, `fun_asr_ctx` 结构体
- [x] 2.2 声明公开函数：`fun_asr_init(model_dir, device)`, `fun_asr_transcribe(ctx, wav_path)`, `fun_asr_free(ctx)`

## 3. GGUF 加载（gguf-loader spec）

- [x] 3.1 实现 `load_llm_weights(path, backend)`：`gguf_init_from_file(no_alloc=true)` → `ggml_backend_alloc_ctx_tensors` → fread 循环 → `ggml_backend_tensor_set`，验证 311 tensors 全部命中
- [x] 3.2 实现 `load_mmproj_weights(path, backend)`：同上，验证 950 tensors 全部命中；特别确认 `a.blk.0.self_attn.linear_qkv.weight` 形状为 [560,1536]
- [x] 3.3 实现 backend 初始化：`ggml_backend_cuda_init(0)` → fallback `ggml_backend_cpu_init()`
- [x] 3.4 实现 `fun_asr_free`：按顺序释放 `ggml_backend_buffer_free`, `gguf_free`, `ggml_free`, `ggml_backend_free`

## 4. 音频前端（wav-frontend spec）

- [x] 4.1 实现 WAV 文件读取：解析 RIFF 头，支持 16-bit PCM mono/stereo；stereo 平均为 mono；输出 float32 归一化到 [-1,1]
- [x] 4.2 实现样本上缩放：读取后立即 `× 32768.0`（upscale_samples=True）
- [x] 4.3 实现 Hamming 窗：`w[n] = 0.54 - 0.46 * cos(2π*n/399)` for n∈[0,399]，预计算并缓存
- [x] 4.4 实现 512 点 FFT（radix-2 或使用 voxtral.cpp-cuda 的 FFT 实现直接复用）
- [x] 4.5 实现 80-mel 三角滤波器组（16kHz，snip_edges=True）；帧数公式：`floor((N-400)/160)+1`
- [x] 4.6 实现 LFR 拼帧：stack m=7 帧，stride n=6，输出 [T', 560]；T' < 7 时返回空

## 5. SenseVoice 编码器（sensevoice-encoder spec）

- [x] 5.1 实现 `build_encoder_graph(gctx, input [560,T'])`：`ggml_new_graph_custom(gctx, 4096, false)`
- [x] 5.2 实现 blk.0 的前向传播：`ggml_norm` LayerNorm(dim=560) → SANM attention（QKV [560,1536]）→ FSMN conv → linear_out → norm2(512) → ReLU FFN(512→2048→512)
- [x] 5.3 实现 blk.1..50 的前向传播（循环，dim=512 全程不变）：LayerNorm → SANM → FSMN → linear_out → LayerNorm → ReLU FFN
- [x] 5.4 实现 after_norm（`ggml_norm + ggml_mul + ggml_add`，eps=1e-5）
- [x] 5.5 实现 tp.blk.0..19 的前向传播（与 blk.1..50 结构相同）
- [x] 5.6 实现 tp_norm；最终输出 encoder_out [512, T']
- [x] 5.7 验证 SANM FSMN 公式：`fsmn_m = ggml_conv_1d(x_norm, fsmn_w, stride=1, pad=5)`；组合：`linear_out(mha_out + fsmn_m)`

## 6. MLP 投影器 + Adaptor（audio-projector spec）

- [x] 6.1 实现 MLP 投影：`matmul(linear1_w, enc_out) + linear1_b` → `ggml_silu` → `matmul(linear2_w) + linear2_b`，输出 [1024, T']
- [x] 6.2 实现 Adaptor blk.0 和 blk.1（2 层 Transformer，dim=1024，4 heads，head_dim=256，FFN dim=256）
- [x] 6.3 Adaptor LayerNorm：`ggml_norm`（eps≈1e-12）+ weight + bias（均为 [1024]）
- [x] 6.4 Adaptor attention：分离 Q/K/V 各 [1024,1024]，无 RoPE，全局注意力（`ggml_flash_attn_ext`）
- [x] 6.5 Adaptor FFN：`w1[1024,256]→ReLU→w2[256,1024]`，均有 bias
- [x] 6.6 Adaptor 输出作为 audio_tokens [1024, T']，存入持久 buffer

## 7. BPE Tokenizer（bpe-tokenizer spec）

- [x] 7.1 实现 `load_tokenizer`：从 LLM GGUF 读取 `tokenizer.ggml.tokens`（151936 条）和 `tokenizer.ggml.merges` 到内存
- [x] 7.2 实现 `find_token_id(str)` → token id（用于定位 `<|startofspeech|>` 等特殊 token）
- [x] 7.3 实现 `tokenize(text)` → `std::vector<int32_t>`（GPT-2 BPE 编码，qwen2 pre-tokenizer）
- [x] 7.4 实现 `decode_tokens(ids)` → `std::string`：替换 Ġ→空格、Ċ→换行，跳过 EOS(151645)

## 8. Qwen3 解码器（qwen3-decoder spec）

- [x] 8.1 实现持久 KV cache 分配：`[1024, 2048, 28]` × 2（K+V）在 context 初始化时分配到 device
- [x] 8.2 实现 prompt 构建：tokenize system + prefix → 拼接 audio_tokens → tokenize suffix（含 `<think>\n\n</think>\n\n`）→ 混合 embedding 序列
- [x] 8.3 实现 prefill 图构建：全序列送入 28 层 Qwen3，每层写 KV cache；`kv_used = prefill_len`
- [x] 8.4 实现单层 Qwen3 forward（可复用于 prefill 和 decode）：RMSNorm(eps=1e-6) → GQA(16Q/8KV) with QK-Norm → RoPE(freq=1e6,head_dim=128,neox) → flash_attn → output_proj → SwiGLU FFN(gate×up→down，无 bias）
- [x] 8.5 实现 QK-Norm：reshape [128,n_heads,T] → `ggml_rms_norm` → scale → reshape back（Q 和 K 各自）
- [x] 8.6 实现 decode 单 token 图（position + kv cache read/write）
- [x] 8.7 实现 KV cache shift-left eviction：`kv_used == 2048` 时 memmove rows[1..2047]→[0..2046]
- [x] 8.8 实现自回归循环：直到 token==151645(EOS) 或 max_tokens；decode_tokens 转换输出

## 9. CLI 入口（main.cpp）

- [x] 9.1 实现 `src/main.cpp`：解析 `--model-dir`, `--wav`, `--device [cuda|cpu]`, `--max-tokens [512]`
- [x] 9.2 调用 `fun_asr_init` → `fun_asr_transcribe` → 打印结果 → `fun_asr_free`
- [x] 9.3 打印 timing 信息：frontend_ms, encoder_ms, projector_ms, decode_ms

## 10. 构建与验证

- [ ] 10.1 构建：`cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON && cmake --build build -j$(nproc)`
- [ ] 10.2 验证无外部依赖：`ldd ./build/fun-asr-nano | grep -E 'onnx|llama'` → 空输出
- [ ] 10.3 运行基础测试：`./build/fun-asr-nano --model-dir ./models --wav test.wav` → 有效中文 UTF-8 输出
- [ ] 10.4 CUDA 验证：运行时无 "falling back to CPU" 日志；CUDA device 0 被使用
- [ ] 10.5 形状验证：encoder_out shape [512, T']，audio_tokens shape [1024, T']，prefill 无越界
