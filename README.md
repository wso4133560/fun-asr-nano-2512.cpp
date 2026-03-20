# fun-asr-nano-2512.cpp

纯 C++ / GGML-CUDA 推理后端，运行 [FunASR-Nano-2512](https://modelscope.cn/models/iic/FunAudioLLM-SenseVoice-GGUF) 多语言语音识别模型。

## 架构

- **音频编码器**：SenseVoiceEncoderSmall（SANM，50 层主编码 + 20 层 TP 编码）
- **语言模型**：Qwen3-0.6B（28 层，KV-cache 解码）
- **前端**：Hamming 窗 80-mel Fbank，LFR(m=7,n=6)，16 kHz
- **量化**：LLM Q5_K_M，音频编码器 F16

## 性能（RTX 3080，7.2s 日语音频）

| 阶段 | 耗时 |
|------|------|
| 编码器 | ~120 ms |
| Prefill（86 tokens） | ~18 ms |
| 解码（25 tokens） | ~92 ms |
| **端到端转写** | **~0.26 s** |

| 指标 | 数值 |
|------|------|
| 解码速度 | ~280 tok/s |
| 整体吞吐 | ~97 tok/s |

## 构建

```bash
# 拷贝 ggml 子目录（或使用项目内已有的）
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j$(nproc)
```

## 使用

```bash
./build/fun-asr-nano \
    --model-dir ./models \
    --wav input.wav \
    [--lang ja|zh|en|...] \
    [--task transcribe|translate] \
    [--max-tokens 512] \
    [--dump-json]
```

### 可选：CTC 提示（提升专有名词准确率）

```bash
./build/fun-asr-nano \
    --model-dir ./models \
    --wav input.wav \
    --auto-ctc \
    --ctc-model-dir /path/to/ctc_model
```

## 模型下载

```bash
# 默认下载 Q5_K_M（推荐）
bash tools/download_models.sh

# 指定量化版本和输出目录
bash tools/download_models.sh --quant Q4_K_M --dir /path/to/models
```

可选量化：`F16` / `Q4_K_M` / `Q5_K_M`（默认）。
模型来源：[HuggingFace wso4133560freewind/Fun-ASR-Nano-2512-gguf](https://huggingface.co/wso4133560freewind/Fun-ASR-Nano-2512-gguf)

| 文件 | 说明 |
|------|------|
| `Fun-ASR-Nano-2512-llm-Q5_K_M.gguf` | Qwen3-0.6B 解码器（推荐） |
| `Fun-ASR-Nano-2512-llm-Q4_K_M.gguf` | Qwen3-0.6B 解码器（更小） |
| `Fun-ASR-Nano-2512-llm-F16.gguf` | Qwen3-0.6B 解码器（全精度） |
| `Fun-ASR-Nano-2512-audio-mmproj-mtmd-F16.gguf` | SenseVoice 编码器 + MLP 投影 + Adaptor |
