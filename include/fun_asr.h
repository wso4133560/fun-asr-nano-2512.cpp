#ifndef FUN_ASR_H
#define FUN_ASR_H

#include "ggml.h"
#include "ggml-backend.h"

#include <stdint.h>
#include <string>
#include <vector>

// ============================================================================
// Constants
// ============================================================================

// LLM (Qwen3-0.6B)
#define FUN_ASR_LLM_LAYERS      28
#define FUN_ASR_LLM_DIM         1024
#define FUN_ASR_LLM_FFN_DIM     3072
#define FUN_ASR_LLM_N_Q_HEADS   16
#define FUN_ASR_LLM_N_KV_HEADS  8
#define FUN_ASR_LLM_HEAD_DIM    128
#define FUN_ASR_LLM_NORM_EPS    1e-6f
#define FUN_ASR_LLM_ROPE_THETA  1000000.0f
#define FUN_ASR_LLM_VOCAB_SIZE  151936
#define FUN_ASR_LLM_EOS         151645
#define FUN_ASR_KV_DIM          (FUN_ASR_LLM_N_KV_HEADS * FUN_ASR_LLM_HEAD_DIM)  // 1024
#define FUN_ASR_KV_WINDOW       2048

// Audio encoder (SenseVoiceEncoderSmall)
#define FUN_ASR_ENC_DIM         512
#define FUN_ASR_ENC_INPUT_DIM   560
#define FUN_ASR_ENC_LAYERS      50   // blk.0..49
#define FUN_ASR_ENC_TP_LAYERS   20   // tp.blk.0..19
#define FUN_ASR_ENC_HEADS       4
#define FUN_ASR_ENC_NORM_EPS    1e-5f

// Audio frontend
#define FUN_ASR_SAMPLE_RATE     16000
#define FUN_ASR_NUM_MEL_BINS    80
#define FUN_ASR_FRAME_LEN       400
#define FUN_ASR_FRAME_SHIFT     160
#define FUN_ASR_LFR_M           7
#define FUN_ASR_LFR_N           6
#define FUN_ASR_MAX_ENC_SEQ     3000

// ============================================================================
// Weights structs (opaque; defined in fun_asr.cpp)
// ============================================================================

struct fun_asr_llm;
struct fun_asr_audio;

// ============================================================================
// Context
// ============================================================================

struct fun_asr_ctx {
    fun_asr_llm   * llm   = nullptr;
    fun_asr_audio * audio = nullptr;

    ggml_backend_t        backend     = nullptr;
    ggml_backend_buffer_t llm_buf     = nullptr;
    ggml_backend_buffer_t audio_buf   = nullptr;
    ggml_backend_buffer_t kv_buf      = nullptr;

    // KV cache: [kv_dim, window, layers] x 2 (K, V)
    ggml_tensor * k_cache = nullptr;
    ggml_tensor * v_cache = nullptr;
    int32_t       kv_used = 0;

    // Persistent ggml contexts for weight tensors
    ggml_context * llm_gctx   = nullptr;
    ggml_context * audio_gctx = nullptr;

    // Persistent decode graph (built once, reserved, reused every step)
    ggml_context * dec_gctx   = nullptr;
    ggml_cgraph  * dec_graph  = nullptr;
    ggml_tensor  * dec_tok_t  = nullptr;  // input: token id [1] i32
    ggml_tensor  * dec_pos_t  = nullptr;  // input: position [1] i32
    ggml_tensor  * dec_out_t  = nullptr;  // output: next token [1] i32
    bool           dec_ready  = false;

    // Last decode diagnostics, populated by fun_asr_transcribe().
    int32_t last_generated_tokens = 0;
    int32_t last_attempts_used = 0;
    int32_t last_repetition_retries = 0;
    int32_t last_punctuation_retries = 0;
    float   last_final_temperature = 0.0f;
};

struct fun_asr_decode_options {
    std::string language;
    std::string task;
    std::string context;
    std::string ctc_text;
    float temperature_base = 0.0f;
    float temperature_step = 0.3f;
    int32_t top_k = 50;
    int32_t retry_attempts = 4;
    int32_t seed = -1;
};

// ============================================================================
// Public API
// ============================================================================

// Initialize context: load both GGUF files, allocate KV cache.
// model_dir: directory containing Fun-ASR-Nano-2512-llm-Q5_K_M.gguf
//            and Fun-ASR-Nano-2512-audio-mmproj-mtmd-F16.gguf
// use_cuda:  true = CUDA backend (device 0); false = CPU backend
fun_asr_ctx * fun_asr_init(const std::string & model_dir, bool use_cuda);

// Transcribe a WAV file. Returns UTF-8 text, or empty string on failure.
std::string fun_asr_transcribe(
    fun_asr_ctx * ctx,
    const std::string & wav_path,
    int32_t max_tokens = 512,
    const fun_asr_decode_options * options = nullptr);

// Free all resources.
void fun_asr_free(fun_asr_ctx * ctx);

#endif // FUN_ASR_H
