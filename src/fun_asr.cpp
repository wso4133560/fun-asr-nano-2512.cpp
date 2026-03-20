#include "fun_asr.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ============================================================================
// Internal constants
// ============================================================================

static constexpr float PI = 3.14159265358979323846f;
static constexpr int32_t N_FFT       = FUN_ASR_FRAME_LEN;      // 400
static constexpr int32_t N_FREQ      = N_FFT / 2 + 1;           // 201 (matches torchaudio rfft n=400)
static constexpr float   MEL_LOW_HZ  = 20.0f;
static constexpr float   MEL_HIGH_HZ = 8000.0f; // kaldi nyquist
static constexpr int32_t MAX_ENC_SEQ = FUN_ASR_MAX_ENC_SEQ;    // 3000 (raw LFR frames)
static constexpr int32_t FUN_ASR_LLM_STOP_TOKEN = 151643;

static std::string to_ascii_lower(std::string s) {
    for (char & ch : s) {
        ch = (char)std::tolower((unsigned char)ch);
    }
    return s;
}

static std::string canonical_language_hint(const std::string & raw) {
    if (raw.empty()) return "";

    const std::string lower = to_ascii_lower(raw);
    if (lower == "zh" || lower == "zh-cn" || lower == "cn" || lower == "chinese") return "中文";
    if (lower == "en" || lower == "en-us" || lower == "en-gb" || lower == "english") return "英文";
    if (lower == "ja" || lower == "jp" || lower == "ja-jp" || lower == "japanese") return "日语";
    if (lower == "yue" || lower == "cantonese") return "粤语";
    if (lower == "ko" || lower == "korean") return "韩文";
    return raw;
}

static std::string canonical_task_hint(const std::string & raw) {
    if (raw.empty()) return "";

    const std::string lower = to_ascii_lower(raw);
    if (lower == "transcribe" || lower == "transcription" || lower == "asr" || lower == "recognize") {
        return "语音转写";
    }
    if (lower == "translate" || lower == "translation") {
        return "语音翻译";
    }
    return raw;
}

static std::string build_prompt_suffix(const fun_asr_decode_options * options) {
    std::string suffix;
    if (options && !options->ctc_text.empty()) {
        suffix += "\n参考转写：" + options->ctc_text;
    }
    suffix += "\n请直接输出最终转写文本，不要添加解释。<|im_end|>\n<|im_start|>assistant\n";
    return suffix;
}

static std::string build_prompt_prefix(const fun_asr_decode_options * options) {
    std::string prefix =
        "<|im_start|>system\n"
        "You are a helpful assistant for multilingual speech recognition."
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "请根据音频内容完成高质量语音转写，只输出最终文本。\n";

    if (options) {
        const std::string language = canonical_language_hint(options->language);
        const std::string task     = canonical_task_hint(options->task);
        if (!language.empty()) {
            prefix += "目标语言：" + language + "\n";
        }
        if (options && !options->context.empty()) {
            prefix += "上下文信息：" + options->context + "\n";
        }
        if (!task.empty()) {
            prefix += "任务：" + task + "\n";
        }
    }
    prefix += "请结合音频与参考文本纠正专有名词、数字和标点。\n";
    return prefix;
}

// ============================================================================
// Internal weight structures
// ============================================================================

struct enc_layer {
    int32_t norm1_dim;     // 560 for blk.0, 512 for blk.1+ and tp.blk
    ggml_tensor * norm1_w; // [norm1_dim]
    ggml_tensor * norm1_b;
    ggml_tensor * qkv_w;   // [norm1_dim, 1536] (GGML: ne[0]=norm1_dim, ne[1]=1536)
    ggml_tensor * qkv_b;   // [1536]
    ggml_tensor * fsmn_w;  // [11, 1, 512]
    ggml_tensor * out_w;   // [512, 512]  (ne[0]=512, ne[1]=512)
    ggml_tensor * out_b;   // [512]
    ggml_tensor * norm2_w; // [512]
    ggml_tensor * norm2_b;
    ggml_tensor * ffn_w1;  // [512, 2048] (ne[0]=512, ne[1]=2048)
    ggml_tensor * ffn_b1;  // [2048]
    ggml_tensor * ffn_w2;  // [2048, 512] (ne[0]=2048, ne[1]=512)
    ggml_tensor * ffn_b2;  // [512]
};

struct ada_layer {
    ggml_tensor * norm1_w; // [1024]
    ggml_tensor * norm1_b;
    ggml_tensor * q_w;     // [1024, 1024]
    ggml_tensor * q_b;     // [1024]
    ggml_tensor * k_w;
    ggml_tensor * k_b;
    ggml_tensor * v_w;
    ggml_tensor * v_b;
    ggml_tensor * o_w;     // [1024, 1024]
    ggml_tensor * o_b;
    ggml_tensor * norm2_w; // [1024]
    ggml_tensor * norm2_b;
    ggml_tensor * ffn_w1;  // [1024, 256] (ne[0]=1024, ne[1]=256 → 1024→256)
    ggml_tensor * ffn_b1;  // [256]
    ggml_tensor * ffn_w2;  // [256, 1024]
    ggml_tensor * ffn_b2;  // [1024]
};

struct fun_asr_audio {
    enc_layer blk[FUN_ASR_ENC_LAYERS];        // blk.0..50
    ggml_tensor * after_norm_w;                // [512]
    ggml_tensor * after_norm_b;
    enc_layer tp_blk[FUN_ASR_ENC_TP_LAYERS];  // tp.blk.0..19
    ggml_tensor * tp_norm_w;                   // [512]
    ggml_tensor * tp_norm_b;

    // MLP projector
    ggml_tensor * proj_w1;  // [512, 2048]
    ggml_tensor * proj_b1;  // [2048]
    ggml_tensor * proj_w2;  // [2048, 1024]
    ggml_tensor * proj_b2;  // [1024]

    // Adaptor blocks (2 layers)
    ada_layer ada[2];

    // Persistent inference buffers
    ggml_context          * out_gctx     = nullptr;
    ggml_backend_buffer_t   out_buf      = nullptr;
    ggml_tensor           * enc_out      = nullptr; // [512, MAX_ENC_SEQ]
    ggml_tensor           * audio_tokens = nullptr; // [1024, MAX_ENC_SEQ]
};

struct llm_layer {
    ggml_tensor * attn_norm;   // [1024]
    ggml_tensor * attn_q;      // [1024, 2048]
    ggml_tensor * attn_k;      // [1024, 1024]
    ggml_tensor * attn_v;      // [1024, 1024]
    ggml_tensor * attn_out;    // [2048, 1024]
    ggml_tensor * attn_q_norm; // [128]
    ggml_tensor * attn_k_norm; // [128]
    ggml_tensor * ffn_norm;    // [1024]
    ggml_tensor * ffn_gate;    // [1024, 3072]
    ggml_tensor * ffn_up;      // [1024, 3072]
    ggml_tensor * ffn_down;    // [3072, 1024]
};

struct fun_asr_llm {
    ggml_tensor * tok_embd;     // [1024, 151936]
    ggml_tensor * output_w;     // [1024, 151936] (shared or separate)
    ggml_tensor * output_norm;  // [1024]
    llm_layer     layers[FUN_ASR_LLM_LAYERS];

    // Tokenizer data (GPT-2 BPE)
    std::vector<std::string>             vocab;        // id → string
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_ranks; // "A B" → rank
};

// Extended context fields stored alongside the public struct
struct fun_asr_ctx_ext {
    ggml_backend_t       backend_cpu = nullptr;
    ggml_backend_sched_t sched       = nullptr;
    ggml_gallocr_t       galloc_dec  = nullptr;  // decode graph allocator
    ggml_gallocr_t       galloc_enc  = nullptr;  // encoder graph allocator
    ggml_gallocr_t       galloc_proj = nullptr;  // projector graph allocator
    ggml_gallocr_t       galloc_pf   = nullptr;  // prefill graph allocator
};

// We keep a parallel ext struct keyed to ctx pointer
// (added as a simple global map for this single-context use case)
static fun_asr_ctx_ext g_ext;

// ============================================================================
// GGUF tensor helper
// ============================================================================

static ggml_tensor * require_tensor(ggml_context * gctx, const char * name) {
    ggml_tensor * t = ggml_get_tensor(gctx, name);
    if (!t) {
        fprintf(stderr, "fun_asr: tensor '%s' not found\n", name);
    }
    return t;
}

// ============================================================================
// GGUF weight loading
// ============================================================================

static bool load_gguf(
    const std::string       & path,
    ggml_backend_t            backend,
    ggml_context           ** out_gctx,
    ggml_backend_buffer_t   * out_buf)
{
    ggml_init_params meta_p = { 0, nullptr, true };
    ggml_context * ctx_meta = nullptr;

    gguf_init_params gp = { true, &ctx_meta };
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gp);
    if (!gctx || !ctx_meta) {
        fprintf(stderr, "fun_asr: failed to open %s\n", path.c_str());
        return false;
    }

    *out_buf = ggml_backend_alloc_ctx_tensors(ctx_meta, backend);
    if (!*out_buf) {
        fprintf(stderr, "fun_asr: failed to allocate weight buffer for %s\n", path.c_str());
        gguf_free(gctx);
        ggml_free(ctx_meta);
        return false;
    }

    FILE * fp = fopen(path.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "fun_asr: cannot open %s for reading\n", path.c_str());
        ggml_backend_buffer_free(*out_buf);
        gguf_free(gctx);
        ggml_free(ctx_meta);
        return false;
    }

    const int n = gguf_get_n_tensors(gctx);
    for (int i = 0; i < n; i++) {
        const char    * name   = gguf_get_tensor_name(gctx, i);
        ggml_tensor   * t      = ggml_get_tensor(ctx_meta, name);
        if (!t) continue;
        const size_t    offset = gguf_get_data_offset(gctx) + gguf_get_tensor_offset(gctx, i);
        const size_t    nbytes = ggml_nbytes(t);
        std::vector<uint8_t> tmp(nbytes);
        if (fseek(fp, (long)offset, SEEK_SET) != 0 ||
            fread(tmp.data(), 1, nbytes, fp) != nbytes) {
            fprintf(stderr, "fun_asr: failed to read tensor '%s'\n", name);
            fclose(fp);
            ggml_backend_buffer_free(*out_buf);
            gguf_free(gctx);
            ggml_free(ctx_meta);
            return false;
        }
        ggml_backend_tensor_set(t, tmp.data(), 0, nbytes);
    }
    fclose(fp);

    // Load tokenizer data before freeing gctx if present (done in load_llm_weights)
    gguf_free(gctx);
    *out_gctx = ctx_meta;
    return true;
}

// Load LLM weights from GGUF
static bool load_llm_weights(const std::string & path, ggml_backend_t backend, fun_asr_ctx * ctx) {
    // First pass: load tokenizer metadata
    {
        ggml_init_params mp = { 0, nullptr, true };
        ggml_context * ctx_meta = nullptr;
        gguf_init_params gp = { true, &ctx_meta };
        gguf_context * gctx = gguf_init_from_file(path.c_str(), gp);
        if (!gctx) return false;

        fun_asr_llm * llm = ctx->llm;

        // tokens
        int64_t tok_key = gguf_find_key(gctx, "tokenizer.ggml.tokens");
        if (tok_key >= 0) {
            size_t n = gguf_get_arr_n(gctx, tok_key);
            llm->vocab.resize(n);
            for (size_t i = 0; i < n; i++) {
                llm->vocab[i] = gguf_get_arr_str(gctx, tok_key, i);
                llm->token_to_id[llm->vocab[i]] = (int32_t)i;
            }
        }
        // merges
        int64_t merge_key = gguf_find_key(gctx, "tokenizer.ggml.merges");
        if (merge_key >= 0) {
            size_t n = gguf_get_arr_n(gctx, merge_key);
            llm->merge_ranks.reserve(n);
            for (size_t i = 0; i < n; i++) {
                std::string s = gguf_get_arr_str(gctx, merge_key, i);
                llm->merge_ranks[s] = (int32_t)i;
            }
        }

        gguf_free(gctx);
        ggml_free(ctx_meta);
    }

    // Second pass: load tensor data
    if (!load_gguf(path, backend, &ctx->llm_gctx, &ctx->llm_buf)) return false;

    fun_asr_llm * llm = ctx->llm;
    ggml_context * gc = ctx->llm_gctx;

    llm->tok_embd    = require_tensor(gc, "token_embd.weight");
    llm->output_w    = require_tensor(gc, "output.weight");
    llm->output_norm = require_tensor(gc, "output_norm.weight");

    char buf[128];
    for (int i = 0; i < FUN_ASR_LLM_LAYERS; i++) {
        auto & L = llm->layers[i];
        #define T(name) do { \
            snprintf(buf, sizeof(buf), "blk.%d." name, i); \
            L.name##_ = require_tensor(gc, buf); \
        } while(0)
        // manual because macro tricks won't work for compound names
        snprintf(buf, sizeof(buf), "blk.%d.attn_norm.weight", i);
        L.attn_norm = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "blk.%d.attn_q.weight", i);
        L.attn_q = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "blk.%d.attn_k.weight", i);
        L.attn_k = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "blk.%d.attn_v.weight", i);
        L.attn_v = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "blk.%d.attn_output.weight", i);
        L.attn_out = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "blk.%d.attn_q_norm.weight", i);
        L.attn_q_norm = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "blk.%d.attn_k_norm.weight", i);
        L.attn_k_norm = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "blk.%d.ffn_norm.weight", i);
        L.ffn_norm = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "blk.%d.ffn_gate.weight", i);
        L.ffn_gate = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "blk.%d.ffn_up.weight", i);
        L.ffn_up = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "blk.%d.ffn_down.weight", i);
        L.ffn_down = require_tensor(gc, buf);
        #undef T
    }
    return true;
}

// Load audio encoder + projector + adaptor weights from mmproj GGUF
static bool load_mmproj_weights(const std::string & path, ggml_backend_t backend, fun_asr_ctx * ctx) {
    if (!load_gguf(path, backend, &ctx->audio_gctx, &ctx->audio_buf)) return false;

    fun_asr_audio * au = ctx->audio;
    ggml_context  * gc = ctx->audio_gctx;
    char buf[256];

    // Encoder blk.0..50
    for (int i = 0; i < FUN_ASR_ENC_LAYERS; i++) {
        auto & L = au->blk[i];
        L.norm1_dim = (i == 0) ? 560 : 512;

        snprintf(buf, sizeof(buf), "a.blk.%d.norm1.weight", i);
        L.norm1_w = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "a.blk.%d.norm1.bias", i);
        L.norm1_b = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "a.blk.%d.self_attn.linear_qkv.weight", i);
        L.qkv_w = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "a.blk.%d.self_attn.linear_qkv.bias", i);
        L.qkv_b = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "a.blk.%d.self_attn.fsmn.weight", i);
        L.fsmn_w = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "a.blk.%d.self_attn.linear_out.weight", i);
        L.out_w = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "a.blk.%d.self_attn.linear_out.bias", i);
        L.out_b = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "a.blk.%d.norm2.weight", i);
        L.norm2_w = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "a.blk.%d.norm2.bias", i);
        L.norm2_b = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "a.blk.%d.feed_forward.w1.weight", i);
        L.ffn_w1 = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "a.blk.%d.feed_forward.w1.bias", i);
        L.ffn_b1 = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "a.blk.%d.feed_forward.w2.weight", i);
        L.ffn_w2 = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "a.blk.%d.feed_forward.w2.bias", i);
        L.ffn_b2 = require_tensor(gc, buf);
    }

    au->after_norm_w = require_tensor(gc, "a.after_norm.weight");
    au->after_norm_b = require_tensor(gc, "a.after_norm.bias");

    // tp.blk.0..19
    for (int i = 0; i < FUN_ASR_ENC_TP_LAYERS; i++) {
        auto & L = au->tp_blk[i];
        L.norm1_dim = 512;

        snprintf(buf, sizeof(buf), "a.tp.blk.%d.norm1.weight", i);
        L.norm1_w = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "a.tp.blk.%d.norm1.bias", i);
        L.norm1_b = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "a.tp.blk.%d.self_attn.linear_qkv.weight", i);
        L.qkv_w = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "a.tp.blk.%d.self_attn.linear_qkv.bias", i);
        L.qkv_b = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "a.tp.blk.%d.self_attn.fsmn.weight", i);
        L.fsmn_w = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "a.tp.blk.%d.self_attn.linear_out.weight", i);
        L.out_w = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "a.tp.blk.%d.self_attn.linear_out.bias", i);
        L.out_b = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "a.tp.blk.%d.norm2.weight", i);
        L.norm2_w = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "a.tp.blk.%d.norm2.bias", i);
        L.norm2_b = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "a.tp.blk.%d.feed_forward.w1.weight", i);
        L.ffn_w1 = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "a.tp.blk.%d.feed_forward.w1.bias", i);
        L.ffn_b1 = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "a.tp.blk.%d.feed_forward.w2.weight", i);
        L.ffn_w2 = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "a.tp.blk.%d.feed_forward.w2.bias", i);
        L.ffn_b2 = require_tensor(gc, buf);
    }

    au->tp_norm_w = require_tensor(gc, "a.tp_norm.weight");
    au->tp_norm_b = require_tensor(gc, "a.tp_norm.bias");

    // MLP projector
    au->proj_w1 = require_tensor(gc, "mm.a.linear1.weight");
    au->proj_b1 = require_tensor(gc, "mm.a.linear1.bias");
    au->proj_w2 = require_tensor(gc, "mm.a.linear2.weight");
    au->proj_b2 = require_tensor(gc, "mm.a.linear2.bias");

    // Adaptor blocks
    for (int i = 0; i < 2; i++) {
        auto & A = au->ada[i];
        snprintf(buf, sizeof(buf), "mm.a.blk.%d.norm1.weight", i);
        A.norm1_w = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "mm.a.blk.%d.norm1.bias", i);
        A.norm1_b = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "mm.a.blk.%d.self_attn.linear_q.weight", i);
        A.q_w = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "mm.a.blk.%d.self_attn.linear_q.bias", i);
        A.q_b = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "mm.a.blk.%d.self_attn.linear_k.weight", i);
        A.k_w = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "mm.a.blk.%d.self_attn.linear_k.bias", i);
        A.k_b = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "mm.a.blk.%d.self_attn.linear_v.weight", i);
        A.v_w = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "mm.a.blk.%d.self_attn.linear_v.bias", i);
        A.v_b = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "mm.a.blk.%d.self_attn.linear_out.weight", i);
        A.o_w = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "mm.a.blk.%d.self_attn.linear_out.bias", i);
        A.o_b = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "mm.a.blk.%d.norm2.weight", i);
        A.norm2_w = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "mm.a.blk.%d.norm2.bias", i);
        A.norm2_b = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "mm.a.blk.%d.feed_forward.w1.weight", i);
        A.ffn_w1 = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "mm.a.blk.%d.feed_forward.w1.bias", i);
        A.ffn_b1 = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "mm.a.blk.%d.feed_forward.w2.weight", i);
        A.ffn_w2 = require_tensor(gc, buf);
        snprintf(buf, sizeof(buf), "mm.a.blk.%d.feed_forward.w2.bias", i);
        A.ffn_b2 = require_tensor(gc, buf);
    }
    return true;
}

// ============================================================================
// WAV loading
// ============================================================================

static bool load_wav(const std::string & path, std::vector<float> & out) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) return false;

    char riff[4]; fin.read(riff, 4);
    if (memcmp(riff, "RIFF", 4) != 0) return false;
    uint32_t chunk_size; fin.read(reinterpret_cast<char*>(&chunk_size), 4);
    char wave[4]; fin.read(wave, 4);
    if (memcmp(wave, "WAVE", 4) != 0) return false;

    uint16_t audio_format = 0, channels = 0, bits = 0;
    uint32_t sample_rate = 0, data_size = 0;
    bool found_fmt = false, found_data = false;

    while (fin.good() && !(found_fmt && found_data)) {
        char id[4]; fin.read(id, 4);
        uint32_t sz; fin.read(reinterpret_cast<char*>(&sz), 4);
        if (!fin.good()) break;
        if (memcmp(id, "fmt ", 4) == 0) {
            fin.read(reinterpret_cast<char*>(&audio_format), 2);
            fin.read(reinterpret_cast<char*>(&channels), 2);
            fin.read(reinterpret_cast<char*>(&sample_rate), 4);
            uint32_t br; fin.read(reinterpret_cast<char*>(&br), 4);
            uint16_t ba; fin.read(reinterpret_cast<char*>(&ba), 2);
            fin.read(reinterpret_cast<char*>(&bits), 2);
            if (sz > 16) fin.seekg(sz - 16, std::ios::cur);
            found_fmt = true;
        } else if (memcmp(id, "data", 4) == 0) {
            data_size = sz;
            found_data = true;
        } else {
            fin.seekg(sz, std::ios::cur);
        }
    }

    if (!found_fmt || !found_data) return false;
    if (audio_format != 1 && audio_format != 3) return false;
    if (!fin.good()) return false;

    const int32_t n_total = data_size / (bits / 8);
    const int32_t n_samples = n_total / channels;

    if (audio_format == 1 && bits == 16) {
        std::vector<int16_t> raw(n_total);
        fin.read(reinterpret_cast<char*>(raw.data()), data_size);
        out.resize(n_samples);
        for (int32_t i = 0; i < n_samples; i++) {
            float sum = 0.0f;
            for (int32_t c = 0; c < channels; c++) sum += raw[i * channels + c];
            out[i] = sum / (channels * 32768.0f);
        }
    } else if (audio_format == 3 && bits == 32) {
        std::vector<float> raw(n_total);
        fin.read(reinterpret_cast<char*>(raw.data()), data_size);
        out.resize(n_samples);
        for (int32_t i = 0; i < n_samples; i++) {
            float sum = 0.0f;
            for (int32_t c = 0; c < channels; c++) sum += raw[i * channels + c];
            out[i] = sum / channels;
        }
    } else {
        return false;
    }
    return true;
}

// ============================================================================
// Mel filterbank frontend (HTK-style, kaldi-compatible)
// ============================================================================

static float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

// Pre-compute windowed DFT basis: hamming[n] * exp(-j2πkn/N) for 400-point rfft → 201 bins.
// Baking the window into the basis turns the per-frame inner loop into a pure dot product
// that auto-vectorizes well with SSE/AVX.
struct DFTPlan {
    // wcos[k * N_FFT + n] = hamming[n] * cos(2πkn/N)
    // wsin[k * N_FFT + n] = hamming[n] * sin(2πkn/N)
    std::vector<float> wcos; // [N_FREQ * N_FFT]
    std::vector<float> wsin;
};

static const DFTPlan & get_dft_plan() {
    static DFTPlan plan = [] {
        DFTPlan p;
        // Periodic Hamming window (torchaudio periodic=True: /N not /(N-1))
        std::vector<float> hamming(N_FFT);
        for (int n = 0; n < N_FFT; n++) {
            hamming[n] = 0.54f - 0.46f * cosf(2.0f * PI * n / N_FFT);
        }
        p.wcos.resize((size_t)N_FREQ * N_FFT);
        p.wsin.resize((size_t)N_FREQ * N_FFT);
        for (int k = 0; k < N_FREQ; k++) {
            for (int n = 0; n < N_FFT; n++) {
                const float angle = 2.0f * PI * k * n / N_FFT;
                p.wcos[k * N_FFT + n] = hamming[n] * cosf(angle);
                p.wsin[k * N_FFT + n] = hamming[n] * sinf(angle);
            }
        }
        return p;
    }();
    return plan;
}

// Compute mel filterbank features (matches torchaudio/FunASR Python pipeline)
// Input: audio [n_samples] (already upscaled ×32768)
// Output: fbank [n_frames, n_mel] (frame-major)
static void compute_fbank(
    const std::vector<float> & audio,
    std::vector<float>       & fbank_out,
    int32_t                  * out_frames)
{
    const int32_t n_samples   = (int32_t)audio.size();
    const int32_t frame_len   = FUN_ASR_FRAME_LEN;   // 400
    const int32_t frame_shift = FUN_ASR_FRAME_SHIFT;  // 160
    const int32_t n_mel       = FUN_ASR_NUM_MEL_BINS; // 80
    const int32_t n_freq      = N_FREQ;               // 201
    const int32_t half_n_fft  = frame_len / 2;        // 200

    // Global mean subtraction (matches Python: audio = audio - np.mean(audio))
    float global_mean = 0.0f;
    for (int i = 0; i < n_samples; i++) global_mean += audio[i];
    global_mean /= n_samples;

    // Global pre-emphasis (matches Python: audio[1:] - 0.97 * audio[:-1])
    std::vector<float> audio_pe(n_samples);
    audio_pe[0] = audio[0] - global_mean;
    for (int i = 1; i < n_samples; i++) {
        audio_pe[i] = (audio[i] - global_mean) - 0.97f * (audio[i - 1] - global_mean);
    }

    // Center padding: pad half_n_fft zeros on each side (matches torchaudio center=True)
    const int32_t padded_len = half_n_fft + n_samples + half_n_fft;
    std::vector<float> padded(padded_len, 0.0f);
    std::copy(audio_pe.begin(), audio_pe.end(), padded.begin() + half_n_fft);

    // Frame count with center padding
    const int32_t n_frames = 1 + (padded_len - frame_len) / frame_shift;
    *out_frames = n_frames;
    if (n_frames == 0) return;

    // Mel filterbank (matches torchaudio: linspace(0, sr//2, n_fft//2+1))
    std::vector<float> mel_filters(n_freq * n_mel, 0.0f);
    {
        std::vector<float> all_freqs(n_freq);
        for (int k = 0; k < n_freq; k++) {
            all_freqs[k] = (float)k * (float)(FUN_ASR_SAMPLE_RATE / 2) / (n_freq - 1);
        }
        const float low_mel  = hz_to_mel(MEL_LOW_HZ);
        const float high_mel = hz_to_mel(MEL_HIGH_HZ);
        std::vector<float> f_pts(n_mel + 2);
        for (int m = 0; m <= n_mel + 1; m++) {
            float mel_m = low_mel + (high_mel - low_mel) * m / (n_mel + 1);
            f_pts[m] = mel_to_hz(mel_m);
        }
        std::vector<float> f_diff(n_mel + 1);
        for (int m = 0; m < n_mel + 1; m++) f_diff[m] = f_pts[m + 1] - f_pts[m];
        for (int m = 0; m < n_mel; m++) {
            for (int k = 0; k < n_freq; k++) {
                float down = (all_freqs[k] - f_pts[m]) / f_diff[m];
                float up   = (f_pts[m + 2] - all_freqs[k]) / f_diff[m + 1];
                mel_filters[k * n_mel + m] = std::max(0.0f, std::min(down, up));
            }
        }
    }

    fbank_out.resize((size_t)n_mel * n_frames);
    const DFTPlan & plan = get_dft_plan();

    #pragma omp parallel for schedule(static)
    for (int32_t fr = 0; fr < n_frames; fr++) {
        const float * frame = padded.data() + fr * frame_shift;
        float power[N_FREQ];

        // Windowed DFT via pre-baked basis (pure dot products → auto-vectorizes)
        for (int k = 0; k < N_FREQ; k++) {
            const float * wc = plan.wcos.data() + (size_t)k * N_FFT;
            const float * ws = plan.wsin.data() + (size_t)k * N_FFT;
            float re = 0.0f, im = 0.0f;
            for (int n = 0; n < N_FFT; n++) {
                const float s = frame[n];
                re += s * wc[n];
                im -= s * ws[n];
            }
            power[k] = re * re + im * im;
        }

        // Apply mel filters and log
        float * row = fbank_out.data() + (size_t)fr * n_mel;
        for (int m = 0; m < n_mel; m++) row[m] = 0.0f;
        for (int k = 0; k < N_FREQ; k++) {
            const float * w = mel_filters.data() + k * n_mel;
            for (int m = 0; m < n_mel; m++) row[m] += w[m] * power[k];
        }
        for (int m = 0; m < n_mel; m++) {
            row[m] = logf(row[m] + 1e-7f);
        }
    }
}

// LFR: stack m=7 frames with stride n=6, output [T', 560]
// Matches Python apply_lfr: left-pad (m-1)//2 copies of frame[0], T'=ceil(n_frames/n)
// fbank input: [n_frames, 80] (frame-major)
// output: [T', 560] (frame-major, each row = 7 stacked mel frames)
static void apply_lfr(
    const std::vector<float> & fbank,  // [n_frames, 80]
    int32_t                    n_frames,
    std::vector<float>       & lfr_out,
    int32_t                  * out_T)
{
    const int m = FUN_ASR_LFR_M;  // 7
    const int n = FUN_ASR_LFR_N;  // 6
    const int mel = FUN_ASR_NUM_MEL_BINS; // 80
    const int out_dim = mel * m;   // 560
    const int left_pad = (m - 1) / 2;  // 3

    if (n_frames == 0) { *out_T = 0; return; }

    // T' = ceil(n_frames / n)
    const int32_t T = (n_frames + n - 1) / n;
    *out_T = T;
    lfr_out.resize((size_t)T * out_dim);

    // padded length = left_pad + n_frames; frame index in padded = i -> real frame max(0, i - left_pad)
    // right-pad by clamping to last frame
    const int32_t padded_len = left_pad + n_frames;
    for (int32_t t = 0; t < T; t++) {
        float * dst = lfr_out.data() + (size_t)t * out_dim;
        for (int k = 0; k < m; k++) {
            int32_t padded_idx = t * n + k;  // index in padded sequence
            int32_t src_frame;
            if (padded_idx < left_pad) {
                src_frame = 0;  // left padding: repeat frame[0]
            } else {
                src_frame = padded_idx - left_pad;
                if (src_frame >= n_frames) src_frame = n_frames - 1;  // right clamp
            }
            const float * src = fbank.data() + (size_t)src_frame * mel;
            std::copy(src, src + mel, dst + k * mel);
        }
    }
}

// Cast to F32 if not already (needed for element-wise ops with F16 audio weights)
static ggml_tensor * e32(ggml_context * gctx, ggml_tensor * t) {
    return (t->type == GGML_TYPE_F32) ? t : ggml_cast(gctx, t, GGML_TYPE_F32);
}

// ============================================================================
// FSMN depthwise convolution (manual 11-loop accumulation)
// fsmn_w: [11, 1, dim] F16 weight
// x: [dim, T] F32 input (already in graph)
// Returns [dim, T] F32 output
// ============================================================================

static ggml_tensor * build_fsmn(
    ggml_context * gctx,
    ggml_tensor  * x,       // [dim, T]
    ggml_tensor  * fsmn_w,  // [11, 1, dim] F16
    int32_t        dim,
    int32_t        T)
{
    // Depthwise 1D conv via conv_2d_dw: each feature dim is a channel
    // Transpose x [dim, T] → [T, dim] contiguous, then reshape to [T, 1, dim, 1]
    ggml_tensor * xt = ggml_cont(gctx, ggml_transpose(gctx, x)); // [T, dim] contiguous
    ggml_tensor * x4 = ggml_reshape_4d(gctx, xt, T, 1, dim, 1);  // [W=T, H=1, C=dim, N=1]

    // Kernel [11, 1, dim] → [KW=11, KH=1, 1, C=dim]
    ggml_tensor * fw = ggml_reshape_4d(gctx, fsmn_w, 11, 1, 1, dim);

    ggml_tensor * conv = ggml_conv_2d_dw(gctx, fw, x4,
        /*s0=*/1, /*s1=*/1, /*p0=*/5, /*p1=*/0, /*d0=*/1, /*d1=*/1);
    // conv: [T, 1, dim, 1]

    // Reshape to [T, dim], transpose back to [dim, T]
    conv = ggml_cont(gctx, ggml_transpose(gctx,
        ggml_reshape_2d(gctx, conv, T, dim))); // [dim, T]

    return ggml_add(gctx, conv, x);
}

// ============================================================================
// SenseVoice encoder graph builder
// ============================================================================

static ggml_cgraph * build_encoder_graph(
    fun_asr_ctx    * ctx,
    ggml_context   * gctx,
    const float    * lfr_data,  // [T', 560] row-major
    int32_t          T)
{
    fun_asr_audio * au = ctx->audio;

    ggml_cgraph * gf = ggml_new_graph_custom(gctx, 32768, false);

    // Upload LFR input [560, T'] in GGML column-major
    ggml_tensor * inp = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, FUN_ASR_ENC_INPUT_DIM, T);
    ggml_set_name(inp, "enc_input");

    // Transpose lfr_data from [T', 560] to [560, T'] before set (done after alloc)
    // We'll use ggml_transpose in the graph
    // Actually inp shape is [560, T], lfr_data is row-major [T, 560]:
    // set as [560, T] → need to upload transposed
    // We'll handle the upload after graph allocation

    // Scale input by sqrt(d_model) before positional encoding (matches Python: x * 512**0.5)
    inp = ggml_scale(gctx, inp, sqrtf((float)FUN_ASR_ENC_DIM));

    // Add SinusoidalPositionEncoder: inp += pos_enc(positions 1..T, depth=560)
    {
        const int depth = FUN_ASR_ENC_INPUT_DIM; // 560
        ggml_tensor * pos_enc_t = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, depth, T);
        ggml_set_name(pos_enc_t, "pos_enc");
        inp = ggml_add(gctx, inp, pos_enc_t);
    }

    // blk.0 (input dim=560)
    ggml_tensor * x;
    {
        auto & L = au->blk[0];
        // LayerNorm(x_in, norm1) over dim=560
        ggml_tensor * xn = ggml_norm(gctx, inp, FUN_ASR_ENC_NORM_EPS);      // [560, T]
        xn = ggml_add(gctx, ggml_mul(gctx, xn, e32(gctx, L.norm1_w)), e32(gctx, L.norm1_b));

        // SANM: QKV projection [560, 1536]
        ggml_tensor * qkv = ggml_mul_mat(gctx, L.qkv_w, xn);               // [1536, T]
        qkv = ggml_add(gctx, qkv, e32(gctx, L.qkv_b));

        // Split into Q [512, T], K [512, T], V [512, T]; make contiguous for reshape
        ggml_tensor * Q = ggml_cont(gctx, ggml_view_2d(gctx, qkv, FUN_ASR_ENC_DIM, T,
            qkv->nb[1], 0));
        ggml_tensor * K = ggml_cont(gctx, ggml_view_2d(gctx, qkv, FUN_ASR_ENC_DIM, T,
            qkv->nb[1], (size_t)FUN_ASR_ENC_DIM * qkv->nb[0]));
        ggml_tensor * V = ggml_cont(gctx, ggml_view_2d(gctx, qkv, FUN_ASR_ENC_DIM, T,
            qkv->nb[1], (size_t)2 * FUN_ASR_ENC_DIM * qkv->nb[0]));

        // Reshape → permute → cont for flash_attn_ext: [head_dim, T, n_heads]
        const int head_dim = FUN_ASR_ENC_DIM / FUN_ASR_ENC_HEADS; // 128
        Q = ggml_cont(gctx, ggml_permute(gctx,
            ggml_reshape_3d(gctx, Q, head_dim, FUN_ASR_ENC_HEADS, T), 0, 2, 1, 3));
        K = ggml_cont(gctx, ggml_permute(gctx,
            ggml_reshape_3d(gctx, K, head_dim, FUN_ASR_ENC_HEADS, T), 0, 2, 1, 3));
        V = ggml_cont(gctx, ggml_permute(gctx,
            ggml_reshape_3d(gctx, V, head_dim, FUN_ASR_ENC_HEADS, T), 0, 2, 1, 3));

        // Flash attention (full self-attention, no mask)
        const float scale = 1.0f / sqrtf((float)head_dim);
        ggml_tensor * mha = ggml_flash_attn_ext(gctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        // Output: [head_dim, T, n_heads] → permute(0,2,1) → [head_dim, n_heads, T] → reshape to [512, T]
        // FA output: [head_dim, n_heads, T] — direct reshape to [n_heads*head_dim, T]
        mha = ggml_reshape_2d(gctx, mha, FUN_ASR_ENC_DIM, T);

        // FSMN on V (raw V before permute for attention), then residual internally
        ggml_tensor * V_raw = ggml_cont(gctx, ggml_view_2d(gctx, qkv, FUN_ASR_ENC_DIM, T,
            qkv->nb[1], (size_t)2 * FUN_ASR_ENC_DIM * qkv->nb[0]));
        ggml_tensor * fsmn = build_fsmn(gctx, V_raw, L.fsmn_w, FUN_ASR_ENC_DIM, T);

        // linear_out(mha) + fsmn  [matches: att_outs = linear_out(attn@V); return att_outs + fsmn_memory]
        ggml_tensor * out = ggml_mul_mat(gctx, L.out_w, mha);              // [512, T]
        out = ggml_add(gctx, out, e32(gctx, L.out_b));
        out = ggml_add(gctx, out, fsmn);

        // No residual for blk.0 (dim change 560→512)
        ggml_tensor * h_attn = out;

        // norm2 + FFN with residual
        ggml_tensor * xn2 = ggml_norm(gctx, h_attn, FUN_ASR_ENC_NORM_EPS);
        xn2 = ggml_add(gctx, ggml_mul(gctx, xn2, e32(gctx, L.norm2_w)), e32(gctx, L.norm2_b));
        ggml_tensor * ffn = ggml_mul_mat(gctx, L.ffn_w1, xn2);             // [2048, T]
        ffn = ggml_add(gctx, ffn, e32(gctx, L.ffn_b1));
        ffn = ggml_relu(gctx, ffn);
        ffn = ggml_mul_mat(gctx, L.ffn_w2, ffn);                            // [512, T]
        ffn = ggml_add(gctx, ffn, e32(gctx, L.ffn_b2));

        x = ggml_add(gctx, h_attn, ffn);
    }


    // blk.1..49 (dim=512)
    for (int i = 1; i < FUN_ASR_ENC_LAYERS; i++) {
        auto & L = au->blk[i];

        ggml_tensor * xn = ggml_norm(gctx, x, FUN_ASR_ENC_NORM_EPS);
        xn = ggml_add(gctx, ggml_mul(gctx, xn, e32(gctx, L.norm1_w)), e32(gctx, L.norm1_b));

        ggml_tensor * qkv = ggml_mul_mat(gctx, L.qkv_w, xn);               // [1536, T]
        qkv = ggml_add(gctx, qkv, e32(gctx, L.qkv_b));

        const int head_dim = FUN_ASR_ENC_DIM / FUN_ASR_ENC_HEADS;
        ggml_tensor * Q = ggml_cont(gctx, ggml_view_2d(gctx, qkv, FUN_ASR_ENC_DIM, T, qkv->nb[1], 0));
        ggml_tensor * K = ggml_cont(gctx, ggml_view_2d(gctx, qkv, FUN_ASR_ENC_DIM, T, qkv->nb[1],
            (size_t)FUN_ASR_ENC_DIM * qkv->nb[0]));
        ggml_tensor * V = ggml_cont(gctx, ggml_view_2d(gctx, qkv, FUN_ASR_ENC_DIM, T, qkv->nb[1],
            (size_t)2 * FUN_ASR_ENC_DIM * qkv->nb[0]));

        Q = ggml_cont(gctx, ggml_permute(gctx,
            ggml_reshape_3d(gctx, Q, head_dim, FUN_ASR_ENC_HEADS, T), 0, 2, 1, 3));
        K = ggml_cont(gctx, ggml_permute(gctx,
            ggml_reshape_3d(gctx, K, head_dim, FUN_ASR_ENC_HEADS, T), 0, 2, 1, 3));
        V = ggml_cont(gctx, ggml_permute(gctx,
            ggml_reshape_3d(gctx, V, head_dim, FUN_ASR_ENC_HEADS, T), 0, 2, 1, 3));

        const float scale = 1.0f / sqrtf((float)head_dim);
        ggml_tensor * mha = ggml_flash_attn_ext(gctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        // FA output: [head_dim, n_heads, T] — direct reshape to [n_heads*head_dim, T]
        mha = ggml_reshape_2d(gctx, mha, FUN_ASR_ENC_DIM, T);

        ggml_tensor * V_raw = ggml_cont(gctx, ggml_view_2d(gctx, qkv, FUN_ASR_ENC_DIM, T,
            qkv->nb[1], (size_t)2 * FUN_ASR_ENC_DIM * qkv->nb[0]));
        ggml_tensor * fsmn = build_fsmn(gctx, V_raw, L.fsmn_w, FUN_ASR_ENC_DIM, T);

        ggml_tensor * out = ggml_mul_mat(gctx, L.out_w, mha);
        out = ggml_add(gctx, out, e32(gctx, L.out_b));
        out = ggml_add(gctx, out, fsmn);

        ggml_tensor * h_attn = ggml_add(gctx, x, out);   // residual

        ggml_tensor * xn2 = ggml_norm(gctx, h_attn, FUN_ASR_ENC_NORM_EPS);
        xn2 = ggml_add(gctx, ggml_mul(gctx, xn2, e32(gctx, L.norm2_w)), e32(gctx, L.norm2_b));
        ggml_tensor * ffn = ggml_mul_mat(gctx, L.ffn_w1, xn2);
        ffn = ggml_add(gctx, ffn, e32(gctx, L.ffn_b1));
        ffn = ggml_relu(gctx, ffn);
        ffn = ggml_mul_mat(gctx, L.ffn_w2, ffn);
        ffn = ggml_add(gctx, ffn, e32(gctx, L.ffn_b2));

        x = ggml_add(gctx, h_attn, ffn);
    }

    // after_norm
    x = ggml_norm(gctx, x, FUN_ASR_ENC_NORM_EPS);
    x = ggml_add(gctx, ggml_mul(gctx, x, e32(gctx, au->after_norm_w)), e32(gctx, au->after_norm_b));

    // tp.blk.0..19
    for (int i = 0; i < FUN_ASR_ENC_TP_LAYERS; i++) {
        auto & L = au->tp_blk[i];

        ggml_tensor * xn = ggml_norm(gctx, x, FUN_ASR_ENC_NORM_EPS);
        xn = ggml_add(gctx, ggml_mul(gctx, xn, e32(gctx, L.norm1_w)), e32(gctx, L.norm1_b));

        ggml_tensor * qkv = ggml_mul_mat(gctx, L.qkv_w, xn);
        qkv = ggml_add(gctx, qkv, e32(gctx, L.qkv_b));

        const int head_dim = FUN_ASR_ENC_DIM / FUN_ASR_ENC_HEADS;
        ggml_tensor * Q = ggml_cont(gctx, ggml_view_2d(gctx, qkv, FUN_ASR_ENC_DIM, T, qkv->nb[1], 0));
        ggml_tensor * K = ggml_cont(gctx, ggml_view_2d(gctx, qkv, FUN_ASR_ENC_DIM, T, qkv->nb[1],
            (size_t)FUN_ASR_ENC_DIM * qkv->nb[0]));
        ggml_tensor * V = ggml_cont(gctx, ggml_view_2d(gctx, qkv, FUN_ASR_ENC_DIM, T, qkv->nb[1],
            (size_t)2 * FUN_ASR_ENC_DIM * qkv->nb[0]));

        Q = ggml_cont(gctx, ggml_permute(gctx,
            ggml_reshape_3d(gctx, Q, head_dim, FUN_ASR_ENC_HEADS, T), 0, 2, 1, 3));
        K = ggml_cont(gctx, ggml_permute(gctx,
            ggml_reshape_3d(gctx, K, head_dim, FUN_ASR_ENC_HEADS, T), 0, 2, 1, 3));
        V = ggml_cont(gctx, ggml_permute(gctx,
            ggml_reshape_3d(gctx, V, head_dim, FUN_ASR_ENC_HEADS, T), 0, 2, 1, 3));

        const float scale = 1.0f / sqrtf((float)head_dim);
        ggml_tensor * mha = ggml_flash_attn_ext(gctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        // FA output: [head_dim, n_heads, T] — direct reshape to [n_heads*head_dim, T]
        mha = ggml_reshape_2d(gctx, mha, FUN_ASR_ENC_DIM, T);

        // FSMN on raw V, linear_out(mha) + fsmn(V)
        ggml_tensor * V_raw = ggml_cont(gctx, ggml_view_2d(gctx, qkv, FUN_ASR_ENC_DIM, T,
            qkv->nb[1], (size_t)2 * FUN_ASR_ENC_DIM * qkv->nb[0]));
        ggml_tensor * fsmn = build_fsmn(gctx, V_raw, L.fsmn_w, FUN_ASR_ENC_DIM, T);

        ggml_tensor * out = ggml_mul_mat(gctx, L.out_w, mha);
        out = ggml_add(gctx, out, e32(gctx, L.out_b));
        out = ggml_add(gctx, out, fsmn);

        ggml_tensor * h_attn = ggml_add(gctx, x, out);

        ggml_tensor * xn2 = ggml_norm(gctx, h_attn, FUN_ASR_ENC_NORM_EPS);
        xn2 = ggml_add(gctx, ggml_mul(gctx, xn2, e32(gctx, L.norm2_w)), e32(gctx, L.norm2_b));
        ggml_tensor * ffn = ggml_mul_mat(gctx, L.ffn_w1, xn2);
        ffn = ggml_add(gctx, ffn, e32(gctx, L.ffn_b1));
        ffn = ggml_relu(gctx, ffn);
        ffn = ggml_mul_mat(gctx, L.ffn_w2, ffn);
        ffn = ggml_add(gctx, ffn, e32(gctx, L.ffn_b2));

        x = ggml_add(gctx, h_attn, ffn);
    }

    // tp_norm
    x = ggml_norm(gctx, x, FUN_ASR_ENC_NORM_EPS);
    x = ggml_add(gctx, ggml_mul(gctx, x, e32(gctx, au->tp_norm_w)), e32(gctx, au->tp_norm_b));

    // Copy result to persistent enc_out [512, MAX_ENC_SEQ]
    ggml_tensor * enc_view = ggml_view_2d(gctx, au->enc_out,
        FUN_ASR_ENC_DIM, T, au->enc_out->nb[1], 0);
    ggml_build_forward_expand(gf, ggml_cpy(gctx, x, enc_view));

    return gf;
}

// ============================================================================
// MLP Projector + Adaptor graph builder
// ============================================================================

static ggml_cgraph * build_projector_graph(
    fun_asr_ctx  * ctx,
    ggml_context * gctx,
    int32_t        T)
{
    fun_asr_audio * au = ctx->audio;
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, 2048, false);

    // Read encoder output [512, T]
    ggml_tensor * enc = ggml_view_2d(gctx, au->enc_out,
        FUN_ASR_ENC_DIM, T, au->enc_out->nb[1], 0);

    // MLP projector: [512→2048] ReLU [2048→1024]
    ggml_tensor * h = ggml_mul_mat(gctx, au->proj_w1, enc);  // [2048, T]
    h = ggml_add(gctx, h, e32(gctx, au->proj_b1));
    h = ggml_relu(gctx, h);
    h = ggml_mul_mat(gctx, au->proj_w2, h);                  // [1024, T]
    h = ggml_add(gctx, h, e32(gctx, au->proj_b2));

    // Adaptor blocks (2 transformer layers, dim=1024, 8 heads, head_dim=128)
    const int ada_dim = FUN_ASR_LLM_DIM;     // 1024
    const int ada_heads = 8;
    const int ada_head_dim = ada_dim / ada_heads; // 128

    for (int i = 0; i < 2; i++) {
        auto & A = au->ada[i];

        // LayerNorm (eps≈1e-12)
        ggml_tensor * xn = ggml_norm(gctx, h, 1e-12f);
        xn = ggml_add(gctx, ggml_mul(gctx, xn, e32(gctx, A.norm1_w)), e32(gctx, A.norm1_b));

        // Separate Q, K, V projections [1024, 1024]
        ggml_tensor * Q = ggml_mul_mat(gctx, A.q_w, xn);  // [1024, T]
        Q = ggml_add(gctx, Q, e32(gctx, A.q_b));
        ggml_tensor * K = ggml_mul_mat(gctx, A.k_w, xn);
        K = ggml_add(gctx, K, e32(gctx, A.k_b));
        ggml_tensor * V = ggml_mul_mat(gctx, A.v_w, xn);
        V = ggml_add(gctx, V, e32(gctx, A.v_b));

        // Reshape for attention: [head_dim, T, n_heads]
        Q = ggml_cont(gctx, ggml_permute(gctx,
            ggml_reshape_3d(gctx, Q, ada_head_dim, ada_heads, T), 0, 2, 1, 3));
        K = ggml_cont(gctx, ggml_permute(gctx,
            ggml_reshape_3d(gctx, K, ada_head_dim, ada_heads, T), 0, 2, 1, 3));
        V = ggml_cont(gctx, ggml_permute(gctx,
            ggml_reshape_3d(gctx, V, ada_head_dim, ada_heads, T), 0, 2, 1, 3));

        // Global attention (no mask, no RoPE)
        const float scale = 1.0f / sqrtf((float)ada_head_dim);
        ggml_tensor * attn = ggml_flash_attn_ext(gctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        // FA output: [head_dim, n_heads, T] — direct reshape
        attn = ggml_reshape_2d(gctx, attn, ada_dim, T);

        // Output projection
        ggml_tensor * out = ggml_mul_mat(gctx, A.o_w, attn);
        out = ggml_add(gctx, out, e32(gctx, A.o_b));

        h = ggml_add(gctx, h, out);  // residual

        // FFN sublayer
        ggml_tensor * xn2 = ggml_norm(gctx, h, 1e-12f);
        xn2 = ggml_add(gctx, ggml_mul(gctx, xn2, e32(gctx, A.norm2_w)), e32(gctx, A.norm2_b));

        ggml_tensor * ffn = ggml_mul_mat(gctx, A.ffn_w1, xn2);  // [256, T]
        ffn = ggml_add(gctx, ffn, e32(gctx, A.ffn_b1));
        ffn = ggml_relu(gctx, ffn);
        ffn = ggml_mul_mat(gctx, A.ffn_w2, ffn);                // [1024, T]
        ffn = ggml_add(gctx, ffn, e32(gctx, A.ffn_b2));

        h = ggml_add(gctx, h, ffn);  // residual
    }

    // Copy to persistent audio_tokens [1024, MAX_ENC_SEQ]
    ggml_tensor * tok_view = ggml_view_2d(gctx, au->audio_tokens,
        FUN_ASR_LLM_DIM, T, au->audio_tokens->nb[1], 0);
    ggml_build_forward_expand(gf, ggml_cpy(gctx, h, tok_view));

    return gf;
}

// ============================================================================
// BPE Tokenizer
// ============================================================================

// GPT-2 bytes-to-unicode table: maps each byte value to a UTF-8 encoded unicode codepoint.
// Printable bytes (33-126, 161-172, 174-255) map to themselves; others map to chr(256+n).
static std::string g_byte_enc[256];
static std::unordered_map<std::string, uint8_t> g_byte_dec;
static bool        g_byte_enc_inited = false;

static void init_byte_enc() {
    if (g_byte_enc_inited) return;
    bool printable[256] = {};
    for (int b = 33;  b <= 126; b++) printable[b] = true;
    for (int b = 161; b <= 172; b++) printable[b] = true;
    for (int b = 174; b <= 255; b++) printable[b] = true;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        uint32_t cp = printable[b] ? (uint32_t)b : (uint32_t)(256 + n++);
        char buf[4]; int len;
        if (cp < 0x80)        { buf[0]=(char)cp; len=1; }
        else if (cp < 0x800)  { buf[0]=(char)(0xC0|(cp>>6)); buf[1]=(char)(0x80|(cp&0x3F)); len=2; }
        else                  { buf[0]=(char)(0xE0|(cp>>12)); buf[1]=(char)(0x80|((cp>>6)&0x3F)); buf[2]=(char)(0x80|(cp&0x3F)); len=3; }
        g_byte_enc[b] = std::string(buf, len);
        g_byte_dec[g_byte_enc[b]] = (uint8_t)b;
    }
    g_byte_enc_inited = true;
}

// Split UTF-8 string into individual UTF-8 character strings
static std::vector<std::string> utf8_chars(const std::string & s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size(); ) {
        uint8_t c = (uint8_t)s[i];
        size_t len = (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
        out.push_back(s.substr(i, std::min(len, s.size() - i)));
        i += len;
    }
    return out;
}

// BPE encode a single word (pre-tokenized unit, already has Ġ prefix if needed)
static std::vector<int32_t> bpe_word(
    const fun_asr_llm & llm,
    const std::string & word)
{
    // Try direct lookup first (common for single-token words)
    {
        auto it = llm.token_to_id.find(word);
        if (it != llm.token_to_id.end()) return { it->second };
    }

    // Split into UTF-8 characters
    auto syms = utf8_chars(word);
    if (syms.empty()) return {};

    // Iteratively apply merges (greedy, rank-based)
    while (syms.size() > 1) {
        int32_t best_rank = INT32_MAX;
        size_t  best_pos  = SIZE_MAX;

        for (size_t i = 0; i + 1 < syms.size(); i++) {
            std::string pair = syms[i] + " " + syms[i + 1];
            auto it = llm.merge_ranks.find(pair);
            if (it != llm.merge_ranks.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos  = i;
            }
        }
        if (best_pos == SIZE_MAX) break;

        std::string merged = syms[best_pos] + syms[best_pos + 1];
        syms.erase(syms.begin() + (ptrdiff_t)best_pos + 1);
        syms[best_pos] = std::move(merged);
    }

    // Convert symbols to IDs
    std::vector<int32_t> ids;
    ids.reserve(syms.size());
    for (const auto & s : syms) {
        auto it = llm.token_to_id.find(s);
        if (it != llm.token_to_id.end()) {
            ids.push_back(it->second);
        } else {
            fprintf(stderr, "fun_asr: unknown token '%s'\n", s.c_str());
        }
    }
    return ids;
}

// Tokenize text using GPT-2 BPE with Qwen2 pre-tokenizer
static std::vector<int32_t> tokenize(const fun_asr_llm & llm, const std::string & text) {
    init_byte_enc();
    std::vector<int32_t> result;
    const size_t n = text.size();
    size_t i = 0;
    bool prev_space = false;

    static const std::string NEWLINE_TOK = "\xc4\x8a";  // Ċ (U+010A)
    static const std::string SPACE_TOK   = "\xc4\xa0";  // Ġ (U+0120)

    while (i < n) {
        // Try to match a special token starting with '<'
        if (text[i] == '<') {
            size_t end = text.find('>', i + 1);
            if (end != std::string::npos) {
                std::string special = text.substr(i, end - i + 1);
                auto it = llm.token_to_id.find(special);
                if (it != llm.token_to_id.end()) {
                    result.push_back(it->second);
                    i = end + 1;
                    prev_space = false;
                    continue;
                }
            }
        }

        // Newline
        if (text[i] == '\n') {
            auto it = llm.token_to_id.find(NEWLINE_TOK);
            if (it != llm.token_to_id.end()) {
                result.push_back(it->second);
            }
            i++;
            prev_space = false;
            continue;
        }

        // Space
        if (text[i] == ' ') {
            prev_space = true;
            i++;
            continue;
        }

        // Collect a word
        std::string word;
        if (std::isalpha((unsigned char)text[i])) {
            while (i < n && std::isalpha((unsigned char)text[i])) word += text[i++];
        } else if (std::isdigit((unsigned char)text[i])) {
            while (i < n && std::isdigit((unsigned char)text[i])) word += text[i++];
        } else if ((unsigned char)text[i] >= 0x80) {
            // Collect the entire run of non-ASCII codepoints (e.g., CJK sequences like 语音)
            // and apply GPT-2 byte encoding to each raw byte.
            while (i < n && (unsigned char)text[i] >= 0x80) {
                uint8_t c = (uint8_t)text[i];
                size_t clen = (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
                for (size_t k = 0; k < clen && i < n; k++) {
                    word += g_byte_enc[(unsigned char)text[i++]];
                }
            }
        } else {
            word = std::string(1, text[i++]);
        }

        if (prev_space) word = SPACE_TOK + word;
        prev_space = false;

        auto ids = bpe_word(llm, word);
        result.insert(result.end(), ids.begin(), ids.end());
    }
    return result;
}

// Decode token IDs to UTF-8 text
static std::string decode_tokens(const fun_asr_llm & llm, const std::vector<int32_t> & ids) {
    init_byte_enc();

    std::string out;
    for (int32_t id : ids) {
        if (id < 0 || id >= (int32_t)llm.vocab.size()) continue;
        if (id == FUN_ASR_LLM_EOS) break;
        const std::string & tok = llm.vocab[id];

        std::string bytes;
        for (const std::string & ch : utf8_chars(tok)) {
            auto it = g_byte_dec.find(ch);
            if (it != g_byte_dec.end()) {
                bytes.push_back((char)it->second);
            } else {
                bytes += ch;
            }
        }
        out += bytes;
    }
    return out;
}

static bool has_repetitive_tail(
    const std::vector<int32_t> & ids,
    size_t window = 30,
    size_t max_unique = 3) {
    if (ids.size() < window) return false;

    std::unordered_set<int32_t> uniq;
    const size_t start = ids.size() - window;
    for (size_t i = start; i < ids.size(); ++i) {
        uniq.insert(ids[i]);
        if (uniq.size() > max_unique) return false;
    }
    return true;
}

static bool is_stop_token(int32_t token_id) {
    return token_id == FUN_ASR_LLM_EOS || token_id == FUN_ASR_LLM_STOP_TOKEN;
}

static bool has_sentence_punctuation(const std::string & text) {
    static const char * kMarks[] = {
        "。", "！", "？", "，", "、", "；", "：",
        ".", "!", "?", ",", ";", ":"
    };
    for (const char * mark : kMarks) {
        if (text.find(mark) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static int32_t sample_from_logits(
    const std::vector<float> & logits,
    float temperature,
    int32_t top_k,
    std::mt19937 & rng) {
    if (logits.empty()) return FUN_ASR_LLM_EOS;

    std::vector<int32_t> candidates(logits.size());
    std::iota(candidates.begin(), candidates.end(), 0);

    const size_t k = std::min<size_t>(
        std::max<int32_t>(1, top_k),
        (int32_t)candidates.size());
    std::partial_sort(
        candidates.begin(), candidates.begin() + (ptrdiff_t)k, candidates.end(),
        [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
    candidates.resize(k);

    if (temperature <= 0.0f || k == 1) {
        return candidates[0];
    }

    float max_logit = -std::numeric_limits<float>::infinity();
    for (int32_t id : candidates) {
        max_logit = std::max(max_logit, logits[id]);
    }

    std::vector<double> weights;
    weights.reserve(candidates.size());
    for (int32_t id : candidates) {
        const double scaled = (double)(logits[id] - max_logit) / (double)temperature;
        weights.push_back(std::exp(scaled));
    }

    std::discrete_distribution<int32_t> dist(weights.begin(), weights.end());
    return candidates[(size_t)dist(rng)];
}

// ============================================================================
// KV cache shift-left eviction
// ============================================================================

static void kv_shift_left(fun_asr_ctx * ctx) {
    const int32_t window   = FUN_ASR_KV_WINDOW;   // 2048
    const int32_t n_layers = FUN_ASR_LLM_LAYERS;  // 28
    const size_t  row_bytes   = ctx->k_cache->nb[1];
    const size_t  layer_stride = ctx->k_cache->nb[2];
    const size_t  total_bytes = ggml_nbytes(ctx->k_cache);

    std::vector<uint8_t> k_buf(total_bytes), v_buf(total_bytes);
    ggml_backend_tensor_get(ctx->k_cache, k_buf.data(), 0, total_bytes);
    ggml_backend_tensor_get(ctx->v_cache, v_buf.data(), 0, total_bytes);

    for (int l = 0; l < n_layers; l++) {
        uint8_t * kb = k_buf.data() + (size_t)l * layer_stride;
        uint8_t * vb = v_buf.data() + (size_t)l * layer_stride;
        memmove(kb, kb + row_bytes, (size_t)(window - 1) * row_bytes);
        memmove(vb, vb + row_bytes, (size_t)(window - 1) * row_bytes);
        memset(kb + (size_t)(window - 1) * row_bytes, 0, row_bytes);
        memset(vb + (size_t)(window - 1) * row_bytes, 0, row_bytes);
    }

    ggml_backend_tensor_set(ctx->k_cache, k_buf.data(), 0, total_bytes);
    ggml_backend_tensor_set(ctx->v_cache, v_buf.data(), 0, total_bytes);
    // kv_used stays at window after shift
}

// ============================================================================
// Qwen3 decoder layer
// ============================================================================

static ggml_tensor * build_qwen3_layer(
    fun_asr_ctx  * ctx,
    ggml_context * gctx,
    ggml_cgraph  * gf,
    ggml_tensor  * x,         // [1024, n_tokens]
    ggml_tensor  * positions,  // [n_tokens] int32
    int32_t        layer_idx,
    int32_t        n_tokens,
    int32_t        kv_offset,  // where to write K,V in cache
    ggml_tensor  * attn_mask)  // [n_kv, n_tokens] F16 or nullptr
{
    auto & L = ctx->llm->layers[layer_idx];
    const int32_t kv_dim     = FUN_ASR_KV_DIM;          // 1024
    const int32_t n_q_heads  = FUN_ASR_LLM_N_Q_HEADS;   // 16
    const int32_t n_kv_heads = FUN_ASR_LLM_N_KV_HEADS;  // 8
    const int32_t head_dim   = FUN_ASR_LLM_HEAD_DIM;    // 128
    const int32_t n_kv       = kv_offset + n_tokens;

    // Pre-attn RMSNorm
    ggml_tensor * residual = x;
    ggml_tensor * xn = ggml_rms_norm(gctx, x, FUN_ASR_LLM_NORM_EPS);
    xn = ggml_mul(gctx, xn, L.attn_norm);

    // Q, K, V projections (no bias)
    ggml_tensor * Q = ggml_mul_mat(gctx, L.attn_q, xn); // [2048, n_tokens]
    ggml_tensor * K = ggml_mul_mat(gctx, L.attn_k, xn); // [1024, n_tokens]
    ggml_tensor * V = ggml_mul_mat(gctx, L.attn_v, xn); // [1024, n_tokens]

    // QK-Norm: per-head RMSNorm on Q then K (applied before RoPE)
    Q = ggml_reshape_3d(gctx, Q, head_dim, n_q_heads, n_tokens);
    Q = ggml_rms_norm(gctx, Q, FUN_ASR_LLM_NORM_EPS);
    Q = ggml_mul(gctx, Q, L.attn_q_norm);

    K = ggml_reshape_3d(gctx, K, head_dim, n_kv_heads, n_tokens);
    K = ggml_rms_norm(gctx, K, FUN_ASR_LLM_NORM_EPS);
    K = ggml_mul(gctx, K, L.attn_k_norm);

    // RoPE (NEOX mode=2, freq_base=1e6) — Q,K already [head_dim, n_heads, n_tokens]
    Q = ggml_rope_ext(gctx, Q, positions, nullptr,
        head_dim, GGML_ROPE_TYPE_NEOX, 0,
        FUN_ASR_LLM_ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_ext(gctx, K, positions, nullptr,
        head_dim, GGML_ROPE_TYPE_NEOX, 0,
        FUN_ASR_LLM_ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // Flatten back for KV cache write
    Q = ggml_cont(gctx, ggml_reshape_2d(gctx, Q, n_q_heads * head_dim, n_tokens));
    K = ggml_cont(gctx, ggml_reshape_2d(gctx, K, kv_dim, n_tokens));

    // Write K, V to cache at [kv_offset .. kv_offset+n_tokens-1]
    {
        ggml_tensor * k_slice = ggml_view_2d(gctx, ctx->k_cache,
            kv_dim, n_tokens, ctx->k_cache->nb[1],
            (size_t)layer_idx * ctx->k_cache->nb[2] + (size_t)kv_offset * ctx->k_cache->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(gctx, K, k_slice));

        ggml_tensor * v_slice = ggml_view_2d(gctx, ctx->v_cache,
            kv_dim, n_tokens, ctx->v_cache->nb[1],
            (size_t)layer_idx * ctx->v_cache->nb[2] + (size_t)kv_offset * ctx->v_cache->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(gctx, V, v_slice));
    }

    // Read full K, V from cache [kv_dim, n_kv]
    ggml_tensor * k_full = ggml_view_2d(gctx, ctx->k_cache,
        kv_dim, n_kv, ctx->k_cache->nb[1],
        (size_t)layer_idx * ctx->k_cache->nb[2]);
    ggml_tensor * v_full = ggml_view_2d(gctx, ctx->v_cache,
        kv_dim, n_kv, ctx->v_cache->nb[1],
        (size_t)layer_idx * ctx->v_cache->nb[2]);

    // Flash attention (GQA: 16 Q heads, 8 KV heads)
    ggml_tensor * q3 = ggml_permute(gctx,
        ggml_reshape_3d(gctx, Q, head_dim, n_q_heads, n_tokens), 0, 2, 1, 3);
    ggml_tensor * k3 = ggml_permute(gctx,
        ggml_reshape_3d(gctx, k_full, head_dim, n_kv_heads, n_kv), 0, 2, 1, 3);
    ggml_tensor * v3 = ggml_permute(gctx,
        ggml_reshape_3d(gctx, v_full, head_dim, n_kv_heads, n_kv), 0, 2, 1, 3);

    const float scale = 1.0f / sqrtf((float)head_dim);
    ggml_tensor * attn_out = ggml_flash_attn_ext(gctx, q3, k3, v3, attn_mask, scale, 0.0f, 0.0f);
    // FA output: [head_dim, n_heads, n_tokens] — direct reshape
    attn_out = ggml_reshape_2d(gctx, attn_out, n_q_heads * head_dim, n_tokens);

    // Output projection + residual
    ggml_tensor * out = ggml_mul_mat(gctx, L.attn_out, attn_out); // [1024, n_tokens]
    x = ggml_add(gctx, residual, out);

    // Pre-FFN RMSNorm
    residual = x;
    ggml_tensor * xn2 = ggml_rms_norm(gctx, x, FUN_ASR_LLM_NORM_EPS);
    xn2 = ggml_mul(gctx, xn2, L.ffn_norm);

    // SwiGLU FFN: gate = silu(gate_w @ x), up = up_w @ x, out = down_w @ (gate * up)
    ggml_tensor * gate = ggml_silu(gctx, ggml_mul_mat(gctx, L.ffn_gate, xn2)); // [3072, n_tokens]
    ggml_tensor * up   = ggml_mul_mat(gctx, L.ffn_up, xn2);                    // [3072, n_tokens]
    ggml_tensor * ffn  = ggml_mul_mat(gctx, L.ffn_down, ggml_mul(gctx, gate, up)); // [1024, n_tokens]

    return ggml_add(gctx, residual, ffn);
}

// ============================================================================
// Helper: run a graph via the scheduler
// ============================================================================

static bool run_graph(fun_asr_ctx * ctx, ggml_context * gctx, ggml_cgraph * gf) {
    ggml_backend_sched_reset(g_ext.sched);
    if (!ggml_backend_sched_alloc_graph(g_ext.sched, gf)) {
        fprintf(stderr, "fun_asr: sched alloc failed\n");
        return false;
    }
    if (ggml_backend_sched_graph_compute(g_ext.sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "fun_asr: graph compute failed\n");
        return false;
    }
    return true;
}

// ============================================================================
// Encoder inference
// ============================================================================

static bool run_encoder(fun_asr_ctx * ctx, const float * lfr_data, int32_t T) {
    const size_t meta_size = ggml_tensor_overhead() * 32768 + ggml_graph_overhead_custom(32768, false);
    std::vector<uint8_t> meta_buf(meta_size);
    ggml_init_params ip = { meta_size, meta_buf.data(), true };
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) return false;

    ggml_cgraph * gf = build_encoder_graph(ctx, gctx, lfr_data, T);

    if (!ggml_gallocr_alloc_graph(g_ext.galloc_enc, gf)) {
        fprintf(stderr, "fun_asr: encoder alloc_graph failed\n");
        ggml_free(gctx); return false;
    }

    ggml_tensor * inp = ggml_graph_get_tensor(gf, "enc_input");
    if (inp) {
        ggml_backend_tensor_set(inp, lfr_data, 0,
            (size_t)FUN_ASR_ENC_INPUT_DIM * T * sizeof(float));
    }

    ggml_tensor * pos_enc_t = ggml_graph_get_tensor(gf, "pos_enc");
    if (pos_enc_t) {
        const int depth = FUN_ASR_ENC_INPUT_DIM;
        std::vector<float> pe((size_t)depth * T);
        const float log_ts_inc = logf(10000.0f) / (depth / 2 - 1);
        for (int t = 0; t < T; t++) {
            float pos = (float)(t + 1);
            for (int i = 0; i < depth / 2; i++) {
                float inv_ts = expf(-(float)i * log_ts_inc);
                float angle = pos * inv_ts;
                pe[i + t * depth]               = sinf(angle);
                pe[(i + depth/2) + t * depth]  = cosf(angle);
            }
        }
        ggml_backend_tensor_set(pos_enc_t, pe.data(), 0,
            (size_t)depth * T * sizeof(float));
    }

    bool ok = (ggml_backend_graph_compute(ctx->backend, gf) == GGML_STATUS_SUCCESS);
    if (!ok) fprintf(stderr, "fun_asr: encoder graph_compute failed\n");

    ggml_free(gctx);
    return ok;
}

// ============================================================================
// Projector inference
// ============================================================================

static bool run_projector(fun_asr_ctx * ctx, int32_t T) {
    const size_t meta_size = ggml_tensor_overhead() * 2048 + ggml_graph_overhead_custom(2048, false);
    std::vector<uint8_t> meta_buf(meta_size);
    ggml_init_params ip = { meta_size, meta_buf.data(), true };
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) return false;

    ggml_cgraph * gf = build_projector_graph(ctx, gctx, T);
    if (!ggml_gallocr_alloc_graph(g_ext.galloc_proj, gf)) {
        ggml_free(gctx); return false;
    }
    bool ok = (ggml_backend_graph_compute(ctx->backend, gf) == GGML_STATUS_SUCCESS);
    ggml_free(gctx);
    return ok;
}

// ============================================================================
// Prefill: assemble embeddings + run all 28 Qwen3 layers
// ============================================================================

static int32_t run_prefill(
    fun_asr_ctx                  * ctx,
    const std::vector<int32_t>   & prefix_ids,
    int32_t                        T_audio,
    const std::vector<int32_t>   & suffix_ids)
{
    const int32_t n_prefix = (int32_t)prefix_ids.size();
    const int32_t n_suffix = (int32_t)suffix_ids.size();
    const int32_t n_total  = n_prefix + T_audio + n_suffix;
    const int32_t llm_dim  = FUN_ASR_LLM_DIM;

    if (n_total > FUN_ASR_KV_WINDOW) {
        fprintf(stderr, "fun_asr: prefill length %d exceeds KV window %d\n",
            n_total, FUN_ASR_KV_WINDOW);
        return -1;
    }

    // Dequantize prefix/suffix embeddings on CPU
    const ggml_tensor * embd_w = ctx->llm->tok_embd;
    const size_t row_size = ggml_row_size(embd_w->type, llm_dim);
    const ggml_type_traits * traits = ggml_get_type_traits(embd_w->type);

    std::vector<float> combined_emb((size_t)n_total * llm_dim);
    {
        std::vector<uint8_t> qrow(row_size);
        // Prefix embeddings
        for (int i = 0; i < n_prefix; i++) {
            ggml_backend_tensor_get(embd_w, qrow.data(), (size_t)prefix_ids[i] * row_size, row_size);
            traits->to_float(qrow.data(), combined_emb.data() + (size_t)i * llm_dim, llm_dim);
        }
        // Audio tokens: copy from GPU
        ggml_backend_tensor_get(ctx->audio->audio_tokens,
            combined_emb.data() + (size_t)n_prefix * llm_dim,
            0, (size_t)T_audio * llm_dim * sizeof(float));
        // Suffix embeddings
        for (int i = 0; i < n_suffix; i++) {
            ggml_backend_tensor_get(embd_w, qrow.data(), (size_t)suffix_ids[i] * row_size, row_size);
            traits->to_float(qrow.data(), combined_emb.data() + (size_t)(n_prefix + T_audio + i) * llm_dim, llm_dim);
        }
    }

    const size_t meta_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false);
    std::vector<uint8_t> meta_buf(meta_size);
    ggml_init_params ip = { meta_size, meta_buf.data(), true };
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) return -1;

    ggml_cgraph * gf = ggml_new_graph_custom(gctx, 8192, false);

    // Combined embedding input [1024, n_total] F32
    ggml_tensor * combined_t = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, llm_dim, n_total);
    ggml_set_name(combined_t, "combined_emb");

    ggml_tensor * pos_t = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_total);
    ggml_set_name(pos_t, "positions");

    ggml_tensor * mask_t = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, n_total, n_total);
    ggml_set_name(mask_t, "causal_mask");

    // Run 28 Qwen3 layers
    ggml_tensor * h = combined_t;
    for (int i = 0; i < FUN_ASR_LLM_LAYERS; i++) {
        h = build_qwen3_layer(ctx, gctx, gf, h, pos_t, i, n_total, 0, mask_t);
    }

    // Final norm (no logits needed from prefill)
    ggml_tensor * h_last = ggml_view_1d(gctx, h, llm_dim,
        (size_t)(n_total - 1) * h->nb[1]);
    h_last = ggml_rms_norm(gctx, h_last, FUN_ASR_LLM_NORM_EPS);
    ggml_build_forward_expand(gf, ggml_mul(gctx, h_last, ctx->llm->output_norm));

    if (!ggml_gallocr_alloc_graph(g_ext.galloc_pf, gf)) {
        ggml_free(gctx); return -1;
    }

    // Upload inputs
    ggml_backend_tensor_set(combined_t, combined_emb.data(), 0,
        (size_t)n_total * llm_dim * sizeof(float));

    std::vector<int32_t> pos_data(n_total);
    for (int i = 0; i < n_total; i++) pos_data[i] = i;
    ggml_backend_tensor_set(pos_t, pos_data.data(), 0, n_total * sizeof(int32_t));

    {
        std::vector<ggml_fp16_t> mask_data((size_t)n_total * n_total);
        for (int q = 0; q < n_total; q++) {
            for (int k = 0; k < n_total; k++) {
                mask_data[(size_t)q * n_total + k] =
                    ggml_fp32_to_fp16((k <= q) ? 0.0f : -INFINITY);
            }
        }
        ggml_backend_tensor_set(mask_t, mask_data.data(), 0,
            (size_t)n_total * n_total * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_graph_compute(ctx->backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(gctx); return -1;
    }

    ggml_free(gctx);
    ctx->kv_used = n_total;
    return n_total;
}

// ============================================================================
// Decode single token step
// ============================================================================

static int32_t run_decode_step(
    fun_asr_ctx * ctx,
    int32_t token_id,
    float temperature,
    int32_t top_k,
    std::mt19937 & rng) {
    const int32_t kv_used = ctx->kv_used;
    const int32_t llm_dim = FUN_ASR_LLM_DIM;
    const bool greedy = (temperature <= 0.0f);

    const size_t meta_size = ggml_tensor_overhead() * 2048 + ggml_graph_overhead_custom(2048, false);
    std::vector<uint8_t> meta_buf(meta_size);
    ggml_init_params ip = { meta_size, meta_buf.data(), true };
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) return FUN_ASR_LLM_EOS;

    ggml_cgraph * gf = ggml_new_graph_custom(gctx, 2048, false);

    // Embedding input: pre-dequantized on CPU, uploaded to GPU
    ggml_tensor * emb_t = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, llm_dim);
    ggml_set_name(emb_t, "emb_input");

    ggml_tensor * pos_t = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ggml_set_name(pos_t, "position");

    ggml_tensor * h = ggml_reshape_2d(gctx, emb_t, llm_dim, 1);
    for (int i = 0; i < FUN_ASR_LLM_LAYERS; i++) {
        h = build_qwen3_layer(ctx, gctx, gf, h, pos_t, i, 1, kv_used, nullptr);
    }

    ggml_tensor * h_flat = ggml_reshape_1d(gctx, h, llm_dim);
    h_flat = ggml_rms_norm(gctx, h_flat, FUN_ASR_LLM_NORM_EPS);
    h_flat = ggml_mul(gctx, h_flat, ctx->llm->output_norm);
    ggml_tensor * logits = ggml_mul_mat(gctx, ctx->llm->output_w, h_flat);

    ggml_tensor * out_t = nullptr;
    if (greedy) {
        ggml_tensor * logits_2d = ggml_reshape_2d(gctx, logits, FUN_ASR_LLM_VOCAB_SIZE, 1);
        out_t = ggml_argmax(gctx, logits_2d);
        ggml_set_name(out_t, "argmax_out");
        ggml_build_forward_expand(gf, out_t);
    } else {
        out_t = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, FUN_ASR_LLM_VOCAB_SIZE);
        ggml_set_name(out_t, "out_logits");
        ggml_build_forward_expand(gf, ggml_cpy(gctx, logits, out_t));
    }

    // Use lightweight graph allocator + direct CUDA compute (bypass scheduler)
    if (!ggml_gallocr_alloc_graph(g_ext.galloc_dec, gf)) {
        ggml_free(gctx); return FUN_ASR_LLM_EOS;
    }

    // Dequantize embedding row on CPU, upload to GPU
    {
        const ggml_tensor * embd_w = ctx->llm->tok_embd;
        const size_t row_size = ggml_row_size(embd_w->type, llm_dim);
        std::vector<uint8_t> qrow(row_size);
        ggml_backend_tensor_get(embd_w, qrow.data(), (size_t)token_id * row_size, row_size);
        float emb_f32[FUN_ASR_LLM_DIM];
        const ggml_type_traits * traits = ggml_get_type_traits(embd_w->type);
        traits->to_float(qrow.data(), emb_f32, llm_dim);
        ggml_backend_tensor_set(emb_t, emb_f32, 0, llm_dim * sizeof(float));
    }
    ggml_backend_tensor_set(pos_t, &kv_used, 0, sizeof(int32_t));

    if (ggml_backend_graph_compute(ctx->backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(gctx); return FUN_ASR_LLM_EOS;
    }

    int32_t result;
    if (greedy) {
        ggml_backend_tensor_get(out_t, &result, 0, sizeof(int32_t));
    } else {
        std::vector<float> host_logits(FUN_ASR_LLM_VOCAB_SIZE);
        ggml_backend_tensor_get(out_t, host_logits.data(), 0, host_logits.size() * sizeof(float));
        result = sample_from_logits(host_logits, temperature, top_k, rng);
    }

    ggml_free(gctx);
    return result;
}

// ============================================================================
// Public API: fun_asr_init
// ============================================================================

fun_asr_ctx * fun_asr_init(const std::string & model_dir, bool use_cuda) {
    fun_asr_ctx * ctx = new fun_asr_ctx();
    ctx->llm   = new fun_asr_llm();
    ctx->audio = new fun_asr_audio();

    // Initialize backends
#ifdef GGML_USE_CUDA
    if (use_cuda) {
        ctx->backend = ggml_backend_cuda_init(0);
        if (!ctx->backend) {
            fprintf(stderr, "fun_asr: CUDA backend init failed, falling back to CPU\n");
        }
    }
#endif
    if (!ctx->backend) {
        ctx->backend = ggml_backend_cpu_init();
    }
    if (!ctx->backend) {
        fprintf(stderr, "fun_asr: failed to init any backend\n");
        fun_asr_free(ctx); return nullptr;
    }

    g_ext.backend_cpu = ggml_backend_cpu_init();

    // Setup scheduler
    ggml_backend_t backends[2] = { ctx->backend, g_ext.backend_cpu };
    int n_backends = (ctx->backend == g_ext.backend_cpu) ? 1 : 2;
    const bool op_offload = (n_backends > 1);
    g_ext.sched = ggml_backend_sched_new(backends, nullptr, n_backends,
        GGML_DEFAULT_GRAPH_SIZE * 16, false, op_offload);
    if (!g_ext.sched) {
        fprintf(stderr, "fun_asr: failed to create scheduler\n");
        fun_asr_free(ctx); return nullptr;
    }

    // Load model weights
    const std::string llm_path   = model_dir + "/Fun-ASR-Nano-2512-llm-Q5_K_M.gguf";
    const std::string audio_path = model_dir + "/Fun-ASR-Nano-2512-audio-mmproj-mtmd-F16.gguf";

    fprintf(stderr, "fun_asr: loading LLM from %s\n", llm_path.c_str());
    if (!load_llm_weights(llm_path, ctx->backend, ctx)) {
        fun_asr_free(ctx); return nullptr;
    }

    fprintf(stderr, "fun_asr: loading audio encoder from %s\n", audio_path.c_str());
    if (!load_mmproj_weights(audio_path, ctx->backend, ctx)) {
        fun_asr_free(ctx); return nullptr;
    }

    // Allocate KV cache: [kv_dim, window, layers] × 2
    {
        ggml_init_params kvm = { ggml_tensor_overhead() * 4, nullptr, true };
        ggml_context * kv_gctx = ggml_init(kvm);

        ctx->k_cache = ggml_new_tensor_3d(kv_gctx, GGML_TYPE_F32,
            FUN_ASR_KV_DIM, FUN_ASR_KV_WINDOW, FUN_ASR_LLM_LAYERS);
        ctx->v_cache = ggml_new_tensor_3d(kv_gctx, GGML_TYPE_F32,
            FUN_ASR_KV_DIM, FUN_ASR_KV_WINDOW, FUN_ASR_LLM_LAYERS);

        ctx->kv_buf = ggml_backend_alloc_ctx_tensors(kv_gctx, ctx->backend);
        if (!ctx->kv_buf) {
            fprintf(stderr, "fun_asr: failed to allocate KV cache\n");
            fun_asr_free(ctx); return nullptr;
        }
        // Zero-initialize KV cache
        const size_t kv_bytes = ggml_nbytes(ctx->k_cache);
        std::vector<uint8_t> zeros(kv_bytes, 0);
        ggml_backend_tensor_set(ctx->k_cache, zeros.data(), 0, kv_bytes);
        ggml_backend_tensor_set(ctx->v_cache, zeros.data(), 0, kv_bytes);
        ctx->kv_used = 0;

        // Note: kv_gctx is not freed (tensors live in it). This is a deliberate leak
        // matching the persistent pattern. We store it nowhere since llm_gctx is separate.
    }

    // Allocate persistent audio output buffers [enc_out, audio_tokens]
    {
        const size_t buf_overhead = ggml_tensor_overhead() * 4;
        ggml_init_params omp = { buf_overhead, nullptr, true }; // GGML-owned heap buffer
        ctx->audio->out_gctx = ggml_init(omp);

        ctx->audio->enc_out = ggml_new_tensor_2d(ctx->audio->out_gctx,
            GGML_TYPE_F32, FUN_ASR_ENC_DIM, MAX_ENC_SEQ);
        ctx->audio->audio_tokens = ggml_new_tensor_2d(ctx->audio->out_gctx,
            GGML_TYPE_F32, FUN_ASR_LLM_DIM, MAX_ENC_SEQ);

        ctx->audio->out_buf = ggml_backend_alloc_ctx_tensors(
            ctx->audio->out_gctx, ctx->backend);
        if (!ctx->audio->out_buf) {
            fprintf(stderr, "fun_asr: failed to allocate output buffers\n");
            fun_asr_free(ctx); return nullptr;
        }
    }

    // Initialize decode graph allocator (bypasses scheduler for faster decode)
    {
        g_ext.galloc_dec = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(ctx->backend));

        // Build a worst-case decode graph to reserve buffer space
        const size_t msz = ggml_tensor_overhead() * 2048 + ggml_graph_overhead_custom(2048, false);
        std::vector<uint8_t> mbuf(msz);
        ggml_init_params mp = { msz, mbuf.data(), true };
        ggml_context * mgctx = ggml_init(mp);
        ggml_cgraph * mgf = ggml_new_graph_custom(mgctx, 2048, false);

        ggml_tensor * memb = ggml_new_tensor_1d(mgctx, GGML_TYPE_F32, FUN_ASR_LLM_DIM);
        ggml_tensor * mpos = ggml_new_tensor_1d(mgctx, GGML_TYPE_I32, 1);
        ggml_tensor * mh = ggml_reshape_2d(mgctx, memb, FUN_ASR_LLM_DIM, 1);
        for (int i = 0; i < FUN_ASR_LLM_LAYERS; i++) {
            mh = build_qwen3_layer(ctx, mgctx, mgf, mh, mpos, i, 1,
                FUN_ASR_KV_WINDOW - 1, nullptr);
        }
        mh = ggml_reshape_1d(mgctx, mh, FUN_ASR_LLM_DIM);
        mh = ggml_rms_norm(mgctx, mh, FUN_ASR_LLM_NORM_EPS);
        mh = ggml_mul(mgctx, mh, ctx->llm->output_norm);
        ggml_tensor * mlogits = ggml_mul_mat(mgctx, ctx->llm->output_w, mh);
        ggml_tensor * mlogits_2d = ggml_reshape_2d(mgctx, mlogits, FUN_ASR_LLM_VOCAB_SIZE, 1);
        ggml_build_forward_expand(mgf, ggml_argmax(mgctx, mlogits_2d));
        // Also account for sampling path (needs larger output buffer)
        ggml_tensor * mout = ggml_new_tensor_1d(mgctx, GGML_TYPE_F32, FUN_ASR_LLM_VOCAB_SIZE);
        ggml_build_forward_expand(mgf, ggml_cpy(mgctx, mlogits, mout));

        ggml_gallocr_reserve(g_ext.galloc_dec, mgf);
        ggml_free(mgctx);
    }

    // Initialize encoder graph allocator
    {
        g_ext.galloc_enc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(ctx->backend));
        const size_t msz = ggml_tensor_overhead() * 32768 + ggml_graph_overhead_custom(32768, false);
        std::vector<uint8_t> mbuf(msz);
        ggml_init_params mp = { msz, mbuf.data(), true };
        ggml_context * mgctx = ggml_init(mp);
        float dummy[FUN_ASR_ENC_INPUT_DIM * FUN_ASR_MAX_ENC_SEQ] = {};
        ggml_cgraph * mgf = build_encoder_graph(ctx, mgctx, dummy, FUN_ASR_MAX_ENC_SEQ);
        ggml_gallocr_reserve(g_ext.galloc_enc, mgf);
        ggml_free(mgctx);
    }

    // Initialize projector graph allocator
    {
        g_ext.galloc_proj = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(ctx->backend));
        const size_t msz = ggml_tensor_overhead() * 2048 + ggml_graph_overhead_custom(2048, false);
        std::vector<uint8_t> mbuf(msz);
        ggml_init_params mp = { msz, mbuf.data(), true };
        ggml_context * mgctx = ggml_init(mp);
        ggml_cgraph * mgf = build_projector_graph(ctx, mgctx, FUN_ASR_MAX_ENC_SEQ);
        ggml_gallocr_reserve(g_ext.galloc_proj, mgf);
        ggml_free(mgctx);
    }

    // Initialize prefill graph allocator (worst case: full KV window)
    {
        g_ext.galloc_pf = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(ctx->backend));
        const int max_pf = FUN_ASR_KV_WINDOW;
        const size_t msz = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false);
        std::vector<uint8_t> mbuf(msz);
        ggml_init_params mp = { msz, mbuf.data(), true };
        ggml_context * mgctx = ggml_init(mp);
        ggml_cgraph * mgf = ggml_new_graph_custom(mgctx, 8192, false);
        ggml_tensor * mc = ggml_new_tensor_2d(mgctx, GGML_TYPE_F32, FUN_ASR_LLM_DIM, max_pf);
        ggml_tensor * mp_t = ggml_new_tensor_1d(mgctx, GGML_TYPE_I32, max_pf);
        ggml_tensor * mm = ggml_new_tensor_2d(mgctx, GGML_TYPE_F16, max_pf, max_pf);
        ggml_tensor * mh = mc;
        for (int i = 0; i < FUN_ASR_LLM_LAYERS; i++) {
            mh = build_qwen3_layer(ctx, mgctx, mgf, mh, mp_t, i, max_pf, 0, mm);
        }
        ggml_tensor * ml = ggml_view_1d(mgctx, mh, FUN_ASR_LLM_DIM,
            (size_t)(max_pf - 1) * mh->nb[1]);
        ml = ggml_rms_norm(mgctx, ml, FUN_ASR_LLM_NORM_EPS);
        ggml_build_forward_expand(mgf, ggml_mul(mgctx, ml, ctx->llm->output_norm));
        ggml_gallocr_reserve(g_ext.galloc_pf, mgf);
        ggml_free(mgctx);
    }

    fprintf(stderr, "fun_asr: initialized successfully\n");
    return ctx;
}

// ============================================================================
// Public API: fun_asr_transcribe
// ============================================================================

std::string fun_asr_transcribe(
    fun_asr_ctx * ctx,
    const std::string & wav_path,
    int32_t max_tokens,
    const fun_asr_decode_options * options) {
    ctx->last_generated_tokens = 0;
    ctx->last_attempts_used = 0;
    ctx->last_repetition_retries = 0;
    ctx->last_punctuation_retries = 0;
    ctx->last_final_temperature = 0.0f;

    auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&](const char * tag) {
        auto dt = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "fun_asr: %s %.1f ms\n", tag, dt);
        t0 = std::chrono::steady_clock::now();
    };

    // 1. Load WAV
    std::vector<float> audio;
    if (!load_wav(wav_path, audio)) {
        fprintf(stderr, "fun_asr: failed to load WAV: %s\n", wav_path.c_str());
        return "";
    }
    elapsed("wav_load");

    // 2. Compute mel filterbank [n_frames, 80] (audio is already normalized [-1, 1])
    std::vector<float> fbank;
    int32_t n_frames = 0;
    compute_fbank(audio, fbank, &n_frames);
    if (n_frames == 0) {
        fprintf(stderr, "fun_asr: audio too short\n");
        return "";
    }
    elapsed("fbank");

    // 4. LFR stacking → [T', 560]
    std::vector<float> lfr;
    int32_t T_prime = 0;
    apply_lfr(fbank, n_frames, lfr, &T_prime);
    if (T_prime < 1) {
        fprintf(stderr, "fun_asr: LFR output empty\n");
        return "";
    }
    if (T_prime > MAX_ENC_SEQ) {
        T_prime = MAX_ENC_SEQ;
        fprintf(stderr, "fun_asr: truncating encoder input to %d frames\n", MAX_ENC_SEQ);
    }
    elapsed("lfr");

    // 5. SenseVoice encoder → enc_out [512, T']
    if (!run_encoder(ctx, lfr.data(), T_prime)) {
        fprintf(stderr, "fun_asr: encoder failed\n");
        return "";
    }
    elapsed("encoder");

    // 6. MLP projector + Adaptor → audio_tokens [1024, T']
    if (!run_projector(ctx, T_prime)) {
        fprintf(stderr, "fun_asr: projector failed\n");
        return "";
    }
    elapsed("projector");

    // 6b. Compute effective audio token count (matches Python's convolutional length formula)
    // The model was trained with this truncation applied to adaptor output
    {
        const int32_t T_mel_valid = (int32_t)(audio.size() / FUN_ASR_FRAME_SHIFT) + 1;
        const int32_t T_lfr_valid = (T_mel_valid + FUN_ASR_LFR_N - 1) / FUN_ASR_LFR_N;
        const int32_t olens_1 = 1 + (T_lfr_valid - 3 + 2) / 2;
        const int32_t target_len = (1 + (olens_1 - 3 + 2) / 2 - 1) / 2 + 1;
        T_prime = std::min(T_prime, std::max(target_len, 1));
    }

    // 7. Tokenize prompt
    const std::string prefix_text = build_prompt_prefix(options);
    const std::string suffix_text = build_prompt_suffix(options);

    std::vector<int32_t> prefix_ids = tokenize(*ctx->llm, prefix_text);
    std::vector<int32_t> suffix_ids = tokenize(*ctx->llm, suffix_text);

    const int32_t n_audio = T_prime;
    fprintf(stderr, "fun_asr: prefill prefix=%zu audio=%d suffix=%zu total=%zu\n",
        prefix_ids.size(), n_audio, suffix_ids.size(),
        prefix_ids.size() + n_audio + suffix_ids.size());

    const int32_t n_total = (int32_t)(prefix_ids.size() + T_prime + suffix_ids.size());
    const int32_t seed = options ? options->seed : -1;
    std::random_device rd;
    std::mt19937 rng(seed >= 0 ? (uint32_t)seed : rd());
    std::vector<int32_t> best_tokens;

    const float temperature_base = options ? options->temperature_base : 0.0f;
    const float temperature_step = options ? options->temperature_step : 0.3f;
    const int32_t top_k = options ? std::max<int32_t>(1, options->top_k) : 50;
    const int32_t retry_attempts = options ? std::max<int32_t>(1, options->retry_attempts) : 4;

    for (int attempt = 0; attempt < retry_attempts; ++attempt) {
        const float temperature = temperature_base + temperature_step * attempt;
        bool aborted = false;

        // 8. Prefill (all tokens except the last one)
        ctx->kv_used = 0;
        if (n_total > 1) {
            std::vector<int32_t> prefill_prefix = prefix_ids;
            std::vector<int32_t> prefill_suffix = suffix_ids;
            if (!prefill_suffix.empty()) {
                prefill_suffix.pop_back();
            }

            int32_t prefill_len = run_prefill(ctx, prefill_prefix, T_prime, prefill_suffix);
            if (prefill_len < 0) {
                fprintf(stderr, "fun_asr: prefill failed\n");
                return "";
            }
            if (attempt == 0) {
                elapsed("prefill");
            }
        }

        // 9. First decode step with the last suffix token at its correct position
        int32_t cur_token = suffix_ids.empty() ? FUN_ASR_LLM_EOS : suffix_ids.back();
        int32_t first_token = run_decode_step(
            ctx, cur_token, temperature, top_k, rng);
        ctx->kv_used++;

        if (attempt > 0) {
            fprintf(stderr, "fun_asr: retry attempt=%d temperature=%.1f\n",
                attempt + 1, temperature);
        }
        ctx->last_attempts_used = attempt + 1;
        ctx->last_final_temperature = temperature;

        if (is_stop_token(first_token)) {
            elapsed("decode");
            return "";
        }

        std::vector<int32_t> out_tokens;
        out_tokens.push_back(first_token);
        cur_token = first_token;
        elapsed("first_decode");

        // 10. Autoregressive decode loop
        auto t_decode_start = std::chrono::steady_clock::now();
        for (int step = 1; step < max_tokens; step++) {
            if (ctx->kv_used >= FUN_ASR_KV_WINDOW) {
                kv_shift_left(ctx);
            }

            int32_t next = run_decode_step(
                ctx, cur_token, temperature, top_k, rng);
            ctx->kv_used++;

            if (is_stop_token(next)) break;
            out_tokens.push_back(next);
            cur_token = next;

            if (has_repetitive_tail(out_tokens)) {
                fprintf(stderr,
                    "fun_asr: repetitive tail detected on attempt %d, retrying\n",
                    attempt + 1);
                ctx->last_repetition_retries++;
                aborted = true;
                break;
            }

            if (out_tokens.size() == 30 &&
                !has_sentence_punctuation(decode_tokens(*ctx->llm, out_tokens))) {
                fprintf(stderr,
                    "fun_asr: no punctuation in first 30 tokens on attempt %d, retrying\n",
                    attempt + 1);
                ctx->last_punctuation_retries++;
                aborted = true;
                break;
            }
        }
        {
            double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_decode_start).count();
            const int n_tok = (int)out_tokens.size();
            const double tps = n_tok / std::max(ms / 1000.0, 1e-6);
            fprintf(stderr, "fun_asr: decode %d tokens in %.1f ms (%.1f tok/s)\n",
                n_tok, ms, tps);
        }

        best_tokens = std::move(out_tokens);
        if (!aborted) break;
    }

    ctx->last_generated_tokens = (int32_t)best_tokens.size();
    return decode_tokens(*ctx->llm, best_tokens);
}

// ============================================================================
// Public API: fun_asr_free
// ============================================================================

void fun_asr_free(fun_asr_ctx * ctx) {
    if (!ctx) return;

    if (g_ext.galloc_pf)   { ggml_gallocr_free(g_ext.galloc_pf);   g_ext.galloc_pf = nullptr; }
    if (g_ext.galloc_proj) { ggml_gallocr_free(g_ext.galloc_proj); g_ext.galloc_proj = nullptr; }
    if (g_ext.galloc_enc)  { ggml_gallocr_free(g_ext.galloc_enc);  g_ext.galloc_enc = nullptr; }
    if (g_ext.galloc_dec)  { ggml_gallocr_free(g_ext.galloc_dec);  g_ext.galloc_dec = nullptr; }
    if (g_ext.sched) {
        ggml_backend_sched_free(g_ext.sched);
        g_ext.sched = nullptr;
    }
    if (g_ext.backend_cpu) {
        ggml_backend_free(g_ext.backend_cpu);
        g_ext.backend_cpu = nullptr;
    }

    if (ctx->audio) {
        if (ctx->audio->out_buf)  ggml_backend_buffer_free(ctx->audio->out_buf);
        if (ctx->audio->out_gctx) ggml_free(ctx->audio->out_gctx);
        delete ctx->audio;
        ctx->audio = nullptr;
    }
    if (ctx->kv_buf)   ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->audio_buf) ggml_backend_buffer_free(ctx->audio_buf);
    if (ctx->llm_buf)  ggml_backend_buffer_free(ctx->llm_buf);
    if (ctx->audio_gctx) ggml_free(ctx->audio_gctx);
    if (ctx->llm_gctx)   ggml_free(ctx->llm_gctx);
    if (ctx->backend)    ggml_backend_free(ctx->backend);

    delete ctx->llm;
    delete ctx;
}
