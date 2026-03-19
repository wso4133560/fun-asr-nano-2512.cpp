#include "fun_asr.h"
#include "gguf.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================================
// Internal constants
// ============================================================================

static constexpr float PI = 3.14159265358979323846f;
static constexpr int32_t N_FFT       = FUN_ASR_FRAME_LEN;      // 400
static constexpr int32_t N_FFT_PAD   = 512;                     // next power of 2
static constexpr int32_t N_FREQ      = N_FFT_PAD / 2 + 1;      // 257 (kaldi uses padded FFT size)
static constexpr float   MEL_LOW_HZ  = 20.0f;
static constexpr float   MEL_HIGH_HZ = 8000.0f; // kaldi nyquist
static constexpr int32_t MAX_ENC_SEQ = FUN_ASR_MAX_ENC_SEQ;    // 3000 (raw LFR frames)

static constexpr const char* PROMPT_PREFIX =
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    "\xe8\xaf\xad\xe9\x9f\xb3\xe8\xbd\xac\xe5\x86\x99\xef\xbc\x9a"; // 语音转写：
static constexpr const char* PROMPT_SUFFIX =
    "<|im_end|>\n<|im_start|>assistant\n";

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

// FFT state (radix-2, zero-padded to N_FFT_PAD=512)
struct FFTPlan {
    std::vector<uint32_t> bit_rev;
    std::vector<float>    tw_re, tw_im;
};

static const FFTPlan & get_fft_plan() {
    static FFTPlan plan = [] {
        FFTPlan p;
        const int N = N_FFT_PAD;
        const int log2N = __builtin_ctz((uint32_t)N);
        p.bit_rev.resize(N);
        for (int i = 0; i < N; i++) {
            uint32_t rev = 0, val = i;
            for (int b = 0; b < log2N; b++) { rev = (rev << 1) | (val & 1); val >>= 1; }
            p.bit_rev[i] = rev;
        }
        p.tw_re.resize(N / 2);
        p.tw_im.resize(N / 2);
        for (int i = 0; i < N / 2; i++) {
            const float a = -2.0f * PI * i / N;
            p.tw_re[i] = cosf(a);
            p.tw_im[i] = sinf(a);
        }
        return p;
    }();
    return plan;
}

static void fft_radix2(float * re, float * im, const FFTPlan & plan) {
    const int N = N_FFT_PAD;
    const int log2N = __builtin_ctz((uint32_t)N);
    for (int i = 0; i < N; i++) {
        int j = (int)plan.bit_rev[i];
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    for (int s = 1; s <= log2N; s++) {
        int m = 1 << s, half = m >> 1, step = N / m;
        for (int k = 0; k < N; k += m) {
            for (int j = 0; j < half; j++) {
                float wr = plan.tw_re[j * step], wi = plan.tw_im[j * step];
                float tr = wr * re[k+j+half] - wi * im[k+j+half];
                float ti = wr * im[k+j+half] + wi * re[k+j+half];
                re[k+j+half] = re[k+j] - tr; im[k+j+half] = im[k+j] - ti;
                re[k+j] += tr; im[k+j] += ti;
            }
        }
    }
}

// Compute mel filterbank features
// Input: audio [n_samples] (already upscaled ×32768)
// Output: fbank [n_mel, n_frames] stored as mel-major (row = mel bin)
static void compute_fbank(
    const std::vector<float> & audio,
    std::vector<float>       & fbank_out,
    int32_t                  * out_frames)
{
    const int32_t n_samples  = (int32_t)audio.size();
    const int32_t frame_len  = FUN_ASR_FRAME_LEN;  // 400
    const int32_t frame_shift = FUN_ASR_FRAME_SHIFT; // 160
    const int32_t n_mel      = FUN_ASR_NUM_MEL_BINS; // 80
    const int32_t n_freq     = N_FREQ;               // 201

    // snip_edges=True: n_frames = floor((n_samples - frame_len) / frame_shift) + 1
    const int32_t n_frames = (n_samples >= frame_len)
        ? (n_samples - frame_len) / frame_shift + 1 : 0;
    *out_frames = n_frames;
    if (n_frames == 0) return;

    // Pre-compute Hamming window
    std::vector<float> hamming(frame_len);
    for (int i = 0; i < frame_len; i++) {
        hamming[i] = 0.54f - 0.46f * cosf(2.0f * PI * i / (frame_len - 1));
    }

    // Pre-compute HTK mel filters: filters[k * n_mel + m] = weight for FFT bin k, mel m
    std::vector<float> mel_filters(n_freq * n_mel, 0.0f);
    {
        const float low_mel  = hz_to_mel(MEL_LOW_HZ);
        const float high_mel = hz_to_mel(MEL_HIGH_HZ);
        const float fft_res  = (float)FUN_ASR_SAMPLE_RATE / (2.0f * (n_freq - 1)); // Hz per bin

        std::vector<float> mel_pts(n_mel + 2), hz_pts(n_mel + 2);
        for (int m = 0; m <= n_mel + 1; m++) {
            mel_pts[m] = low_mel + (high_mel - low_mel) * m / (n_mel + 1);
            hz_pts[m]  = mel_to_hz(mel_pts[m]);
        }
        for (int m = 0; m < n_mel; m++) {
            float fl = hz_pts[m], fc = hz_pts[m+1], fr = hz_pts[m+2];
            for (int k = 0; k < n_freq; k++) {
                float f = k * fft_res;
                float v = 0.0f;
                if (f > fl && f < fc)       v = (f - fl) / (fc - fl);
                else if (f >= fc && f < fr) v = (fr - f) / (fr - fc);
                mel_filters[k * n_mel + m] = v;
            }
        }
    }

    fbank_out.resize((size_t)n_mel * n_frames, 0.0f);
    const FFTPlan & plan = get_fft_plan();
    std::vector<float> fft_re(N_FFT_PAD), fft_im(N_FFT_PAD);
    std::vector<float> power(n_freq);

    for (int32_t fr = 0; fr < n_frames; fr++) {
        const int32_t start = fr * frame_shift;
        // Remove DC offset
        float dc = 0.0f;
        for (int i = 0; i < frame_len; i++) dc += audio[start + i];
        dc /= frame_len;
        // Apply preemphasis (coeff=0.97, matching kaldi default), Hamming window + zero-pad
        std::fill(fft_re.begin(), fft_re.end(), 0.0f);
        std::fill(fft_im.begin(), fft_im.end(), 0.0f);
        fft_re[0] = (audio[start + 0] - dc) * hamming[0];
        for (int i = 1; i < frame_len; i++) {
            float cur = (audio[start + i] - dc) - 0.97f * (audio[start + i - 1] - dc);
            fft_re[i] = cur * hamming[i];
        }
        fft_radix2(fft_re.data(), fft_im.data(), plan);

        // Power spectrum
        for (int k = 0; k < n_freq; k++) {
            power[k] = fft_re[k] * fft_re[k] + fft_im[k] * fft_im[k];
        }

        // Apply mel filters and log
        float * row = fbank_out.data() + (size_t)fr * n_mel;
        std::fill(row, row + n_mel, 0.0f);
        for (int k = 0; k < n_freq; k++) {
            const float * w = mel_filters.data() + k * n_mel;
            for (int m = 0; m < n_mel; m++) row[m] += w[m] * power[k];
        }
        for (int m = 0; m < n_mel; m++) {
            row[m] = logf(std::max(row[m], 1e-10f));
        }
    }
    // fbank_out is [n_frames, n_mel] (frame-major)
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
    ggml_tensor  * fsmn_w,  // [11, 1, dim]
    int32_t        dim,
    int32_t        T)
{
    // Pad x along time dimension (dim 1): 5 zeros on each side → [dim, T+10]
    ggml_tensor * x_pad = ggml_pad_ext(gctx, x, 0, 0, 5, 5, 0, 0, 0, 0); // [dim, T+10]

    // Transpose fsmn_w [11, 1, dim] → contiguous [dim, 11]
    // permute(2, 0, 1, 3): new ne = [dim, 11, 1, 1]
    ggml_tensor * fw = ggml_cont(gctx, ggml_permute(gctx, fsmn_w, 2, 0, 1, 3));
    fw = ggml_reshape_2d(gctx, fw, dim, 11); // [dim, 11] contiguous

    // Convert weight to F32 (it's F16 in the audio model)
    fw = ggml_cast(gctx, fw, GGML_TYPE_F32); // [dim, 11] F32

    ggml_tensor * acc = nullptr;
    for (int k = 0; k < 11; k++) {
        // Weight for position k: column k of fw → [dim] F32
        ggml_tensor * w_k = ggml_view_1d(gctx, fw, dim,
            (size_t)k * fw->nb[1]); // [dim] F32
        // Reshape to [dim, 1] for broadcasting
        w_k = ggml_reshape_2d(gctx, w_k, dim, 1);

        // Input slice at position k: x_pad[:,k:k+T] → [dim, T]
        ggml_tensor * x_k = ggml_view_2d(gctx, x_pad, dim, T,
            x_pad->nb[1], (size_t)k * x_pad->nb[1]);

        // Elementwise multiply with broadcast: [dim, T] * [dim, 1] → [dim, T]
        ggml_tensor * contrib = ggml_mul(gctx, x_k, w_k);

        acc = (acc == nullptr) ? contrib : ggml_add(gctx, acc, contrib);
    }
    // FSMN internal residual: conv(x) + x  (matches forward_fsmn: x += inputs)
    return ggml_add(gctx, acc, x);
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
    ggml_backend_sched_set_tensor_backend(g_ext.sched, inp, ctx->backend);

    // Transpose lfr_data from [T', 560] to [560, T'] before set (done after alloc)
    // We'll use ggml_transpose in the graph
    // Actually inp shape is [560, T], lfr_data is row-major [T, 560]:
    // set as [560, T] → need to upload transposed
    // We'll handle the upload after graph allocation

    // Add SinusoidalPositionEncoder: inp += pos_enc(positions 1..T, depth=560)
    // Formula: inv_ts[i] = exp(-i * log(10000) / (depth/2-1)), i in [0, depth/2)
    //          pos_enc = [sin(pos * inv_ts), cos(pos * inv_ts)]
    {
        const int depth = FUN_ASR_ENC_INPUT_DIM; // 560
        // Build position encoding tensor [560, T] on CPU, then upload
        ggml_tensor * pos_enc_t = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, depth, T);
        ggml_set_name(pos_enc_t, "pos_enc");
        ggml_backend_sched_set_tensor_backend(g_ext.sched, pos_enc_t, g_ext.backend_cpu);
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
        mha = ggml_cont(gctx, ggml_permute(gctx, mha, 2, 0, 1, 3));
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
        ggml_set_name(x, "dbg_blk0");
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
        mha = ggml_cont(gctx, ggml_permute(gctx, mha, 2, 0, 1, 3));
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
        mha = ggml_cont(gctx, ggml_permute(gctx, mha, 2, 0, 1, 3));
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

    // MLP projector: [512→2048] SiLU [2048→1024]
    ggml_tensor * h = ggml_mul_mat(gctx, au->proj_w1, enc);  // [2048, T]
    h = ggml_add(gctx, h, e32(gctx, au->proj_b1));
    h = ggml_silu(gctx, h);
    h = ggml_mul_mat(gctx, au->proj_w2, h);                  // [1024, T]
    h = ggml_add(gctx, h, e32(gctx, au->proj_b2));

    // Adaptor blocks (2 transformer layers, dim=1024, 4 heads, head_dim=256)
    const int ada_dim = FUN_ASR_LLM_DIM;     // 1024
    const int ada_heads = 4;
    const int ada_head_dim = ada_dim / ada_heads; // 256

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
        attn = ggml_cont(gctx, ggml_permute(gctx, attn, 2, 0, 1, 3));
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
    std::string out;
    for (int32_t id : ids) {
        if (id < 0 || id >= (int32_t)llm.vocab.size()) continue;
        if (id == FUN_ASR_LLM_EOS) break;
        const std::string & tok = llm.vocab[id];
        // Replace Ġ (0xC4 0xA0) → space, Ċ (0xC4 0x8A) → newline
        std::string s;
        for (size_t i = 0; i < tok.size(); ) {
            uint8_t c = (uint8_t)tok[i];
            if (c == 0xC4 && i + 1 < tok.size()) {
                uint8_t c2 = (uint8_t)tok[i+1];
                if (c2 == 0xA0) { s += ' '; i += 2; continue; }
                if (c2 == 0x8A) { s += '\n'; i += 2; continue; }
            }
            s += tok[i++];
        }
        out += s;
    }
    return out;
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
    // Q: [2048, n_tokens] → reshape [128, 16, n_tokens] → rms_norm → scale → reshape back
    Q = ggml_reshape_3d(gctx, Q, head_dim, n_q_heads, n_tokens);
    Q = ggml_rms_norm(gctx, Q, FUN_ASR_LLM_NORM_EPS);
    Q = ggml_mul(gctx, Q, L.attn_q_norm);  // broadcast [128] with [128, 16, n_tokens]
    Q = ggml_cont(gctx, Q);
    Q = ggml_reshape_2d(gctx, Q, n_q_heads * head_dim, n_tokens); // [2048, n_tokens]

    K = ggml_reshape_3d(gctx, K, head_dim, n_kv_heads, n_tokens);
    K = ggml_rms_norm(gctx, K, FUN_ASR_LLM_NORM_EPS);
    K = ggml_mul(gctx, K, L.attn_k_norm);  // broadcast [128] with [128, 8, n_tokens]
    K = ggml_cont(gctx, K);
    K = ggml_reshape_2d(gctx, K, kv_dim, n_tokens);  // [1024, n_tokens]

    // Reshape Q, K for RoPE: [head_dim, n_heads, n_tokens]
    Q = ggml_reshape_3d(gctx, Q, head_dim, n_q_heads, n_tokens);
    K = ggml_reshape_3d(gctx, K, head_dim, n_kv_heads, n_tokens);

    // RoPE (NEOX mode=2, freq_base=1e6, head_dim=128)
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
    ggml_tensor * q3 = ggml_cont(gctx, ggml_permute(gctx,
        ggml_reshape_3d(gctx, Q, head_dim, n_q_heads, n_tokens), 0, 2, 1, 3));
    ggml_tensor * k3 = ggml_cont(gctx, ggml_permute(gctx,
        ggml_reshape_3d(gctx, k_full, head_dim, n_kv_heads, n_kv), 0, 2, 1, 3));
    ggml_tensor * v3 = ggml_cont(gctx, ggml_permute(gctx,
        ggml_reshape_3d(gctx, v_full, head_dim, n_kv_heads, n_kv), 0, 2, 1, 3));

    const float scale = 1.0f / sqrtf((float)head_dim);
    ggml_tensor * attn_out = ggml_flash_attn_ext(gctx, q3, k3, v3, attn_mask, scale, 0.0f, 0.0f);
    attn_out = ggml_cont(gctx, ggml_permute(gctx, attn_out, 2, 0, 1, 3));
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
    // Build graph
    const size_t meta_size = ggml_tensor_overhead() * 32768 + ggml_graph_overhead_custom(32768, false);
    std::vector<uint8_t> meta_buf(meta_size);
    ggml_init_params ip = { meta_size, meta_buf.data(), true };
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) return false;

    ggml_cgraph * gf = build_encoder_graph(ctx, gctx, lfr_data, T);

    ggml_backend_sched_reset(g_ext.sched);
    if (!ggml_backend_sched_alloc_graph(g_ext.sched, gf)) {
        fprintf(stderr, "fun_asr: encoder alloc_graph failed\n");
        ggml_free(gctx); return false;
    }

    // Upload input: lfr_data is [T, 560] row-major, need [560, T] column-major
    // The tensor inp in the graph is [560, T] (ne[0]=560, ne[1]=T)
    // We transpose on upload
    ggml_tensor * inp = ggml_graph_get_tensor(gf, "enc_input");
    if (inp) {
        // Transpose: lfr_data[t, d] → inp[d, t]
        std::vector<float> transposed((size_t)FUN_ASR_ENC_INPUT_DIM * T);
        for (int t = 0; t < T; t++) {
            for (int d = 0; d < FUN_ASR_ENC_INPUT_DIM; d++) {
                transposed[d * T + t] = lfr_data[t * FUN_ASR_ENC_INPUT_DIM + d];
            }
        }
        ggml_backend_tensor_set(inp, transposed.data(), 0,
            (size_t)FUN_ASR_ENC_INPUT_DIM * T * sizeof(float));
    }

    // Upload sinusoidal position encoding [560, T]
    ggml_tensor * pos_enc_t = ggml_graph_get_tensor(gf, "pos_enc");
    if (pos_enc_t) {
        const int depth = FUN_ASR_ENC_INPUT_DIM; // 560
        std::vector<float> pe((size_t)depth * T);
        const float log_ts_inc = logf(10000.0f) / (depth / 2 - 1);
        for (int t = 0; t < T; t++) {
            float pos = (float)(t + 1); // positions 1..T
            for (int i = 0; i < depth / 2; i++) {
                float inv_ts = expf(-(float)i * log_ts_inc);
                float angle = pos * inv_ts;
                // GGML tensor [depth, T]: element [d, t] is at index d + t*depth
                pe[i + t * depth]               = sinf(angle);
                pe[(i + depth/2) + t * depth]  = cosf(angle);
            }
        }
        ggml_backend_tensor_set(pos_enc_t, pe.data(), 0,
            (size_t)depth * T * sizeof(float));
    }

    bool ok = (ggml_backend_sched_graph_compute(g_ext.sched, gf) == GGML_STATUS_SUCCESS);
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
    bool ok = run_graph(ctx, gctx, gf);
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

    // Build the full prefill embedding on CPU, then upload
    // Prefix: token embedding lookup via CPU dequant (avoid building graph for embd lookup)
    // Instead, build a GGML graph that assembles everything

    const size_t meta_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false);
    std::vector<uint8_t> meta_buf(meta_size);
    ggml_init_params ip = { meta_size, meta_buf.data(), true };
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) return -1;

    ggml_cgraph * gf = ggml_new_graph_custom(gctx, 8192, false);

    // Build combined input embedding tensor as graph operations
    // prefix_ids_t: [n_prefix] int32
    ggml_tensor * pref_ids_t = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_prefix);
    ggml_set_name(pref_ids_t, "prefix_ids");
    ggml_backend_sched_set_tensor_backend(g_ext.sched, pref_ids_t, ctx->backend);

    ggml_tensor * suf_ids_t = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_suffix);
    ggml_set_name(suf_ids_t, "suffix_ids");
    ggml_backend_sched_set_tensor_backend(g_ext.sched, suf_ids_t, ctx->backend);

    // Positions: [n_total] int32
    ggml_tensor * pos_t = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_total);
    ggml_set_name(pos_t, "positions");
    ggml_backend_sched_set_tensor_backend(g_ext.sched, pos_t, ctx->backend);

    // Causal mask: [n_total, n_total] F16
    ggml_tensor * mask_t = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, n_total, n_total);
    ggml_set_name(mask_t, "causal_mask");
    ggml_backend_sched_set_tensor_backend(g_ext.sched, mask_t, ctx->backend);

    // Prefix embeddings
    ggml_tensor * pref_emb = ggml_get_rows(gctx,
        ctx->llm->tok_embd, pref_ids_t);  // [1024, n_prefix]

    // Audio tokens view [1024, T_audio]
    ggml_tensor * audio_view = ggml_view_2d(gctx, ctx->audio->audio_tokens,
        llm_dim, T_audio, ctx->audio->audio_tokens->nb[1], 0);

    // Suffix embeddings
    ggml_tensor * suf_emb = ggml_get_rows(gctx,
        ctx->llm->tok_embd, suf_ids_t);  // [1024, n_suffix]

    // Persistent output buffer for assembled embeddings: [1024, n_total]
    // Use a transient tensor allocated by sched
    ggml_tensor * combined = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, llm_dim, n_total);
    ggml_set_name(combined, "combined_emb");

    // Copy parts into combined
    ggml_tensor * c_pref = ggml_view_2d(gctx, combined, llm_dim, n_prefix,
        combined->nb[1], 0);
    ggml_build_forward_expand(gf, ggml_cpy(gctx, pref_emb, c_pref));

    ggml_tensor * c_audio = ggml_view_2d(gctx, combined, llm_dim, T_audio,
        combined->nb[1], (size_t)n_prefix * combined->nb[1]);
    ggml_build_forward_expand(gf, ggml_cpy(gctx, audio_view, c_audio));

    ggml_tensor * c_suf = ggml_view_2d(gctx, combined, llm_dim, n_suffix,
        combined->nb[1], (size_t)(n_prefix + T_audio) * combined->nb[1]);
    ggml_build_forward_expand(gf, ggml_cpy(gctx, suf_emb, c_suf));

    // Run 28 Qwen3 layers
    ggml_tensor * h = combined;
    for (int i = 0; i < FUN_ASR_LLM_LAYERS; i++) {
        h = build_qwen3_layer(ctx, gctx, gf, h, pos_t, i, n_total, 0, mask_t);
    }

    // Final norm + logits for last token (to get first decode token)
    ggml_tensor * h_last = ggml_view_1d(gctx, h, llm_dim,
        (size_t)(n_total - 1) * h->nb[1]);
    h_last = ggml_rms_norm(gctx, h_last, FUN_ASR_LLM_NORM_EPS);
    h_last = ggml_mul(gctx, h_last, ctx->llm->output_norm);
    // We don't need logits from prefill since we decode the last suffix token

    ggml_backend_sched_reset(g_ext.sched);
    if (!ggml_backend_sched_alloc_graph(g_ext.sched, gf)) {
        ggml_free(gctx); return -1;
    }

    // Upload prefix/suffix ids
    ggml_backend_tensor_set(pref_ids_t, prefix_ids.data(), 0, n_prefix * sizeof(int32_t));
    ggml_backend_tensor_set(suf_ids_t,  suffix_ids.data(), 0, n_suffix * sizeof(int32_t));

    // Upload positions [0, 1, ..., n_total-1]
    std::vector<int32_t> pos_data(n_total);
    for (int i = 0; i < n_total; i++) pos_data[i] = i;
    ggml_backend_tensor_set(pos_t, pos_data.data(), 0, n_total * sizeof(int32_t));

    // Upload causal mask: -inf for future positions
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

    if (ggml_backend_sched_graph_compute(g_ext.sched, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(gctx); return -1;
    }

    ggml_free(gctx);
    ctx->kv_used = n_total;
    return n_total;
}

// ============================================================================
// Decode single token step
// ============================================================================

static int32_t run_decode_step(fun_asr_ctx * ctx, int32_t token_id) {
    const int32_t kv_used = ctx->kv_used;
    const int32_t llm_dim = FUN_ASR_LLM_DIM;

    const size_t meta_size = ggml_tensor_overhead() * 2048 + ggml_graph_overhead_custom(2048, false);
    std::vector<uint8_t> meta_buf(meta_size);
    ggml_init_params ip = { meta_size, meta_buf.data(), true };
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) return FUN_ASR_LLM_EOS;

    ggml_cgraph * gf = ggml_new_graph_custom(gctx, 2048, false);

    // Input token ID: [1] int32
    ggml_tensor * tok_t = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ggml_set_name(tok_t, "token_id");
    ggml_backend_sched_set_tensor_backend(g_ext.sched, tok_t, ctx->backend);

    // Position: [1] int32
    ggml_tensor * pos_t = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ggml_set_name(pos_t, "position");
    ggml_backend_sched_set_tensor_backend(g_ext.sched, pos_t, ctx->backend);

    // Token embedding: [1024, 1]
    ggml_tensor * emb = ggml_get_rows(gctx, ctx->llm->tok_embd, tok_t);

    // Run 28 layers (n_tokens=1, kv_offset=kv_used)
    ggml_tensor * h = emb;
    for (int i = 0; i < FUN_ASR_LLM_LAYERS; i++) {
        h = build_qwen3_layer(ctx, gctx, gf, h, pos_t, i, 1, kv_used, nullptr);
    }

    // Final norm + logits
    ggml_tensor * h_flat = ggml_reshape_1d(gctx, h, llm_dim);
    h_flat = ggml_rms_norm(gctx, h_flat, FUN_ASR_LLM_NORM_EPS);
    h_flat = ggml_mul(gctx, h_flat, ctx->llm->output_norm);
    ggml_tensor * logits = ggml_mul_mat(gctx, ctx->llm->output_w, h_flat); // [151936]

    // Argmax: use ggml_argmax on [vocab_size, 1]
    ggml_tensor * logits_2d = ggml_reshape_2d(gctx, logits, FUN_ASR_LLM_VOCAB_SIZE, 1);
    ggml_tensor * next_tok   = ggml_argmax(gctx, logits_2d); // [1] int32

    // Persistent output: store in a 1D tensor we can read
    ggml_tensor * out_tok = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ggml_set_name(out_tok, "out_token");
    ggml_backend_sched_set_tensor_backend(g_ext.sched, out_tok, ctx->backend);
    ggml_build_forward_expand(gf, ggml_cpy(gctx, next_tok, out_tok));

    ggml_backend_sched_reset(g_ext.sched);
    if (!ggml_backend_sched_alloc_graph(g_ext.sched, gf)) {
        ggml_free(gctx); return FUN_ASR_LLM_EOS;
    }

    // Set inputs
    ggml_backend_tensor_set(tok_t, &token_id, 0, sizeof(int32_t));
    ggml_backend_tensor_set(pos_t, &kv_used, 0, sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(g_ext.sched, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(gctx); return FUN_ASR_LLM_EOS;
    }

    int32_t next_id = FUN_ASR_LLM_EOS;
    ggml_backend_tensor_get(out_tok, &next_id, 0, sizeof(int32_t));

    ggml_free(gctx);
    return next_id;
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

    fprintf(stderr, "fun_asr: initialized successfully\n");
    return ctx;
}

// ============================================================================
// Public API: fun_asr_transcribe
// ============================================================================

std::string fun_asr_transcribe(fun_asr_ctx * ctx, const std::string & wav_path, int32_t max_tokens) {
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

    // 2. Upscale ×32768
    for (auto & s : audio) s *= 32768.0f;

    // 3. Compute mel filterbank [n_frames, 80]
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
    fprintf(stderr, "fun_asr: T'=%d\n", T_prime);
    fprintf(stderr, "fun_asr: lfr[0,:5] = %.6f %.6f %.6f %.6f %.6f\n", lfr[0], lfr[1], lfr[2], lfr[3], lfr[4]);
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

    // Debug: check enc_out and audio_tokens
    {
        std::vector<float> dbg(5);
        ggml_backend_tensor_get(ctx->audio->enc_out, dbg.data(), 0, 5 * sizeof(float));
        fprintf(stderr, "fun_asr: enc_out[0..4] = %.6f %.6f %.6f %.6f %.6f\n",
            dbg[0], dbg[1], dbg[2], dbg[3], dbg[4]);
        ggml_backend_tensor_get(ctx->audio->audio_tokens, dbg.data(), 0, 5 * sizeof(float));
        fprintf(stderr, "fun_asr: audio_tokens[0..4] = %.4f %.4f %.4f %.4f %.4f\n",
            dbg[0], dbg[1], dbg[2], dbg[3], dbg[4]);
    }

    // 7. Tokenize prompt
    const std::string prefix_text = PROMPT_PREFIX;
    const std::string suffix_text = PROMPT_SUFFIX;

    std::vector<int32_t> prefix_ids = tokenize(*ctx->llm, prefix_text);
    std::vector<int32_t> suffix_ids = tokenize(*ctx->llm, suffix_text);

    // Debug: dump prefix/suffix token IDs
    fprintf(stderr, "fun_asr: prefix_ids =");
    for (int id : prefix_ids) fprintf(stderr, " %d", id);
    fprintf(stderr, "\nfun_asr: suffix_ids =");
    for (int id : suffix_ids) fprintf(stderr, " %d", id);
    fprintf(stderr, "\n");

    fprintf(stderr, "fun_asr: prefix=%zu suffix=%zu audio=%d total_prefill=%zu\n",
        prefix_ids.size(), suffix_ids.size(), T_prime,
        prefix_ids.size() + T_prime + suffix_ids.size());

    // 8. Prefill (all tokens except the last one)
    ctx->kv_used = 0;
    const int32_t n_total = (int32_t)(prefix_ids.size() + T_prime + suffix_ids.size());

    if (n_total > 1) {
        // Build prefill with all tokens except the last suffix token
        std::vector<int32_t> prefill_prefix = prefix_ids;
        std::vector<int32_t> prefill_suffix = suffix_ids;
        if (!prefill_suffix.empty()) {
            prefill_suffix.pop_back();  // Remove last token for separate first step
        }

        int32_t prefill_len = run_prefill(ctx, prefill_prefix, T_prime, prefill_suffix);
        if (prefill_len < 0) {
            fprintf(stderr, "fun_asr: prefill failed\n");
            return "";
        }
        elapsed("prefill");
    }

    // 9. First decode step with the last suffix token at its correct position
    int32_t cur_token = suffix_ids.empty() ? FUN_ASR_LLM_EOS : suffix_ids.back();
    int32_t first_token = run_decode_step(ctx, cur_token);
    ctx->kv_used++;

    fprintf(stderr, "fun_asr: first_token=%d\n", first_token);
    if (first_token == FUN_ASR_LLM_EOS) {
        elapsed("decode");
        return "";
    }

    std::vector<int32_t> out_tokens;
    out_tokens.push_back(first_token);
    cur_token = first_token;
    elapsed("first_step");

    // 10. Autoregressive decode loop
    for (int step = 1; step < max_tokens; step++) {
        if (ctx->kv_used >= FUN_ASR_KV_WINDOW) {
            kv_shift_left(ctx);
        }

        int32_t next = run_decode_step(ctx, cur_token);
        ctx->kv_used++;

        fprintf(stderr, "fun_asr: step=%d token=%d\n", step, next);
        if (next == FUN_ASR_LLM_EOS) break;
        out_tokens.push_back(next);
        cur_token = next;
    }
    elapsed("decode");

    fprintf(stderr, "fun_asr: generated %zu tokens\n", out_tokens.size());
    return decode_tokens(*ctx->llm, out_tokens);
}

// ============================================================================
// Public API: fun_asr_free
// ============================================================================

void fun_asr_free(fun_asr_ctx * ctx) {
    if (!ctx) return;

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
