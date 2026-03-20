#include "fun_asr.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <cstdint>
#include <string>

static std::string shell_quote(const std::string & s) {
    std::string out = "'";
    for (char ch : s) {
        if (ch == '\'') out += "'\\''";
        else out += ch;
    }
    out += "'";
    return out;
}

static std::string run_ctc_hint(
    const std::string & wav_path,
    const std::string & ctc_model_dir,
    const std::string & language) {
    std::string cmd = "python3 ./tools/ctc_hint.py --wav " + shell_quote(wav_path) +
                      " --model-dir " + shell_quote(ctc_model_dir);
    if (!language.empty()) {
        cmd += " --lang " + shell_quote(language);
    }

    FILE * pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";

    std::string output;
    char buf[512];
    while (fgets(buf, sizeof(buf), pipe)) {
        output += buf;
    }
    if (pclose(pipe) != 0) {
        return "";
    }

    while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) {
        output.pop_back();
    }
    return output;
}

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (unsigned char ch : s) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (ch < 0x20) {
                    char buf[7];
                    snprintf(buf, sizeof(buf), "\\u%04x", ch);
                    out += buf;
                } else {
                    out.push_back((char)ch);
                }
                break;
        }
    }
    return out;
}

static void print_usage(const char * prog) {
    fprintf(stderr,
            "Usage: %s --model-dir <dir> --wav <file> [--device cuda|cpu] [--max-tokens <n>]\n"
            "          [--lang <name>] [--task <transcribe|translate>] [--context <text>]\n"
            "          [--auto-ctc] [--ctc-model-dir <dir>]\n"
            "          [--temperature-base <f>] [--temperature-step <f>]\n"
            "          [--top-k <n>] [--retry-attempts <n>] [--seed <n>] [--dump-json]\n",
            prog);
}

int main(int argc, char ** argv) {
    std::string model_dir;
    std::string wav_path;
    std::string ctc_model_dir = "/home/tanglin/workspace2/Fun-ASR-GGUF/model";
    fun_asr_decode_options options;
    bool use_cuda = true;
    bool auto_ctc = false;
    bool dump_json = false;
    int32_t max_tokens = 512;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "--wav") == 0 && i + 1 < argc) {
            wav_path = argv[++i];
        } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            ++i;
            use_cuda = strcmp(argv[i], "cuda") == 0;
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lang") == 0 && i + 1 < argc) {
            options.language = argv[++i];
        } else if (strcmp(argv[i], "--task") == 0 && i + 1 < argc) {
            options.task = argv[++i];
        } else if (strcmp(argv[i], "--context") == 0 && i + 1 < argc) {
            options.context = argv[++i];
        } else if (strcmp(argv[i], "--auto-ctc") == 0) {
            auto_ctc = true;
        } else if (strcmp(argv[i], "--ctc-model-dir") == 0 && i + 1 < argc) {
            ctc_model_dir = argv[++i];
        } else if (strcmp(argv[i], "--temperature-base") == 0 && i + 1 < argc) {
            options.temperature_base = strtof(argv[++i], nullptr);
        } else if (strcmp(argv[i], "--temperature-step") == 0 && i + 1 < argc) {
            options.temperature_step = strtof(argv[++i], nullptr);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            options.top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--retry-attempts") == 0 && i + 1 < argc) {
            options.retry_attempts = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            options.seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dump-json") == 0) {
            dump_json = true;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_dir.empty() || wav_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    if (auto_ctc) {
        options.ctc_text = run_ctc_hint(wav_path, ctc_model_dir, options.language);
        if (!options.ctc_text.empty()) {
            fprintf(stderr, "fun_asr: ctc_hint=%s\n", options.ctc_text.c_str());
        } else {
            fprintf(stderr, "fun_asr: ctc hint unavailable\n");
        }
    }

    auto t0 = std::chrono::steady_clock::now();

    fun_asr_ctx * ctx = fun_asr_init(model_dir, use_cuda);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize fun_asr context\n");
        return 1;
    }

    auto t1 = std::chrono::steady_clock::now();

    std::string result = fun_asr_transcribe(ctx, wav_path, max_tokens, &options);

    auto t2 = std::chrono::steady_clock::now();

    auto init_ms       = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    auto transcribe_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    if (dump_json) {
        printf("{\"text\":\"%s\",\"generated_tokens\":%d,\"attempts_used\":%d,"
               "\"repetition_retries\":%d,\"punctuation_retries\":%d,"
               "\"final_temperature\":%.3f,\"temperature_base\":%.3f,"
               "\"temperature_step\":%.3f,\"top_k\":%d,\"retry_attempts\":%d,"
               "\"seed\":%d,\"max_tokens\":%d,\"language\":\"%s\",\"task\":\"%s\","
               "\"ctc_text\":\"%s\"}\n",
               json_escape(result).c_str(),
               ctx->last_generated_tokens,
               ctx->last_attempts_used,
               ctx->last_repetition_retries,
               ctx->last_punctuation_retries,
               ctx->last_final_temperature,
               options.temperature_base,
               options.temperature_step,
               options.top_k,
               options.retry_attempts,
               options.seed,
               max_tokens,
               json_escape(options.language).c_str(),
               json_escape(options.task).c_str(),
               json_escape(options.ctc_text).c_str());
    } else if (!result.empty()) {
        printf("%s\n", result.c_str());
    } else {
        fprintf(stderr, "Transcription failed or returned empty result\n");
    }

    const int n_gen = ctx->last_generated_tokens;
    const double tok_per_sec = n_gen > 0 ? n_gen / (transcribe_ms / 1000.0) : 0.0;
    fprintf(stderr, "\n[timing] init=%.3fs  transcribe=%.3fs  tokens=%d  tok/s=%.1f\n",
            init_ms / 1000.0, transcribe_ms / 1000.0, n_gen, tok_per_sec);

    fun_asr_free(ctx);
    return result.empty() ? 1 : 0;
}
