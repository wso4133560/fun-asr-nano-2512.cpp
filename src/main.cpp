#include "fun_asr.h"
#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s --model-dir <dir> --wav <file> [--device cuda|cpu] [--max-tokens <n>]\n", prog);
}

int main(int argc, char ** argv) {
    std::string model_dir;
    std::string wav_path;
    bool use_cuda = true;
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

    auto t0 = std::chrono::steady_clock::now();

    fun_asr_ctx * ctx = fun_asr_init(model_dir, use_cuda);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize fun_asr context\n");
        return 1;
    }

    auto t1 = std::chrono::steady_clock::now();

    std::string result = fun_asr_transcribe(ctx, wav_path, max_tokens);

    auto t2 = std::chrono::steady_clock::now();

    auto init_ms       = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    auto transcribe_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    if (!result.empty()) {
        printf("%s\n", result.c_str());
    } else {
        fprintf(stderr, "Transcription failed or returned empty result\n");
    }

    fprintf(stderr, "\n[timing] init=%.3fs  transcribe=%.3fs\n",
            init_ms / 1000.0, transcribe_ms / 1000.0);

    fun_asr_free(ctx);
    return result.empty() ? 1 : 0;
}
