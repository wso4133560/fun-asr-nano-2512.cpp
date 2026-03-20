#!/usr/bin/env bash
set -euo pipefail

REPO="wso4133560freewind/Fun-ASR-Nano-2512-gguf"
BASE_URL="https://huggingface.co/${REPO}/resolve/main"
DEFAULT_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"

# --- defaults ---
LLM_QUANT="Q5_K_M"
OUT_DIR="${DEFAULT_DIR}"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Download Fun-ASR-Nano-2512 GGUF model files.

Options:
  -d, --dir <path>      Output directory (default: ./models)
  -q, --quant <quant>   LLM quantization: F16 | Q4_K_M | Q5_K_M (default: Q5_K_M)
  -h, --help            Show this help

Files downloaded:
  Fun-ASR-Nano-2512-llm-<quant>.gguf
  Fun-ASR-Nano-2512-audio-mmproj-mtmd-F16.gguf
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dir)   OUT_DIR="$2"; shift 2 ;;
        -q|--quant) LLM_QUANT="$2"; shift 2 ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

case "${LLM_QUANT}" in
    F16|Q4_K_M|Q5_K_M) ;;
    *) echo "Invalid quant '${LLM_QUANT}'. Choose F16, Q4_K_M, or Q5_K_M." >&2; exit 1 ;;
esac

mkdir -p "${OUT_DIR}"

download() {
    local filename="$1"
    local dest="${OUT_DIR}/${filename}"
    local url="${BASE_URL}/${filename}"

    if [[ -f "${dest}" ]]; then
        echo "[skip] ${filename} already exists"
        return
    fi

    echo "[download] ${filename}"
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "${dest}.tmp" "${url}"
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "${dest}.tmp" "${url}"
    else
        echo "Neither wget nor curl found." >&2
        exit 1
    fi
    mv "${dest}.tmp" "${dest}"
    echo "[done] ${dest}"
}

download "Fun-ASR-Nano-2512-llm-${LLM_QUANT}.gguf"
download "Fun-ASR-Nano-2512-audio-mmproj-mtmd-F16.gguf"

echo ""
echo "Models saved to: ${OUT_DIR}"
echo "Run inference with:"
echo "  ./build/fun-asr-nano --model-dir ${OUT_DIR} --wav input.wav"
