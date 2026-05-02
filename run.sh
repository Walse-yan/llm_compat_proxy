#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash run.sh [options]

Options:
  --host VALUE                       Bind host, default: 0.0.0.0
  --port VALUE                       Bind port, default: 8090
  --llama-url VALUE                  llama.cpp base URL, default: http://127.0.0.1:8089
  --model VALUE                      Fallback model id
  --embeddings-cache 0|1             Enable/disable /v1/embeddings cache, default: 1
  --embeddings-cache-max-items NUM   Max cached embedding responses, default: 1024
  --models-cache-ttl SECONDS         /v1/models cache TTL, default: 5. Set 0 to disable
  --help                             Show this help
EOF
}

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8090}"
LLAMA_CPP_BASE_URL="${LLAMA_CPP_BASE_URL:-http://127.0.0.1:8089}"
PROXY_MODEL_ID="${PROXY_MODEL_ID:-Qwopus3.6-27B-v1-preview-Q5_K_M.gguf}"
EMBEDDINGS_CACHE_ENABLED="${EMBEDDINGS_CACHE_ENABLED:-1}"
EMBEDDINGS_CACHE_MAX_ITEMS="${EMBEDDINGS_CACHE_MAX_ITEMS:-1024}"
MODELS_CACHE_TTL_SECONDS="${MODELS_CACHE_TTL_SECONDS:-5}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --llama-url)
      LLAMA_CPP_BASE_URL="$2"
      shift 2
      ;;
    --model)
      PROXY_MODEL_ID="$2"
      shift 2
      ;;
    --embeddings-cache)
      EMBEDDINGS_CACHE_ENABLED="$2"
      shift 2
      ;;
    --embeddings-cache-max-items)
      EMBEDDINGS_CACHE_MAX_ITEMS="$2"
      shift 2
      ;;
    --models-cache-ttl)
      MODELS_CACHE_TTL_SECONDS="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

export LLAMA_CPP_BASE_URL
export PROXY_MODEL_ID
export EMBEDDINGS_CACHE_ENABLED
export EMBEDDINGS_CACHE_MAX_ITEMS
export MODELS_CACHE_TTL_SECONDS

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate LLM-env
fi

python -m uvicorn app.main:app --host "$HOST" --port "$PORT"
