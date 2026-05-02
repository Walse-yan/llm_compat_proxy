import os


LLAMA_CPP_BASE_URL = os.getenv("LLAMA_CPP_BASE_URL", "http://127.0.0.1:8089").rstrip("/")
PROXY_MODEL_ID = os.getenv("PROXY_MODEL_ID", "Qwopus3.6-27B-v1-preview-Q5_K_M.gguf")
UPSTREAM_TIMEOUT_SECONDS = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "600"))
EMBEDDINGS_CACHE_ENABLED = os.getenv("EMBEDDINGS_CACHE_ENABLED", "1") not in {"0", "false", "False"}
EMBEDDINGS_CACHE_MAX_ITEMS = int(os.getenv("EMBEDDINGS_CACHE_MAX_ITEMS", "1024"))
MODELS_CACHE_TTL_SECONDS = float(os.getenv("MODELS_CACHE_TTL_SECONDS", "5"))
