# LLM Compat Proxy

[中文文档](README.zh-CN.md)

A lightweight FastAPI proxy that wraps a local `llama.cpp` server with OpenAI-compatible and Anthropic-compatible APIs. It is designed for local GGUF models and clients such as LobeHub/LobeChat, OpenClaw, opencode, and other tools that expect `/v1` style endpoints.

## Features

- OpenAI-compatible endpoints: `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`
- Anthropic-compatible endpoint: `/v1/messages`
- Automatic model discovery from `llama.cpp` `/v1/models`
- Reasoning-aware Responses API adapter for `reasoning_content`
- CORS and `OPTIONS` preflight handling for browser clients
- Safe proxy-level cache for embeddings and model lists
- Command-line startup script with cache switches

## Requirements

- Python 3.11+
- A running `llama.cpp` server, usually at `http://127.0.0.1:8089`
- Python packages listed in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

If you use conda:

```bash
conda activate LLM-env
pip install -r requirements.txt
```

## Quick Start

Start the proxy with defaults:

```bash
cd /mnt/yanjq/program/LLM/llm_compat_proxy
bash run.sh
```

Or run uvicorn directly:

```bash
conda activate LLM-env
cd /mnt/yanjq/program/LLM/llm_compat_proxy
python -m uvicorn app.main:app --host 0.0.0.0 --port 8090
```

OpenAI-compatible base URL:

```text
http://127.0.0.1:8090/v1
```

API key can be any non-empty value, for example `local`.

## Command-Line Options

```bash
bash run.sh \
  --host 0.0.0.0 \
  --port 8090 \
  --llama-url http://127.0.0.1:8089 \
  --embeddings-cache 1 \
  --embeddings-cache-max-items 1024 \
  --models-cache-ttl 5
```

Disable proxy caches:

```bash
bash run.sh --embeddings-cache 0 --models-cache-ttl 0
```

Show all options:

```bash
bash run.sh --help
```

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `LLAMA_CPP_BASE_URL` | `http://127.0.0.1:8089` | Upstream `llama.cpp` server URL |
| `PROXY_MODEL_ID` | `Qwopus3.6-27B-v1-preview-Q5_K_M.gguf` | Fallback model ID if upstream model discovery fails |
| `UPSTREAM_TIMEOUT_SECONDS` | `600` | Timeout for upstream requests |
| `EMBEDDINGS_CACHE_ENABLED` | `1` | Enable `/v1/embeddings` cache; set `0` to disable |
| `EMBEDDINGS_CACHE_MAX_ITEMS` | `1024` | Maximum cached embedding responses |
| `MODELS_CACHE_TTL_SECONDS` | `5` | TTL for `/v1/models` cache; set `0` to disable |

`PROXY_MODEL_ID` is only a fallback. During normal use, the proxy asks `llama.cpp` for `GET /v1/models` and uses the first upstream model when a request omits `model`.

## Client Configuration

### LobeHub / LobeChat

Use an OpenAI-compatible provider:

```text
Base URL: http://YOUR_HOST:8090/v1
API Key: local
Model: use the model returned by /v1/models
```

For reasoning-capable local models, make sure your LobeHub model configuration enables reasoning-related options such as `enableReasoning` and `reasoningBudgetToken` when your LobeHub version requires them.

### OpenClaw / opencode / Other OpenAI Clients

```text
OPENAI_BASE_URL=http://YOUR_HOST:8090/v1
OPENAI_API_KEY=local
OPENAI_MODEL=<model returned by /v1/models>
```

### Anthropic-Compatible Clients

```text
ANTHROPIC_BASE_URL=http://YOUR_HOST:8090
ANTHROPIC_API_KEY=local
ANTHROPIC_MODEL=claude-3-5-sonnet-latest
```

The Anthropic endpoint ignores remote provider model names and routes requests to the current local model reported by `llama.cpp`.

## Examples

List models:

```bash
curl http://127.0.0.1:8090/v1/models
```

OpenAI chat completions:

```bash
curl http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain spatial transcriptomics briefly."}],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

OpenAI Responses API:

```bash
curl http://127.0.0.1:8090/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Reply with OK.",
    "max_output_tokens": 32
  }'
```

Anthropic messages:

```bash
curl http://127.0.0.1:8090/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: local" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-3-5-sonnet-latest",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

## Cache

The proxy keeps a small in-memory cache shared by all clients using it.

- `/v1/embeddings` is cached by normalized request payload.
- `/v1/models` is cached for a short TTL.
- Chat and responses are not cached by default.

Cache status is exposed through the `x-proxy-cache` response header:

```text
HIT | MISS | BYPASS
```

Check current cache settings:

```bash
curl http://127.0.0.1:8090/cache/status
```

## Development

Run tests:

```bash
pytest -q
```

Syntax check:

```bash
python -m compileall app
```

## Notes

- The proxy does not load models by itself. It forwards requests to your running `llama.cpp` server.
- For real model switching, switch the model in `llama.cpp`; the proxy will pick it up through `/v1/models`.
- Embedding support depends on whether your `llama.cpp` server and loaded model support embedding requests.
