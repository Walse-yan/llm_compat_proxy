# LLM Compat Proxy

[English README](README.md)

一个轻量级 FastAPI 代理服务，用来把本地 `llama.cpp` 服务封装成 OpenAI 兼容接口和 Anthropic 兼容接口。适合本地 GGUF 模型，也适合 LobeHub/LobeChat、OpenClaw、opencode 以及其他需要 `/v1` 接口的客户端。

## 功能

- OpenAI 兼容接口：`/v1/models`、`/v1/chat/completions`、`/v1/completions`、`/v1/responses`、`/v1/embeddings`
- Anthropic 兼容接口：`/v1/messages`
- 自动从 `llama.cpp` 的 `/v1/models` 获取当前模型
- 支持将 `reasoning_content` 转换为 Responses API 的推理事件
- 支持浏览器客户端需要的 CORS 和 `OPTIONS` 预检
- 支持安全的代理层缓存：embedding 缓存和模型列表短缓存
- 提供带命令行参数的启动脚本

## 环境要求

- Python 3.11+
- 已运行的 `llama.cpp` 服务，默认地址为 `http://127.0.0.1:8089`
- `requirements.txt` 中列出的 Python 依赖

安装依赖：

```bash
pip install -r requirements.txt
```

如果使用 conda：

```bash
conda activate LLM-env
pip install -r requirements.txt
```

## 快速启动

使用默认配置启动：

```bash
cd /mnt/yanjq/program/LLM/llm_compat_proxy
bash run.sh
```

也可以直接启动 uvicorn：

```bash
conda activate LLM-env
cd /mnt/yanjq/program/LLM/llm_compat_proxy
python -m uvicorn app.main:app --host 0.0.0.0 --port 8090
```

OpenAI 兼容 Base URL：

```text
http://127.0.0.1:8090/v1
```

API Key 可以随便填一个非空值，例如 `local`。

## 命令行启动参数

```bash
bash run.sh \
  --host 0.0.0.0 \
  --port 8090 \
  --llama-url http://127.0.0.1:8089 \
  --embeddings-cache 1 \
  --embeddings-cache-max-items 1024 \
  --models-cache-ttl 5
```

关闭代理层缓存：

```bash
bash run.sh --embeddings-cache 0 --models-cache-ttl 0
```

查看全部参数：

```bash
bash run.sh --help
```

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LLAMA_CPP_BASE_URL` | `http://127.0.0.1:8089` | 上游 `llama.cpp` 服务地址 |
| `PROXY_MODEL_ID` | `Qwopus3.6-27B-v1-preview-Q5_K_M.gguf` | 上游模型发现失败时使用的兜底模型名 |
| `UPSTREAM_TIMEOUT_SECONDS` | `600` | 上游请求超时时间 |
| `EMBEDDINGS_CACHE_ENABLED` | `1` | 是否开启 `/v1/embeddings` 缓存；设为 `0` 可关闭 |
| `EMBEDDINGS_CACHE_MAX_ITEMS` | `1024` | 最多缓存多少条 embedding 响应 |
| `MODELS_CACHE_TTL_SECONDS` | `5` | `/v1/models` 缓存秒数；设为 `0` 可关闭 |

`PROXY_MODEL_ID` 只是兜底值。正常情况下，代理会请求 `llama.cpp` 的 `GET /v1/models`，并在客户端没有传 `model` 时使用上游当前模型。

## 客户端配置

### LobeHub / LobeChat

使用 OpenAI 兼容供应商：

```text
Base URL: http://你的地址:8090/v1
API Key: local
Model: 使用 /v1/models 返回的模型名
```

如果你的本地模型支持推理过程，并且 LobeHub 版本要求手动开启模型能力，请在 LobeHub 的模型配置中启用 `enableReasoning`、`reasoningBudgetToken` 等 reasoning 相关参数。

### OpenClaw / opencode / 其他 OpenAI 客户端

```text
OPENAI_BASE_URL=http://你的地址:8090/v1
OPENAI_API_KEY=local
OPENAI_MODEL=<从 /v1/models 获取到的模型名>
```

### Anthropic 兼容客户端

```text
ANTHROPIC_BASE_URL=http://你的地址:8090
ANTHROPIC_API_KEY=local
ANTHROPIC_MODEL=claude-3-5-sonnet-latest
```

Anthropic 接口会忽略远端供应商模型名，并自动路由到 `llama.cpp` 当前加载的本地模型。

## 调用示例

查看模型：

```bash
curl http://127.0.0.1:8090/v1/models
```

OpenAI Chat Completions：

```bash
curl http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "简单解释一下空间转录组学"}],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

OpenAI Responses API：

```bash
curl http://127.0.0.1:8090/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "input": "只回复 OK",
    "max_output_tokens": 32
  }'
```

Anthropic Messages：

```bash
curl http://127.0.0.1:8090/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: local" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-3-5-sonnet-latest",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

## 缓存

代理内置一个所有客户端共享的进程内缓存。

- `/v1/embeddings` 会按标准化后的请求内容缓存。
- `/v1/models` 会按短 TTL 缓存。
- 聊天和 Responses 回答默认不缓存，避免不同上下文串答案。

缓存状态会通过响应头返回：

```text
x-proxy-cache: HIT | MISS | BYPASS
```

查看当前缓存设置：

```bash
curl http://127.0.0.1:8090/cache/status
```

## 开发

运行测试：

```bash
pytest -q
```

语法检查：

```bash
python -m compileall app
```

## 说明

- 代理本身不加载模型，只负责转发请求到正在运行的 `llama.cpp`。
- 如果要切换模型，请在 `llama.cpp` 中切换；代理会通过 `/v1/models` 自动发现当前模型。
- `/v1/embeddings` 是否真正可用，取决于你的 `llama.cpp` 服务和当前模型是否支持 embedding。
