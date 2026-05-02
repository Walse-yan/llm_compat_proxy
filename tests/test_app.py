import pytest
from httpx import ASGITransport, AsyncClient

from app.cache import clear_caches
from app.main import app


@pytest.fixture(autouse=True)
def clear_proxy_caches():
    clear_caches()


@pytest.mark.asyncio
async def test_models_endpoint_returns_upstream_model_shape(monkeypatch):
    async def fake_model_list():
        return {
            "object": "list",
            "data": [
                {
                    "id": "dynamic-model.gguf",
                    "object": "model",
                    "created": 123,
                    "owned_by": "llamacpp",
                }
            ],
        }

    monkeypatch.setattr("app.main.upstream_model_list", fake_model_list)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/v1/models")

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "dynamic-model.gguf"


@pytest.mark.asyncio
async def test_anthropic_messages_calls_upstream_and_returns_anthropic_shape(monkeypatch):
    captured = {}

    async def fake_current_model_id():
        return "dynamic-model.gguf"

    async def fake_chat_completion(payload):
        captured["payload"] = payload
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "local answer"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        }

    monkeypatch.setattr("app.main.get_current_model_id", fake_current_model_id)
    monkeypatch.setattr("app.main.upstream_chat_completion", fake_chat_completion)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-latest",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "你好"}],
            },
        )

    assert response.status_code == 200
    assert captured["payload"]["model"] == "dynamic-model.gguf"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "你好"}]
    assert response.json()["content"] == [{"type": "text", "text": "local answer"}]


@pytest.mark.asyncio
async def test_openai_chat_uses_current_model_when_request_omits_model(monkeypatch):
    captured = {}

    async def fake_current_model_id():
        return "dynamic-model.gguf"

    async def fake_chat_completion(payload):
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    monkeypatch.setattr("app.main.get_current_model_id", fake_current_model_id)
    monkeypatch.setattr("app.main.upstream_chat_completion", fake_chat_completion)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert captured["payload"]["model"] == "dynamic-model.gguf"


@pytest.mark.asyncio
async def test_openai_chat_keeps_requested_model(monkeypatch):
    captured = {}

    async def fake_current_model_id():
        return "dynamic-model.gguf"

    async def fake_chat_completion(payload):
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    monkeypatch.setattr("app.main.get_current_model_id", fake_current_model_id)
    monkeypatch.setattr("app.main.upstream_chat_completion", fake_chat_completion)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "client-selected.gguf",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    assert captured["payload"]["model"] == "client-selected.gguf"


@pytest.mark.asyncio
async def test_cors_preflight_for_responses_endpoint():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.options(
            "/v1/responses",
            headers={
                "Origin": "http://localhost:3210",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"


@pytest.mark.asyncio
async def test_bare_options_for_responses_endpoint():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.options("/v1/responses")

    assert response.status_code == 200
    assert response.headers["allow"] == "GET, POST, PUT, PATCH, DELETE, OPTIONS"


@pytest.mark.asyncio
async def test_openai_responses_endpoint_maps_input_to_chat_and_returns_response_shape(monkeypatch):
    captured = {}

    async def fake_current_model_id():
        return "dynamic-model.gguf"

    async def fake_chat_completion(payload):
        captured["payload"] = payload
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "response answer"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }

    monkeypatch.setattr("app.main.get_current_model_id", fake_current_model_id)
    monkeypatch.setattr("app.main.upstream_chat_completion", fake_chat_completion)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/v1/responses",
            json={
                "instructions": "Be brief.",
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "你好"}],
                    }
                ],
                "max_output_tokens": 32,
            },
        )

    assert response.status_code == 200
    assert captured["payload"]["model"] == "dynamic-model.gguf"
    assert captured["payload"]["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "你好"},
    ]
    assert captured["payload"]["max_tokens"] == 32
    body = response.json()
    assert body["object"] == "response"
    assert body["output_text"] == "response answer"


@pytest.mark.asyncio
async def test_openai_responses_endpoint_keeps_reasoning_separate(monkeypatch):
    async def fake_current_model_id():
        return "dynamic-model.gguf"

    async def fake_chat_completion(_payload):
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "thinking text",
                        "content": "answer text",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
        }

    monkeypatch.setattr("app.main.get_current_model_id", fake_current_model_id)
    monkeypatch.setattr("app.main.upstream_chat_completion", fake_chat_completion)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/v1/responses",
            json={"input": "hi"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["output_text"] == "answer text"
    assert body["output"][0]["type"] == "reasoning"
    assert body["output"][0]["content"] == [{"type": "reasoning_text", "text": "thinking text"}]
    assert body["output"][1]["type"] == "message"


@pytest.mark.asyncio
async def test_openai_responses_stream_emits_reasoning_events_separately(monkeypatch):
    async def fake_current_model_id():
        return "dynamic-model.gguf"

    async def fake_stream(_payload):
        yield b'data: {"choices":[{"delta":{"reasoning_content":"think "}}]}\n\n'
        yield b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    monkeypatch.setattr("app.main.get_current_model_id", fake_current_model_id)
    monkeypatch.setattr("app.main.upstream_chat_completion_stream", fake_stream)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/v1/responses",
            json={"input": "hi", "stream": True},
        )

    assert response.status_code == 200
    text = response.text
    assert "response.reasoning_text.delta" in text
    assert '"delta": "think "' in text
    assert "response.output_text.delta" in text
    assert '"delta": "answer"' in text


@pytest.mark.asyncio
async def test_openai_embeddings_endpoint_forwards_to_upstream(monkeypatch):
    captured = {}

    async def fake_current_model_id():
        return "embedding-model.gguf"

    async def fake_embeddings(payload):
        captured["payload"] = payload
        return {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
            "model": payload["model"],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }

    monkeypatch.setattr("app.main.get_current_model_id", fake_current_model_id)
    monkeypatch.setattr("app.main.upstream_embeddings", fake_embeddings)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post("/v1/embeddings", json={"input": "hello"})

    assert response.status_code == 200
    assert captured["payload"] == {"input": "hello", "model": "embedding-model.gguf"}
    assert response.json()["data"][0]["embedding"] == [0.1, 0.2]


@pytest.mark.asyncio
async def test_openai_embeddings_endpoint_uses_cache_for_repeated_payload(monkeypatch):
    calls = 0

    async def fake_current_model_id():
        return "embedding-model.gguf"

    async def fake_embeddings(payload):
        nonlocal calls
        calls += 1
        return {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [calls, 0.2], "index": 0}],
            "model": payload["model"],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }

    monkeypatch.setattr("app.main.get_current_model_id", fake_current_model_id)
    monkeypatch.setattr("app.main.upstream_embeddings", fake_embeddings)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        first = await client.post("/v1/embeddings", json={"input": "hello"})
        second = await client.post("/v1/embeddings", json={"input": "hello"})

    assert calls == 1
    assert first.headers["x-proxy-cache"] == "MISS"
    assert second.headers["x-proxy-cache"] == "HIT"
    assert second.json()["data"][0]["embedding"] == [1, 0.2]


@pytest.mark.asyncio
async def test_models_endpoint_uses_short_cache(monkeypatch):
    calls = 0

    async def fake_model_list():
        nonlocal calls
        calls += 1
        return {
            "object": "list",
            "data": [{"id": f"dynamic-{calls}.gguf", "object": "model"}],
        }

    monkeypatch.setattr("app.main.upstream_model_list", fake_model_list)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        first = await client.get("/v1/models")
        second = await client.get("/v1/models")

    assert calls == 1
    assert first.headers["x-proxy-cache"] == "MISS"
    assert second.headers["x-proxy-cache"] == "HIT"
    assert second.json()["data"][0]["id"] == "dynamic-1.gguf"


@pytest.mark.asyncio
async def test_cache_status_endpoint_reports_cache_settings():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/cache/status")

    assert response.status_code == 200
    body = response.json()
    assert "embeddings" in body
    assert "models" in body
    assert body["embeddings"]["max_items"] >= 0
    assert body["models"]["ttl_seconds"] >= 0
