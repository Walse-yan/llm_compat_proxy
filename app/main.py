from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .cache import (
    cache_status,
    get_embeddings_cache,
    get_models_cache,
    make_cache_key,
    set_embeddings_cache,
    set_models_cache,
)
from .config import EMBEDDINGS_CACHE_ENABLED, LLAMA_CPP_BASE_URL, PROXY_MODEL_ID
from .transforms import (
    anthropic_messages_to_openai,
    make_anthropic_model_list,
    make_openai_model_list,
    normalize_openai_chat_payload,
    openai_chat_to_anthropic_message,
    openai_chat_to_response,
    responses_to_openai_chat,
)
from .upstream import (
    chat_completion as upstream_chat_completion,
    chat_completion_stream as upstream_chat_completion_stream,
    completion as upstream_completion,
    completion_stream as upstream_completion_stream,
    embeddings as upstream_embeddings,
    model_list as upstream_model_list,
)


app = FastAPI(title="Local LLM OpenAI/Anthropic Compatibility Proxy")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _with_default_model(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["model"] = result.get("model") or PROXY_MODEL_ID
    return result


async def get_current_model_id() -> str:
    try:
        models = await get_model_list()
    except httpx.HTTPError:
        return PROXY_MODEL_ID

    data = models.get("data") or models.get("models") or []
    if not data:
        return PROXY_MODEL_ID

    first = data[0]
    if isinstance(first, dict):
        return first.get("id") or first.get("model") or first.get("name") or PROXY_MODEL_ID
    return str(first)


async def _with_current_model(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["model"] = result.get("model") or await get_current_model_id()
    return result


async def get_model_list() -> dict[str, Any]:
    cached = get_models_cache()
    if cached is not None:
        return cached
    models = await upstream_model_list()
    set_models_cache(models)
    return models


def _upstream_error(exc: httpx.HTTPStatusError) -> HTTPException:
    detail: Any
    try:
        detail = exc.response.json()
    except ValueError:
        detail = exc.response.text
    return HTTPException(status_code=exc.response.status_code, detail=detail)


@app.exception_handler(httpx.HTTPStatusError)
async def httpx_status_handler(_request: Request, exc: httpx.HTTPStatusError) -> JSONResponse:
    error = _upstream_error(exc)
    return JSONResponse(status_code=error.status_code, content={"error": error.detail})


@app.options("/{path:path}")
async def bare_options(path: str) -> Response:
    return Response(
        status_code=200,
        headers={
            "allow": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
            "access-control-allow-origin": "*",
            "access-control-allow-methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
            "access-control-allow-headers": "*",
        },
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": await get_current_model_id(),
        "upstream": LLAMA_CPP_BASE_URL,
    }


@app.get("/cache/status")
async def proxy_cache_status() -> dict[str, Any]:
    return cache_status()


@app.get("/v1/models")
async def list_openai_models() -> dict[str, Any]:
    try:
        cached = get_models_cache()
        if cached is not None:
            return JSONResponse(content=cached, headers={"x-proxy-cache": "HIT"})
        models = await upstream_model_list()
        set_models_cache(models)
        return JSONResponse(content=models, headers={"x-proxy-cache": "MISS"})
    except httpx.HTTPError:
        return JSONResponse(
            content=make_openai_model_list(PROXY_MODEL_ID),
            headers={"x-proxy-cache": "BYPASS"},
        )


@app.get("/anthropic/v1/models")
async def list_anthropic_models() -> dict[str, Any]:
    return make_anthropic_model_list(await get_current_model_id())


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    payload = normalize_openai_chat_payload(
        await request.json(),
        default_model=await get_current_model_id(),
    )
    if payload.get("stream"):
        return StreamingResponse(
            upstream_chat_completion_stream(payload),
            media_type="text/event-stream",
        )
    return await upstream_chat_completion(payload)


@app.post("/v1/completions")
async def openai_completions(request: Request):
    payload = await _with_current_model(await request.json())
    if payload.get("stream"):
        return StreamingResponse(
            upstream_completion_stream(payload),
            media_type="text/event-stream",
        )
    return await upstream_completion(payload)


@app.post("/v1/responses")
async def openai_responses(request: Request):
    payload = await request.json()
    response_id = f"resp_{uuid.uuid4().hex}"
    created_at = int(time.time())
    chat_payload = responses_to_openai_chat(
        payload,
        default_model=payload.get("model") or await get_current_model_id(),
    )

    if chat_payload.get("stream"):
        return StreamingResponse(
            _responses_stream(chat_payload, response_id, created_at),
            media_type="text/event-stream",
        )

    response = await upstream_chat_completion(chat_payload)
    return openai_chat_to_response(
        response,
        model=chat_payload["model"],
        response_id=response_id,
        created_at=created_at,
    )


@app.post("/v1/embeddings")
async def openai_embeddings(request: Request):
    payload = await _with_current_model(await request.json())
    if EMBEDDINGS_CACHE_ENABLED:
        cache_key = make_cache_key(payload)
        cached = get_embeddings_cache(cache_key)
        if cached is not None:
            return JSONResponse(content=cached, headers={"x-proxy-cache": "HIT"})
    response = await upstream_embeddings(payload)
    if EMBEDDINGS_CACHE_ENABLED:
        set_embeddings_cache(cache_key, response)
        return JSONResponse(content=response, headers={"x-proxy-cache": "MISS"})
    return JSONResponse(content=response, headers={"x-proxy-cache": "BYPASS"})


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    payload = await request.json()
    request_id = f"msg_{uuid.uuid4().hex}"
    model = await get_current_model_id()
    openai_payload = anthropic_messages_to_openai(payload, default_model=model)

    if openai_payload.get("stream"):
        return StreamingResponse(
            _anthropic_stream(openai_payload, request_id, model),
            media_type="text/event-stream",
        )

    response = await upstream_chat_completion(openai_payload)
    return openai_chat_to_anthropic_message(
        response,
        model=openai_payload["model"],
        request_id=request_id,
    )


async def _anthropic_stream(payload: dict[str, Any], request_id: str, model: str) -> AsyncIterator[bytes]:
    created = int(time.time())
    message = {
        "id": request_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    yield _sse("message_start", {"type": "message_start", "message": message})
    yield _sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    )

    async for raw in upstream_chat_completion_stream(payload):
        for line in raw.decode("utf-8", errors="ignore").splitlines():
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if not data or data == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if text:
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": text},
                    },
                )

            finish_reason = choice.get("finish_reason")
            if finish_reason:
                yield _sse(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": 0},
                    },
                )

    yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield _sse("message_stop", {"type": "message_stop"})


async def _responses_stream(payload: dict[str, Any], response_id: str, created_at: int) -> AsyncIterator[bytes]:
    response_suffix = response_id.removeprefix("resp_")
    reasoning_item_id = f"rs_{response_suffix}"
    output_item_id = f"msg_{response_id.removeprefix('resp_')}"
    response_shell = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "in_progress",
        "model": payload["model"],
        "output": [],
    }
    yield _sse("response.created", {"type": "response.created", "response": response_shell})
    yield _sse("response.in_progress", {"type": "response.in_progress", "response": response_shell})

    output_started = False
    reasoning_started = False
    answer_parts: list[str] = []
    reasoning_parts: list[str] = []
    async for raw in upstream_chat_completion_stream(payload):
        for line in raw.decode("utf-8", errors="ignore").splitlines():
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if not data or data == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            reasoning_text = delta.get("reasoning_content")
            if reasoning_text:
                if not reasoning_started:
                    reasoning_started = True
                    yield _sse(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": 0,
                            "item": {
                                "id": reasoning_item_id,
                                "type": "reasoning",
                                "summary": [],
                                "content": [],
                            },
                        },
                    )
                    yield _sse(
                        "response.reasoning_text_part.added",
                        {
                            "type": "response.reasoning_text_part.added",
                            "item_id": reasoning_item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "part": {"type": "reasoning_text", "text": ""},
                        },
                    )
                reasoning_parts.append(reasoning_text)
                yield _sse(
                    "response.reasoning_text.delta",
                    {
                        "type": "response.reasoning_text.delta",
                        "item_id": reasoning_item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": reasoning_text,
                    },
                )

            text = delta.get("content")
            if text:
                if not output_started:
                    output_started = True
                    output_index = 1 if reasoning_started else 0
                    yield _sse(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": output_index,
                            "item": {
                                "id": output_item_id,
                                "type": "message",
                                "status": "in_progress",
                                "role": "assistant",
                                "content": [],
                            },
                        },
                    )
                    yield _sse(
                        "response.content_part.added",
                        {
                            "type": "response.content_part.added",
                            "item_id": output_item_id,
                            "output_index": output_index,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": "", "annotations": []},
                        },
                    )
                answer_parts.append(text)
                yield _sse(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": output_item_id,
                        "output_index": 1 if reasoning_started else 0,
                        "content_index": 0,
                        "delta": text,
                    },
                )

    final_reasoning = "".join(reasoning_parts)
    final_text = "".join(answer_parts)
    output_index = 1 if reasoning_started else 0

    if reasoning_started:
        yield _sse(
            "response.reasoning_text.done",
            {
                "type": "response.reasoning_text.done",
                "item_id": reasoning_item_id,
                "output_index": 0,
                "content_index": 0,
                "text": final_reasoning,
            },
        )
        yield _sse(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": reasoning_item_id,
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": final_reasoning}],
                    "content": [{"type": "reasoning_text", "text": final_reasoning}],
                },
            },
        )

    if not output_started:
        yield _sse(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": {
                    "id": output_item_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            },
        )
        yield _sse(
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "item_id": output_item_id,
                "output_index": output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        )

    yield _sse(
        "response.output_text.done",
        {
            "type": "response.output_text.done",
            "item_id": output_item_id,
            "output_index": output_index,
            "content_index": 0,
            "text": final_text,
        },
    )
    yield _sse(
        "response.content_part.done",
        {
            "type": "response.content_part.done",
            "item_id": output_item_id,
            "output_index": output_index,
            "content_index": 0,
            "part": {"type": "output_text", "text": final_text, "annotations": []},
        },
    )
    yield _sse(
        "response.output_item.done",
        {
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": {
                "id": output_item_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": final_text, "annotations": []}],
            },
        },
    )
    completed = dict(response_shell)
    completed["status"] = "completed"
    completed_output = []
    if reasoning_started:
        completed_output.append(
            {
                "id": reasoning_item_id,
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": final_reasoning}],
                "content": [{"type": "reasoning_text", "text": final_reasoning}],
            }
        )
    completed_output.append(
        {
            "id": output_item_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": final_text, "annotations": []}],
        }
    )
    completed["output"] = completed_output
    completed["output_text"] = final_text
    yield _sse("response.completed", {"type": "response.completed", "response": completed})
    yield b"data: [DONE]\n\n"


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")
