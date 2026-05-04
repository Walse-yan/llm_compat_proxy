from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
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
    ThinkTagStreamFilter,
    anthropic_messages_to_openai,
    clean_visible_text,
    make_anthropic_model_list,
    make_openai_model_list,
    normalize_openai_chat_payload,
    openai_chat_to_anthropic_message,
    openai_chat_to_response,
    openai_tool_calls_to_response_items,
    responses_to_openai_chat,
    split_think_text,
)
from .upstream import (
    chat_completion as upstream_chat_completion,
    chat_completion_stream as upstream_chat_completion_stream,
    completion as upstream_completion,
    completion_stream as upstream_completion_stream,
    embeddings as upstream_embeddings,
    model_list as upstream_model_list,
    open_chat_completion_stream as upstream_open_chat_completion_stream,
    open_completion_stream as upstream_open_completion_stream,
)


class CollapseDuplicateV1Middleware:
    def __init__(self, inner_app):
        self.inner_app = inner_app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            path = scope.get("path", "")
            if path.startswith("/v1/v1/"):
                scope = dict(scope)
                scope["path"] = "/v1/" + path.removeprefix("/v1/v1/")
        await self.inner_app(scope, receive, send)


app = FastAPI(title="Local LLM OpenAI/Anthropic Compatibility Proxy")
app.add_middleware(CollapseDuplicateV1Middleware)
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


def _log_llm_exchange(
    *,
    endpoint: str,
    model: str | None,
    stream: bool,
    question: str,
    answer: str,
    reasoning: str = "",
    tool_context: dict[str, Any] | None = None,
) -> None:
    record = {
        "event": "llm_exchange",
        "time": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "model": model,
        "stream": stream,
        "question": question,
        "answer": answer,
        "display_answer": answer or reasoning,
        "reasoning": reasoning,
    }
    if tool_context:
        record.update(tool_context)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def _messages_question(payload: dict[str, Any]) -> str:
    messages = payload.get("messages") or []
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return ""


def _chat_completion_text(payload: dict[str, Any]) -> tuple[str, str]:
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return str(message.get("content") or ""), str(message.get("reasoning_content") or "")


def _completion_text(payload: dict[str, Any]) -> str:
    choice = (payload.get("choices") or [{}])[0]
    return str(choice.get("text") or "")


def _responses_reasoning_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        for content in item.get("content") or item.get("summary") or []:
            if isinstance(content, dict):
                text = content.get("text")
                if text:
                    parts.append(str(text))
    return "\n".join(parts)


def _tool_context(payload: dict[str, Any]) -> dict[str, Any]:
    tools = payload.get("tools") or []
    tool_names: list[str] = []
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            name = tool.get("name")
            if not name and isinstance(tool.get("function"), dict):
                name = tool["function"].get("name")
            if not name:
                name = tool.get("type")
            if name:
                tool_names.append(str(name))
    return {
        "tools_count": len(tools) if isinstance(tools, list) else 0,
        "tool_names": tool_names[:20],
        "tool_choice": payload.get("tool_choice"),
        "parallel_tool_calls": payload.get("parallel_tool_calls"),
    }


def _tool_calls_returned(payload: dict[str, Any]) -> int:
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    tool_calls = message.get("tool_calls") or []
    return len(tool_calls) if isinstance(tool_calls, list) else 0


def _merge_stream_tool_call_delta(accumulator: dict[int, dict[str, Any]], delta_tool_calls: Any) -> None:
    if not isinstance(delta_tool_calls, list):
        return

    for raw_tool_call in delta_tool_calls:
        if not isinstance(raw_tool_call, dict):
            continue
        index = int(raw_tool_call.get("index") or 0)
        current = accumulator.setdefault(index, {"type": "function", "function": {"name": "", "arguments": ""}})

        if raw_tool_call.get("id"):
            current["id"] = raw_tool_call["id"]
        if raw_tool_call.get("type"):
            current["type"] = raw_tool_call["type"]

        raw_function = raw_tool_call.get("function")
        if isinstance(raw_function, dict):
            function = current.setdefault("function", {"name": "", "arguments": ""})
            if raw_function.get("name"):
                function["name"] = str(function.get("name") or "") + str(raw_function["name"])
            if raw_function.get("arguments"):
                function["arguments"] = str(function.get("arguments") or "") + str(raw_function["arguments"])


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


async def _open_chat_stream(payload: dict[str, Any]) -> AsyncIterator[bytes]:
    return await upstream_open_chat_completion_stream(payload)


async def _open_completion_stream(payload: dict[str, Any]) -> AsyncIterator[bytes]:
    return await upstream_open_completion_stream(payload)


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


@app.api_route("/v1", methods=["GET", "HEAD", "POST"])
async def v1_probe():
    return {"status": "ok", "api": "llm-compat-proxy", "base": "/v1"}


@app.get("/cache/status")
async def proxy_cache_status() -> dict[str, Any]:
    return cache_status()


@app.get("/models")
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


@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    payload = normalize_openai_chat_payload(
        await request.json(),
        default_model=await get_current_model_id(),
    )
    if payload.get("stream"):
        upstream_stream = await _open_chat_stream(payload)
        return StreamingResponse(
            _chat_completion_stream(
                upstream_stream,
                endpoint=str(request.url.path),
                model=payload.get("model"),
                question=_messages_question(payload),
                tool_context=_tool_context(payload),
            ),
            media_type="text/event-stream",
        )
    response = _clean_chat_completion_response(await upstream_chat_completion(payload))
    answer, reasoning = _chat_completion_text(response)
    _log_llm_exchange(
        endpoint=str(request.url.path),
        model=payload.get("model"),
        stream=False,
        question=_messages_question(payload),
        answer=answer,
        reasoning=reasoning,
        tool_context={**_tool_context(payload), "tool_calls_returned": _tool_calls_returned(response)},
    )
    return response


@app.post("/completions")
@app.post("/v1/completions")
async def openai_completions(request: Request):
    payload = await _with_current_model(await request.json())
    if payload.get("stream"):
        upstream_stream = await _open_completion_stream(payload)
        return StreamingResponse(
            _passthrough_stream(upstream_stream),
            media_type="text/event-stream",
        )
    response = await upstream_completion(payload)
    _log_llm_exchange(
        endpoint=str(request.url.path),
        model=payload.get("model"),
        stream=False,
        question=str(payload.get("prompt") or ""),
        answer=_completion_text(response),
    )
    return response


@app.post("/responses")
@app.post("/v1/responses")
async def openai_responses(request: Request):
    payload = await request.json()
    response_id = f"resp_{uuid.uuid4().hex}"
    created_at = int(time.time())
    tool_context = _tool_context(payload)
    chat_payload = responses_to_openai_chat(
        payload,
        default_model=payload.get("model") or await get_current_model_id(),
    )

    if chat_payload.get("stream"):
        upstream_stream = await _open_chat_stream(chat_payload)
        return StreamingResponse(
            _responses_stream(
                chat_payload,
                upstream_stream,
                response_id,
                created_at,
                endpoint=str(request.url.path),
                question=_messages_question(chat_payload),
                tool_context=tool_context,
            ),
            media_type="text/event-stream",
        )

    response = await upstream_chat_completion(chat_payload)
    response_payload = openai_chat_to_response(
        response,
        model=chat_payload["model"],
        response_id=response_id,
        created_at=created_at,
    )
    _log_llm_exchange(
        endpoint=str(request.url.path),
        model=chat_payload.get("model"),
        stream=False,
        question=_messages_question(chat_payload),
        answer=str(response_payload.get("output_text") or ""),
        reasoning=_responses_reasoning_text(response_payload),
        tool_context={**tool_context, "tool_calls_returned": _tool_calls_returned(response)},
    )
    return response_payload


@app.post("/embeddings")
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


@app.post("/messages")
@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    payload = await request.json()
    request_id = f"msg_{uuid.uuid4().hex}"
    model = await get_current_model_id()
    openai_payload = anthropic_messages_to_openai(payload, default_model=model)

    if openai_payload.get("stream"):
        upstream_stream = await _open_chat_stream(openai_payload)
        return StreamingResponse(
            _anthropic_stream(openai_payload, upstream_stream, request_id, model, endpoint=str(request.url.path)),
            media_type="text/event-stream",
        )

    response = await upstream_chat_completion(openai_payload)
    response = _clean_chat_completion_response(response)
    response_payload = openai_chat_to_anthropic_message(
        response,
        model=openai_payload["model"],
        request_id=request_id,
    )
    answer = "\n".join(
        str(item.get("text") or "")
        for item in response_payload.get("content", [])
        if isinstance(item, dict)
    )
    _log_llm_exchange(
        endpoint=str(request.url.path),
        model=openai_payload.get("model"),
        stream=False,
        question=_messages_question(openai_payload),
        answer=answer,
        reasoning=_chat_completion_text(response)[1],
        tool_context=_tool_context(payload),
    )
    return response_payload


def _clean_chat_completion_response(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    choices = []
    for choice in result.get("choices", []):
        if not isinstance(choice, dict):
            choices.append(choice)
            continue
        clean_choice = dict(choice)
        message = clean_choice.get("message")
        if isinstance(message, dict):
            clean_message = dict(message)
            content = clean_message.get("content")
            if isinstance(content, str):
                visible, inline_reasoning = split_think_text(content)
                clean_message["content"] = visible
                if inline_reasoning and not clean_message.get("reasoning_content"):
                    clean_message["reasoning_content"] = inline_reasoning
            clean_choice["message"] = clean_message
        choices.append(clean_choice)
    result["choices"] = choices
    return result


async def _chat_completion_stream(
    stream: AsyncIterator[bytes],
    *,
    endpoint: str,
    model: str | None,
    question: str,
    tool_context: dict[str, Any] | None = None,
) -> AsyncIterator[bytes]:
    think_filter = ThinkTagStreamFilter()
    answer_parts: list[str] = []
    reasoning_parts: list[str] = []
    try:
        async for raw in stream:
            for line in raw.decode("utf-8", errors="ignore").splitlines():
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if not data:
                    continue
                if data == "[DONE]":
                    visible, reasoning = think_filter.flush()
                    if visible or reasoning:
                        if visible:
                            answer_parts.append(visible)
                        if reasoning:
                            reasoning_parts.append(reasoning)
                        yield _chat_stream_filter_flush_event(visible, reasoning)
                    _log_llm_exchange(
                        endpoint=endpoint,
                        model=model,
                        stream=True,
                        question=question,
                        answer="".join(answer_parts),
                        reasoning="".join(reasoning_parts),
                        tool_context=tool_context,
                    )
                    yield b"data: [DONE]\n\n"
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    yield raw
                    continue

                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta")
                if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                    visible, reasoning = think_filter.feed(delta["content"])
                    if reasoning:
                        delta["reasoning_content"] = (delta.get("reasoning_content") or "") + reasoning
                    delta["content"] = visible
                    if not visible and not reasoning and set(delta.keys()) == {"content"}:
                        continue
                if isinstance(delta, dict):
                    if isinstance(delta.get("content"), str):
                        delta["content"] = clean_visible_text(delta["content"])
                        answer_parts.append(delta["content"])
                    if isinstance(delta.get("reasoning_content"), str):
                        reasoning_parts.append(delta["reasoning_content"])
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose:
            await aclose()


async def _passthrough_stream(stream: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    try:
        async for chunk in stream:
            yield chunk
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose:
            await aclose()


def _chat_stream_filter_flush_event(visible: str, reasoning: str) -> bytes:
    delta: dict[str, Any] = {}
    if visible:
        delta["content"] = visible
    if reasoning:
        delta["reasoning_content"] = reasoning
    chunk = {
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": None,
            }
        ]
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")


async def _anthropic_stream(
    payload: dict[str, Any],
    stream: AsyncIterator[bytes],
    request_id: str,
    model: str,
    *,
    endpoint: str,
) -> AsyncIterator[bytes]:
    created = int(time.time())
    answer_parts: list[str] = []
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

    try:
        async for raw in stream:
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
                text = clean_visible_text(delta.get("content") or "")
                if text:
                    answer_parts.append(text)
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
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose:
            await aclose()

    yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield _sse("message_stop", {"type": "message_stop"})
    _log_llm_exchange(
        endpoint=endpoint,
        model=model,
        stream=True,
        question=_messages_question(payload),
        answer="".join(answer_parts),
    )


async def _responses_stream(
    payload: dict[str, Any],
    stream: AsyncIterator[bytes],
    response_id: str,
    created_at: int,
    *,
    endpoint: str,
    question: str,
    tool_context: dict[str, Any] | None = None,
) -> AsyncIterator[bytes]:
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
    tool_call_accumulator: dict[int, dict[str, Any]] = {}
    think_filter = ThinkTagStreamFilter()
    try:
        async for raw in stream:
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
                _merge_stream_tool_call_delta(tool_call_accumulator, delta.get("tool_calls"))
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
                inline_reasoning = ""
                if text:
                    text, inline_reasoning = think_filter.feed(text)
                if inline_reasoning:
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
                    reasoning_parts.append(inline_reasoning)
                    yield _sse(
                        "response.reasoning_text.delta",
                        {
                            "type": "response.reasoning_text.delta",
                            "item_id": reasoning_item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": inline_reasoning,
                        },
                    )
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
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose:
            await aclose()

    tail_text, tail_reasoning = think_filter.flush()
    if tail_reasoning:
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
        reasoning_parts.append(tail_reasoning)
        yield _sse(
            "response.reasoning_text.delta",
            {
                "type": "response.reasoning_text.delta",
                "item_id": reasoning_item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": tail_reasoning,
            },
        )
    if tail_text:
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
        answer_parts.append(tail_text)
        yield _sse(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": output_item_id,
                "output_index": 1 if reasoning_started else 0,
                "content_index": 0,
                "delta": tail_text,
            },
        )

    final_reasoning = "".join(reasoning_parts)
    final_text = "".join(answer_parts)
    output_index = 1 if reasoning_started else 0
    tool_call_items = openai_tool_calls_to_response_items(
        [tool_call_accumulator[index] for index in sorted(tool_call_accumulator)],
        response_id,
    )

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

    if not output_started and not tool_call_items:
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

    if output_started or not tool_call_items:
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

    next_output_index = int(reasoning_started) + int(output_started or not tool_call_items)
    for tool_index, tool_call_item in enumerate(tool_call_items, start=next_output_index):
        yield _sse(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": tool_index,
                "item": dict(tool_call_item, status="in_progress", arguments=""),
            },
        )
        yield _sse(
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "item_id": tool_call_item["id"],
                "output_index": tool_index,
                "delta": tool_call_item["arguments"],
            },
        )
        yield _sse(
            "response.function_call_arguments.done",
            {
                "type": "response.function_call_arguments.done",
                "item_id": tool_call_item["id"],
                "output_index": tool_index,
                "arguments": tool_call_item["arguments"],
            },
        )
        yield _sse(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": tool_index,
                "item": tool_call_item,
            },
        )
    completed = dict(response_shell)
    completed["status"] = "requires_action" if tool_call_items else "completed"
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
    if final_text or not tool_call_items:
        completed_output.append(
            {
                "id": output_item_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": final_text, "annotations": []}],
            }
        )
    completed_output.extend(tool_call_items)
    completed["output"] = completed_output
    completed["output_text"] = final_text
    yield _sse("response.completed", {"type": "response.completed", "response": completed})
    _log_llm_exchange(
        endpoint=endpoint,
        model=payload.get("model"),
        stream=True,
        question=question,
        answer=final_text,
        reasoning=final_reasoning,
        tool_context={**(tool_context or {}), "tool_calls_returned": len(tool_call_items)},
    )
    yield b"data: [DONE]\n\n"


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")
