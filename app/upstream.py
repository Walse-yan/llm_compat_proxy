from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from .config import LLAMA_CPP_BASE_URL, UPSTREAM_TIMEOUT_SECONDS


def _timeout() -> httpx.Timeout:
    return httpx.Timeout(UPSTREAM_TIMEOUT_SECONDS, connect=30.0)


async def post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=_timeout(), trust_env=False) as client:
        response = await client.post(f"{LLAMA_CPP_BASE_URL}{path}", json=payload)
        response.raise_for_status()
        return response.json()


async def get_json(path: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=_timeout(), trust_env=False) as client:
        response = await client.get(f"{LLAMA_CPP_BASE_URL}{path}")
        response.raise_for_status()
        return response.json()


async def stream_post(path: str, payload: dict[str, Any]) -> AsyncIterator[bytes]:
    async with httpx.AsyncClient(timeout=_timeout(), trust_env=False) as client:
        async with client.stream("POST", f"{LLAMA_CPP_BASE_URL}{path}", json=payload) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                if chunk:
                    yield chunk


async def chat_completion(payload: dict[str, Any]) -> dict[str, Any]:
    return await post_json("/v1/chat/completions", payload)


async def completion(payload: dict[str, Any]) -> dict[str, Any]:
    return await post_json("/v1/completions", payload)


async def embeddings(payload: dict[str, Any]) -> dict[str, Any]:
    return await post_json("/v1/embeddings", payload)


async def model_list() -> dict[str, Any]:
    return await get_json("/v1/models")


def chat_completion_stream(payload: dict[str, Any]) -> AsyncIterator[bytes]:
    return stream_post("/v1/chat/completions", payload)


def completion_stream(payload: dict[str, Any]) -> AsyncIterator[bytes]:
    return stream_post("/v1/completions", payload)
