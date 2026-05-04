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
    stream = await open_stream_post(path, payload)
    try:
        async for chunk in stream:
            yield chunk
    finally:
        await stream.aclose()


class UpstreamByteStream:
    def __init__(self, client: httpx.AsyncClient, response: httpx.Response) -> None:
        self.client = client
        self.response = response

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iter_bytes()

    async def _iter_bytes(self) -> AsyncIterator[bytes]:
        async for chunk in self.response.aiter_bytes():
            if chunk:
                yield chunk

    async def aclose(self) -> None:
        await self.response.aclose()
        await self.client.aclose()


async def open_stream_post(path: str, payload: dict[str, Any]) -> UpstreamByteStream:
    client = httpx.AsyncClient(timeout=_timeout(), trust_env=False)
    try:
        request = client.build_request("POST", f"{LLAMA_CPP_BASE_URL}{path}", json=payload)
        response = await client.send(request, stream=True)
        response.raise_for_status()
        return UpstreamByteStream(client, response)
    except Exception:
        await client.aclose()
        raise


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


async def open_chat_completion_stream(payload: dict[str, Any]) -> UpstreamByteStream:
    return await open_stream_post("/v1/chat/completions", payload)


async def open_completion_stream(payload: dict[str, Any]) -> UpstreamByteStream:
    return await open_stream_post("/v1/completions", payload)
