from __future__ import annotations

import copy
import json
import time
from collections import OrderedDict
from typing import Any

from .config import (
    EMBEDDINGS_CACHE_ENABLED,
    EMBEDDINGS_CACHE_MAX_ITEMS,
    MODELS_CACHE_TTL_SECONDS,
)


_embeddings_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
_models_cache: tuple[float, dict[str, Any]] | None = None


def make_cache_key(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def get_embeddings_cache(key: str) -> dict[str, Any] | None:
    if not EMBEDDINGS_CACHE_ENABLED:
        return None
    value = _embeddings_cache.get(key)
    if value is None:
        return None
    _embeddings_cache.move_to_end(key)
    return copy.deepcopy(value)


def set_embeddings_cache(key: str, value: dict[str, Any]) -> None:
    if not EMBEDDINGS_CACHE_ENABLED or EMBEDDINGS_CACHE_MAX_ITEMS <= 0:
        return
    _embeddings_cache[key] = copy.deepcopy(value)
    _embeddings_cache.move_to_end(key)
    while len(_embeddings_cache) > EMBEDDINGS_CACHE_MAX_ITEMS:
        _embeddings_cache.popitem(last=False)


def get_models_cache() -> dict[str, Any] | None:
    if MODELS_CACHE_TTL_SECONDS <= 0:
        return None
    if _models_cache is None:
        return None
    expires_at, value = _models_cache
    if time.monotonic() >= expires_at:
        return None
    return copy.deepcopy(value)


def set_models_cache(value: dict[str, Any]) -> None:
    global _models_cache
    if MODELS_CACHE_TTL_SECONDS <= 0:
        return
    _models_cache = (time.monotonic() + MODELS_CACHE_TTL_SECONDS, copy.deepcopy(value))


def clear_caches() -> None:
    global _models_cache
    _embeddings_cache.clear()
    _models_cache = None


def cache_status() -> dict[str, Any]:
    return {
        "embeddings": {
            "enabled": EMBEDDINGS_CACHE_ENABLED,
            "items": len(_embeddings_cache),
            "max_items": EMBEDDINGS_CACHE_MAX_ITEMS,
        },
        "models": {
            "enabled": MODELS_CACHE_TTL_SECONDS > 0,
            "ttl_seconds": MODELS_CACHE_TTL_SECONDS,
            "items": 1 if _models_cache is not None else 0,
        },
    }
