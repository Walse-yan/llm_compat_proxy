from __future__ import annotations

import json
from typing import Any


_STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "stop_sequence",
}


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, dict) and item.get("type") == "tool_result":
                parts.append(_content_to_text(item.get("content", "")))
            elif isinstance(item, dict) and item.get("type") in {"reasoning", "thinking"}:
                parts.append(str(item.get("reasoning") or item.get("thinking") or item.get("text") or ""))
            elif isinstance(item, dict):
                for key in ("text", "content", "reasoning", "thinking", "summary"):
                    if key in item and item[key]:
                        parts.append(_content_to_text(item[key]))
                        break
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(part for part in parts if part)
    return str(content)


def normalize_openai_chat_payload(payload: dict[str, Any], default_model: str) -> dict[str, Any]:
    result = dict(payload)
    result["model"] = result.get("model") or default_model
    normalized_messages: list[dict[str, Any]] = []

    for message in result.get("messages", []):
        if not isinstance(message, dict):
            continue
        role = message.get("role", "user")
        if role not in {"system", "user", "assistant", "tool"}:
            role = "user"
        normalized_messages.append(
            {
                "role": role,
                "content": _content_to_text(message.get("content", "")),
            }
        )

    result["messages"] = normalized_messages
    return result


def responses_to_openai_chat(payload: dict[str, Any], default_model: str) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    instructions = payload.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": _content_to_text(instructions)})

    input_value = payload.get("input", "")
    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
    elif isinstance(input_value, list):
        for item in input_value:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                role = item.get("role", "user")
                if role not in {"system", "user", "assistant", "tool"}:
                    role = "user"
                messages.append({"role": role, "content": _content_to_text(item.get("content", ""))})

    result: dict[str, Any] = {
        "model": payload.get("model") or default_model,
        "messages": messages,
        "stream": bool(payload.get("stream", False)),
    }

    field_map = {
        "max_output_tokens": "max_tokens",
        "max_tokens": "max_tokens",
        "temperature": "temperature",
        "top_p": "top_p",
        "stop": "stop",
    }
    for source, target in field_map.items():
        if source in payload and payload[source] is not None:
            result[target] = payload[source]

    return normalize_openai_chat_payload(result, default_model=default_model)


def openai_chat_to_response(payload: dict[str, Any], model: str, response_id: str, created_at: int) -> dict[str, Any]:
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content") or ""
    reasoning_text = message.get("reasoning_content") or ""
    usage = payload.get("usage") or {}
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    output: list[dict[str, Any]] = []

    if reasoning_text:
        output.append(
            {
                "id": f"rs_{response_id.removeprefix('resp_')}",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": reasoning_text}],
                "content": [{"type": "reasoning_text", "text": reasoning_text}],
            }
        )

    output.append(
        {
            "id": f"msg_{response_id.removeprefix('resp_')}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": text or reasoning_text,
                    "annotations": [],
                }
            ],
        }
    )

    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "model": model,
        "output": output,
        "output_text": text or reasoning_text,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": usage.get("total_tokens", input_tokens + output_tokens),
        },
    }


def anthropic_messages_to_openai(payload: dict[str, Any], default_model: str) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    system = payload.get("system")
    if system:
        messages.append({"role": "system", "content": _content_to_text(system)})

    for message in payload.get("messages", []):
        role = message.get("role", "user")
        if role not in {"user", "assistant", "system"}:
            role = "user"
        messages.append({"role": role, "content": _content_to_text(message.get("content", ""))})

    result: dict[str, Any] = {
        "model": default_model,
        "messages": messages,
        "stream": bool(payload.get("stream", False)),
    }

    field_map = {
        "max_tokens": "max_tokens",
        "temperature": "temperature",
        "top_p": "top_p",
    }
    for source, target in field_map.items():
        if source in payload and payload[source] is not None:
            result[target] = payload[source]

    if payload.get("stop_sequences"):
        result["stop"] = payload["stop_sequences"]

    return result


def openai_chat_to_anthropic_message(
    payload: dict[str, Any],
    model: str,
    request_id: str,
) -> dict[str, Any]:
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or message.get("reasoning_content") or ""
    finish_reason = choice.get("finish_reason")
    usage = payload.get("usage") or {}

    return {
        "id": request_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": content}],
        "stop_reason": _STOP_REASON_MAP.get(finish_reason, finish_reason or "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def make_openai_model_list(model_id: str) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


def make_anthropic_model_list(model_id: str) -> dict[str, Any]:
    return {
        "data": [
            {
                "id": model_id,
                "type": "model",
                "display_name": model_id,
                "created_at": "2026-05-02T00:00:00Z",
            }
        ],
        "has_more": False,
        "first_id": model_id,
        "last_id": model_id,
    }
