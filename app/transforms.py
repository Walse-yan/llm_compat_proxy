from __future__ import annotations

import json
import re
from typing import Any


_STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "stop_sequence",
}

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_TAG_RE = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)
_TOOL_CALL_TAG_RE = re.compile(r"</?tool_call\b[^>]*>", re.IGNORECASE)


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


def strip_think_tags(text: str) -> str:
    without_blocks = _THINK_BLOCK_RE.sub("", text)
    return _THINK_TAG_RE.sub("", without_blocks)


def strip_pseudo_tool_call_tags(text: str) -> str:
    return _TOOL_CALL_TAG_RE.sub("", text)


def clean_visible_text(text: str) -> str:
    return strip_pseudo_tool_call_tags(strip_think_tags(text))


def split_think_text(text: str) -> tuple[str, str]:
    reasoning_parts = [match.group(0) for match in _THINK_BLOCK_RE.finditer(text)]
    reasoning = "\n".join(_THINK_TAG_RE.sub("", part).strip() for part in reasoning_parts if part)
    visible = clean_visible_text(text)
    return visible, reasoning


class ThinkTagStreamFilter:
    def __init__(self) -> None:
        self._buffer = ""
        self._in_think = False

    def feed(self, text: str) -> tuple[str, str]:
        self._buffer += text
        visible_parts: list[str] = []
        reasoning_parts: list[str] = []

        while self._buffer:
            lower = self._buffer.lower()
            if self._in_think:
                close_index = lower.find("</think>")
                if close_index == -1:
                    keep = min(len(self._buffer), 7)
                    if len(self._buffer) > keep:
                        reasoning_parts.append(self._buffer[:-keep])
                        self._buffer = self._buffer[-keep:]
                    break
                reasoning_parts.append(self._buffer[:close_index])
                self._buffer = self._buffer[close_index + len("</think>") :]
                self._in_think = False
                continue

            open_index = lower.find("<think")
            close_index = lower.find("</think>")
            tag_index_candidates = [index for index in (open_index, close_index) if index != -1]
            if not tag_index_candidates:
                keep = min(len(self._buffer), 7)
                if len(self._buffer) > keep:
                    visible_parts.append(self._buffer[:-keep])
                    self._buffer = self._buffer[-keep:]
                break

            tag_index = min(tag_index_candidates)
            visible_parts.append(self._buffer[:tag_index])
            if tag_index == close_index:
                self._buffer = self._buffer[close_index + len("</think>") :]
                continue

            end_index = self._buffer.find(">", tag_index)
            if end_index == -1:
                break
            self._buffer = self._buffer[end_index + 1 :]
            self._in_think = True

        return "".join(visible_parts), "".join(reasoning_parts)

    def flush(self) -> tuple[str, str]:
        if not self._buffer:
            return "", ""
        text = self._buffer
        self._buffer = ""
        if self._in_think:
            self._in_think = False
            return "", clean_visible_text(text)
        return clean_visible_text(text), ""


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
        normalized_message: dict[str, Any] = {
            "role": role,
            "content": _content_to_text(message.get("content", "")),
        }
        for key in ("name", "tool_call_id"):
            if key in message and message[key] is not None:
                normalized_message[key] = message[key]
        if message.get("tool_calls"):
            normalized_message["tool_calls"] = message["tool_calls"]
        normalized_messages.append(normalized_message)

    result["messages"] = normalized_messages
    return result


def responses_to_openai_chat(payload: dict[str, Any], default_model: str) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
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
                item_type = item.get("type")
                if item_type == "function_call_output":
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": item.get("call_id") or item.get("id") or "",
                            "content": _content_to_text(item.get("output", "")),
                        }
                    )
                    continue
                if item_type == "function_call":
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": item.get("call_id") or item.get("id") or "",
                                    "type": "function",
                                    "function": {
                                        "name": item.get("name") or "",
                                        "arguments": item.get("arguments") or "{}",
                                    },
                                }
                            ],
                        }
                    )
                    continue
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

    tools = responses_tools_to_openai_tools(payload.get("tools") or [])
    if tools:
        result["tools"] = tools
        result["tool_choice"] = payload.get("tool_choice") or "auto"
    if "parallel_tool_calls" in payload:
        result["parallel_tool_calls"] = payload["parallel_tool_calls"]

    return normalize_openai_chat_payload(result, default_model=default_model)


def responses_tools_to_openai_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []

    result: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            result.append(tool)
            continue

        name = tool.get("name") or tool.get("type")
        if not name:
            continue

        parameters = tool.get("parameters") or tool.get("input_schema")
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}, "additionalProperties": True}

        result.append(
            {
                "type": "function",
                "function": {
                    "name": str(name),
                    "description": str(tool.get("description") or f"Codex tool: {name}"),
                    "parameters": parameters,
                },
            }
        )
    return result


def openai_tool_calls_to_response_items(tool_calls: Any, response_id: str) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []

    response_suffix = response_id.removeprefix("resp_")
    items: list[dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        name = function.get("name") or tool_call.get("name")
        if not name:
            continue
        arguments = function.get("arguments")
        if arguments is None:
            arguments = tool_call.get("arguments") or "{}"
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)

        call_id = tool_call.get("id") or tool_call.get("call_id") or f"call_{response_suffix}_{index}"
        items.append(
            {
                "id": f"fc_{response_suffix}_{index}",
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": str(name),
                "arguments": arguments,
            }
        )
    return items


def openai_chat_to_response(payload: dict[str, Any], model: str, response_id: str, created_at: int) -> dict[str, Any]:
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content") or ""
    text, inline_reasoning = split_think_text(text)
    reasoning_text = message.get("reasoning_content") or inline_reasoning
    tool_call_items = openai_tool_calls_to_response_items(message.get("tool_calls"), response_id)
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

    if text or reasoning_text or not tool_call_items:
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

    output.extend(tool_call_items)

    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "requires_action" if tool_call_items else "completed",
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
    content = strip_think_tags(message.get("content") or message.get("reasoning_content") or "")
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
