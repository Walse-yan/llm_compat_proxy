from app.transforms import (
    anthropic_messages_to_openai,
    normalize_openai_chat_payload,
    openai_chat_to_anthropic_message,
    openai_chat_to_response,
)


def test_anthropic_messages_to_openai_moves_system_and_text_parts():
    payload = {
        "model": "claude-3-5-sonnet-latest",
        "system": "You are concise.",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "world"},
                ],
            }
        ],
        "max_tokens": 128,
        "temperature": 0.2,
        "stop_sequences": ["END"],
    }

    result = anthropic_messages_to_openai(payload, default_model="local-model")

    assert result == {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Hello\nworld"},
        ],
        "max_tokens": 128,
        "temperature": 0.2,
        "stop": ["END"],
        "stream": False,
    }


def test_openai_chat_to_anthropic_message_maps_content_and_usage():
    openai_payload = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hi there"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }

    result = openai_chat_to_anthropic_message(
        openai_payload,
        model="local-model",
        request_id="msg_test",
    )

    assert result == {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "local-model",
        "content": [{"type": "text", "text": "Hi there"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 3},
    }


def test_normalize_openai_chat_payload_flattens_lobehub_content_items():
    payload = {
        "model": "local-model",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "在吗"}],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "reasoning", "reasoning": "Need answer briefly."},
                    {"type": "text", "text": "在呢！"},
                ],
                "tool_calls": [],
            },
            {
                "role": "user",
                "content": [{"text": "空间转录是什么"}],
            },
        ],
        "stream": True,
    }

    result = normalize_openai_chat_payload(payload, default_model="local-model")

    assert result["messages"] == [
        {"role": "user", "content": "在吗"},
        {"role": "assistant", "content": "Need answer briefly.\n在呢！"},
        {"role": "user", "content": "空间转录是什么"},
    ]
    assert "tool_calls" not in result["messages"][1]
    assert result["stream"] is True


def test_openai_chat_to_response_keeps_reasoning_separate_from_answer():
    openai_payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "reasoning_content": "Think first.",
                    "content": "Final answer.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    result = openai_chat_to_response(
        openai_payload,
        model="local-model",
        response_id="resp_test",
        created_at=123,
    )

    assert result["output_text"] == "Final answer."
    assert result["output"][0] == {
        "id": "rs_test",
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "Think first."}],
        "content": [{"type": "reasoning_text", "text": "Think first."}],
    }
    assert result["output"][1]["content"] == [
        {"type": "output_text", "text": "Final answer.", "annotations": []}
    ]
