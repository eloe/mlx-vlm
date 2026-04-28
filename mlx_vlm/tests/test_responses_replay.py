"""Pure unit tests for Responses replay helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest


def _ensure_package():
    package = types.ModuleType("mlx_vlm")
    package.__path__ = [str(Path(__file__).parent.parent)]
    sys.modules.setdefault("mlx_vlm", package)


def _load_module(name: str, filename: str):
    """Load a sibling mlx_vlm module without importing mlx_vlm.__init__."""
    _ensure_package()
    module_name = f"mlx_vlm.{name}"
    module_path = Path(__file__).parent.parent / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


responses_store = _load_module("responses_store", "responses_store.py")
responses_replay = _load_module("responses_replay", "responses_replay.py")

ResponseStore = responses_store.ResponseStore


def test_resolve_replay_uses_previous_response_context():
    store = ResponseStore()
    store.save(
        "resp_1",
        "Hello",
        [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi there"}],
            }
        ],
    )

    expanded = responses_replay.resolve_responses_input_items(
        "Follow up",
        previous_response_id="resp_1",
        response_store=store,
    )

    assert expanded == [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hi there"}],
        },
        {"role": "user", "content": "Follow up"},
    ]


def test_resolve_replay_supports_chained_previous_response_ids():
    store = ResponseStore()
    first_input = "Hello"
    first_output = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hi there"}],
        }
    ]
    store.save("resp_1", first_input, first_output)

    second_expanded = responses_replay.resolve_responses_input_items(
        "Follow up",
        previous_response_id="resp_1",
        response_store=store,
    )
    store.save(
        "resp_2",
        second_expanded,
        [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Second answer"}],
            }
        ],
    )

    third_expanded = responses_replay.resolve_responses_input_items(
        "Third turn",
        previous_response_id="resp_2",
        response_store=store,
    )

    assert third_expanded == [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hi there"}],
        },
        {"role": "user", "content": "Follow up"},
        {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Second answer"}],
        },
        {"role": "user", "content": "Third turn"},
    ]


def test_resolve_replay_missing_previous_response_raises_lookup_error():
    store = ResponseStore()

    with pytest.raises(LookupError, match="resp_missing"):
        responses_replay.resolve_responses_input_items(
            "Hello",
            previous_response_id="resp_missing",
            response_store=store,
        )


def test_responses_input_to_messages_extracts_text_images_and_instructions():
    chat_messages, images, instructions = responses_replay.responses_input_to_messages(
        [
            {"role": "system", "content": "Be brief."},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this"},
                    {"type": "input_image", "image_url": "data:image/png;base64,abc"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "It is a cat."}],
            },
        ]
    )

    assert chat_messages == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Describe this"},
        {"role": "assistant", "content": "It is a cat."},
    ]
    assert images == ["data:image/png;base64,abc"]
    assert instructions == "Be brief."


def test_responses_input_to_messages_converts_function_calls():
    chat_messages, images, instructions = responses_replay.responses_input_to_messages(
        [
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "lookup",
                "arguments": '{"city":"NYC"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "72F",
            },
        ]
    )

    assert chat_messages == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"city":"NYC"}',
                    },
                }
            ],
        },
        {"role": "tool", "content": "72F", "tool_call_id": "call_123"},
    ]
    assert images == []
    assert instructions is None
