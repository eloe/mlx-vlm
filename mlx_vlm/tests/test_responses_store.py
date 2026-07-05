"""Pure unit tests for the standalone response store module."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


def _load_module(name: str, filename: str):
    """Load a sibling module without importing mlx_vlm.__init__."""
    module_path = Path(__file__).parent.parent / filename
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


responses_store = _load_module("responses_store", "responses_store.py")
ResponseStore = responses_store.ResponseStore
StoredResponse = responses_store.StoredResponse


class DumpableItem:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


def test_store_rejects_invalid_maxsize():
    with pytest.raises(ValueError, match="maxsize"):
        ResponseStore(maxsize=0)


def test_store_save_and_get_round_trip():
    store = ResponseStore()
    store.save("resp_1", "hello", [{"type": "message"}])

    entry = store.get("resp_1")

    assert entry == StoredResponse(
        input_items="hello",
        output_items=[{"type": "message"}],
    )


def test_store_returns_snapshots_not_live_references():
    input_items = [{"role": "user", "content": "hello"}]
    output_items = [{"type": "function_call", "name": "lookup", "arguments": "{}"}]

    store = ResponseStore()
    store.save("resp_1", input_items, output_items)

    input_items[0]["content"] = "mutated"
    output_items[0]["name"] = "changed"

    entry = store.get("resp_1")
    assert entry is not None
    assert entry.input_items[0]["content"] == "hello"
    assert entry.output_items[0]["name"] == "lookup"

    entry.output_items[0]["name"] = "client-change"
    assert store.get("resp_1").output_items[0]["name"] == "lookup"


def test_store_get_missing_returns_none():
    store = ResponseStore()
    assert store.get("resp_missing") is None


def test_store_lru_eviction_respects_recent_reads():
    store = ResponseStore(maxsize=2)
    store.save("resp_a", "a", [])
    store.save("resp_b", "b", [])

    assert store.get("resp_a") is not None

    store.save("resp_c", "c", [])

    assert store.get("resp_a") is not None
    assert store.get("resp_b") is None
    assert store.get("resp_c") is not None


def test_store_overwrite_does_not_grow_store():
    store = ResponseStore(maxsize=2)
    store.save("resp_1", "a", [])
    store.save("resp_1", "b", [{"type": "message"}])

    assert len(store) == 1
    assert store.get("resp_1") == StoredResponse(
        input_items="b",
        output_items=[{"type": "message"}],
    )


def test_replay_input_rehydrates_string_input_and_output_text():
    store = ResponseStore()
    store.save(
        "resp_1",
        "hello",
        [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Hello"},
                    {"type": "ignored", "text": "skip"},
                    {"type": "output_text", "text": " world"},
                ],
            }
        ],
    )

    replay_items = store.replay_input("resp_1")

    assert replay_items == [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "Hello"},
                {"type": "output_text", "text": " world"},
            ],
        },
    ]


def test_replay_input_preserves_message_list_input():
    original = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    store = ResponseStore()
    store.save("resp_1", original, [])

    replay_items = store.replay_input("resp_1")

    assert replay_items == original
    replay_items[0]["content"] = "changed"
    assert store.get("resp_1").input_items[0]["content"] == "You are helpful."


def test_replay_input_includes_function_calls():
    store = ResponseStore()
    store.save(
        "resp_1",
        "hello",
        [
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "get_weather",
                "arguments": '{"city":"NYC"}',
            }
        ],
    )

    replay_items = store.replay_input("resp_1")

    assert replay_items == [
        {"role": "user", "content": "hello"},
        {
            "type": "function_call",
            "call_id": "call_123",
            "name": "get_weather",
            "arguments": '{"city":"NYC"}',
        },
    ]


def test_replay_input_accepts_model_dump_items():
    store = ResponseStore()
    store.save(
        "resp_1",
        "hello",
        [
            DumpableItem(
                {
                    "type": "message",
                    "content": [DumpableItem({"type": "output_text", "text": "Hi"})],
                }
            )
        ],
    )

    assert store.replay_input("resp_1") == [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hi"}],
        },
    ]


def test_store_clear_resets_entries():
    store = ResponseStore()
    store.save("resp_1", "a", [])
    store.save("resp_2", "b", [])

    store.clear()

    assert len(store) == 0
    assert store.get("resp_1") is None
    assert store.replay_input("resp_2") is None
