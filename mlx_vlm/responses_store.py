"""Standalone in-memory LRU store for Responses API replay state."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import threading
from typing import Any, Mapping


@dataclass(frozen=True)
class StoredResponse:
    """Snapshot of a response's input and output items."""

    input_items: Any
    output_items: list[Any]


class ResponseStore:
    """Bounded thread-safe LRU store keyed by response id."""

    def __init__(self, maxsize: int = 256):
        if maxsize < 1:
            raise ValueError("maxsize must be at least 1")

        self._store: OrderedDict[str, StoredResponse] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def save(
        self,
        response_id: str,
        input_items: Any,
        response_output: list[Any],
    ) -> None:
        """Store a replayable snapshot for a response id."""
        entry = StoredResponse(
            input_items=deepcopy(input_items),
            output_items=deepcopy(response_output),
        )

        with self._lock:
            if response_id in self._store:
                self._store.move_to_end(response_id)
            self._store[response_id] = entry
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)

    def get(self, response_id: str) -> StoredResponse | None:
        """Return a stored snapshot and mark it as recently used."""
        with self._lock:
            entry = self._store.get(response_id)
            if entry is None:
                return None
            self._store.move_to_end(response_id)
            return deepcopy(entry)

    def replay_input(self, response_id: str) -> list[dict[str, Any]] | None:
        """Rebuild prior request state as input items for future replay."""
        entry = self.get(response_id)
        if entry is None:
            return None

        items: list[dict[str, Any]] = []
        original_input = entry.input_items

        if isinstance(original_input, str):
            items.append({"role": "user", "content": original_input})
        elif isinstance(original_input, list):
            items.extend(deepcopy(original_input))

        for output_item in entry.output_items:
            output_dict = _maybe_mapping(output_item)
            if output_dict is None:
                continue

            item_type = output_dict.get("type", "")
            if item_type == "message":
                content = output_dict.get("content", [])
                output_text_parts = []
                for part in content:
                    part_dict = _maybe_mapping(part)
                    if part_dict is None:
                        continue
                    if part_dict.get("type") == "output_text":
                        output_text_parts.append(
                            {
                                "type": "output_text",
                                "text": part_dict.get("text", ""),
                            }
                        )

                if output_text_parts:
                    items.append(
                        {
                            "role": output_dict.get("role", "assistant"),
                            "content": output_text_parts,
                        }
                    )
            elif item_type == "function_call":
                items.append(
                    {
                        "type": "function_call",
                        "call_id": output_dict.get("call_id", ""),
                        "name": output_dict.get("name", ""),
                        "arguments": output_dict.get("arguments", ""),
                    }
                )

        return items

    def clear(self) -> None:
        """Remove all stored entries."""
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


def _maybe_mapping(value: Any) -> Mapping[str, Any] | None:
    """Normalize dict-like or model-like objects into mappings."""
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return None
