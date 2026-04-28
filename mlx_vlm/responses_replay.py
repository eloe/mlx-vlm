"""Helpers for expanding and parsing Responses API replay context."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .responses_store import ResponseStore


def resolve_responses_input_items(
    input_items: str | list[Any],
    previous_response_id: str | None = None,
    response_store: ResponseStore | None = None,
) -> str | list[Any]:
    """Expand ``previous_response_id`` into replayable response input items."""
    if previous_response_id is None:
        return deepcopy(input_items)

    if response_store is None:
        raise ValueError("response_store is required when previous_response_id is set.")

    replayed = response_store.replay_input(previous_response_id)
    if replayed is None:
        raise LookupError(previous_response_id)

    if isinstance(input_items, str):
        current_items: list[Any] = [{"role": "user", "content": input_items}]
    elif isinstance(input_items, list):
        current_items = deepcopy(input_items)
    else:
        raise ValueError("Invalid input format.")

    return replayed + current_items


def responses_input_to_messages(
    input_items: str | list[Any],
) -> tuple[list[dict[str, Any]], list[str], str | None]:
    """Convert Responses API input items into server chat messages and images."""
    if isinstance(input_items, str):
        return [{"role": "user", "content": input_items}], [], None
    if not isinstance(input_items, list):
        raise ValueError("Invalid input format.")

    chat_messages: list[dict[str, Any]] = []
    images: list[str] = []
    instructions: str | None = None

    for message in input_items:
        message_dict = _as_mapping(message)
        if message_dict is None:
            raise ValueError("Invalid input format.")

        item_type = message_dict.get("type", "")
        if item_type == "function_call_output":
            chat_messages.append(
                {
                    "role": "tool",
                    "content": message_dict.get("output", ""),
                    "tool_call_id": message_dict.get("call_id", "unknown"),
                }
            )
            continue

        if item_type == "function_call":
            chat_messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": message_dict.get("call_id", ""),
                            "type": "function",
                            "function": {
                                "name": message_dict.get("name", ""),
                                "arguments": message_dict.get("arguments", ""),
                            },
                        }
                    ],
                }
            )
            continue

        role = message_dict.get("role")
        if role is None:
            raise ValueError("Invalid input format.")

        text_content, new_images = _extract_content(role, message_dict.get("content"))
        chat_message: dict[str, Any] = {"role": role, "content": text_content}

        if message_dict.get("tool_calls") is not None:
            chat_message["tool_calls"] = deepcopy(message_dict["tool_calls"])
        if message_dict.get("tool_call_id") is not None:
            chat_message["tool_call_id"] = message_dict["tool_call_id"]
        if message_dict.get("name") is not None:
            chat_message["name"] = message_dict["name"]

        chat_messages.append(chat_message)
        images.extend(new_images)

        if role in {"system", "developer"} and text_content:
            instructions = text_content

    return chat_messages, images, instructions


def _extract_content(role: str, content: Any) -> tuple[str | None, list[str]]:
    if content is None or isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        raise ValueError("Invalid input format.")

    text_parts: list[str] = []
    images: list[str] = []
    for item in content:
        item_dict = _as_mapping(item)
        if item_dict is None:
            raise ValueError("Missing type in input item.")

        item_type = item_dict.get("type", "")
        if item_type in {"input_text", "text", "output_text"}:
            text_parts.append(item_dict.get("text", ""))
        elif role == "user" and item_type == "input_image":
            images.append(item_dict.get("image_url", ""))
        elif role == "user" and item_type == "image_url":
            image_url = item_dict.get("image_url", {})
            if isinstance(image_url, Mapping):
                images.append(image_url.get("url", ""))
            else:
                raise ValueError("Invalid input item type.")
        else:
            raise ValueError("Invalid input item type.")

    return "".join(text_parts), images


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return None
