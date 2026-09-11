# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One stable-prefix boundary policy, applied after provider projection."""

from collections.abc import Mapping
from typing import Any, Literal


def reject_legacy_cache_config(config: Mapping[str, Any]) -> None:
    extra = config.get("extra_body")
    if "cache_control_injection_points" in config or (
        isinstance(extra, Mapping) and "cache_control_injection_points" in extra
    ):
        raise ValueError(
            "cache_control_injection_points was removed. Use cache_breakpoint="
            "'auto', 'anthropic', 'openai' (Responses only), or None; place "
            "{'role': 'metadata', 'nooa_cache_boundary': True} before dynamic context."
        )


def _mark_responses_text(content: Any) -> tuple[Any, bool]:
    """Attach an OpenAI explicit breakpoint to the last input-text block."""
    marker = {"mode": "explicit"}
    if isinstance(content, str):
        return [
            {
                "type": "input_text",
                "text": content,
                "prompt_cache_breakpoint": marker,
            }
        ], True
    if isinstance(content, list):
        for index in range(len(content) - 1, -1, -1):
            block = content[index]
            if isinstance(block, dict) and block.get("type") == "input_text":
                updated = list(content)
                updated[index] = {**block, "prompt_cache_breakpoint": marker}
                return updated, True
    return content, False


def _mark_responses_cache_breakpoint(messages: list[dict[str, Any]], boundary: int) -> bool:
    """Mark the latest eligible Responses input block before ``boundary``."""
    for index in range(boundary - 1, -1, -1):
        item = messages[index]
        if item.get("type") == "function_call_output":
            output, marked = _mark_responses_text(item.get("output"))
            if marked:
                messages[index] = {**item, "output": output}
                return True
        # Assistant output uses output_text, which is not an eligible input block.
        if item.get("role") in {"system", "developer", "user"}:
            content, marked = _mark_responses_text(item.get("content"))
            if marked:
                messages[index] = {**item, "content": content}
                return True
    return False


def enable_openai_explicit_cache(api_params: dict[str, Any]) -> None:
    extra = api_params.get("extra_body")
    if extra is not None and not isinstance(extra, Mapping):
        raise ValueError("extra_body must be a mapping")
    extra = dict(extra or {})
    options = extra.get("prompt_cache_options")
    if options is not None and not isinstance(options, Mapping):
        raise ValueError("extra_body.prompt_cache_options must be a mapping")
    extra["prompt_cache_options"] = {**(options or {}), "mode": "explicit"}
    api_params["extra_body"] = extra


def _mark_anthropic(message: dict[str, Any], *, responses: bool) -> dict[str, Any] | None:
    content = message.get("content")
    marker = {"type": "ephemeral"}
    if isinstance(content, str) and content:
        return {
            **message,
            "content": [
                {
                    "type": "input_text" if responses else "text",
                    "text": content,
                    "cache_control": marker,
                }
            ],
        }
    if isinstance(content, list):
        for i in range(len(content) - 1, -1, -1):
            block = content[i]
            if isinstance(block, dict) and block.get("type") in {
                "text",
                "input_text",
                "output_text",
                "tool_result",
                "image",
            }:
                blocks = list(content)
                blocks[i] = {**block, "cache_control": marker}
                return {**message, "content": blocks}
    return None


def validate_cache_boundary(message: Mapping[str, Any]) -> None:
    """Reject malformed markers before projection can discard their fields."""
    if message["nooa_cache_boundary"] is not True:
        raise ValueError("nooa_cache_boundary must be true")
    if message.get("role") != "metadata" or set(message) != {"role", "nooa_cache_boundary"}:
        raise ValueError(
            "Use a separate {'role': 'metadata', 'nooa_cache_boundary': True} "
            "element before dynamic context, not a marker on a model message."
        )


def apply_cache_policy(
    messages: list[dict[str, Any]],
    mapping: Literal["anthropic", "openai"] | None,
    *,
    responses: bool,
    instructions: str | None = None,
) -> tuple[list[dict[str, Any]], str | None, bool]:
    """Consume one boundary; direct callers default to their leading instructions."""
    clean = []
    boundary = None
    for message in messages:
        if "nooa_cache_boundary" in message:
            validate_cache_boundary(message)
            if boundary is not None:
                raise ValueError("Rendered history contains more than one cache boundary")
            boundary = len(clean)
            continue
        clean.append(message)
    if mapping is None:
        return clean, instructions, False
    if boundary is None:
        boundary = 0
        for message in clean:
            if message.get("role") not in {"system", "developer"}:
                break
            boundary += 1
    if mapping == "anthropic":
        # Responses normally lifts stable system text into instructions. Put it
        # back into an eligible block only when this bridge needs a cache marker.
        if instructions:
            clean.insert(0, {"role": "system", "content": instructions})
            instructions = None
            boundary += 1
        for i in range(boundary - 1, -1, -1):
            marked = _mark_anthropic(clean[i], responses=responses)
            if marked is not None:
                clean[i] = marked
                break
        return clean, instructions, False
    if not responses:
        raise ValueError("The OpenAI explicit cache mapping requires ResponsesClient")
    marked = _mark_responses_cache_breakpoint(clean, boundary)
    if not marked and instructions:
        content, _ = _mark_responses_text(instructions)
        clean.insert(0, {"role": "system", "content": content})
        instructions = None
    # No eligible stable text: explicit mode deliberately avoids caching a
    # changing suffix. Never invent an empty text block just to host a marker.
    return clean, instructions, True
