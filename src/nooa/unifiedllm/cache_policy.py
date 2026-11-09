# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One stable-prefix boundary policy, applied after provider projection."""

import logging
from collections.abc import Mapping
from typing import Any, Literal

from nooa.llm_types import CacheBoundary

logger = logging.getLogger(__name__)


def reject_legacy_cache_config(config: Mapping[str, Any]) -> None:
    extra = config.get("extra_body")
    if "cache_control_injection_points" in config or (
        isinstance(extra, Mapping) and "cache_control_injection_points" in extra
    ):
        raise ValueError(
            "cache_control_injection_points was removed. Use cache_breakpoint="
            "'auto', 'anthropic', 'openai' (Responses only), or None; place "
            "CacheBoundary() before dynamic context."
        )


def wrap_responses_text(text: str, kind: str = "input_text") -> dict[str, Any]:
    """The one stable Responses content-block shape for a string.

    ``input_text`` for every non-assistant message, ``output_text`` for
    assistant messages -- the wire-shape-stability wrap outside this module
    (``unifiedllm._transform_messages``, run for every message regardless of
    whether it's marked this turn) and marking below must produce the exact
    same block shape, or a message renders differently once it stops being
    "the newest eligible block" and the provider's literal-prefix cache
    match breaks there on every later turn.
    """
    return {"type": kind, "text": text}


def _mark_responses_content(content: Any) -> tuple[Any, bool]:
    """Mark the last cacheable input block, including stable images and files."""
    marker = {"mode": "explicit"}
    if isinstance(content, str):
        return [{**wrap_responses_text(content), "prompt_cache_breakpoint": marker}], True
    if isinstance(content, list):
        for index in range(len(content) - 1, -1, -1):
            block = content[index]
            if isinstance(block, dict) and block.get("type") in {
                "input_text",
                "input_image",
                "input_file",
            }:
                updated = list(content)
                updated[index] = {**block, "prompt_cache_breakpoint": marker}
                return updated, True
    return content, False


def _mark_responses_cache_breakpoint(messages: list[dict[str, Any]], boundary: int) -> bool:
    """Reconstruct the latest 80 eligible message endpoints before ``boundary``.

    Explicit lookup considers the latest 80 breakpoints (writes use the latest
    four). Retain recent checkpoints as history grows, not just the newest one.
    Only marked containers are copied; content strings and all other items are
    shared. Stop after 80 endpoints, skipping ineligible assistant/native items.
    """
    count = 0
    for index in range(boundary - 1, -1, -1):
        item = messages[index]
        if item.get("type") == "function_call_output":
            output, marked = _mark_responses_content(item.get("output"))
            if marked:
                messages[index] = {**item, "output": output}
                count += 1
        # Assistant output uses output_text, which is not an eligible input block.
        elif item.get("role") in {"system", "developer", "user"}:
            content, marked = _mark_responses_content(item.get("content"))
            if marked:
                messages[index] = {**item, "content": content}
                count += 1
        if count == 80:
            break
    return count > 0


SESSION_AFFINITY_HEADER = "x-session-affinity"


def add_session_affinity_header(api_params: dict[str, Any]) -> None:
    """Mirror ``prompt_cache_key`` into the ``x-session-affinity`` request header.

    A prompt cache only helps if a conversation's requests reach the worker
    that holds its prefix. Baseten-served Hub routes (Kimi, GLM, Nemotron)
    pick a worker per request, and every switch to a cold prefill worker is
    a full cache miss. Baseten reads this header as a routing hint and
    echoes it back as ``x-baseten-session-id``; on Kimi's disaggregated
    deployment it kept a conversation on one prefill worker for every
    follow-up where the same conversation without it rotated workers and
    missed. Routes that don't know the header ignore it (Azure OpenAI,
    DeepSeek, non-disaggregated Nemotron: no change in behaviour).

    ``prompt_cache_key`` already identifies the conversation for the
    provider's cache, so it is the right affinity value too. A caller's
    explicit ``extra_headers`` entry for this header wins.
    """
    key = api_params.get("prompt_cache_key")
    if not isinstance(key, str) or not key:
        return
    headers = api_params.get("extra_headers")
    if headers is not None and not isinstance(headers, Mapping):
        raise ValueError("extra_headers must be a mapping")
    merged = dict(headers or {})
    # Header names are case-insensitive; a caller's X-Session-Affinity wins too.
    if not any(
        isinstance(name, str) and name.lower() == SESSION_AFFINITY_HEADER for name in merged
    ):
        merged[SESSION_AFFINITY_HEADER] = key
    api_params["extra_headers"] = merged


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


def wrap_anthropic_text(text: str) -> dict[str, Any]:
    """The one stable Anthropic content-block shape for a string.

    The wire-shape-stability wrap outside this module (``chat_parts.
    project_chat_turn``, ``replay_state.prepare_chat_messages`` -- both
    already conditioned on whether marking applies this turn) and marking
    below must produce the exact same block shape, for the same reason as
    ``wrap_responses_text``.
    """
    return {"type": "text", "text": text}


def _mark_anthropic(message: dict[str, Any]) -> dict[str, Any] | None:
    content = message.get("content")
    marker = {"type": "ephemeral"}
    if isinstance(content, str) and content:
        return {
            **message,
            "content": [{**wrap_anthropic_text(content), "cache_control": marker}],
        }
    if isinstance(content, list):
        for i in range(len(content) - 1, -1, -1):
            block = content[i]
            if isinstance(block, dict) and block.get("type") in {
                "text",
                "tool_result",
                "image",
                "image_url",
                "document",
                "file",
            }:
                blocks = list(content)
                blocks[i] = {**block, "cache_control": marker}
                return {**message, "content": blocks}
    return None


def reject_boundary_dict(message: Mapping[str, Any]) -> None:
    """JSON projections are not cache-policy inputs; use the typed boundary."""
    if "nooa_cache_boundary" in message:
        raise ValueError(
            "Pass CacheBoundary() from nooa.unifiedllm before dynamic context, "
            "not a nooa_cache_boundary dictionary."
        )


def apply_cache_policy(
    messages: list[dict[str, Any] | CacheBoundary],
    mapping: Literal["auto", "anthropic", "openai"] | None,
    *,
    responses: bool,
    instructions: str | None = None,
) -> tuple[list[dict[str, Any]], str | None, bool]:
    """Consume one boundary; direct callers default to their leading instructions."""
    clean = []
    boundary = None
    for message in messages:
        if isinstance(message, CacheBoundary):
            if boundary is not None:
                raise ValueError("Rendered history contains more than one cache boundary")
            boundary = len(clean)
            continue
        reject_boundary_dict(message)
        clean.append(message)
    if mapping is None:
        return clean, instructions, False
    automatic = mapping == "auto"
    if automatic and boundary is None:
        return clean, instructions, False
    if boundary is None:
        boundary = 0
        for message in clean:
            if message.get("role") not in {"system", "developer"}:
                break
            boundary += 1
    if mapping == "anthropic":
        if responses:
            raise ValueError("The Anthropic cache mapping requires CompletionClient")
        for i in range(boundary - 1, -1, -1):
            marked = _mark_anthropic(clean[i])
            if marked is not None:
                clean[i] = marked
                break
        return clean, instructions, False
    if not responses:
        raise ValueError("The OpenAI explicit cache mapping requires ResponsesClient")
    marked = _mark_responses_cache_breakpoint(clean, boundary)
    if not marked and instructions:
        content, marked = _mark_responses_content(instructions)
        clean.insert(0, {"role": "system", "content": content})
        instructions = None
    if not marked and not automatic:
        logger.warning(
            "OpenAI explicit cache policy found no eligible stable block; this request "
            "will not use prompt caching. Add stable instructions or place "
            "CacheBoundary() after reusable input text to enable cache writes."
        )
    # No eligible stable input: explicit mode deliberately avoids caching a
    # changing suffix. Never invent an empty text block just to host a marker.
    return clean, instructions, marked or not automatic
