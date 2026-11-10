# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Presentation of NOOA's sanitized Relay events as ATIF."""

from __future__ import annotations

import json
from typing import Any


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"Non-JSON number: {value}")


def _provider_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    raw_calls = message.get("tool_calls")
    for call in raw_calls if isinstance(raw_calls, list) else []:
        if not isinstance(call, dict):
            continue
        identifier, function = call.get("id"), call.get("function")
        if not isinstance(identifier, str) or not identifier or not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = function.get("arguments", {})
        raw_arguments = arguments
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments, parse_constant=_reject_json_constant)
            except (ValueError, RecursionError):
                pass
        projected = {"tool_call_id": identifier, "function_name": name, "arguments": arguments}
        if not isinstance(arguments, dict):
            # ATIF requires an argument object, even when the model emitted
            # malformed JSON. Preserve that evidence without inventing inputs.
            projected["arguments"] = {}
            projected["extra"] = {"raw_arguments": raw_arguments}
        calls.append(projected)
    return calls


def _normalize_atif(trajectory: dict[str, Any]) -> dict[str, Any]:
    from nooa.nemo_relay_middleware import _tool_display

    steps = trajectory.get("steps", [])
    issuers: dict[str, list[int]] = {}
    provider_calls: dict[int, list[dict[str, Any]]] = {}
    for index, step in enumerate(steps):
        extra = step.get("extra")
        response = extra.get("llm_response", {}) if isinstance(extra, dict) else {}
        message = response.get("message") if isinstance(response, dict) else None
        if not isinstance(message, dict):
            continue
        # This is the sanitized legacy payload, never a saved pre-redaction copy.
        content = message.get("content")
        step["message"] = (
            content
            if isinstance(content, str)
            else ("" if content is None else json.dumps(content, ensure_ascii=False))
        )
        reasoning = message.pop("reasoning_content", None)
        if isinstance(reasoning, str):
            step["reasoning_content"] = reasoning
        calls = _provider_calls(message)
        provider_calls[index] = calls
        for call in calls:
            issuers.setdefault(call["tool_call_id"], []).append(index)

    # IDs are the only reliable join under concurrency. Repeated provider IDs
    # remain untouched: names, arguments, and timing cannot disambiguate them.
    unique = {identifier: owners[0] for identifier, owners in issuers.items() if len(owners) == 1}
    calls_by_step: list[list[dict[str, Any]]] = [[] for _ in steps]
    observations_by_step: list[list[dict[str, Any]]] = [[] for _ in steps]
    event_owners: dict[str, int] = {}
    for index, step in enumerate(steps):
        emitted = provider_calls.get(index, [])
        calls_by_step[index].extend(emitted)
        for call in step.get("tool_calls", []):
            identifier = call.get("tool_call_id")
            if identifier not in unique:
                if not any(
                    existing.get("tool_call_id") == identifier for existing in calls_by_step[index]
                ):
                    calls_by_step[index].append(call)
        for observation in step.get("observation", {}).get("results", []):
            extra = observation.get("extra")
            if not isinstance(extra, dict):
                extra = {}
            annotation = extra.get("tool_result_annotation")
            if isinstance(annotation, dict) and isinstance(annotation.get("is_error"), bool):
                extra["is_error"] = annotation["is_error"]
            if "content" not in observation and "tool_result" in extra:
                observation["content"] = _tool_display(extra["tool_result"])
            elif isinstance(observation.get("content"), str):
                observation["content"] = _tool_display(observation["content"])
            # The display is bounded; retaining its raw duplicate defeats that
            # bound and can include opaque pickle payloads.
            if "content" in observation:
                extra.pop("tool_result", None)
            target = unique.get(observation.get("source_call_id"), index)
            if observation.get("source_call_id") in unique and isinstance(
                extra.get("event_uuid"), str
            ):
                event_owners[extra["event_uuid"]] = target
            observations_by_step[target].append(observation)

    # These native arrays contain explicit IDs, so move their metadata with
    # the observation rather than leaving it attached to an unrelated model.
    for field, identifier, owners in (
        ("tool_invocations", "invocation_id", unique),
        ("tool_ancestry", "function_id", event_owners),
    ):
        routed: list[list[dict[str, Any]]] = [[] for _ in steps]
        for index, step in enumerate(steps):
            extra = step.get("extra")
            if not isinstance(extra, dict):
                continue
            for item in extra.pop(field, []) or []:
                routed[owners.get(item.get(identifier), index)].append(item)
        for index, items in enumerate(routed):
            if items:
                steps[index].setdefault("extra", {})[field] = items

    for index, step in enumerate(steps):
        if calls_by_step[index]:
            step["tool_calls"] = calls_by_step[index]
        else:
            step.pop("tool_calls", None)
        if observations_by_step[index]:
            step["observation"] = {"results": observations_by_step[index]}
        else:
            step.pop("observation", None)
    return trajectory


async def export_nemo_relay_atif(exporter: Any) -> dict[str, Any]:
    """Flush Relay and return readable ATIF built solely from sanitized events.

    Use this instead of parsing ``exporter.export_json()`` directly for NOOA
    events. Provider call IDs are correlated only when unique. Host-generated
    calls retain Relay's placement, and missing/redacted observations are never
    reconstructed. Raw duplicate tool results are replaced with bounded text.
    """
    import nemo_relay  # type: ignore[import]

    await nemo_relay.subscribers.flush_async()
    return _normalize_atif(json.loads(exporter.export_json()))
