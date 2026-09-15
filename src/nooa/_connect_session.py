# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded multi-turn onboarding checks; raw replay state stays in memory."""

import asyncio
import json
from copy import deepcopy

REPLY_CAP = 2048
MAX_LENGTH_RETRIES = 2
# Includes padding, schema/instructions, and up to two prior reply-sized items.
CALL_RESERVATION = 8192 + 3 * REPLY_CAP
TOKEN_RESERVATION = 3 * CALL_RESERVATION


def reply_budget(entry, budget_tokens):
    """Prefer the saved cap; fall back within the already approved total budget."""
    configured = entry.get("max_tokens", REPLY_CAP)
    cap = configured
    reservation = 8192 + 3 * cap
    if 3 * reservation > budget_tokens:
        cap = min(REPLY_CAP, configured)
        reservation = 8192 + 3 * cap
    return configured, cap, reservation


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _strings(child)


def _contains(expected, actual):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains(value, actual[key]) for key, value in expected.items()
        )
    return expected == actual


def settings_on_wire(settings, wire):
    """Compare declared settings after the API's reply-limit field translation."""
    expected = deepcopy(settings)
    caps = {"max_tokens", "max_completion_tokens", "max_output_tokens"}
    for key in caps & expected.keys():
        if key not in wire and len(caps & wire.keys()) == 1:
            expected[next(iter(caps & wire.keys()))] = expected.pop(key)
    return _contains(expected, wire)


def _wire_reasoning(value):
    """Only replay fields count; portable text is not native reasoning replay."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {
                "reasoning_content",
                "encrypted_content",
                "signature",
                "thought_signature",
                "thought_signatures",
            }:
                yield from _strings(child)
            elif key == "thinking" and value.get("type") == "thinking":
                yield from _strings(child)
            elif key == "data" and value.get("type") == "redacted_thinking":
                yield from _strings(child)
            elif key == "summary" and value.get("type") == "reasoning":
                yield from _strings(child)
            elif key == "id" and isinstance(child, str) and "__thought__" in child:
                yield child.split("__thought__", 1)[1]
            else:
                yield from _wire_reasoning(child)
    elif isinstance(value, list):
        for child in value:
            yield from _wire_reasoning(child)


def _cache_markers(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"cache_control", "prompt_cache_breakpoint"}:
                yield key
            else:
                yield from _cache_markers(child)
    elif isinstance(value, list):
        for child in value:
            yield from _cache_markers(child)


def _reasoning_values(response):
    """Compare meaningful reasoning payloads, never persist their contents."""
    from nooa._immutable_json import json_containers

    def payloads(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {
                    "encrypted_content",
                    "signature",
                    "data",
                    "thought_signature",
                    "thought_signatures",
                    "inline_thought_signature",
                }:
                    yield from _strings(child)
                else:
                    yield from payloads(child)
        elif isinstance(value, list):
            for child in value:
                yield from payloads(child)

    for part in response.parts:
        if part.kind == "reasoning" and part.text:
            yield part.text
        native = json_containers(part.native) if part.native else {}
        yield from payloads(native)


async def session_steps(alias, entry, *, api_key, budget_tokens):
    """Three turns, with bounded length-only retries; never execute model tools."""
    from nooa.connect import ProbeUpdate, _include_rejected
    from nooa.context_blocks.formatter import OpenAIProviderFormatter, ResponsesProviderFormatter
    from nooa.context_blocks.models import BlockMetadata, ResolvedBlock, Role
    from nooa.context_blocks.renderer import render_context
    from nooa.context_blocks.renderers.cached import CachedBlockFormatter
    from nooa.unifiedllm import CacheBoundary, RetryConfig, Tool
    from nooa.unifiedllm.registry import client_from_config

    configured_cap, reply_cap, reservation = reply_budget(entry, budget_tokens)
    if budget_tokens < 3 * reservation:
        yield ProbeUpdate("session", {"outcome": "not_probed", "reason": "budget exhausted"})
        return
    window = entry.get("context_window")
    if isinstance(window, int) and window < reservation:
        yield ProbeUpdate(
            "session",
            {"outcome": "not_probed", "reason": "context window too small for this check"},
        )
        return
    levels = entry.get("reasoning_levels", {})
    level = entry.get("reasoning_default")
    if level not in levels:
        level = next(iter(levels), None)
    settings = levels.get(level, {})
    controls = {
        key: value
        for key, value in {**entry, **settings}.items()
        if key
        in {"thinking", "reasoning", "reasoning_effort", "output_config", "chat_template_kwargs"}
    }
    if any(
        settings.get(key, reply_cap) != reply_cap
        for key in ("max_tokens", "max_output_tokens", "max_completion_tokens")
    ):
        yield ProbeUpdate(
            "session",
            {
                "outcome": "not_probed",
                "reason": "selected reasoning level changes the approved reply cap",
            },
        )
        return

    def probe_tool(value: str):
        raise RuntimeError("Setup checks never execute model tools")

    formatter = CachedBlockFormatter()
    provider = (
        ResponsesProviderFormatter()
        if entry["api_style"] == "responses"
        else OpenAIProviderFormatter()
    )
    # Non-repetitive lines give a reusable prefix without any user's private data.
    padding = "\n".join(
        f"Reference record {i:04d}: item {i * 17 + 3:06d} belongs to batch {i % 97:02d}."
        for i in range(360)
    )
    messages = render_context(
        [
            ResolvedBlock(
                key="instructions",
                role=Role.SYSTEM,
                content="Use the reference to solve the task. Do not repeat the reference.",
                metadata=BlockMetadata(static=True),
            ),
            ResolvedBlock(
                key="reference",
                role=Role.USER,
                content=padding,
                metadata=BlockMetadata(static=True),
            ),
            ResolvedBlock(
                key="task",
                role=Role.USER,
                content="Find the sum of item numbers for records 0013 and 0027. Call probe_tool with the sum as a string, or answer briefly.",
                metadata=BlockMetadata(static=False, user_block=True),
            ),
        ],
        block_formatter=formatter,
        provider_formatter=provider,
    ).output
    params = {
        "tools": [
            Tool(name="probe_tool", description="Record a computed value", callable=probe_tool)
        ],
        "max_output_tokens" if entry["api_style"] == "responses" else "max_tokens": reply_cap,
    }
    if level is not None:
        params["reasoning_level"] = level
        for key in ("max_tokens", "max_output_tokens", "max_completion_tokens"):
            if key in settings:
                params.pop(
                    "max_output_tokens" if entry["api_style"] == "responses" else "max_tokens", None
                )
    bodies = []
    successful_bodies = []

    async def capture(request):
        bodies.append(json.loads(request.content))

    try:
        client = client_from_config(
            alias,
            entry,
            api_key=api_key,
            num_retries=0,
            retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
        )
    except Exception as exc:
        yield ProbeUpdate("session", {"outcome": "not_confirmed", "error": type(exc).__name__})
        return
    # Hook only this owned client. Never monkey-patch global HTTP or store headers.
    hooks = client._http.httpx_async.event_hooks["request"]
    hooks.append(capture)
    spent = 0
    first = None
    replay = None
    last_usage = None
    observations = []
    settings_ok = True
    try:
        for index, name in enumerate(("seed", "replay", "repeat")):
            if spent + reservation > budget_tokens:
                yield ProbeUpdate(
                    "session",
                    {
                        "outcome": "not_probed",
                        "reason": "budget exhausted",
                        "tokens_charged_to_budget": spent,
                    },
                )
                return
            yield ProbeUpdate(f"session:{name}", {"outcome": "running"})
            call_messages = (
                messages
                if index == 0
                else [
                    *replay,
                    CacheBoundary(),
                    {
                        "role": "user",
                        "content": f"Check {index}: verify the recorded result briefly.",
                    },
                ]
            )
            attempts = []
            for attempt in range(MAX_LENGTH_RETRIES + 1):
                before = len(bodies)
                spent += reservation
                try:
                    async with asyncio.timeout(120):
                        response = await client.acall(call_messages, **params)
                except Exception as exc:
                    record = {
                        "outcome": "not_confirmed",
                        "error": type(exc).__name__,
                        "tokens_charged_to_budget": spent,
                        "attempts": deepcopy(attempts),
                    }
                    status = getattr(exc, "status_code", None)
                    if isinstance(status, int):
                        record["status_code"] = status
                    if entry.get("include") == [
                        "reasoning.encrypted_content"
                    ] and _include_rejected(exc):
                        record["include_rejected"] = True
                    yield ProbeUpdate("session", record)
                    return
                usage = response.usage
                last_usage = usage
                total = usage.input_tokens + usage.output_tokens if usage else 0
                spent += max(0, total - reservation)
                observed = bool(
                    any(p.kind == "reasoning" for p in response.parts)
                    or (usage and usage.reasoning_tokens)
                )
                record = {
                    "outcome": "accepted",
                    "reasoning_observed": observed,
                    "input_tokens": usage.input_tokens if usage else None,
                    "output_tokens": usage.output_tokens if usage else None,
                    "cached_input_tokens": usage.cached_input_tokens if usage else None,
                    "finish_reason": response.finish_reason,
                    "tested_reasoning_level": level,
                    "configured_reply_tokens": configured_cap,
                    "tested_reply_tokens": reply_cap,
                    "reply_limit_reduced_for_check": reply_cap != configured_cap,
                }
                attempts.append(
                    {
                        k: record[k]
                        for k in (
                            "input_tokens",
                            "output_tokens",
                            "finish_reason",
                            "tested_reply_tokens",
                        )
                    }
                )
                record["attempts"] = deepcopy(attempts)
                reason = None
                if len(bodies) != before + 1 or (usage and usage.output_tokens > reply_cap):
                    reason = "request capture missing or server exceeded the reply cap"
                elif response.finish_reason == "length":
                    next_cap = min(reply_cap * 2, configured_cap)
                    next_reservation = 8192 + 3 * next_cap
                    cap_key = (
                        "max_output_tokens" if entry["api_style"] == "responses" else "max_tokens"
                    )
                    if next_cap <= reply_cap:
                        reason = "reply truncated at the saved model limit"
                    elif attempt == MAX_LENGTH_RETRIES:
                        reason = "reply still truncated after two larger-limit retries"
                    elif cap_key not in params:
                        reason = (
                            "reply truncated; the selected reasoning level fixes the reply limit"
                        )
                    elif isinstance(window, int) and window < next_reservation:
                        reason = (
                            "reply truncated; a larger check would exceed the context allowance"
                        )
                    elif spent + (3 - index) * next_reservation > budget_tokens:
                        reason = "reply truncated; insufficient approved budget to retry and finish the checks"
                    else:
                        yield ProbeUpdate(
                            f"session:{name}",
                            {
                                "outcome": "retrying",
                                "reason": "reply limit reached",
                                "previous_reply_tokens": reply_cap,
                                "tested_reply_tokens": next_cap,
                                "attempts": deepcopy(attempts),
                            },
                        )
                        reply_cap, reservation = next_cap, next_reservation
                        params[cap_key] = reply_cap
                        continue  # Same input; never replay the truncated response.
                elif response.finish_reason in {"error", "content_filter"}:
                    reason = "conversation reply failed or was filtered"
                if reason:
                    record.update(outcome="not_confirmed", reason=reason)
                    yield ProbeUpdate(f"session:{name}", record)
                    yield ProbeUpdate(
                        "session",
                        {
                            "outcome": "not_confirmed",
                            "reason": reason,
                            "tokens_charged_to_budget": spent,
                        },
                    )
                    return
                yield ProbeUpdate(f"session:{name}", record)
                break
            observations.append(observed)
            successful_bodies.append(bodies[-1])
            wire = bodies[-1]
            settings_ok &= settings_on_wire(controls, wire)
            if index == 0:
                first = response
                replay = [m for m in messages if not isinstance(m, CacheBoundary)] + [first]
                for call in first.tool_calls:
                    if call.name != "probe_tool":
                        yield ProbeUpdate(
                            "session",
                            {
                                "outcome": "not_confirmed",
                                "reason": "unexpected tool requested",
                                "tokens_charged_to_budget": spent,
                            },
                        )
                        return
                    replay.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": "Probe result recorded.",
                        }
                    )

        # Compare all provider fields; only the final user message may differ.
        left, right = deepcopy(successful_bodies[1]), deepcopy(successful_bodies[2])
        key = "input" if entry["api_style"] == "responses" else "messages"
        for body in (left, right):
            # The reply allowance may have grown; it is not part of the input prefix.
            for cap_key in ("max_tokens", "max_output_tokens", "max_completion_tokens"):
                body.pop(cap_key, None)
            if entry["api_style"] == "anthropic":
                body[key][-1]["content"][-1]["text"] = "<volatile>"
            else:
                body[key][-1]["content"] = "<volatile>"
        stable = left == right
        cached = last_usage.cached_input_tokens if last_usage else None
        seed_input = first.usage.input_tokens if first.usage else 0
        markers = list(_cache_markers(successful_bodies[2]))
        explicit = bool(markers)
        substantial = bool(stable and seed_input > 0 and cached and cached >= seed_input / 2)
        cache_reason = (
            "Reusable conversation cached"
            if substantial
            else "Only a small portion was cached; try a longer prompt or check the server's caching support"
            if cached
            else "Cache markers were sent, but the server reported no reuse; retry later or check server support"
            if explicit
            else "No cache markers or cache reads observed; check the runtime's cache defaults and server support"
        )
        yield ProbeUpdate(
            "cache",
            {
                "outcome": "confirmed" if substantial else "not_confirmed",
                "cached_input_tokens": cached,
                "input_tokens": last_usage.input_tokens if last_usage else None,
                "stable_prefix": stable,
                "marker_count": len(markers),
                "explicit_mode": successful_bodies[2].get("prompt_cache_options", {}).get("mode")
                == "explicit",
                "reason": cache_reason,
            },
        )
        expected = list(_reasoning_values(first))
        actual = list(_wire_reasoning(successful_bodies[1]))
        retained = bool(expected) and all(value in actual for value in expected)
        yield ProbeUpdate(
            "reasoning_retention",
            {
                "outcome": "confirmed" if retained and settings_ok else "not_confirmed",
                "reasoning_observed_by_turn": observations,
                "settings_retained": settings_ok if controls else None,
                "state_retained": retained if expected else None,
                "tested_reasoning_level": level,
                "reason": "Reasoning preserved across turns"
                if retained and settings_ok
                else "Reasoning state was replayed, but the selected reasoning settings did not reach every request; check parameter filtering in the client"
                if retained and not settings_ok
                else "Reasoning was returned but not preserved in the follow-up request; check replay settings or try another interface"
                if expected
                else "No replayable reasoning returned; check reasoning/replay settings or try another interface. This does not mean reasoning is off",
            },
        )
        yield ProbeUpdate("session", {"outcome": "completed", "tokens_charged_to_budget": spent})
    finally:
        hooks.remove(capture)
        await client.aclose()
