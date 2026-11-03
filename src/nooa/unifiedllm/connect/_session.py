# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded multi-turn onboarding checks; raw replay state stays in memory."""

import asyncio
import json
import uuid
from copy import deepcopy

from . import REASONING_CHECK_PROMPT

REPLY_CAP = 2048
# Includes padding, schema/instructions, and up to two prior reply-sized items.
TOKEN_RESERVATION = 3 * (8192 + 3 * REPLY_CAP)
# Module-level so tests can monkeypatch it to 0 -- these three calls land back
# to back on the same endpoint in production; spurious 5xx/rate-limit
# failures are common enough there to be worth a short, fixed pace-out.
SESSION_CALL_PACING_SECONDS = 0.5


def reply_budget(entry, budget_tokens):
    """Reserve the configured request, never substitute a smaller check cap."""
    from . import configured_reply_cap

    levels = entry.get("reasoning_levels", {})
    level = entry.get("reasoning_default")
    if level not in levels:
        level = next(iter(levels), None)
    cap = configured_reply_cap(entry, level)
    return cap, cap, 8192 + 3 * cap


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


def _strip_cache_markers(value):
    """Compare underlying content stability independently of cache checkpoints.

    Stripping marker keys isolates content drift, but does not prove cache
    lookup eligibility. OpenAI explicit caching also needs a warmed checkpoint
    present in the later request; a moving marker alone can lose prefix lookup.
    """
    if isinstance(value, dict):
        return {
            key: _strip_cache_markers(child)
            for key, child in value.items()
            if key not in {"cache_control", "prompt_cache_breakpoint"}
        }
    if isinstance(value, list):
        return [_strip_cache_markers(child) for child in value]
    return value


def _history_prefix_stable(prior_body, later_body, key):
    """Did every message an earlier turn sent survive, unchanged, as later history?

    Explicit cache breakpoints only help the provider if the exact marked
    message renders identically once it stops being "the newest eligible
    block" and becomes ordinary history on a later turn. If the renderer
    reconstructs that message with a different wire shape at that point (for
    example a plain string that was list-wrapped only while carrying the
    marker), the provider's literal-prefix cache match breaks there on every
    later turn -- silently, since nothing else about the conversation looks
    wrong. This compares ``prior_body``'s entire input list, markers aside,
    against the same leading slice of ``later_body``'s input list, which is
    exactly the byte-stability a growing explicit-breakpoint conversation
    depends on.

    Anthropic's cacheable prefix and marker can also live in a top-level
    ``system`` field, separate from ``messages`` -- litellm renders a leading
    system-role message that way. A regression confined to that field would
    otherwise report as stable.
    """
    prior_list = [_strip_cache_markers(m) for m in prior_body.get(key, [])]
    later_list = [_strip_cache_markers(m) for m in later_body.get(key, [])]
    if len(later_list) < len(prior_list):
        return False
    if later_list[: len(prior_list)] != prior_list:
        return False
    return _strip_cache_markers(prior_body.get("system")) == _strip_cache_markers(
        later_body.get("system")
    )


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
    """Three configured-cap turns, without retries; never execute model tools."""
    from nooa.context_blocks.formatter import OpenAIProviderFormatter
    from nooa.context_blocks.models import BlockMetadata, ResolvedBlock, Role
    from nooa.context_blocks.renderer import render_context
    from nooa.context_blocks.renderers.cached import CachedBlockFormatter
    from nooa.unifiedllm import CacheBoundary, RetryConfig, Tool
    from nooa.unifiedllm.connect import ProbeUpdate, _include_rejected
    from nooa.unifiedllm.http_config import HttpConfig
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
    provider = OpenAIProviderFormatter()
    # Non-repetitive lines give a reusable prefix without any user's private data.
    # Short hex ids in place of zero-padded decimal digit runs measurably lower
    # (but do not eliminate) a stochastic provider content filter observed live
    # on this check's arithmetic-verification turn; see CHANGELOG.
    train_ids = [uuid.uuid5(uuid.NAMESPACE_DNS, f"train-{i}").hex[:6] for i in range(360)]
    padding = "\n".join(
        f"Train {train_ids[i]} departs platform {1 + i % 12}; its trip takes "
        f"{15 + (i * 17 + 3) % 180} minutes."
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
                # The cached formatter partitions SYSTEM context blocks and
                # renders the dynamic half as a trailing USER message.
                #
                # Reuses the reasoning-level puzzle rather than a trivial
                # arithmetic task: a model with adaptive/content-dependent
                # reasoning effort can legitimately skip reasoning on
                # something this easy even at a real reasoning level, which
                # previously showed up as a false "reasoning not retained"
                # result indistinguishable from an actual replay bug
                # (observed live for gpt-6-astra). The puzzle is hard enough
                # to force genuine reasoning at any configured level. As
                # with level checks, a wrong answer here is not graded —
                # only whether reasoning was observed at all.
                role=Role.SYSTEM,
                content=(
                    f"{REASONING_CHECK_PROMPT}\n\nCall probe_tool with your eight-letter "
                    "answer, or answer briefly."
                ),
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
            http_config=HttpConfig(read_timeout=120),
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
    readings = []
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
                        # A plain "verify the answer" ask is itself trivial —
                        # answerable from memory without reasoning again, so a
                        # model with adaptive/content-dependent reasoning
                        # effort can legitimately skip it on this turn even
                        # though it reasoned on the original puzzle (observed
                        # live for gpt-6-astra). Changing one rule forces a
                        # genuine new reasoning pass rather than a recall.
                        "content": (
                            f"Check {index}: one rule changed — H now runs first, not "
                            "last. Solve the puzzle again with this update. Call "
                            "probe_tool with your new eight-letter answer, or answer "
                            "briefly."
                        ),
                    },
                ]
            )
            attempts = []
            before = len(bodies)
            spent += reservation
            if index > 0 and SESSION_CALL_PACING_SECONDS:
                await asyncio.sleep(SESSION_CALL_PACING_SECONDS)
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
                if entry.get("include") == ["reasoning.encrypted_content"] and _include_rejected(
                    exc
                ):
                    record["include_rejected"] = True
                yield ProbeUpdate("session", record)
                return
            usage = response.usage
            total = usage.input_tokens + usage.output_tokens if usage else 0
            spent += max(0, total - reservation)
            # Billed usage.reasoning_tokens alone is not evidence -- confirmed
            # live, Kimi/Qwen bill nonzero reasoning tokens on routes that
            # never put a reasoning item on the wire, so there is nothing for
            # capture_parts to attach and parts stays empty.
            observed = any(p.kind == "reasoning" for p in response.parts)
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
                "transport": getattr(client, "transport", "litellm"),
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
            elif not settings_on_wire({"max_tokens": reply_cap}, bodies[-1]):
                reason = "configured reply cap did not reach the request"
            elif response.finish_reason == "length":
                reason = "reply truncated at the saved model limit"
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
            record["settings_sent"] = True
            yield ProbeUpdate(f"session:{name}", record)
            observations.append(observed)
            successful_bodies.append(bodies[-1])
            if index:
                readings.append(
                    {
                        "turn": name,
                        "input_tokens": usage.input_tokens if usage else None,
                        "cached_input_tokens": usage.cached_input_tokens if usage else None,
                    }
                )
            wire = bodies[-1]
            settings_ok &= settings_on_wire(controls, wire)
            if index == 0:
                first = response
            # Extend after every turn, not just the seed, so the next call is a
            # genuine continuation: whatever this turn just sent (including any
            # message an explicit cache breakpoint marked) must reappear as
            # ordinary history next time, the same way a real multi-turn
            # session grows. Reusing a frozen base across turns (the previous
            # behavior) never lets a marked message become history within this
            # probe, so it could never catch a renderer that reconstructs that
            # message differently once replayed -- see the cache stability
            # check below, which depends on this actually growing.
            replay = [m for m in call_messages if not isinstance(m, CacheBoundary)] + [response]
            for call in response.tool_calls:
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

        # "repeat" (successful_bodies[2]) is now a genuine continuation of
        # "replay" (successful_bodies[1]) -- see the extension above. Everything
        # "replay" actually sent, including whichever message an explicit cache
        # breakpoint marked, must reappear byte-identical (that transient marker
        # key aside) as a leading slice of "repeat"'s input for the provider's
        # cache to extend across turns. A frozen replay base across both calls
        # (the previous check) could never catch this: it never let a marked
        # message become history within the probe at all.
        key = "input" if entry["api_style"] == "responses" else "messages"
        stable = _history_prefix_stable(successful_bodies[1], successful_bodies[2], key)
        best = max(readings, key=lambda r: r["cached_input_tokens"] or 0)
        cached = best["cached_input_tokens"]
        seed_input = first.usage.input_tokens if first.usage else 0
        markers = list(_cache_markers(successful_bodies[2]))
        explicit = bool(markers)
        substantial = bool(stable and seed_input > 0 and cached and cached >= seed_input / 2)
        cache_reason = (
            "Reusable conversation cached"
            if substantial
            else "A message from an earlier turn rendered differently once replayed as "
            "history; explicit cache breakpoints will not accumulate hits across turns "
            "until the renderer keeps that message's wire shape stable"
            if not stable
            else "Only a small portion was cached; try a longer prompt or check the server's caching support"
            if cached
            else "Cache markers were sent, but the server reported no reuse; retry later or check server support"
            if explicit
            else "No cache reuse reported on these continuations; provider caching can vary. The model connection still works"
            if entry["api_style"] == "chat"
            else "No cache markers or cache reads observed; check the runtime's cache defaults and server support"
        )
        yield ProbeUpdate(
            "cache",
            {
                "outcome": "confirmed"
                if substantial
                else "warning"
                if stable and (explicit or entry["api_style"] == "chat")
                else "not_confirmed",
                "cached_input_tokens": cached,
                "input_tokens": best["input_tokens"],
                "readings": readings,
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
