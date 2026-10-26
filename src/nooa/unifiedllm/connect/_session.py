# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-turn session test: three related calls to one model entry.

The session test sends a conversation of three turns ("seed", "replay",
"repeat"), each continuing the previous one, and reports three checks:

- ``session``: whether the conversation worked at the configured reply limit.
- ``cache``: whether the provider reused cached input across turns, and
  whether the earlier turns were sent again byte-for-byte as history (needed
  for prompt caching to work).
- ``reasoning_retention``: whether the reasoning state returned on the first
  turn was sent back in the next request ("replayed"), as multi-turn
  reasoning models require.

Model tools are never executed, nothing is retried, and the raw requests and
reasoning stay in memory only; results contain counts and flags, not content.
"""

import asyncio
import json
import uuid
from copy import deepcopy

from . import REASONING_CHECK_PROMPT

REPLY_CAP = 2048
# Token allowance per call: the ~8k padding prompt plus instructions, plus up
# to three reply-sized items (two earlier replies and the new reply).
TOKEN_RESERVATION = 3 * (8192 + 3 * REPLY_CAP)
# Pause between the three calls, which would otherwise hit the same endpoint
# back to back and often trigger spurious 5xx or rate-limit errors. A module
# constant so tests can set it to 0.
SESSION_CALL_PACING_SECONDS = 0.5


def reply_budget(entry, budget_tokens):
    """Return ``(configured_cap, tested_cap, token_reservation)`` for one session call.

    The session test uses the entry's own configured reply limit, never a
    smaller limit just for the test, so it tests the real configuration. The
    reservation is the tokens set aside per call. Building the selected
    reasoning level first makes an invalid level raise. ``budget_tokens`` is
    currently unused.
    """
    from nooa.unifiedllm.reasoning import ReasoningConfig

    from . import configured_reply_cap

    level = entry.get("reasoning_level")
    if level is not None:
        ReasoningConfig(
            levels=entry.get("reasoning_levels"), default=entry.get("reasoning_default")
        ).settings(level)
    cap = configured_reply_cap(entry, level)
    return cap, cap, 8192 + 3 * cap


def _strings(value):
    """Yield every string inside ``value``, searching dicts, lists and tuples."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _strings(child)


def _contains(expected, actual):
    """Return True if ``actual`` contains ``expected``; dicts may have extra keys."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains(value, actual[key]) for key, value in expected.items()
        )
    return expected == actual


def settings_on_wire(settings, wire):
    """Return True if every value in ``settings`` appears in the sent request ``wire``.

    Clients rename the reply limit to the field their API uses
    (``max_tokens``, ``max_completion_tokens`` or ``max_output_tokens``). If the
    request contains exactly one of these fields, the expected limit is
    compared against that field.
    """
    expected = deepcopy(settings)
    caps = {"max_tokens", "max_completion_tokens", "max_output_tokens"}
    for key in caps & expected.keys():
        if key not in wire and len(caps & wire.keys()) == 1:
            expected[next(iter(caps & wire.keys()))] = expected.pop(key)
    return _contains(expected, wire)


def _wire_reasoning(value):
    """Yield the reasoning values found in a request body sent to the provider.

    Only the fields providers use to receive earlier reasoning back count
    (``reasoning_content``, encrypted content, signatures, Anthropic thinking
    blocks, Responses reasoning summaries, and thought signatures that LiteLLM
    encodes into tool call IDs after ``__thought__``). Reasoning repeated as
    plain message text does not count as reasoning replay.
    """
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
    """Yield the name of every prompt-cache marker key found in a request body."""
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
    """Return a copy of ``value`` with all prompt-cache marker keys removed.

    This lets two requests be compared for content changes while ignoring
    where the cache markers are. Equal content alone does not guarantee a
    cache hit: with OpenAI explicit caching, the later request must also carry
    a marker at a position that was already cached, and a marker that moved
    can prevent the cached prefix from being found.
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
    """Return True if the later request starts with the earlier request's messages, unchanged.

    Provider prompt caches match the exact start (prefix) of a request. A
    message that carried a cache marker on one turn becomes ordinary history
    on the next. If it is then sent in a different form (for example as a
    plain string where it was a list of content parts while it carried the
    marker), the cached prefix no longer matches on any later turn, and
    nothing else in the conversation shows the problem.

    ``key`` names the message list ("messages" or "input"). The whole list
    in ``prior_body`` must equal the first items of the list in
    ``later_body``, ignoring cache markers. The top-level ``system`` field
    must also be equal, because Anthropic requests (and LiteLLM, for a leading
    system message) keep the system prompt and its marker there instead of
    in the message list.
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
    """Yield the reasoning values in ``response`` that should be sent back next turn.

    These are the reasoning text (split into the original summary blocks for
    the Responses API) and the encrypted/signature fields in each part's
    provider-specific data. The values are only compared with the next
    request, never stored.
    """
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
        native = json_containers(part.native) if part.native else {}
        if part.kind == "reasoning" and part.text:
            # Responses API summaries are stored as one joined string but sent
            # back as separate blocks, so yield each block's text separately.
            summary = native.get("summary") if isinstance(native, dict) else None
            if (
                isinstance(summary, list)
                and summary
                and all(isinstance(b, dict) and "_text_length" in b for b in summary)
            ):
                offset = 0
                for block in summary:
                    length = block["_text_length"]
                    if length:
                        yield part.text[offset : offset + length]
                    offset += length + 1  # matches the "\n" used when capturing
            else:
                yield part.text
        yield from payloads(native)


LITELLM_ENVELOPE = "litellm_enc:"


def _replayed(value, actual):
    """Return True if reasoning ``value`` was sent in the follow-up request.

    ``actual`` lists the reasoning values found in that request. A LiteLLM
    proxy wraps encrypted reasoning as ``litellm_enc:<routing>;<state>`` and a
    LiteLLM client unwraps it again before sending, so the request may contain
    only the inner state. Such a value therefore also matches if a sent value
    of at least 32 characters appears inside it.
    """
    if value in actual:
        return True
    return value.startswith(LITELLM_ENVELOPE) and any(
        len(sent) >= 32 and sent in value for sent in actual
    )


async def _session_steps(alias, entry, *, api_key, budget_tokens):
    """Run the three session calls and yield a ``ProbeUpdate`` per step and check.

    Each call uses the configured reply limit and reasoning level, no
    retries and a 120 second timeout. The model may call ``probe_tool``; the
    tool is never executed, and a fixed result is sent back instead. The test
    is skipped (``not_probed``) if the token budget, the context window or
    the reasoning level's own reply limit does not allow it, and stops early
    if any call fails or is incomplete. Requests are captured through an HTTP
    hook on the test's own client only.
    """
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
    levels = entry.get("reasoning_levels") or {}
    level = entry.get("reasoning_level")
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
        # Offered to the model, but Connect never runs tools during tests.
        raise RuntimeError("Setup checks never execute model tools")

    formatter = CachedBlockFormatter()
    provider = OpenAIProviderFormatter()
    # Generated, varied filler text forms a long prefix that the provider can
    # cache, without using any user data. Short hex IDs are used instead of
    # zero-padded numbers because they make a provider content filter, seen
    # in live tests on this check, trigger less often (not never); see CHANGELOG.
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
                # The cached formatter splits SYSTEM blocks into static and
                # dynamic parts and sends the dynamic part (this task) as the
                # last USER message.
                #
                # The task is the reasoning-level puzzle, not simple
                # arithmetic: models that decide how much to reason based on
                # the question may skip reasoning on an easy task, which
                # looked exactly like a reasoning replay bug (seen live with
                # gpt-6-astra). The puzzle makes the model reason at any
                # level. As in the level checks, the answer is not graded;
                # only whether reasoning was observed.
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
    # Capture request bodies through a hook on this client only; global HTTP
    # settings are not patched, and headers (which hold the key) are not kept.
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
                        # Asking only to verify the earlier answer is easy to
                        # answer from memory, so a model that adapts its
                        # reasoning effort may skip reasoning on this turn
                        # (seen live with gpt-6-astra). Changing one rule makes
                        # it reason again instead of recalling the answer.
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
            # Reasoning counts as observed only if the reply contains a
            # reasoning part. usage.reasoning_tokens is not enough: in live
            # tests Kimi/Qwen reported reasoning tokens on endpoints that never
            # return reasoning content, so nothing could be replayed.
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
            # Grow the history after every turn, as a real conversation does:
            # everything this turn sent (including a message that carried a
            # cache marker) is sent again as plain history next turn. The
            # cache stability check below relies on this; with a fixed history
            # a marked message would never become history, and a renderer that
            # sends it differently afterwards would go unnoticed.
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

        # The "repeat" request (successful_bodies[2]) continues the "replay"
        # request (successful_bodies[1]). For the provider's cache to keep
        # working across turns, everything "replay" sent must reappear,
        # identical apart from cache markers, at the start of "repeat".
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
        retained = bool(expected) and all(_replayed(value, actual) for value in expected)
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


async def session_steps(alias, entry, *, api_key, budget_tokens):
    """Run the session test (see ``_session_steps``) and yield its updates.

    Each finished result is stamped with the configuration it was run with
    (see ``stamp_verification``), using a copy of ``entry`` taken before the
    test started, so later edits to the entry make the result out of date.
    """
    from contextlib import aclosing
    from copy import deepcopy

    from ._verification import stamp_verification

    executed_entry = deepcopy(entry)
    async with aclosing(
        _session_steps(alias, entry, api_key=api_key, budget_tokens=budget_tokens)
    ) as steps:
        async for update in steps:
            if update.outcome.get("outcome") not in {"not_probed", "running"}:
                stamp_verification(update.outcome, executed_entry, update.name, timeout_seconds=120)
            yield update
