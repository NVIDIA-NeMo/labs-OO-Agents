# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small, responsive terminal presentation helpers for the Connect wizard."""

import os
import shutil
import sys
import textwrap
import time
from contextlib import contextmanager

import click


@contextmanager
def quiet_provider_messages():
    """Hide legacy-library help banners while this CLI displays safe outcomes."""
    legacy = sys.modules.get("litellm")
    previous = getattr(legacy, "suppress_debug_info", False)
    if legacy is not None:
        legacy.suppress_debug_info = True
    try:
        yield
    finally:
        if legacy is not None:
            legacy.suppress_debug_info = previous


def format_budget(tokens):
    """Render a check-token budget, spelling out the unset-flag sentinel as unlimited."""
    from nooa.unifiedllm.connect import DEFAULT_CHECK_BUDGET

    # DEFAULT_CHECK_BUDGET is a large finite sentinel, not float("inf"), so it
    # survives JSON encoding and every existing int arithmetic site unchanged
    # (see its definition). "Budget remaining" values are the sentinel minus
    # whatever a run has spent so far (observed up to ~1.2M tokens for a full
    # run; see TOKEN_RESERVATION), never exactly equal to it — so this stays
    # a threshold, not an exact-equality check, but one anchored to the
    # actual sentinel with a generous fixed buffer for spend, rather than a
    # fraction of it. --budget-tokens has no declared upper bound, so a
    # threshold that scaled down with the sentinel (e.g. a fraction of it)
    # would mislabel a genuine, very large, explicitly-chosen budget as
    # unlimited; a fixed buffer close to the sentinel does not.
    return "unlimited" if tokens >= DEFAULT_CHECK_BUDGET - 10**9 else f"{tokens:,}"


def _reasoning_tokens_label(record):
    """Describe what's actually known about a level check's reasoning cost.

    A real, positive count from the endpoint/litellm is shown as-is. Anthropic
    can withhold the visible thinking text in more than one wire shape — a
    genuine redacted_thinking block (an opaque encrypted blob, no text field
    at all), or a normal *signed* "thinking" block whose text simply comes
    back empty (observed live for Claude Sonnet 5/Opus 5 via Azure/Bedrock) —
    but either way litellm's reasoning_tokens estimate is a text-length count
    that reads 0 when there is no visible text, regardless of how much
    thinking actually happened. Showing that 0 as though it were measured
    would be misleading. output_tokens is shown instead where available,
    since Anthropic bills thinking tokens as ordinary output tokens without
    splitting them out.

    litellm's text-length estimate only exists for Anthropic/Bedrock at all —
    for every other provider (observed live for Qwen and DeepSeek), it never
    attempts one, even when real, non-empty reasoning text came back. When we
    have that text in hand, its character count is shown as a size signal
    instead of just "not reported separately" with nothing further. Returns
    "" when nothing is known.
    """
    tokens = record.get("reasoning_tokens")
    if isinstance(tokens, int) and tokens > 0:
        return f"{tokens:,} reasoning tokens"
    output_tokens = record.get("output_tokens")
    if not isinstance(output_tokens, int):
        return ""
    if record.get("reasoning_encrypted"):
        size = record.get("reasoning_encrypted_bytes")
        size_note = f"; ~{size:,} bytes of encrypted state" if isinstance(size, int) else ""
        return (
            f"reasoning text withheld by the provider ({output_tokens:,} output tokens, "
            f"not split out{size_note})"
        )
    if record.get("reasoning_observed"):
        chars = record.get("reasoning_text_chars")
        chars_note = f"; ~{chars:,} reasoning chars" if isinstance(chars, int) else ""
        return f"{output_tokens:,} output tokens (reasoning tokens not reported separately{chars_note})"
    return ""


def _reasoning_tokens_summary(record):
    """Compact form of _reasoning_tokens_label for the end-of-run summary line.

    Mirrors _reasoning_tokens_label's gating exactly: the final "N output
    (chars)" fallback only applies when reasoning was actually observed.
    Without that gate, a level where the model reported real output tokens
    but never reasoned at all would print as though a reasoning cost was
    measured for it.
    """
    tokens = record.get("reasoning_tokens")
    if isinstance(tokens, int) and tokens > 0:
        return f"{tokens:,}"
    output_tokens = record.get("output_tokens")
    if not isinstance(output_tokens, int):
        return "reasoning observed" if record.get("reasoning_observed") else "0"
    if record.get("reasoning_encrypted"):
        size = record.get("reasoning_encrypted_bytes")
        size_note = f", ~{size:,}B" if isinstance(size, int) else ""
        return f"{output_tokens:,} output (withheld{size_note})"
    if not record.get("reasoning_observed"):
        return "0"
    chars = record.get("reasoning_text_chars")
    chars_note = f", ~{chars:,} chars" if isinstance(chars, int) else ""
    return f"{output_tokens:,} output{chars_note}"


def check_failure(outcome):
    """Translate sanitized error classes, never show a provider error body."""
    error = outcome.get("error", "")
    status = outcome.get("status_code")
    if status in {401, 403} or error in {"AuthenticationError", "PermissionDeniedError"}:
        return "Key rejected by this server. Check the key and server URL."
    if status == 404 or error == "NotFoundError":
        return "No route found for this interface or model."
    if status == 429 or error == "RateLimitError":
        return "Server rate limit reached. Try again later."
    if isinstance(status, int) and status >= 500:
        return "Server error. Try again later."
    if "Timeout" in error:
        return "Model response timed out; the route may be slow."
    if error == "ReasoningReplayError":
        return "Reply not understood (ReasoningReplayError)."
    if error == "APIConnectionError":
        return "Could not reach the server. Check the connection or try again."
    if isinstance(status, int) and 400 <= status < 500:
        return "Request rejected by this server. Check the request settings."
    return None


def local_failure(exc, *, api_key=None, api_key_env=None):
    """Explain local setup errors without YAML excerpts or provider error bodies."""
    import yaml

    if isinstance(exc, yaml.YAMLError):
        mark = getattr(exc, "problem_mark", None)
        detail = (
            f"Invalid YAML in {mark.name}, line {mark.line + 1}, column {mark.column + 1}."
            if mark is not None
            else "Invalid YAML configuration."
        )
    elif isinstance(exc, (OSError, ValueError)):
        detail = str(exc)
    else:
        detail = "Check the connection and credentials; see the diagnostic prompt."
    for secret in (api_key, os.environ.get(api_key_env) if api_key_env else None):
        if secret:
            detail = detail.replace(secret, "[redacted]")
    return detail[:1000]


def line(text, *, fg=None, bold=False, dim=False):
    width = max(24, min(84, shutil.get_terminal_size((80, 24)).columns - 4))
    for part in textwrap.wrap(text, width=width, break_on_hyphens=False) or [""]:
        if "NO_COLOR" not in os.environ:
            part = click.style(part, fg=fg, bold=bold, dim=dim)
        click.echo("  " + part)


def intro(*, checks, output_tokens, budget_tokens, reasoning_output_tokens=4096):
    click.echo()
    line("NOOA  /  CONNECT", fg="bright_cyan", bold=True)
    line("Add a model to your workspace.", dim=True)
    click.echo()
    if checks:
        line("API checks may incur charges.", fg="yellow")
        line("Checks use the same model client as your agents.", dim=True)
        line(
            "Up to 3 interface calls, then tools, each proposed reasoning level, and a 3-turn conversation check.",
            dim=True,
        )
        line(
            f"Interface discovery: {output_tokens:,} output tokens / call. After setup, all checks send the saved reply cap (including reasoning-level overrides); checks that do not fit the approved budget are skipped.",
            dim=True,
        )
        line(
            f"{format_budget(budget_tokens)} shared token budget"
            if budget_tokens is not None
            else "Up to 3 interface calls, then tools and each proposed reasoning level.",
            dim=True,
        )
        line(
            "No automatic retries or capacity probes. Caps are estimates, not billing limits.",
            dim=True,
        )
        line(
            "If encrypted reasoning is explicitly rejected, one check without it may use the same approved budget.",
            dim=True,
        )
        line(
            "Includes a longer reusable prompt to measure cache reads. Dollar cost depends on your model; servers can ignore caps. Checks stop when the approved budget is exhausted.",
            dim=True,
        )
    else:
        line("No generation calls. Listing and metadata may still be fetched.", dim=True)
    line("Skip paid checks: --no-probe    Help: F1", dim=True)
    if not sys.stdin.isatty():
        line("Manual setup: docs/model-configuration.md", dim=True)
        line("Agent skill: nooa-model-configuration", dim=True)


def step(number, title):
    click.echo()
    line(f"{number} / 4  ·  {title}", fg="bright_cyan", bold=True)
    click.echo()


class CheckProgress:
    """Compact human progress; replace only the active line on a real terminal."""

    def __init__(self):
        self.inline = click.get_text_stream("stdout").isatty()
        self.active = False
        self.started = {}
        self.results = {}
        self.reasoning_levels = {}

    def _clear(self):
        if self.active:
            click.echo("\r\033[2K", nl=False, color=True)
            self.active = False

    def update(self, name, record, *, missing_reasoning=False):
        labels = {
            "chat": "Chat interface",
            "responses": "Responses interface",
            "anthropic": "Anthropic interface",
            "routing": "Connection",
            "tools": "Tool use",
            "session:seed": "Conversation 1/3 · start",
            "session:replay": "Conversation 2/3 · continue",
            "session:repeat": "Conversation 3/3 · repeat",
            "cache": "Cache reuse",
            "reasoning_retention": "Reasoning carried forward",
            "encrypted_reasoning": "Encrypted reasoning",
            "session": "Conversation checks",
        }
        label = "Reasoning · " + name[6:] if name.startswith("level:") else labels.get(name, name)
        outcome = record.get("outcome")
        self._clear()
        if outcome == "running":
            self.started[name] = time.monotonic()
            text = f"  … {label} — checking"
            if self.inline:
                width = max(20, shutil.get_terminal_size((80, 24)).columns - 1)
                click.echo(text[:width], nl=False)
                self.active = True
            else:
                line(text.strip(), dim=True)
            return
        if name == "session" and outcome == "completed":
            return  # Cache and reasoning rows already describe the result.
        if outcome == "retrying":
            line(
                f"↻ {label}: Reached the test limit; retrying with {record['tested_reply_tokens']:,} tokens (saved setting unchanged).",
                fg="yellow",
            )
            return
        # `status` here is display-only scratch state for the fallback table
        # below (`fallback[status]`); check_status() a few lines down is the
        # sole source of truth for the icon/color, and always overwrites
        # whatever this block computes. Do not read `status` above that call.
        detail = check_failure(record) or record.get("reason")
        if outcome == "accepted":
            detail = (
                "Connected"
                if name in {"routing", "chat", "responses", "anthropic"}
                else "Reply received"
            )
            if name == "tools":
                detail = (
                    "Tool call returned" if record.get("tool_observed") else "No tool call returned"
                )
            elif name.startswith("level:"):
                detail = (
                    "Reasoning returned" if record.get("reasoning_observed") else "Request accepted"
                )
                if missing_reasoning:
                    detail = "No reasoning details returned"
            elif record.get("reasoning_observed"):
                detail += " · reasoning returned"
            if record.get("finish_reason") == "length":
                detail = "Ran out of reply tokens before finishing"
                if name.startswith("level:"):
                    detail += (
                        " — if you plan to use this reasoning level, increase the reply budget"
                    )
            elif record.get("finish_reason") in {"error", "content_filter"}:
                detail = "Reply incomplete; check not conclusive"
            if record.get("reason") == "previous result reused":
                detail += " · already checked"
            if name.startswith("level:") and isinstance(record.get("answer_correct"), bool):
                tokens_label = _reasoning_tokens_label(record)
                if tokens_label:
                    detail += f" · {tokens_label}"
                if record.get("reasoning_observed"):
                    # Record every level where reasoning genuinely happened,
                    # even when no token count/label is available (e.g. the
                    # response carried no usage) — omitting it here silently
                    # drops the level from the finish() summary line,
                    # defeating the point of a complete cross-level
                    # comparison. Levels where reasoning was never observed
                    # stay out entirely: a bare "0" there would look like a
                    # measured reasoning cost instead of an absence of
                    # evidence (see unobserved_reasoning_levels).
                    self.reasoning_levels[name[6:]] = record
                detail += " · answer correct" if record["answer_correct"] else " · answer incorrect"
        elif name == "cache" and outcome == "confirmed" and record.get("input_tokens"):
            cached, total = record.get("cached_input_tokens", 0), record["input_tokens"]
            detail = f"Reused {cached / total:.0%} of input ({cached:,} / {total:,} tokens)"
            if record.get("readings"):
                detail += " · " + ", ".join(
                    f"{r['turn']}: {r.get('cached_input_tokens') or 0:,} cached"
                    for r in record["readings"]
                )
        elif name == "reasoning_retention" and outcome == "confirmed":
            detail = "Preserved in the next request"
        from nooa.unifiedllm.connect._records import check_status

        status = check_status(name, record, missing_reasoning=missing_reasoning)
        icon, color = {
            "passed": ("✓", "green"),
            "attention": ("!", "yellow"),
            "skipped": ("–", None),
        }[status]
        elapsed = (
            f" · {time.monotonic() - self.started.pop(name):.1f}s" if name in self.started else ""
        )
        fallback = {
            "not_probed": "Not checked",
            "not_confirmed": "Not confirmed",
            "rejected": "Request rejected",
        }.get(outcome, outcome)
        line(f"{icon} {label}: {detail or fallback}{elapsed}", fg=color)
        self.results[name] = status

    def finish(self, *, summary=True):
        self._clear()
        if self.reasoning_levels and summary:
            parts = [
                f"{level}: {_reasoning_tokens_summary(record)}"
                f"{'' if record['answer_correct'] else ' (wrong)'}"
                for level, record in self.reasoning_levels.items()
            ]
            line("Reasoning tokens · " + " · ".join(parts), dim=True)
        if self.results and summary:
            counts = [
                f"{sum(s == status for s in self.results.values())} {label}"
                for status, label in (
                    ("passed", "passed"),
                    ("attention", "need attention"),
                    ("skipped", "skipped"),
                )
            ]
            line("Results · " + " · ".join(counts), bold=True)
            click.echo()


def model_details(model, *, output_tokens, edited=False):
    """Show published model information without changing any request settings."""

    def tokens(value):
        return (
            f"{value:,} tokens"
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
            else "Not listed"
        )

    reasoning = model.get("reasoning") or {}
    levels = reasoning.get("supported_efforts") or []
    level_text = ", ".join(levels) if levels else "Not listed"
    default_text = reasoning.get("default_effort") or "Not listed"
    if not levels and isinstance(reasoning.get("default_enabled"), bool):
        level_text = (
            "Always on"
            if reasoning.get("mandatory") is True
            else "Thinking on/off; no named levels listed"
            if reasoning.get("mandatory") is False
            else "Thinking available; named levels not listed"
        )
        default_text = "Thinking on" if reasoning["default_enabled"] else "Thinking off"
    click.echo()
    line(f"Model details · {model['id']}", fg="bright_cyan", bold=True)
    for label, value in (
        ("Context window", tokens(model.get("context_length"))),
        (
            "Reported reply ceiling",
            tokens((model.get("top_provider") or {}).get("max_completion_tokens")),
        ),
        ("Reasoning levels", level_text),
        ("Default reasoning", default_text),
        (
            "Interface discovery limit",
            f"{output_tokens:,} tokens per reply (not reasoning checks)",
        ),
    ):
        source_key = {
            "Context window": "context_length",
            "Reported reply ceiling": "max_completion_tokens",
        }.get(label)
        source = (model.get("limit_sources") or {}).get(source_key)
        source_label = {
            "catalogue": "catalogue",
            "endpoint": "endpoint",
            "endpoint_input_limit": "endpoint input limit; conservative bound",
        }.get(source, source)
        line(f"{label:<25} {value}" + (f" · {source_label}" if source_label else ""))
    line(
        "Your edited settings."
        if edited
        else "Sources: endpoint limits take priority; catalogue information fills gaps."
        if model.get("endpoint_limits")
        else "Source: OpenRouter model listing. Your server may use different limits.",
        dim=True,
    )
    line("Setup checks do not measure maximum limits.", dim=True)
    line(
        "The reply ceiling is metadata, not your per-reply budget. Input, reasoning and the answer share the context window.",
        dim=True,
    )
    context = model.get("context_length")
    ceiling = (model.get("top_provider") or {}).get("max_completion_tokens")
    if (
        isinstance(context, int)
        and isinstance(ceiling, int)
        and context > 0
        and ceiling >= context * 0.8
    ):
        line(
            f"A reply using this whole ceiling would leave only {max(0, context - ceiling):,} tokens for input. Do not use it as an everyday reply budget."
            if ceiling < context
            else "This ceiling meets or exceeds the context window. Verify the server limits before using it as a reply budget.",
            fg="yellow",
        )
    click.echo()
