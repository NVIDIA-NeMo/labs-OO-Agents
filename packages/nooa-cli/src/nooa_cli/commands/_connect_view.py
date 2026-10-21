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
        return "Could not read the reply."
    if error == "APIConnectionError":
        return "Could not reach the server. Check the connection or try again."
    if isinstance(status, int) and 400 <= status < 500:
        return "Request rejected by this server. Check the request settings."
    return None


def connection_detail(record):
    """Explain the configured answer check without exposing its internal contract."""
    from nooa.unifiedllm.connect._records import TYPED_TEXT_BASELINE, text_baseline_passed

    if text_baseline_passed(record):
        return "Passed"
    if record.get("outcome") == "not_probed":
        return (
            "Not tested: check budget used up"
            if record.get("reason") == "budget exhausted"
            else "Not tested"
        )
    failure = check_failure(record)
    if failure:
        return "Failed · " + failure
    if record.get("baseline_kind") != TYPED_TEXT_BASELINE:
        return "Not verified with the selected settings"
    if record.get("finish_reason") == "length":
        return "Failed · Reply reached the token limit before finishing"
    if record.get("finish_reason") in {"error", "content_filter"}:
        return "Failed · Reply failed or was filtered"
    if "finish_reason" not in record:
        return "Failed · No usable answer returned"
    if record.get("finish_reason") != "stop":
        return "Failed · Reply did not finish"
    if record.get("final_answer_observed") is not True:
        return "Failed · No final answer returned"
    if record.get("typed_result_validated") is not True:
        return "Failed · Answer was not in the required format"
    if record.get("answer_correct") is not True:
        return "Failed · Answer did not match the test question"
    return "Not verified: selected settings were not observed in the request"


def check_reason(reason):
    """Short terminal wording; library records and diagnostic detail stay intact."""
    reason = reason or ""
    replacements = {
        "budget exhausted": "check budget used up",
        "Decision baseline not accepted; fresh checks not tested": "Decision connection check did not pass; later checks skipped",
        "Typed decision baseline did not pass; image not tested": "Decision answer checks did not pass; image check skipped",
        "Native typed response accepted but fixture answer incorrect": "Answer format accepted, but the sample answer was incorrect. Try another model.",
        "Image access, known color accuracy and actual wire attachment must all be confirmed": "Could not confirm image delivery and the sample answer. Review image settings or try another connection.",
        "This interface has no native video input mapping here; select video manually as unverified.": "Video testing isn't available for this API format. Use Chat to test video, or enable it without testing.",
        "This interface has no portable inline WAV input shape here; select audio manually as unverified.": "Audio testing isn't available for this API format. Use Chat to test audio, or enable it without testing.",
        "Configured reply cap changed after planning; make a new plan": "Reply limit changed after planning; restart setup",
        "probe timeout must be between 1 and 120 seconds": "Check timeout must be 1–120 seconds",
        "selected reasoning level changes the approved reply cap": "Selected reasoning level needs a different reply limit; restart setup",
        "reply truncated at the saved model limit": "Reply reached the saved token limit before finishing",
        "Only a small portion was cached; try a longer prompt or check the server's caching support": "Little cache reuse reported. Try a longer prompt or review the server's caching settings.",
        "Reply not understood (ReasoningReplayError)": "Could not read the reply",
        "Reply not understood by the runtime": "Could not read the reply",
        "Timed out; server receipt and completion are unknown": "Request timed out; completion is unknown",
        "The client did not send the requested settings; check parameter filtering": "Selected settings were not sent. Review the client settings.",
        "previous check failed": "an earlier check failed",
        "not approved or earlier check failed": "not approved or an earlier check failed",
        "Could not observe the actual request; the runtime used an uninstrumented HTTP client": "Could not verify the settings sent to the server",
        "request capture missing or server exceeded the reply cap": "Could not verify the request or its reply limit",
        "configured reply cap did not reach the request": "The saved reply limit was not sent",
        "declared output cap exceeds the approved probe cap": "Reply limit exceeds the approved test limit",
        "probes require one non-streaming generation": "Checks require one complete reply without streaming",
        "Reasoning state was replayed, but the selected reasoning settings did not reach every request; check parameter filtering in the client": "Reasoning was carried forward, but some selected settings were not sent. Review the client settings.",
        "Reasoning was returned but not preserved in the follow-up request; check replay settings or try another interface": "Reasoning was not carried forward. Review the settings or try another API format.",
        "No replayable reasoning returned; check reasoning/replay settings or try another interface. This does not mean reasoning is off": "Could not confirm reasoning carried forward. Try another API format; reasoning may still be active.",
        "No cache reuse reported on these continuations; provider caching can vary. The model connection still works": "No cache reuse reported. Caching can vary; the connection still works.",
        "Cache markers were sent, but the server reported no reuse; retry later or check server support": "Caching was requested, but no reuse was reported. Check the server's caching settings.",
        "No cache markers or cache reads observed; check the runtime's cache defaults and server support": "Caching was not requested and no reuse was reported. Review the client and server caching settings.",
        "A message from an earlier turn rendered differently once replayed as history; explicit cache breakpoints will not accumulate hits across turns until the renderer keeps that message's wire shape stable": "Earlier messages changed when sent again, preventing cache reuse. Review the client settings.",
    }
    if reason.startswith("Video testing not implemented"):
        return "Video testing isn't available for this API format. Use Chat to test video, or enable it without testing."
    if reason.startswith("Audio testing not implemented"):
        return "Audio testing isn't available for this API format. Use Chat to test audio, or enable it without testing."
    if reason.startswith("Text baseline") or "; media not tested" in reason:
        return "Connection check did not pass; media not tested"
    return replacements.get(reason, reason)


def media_test_detail(kind, record, *, media_inputs=None):
    """Display effective permissions separately from unchanged test evidence."""
    outcome = record.get("outcome")
    if outcome == "passed":
        if record.get("client") == "decision":
            return "Passed · Image sent; access and sample color confirmed"
        return "Passed · Model reported access to this input"
    if outcome == "unavailable":
        setting = (
            f"{media_label(kind)}: enabled (not tested)"
            if media_inputs is not None and kind in media_inputs
            else f"{media_label(kind)}: off"
        )
        return f"Not available · The model reported no access to this sample; {setting}."
    if outcome == "not_probed":
        return f"{media_label(kind).removesuffix('s')} check skipped" + (
            ": " + check_reason(record["reason"]) if record.get("reason") else ""
        )
    reason = record.get("reason", "").removesuffix(f"; {kind} support was not confirmed")
    if record.get("client") == "decision" and not record.get("error"):
        if record.get("attachment_observed") is False:
            reason = "Image was not sent as planned. Review image request settings"
        elif record.get("access_reported") is False:
            reason = "The model reported no access to the sample. Try another connection or leave images off"
        elif record.get("answer_correct") is False:
            reason = (
                "The model did not identify the sample color. Try another model or leave images off"
            )
        else:
            reason = check_reason(reason)
    if outcome == "rejected":
        detail = "Server rejected the media request"
    elif record.get("timeout_kind") or reason.startswith("Timed out"):
        detail = "Request timed out; completion is unknown"
    elif reason.startswith("Service returned HTTP"):
        # Keep the concise HTTP wording explicitly requested for media failures.
        detail = reason
    elif reason == "Could not read the answer":
        detail = reason
    elif record.get("error") or reason.startswith(("Could not connect", "Request failed")):
        detail = check_failure(record) or reason.split(" (")[0] or "Request failed"
    else:
        detail = {
            "Actual request was not observed; media and output settings could not be verified": "Could not verify the attachment and reply settings sent to the server",
            "Generated attachment or output settings were missing from the observed request": "The attachment or reply settings were not sent as planned",
            "Reply was not a completed final answer": "Reply did not finish",
            "Reply not understood by the runtime": "Could not read the reply",
        }.get(reason, reason or f"Could not confirm {kind} from the sample")
    setting = (
        f"{media_label(kind)}: enabled (not confirmed by this test)"
        if media_inputs is not None and kind in media_inputs
        else f"{media_label(kind)}: off"
        if media_inputs is not None
        else f"{media_label(kind)}: not confirmed"
    )
    return f"Failed · {detail.rstrip('.')}; {setting}."


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
        detail = "Check the connection and credentials; use --show-diagnostics for details."
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


def progress_result(icon, label, detail, *, elapsed="", headline=None, fg=None):
    """Keep outcome and duration on the header; explanations always follow below."""
    width = max(8, shutil.get_terminal_size((80, 24)).columns - 2)
    headline = headline or detail
    body = detail.removeprefix(headline).removeprefix(" · ").removeprefix(": ")
    header = f"{icon} {label}: {headline}{elapsed}"
    if len(header) <= width:
        parts = [header]
    else:
        parts = textwrap.wrap(f"{icon} {label}:", width=width, subsequent_indent="  ")
        parts += textwrap.wrap(
            headline + elapsed, width=width, initial_indent="  ", subsequent_indent="  "
        )
    if body:
        parts += textwrap.wrap(
            "· " + body,
            width=width,
            initial_indent="  ",
            subsequent_indent="  ",
            break_on_hyphens=False,
        )
    for part in parts:
        if "NO_COLOR" not in os.environ:
            part = click.style(part, fg=fg)
        click.echo("  " + part)


def check_label(name):
    """One human vocabulary for checks and review rows."""
    if name.startswith("level:"):
        return "Reasoning (" + name[6:] + ")"
    if name.startswith("media:"):
        kind = name[6:]
        return ("PDF" if kind == "pdf" else kind.capitalize()) + " input"
    return {
        "chat": "OpenAI-compatible Chat",
        "responses": "OpenAI Responses",
        "anthropic": "Anthropic Messages",
        "routing": "Connection",
        "decision:typed": "Typed decisions (boolean, choice, score)",
        "decision:positive": "Decision positive case",
        "decision:negative": "Decision negative case",
        "tools": "Tool use",
        "session:seed": "Conversation (1/3)",
        "session:replay": "Conversation (2/3)",
        "session:repeat": "Conversation (3/3)",
        "cache": "Cache reuse",
        "reasoning_retention": "Reasoning retention",
        "encrypted_reasoning": "Encrypted reasoning",
        "session": "Conversation",
    }.get(name, name)


def check_outcome(name, record, status):
    """Presentation only: never upgrade the evidence policy's failed/skipped checks."""
    outcome = record.get("outcome")
    if status == "skipped":
        return "Skipped"
    if outcome == "unavailable" or outcome == "warning" and not record.get("error"):
        return "Not available"
    if (
        name == "reasoning_retention"
        and outcome == "not_confirmed"
        and record.get("reason", "").startswith("No replayable reasoning returned")
    ):
        return "Not available"
    return "Passed" if status == "passed" else "Failed"


def intro(*, checks, output_tokens, budget_tokens, reasoning_output_tokens=4096):
    click.echo()
    line("NOOA  /  CONNECT", fg="bright_cyan", bold=True)
    line("Add a model to your workspace.", dim=True)
    click.echo()
    if checks:
        line(
            "Checks may incur charges. The token budget is an estimate, not a spending cap.",
            fg="yellow",
        )
        line(
            f"Check budget: {format_budget(budget_tokens)}"
            if budget_tokens is not None
            else "Check budget: unlimited",
            dim=True,
        )
        line(
            f"API discovery: one connection call per candidate; chat reply limit {output_tokens:,} tokens. Decision calls have no reply limit.",
            dim=True,
        )
        line(
            f"Each chat API format may also get a reasoning check: {reasoning_output_tokens:,} output tokens per call.",
            dim=True,
        )
        line(
            "After setup: connection, tools, reasoning levels and up to 3 conversation replies, using your saved reply limits. Checks that exceed the token budget are skipped.",
            dim=True,
        )
        line(
            "Optional media tests: one generated sample per selected format (up to 4), after the connection check passes. No user files are used.",
            dim=True,
        )
        line(
            "Conversation checks include a longer prompt to test caching. If encrypted reasoning is rejected, one check without it may follow within the same token budget.",
            dim=True,
        )
    else:
        line(
            "Connection: not tested. Model lists and public details may still be fetched.",
            dim=True,
        )
    line("Skip paid checks: --no-probe    Help: F1", dim=True)
    if not sys.stdin.isatty():
        line("Manual setup: docs/model-configuration.md", dim=True)
        line("Agent workflow: nooa-connect", dim=True)


def step(number, title):
    click.echo()
    line(f"{number} / 4  ·  {title}", fg="bright_cyan", bold=True)
    click.echo()


class CheckProgress:
    """Compact human progress; replace only the active line on a real terminal."""

    def __init__(self, *, diagnostics=False):
        self.diagnostics = diagnostics
        self.inline = click.get_text_stream("stdout").isatty()
        self.active = False
        self.started = {}
        self.results = {}
        self.reasoning_levels = {}

    def _clear(self):
        if self.active:
            click.echo("\r\033[2K", nl=False, color=True)
            self.active = False

    def update(self, name, record, *, missing_reasoning=False, media_inputs=None):
        label = check_label(name)
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
        detail = check_failure(record) or check_reason(record.get("reason"))
        if name.startswith("media:"):
            detail = media_test_detail(name[6:], record, media_inputs=media_inputs)
        if outcome == "accepted":
            detail = ""
            if name == "tools" and not record.get("tool_observed"):
                detail = "No tool call returned"
            elif name.startswith("level:") and missing_reasoning:
                detail = "No reasoning details returned"
            if record.get("finish_reason") == "length":
                detail = "Ran out of reply tokens before finishing"
                if name.startswith("level:"):
                    detail += " — increase the reply limit to use this reasoning level"
            elif record.get("finish_reason") in {"error", "content_filter"}:
                detail = "Reply incomplete; could not confirm this check"
            if (
                name.startswith("level:")
                and record.get("reasoning_observed")
                and isinstance(record.get("answer_correct"), bool)
            ):
                self.reasoning_levels[name[6:]] = record
        elif name == "cache" and outcome == "confirmed" and record.get("input_tokens"):
            cached, total = record.get("cached_input_tokens", 0), record["input_tokens"]
            detail = f"Reused {cached / total:.0%} of input ({cached:,} / {total:,} tokens)"
            if record.get("readings"):
                detail += " · " + ", ".join(
                    f"{r['turn']}: {r.get('cached_input_tokens') or 0:,} cached"
                    for r in record["readings"]
                )
        elif name == "encrypted_reasoning" and outcome == "rejected":
            detail = (
                "Server rejected encrypted reasoning; disabled. Reasoning may not carry forward."
            )
        elif name == "reasoning_retention" and outcome == "confirmed":
            detail = "Preserved in the next request"
        from nooa.unifiedllm.connect._records import check_status

        status = check_status(name, record, missing_reasoning=missing_reasoning)
        if (
            name == "routing"
            and record.get("client") != "decision"
            and record.get("baseline_kind") != "interface_acceptance"
        ):
            detail = connection_detail(record)
        icon, color = {
            "passed": ("✓", "green"),
            "attention": ("!", "yellow"),
            "skipped": ("–", None),
        }[status]
        elapsed = (
            f" · {time.monotonic() - self.started.pop(name):.1f}s" if name in self.started else ""
        )
        headline = check_outcome(name, record, status)
        # Accepted requests and finished replies are evidence, not additional outcomes.
        if (
            headline == "Passed"
            and name not in {"cache", "encrypted_reasoning"}
            and not (name == "media:image" and record.get("client") == "decision")
        ):
            detail = ""
            if record.get("reason") == "previous result reused":
                detail = "Already checked with these settings"
            if name.startswith("level:") and record.get("answer_correct") is False:
                detail = "Reasoning returned; test answer incorrect"
            if self.diagnostics and name.startswith("level:"):
                answer = record.get("answer_correct")
                correctness = (
                    "answer correct"
                    if answer is True
                    else "answer incorrect"
                    if answer is False
                    else ""
                )
                detail = " · ".join(filter(None, (_reasoning_tokens_label(record), correctness)))
        if name == "routing" and record.get("client") != "decision" and status != "passed":
            detail = connection_detail(record)
        if detail:
            for prefix in (
                "Passed · ",
                "Failed · ",
                "Skipped · ",
                "Not tested · ",
                "Not available · ",
            ):
                if detail.startswith(prefix):
                    detail = detail.removeprefix(prefix)
                    break
            if detail == "Not tested":
                detail = ""
            elif detail.startswith("Not tested: "):
                detail = detail.removeprefix("Not tested: ")
        if headline == "Not available":
            icon, color = "–", "yellow"
        progress_result(
            icon, label, detail or headline, elapsed=elapsed, headline=headline, fg=color
        )
        self.results[name] = headline

    def finish(self, *, summary=True):
        self._clear()
        if self.reasoning_levels and summary and self.diagnostics:
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
                    ("Passed", "passed"),
                    ("Not available", "not available"),
                    ("Failed", "failed"),
                    ("Skipped", "skipped"),
                )
            ]
            line("Results · " + " · ".join(counts), bold=True)
            click.echo()


def media_label(kind):
    return {"image": "Images", "audio": "Audio", "video": "Video", "pdf": "PDFs"}.get(
        kind, kind.capitalize()
    )


def media_settings(entry):
    """Show saved permissions separately from the evidence, without raw enum values."""
    enabled = entry.get("media_inputs", [])
    facts = entry.get("provenance", {}).get("probes", {})
    kinds = (
        ("image",) if entry.get("client_type") == "decision" else ("image", "audio", "video", "pdf")
    )
    for kind in kinds:
        record = facts.get("media:" + kind, {})
        state = (
            "off"
            if kind not in enabled
            else "enabled (test passed)"
            if record.get("outcome") == "passed"
            else "enabled (not confirmed by this test)"
            if record.get("outcome") in {"inconclusive", "rejected", "unavailable", "not_confirmed"}
            else "enabled (not tested)"
        )
        line(f"{media_label(kind)}: {state}")


def model_details(model, *, output_tokens, edited=False):
    """Show available facts only; unknown fields remain editable and unknown internally."""

    def tokens(value):
        return (
            f"{value:,} tokens"
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
            else None
        )

    reasoning = model.get("reasoning") or {}
    levels = reasoning.get("supported_efforts") or []
    level_text = ", ".join(levels) if levels else None
    default_text = reasoning.get("default_effort")
    if not levels and isinstance(reasoning.get("default_enabled"), bool):
        level_text = (
            "Always on"
            if reasoning.get("mandatory") is True
            else "Thinking on/off"
            if reasoning.get("mandatory") is False
            else "Thinking available"
        )
        default_text = "Thinking on" if reasoning["default_enabled"] else "Thinking off"
    rows = [
        (label, value)
        for label, value in (
            ("Context window", tokens(model.get("context_length"))),
            (
                "Model reply maximum",
                tokens((model.get("top_provider") or {}).get("max_completion_tokens")),
            ),
            ("Reasoning levels", level_text),
            ("Default reasoning", default_text),
        )
        if value
    ]
    if not rows:
        line(
            "No model settings listed. Enter known limits or continue with NOOA defaults.", dim=True
        )
        return
    click.echo()
    line(f"Model details · {model['id']}", fg="bright_cyan", bold=True)
    for label, value in rows:
        source_key = {
            "Context window": "context_length",
            "Model reply maximum": "max_completion_tokens",
        }.get(label)
        source = (model.get("limit_sources") or {}).get(source_key)
        source_label = {
            "catalogue": "OpenRouter",
            "endpoint": "your server",
            "endpoint_input_limit": "server input limit",
        }.get(source, source)
        line(f"{label:<25} {value}" + (f" · {source_label}" if source_label and not edited else ""))
    line(
        "Your edited settings."
        if edited
        else "Model details from your server, with OpenRouter details where needed."
        if model.get("endpoint_limits")
        else "Model details from OpenRouter. Check your server's limits if they differ.",
        dim=True,
    )
    line(f"API discovery reply limit: {output_tokens:,} tokens.", dim=True)
    line(
        "Choose your reply limit next. Input, thinking and the answer share the context window.",
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
            f"A reply using the model maximum would leave only {max(0, context - ceiling):,} tokens for input. Choose a smaller reply limit."
            if ceiling < context
            else "The model reply maximum reaches the context window. Check your server limits and choose a smaller reply limit to leave room for input.",
            fg="yellow",
        )
    click.echo()
