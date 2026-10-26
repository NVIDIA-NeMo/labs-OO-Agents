# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Terminal output for ``nooa connect``: headings, aligned fields, test call results.

Everything printed by the wizard goes through these helpers. Output is wrapped
to the terminal width, and colour is turned off when ``NO_COLOR`` is set. The
module also turns the records of test calls (from ``nooa.unifiedllm.connect``)
into short, plain messages. Raw provider error bodies and API keys are never
printed; errors are described from their type and HTTP status.
"""

import os
import shutil
import sys
import textwrap
import time
from contextlib import contextmanager

import click


@contextmanager
def quiet_provider_messages():
    """Context manager that stops LiteLLM from printing its own debug/help banners.

    While test calls run, the CLI prints its own short summary of each
    result; LiteLLM's banners would add noise and could show raw provider
    details. The previous setting is restored on exit. Does nothing if
    LiteLLM is not imported.
    """
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
    """Format a test call token budget, showing the "no budget given" value as "unlimited"."""
    from nooa.unifiedllm.connect import DEFAULT_CHECK_BUDGET

    # Without --budget-tokens the budget is DEFAULT_CHECK_BUDGET: a very large
    # integer (not float("inf"), so it stays valid in JSON and integer maths).
    # The remaining budget is that value minus the tokens used so far (up to
    # about 1.2M for a full run; see TOKEN_RESERVATION), so it is never exactly
    # equal. Treat anything within a fixed 10**9 tokens of it as unlimited. A
    # fixed margin, rather than a fraction of the value, keeps a very large
    # budget chosen with --budget-tokens from being shown as unlimited.
    return "unlimited" if tokens >= DEFAULT_CHECK_BUDGET - 10**9 else f"{tokens:,}"


def _reasoning_tokens_label(record):
    """Describe how many reasoning tokens a reasoning-level test call used, as far as known.

    ``record`` is the result of one ``level:<name>`` test call. Returns ""
    when nothing is known.

    - A positive ``reasoning_tokens`` count is shown as is.
    - If the provider withheld the reasoning text, the reasoning count cannot
      be trusted. Anthropic does this in two ways: a ``redacted_thinking``
      block (encrypted, no text), or a signed ``thinking`` block whose text is
      empty (seen with Claude Sonnet 5 / Opus 5 on Azure and Bedrock).
      LiteLLM estimates reasoning tokens from the length of the visible text,
      so it reports 0 however much thinking happened. The total output tokens
      are shown instead; Anthropic counts thinking as ordinary output tokens.
    - LiteLLM only makes that estimate for Anthropic and Bedrock. For other
      providers (seen with Qwen and DeepSeek) no count is reported even when
      reasoning text came back, so the length of that text in characters is
      shown as a rough size.
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
    """Short form of ``_reasoning_tokens_label`` for the summary line at the end of a run.

    Uses the same rules as ``_reasoning_tokens_label``. In particular the
    "N output" form is only used when reasoning was actually seen in the
    reply; otherwise a level where the model returned output but did not
    reason would look as if its reasoning cost had been measured.
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
    """Return a short message for a failed test call, or None if no rule matches.

    The message is chosen from the error type name and HTTP status in
    ``outcome``; the provider's error body is never shown.
    """
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
    """Return the result text for the basic test call (``routing``) of a model entry.

    That call asks a question with a known typed answer. The text says
    whether it passed, was not run, or why it failed (rejected request, reply
    cut off, no final answer, wrong format, wrong answer), in plain words
    rather than the record's field names.
    """
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
    """Return terminal wording for a reason string from a test call record.

    Only the printed text changes; the record and the troubleshooting output
    keep the library's original wording. Unknown reasons are returned as is.
    """
    reason = reason or ""
    replacements = {
        "budget exhausted": "check budget used up",
        "Decision baseline not accepted; fresh checks not tested": "Decision connection check did not pass; later checks skipped",
        "Typed decision baseline did not pass; image not tested": "Decision answer checks did not pass; image check skipped",
        "Native typed response accepted but fixture answer incorrect": "Answer format accepted, but the sample answer was incorrect. Try another model.",
        "Image access, known color accuracy and actual wire attachment must all be confirmed": "Could not confirm image delivery and the sample answer. Review image settings or try another connection.",
        "This interface has no native video input mapping here; select video manually as unverified.": "Native video input is not supported by this API format.",
        "This interface has no portable inline WAV input shape here; select audio manually as unverified.": "Native audio input is not supported by this API format.",
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
        "No replayable reasoning returned; check reasoning/replay settings or try another interface. This does not mean reasoning is off": "No replayable reasoning was returned in this check.",
        "No cache reuse reported on these continuations; provider caching can vary. The model connection still works": "No cache reuse reported. Caching can vary; the connection still works.",
        "Cache markers were sent, but the server reported no reuse; retry later or check server support": "Caching was requested, but no reuse was reported. Check the server's caching settings.",
        "No cache markers or cache reads observed; check the runtime's cache defaults and server support": "Caching was not requested and no reuse was reported. Review the client and server caching settings.",
        "A message from an earlier turn rendered differently once replayed as history; explicit cache breakpoints will not accumulate hits across turns until the renderer keeps that message's wire shape stable": "Earlier messages changed when sent again, preventing cache reuse. Review the client settings.",
    }
    if reason.startswith("Video testing not implemented"):
        return "Native video input is not supported by this API format."
    if reason.startswith("Audio testing not implemented"):
        return "Native audio input is not supported by this API format."
    if reason.startswith("Text baseline") or "; media not tested" in reason:
        return "Connection check did not pass; media not tested"
    return replacements.get(reason, reason)


def media_test_detail(kind, record, *, media_inputs=None):
    """Return the result text for one media test call (``media:<kind>``).

    The text has two parts: what the test call showed (passed, not
    available, rejected, failed and why), and whether ``kind`` is enabled in
    ``media_inputs``. The two are separate because a kind can be enabled
    without a passing test, and a test result does not change the setting.
    """
    outcome = record.get("outcome")
    if record.get("implementation_supported") is False or record.get("reason", "").startswith(
        "Unsupported media input for "
    ):
        return "Unsupported by this request interface · cannot be enabled."
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
    if outcome == "rejected" and record.get("client") != "decision":
        return (
            f"Not available · This model doesn't accept {media_label(kind).lower()}; it stays off."
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
        # Show the library's short "Service returned HTTP ..." message unchanged.
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
    """Return a short message (at most 1,000 characters) for an error during setup.

    YAML errors give only the file, line and column (not the file contents).
    ``OSError`` and ``ValueError`` use their message. Other errors, which may
    carry provider error bodies, get a generic hint. Any occurrence of the
    API key is replaced with ``[redacted]``.
    """
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


def line(text, *, fg=None, bold=False, dim=False, err=False):
    """Print ``text`` wrapped to the terminal width (at most 84 columns), indented and styled."""
    width = max(8, min(84, shutil.get_terminal_size((80, 24)).columns - 2))
    for part in textwrap.wrap(text, width=width, break_on_hyphens=False) or [""]:
        if "NO_COLOR" not in os.environ:
            part = click.style(part, fg=fg, bold=bold or None, dim=dim or None)
        click.echo("  " + part, err=err)


def note(text, *, err=False):
    """Print a dim note, or a yellow one on standard error when ``err`` is set."""
    line(text.strip(), dim=not err, fg="yellow" if err else None, err=err)


def warning(text, *, err=False):
    """Print a yellow warning, dropping a leading ``"Warning: "``."""
    line(text.removeprefix("Warning: "), fg="yellow", err=err)


def prompt_text(text, *, suffix=": "):
    """Return a prompt label wrapped to the terminal width, for Click's line input.

    Used when standard input is not a terminal; the prompt_toolkit widgets
    format their own labels.
    """
    width = max(8, shutil.get_terminal_size((80, 24)).columns - 2 - len(suffix))
    parts = textwrap.wrap(text, width=width, break_on_hyphens=False) or [""]
    label = "\n".join("  " + part for part in parts)
    return label  # No colour codes: Click prints this as is, also into pipes.


def fields(rows):
    """Print ``(label, value)`` rows with the values aligned in one column.

    On narrow terminals each label is printed on its own line above its
    value. Text is wrapped before colour is added, so wrapping never splits a
    colour code.
    """
    rows = [(label + ":", str(value)) for label, value in rows]
    if not rows:
        return
    columns = max(12, shutil.get_terminal_size((80, 24)).columns)
    width = min(100, columns - 4)
    label_width = max(len(label) for label, _ in rows)
    value_width = width - label_width - 1
    stacked = value_width < 16
    for label, value in rows:
        if stacked:
            for part in textwrap.wrap(label, width=width, break_on_hyphens=False):
                if "NO_COLOR" not in os.environ:
                    part = click.style(part, bold=True)
                click.echo("    " + part)
            for part in textwrap.wrap(value, width=width, break_on_hyphens=False) or [""]:
                click.echo("    " + part)
        else:
            parts = textwrap.wrap(value, width=value_width, break_on_hyphens=False) or [""]
            for index, part in enumerate(parts):
                key = label.ljust(label_width) if index == 0 else " " * label_width
                if "NO_COLOR" not in os.environ and index == 0:
                    key = click.style(key, bold=True)
                click.echo("    " + key + " " + part)


def section(title):
    """Print a blank line and then ``title`` as a bold cyan heading."""
    click.echo()
    width = max(10, min(100, shutil.get_terminal_size((80, 24)).columns - 2))
    for part in textwrap.wrap(title, width=width, break_on_hyphens=False):
        if "NO_COLOR" not in os.environ:
            part = click.style(part, bold=True, fg="bright_cyan")
        click.echo("  " + part)


def progress_result(icon, label, detail, *, elapsed="", headline=None, fg=None):
    """Print one test call result: a header line, then any explanation below it.

    The header is ``<icon> <label>: <headline><elapsed>``. The rest of
    ``detail`` (after the headline) goes on the following lines, so the
    result and time stay on the first line however long the explanation is.
    """
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


# Test call names that are API styles: results of the "interfaces" discovery.
INTERFACE_NAMES = {"chat", "responses", "anthropic", "systemone", "openai"}


def check_label(name):
    """Return the display name of a test call (e.g. ``"tools"`` -> ``"Tool use"``).

    Used for both progress lines and review rows so they use the same names.
    """
    if name.startswith("level:"):
        return "Reasoning (" + name[6:] + ")"
    if name.startswith("media:"):
        kind = name[6:]
        return ("PDF" if kind == "pdf" else kind.capitalize()) + " input"
    return {
        "chat": "OpenAI-compatible Chat",
        "responses": "OpenAI Responses",
        "anthropic": "Anthropic Messages",
        "systemone": "System One Decisions",
        "openai": "OpenAI Decisions",
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
        "explicit_cache": "Prompt caching",
        "session": "Conversation",
    }.get(name, name)


def check_outcome(name, record, status):
    """Return the headline for a test call result: Passed, Failed, Skipped or Not available.

    ``status`` is the library's verdict (``check_status``). This function
    only chooses the wording; it may show a failure as "Not available" (the
    model does not offer a feature), but never shows a failed or skipped
    call as passed.
    """
    outcome = record.get("outcome")
    if status == "skipped":
        return "Skipped"
    if outcome == "unavailable" or outcome == "warning" and not record.get("error"):
        return "Not available"
    if name.startswith("media:") and outcome == "rejected":
        return "Not available"  # The model does not accept this media kind; it stays off.
    if (
        name == "reasoning_retention"
        and outcome == "not_confirmed"
        and record.get("reason", "").startswith("No replayable reasoning returned")
    ):
        return "Not available"
    return "Passed" if status == "passed" else "Failed"


def intro(*, checks, output_tokens, budget_tokens, reasoning_output_tokens=4096):
    """Print the wizard's title and short help lines."""
    click.echo()
    line("NOOA  /  CONNECT", fg="bright_cyan", bold=True)
    line("Add a model to your workspace.", dim=True)
    click.echo()
    if not checks:
        line("Tests are off.", dim=True)
    line("Skip tests: --no-probe    Help: F1", dim=True)
    if not sys.stdin.isatty():
        line("Manual setup: docs/model-configuration.md", dim=True)
        line("Agent workflow: nooa-connect", dim=True)


def step(number, title, *, total=5):
    """Print a step heading such as ``2 / 5  ·  Model``."""
    click.echo()
    line(f"{number} / {total}  ·  {title}", fg="bright_cyan", bold=True)
    click.echo()


class CheckProgress:
    """Prints the progress and result of test calls, one line per call.

    Call ``update`` with each record as it changes and ``finish`` at the end.
    On a terminal the "checking" line of the running call is overwritten by
    its result; elsewhere every line is kept. ``results`` maps each call name
    to its headline, for the summary counts.
    """

    def __init__(self, *, diagnostics=False):
        self.diagnostics = diagnostics
        self.inline = click.get_text_stream("stdout").isatty()
        self.active = False
        self.started = {}
        self.results = {}
        self.reasoning_levels = {}
        self.interface_seen = set()

    def _clear(self):
        """Erase the "checking" line of the running call, if one is shown."""
        if self.active:
            click.echo("\r\033[2K", nl=False, color=True)
            self.active = False

    def _interface_update(self, name, label, record):
        """Print a result of API style discovery (the ``interfaces`` stage).

        An API style the endpoint does not accept is shown as "Not
        available", not as a failure. The second result for the same style is
        its reasoning test and is labelled as such.
        """
        elapsed = (
            f" · {time.monotonic() - self.started.pop(name):.1f}s" if name in self.started else ""
        )
        accepted = record.get("outcome") == "accepted"
        reasoning_row = name in self.interface_seen
        self.interface_seen.add(name)
        if reasoning_row:
            label += " reasoning"
            if accepted and record.get("reasoning_observed"):
                icon, headline, detail, color = "✓", "Returned", "", "green"
            elif accepted:
                icon, headline, detail, color = "–", "Not returned", "", None
            else:
                icon, headline, color = "–", "Not available", None
                detail = check_failure(record) or check_reason(record.get("reason")) or ""
        elif accepted:
            icon, headline, detail, color = "✓", "Works", "", "green"
        else:
            icon, headline, color = "–", "Not available", None
            detail = check_failure(record) or check_reason(record.get("reason")) or ""
        progress_result(
            icon, label, detail or headline, elapsed=elapsed, headline=headline, fg=color
        )
        self.results[name] = headline

    def update(
        self, name, record, *, missing_reasoning=False, ignored_off=False, media_inputs=None
    ):
        """Show the state of test call ``name`` from its ``record``.

        A ``running`` record prints a "checking" line; a final record prints
        the result. ``missing_reasoning`` means a reasoning level returned no
        reasoning; ``ignored_off`` means reasoning came back although the
        level turns it off; ``media_inputs`` are the enabled media kinds.
        """
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
            return  # The cache and reasoning lines already show the session result.
        if name in INTERFACE_NAMES:
            self._interface_update(name, label, record)
            return
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
        elif name == "explicit_cache":
            detail = "Not supported by this server; turned off for this model."
        from nooa.unifiedllm.connect._records import check_status

        status = (
            "skipped"
            if name == "explicit_cache"
            else check_status(name, record, missing_reasoning=missing_reasoning)
        )
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
        if ignored_off:
            # The "off" level still returned reasoning, so the setting had no effect.
            headline = "Not available"
            detail = "Reasoning was still returned; this model may ignore this setting."
        # For a passed call, the "accepted" or "finished" details add nothing,
        # so show no detail except for the few cases below.
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
        """Clear the progress line and, if ``summary``, print the result counts."""
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
    """Return the display name of a media kind (``"pdf"`` -> ``"PDFs"``)."""
    return {"image": "Images", "audio": "Audio", "video": "Video", "pdf": "PDFs"}.get(
        kind, kind.capitalize()
    )


def media_settings(entry):
    """Print each media kind with its setting in ``entry`` and its test result.

    For example "enabled (test passed)" or "enabled (not tested)", so the
    user sees whether an enabled kind was actually tested.
    """
    enabled = entry.get("media_inputs", [])
    facts = entry.get("provenance", {}).get("probes", {})
    kinds = (
        ("image",) if entry.get("client_type") == "decision" else ("image", "audio", "video", "pdf")
    )
    rows = []
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
        rows.append((media_label(kind), state))
    fields(rows)


def model_details(
    model, *, output_tokens, edited=False, heading=True, attribution=True, render=True
):
    """Print the known limits and reasoning settings of ``model``, with their sources.

    ``model`` is a catalogue-style dict. Only values that are known are
    shown; unknown values stay unknown (they are not filled with defaults)
    and can be entered later. ``edited`` marks values the user typed in.
    With ``render=False`` nothing is printed and the rows are returned
    instead (an empty list if there are none). Also warns when the model's
    maximum reply length leaves little room for input in the context window.
    """

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
        if not render:
            return []
        line(
            "No model settings listed. Enter known limits or continue with NOOA defaults.", dim=True
        )
        return
    if heading and render:
        section("Model details")
        fields([("Model", model["id"])])
    displayed = []
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
        displayed.append(
            (
                "Input limit (context bound)" if source == "endpoint_input_limit" else label,
                value + (f" · {source_label}" if source_label and not edited else ""),
            )
        )
    if attribution:
        displayed.append(
            (
                "Source",
                "Your settings"
                if edited
                else "This server"
                if model.get("endpoint_limits")
                else "OpenRouter",
            )
        )
    displayed.append(("API discovery reply limit", f"{output_tokens:,} tokens"))
    if not render:
        return displayed
    fields(displayed)
    note("Choose your reply limit next. Input, thinking and the answer share the context window.")
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
