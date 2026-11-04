# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Readable terminal layout without changing the setup's cost or save policy."""

import os

import click
import pytest
from click.testing import CliRunner
from nooa_cli.commands import _connect_view as view


def test_intro_is_wrapped_and_keeps_cost_notice_and_manual_routes(monkeypatch):
    monkeypatch.setattr(view.shutil, "get_terminal_size", lambda *args: os.terminal_size((58, 24)))

    @click.command()
    def command():
        view.intro(checks=True, output_tokens=200, budget_tokens=4096)

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    assert all(len(line) <= 58 for line in result.output.splitlines())
    assert "may incur charges" in result.output
    assert "200" in result.output and "4,096" in result.output
    assert "--no-probe" in result.output
    assert "F1" in result.output
    assert "docs/model-configuration.md" in result.output
    assert "nooa-model-configuration" in result.output


def test_no_color_keeps_readable_titles_and_warning(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")

    @click.command()
    def command():
        view.intro(checks=True, output_tokens=200, budget_tokens=4096)
        view.step(1, "Connection")

    result = CliRunner().invoke(command, color=True)
    assert "\x1b[" not in result.output
    assert "Connection" in result.output
    assert "may incur charges" in result.output


def test_model_details_show_published_limits_separately_from_setup_cap():
    @click.command()
    def command():
        view.model_details(
            {
                "id": "vendor/example",
                "context_length": 128000,
                "top_provider": {"max_completion_tokens": 16384},
                "reasoning": {
                    "supported_efforts": ["low", "medium", "high"],
                    "default_effort": "medium",
                },
            },
            output_tokens=200,
        )

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    for text in (
        "vendor/example",
        "128,000",
        "16,384",
        "low, medium, high",
        "medium",
        "200",
    ):
        assert text in result.output
    assert "Reported reply ceiling" in result.output
    assert "Published output default" not in result.output
    assert "Interface discovery limit" in result.output
    assert "Source: OpenRouter" in result.output
    assert "not proof" not in result.output


def test_missing_model_details_stay_unknown_instead_of_becoming_recommendations():
    @click.command()
    def command():
        view.model_details(
            {
                "id": "vendor/example",
                "context_length": None,
                "top_provider": {"max_completion_tokens": False},
            },
            output_tokens=200,
        )

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    assert result.output.count("Not listed") == 4
    assert "0 tokens" not in result.output.replace("200 tokens", "")


def test_local_diagnostics_keep_validation_but_hide_provider_bodies(monkeypatch):
    import httpx

    monkeypatch.setenv("TEST_ERROR_KEY", "private-key")
    assert view.local_failure(ValueError("reasoning_levels must be a mapping")) == (
        "reasoning_levels must be a mapping"
    )
    error = ValueError("Invalid file private-key, transient-key")
    detail = view.local_failure(error, api_key="transient-key", api_key_env="TEST_ERROR_KEY")
    assert "private-key" not in detail and "transient-key" not in detail
    detail = view.local_failure(httpx.ConnectError("untrusted server text private-key"))
    assert "untrusted server text" not in detail and "private-key" not in detail


def test_reply_ceiling_and_boolean_reasoning_metadata_are_explained():
    @click.command()
    def command():
        view.model_details(
            {
                "id": "example/model",
                "context_length": 262144,
                "top_provider": {"max_completion_tokens": 235929},
                "reasoning": {"mandatory": False, "default_enabled": True},
            },
            output_tokens=200,
            edited=True,
        )

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    assert "26,215" in result.output
    assert "metadata, not your per-reply budget" in result.output
    assert "Thinking on/off" in result.output
    assert "Thinking on" in result.output
    assert "Your edited settings" in result.output


def test_progress_reports_evidence_and_skips_without_payloads(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(view.shutil, "get_terminal_size", lambda *args: os.terminal_size((58, 24)))

    @click.command()
    def command():
        progress = view.CheckProgress()
        progress.update("tools", {"outcome": "running"})
        progress.update(
            "tools", {"outcome": "accepted", "tool_observed": False, "request": "SECRET"}
        )
        progress.update("level:high", {"outcome": "accepted"}, missing_reasoning=True)
        progress.update("level:low", {"outcome": "not_probed", "reason": "budget exhausted"})
        progress.update(
            "cache", {"outcome": "confirmed", "input_tokens": 10000, "cached_input_tokens": 9000}
        )
        progress.update("reasoning_retention", {"outcome": "confirmed"})
        progress.finish()

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    assert "Tool use — checking" in result.output
    assert "No tool call returned" in result.output
    assert "No reasoning details returned" in result.output
    assert "budget exhausted" in result.output
    assert "Reused 90%" in result.output
    assert "2 passed · 0 not available · 2 failed · 1 skipped" in " ".join(result.output.split())
    assert "SECRET" not in result.output
    assert "\x1b" not in result.output
    assert all(len(line) <= 58 for line in result.output.splitlines())


def test_terminal_progress_clears_active_row_on_completion_and_cancel(monkeypatch):
    from io import StringIO

    class Terminal(StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr(click, "get_text_stream", lambda name: Terminal())

    @click.command()
    def command():
        progress = view.CheckProgress()
        progress.update("chat", {"outcome": "running"})
        progress.update("chat", {"outcome": "accepted"})
        progress.update("tools", {"outcome": "running"})
        progress.finish()

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    assert result.stdout.count("\r\x1b[2K") == 2
    assert "OpenAI-compatible Chat: Passed" in result.output


def test_wrong_answer_shows_a_passed_icon_not_attention():
    @click.command()
    def command():
        progress = view.CheckProgress()
        progress.update(
            "level:low",
            {
                "outcome": "accepted",
                "reasoning_observed": True,
                "answer_correct": False,
                "reasoning_tokens": 800,
            },
        )
        progress.finish()

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    normalized_output = " ".join(result.output.split())
    assert "✓ Reasoning (low):" in normalized_output
    assert "answer incorrect" in normalized_output
    assert "Results · 1 passed · 0 not available · 0 failed · 0 skipped" in normalized_output


def test_finish_summarizes_reasoning_tokens_across_levels():
    @click.command()
    def command():
        progress = view.CheckProgress(diagnostics=True)
        progress.update(
            "level:max",
            {
                "outcome": "accepted",
                "reasoning_observed": True,
                "answer_correct": True,
                "reasoning_tokens": 224,
            },
        )
        progress.update(
            "level:low",
            {
                "outcome": "accepted",
                "reasoning_observed": True,
                "answer_correct": False,
                "reasoning_tokens": 800,
            },
        )
        progress.finish()

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    normalized_output = " ".join(result.output.split())
    assert "Reasoning tokens · max: 224 · low: 800 (wrong)" in normalized_output


def test_finish_omits_reasoning_summary_when_no_level_ran():
    @click.command()
    def command():
        progress = view.CheckProgress()
        progress.update("routing", {"outcome": "accepted"})
        progress.finish()

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    assert "Reasoning tokens ·" not in result.output


def test_length_finish_reason_says_it_ran_out_of_tokens():
    @click.command()
    def command():
        progress = view.CheckProgress()
        progress.update(
            "level:xhigh",
            {
                "outcome": "accepted",
                "reasoning_observed": True,
                "answer_correct": False,
                "finish_reason": "length",
            },
        )
        progress.finish()

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    normalized_output = " ".join(result.output.split())
    assert "Ran out of reply tokens before finishing" in normalized_output


@pytest.mark.parametrize("finish_reason", ["error", "content_filter"])
def test_error_or_filtered_finish_reason_keeps_the_generic_message(finish_reason):
    @click.command()
    def command():
        progress = view.CheckProgress()
        progress.update(
            "level:high",
            {
                "outcome": "accepted",
                "reasoning_observed": True,
                "answer_correct": False,
                "finish_reason": finish_reason,
            },
        )
        progress.finish()

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    normalized_output = " ".join(result.output.split())
    assert "Reply incomplete; could not confirm this check" in normalized_output
    assert "Ran out of reply tokens" not in normalized_output


def test_length_advice_to_raise_the_budget_is_only_for_level_checks():
    @click.command()
    def command():
        progress = view.CheckProgress()
        progress.update("tools", {"outcome": "accepted", "finish_reason": "length"})
        progress.finish()

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    assert "Ran out of reply tokens before finishing" in result.output
    assert "increase the reply budget" not in result.output


def test_withheld_reasoning_text_shows_output_tokens_not_a_zero_count():
    """Covers a signed "thinking" block whose text is empty: reasoning_encrypted
    is True but there's no meaningful byte size to report (a signature's
    length doesn't scale with how much was thought).
    """

    @click.command()
    def command():
        progress = view.CheckProgress(diagnostics=True)
        progress.update(
            "level:max",
            {
                "outcome": "accepted",
                "reasoning_observed": True,
                "reasoning_encrypted": True,
                "reasoning_tokens": 0,
                "output_tokens": 687,
                "answer_correct": True,
            },
        )
        progress.finish()

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    normalized_output = " ".join(result.output.split())
    assert "reasoning text withheld by the provider (687 output tokens, not split out)" in (
        normalized_output
    )
    assert "0 reasoning tokens" not in normalized_output
    assert "Reasoning tokens · max: 687 output (withheld)" in normalized_output


def test_withheld_reasoning_text_shows_its_byte_size_when_known():
    """Covers a genuine redacted_thinking block, which does carry a
    measurable opaque data blob.
    """

    @click.command()
    def command():
        progress = view.CheckProgress(diagnostics=True)
        progress.update(
            "level:max",
            {
                "outcome": "accepted",
                "reasoning_observed": True,
                "reasoning_encrypted": True,
                "reasoning_encrypted_bytes": 100,
                "reasoning_tokens": 0,
                "output_tokens": 687,
                "answer_correct": True,
            },
        )
        progress.finish()

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    normalized_output = " ".join(result.output.split())
    assert (
        "reasoning text withheld by the provider "
        "(687 output tokens, not split out; ~100 bytes of encrypted state)"
    ) in normalized_output
    assert "Reasoning tokens · max: 687 output (withheld, ~100B)" in normalized_output


def test_visible_reasoning_with_no_token_estimate_shows_its_char_count():
    """Covers Qwen/DeepSeek-style routes: real reasoning text came back but
    litellm never attempted a token-count estimate for it.
    """

    @click.command()
    def command():
        progress = view.CheckProgress(diagnostics=True)
        progress.update(
            "level:on",
            {
                "outcome": "accepted",
                "reasoning_observed": True,
                "reasoning_encrypted": False,
                "reasoning_tokens": 0,
                "reasoning_text_chars": 2400,
                "output_tokens": 2329,
                "answer_correct": True,
            },
        )
        progress.finish()

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    normalized_output = " ".join(result.output.split())
    assert (
        "2,329 output tokens (reasoning tokens not reported separately; ~2,400 reasoning chars)"
    ) in normalized_output
    assert "Reasoning tokens · on: 2,329 output, ~2,400 chars" in normalized_output


def test_visible_reasoning_without_char_count_omits_the_note():
    @click.command()
    def command():
        progress = view.CheckProgress(diagnostics=True)
        progress.update(
            "level:on",
            {
                "outcome": "accepted",
                "reasoning_observed": True,
                "reasoning_encrypted": False,
                "reasoning_tokens": 0,
                "output_tokens": 2329,
                "answer_correct": True,
            },
        )
        progress.finish()

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    normalized_output = " ".join(result.output.split())
    assert "2,329 output tokens (reasoning tokens not reported separately)" in normalized_output
    assert "reasoning chars" not in normalized_output
    assert "Reasoning tokens · on: 2,329 output" in normalized_output
    assert "chars" not in normalized_output.split("Reasoning tokens ·")[1].split("Results")[0]


def test_unsplit_reasoning_tokens_falls_back_to_output_tokens():
    @click.command()
    def command():
        progress = view.CheckProgress(diagnostics=True)
        progress.update(
            "level:high",
            {
                "outcome": "accepted",
                "reasoning_observed": True,
                "reasoning_tokens": 0,
                "output_tokens": 300,
                "answer_correct": True,
            },
        )
        progress.finish()

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    normalized_output = " ".join(result.output.split())
    assert "300 output tokens (reasoning tokens not reported separately)" in normalized_output
    assert "0 reasoning tokens" not in normalized_output
    assert "Reasoning tokens · high: 300 output" in normalized_output


def test_no_reasoning_signal_at_all_shows_no_token_suffix():
    @click.command()
    def command():
        progress = view.CheckProgress()
        progress.update(
            "level:low",
            {
                "outcome": "accepted",
                "reasoning_observed": False,
                "reasoning_tokens": 0,
                "output_tokens": 20,
                "answer_correct": True,
            },
        )
        progress.finish()

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    assert "output tokens" not in result.output
    assert "reasoning tokens" not in result.output
    assert "Reasoning tokens ·" not in result.output


def test_reasoning_observed_without_usage_still_appears_in_the_summary():
    """A level whose response carried no usage object (so output_tokens
    isn't an int) must not silently vanish from the end-of-run summary just
    because no token count could be computed for it — that defeats the
    point of a complete cross-level comparison. It reads "reasoning
    observed" rather than a misleading "0", since reasoning genuinely did
    happen here; only levels where it never happened stay out entirely.
    """

    @click.command()
    def command():
        progress = view.CheckProgress(diagnostics=True)
        progress.update(
            "level:low",
            {
                "outcome": "accepted",
                "reasoning_observed": True,
                "answer_correct": True,
            },
        )
        progress.finish()

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    assert "Reasoning tokens · low: reasoning observed" in result.output


def test_success_hides_test_cap_but_length_retry_explains_increase():
    @click.command()
    def command():
        progress = view.CheckProgress()
        progress.update(
            "session:seed",
            {
                "outcome": "accepted",
                "reasoning_observed": True,
                "tested_reply_tokens": 2048,
                "configured_reply_tokens": 65536,
                "reply_limit_reduced_for_check": True,
            },
        )
        progress.update(
            "session:replay",
            {
                "outcome": "retrying",
                "tested_reply_tokens": 4096,
            },
        )
        progress.update("session:replay", {"outcome": "accepted"})
        progress.finish()

    result = CliRunner().invoke(command)
    assert result.exit_code == 0
    assert "2,048" not in result.output
    assert "65,536" not in result.output
    text = " ".join(result.output.split())
    assert "retrying with 4,096 tokens" in text
    assert "saved setting unchanged" in text
    assert "2 passed · 0 not available · 0 failed · 0 skipped" in result.output


def test_format_budget_labels_the_sentinel_as_unlimited():
    from nooa.unifiedllm.connect import DEFAULT_CHECK_BUDGET

    assert view.format_budget(DEFAULT_CHECK_BUDGET) == "unlimited"


def test_format_budget_labels_sentinel_minus_realistic_spend_as_unlimited():
    """The default budget's "remaining" value is the sentinel minus whatever
    a run spent so far (observed up to ~1.2M tokens for a full run), never
    exactly equal to it — that must still read as unlimited.
    """
    from nooa.unifiedllm.connect import DEFAULT_CHECK_BUDGET

    assert view.format_budget(DEFAULT_CHECK_BUDGET - 5_000_000) == "unlimited"


def test_format_budget_does_not_mislabel_a_large_explicit_value():
    """--budget-tokens has no declared upper bound. A genuine, very large,
    explicitly-chosen budget must still display as its real number, not be
    conflated with the much larger unlimited sentinel.
    """
    assert view.format_budget(2_000_000_000_000) == "2,000,000,000,000"


def test_format_budget_shows_small_values_as_is():
    assert view.format_budget(4096) == "4,096"


@pytest.mark.parametrize(
    "outcome,label,status,reason",
    [
        ("passed", "Passed", "passed", "Model reported access to this input"),
        (
            "inconclusive",
            "Failed",
            "attention",
            "Service returned HTTP 400; image support was not confirmed",
        ),
        ("rejected", "Failed", "attention", "Provider explicitly rejected this media request"),
        ("not_probed", "Skipped", "skipped", "budget exhausted"),
        (
            "inconclusive",
            "Failed",
            "attention",
            "Timed out; server receipt and completion are unknown; image support was not confirmed",
        ),
    ],
)
def test_media_labels_keep_classification_and_safe_reasons(outcome, label, status, reason):
    from nooa.unifiedllm.connect import check_status

    record = {"outcome": outcome, "reason": reason}
    before = record.copy()

    @click.command()
    def command():
        progress = view.CheckProgress()
        progress.update("media:image", record)
        progress.finish()

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    expected = {
        "passed": "Passed",
        "rejected": "Failed · Server rejected the media request; image support not confirmed.",
        "not_probed": "Skipped · token budget exhausted",
    }.get(
        outcome,
        "Failed · "
        + (
            "Request timed out; completion is unknown"
            if reason.startswith("Timed out")
            else "Service returned HTTP 400"
        )
        + "; image support not confirmed.",
    )
    assert f"Image input: {expected}" in " ".join(result.output.split())
    assert record == before
    assert check_status("media:image", record) == status


@pytest.mark.parametrize(
    "enabled,setting",
    [
        ([], "image support disabled"),
        (["image"], "image input remains enabled (not confirmed by this test)"),
    ],
)
def test_failed_media_label_setting_is_display_only(enabled, setting):
    record = {
        "outcome": "inconclusive",
        "reason": "Service returned HTTP 400; image support was not confirmed",
    }
    before = record.copy()
    assert view.media_test_detail("image", record, media_inputs=enabled) == (
        f"Failed · Service returned HTTP 400; {setting}."
    )
    assert record == before


@pytest.mark.parametrize("correct", [True, False])
def test_typed_baseline_progress_uses_answer_policy_not_connected_label(correct):
    @click.command()
    def command():
        progress = view.CheckProgress()
        progress.update(
            "routing",
            {
                "outcome": "accepted",
                "baseline_kind": "predict_int_v1",
                "typed_result_validated": True,
                "answer_correct": correct,
                "final_answer_observed": True,
                "settings_sent": True,
                "finish_reason": "stop",
            },
        )
        progress.finish()

    result = CliRunner().invoke(command, color=False)
    assert result.exit_code == 0, result.output
    assert "Connected" not in result.output
    if correct:
        assert "Connection: Passed" in result.output
        assert "integer" not in result.output
    else:
        assert "Connection: Failed" in result.output
        assert "Answer did not match the test question" in " ".join(result.output.split())
        assert "✓" not in result.output


@pytest.mark.parametrize("kind", ["image", "audio", "video", "pdf"])
@pytest.mark.parametrize("enabled", [False, True])
def test_false_access_is_clear_and_keeps_effective_permission(kind, enabled):
    record = {
        "outcome": "unavailable",
        "finish_reason": "stop",
        "access_reported": False,
        "typed_result_validated": True,
        "reason": "Model cannot access this input",
    }
    original = record.copy()

    @click.command()
    def command():
        view.CheckProgress().update("media:" + kind, record, media_inputs=[kind] if enabled else [])

    result = CliRunner().invoke(command)
    rendered = " ".join(result.output.split())
    assert result.exit_code == 0, result.output
    assert f"{view.check_label('media:' + kind)}: Not available" in rendered
    assert "Model cannot access this input" in rendered
    assert "Wrong answer" not in rendered and "✓" not in rendered
    assert (
        f"{kind} input remains enabled (manual choice)" if enabled else f"{kind} input disabled"
    ) in rendered
    assert "Request failed" not in rendered and "unsupported" not in rendered
    assert record == original


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({"finish_reason": "length"}, "Reply reached the token limit before finishing"),
        ({"finish_reason": "content_filter"}, "Reply failed or was filtered"),
        ({"final_answer_observed": False}, "No final answer returned"),
        ({"typed_result_validated": False}, "Answer was not in the required format"),
        ({"settings_sent": False}, "selected settings were not observed"),
        (
            {"outcome": "not_probed", "reason": "budget exhausted"},
            "Skipped · token budget exhausted",
        ),
        (
            {"outcome": "rejected", "error": "AuthenticationError", "status_code": 401},
            "Key rejected by this server",
        ),
    ],
)
def test_connection_output_preserves_material_failures(changes, expected):
    record = {
        "outcome": "accepted",
        "baseline_kind": "predict_int_v1",
        "typed_result_validated": True,
        "answer_correct": True,
        "final_answer_observed": True,
        "settings_sent": True,
        "finish_reason": "stop",
        **changes,
    }
    original = record.copy()

    @click.command()
    def command():
        view.CheckProgress().update("routing", record)

    result = CliRunner().invoke(command)
    output = " ".join(result.output.split())
    assert expected in output
    assert "Connection: Passed" not in output
    assert "integer" not in output and "baseline" not in output
    assert record == original


@pytest.mark.parametrize("kind", ["image", "audio", "video", "pdf"])
def test_media_request_errors_are_not_mismatched_answers(kind):
    @click.command()
    def command():
        progress = view.CheckProgress()
        progress.update(
            "media:" + kind,
            {
                "outcome": "inconclusive",
                "error": "BadRequestError",
                "status_code": 400,
                "reason": f"Service returned HTTP 400; {kind} support was not confirmed",
            },
            media_inputs=[],
        )

    result = CliRunner().invoke(command)
    output = " ".join(result.output.split())
    assert "Failed · Service returned HTTP 400" in output
    assert f"{kind} support disabled" in output
    assert "answer did not match" not in output


def test_intro_has_one_plain_spend_warning_and_visible_call_caps():
    @click.command()
    def command():
        view.intro(
            checks=True, output_tokens=200, reasoning_output_tokens=4096, budget_tokens=100000
        )

    result = CliRunner().invoke(command)
    output = " ".join(result.output.split())
    assert output.count("Checks may incur charges.") == 1
    assert "The token budget is an estimate, not a spending cap." in output
    assert "Shared token budget: 100,000" in output
    assert "200 output tokens per call" in output
    assert "4,096 output tokens per call" in output
    for jargon in ("baseline", "native", "wire", "capacity probes", "billing limits"):
        assert jargon not in output


@pytest.mark.parametrize("kind", ["audio", "video"])
def test_local_media_skip_explains_api_format_without_claiming_unsupported(kind):
    from nooa.unifiedllm.connect._media import media_content

    _, _, reason = media_content(kind, "responses")
    detail = view.media_test_detail(kind, {"outcome": "not_probed", "reason": reason})
    assert "Skipped" in detail
    assert f"{kind.capitalize()} samples cannot be tested with this API format" in detail
    assert "Manual selection is unverified" in detail
    assert "native" not in detail and "unsupported" not in detail


@pytest.mark.parametrize("columns", [24, 40, 80, 120, 240])
@pytest.mark.parametrize("color", [False, True, "no_color"])
@pytest.mark.parametrize(
    "name,record,permission",
    [
        (
            "media:image",
            {
                "outcome": "inconclusive",
                "reason": "Service returned HTTP 400; image support was not confirmed",
            },
            [],
        ),
        ("media:image", {"outcome": "unavailable"}, ["image"]),
        ("media:image", {"outcome": "passed"}, []),
        ("media:audio", {"outcome": "not_probed", "reason": "budget exhausted"}, []),
        ("tools", {"outcome": "accepted", "tool_observed": False}, []),
        (
            "level:high",
            {
                "outcome": "accepted",
                "reasoning_observed": True,
                "answer_correct": True,
                "reasoning_tokens": 800,
            },
            [],
        ),
        ("cache", {"outcome": "confirmed", "input_tokens": 10000, "cached_input_tokens": 9000}, []),
        ("routing", {"outcome": "rejected", "status_code": 401}, []),
    ],
)
def test_result_duration_is_with_outcome_and_details_are_lossless(
    monkeypatch, columns, color, name, record, permission
):
    monkeypatch.setattr(
        view.shutil, "get_terminal_size", lambda *args, **kwargs: os.terminal_size((columns, 24))
    )
    monkeypatch.setattr(view.time, "monotonic", lambda: 1.7)
    if color == "no_color":
        monkeypatch.setenv("NO_COLOR", "1")
    else:
        monkeypatch.delenv("NO_COLOR", raising=False)

    @click.command()
    def command():
        progress = view.CheckProgress()
        progress.started[name] = 1.0
        progress.update(name, record, media_inputs=permission)

    result = CliRunner().invoke(command, color=bool(color))
    assert result.exit_code == 0, result.output
    plain = click.unstyle(result.output)
    rows = plain.splitlines()
    timed = next(row for row in rows if "0.7s" in row)
    assert len(timed.split(" · 0.7s")[0].strip()) > 1
    assert all(len(row) <= columns for row in rows)
    if color == "no_color" or not color:
        assert "\x1b[" not in result.output
    if name.startswith("media:"):
        expected = " ".join(
            (
                "Passed"
                if record["outcome"] == "passed"
                else view.media_test_detail(name[6:], record, media_inputs=permission)
            ).split()
        )
        rendered = " ".join(plain.replace(" · 0.7s", "").split())
        assert expected in rendered
    if len(rows) > 1:
        assert all(row.startswith("    ") for row in rows[1:])


def test_general_line_retains_its_existing_width_cap(monkeypatch):
    monkeypatch.setattr(
        view.shutil, "get_terminal_size", lambda *args, **kwargs: os.terminal_size((240, 24))
    )

    @click.command()
    def command():
        view.line("word " * 40)

    result = CliRunner().invoke(command)
    assert result.exit_code == 0
    assert len(result.output.splitlines()) > 1
    assert max(map(len, result.output.splitlines())) <= 86
