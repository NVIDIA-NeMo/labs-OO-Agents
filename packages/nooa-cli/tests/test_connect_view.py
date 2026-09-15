# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Readable terminal layout without changing the setup's cost or save policy."""

import os

import click
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
    assert "nooa-agent-authoring" in result.output


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
    assert "Connection/tool check limit" in result.output
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
    assert "2 passed · 2 need attention · 1 skipped" in result.output
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
    assert "Chat interface: Connected" in result.output


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
    assert "2 passed · 0 need attention · 0 skipped" in result.output
