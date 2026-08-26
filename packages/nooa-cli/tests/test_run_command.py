# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI contract tests for ``nooa run``."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner
from nooa_cli.commands.run import command
from nooa_cli.headless import HeadlessResult


def _result(*, status: str = "done", session_id: str | None = "abc123") -> HeadlessResult:
    return HeadlessResult(
        session_id=session_id,
        status=status,  # type: ignore[arg-type]
        messages=["first", "second"],
        explanation="complete",
    )


def _invoke(args: list[str], *, input: str | None = None, result=None):
    mocked = AsyncMock(return_value=result or _result())
    with (
        patch("nooa.llm_config.llm_config_chain", return_value=[]),
        patch("nooa.secrets.load_secrets_into_env"),
        patch("nooa.unifiedllm.reload_registry"),
        patch("nooa_cli.headless.run_headless", mocked),
        patch("nooa_cli.tui.config.Config.load", return_value=SimpleNamespace()),
    ):
        invocation = CliRunner().invoke(command, args, input=input)
    return invocation, mocked


def test_run_help_documents_headless_contract() -> None:
    result = CliRunner().invoke(command, ["--help"])

    assert result.exit_code == 0
    assert "--format [text|json|jsonl]" in result.output
    assert "--continue" in result.output
    assert "--resume ID_OR_PREFIX" in result.output
    assert "--ephemeral" in result.output


def test_run_joins_positional_prompt_and_piped_context() -> None:
    result, mocked = _invoke(["fix", "the", "tests"], input="extra context\n")

    assert result.exit_code == 0, result.output
    assert result.stdout == "first\n\nsecond\n"
    assert result.stderr == "Session: abc123\n"
    assert mocked.await_args.args[0] == "fix the tests\nextra context"


def test_run_reads_stdin_with_explicit_dash() -> None:
    result, mocked = _invoke(["-", "--ephemeral"], input="from stdin\n")

    assert result.exit_code == 0, result.output
    assert mocked.await_args.args[0] == "from stdin"
    assert mocked.await_args.kwargs["ephemeral"] is True


def test_run_rejects_empty_input() -> None:
    result = CliRunner().invoke(command, [], input="")

    assert result.exit_code == 2
    assert "Provide PROMPT or pipe a prompt" in result.stderr


def test_run_json_is_one_clean_document() -> None:
    result, _mocked = _invoke(["--format", "json", "task"])

    assert result.exit_code == 0
    assert result.stderr == "Session: abc123\n"
    payload = json.loads(result.stdout)
    assert payload == {
        "schema_version": 1,
        "session_id": "abc123",
        "status": "done",
        "messages": ["first", "second"],
        "explanation": "complete",
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
            "cost_usd": 0.0,
        },
        "error": None,
    }


def test_run_blocked_uses_exit_three_and_keeps_question() -> None:
    result, _mocked = _invoke(["task"], result=_result(status="blocked"))

    assert result.exit_code == 3
    assert result.stdout == "first\n\nsecond\n"


def test_run_runtime_failure_is_machine_readable() -> None:
    mocked = AsyncMock(side_effect=RuntimeError("provider unavailable"))
    with (
        patch("nooa.llm_config.llm_config_chain", return_value=[]),
        patch("nooa.secrets.load_secrets_into_env"),
        patch("nooa.unifiedllm.reload_registry"),
        patch("nooa_cli.headless.run_headless", mocked),
        patch("nooa_cli.tui.config.Config.load", return_value=SimpleNamespace()),
    ):
        result = CliRunner().invoke(command, ["--format", "json", "task"])

    assert result.exit_code == 1
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["status"] == "failed"
    assert payload["error"] == {"type": "RuntimeError", "message": "provider unavailable"}


def test_run_rejects_conflicting_session_options() -> None:
    result = CliRunner().invoke(command, ["--ephemeral", "--continue", "task"])

    assert result.exit_code == 2
    assert "cannot be combined" in result.stderr


def test_run_jsonl_streams_events_without_plain_text() -> None:
    async def fake_run(_text, **kwargs):
        kwargs["on_event"]({"schema_version": 1, "type": "session.started"})
        kwargs["on_event"]({"schema_version": 1, "type": "turn.completed"})
        return _result()

    with (
        patch("nooa.llm_config.llm_config_chain", return_value=[]),
        patch("nooa.secrets.load_secrets_into_env"),
        patch("nooa.unifiedllm.reload_registry"),
        patch("nooa_cli.headless.run_headless", fake_run),
        patch("nooa_cli.tui.config.Config.load", return_value=SimpleNamespace()),
    ):
        result = CliRunner().invoke(command, ["--format", "jsonl", "task"])

    assert result.exit_code == 0, result.output
    assert [json.loads(line)["type"] for line in result.stdout.splitlines()] == [
        "session.started",
        "turn.completed",
    ]


def test_run_writes_final_message_file(tmp_path) -> None:
    output = tmp_path / "answer.md"
    result, _mocked = _invoke(["--output", str(output), "task"])

    assert result.exit_code == 0, result.output
    assert output.read_text(encoding="utf-8") == "first\n\nsecond"


def test_run_structured_failure_exits_one() -> None:
    failed = HeadlessResult(
        session_id="abc123",
        status="failed",
        messages=[],
        explanation="tool exploded",
        error={"type": "RuntimeError", "message": "tool exploded"},
    )
    result, _mocked = _invoke(["--format", "json", "task"], result=failed)

    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"]["message"] == "tool exploded"


def test_run_text_failure_writes_diagnostic_to_stderr() -> None:
    failed = HeadlessResult(
        session_id="abc123",
        status="failed",
        messages=[],
        explanation="tool exploded",
        error={"type": "RuntimeError", "message": "tool exploded"},
    )
    result, _mocked = _invoke(["task"], result=failed)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Error: tool exploded" in result.stderr


def test_run_keyboard_interrupt_exits_130() -> None:
    mocked = AsyncMock(side_effect=KeyboardInterrupt)
    with (
        patch("nooa.llm_config.llm_config_chain", return_value=[]),
        patch("nooa.secrets.load_secrets_into_env"),
        patch("nooa.unifiedllm.reload_registry"),
        patch("nooa_cli.headless.run_headless", mocked),
        patch("nooa_cli.tui.config.Config.load", return_value=SimpleNamespace()),
    ):
        result = CliRunner().invoke(command, ["task"])

    assert result.exit_code == 130


def test_run_jsonl_keyboard_interrupt_emits_cancelled_terminal() -> None:
    mocked = AsyncMock(side_effect=KeyboardInterrupt)
    with (
        patch("nooa.llm_config.llm_config_chain", return_value=[]),
        patch("nooa.secrets.load_secrets_into_env"),
        patch("nooa.unifiedllm.reload_registry"),
        patch("nooa_cli.headless.run_headless", mocked),
        patch("nooa_cli.tui.config.Config.load", return_value=SimpleNamespace()),
    ):
        result = CliRunner().invoke(command, ["--format", "jsonl", "task"])

    assert result.exit_code == 130
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert [event["type"] for event in events] == ["turn.cancelled"]
