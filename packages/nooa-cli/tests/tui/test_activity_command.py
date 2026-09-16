# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the /activity slash command (ActivityCommand)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from nooa_cli.tui.commands import ActivityCommand
from nooa_cli.tui.output import TableOutput, TextOutput


def _reset_activity_state():
    import nooa.runtime.debug_handler as dh

    dh._pending_llm_calls.clear()
    dh._llm_call_counter = 0
    dh._pending_code_execs.clear()
    dh._code_exec_counter = 0


@pytest.fixture
def command():
    _reset_activity_state()
    cmd = ActivityCommand(frontend=AsyncMock(), config=MagicMock(), agent=MagicMock())
    yield cmd
    _reset_activity_state()


@pytest.mark.asyncio
async def test_idle(command):
    """With nothing in flight, /activity is a single status one-liner."""
    result = await command.execute([])
    assert result.success
    assert len(result.outputs) == 1
    out = result.outputs[0]
    assert isinstance(out, TextOutput)
    assert not isinstance(out, TableOutput)
    assert "idle" in out.content.lower()


@pytest.mark.asyncio
async def test_executing_python(command):
    """A running code cell reports Executing Python with a preview row."""
    from nooa.runtime.debug_handler import code_exec_context

    with code_exec_context("x = 1\nprint(x)"):
        result = await command.execute([])
    table = result.outputs[0]
    assert table.rows[0] == ["Phase", "Executing Python"]
    # second row carries the cell preview
    assert table.rows[1][1] == "x = 1"
    assert table.rows[1][0].startswith("  python (")
    assert table.footer == "In a code cell — not waiting on the model."


@pytest.mark.asyncio
async def test_preview_fallback_for_blank_cell(command):
    """A blank/whitespace cell falls back to the "(code cell)" preview label."""
    from nooa.runtime.debug_handler import code_exec_context

    with code_exec_context(""):
        result = await command.execute([])
    table = result.outputs[0]
    assert table.rows[1][1] == "(code cell)"


@pytest.mark.asyncio
async def test_waiting_llm(command):
    """An in-flight LLM call reports Waiting on LLM call with the model row."""
    from nooa.runtime.debug_handler import llm_call_context

    with llm_call_context(model="gpt-test"):
        result = await command.execute([])
    table = result.outputs[0]
    assert table.rows[0] == ["Phase", "Waiting on LLM call"]
    assert table.rows[1][1] == "gpt-test"
    assert table.rows[1][0].startswith("  llm (")
    assert table.footer == "Blocked waiting for the model to respond."


@pytest.mark.asyncio
async def test_llm_model_unknown_fallback(command):
    """An in-flight LLM call missing its ``model`` key renders as 'unknown'."""
    import nooa.runtime.debug_handler as dh
    from nooa.runtime.debug_handler import register_llm_call

    call_id = register_llm_call(model="x")
    with dh._llm_call_lock:
        dh._pending_llm_calls[call_id].pop("model")
    result = await command.execute([])
    table = result.outputs[0]
    assert table.rows[1][1] == "unknown"


@pytest.mark.asyncio
async def test_python_and_llm_nested(command):
    """A cell blocked on an LLM call: phase is Executing Python, both rows shown."""
    from nooa.runtime.debug_handler import code_exec_context, llm_call_context

    with code_exec_context("y = 2"):
        with llm_call_context(model="gpt-test"):
            result = await command.execute([])
    table = result.outputs[0]
    assert table.rows[0] == ["Phase", "Executing Python"]
    # both a python row and an llm row are present
    kinds = [r[0].strip().split()[0] for r in table.rows[1:]]
    assert "python" in kinds and "llm" in kinds
    assert table.footer == "Running a code cell that is itself blocked on an LLM call."


@pytest.mark.asyncio
async def test_import_error_fallback(command, monkeypatch):
    """If debug_handler can't be imported, /activity returns a graceful error."""
    import builtins as _builtins

    real_import = _builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "nooa.runtime.debug_handler":
            raise ImportError("boom")
        return real_import(name, *args, **kwargs)

    # Scoped to just the call under test: the `command` fixture's teardown
    # imports nooa.runtime.debug_handler to reset module state, so leaving
    # __import__ patched past this block makes teardown explode. Whether that
    # happened used to depend on the order pytest instantiated `command` vs
    # `monkeypatch` — an autouse fixture requesting monkeypatch was enough to
    # flip it. A context manager makes the test independent of that ordering.
    with monkeypatch.context() as patched:
        patched.setattr(_builtins, "__import__", _fake_import)
        result = await command.execute([])

    assert not result.success
    assert isinstance(result.outputs[0], TextOutput)


@pytest.mark.asyncio
async def test_activity_command_opens_overlay_when_frontend_supports_it():
    """Terminal-like frontends show /activity in the full-screen overlay."""

    class OverlayFrontend:
        def __init__(self) -> None:
            self.outputs = None

        async def open_activity_overlay(self, outputs):
            self.outputs = outputs

    _reset_activity_state()
    frontend = OverlayFrontend()
    cmd = ActivityCommand(frontend=frontend, config=MagicMock(), agent=MagicMock())

    result = await cmd.execute([])

    assert result.success
    assert isinstance(result.outputs[0], TextOutput)
    assert "closed" in result.outputs[0].content
    assert frontend.outputs is not None
    assert isinstance(frontend.outputs[0], TextOutput)
    assert "idle" in frontend.outputs[0].content.lower()
