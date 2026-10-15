# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failure diagnostics must survive CodeAct context rolling between requests."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from nooa.agentdoc import pformat
from nooa.context_blocks.events import ResultStatus
from nooa.events import PythonOutput

from . import _native_harness as harness


async def test_failure_note_keeps_first_and_recent_errors_after_context_rolls(
    tmp_path, monkeypatch
):
    process = SimpleNamespace(
        stdin=SimpleNamespace(close=Mock(), wait_closed=AsyncMock()),
        stdout=asyncio.StreamReader(),
        stderr=SimpleNamespace(read=AsyncMock(return_value=b"")),
        wait=AsyncMock(return_value=0),
        returncode=0,
    )
    connection = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(harness, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(harness, "subprocess", SimpleNamespace(CREATE_NO_WINDOW=0))
    monkeypatch.setattr(harness.shutil, "which", lambda name: name)
    monkeypatch.setattr(harness, "_child_environment", lambda *args: {})
    monkeypatch.setattr(harness.asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    monkeypatch.setattr(harness, "connect_to_agent", lambda *args, **kwargs: connection)

    def output(index: int, error: str, stdout: str = "") -> dict[str, str]:
        event = PythonOutput(
            tool_call_id=f"call-{index}",
            execution_count=index,
            execution_status=ResultStatus.ERROR,
            stdout=stdout,
            error=error,
        )
        return {"role": "user", "content": f'<sys tag="{index}">\n{pformat(event)}\n</sys>'}

    first = output(1, "TimeoutError: ")
    middle = output(2, "Traceback (most recent call last):\nNameError: middle_failure")
    recent = output(3, "Traceback (most recent call last):\nNameError: recent_failure")
    last = output(
        4,
        "Traceback (most recent call last):\nNameError: final_failure",
        "x" * 10_000 + "OMIT_MIDDLE_OUTPUT" + "x" * 10_000,
    )
    system = {"role": "system", "content": "PRIVATE_SYSTEM_PROMPT\n" + first["content"]}
    task = {"role": "user", "content": "PRIVATE_TASK_PROMPT"}
    failure = AssertionError("The native workflow did not complete")
    with pytest.raises(AssertionError) as raised:
        async with harness.launch_native_acp(object(), tmp_path / "workspace", []) as (_, _, model):
            model.requests.extend(
                [
                    {"messages": [system, task, first]},
                    {"messages": [system, task, first, middle]},
                    {"messages": [system, task, middle, recent]},
                    {"messages": [system, task, recent, last]},
                ]
            )
            raise failure

    assert raised.value is failure
    note = "\n".join(failure.__notes__)
    assert note.count("TimeoutError:") == 1
    assert note.count("recent_failure") == 1
    assert note.count("final_failure") == 1
    assert "middle_failure" not in note
    assert "OMIT_MIDDLE_OUTPUT" not in note
    assert "PRIVATE_SYSTEM_PROMPT" not in note
    assert "PRIVATE_TASK_PROMPT" not in note
    assert len(note) < 12_500
    assert process.stdin.close.call_count == 1
    process.wait.assert_awaited_once()
    connection.close.assert_awaited_once()
