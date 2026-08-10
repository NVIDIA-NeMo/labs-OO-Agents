# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Byte/character fidelity contracts for buffered and streamed shell output."""

from __future__ import annotations

import re
import secrets
import shlex
from pathlib import Path

import pytest

from nooa.tools._bash_session import MAX_OUTPUT_CHARS, BashSession


@pytest.fixture
async def session(tmp_path):
    shell = BashSession(cwd=tmp_path)
    await shell.start()
    try:
        yield shell
    finally:
        await shell.close()


def _streams(events: list[tuple[str, str]]) -> tuple[str, str]:
    stdout = "".join(value for name, value in events if name == "stdout")
    stderr = "".join(value for name, value in events if name == "stderr")
    return stdout, stderr


def _remove_artifact_from_notice(output: str) -> None:
    match = re.search(r"full untruncated output .* is in: (.+)\n", output)
    if match is not None:
        Path(match.group(1)).unlink(missing_ok=True)


async def test_buffered_truncation_exact_boundary(session):
    at_limit, stderr, code = await session.run(
        f"python3 -c \"import os; os.write(1, b'x' * {MAX_OUTPUT_CHARS})\""
    )
    over_limit, over_stderr, over_code = await session.run(
        f"python3 -c \"import os; os.write(1, b'y' * {MAX_OUTPUT_CHARS + 1})\""
    )
    try:
        assert (len(at_limit), stderr, code) == (MAX_OUTPUT_CHARS, "", 0)
        assert "<truncated-output>" not in at_limit
        assert (over_stderr, over_code) == ("", 0)
        assert "<truncated-output>" in over_limit
        assert f"Output too large ({MAX_OUTPUT_CHARS + 1:,} chars)" in over_limit
    finally:
        _remove_artifact_from_notice(over_limit)


async def test_buffered_invalid_utf8_uses_replacement_character(session):
    stdout, stderr, code = await session.run(
        "python3 -c \"import os; os.write(1,b'good\\xffend'); os.write(2,b'bad\\xfeend')\""
    )
    assert (stdout, stderr, code) == ("good�end", "bad�end", 0)


async def test_streaming_split_utf8_on_stderr(session):
    source = (
        "import os,time; os.write(2,b'prefix\\xe2'); "
        "time.sleep(.05); os.write(2,b'\\x82\\xac-suffix')"
    )
    events = [item async for item in session.run_stream(f"python3 -c {shlex.quote(source)}")]
    stdout, stderr = _streams(events)
    assert (stdout, stderr) == ("", "prefix€-suffix")
    assert events[-1] == ("__done__", "0,0")


async def test_streaming_invalid_utf8_uses_replacement_character(session):
    events = [
        item
        async for item in session.run_stream(
            "python3 -c \"import os; os.write(1,b'good\\xffend'); os.write(2,b'bad\\xfeend')\""
        )
    ]
    assert _streams(events) == ("good�end", "bad�end")
    assert events[-1] == ("__done__", "0,0")


async def test_streaming_patterned_stdout_and_stderr_are_lossless_and_ordered(session):
    count = 2_000
    source = (
        "import os\n"
        f"for i in range({count}):\n"
        " os.write(1, f'O{i:04d}|'.encode())\n"
        " os.write(2, f'E{i:04d}|'.encode())\n"
    )
    events = [
        item async for item in session.run_stream(f"python3 -c {shlex.quote(source)}", timeout=10)
    ]
    stdout, stderr = _streams(events)

    assert stdout == "".join(f"O{index:04d}|" for index in range(count))
    assert stderr == "".join(f"E{index:04d}|" for index in range(count))
    assert events[-1] == ("__done__", "0,0")


async def test_output_ending_in_marker_prefix_is_not_lost(session, monkeypatch):
    real_token_hex = secrets.token_hex

    def controlled_output_marker(size: int) -> str:
        return "marker" if size == 16 else real_token_hex(size)

    monkeypatch.setattr(secrets, "token_hex", controlled_output_marker)
    marker = "__NOOA_OUTPUT_marker__"
    prefix = marker[:17]
    payload = "x" * (4096 - len(prefix)) + prefix

    events = [
        item
        async for item in session.run_stream(
            f'python3 -c "import os; os.write(1, {payload.encode()!r})"'
        )
    ]

    assert _streams(events) == (payload, "")
    assert events[-1] == ("__done__", "0,0")
