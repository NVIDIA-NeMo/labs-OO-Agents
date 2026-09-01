# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for how BashSession frames a command on the wire.

The control protocol (exit code, cwd, sentinel) travels on the same stdin as
the command, so the command must not be parsed as shell text by the same
parser that has to reach those lines. These tests pin the properties that
depend on it:

1. A command bash cannot parse exits non-zero instead of hanging
2. A command that reads stdin cannot consume the protocol
3. cd / export still persist across commands
4. Command length is not bounded by ARG_MAX
"""

import asyncio

import pytest

from nooa.tools._bash_session import BashSession

# Unbalanced double quote. Parsed as shell text this swallows every following
# line, including the protocol, so the sentinel never arrives.
UNPARSEABLE = """grep -n "needle\\(/tmp --glob '!build' """


@pytest.fixture
async def session(tmp_path):
    """Fresh BashSession for each test."""
    s = BashSession(cwd=tmp_path)
    await s.start()
    yield s
    await s.close()


class TestUnparseableCommands:
    """A command bash cannot parse is an error, not a hang."""

    async def test_unbalanced_quote_returns_promptly(self, session):
        stdout, stderr, code, timed_out = await asyncio.wait_for(
            session.run_with_timeout_flag(UNPARSEABLE, timeout=10.0), timeout=20.0
        )
        assert not timed_out, "unparseable command hung instead of returning"
        assert code != 0

    async def test_unbalanced_quote_reports_the_syntax_error(self, session):
        _stdout, stderr, _code, _timed = await session.run_with_timeout_flag(
            UNPARSEABLE, timeout=10.0
        )
        # The caller is an agent that can correct itself only if it is told why.
        assert "unexpected EOF" in stderr or "syntax error" in stderr

    async def test_session_survives_an_unparseable_command(self, session):
        await session.run(UNPARSEABLE, timeout=10.0)
        stdout, _stderr, code = await session.run("echo alive")
        assert stdout == "alive"
        assert code == 0

    async def test_unparseable_command_does_not_restart_the_session(self, session):
        """A syntax error is handled in-shell, so no reset is needed."""
        before = session._start_count
        await session.run(UNPARSEABLE, timeout=10.0)
        assert session._start_count == before

    @pytest.mark.parametrize(
        "command",
        [
            'echo "unterminated',
            "echo 'unterminated",
            "for i in 1 2 3; do echo $i",
            "cat <<EOF\nno terminator",
            "echo $(",
        ],
    )
    async def test_malformed_shapes_all_return(self, session, command):
        _stdout, _stderr, _code, timed_out = await asyncio.wait_for(
            session.run_with_timeout_flag(command, timeout=10.0), timeout=20.0
        )
        assert not timed_out


class TestProtocolIsolation:
    """A command that reads stdin cannot consume the control protocol."""

    async def test_cat_does_not_consume_the_protocol(self, session):
        stdout, _stderr, code, timed_out = await session.run_with_timeout_flag("cat", timeout=10.0)
        assert not timed_out
        assert code == 0
        assert stdout == ""

    async def test_read_builtin_gets_eof(self, session):
        stdout, _stderr, code, timed_out = await session.run_with_timeout_flag(
            "read line; echo got=$line", timeout=10.0
        )
        assert not timed_out
        assert code == 0
        assert stdout == "got="


class TestStatePersistence:
    """Framing must not cost the session its defining property."""

    async def test_cd_persists(self, session, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        await session.run(f"cd {sub}")
        stdout, _stderr, _code = await session.run("pwd")
        assert stdout.endswith("sub")
        assert session.cwd == sub

    async def test_exported_variables_persist(self, session):
        await session.run("export NOOA_TEST_VAR=persisted")
        stdout, _stderr, _code = await session.run("echo $NOOA_TEST_VAR")
        assert stdout == "persisted"

    async def test_shell_functions_persist(self, session):
        await session.run("noofn() { echo from_function; }")
        stdout, _stderr, _code = await session.run("noofn")
        assert stdout == "from_function"

    async def test_exit_code_is_reported(self, session):
        _stdout, _stderr, code = await session.run("(exit 42)")
        assert code == 42


class TestLongCommands:
    """Command length is bounded by memory, not by ARG_MAX."""

    async def test_200kb_command_runs(self, session):
        payload = "x" * 200_000
        stdout, _stderr, code = await session.run(f"echo {payload} | wc -c", timeout=30.0)
        assert code == 0
        assert stdout.strip() == str(len(payload) + 1)


class TestStreamingUsesTheSameFraming:
    """run_stream shares _build_script, so it gets the same guarantees."""

    async def test_stream_of_unparseable_command_terminates(self, session):
        events = []
        async for name, chunk in session.run_stream(UNPARSEABLE, timeout=10.0):
            events.append((name, chunk))
        kind, payload = events[-1]
        assert kind == "__done__"
        exit_code, timed_out_flag = payload.split(",")
        assert timed_out_flag == "0", "unparseable command timed out instead of returning"
        assert exit_code != "0"

    async def test_stream_yields_output_and_persists_cd(self, session, tmp_path):
        sub = tmp_path / "streamed"
        sub.mkdir()
        chunks = [
            c async for n, c in session.run_stream(f"cd {sub} && echo streamed", timeout=10.0)
        ]
        assert "streamed" in "".join(chunks)
        assert session.cwd == sub
