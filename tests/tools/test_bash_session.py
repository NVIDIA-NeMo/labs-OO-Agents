# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for persistent bash session."""

import pytest

from nooa.tools._bash_session import BashSession


@pytest.fixture
async def session(tmp_path):
    """Create a bash session in a temp directory."""
    s = BashSession(cwd=tmp_path)
    await s.start()
    yield s
    await s.close()


class TestBashSession:
    async def test_simple_command(self, session):
        out, err, code = await session.run("echo hello")
        assert code == 0
        assert "hello" in out

    async def test_exit_code(self, session):
        out, err, code = await session.run("false")
        assert code != 0

    async def test_stderr(self, session):
        out, err, code = await session.run("echo oops 1>&2")
        assert "oops" in err

    async def test_cd_persists(self, session, tmp_path):
        """cd in one command should persist to the next."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        await session.run(f"cd {subdir}")
        out, _, _ = await session.run("pwd")
        assert str(subdir) in out

    async def test_cwd_tracking(self, session, tmp_path):
        """Session.cwd should update after cd."""
        subdir = tmp_path / "deep" / "nested"
        subdir.mkdir(parents=True)

        await session.run(f"cd {subdir}")
        assert session.cwd == subdir

    async def test_env_persists(self, session):
        """Environment variables should persist."""
        await session.run("export MY_VAR=hello123")
        out, _, _ = await session.run("echo $MY_VAR")
        assert "hello123" in out

    async def test_multiline_output(self, session):
        out, _, code = await session.run("echo line1; echo line2; echo line3")
        assert code == 0
        assert "line1" in out
        assert "line2" in out
        assert "line3" in out

    async def test_start_idempotent(self, session):
        """Calling start() twice should be safe."""
        await session.start()  # already started by fixture
        out, _, _ = await session.run("echo still_works")
        assert "still_works" in out

    async def test_output_truncation(self, session):
        """Very long output should be truncated."""
        out, _, _ = await session.run("python3 -c \"print('x' * 50000)\"")
        assert len(out) <= 31000  # MAX_OUTPUT_CHARS + truncation message

    async def test_truncation_keeps_the_head_and_the_tail(self, session):
        """A long output keeps its end too: a failing command usually ends with its error."""
        script = "import sys; print('H' * 20000 + 'M' * 20000 + 'T' * 20000); "
        script += "sys.stderr.write('h' * 20000 + 'm' * 20000 + 't' * 20000)"
        out, err, code = await session.run(f'python3 -c "{script}"')
        assert code == 0
        for text, head, middle, tail in ((out, "H", "M", "T"), (err, "h", "m", "t")):
            # nooa.snapshot's render_output: 15,000 head chars, one cut line, 15,000 tail chars.
            whole = head * 20000 + middle * 20000 + tail * 20000 + ("\n" if head == "H" else "")
            expected = (
                whole[:15_000]
                + f"\n[... {len(whole) - 30_000:,} chars omitted; full output not kept]\n"
                + whole[-15_000:]
            )
            assert text == expected.removesuffix("\n")  # run() strips stdout's last newline

    def test_bounded_matches_one_write_and_keeps_both_ends(self):
        """Chunked feeding gives the same text as one write, for every size around the limit."""
        from nooa.snapshot import TextStream, render_output
        from nooa.tools._bash_session import MAX_OUTPUT_CHARS, _bounded

        assert MAX_OUTPUT_CHARS == 30_000  # the limit is unchanged
        for size in (MAX_OUTPUT_CHARS, MAX_OUTPUT_CHARS + 1, 65_536, 65_537, 300_000):
            text = "".join(chr(ord("a") + i % 26) for i in range(size))
            reference = TextStream(unit="chars", head=15_000, tail=15_000)
            reference.write(text)
            assert _bounded(text) == render_output(reference.close())
            if size > MAX_OUTPUT_CHARS:
                assert _bounded(text) == (
                    text[:15_000]
                    + f"\n[... {size - 30_000:,} chars omitted; full output not kept]\n"
                    + text[-15_000:]
                )
            else:
                assert _bounded(text) == text

    async def test_close_and_restart(self, tmp_path):
        """Session should be closeable and re-startable."""
        s = BashSession(cwd=tmp_path)
        await s.start()
        out1, _, _ = await s.run("echo first")
        assert "first" in out1
        await s.close()

        # Start fresh
        await s.start()
        out2, _, _ = await s.run("echo second")
        assert "second" in out2
        await s.close()

    async def test_pipe_commands(self, session):
        out, _, code = await session.run("echo -e 'a\\nb\\nc' | sort -r")
        assert code == 0

    async def test_command_with_quotes(self, session):
        out, _, code = await session.run("echo 'hello world'")
        assert code == 0
        assert "hello world" in out

    async def test_file_operations(self, session, tmp_path):
        """Write and read a file through the session."""
        await session.run(f"echo 'test content' > {tmp_path}/test.txt")
        out, _, _ = await session.run(f"cat {tmp_path}/test.txt")
        assert "test content" in out


class TestPublicStateHelpers:
    """``is_busy()`` and ``snapshot_cwd_env()``: public, non-blocking state reads."""

    async def test_not_started_is_not_busy_and_has_no_snapshot(self, tmp_path):
        s = BashSession(cwd=tmp_path)
        try:
            assert s.is_busy() is False
            assert await s.snapshot_cwd_env() is None  # does not start bash
        finally:
            await s.close()

    async def test_snapshot_reads_cwd_and_exports(self, session, tmp_path):
        (tmp_path / "sub").mkdir()
        await session.run("cd sub; export SNAP_VAR='a b=c'; NOT_EXPORTED=1")
        snap = await session.snapshot_cwd_env()
        assert snap is not None
        cwd, env = snap
        assert cwd == str(tmp_path / "sub")
        assert env["SNAP_VAR"] == "a b=c" and "NOT_EXPORTED" not in env
        # The capture changes nothing in the session.
        out, _, _ = await session.run('pwd; echo "$SNAP_VAR"')
        assert out.splitlines() == [str(tmp_path / "sub"), "a b=c"]

    async def test_snapshot_reads_a_large_environment_whole(self, session, tmp_path):
        # Over 40 KB of env: its base64 is far past MAX_OUTPUT_CHARS, which a
        # capped read would cut (and then decode to garbage or fail).
        (tmp_path / "sub").mkdir()
        await session.run('cd sub; export BIG_VAR="$(head -c 45000 /dev/zero | tr "\\0" x)"')
        snap = await session.snapshot_cwd_env()
        assert snap is not None
        cwd, env = snap
        assert cwd == str(tmp_path / "sub")
        assert env["BIG_VAR"] == "x" * 45000
        assert "PATH" in env

    async def test_snapshot_is_none_when_the_capture_is_corrupt(self, session, monkeypatch):
        async def cut(command, timeout, *, bounded=True):
            # A cut blob that a lenient decode still accepts (non-base64 chars
            # are dropped) and turns into garbage after "PATH=/bin".
            return "/somewhere\nUEFUSD0vYmlu<truncated-output-x>", "", 0, False

        monkeypatch.setattr(session, "_run_unlocked", cut)
        assert await session.snapshot_cwd_env() is None  # never a wrong environment

    async def test_busy_while_a_command_runs_and_no_snapshot_then(self, session):
        import asyncio

        assert session.is_busy() is False
        running = asyncio.create_task(session.run("sleep 0.5"))
        for _ in range(100):
            if session.is_busy():
                break
            await asyncio.sleep(0.01)
        assert session.is_busy() is True
        assert await session.snapshot_cwd_env() is None  # returns at once, does not wait
        assert not running.done()
        await running
        assert session.is_busy() is False
        assert await session.snapshot_cwd_env() is not None
