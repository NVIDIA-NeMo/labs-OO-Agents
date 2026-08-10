# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for persistent bash session."""

import asyncio
import shlex

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

    async def test_buffered_output_preserves_utf8_split_across_reads(self, session):
        command = (
            'python3 -c "import os,time; '
            "os.write(1,b'\\xe2'); os.write(2,b'\\xe2'); time.sleep(.1); "
            "os.write(1,b'\\x82\\xac'); os.write(2,b'\\x82\\xac')\""
        )

        stdout, stderr, code = await session.run(command)

        assert (stdout, stderr, code) == ("€", "€", 0)

    async def test_buffered_truncation_keeps_head_and_tail_for_both_streams(self, session):
        command = (
            'python3 -c "import os; '
            "os.write(1, b'OUT_HEAD' + b'x' * 50000 + b'OUT_TAIL'); "
            "os.write(2, b'ERR_HEAD' + b'y' * 50000 + b'ERR_TAIL')\""
        )

        stdout, stderr, code = await session.run(command)

        assert code == 0
        assert "OUT_HEAD" in stdout and "OUT_TAIL" in stdout
        assert "ERR_HEAD" in stderr and "ERR_TAIL" in stderr
        assert "<truncated-output>" in stdout
        assert "<truncated-output>" in stderr

    async def test_streaming_output_is_not_truncated(self, session):
        chunks = [
            chunk
            async for stream, chunk in session.run_stream("python3 -c \"print('x' * 100000)\"")
            if stream == "stdout"
        ]
        output = "".join(chunks)
        assert output == "x" * 100_000 + "\n"
        assert "<truncated-output>" not in output

    async def test_streaming_preserves_utf8_split_across_read_chunks(self, session):
        command = "python3 -c \"import os; os.write(1, b'x' * 4095 + '€'.encode() + b'\\n')\""
        chunks = [
            chunk async for stream, chunk in session.run_stream(command) if stream == "stdout"
        ]
        assert "".join(chunks) == "x" * 4095 + "€\n"

    async def test_streaming_output_arrives_before_command_finishes(self, session, tmp_path):
        release = tmp_path / "release-stream"
        command = (
            "printf first; "
            f"while [ ! -f {shlex.quote(str(release))} ]; do sleep 0.01; done; "
            "printf second"
        )
        stream = session.run_stream(command)

        first = await asyncio.wait_for(anext(stream), timeout=1)
        assert first == ("stdout", "first")

        release.write_text("")
        remaining = [item async for item in stream]
        assert ("stdout", "second") in remaining
        assert remaining[-1] == ("__done__", "0,0")

    async def test_abandoned_stream_interrupts_command_and_resynchronizes_shell(self, session):
        stream = session.run_stream("printf started; sleep 30; printf stale")
        first = await asyncio.wait_for(anext(stream), timeout=1)
        assert first == ("stdout", "started")

        await asyncio.wait_for(stream.aclose(), timeout=3)

        out, err, code = await session.run("printf recovered")
        assert (out, err, code) == ("recovered", "", 0)

    async def test_late_stream_close_preserves_completed_shell_state(self, session, tmp_path):
        subdir = tmp_path / "completed-cwd"
        subdir.mkdir()
        await session.run("export PRESERVE=yes")
        start_count = session._start_count
        stream = session.run_stream(f"cd {shlex.quote(str(subdir))}; printf finished")

        assert await asyncio.wait_for(anext(stream), timeout=1) == ("stdout", "finished")
        await asyncio.sleep(0.1)
        await asyncio.wait_for(stream.aclose(), timeout=1)

        assert session._start_count == start_count
        assert session.cwd == subdir
        out, err, code = await session.run('printf "$PRESERVE:$PWD"')
        assert (out, err, code) == (f"yes:{subdir}", "", 0)

    async def test_abandoned_stream_kills_nested_descendants(self, session):
        stream = session.run_stream("printf started; sh -c '(sleep .3; printf stale) & wait'")
        assert await asyncio.wait_for(anext(stream), timeout=1) == ("stdout", "started")

        await asyncio.wait_for(stream.aclose(), timeout=2)

        out, err, code = await session.run("sleep .5; printf recovered")
        assert (out, err, code) == ("recovered", "", 0)

    async def test_background_output_is_isolated_from_next_command(self, session):
        first = [
            item async for item in session.run_stream("(sleep .2; printf late) & printf foreground")
        ]
        assert "".join(value for name, value in first if name == "stdout") == "foreground"
        assert first[-1] == ("__done__", "0,0")

        await asyncio.sleep(0.3)
        out, err, code = await session.run("printf next")
        assert (out, err, code) == ("next", "", 0)

    async def test_cancelling_consumer_task_resynchronizes_shell(self, session):
        received_output = asyncio.Event()

        async def consume() -> None:
            async for name, _ in session.run_stream("printf started; sleep 30"):
                if name == "stdout":
                    received_output.set()

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(received_output.wait(), timeout=1)
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer

        out, err, code = await asyncio.wait_for(session.run("printf recovered"), timeout=2)
        assert (out, err, code) == ("recovered", "", 0)

    async def test_slow_stream_consumer_applies_backpressure(self, session, tmp_path):
        marker = tmp_path / "producer-finished"
        command = (
            "python3 -c \"import os; os.write(1, b'x' * 2000000)\"; "
            f"touch {shlex.quote(str(marker))}"
        )
        stream = session.run_stream(command, timeout=10)

        first_name, first_chunk = await asyncio.wait_for(anext(stream), timeout=1)
        assert first_name == "stdout"
        await asyncio.sleep(0.1)
        assert not marker.exists()

        output_size = len(first_chunk)
        done = None
        async for name, value in stream:
            if name == "stdout":
                output_size += len(value)
            elif name == "__done__":
                done = value
        assert output_size == 2_000_000
        assert done == "0,0"
        assert marker.exists()

    async def test_paused_consumer_does_not_drop_other_completed_stream(self, session):
        stream = session.run_stream("printf OUT; printf ERR >&2")
        first = await asyncio.wait_for(anext(stream), timeout=1)
        await asyncio.sleep(0.1)
        events = [first, *[item async for item in stream]]

        assert "".join(value for name, value in events if name == "stdout") == "OUT"
        assert "".join(value for name, value in events if name == "stderr") == "ERR"
        assert events[-1] == ("__done__", "0,0")

    async def test_stream_reader_boundary_failure_is_reported(self, session):
        with pytest.raises(RuntimeError, match="stdout closed before"):
            _ = [item async for item in session.run_stream("exec 1>&-")]

        out, err, code = await session.run("printf recovered")
        assert (out, err, code) == ("recovered", "", 0)

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
