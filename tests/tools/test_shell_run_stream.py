# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ShellTools.run_stream: live, lossless, bounded and cancellable output."""

import asyncio
import subprocess
import time
import tracemalloc

import pytest

from nooa.tools.shell_tools import ShellTools


@pytest.fixture
async def sh(tmp_path):
    shell = ShellTools(cwd=str(tmp_path))
    yield shell
    await shell.close()


def _text(events, kind):
    return "".join(event.text for event in events if event.kind == kind)


def _running(pattern: str) -> bool:
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


async def test_first_chunk_arrives_before_the_command_ends(sh):
    stream = sh.run_stream("echo first; sleep 3; echo second", timeout=10.0)
    start = time.monotonic()
    first = await asyncio.wait_for(anext(stream), timeout=2.0)
    assert time.monotonic() - start < 2.0
    assert (first.kind, first.text) == ("stdout", "first\n")
    rest = [event async for event in stream]
    assert _text(rest, "stdout") == "second\n"
    assert (rest[-1].kind, rest[-1].returncode, rest[-1].timed_out) == ("done", 0, False)


async def test_large_output_is_complete_and_in_order(sh):
    events = [event async for event in sh.run_stream("seq 1 200000", timeout=30.0)]
    expected = "".join(f"{i}\n" for i in range(1, 200001))
    assert _text(events, "stdout") == expected
    assert events[-1].returncode == 0


async def test_multibyte_text_split_across_reads_is_decoded_intact(sh):
    events = [event async for event in sh.run_stream("yes é | head -n 30000", timeout=30.0)]
    out = _text(events, "stdout")
    assert out.count("\ufffd") == 0
    assert out == "é\n" * 30000


async def test_stdout_and_stderr_stay_separate(sh):
    events = [event async for event in sh.run_stream("echo out; echo err >&2; echo out2; (exit 3)")]
    assert _text(events, "stdout") == "out\nout2\n"
    assert _text(events, "stderr") == "err\n"
    assert (events[-1].kind, events[-1].returncode) == ("done", 3)


async def test_memory_stays_bounded_for_a_large_output(sh):
    await sh.run("true")  # start the session outside the measurement
    size = 30_000_000
    received = 0
    tracemalloc.start()
    try:
        async for event in sh.run_stream(f"head -c {size} /dev/zero | tr '\\0' a", timeout=60.0):
            if event.kind == "stdout":
                received += len(event.text)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert received == size
    assert peak < 5_000_000


async def test_timeout_still_reports_timed_out(sh):
    events = [event async for event in sh.run_stream("echo a; sleep 20", timeout=0.5)]
    assert _text(events, "stdout") == "a\n"
    assert (events[-1].returncode, events[-1].timed_out) == (124, True)
    assert (await sh.run("echo ok")).stdout == "ok"


async def test_aclose_stops_the_command_and_keeps_the_session(sh):
    await sh.run("export KEPT=yes")
    stream = sh.run_stream("echo started; sleep 47.301; echo finished", timeout=60.0)
    first = await asyncio.wait_for(anext(stream), timeout=5.0)
    assert first.text == "started\n"
    assert _running("sleep 47.301")

    start = time.monotonic()
    await stream.aclose()
    assert time.monotonic() - start < 5.0
    assert not _running("sleep 47.301")
    # Nothing from the abandoned command bleeds into the next one, and shell
    # state set before it is still there.
    assert (await sh.run("echo ok")).stdout == "ok"
    assert (await sh.run("echo $KEPT")).stdout == "yes"


async def test_breaking_out_of_the_loop_stops_the_command(sh):
    async def first_chunk():
        async for event in sh.run_stream("echo started; sleep 47.302", timeout=60.0):
            return event

    first = await asyncio.wait_for(first_chunk(), timeout=5.0)
    assert first.text == "started\n"
    # The abandoned stream is closed by the event loop; the next command
    # waits for that and then runs normally.
    result = await asyncio.wait_for(sh.run("echo ok"), timeout=10.0)
    assert result.stdout == "ok"
    assert not _running("sleep 47.302")


async def test_cancelling_the_consumer_stops_the_command(sh):
    seen = asyncio.Event()

    async def consume():
        async for _event in sh.run_stream("echo started; sleep 47.303", timeout=60.0):
            seen.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(seen.wait(), timeout=5.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not _running("sleep 47.303")
    assert (await sh.run("echo ok")).stdout == "ok"
