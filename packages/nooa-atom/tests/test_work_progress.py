# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Throttle-batched managed Shell progress, completion-only by default; no provider."""

import asyncio
import contextlib
import gc
import math
import shlex
import sys
import time
import weakref
from pathlib import Path

import pytest
from nooa_atom.agent.work_manager import WorkManager, _progress_sink, emit
from nooa_atom.agent.work_progress import (
    MAX_BYTES_LIMIT,
    ShellProgress,
    Throttle,
    _keep,
    _ShellProgress,
)
from nooa_atom.agent.work_shell import Shell

from nooa.runtime.channels import QueueManager


@pytest.fixture(autouse=True)
def small_shell_view(monkeypatch):
    """Cut Shell texts at 10,000 characters here (1 MiB by default), so 20 kB is long."""
    from nooa_atom.agent import work_shell

    monkeypatch.setattr(work_shell, "_VIEW_CHARS", 10_000)
    monkeypatch.setattr(work_shell, "_VIEW_TAIL_CHARS", 7_500)


class Clock:
    class Timer:
        def __init__(self, due, callback):
            self.due, self.callback, self.cancelled = due, callback, False

        def cancel(self):
            self.cancelled = True

    def __init__(self):
        self.now, self.timers = 0.0, []

    def call_later(self, seconds, callback):
        timer = self.Timer(self.now + seconds, callback)
        self.timers.append(timer)
        return timer

    def live(self):
        return [timer for timer in self.timers if not timer.cancelled]

    def advance(self, seconds):
        self.now += seconds
        for timer in list(self.timers):
            if not timer.cancelled and timer.due <= self.now:
                timer.cancelled = True
                timer.callback()


class Sink:
    def __init__(self):
        self.items, self.callback = [], None

    def on_close(self, callback):
        self.callback = callback

    def emit(self, item):
        self.items.append(item)

    def close(self):
        callback, self.callback = self.callback, None
        if callback:
            callback()


def texts(sink, stream="stdout"):
    return [getattr(item, stream) for item in sink.items if getattr(item, stream)]


# ---- Throttle ---------------------------------------------------------------


def test_throttle_defaults_repr_and_frozen():
    policy = Throttle()
    assert (policy.items, policy.seconds, policy.max_bytes) == (None, 1.0, 8192)
    assert repr(policy) == "Throttle(seconds=1)"
    assert repr(Throttle(items=20, seconds=None, max_bytes=100)) == (
        "Throttle(items=20, max_bytes=100)"
    )
    assert Throttle(items=3) == Throttle(items=3, seconds=1.0)
    with pytest.raises(AttributeError):
        policy.seconds = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"items": None, "seconds": None}, ValueError),
        ({"items": 0}, ValueError),
        ({"items": -1}, ValueError),
        ({"items": True}, TypeError),
        ({"items": 1.5}, TypeError),
        ({"seconds": 0}, ValueError),
        ({"seconds": -1.0}, ValueError),
        ({"seconds": math.inf}, ValueError),
        ({"seconds": math.nan}, ValueError),
        ({"seconds": "1"}, TypeError),
        ({"seconds": False}, TypeError),
        ({"max_bytes": 0}, ValueError),
        ({"max_bytes": MAX_BYTES_LIMIT + 1}, ValueError),
        ({"max_bytes": 10.0}, TypeError),
    ],
)
def test_throttle_validation(kwargs, error):
    with pytest.raises(error):
        Throttle(**kwargs)


# ---- batching ---------------------------------------------------------------


def test_default_policy_none_is_quiet_and_keeps_nothing():
    clock, sink = Clock(), Sink()
    progress = _ShellProgress(sink, None, loop=clock)
    progress.stdout.feed(b"x" * 100_000)
    clock.advance(100)
    assert sink.items == [] and progress.buffered() == 0 and not clock.timers
    progress.close()


def test_seconds_start_at_oldest_fragment_partial_line_and_no_empty_wakeups():
    clock, sink = Clock(), Sink()
    progress = _ShellProgress(sink, Throttle(), loop=clock)
    progress.stdout.feed(b"a")
    clock.advance(0.75)
    progress.stdout.feed(b"b")  # does not postpone the oldest fragment's deadline
    assert progress.buffered() == 2
    clock.advance(0.25)
    assert sink.items == [ShellProgress(stdout="ab")]  # no newline needed
    clock.advance(10)
    assert len(sink.items) == 1 and progress.stdout.timer is None and not clock.live()
    progress.close()


def test_items_count_lines_and_leave_partial_line_pending():
    clock, sink = Clock(), Sink()
    progress = _ShellProgress(sink, Throttle(items=2, seconds=None), loop=clock)
    progress.stdout.feed(b"one\ntwo")
    assert sink.items == [] and not clock.timers
    progress.stdout.feed(b"\nthree\nfour\nfive\nsix")
    assert texts(sink) == ["one\ntwo\n", "three\nfour\n"]
    assert bytes(progress.stdout.pending) == b"five\nsix"
    clock.advance(1000)  # no seconds: the partial batch waits (the final result has it)
    assert len(sink.items) == 2
    progress.close()


def test_items_or_seconds_whichever_first_restarts_timer_after_a_flush():
    clock, sink = Clock(), Sink()
    progress = _ShellProgress(sink, Throttle(items=2, seconds=1), loop=clock)
    progress.stdout.feed(b"a\n")
    clock.advance(0.5)
    progress.stdout.feed(b"b\nc")  # 2 lines: flush now; "c" is a new oldest fragment
    assert texts(sink) == ["a\nb\n"]
    clock.advance(0.9)
    assert len(sink.items) == 1
    clock.advance(0.1)
    assert texts(sink) == ["a\nb\n", "c"]
    progress.close()


def test_max_bytes_utf8_split_safety_and_invalid_bytes():
    clock, sink = Clock(), Sink()
    progress = _ShellProgress(sink, Throttle(seconds=1, max_bytes=8192), loop=clock)
    text = "a" * 8190 + "🙂" + "漢" * 4000
    wire = text.encode()
    for part in (wire[:8191], wire[8191:8193], wire[8193:]):
        progress.stdout.feed(part)
    progress.stderr.feed(b"bad: \xff\xf0")
    progress.stderr.feed(b"", final=True)
    clock.advance(1)
    assert "".join(texts(sink)) == text
    assert all(len(item.stdout.encode()) <= 8192 for item in sink.items)
    assert texts(sink, "stderr") == ["bad: ��"]
    assert progress.buffered() == 0
    progress.close()


def test_tiny_max_bytes_never_splits_or_stalls_on_a_wide_code_point():
    clock, sink = Clock(), Sink()
    progress = _ShellProgress(sink, Throttle(seconds=None, items=1000, max_bytes=1), loop=clock)
    progress.stdout.feed("a🙂b".encode())
    assert texts(sink) == ["a", "🙂", "b"]
    progress.close()


def test_streams_batch_separately_in_their_own_order():
    clock, sink = Clock(), Sink()
    progress = _ShellProgress(sink, Throttle(items=1, seconds=None), loop=clock)
    progress.stdout.feed(b"o1\n")
    progress.stderr.feed(b"e1\n")
    progress.stdout.feed(b"o2\n")
    progress.stderr.feed(b"e2")
    assert [(i.stdout, i.stderr) for i in sink.items] == [
        ("o1\n", ""),
        ("", "e1\n"),
        ("o2\n", ""),
    ]
    assert bytes(progress.stderr.pending) == b"e2"
    progress.close()


def test_set_policy_flushes_once_cancels_old_timer_and_new_timer_waits_for_next():
    clock, sink = Clock(), Sink()
    progress = _ShellProgress(sink, Throttle(seconds=30), loop=clock)
    progress.stdout.feed(b"pending")
    old = progress.stdout.timer
    progress.set_policy(Throttle(seconds=0.5))  # more chatty: what waited goes now
    assert texts(sink) == ["pending"] and old.cancelled and progress.stdout.timer is None
    clock.advance(10)
    assert len(sink.items) == 1  # no timer until the next fragment
    progress.stdout.feed(b"next")
    clock.advance(0.5)
    assert texts(sink) == ["pending", "next"]
    progress.set_policy(Throttle(seconds=0.5))  # same policy: nothing happens
    progress.stdout.feed(b"x")
    progress.set_policy(None)  # quiet: flushes once, then nothing more
    assert texts(sink)[-1] == "x"
    progress.stdout.feed(b"ignored" * 5000)
    clock.advance(100)
    assert texts(sink) == ["pending", "next", "x"] and progress.buffered() == 0
    progress.close()


def test_enabling_mid_stream_resynchronises_on_a_code_point():
    clock, sink = Clock(), Sink()
    progress = _ShellProgress(sink, None, loop=clock)
    wire = "🙂z".encode()
    progress.stdout.feed(wire[:2])  # quiet: skipped
    progress.set_policy(Throttle(seconds=1))
    progress.stdout.feed(wire[2:])
    clock.advance(1)
    assert texts(sink) == ["z"]  # no replacement characters from the skipped half
    progress.close()


def test_invalid_policy_changes_nothing():
    progress = _ShellProgress(Sink(), Throttle(), loop=Clock())
    with pytest.raises(TypeError):
        progress.set_policy({"seconds": 1})
    assert progress.policy == Throttle()
    progress.close()


def test_terminal_close_discards_tail_and_detaches_timers_and_sink():
    clock, sink = Clock(), Sink()
    progress = _ShellProgress(sink, Throttle(), loop=clock)
    progress.stdout.feed(b"short")
    progress.close()
    progress.stdout.feed(b"late" * 8192)
    clock.advance(100)
    assert not sink.items and progress.sink is None
    assert not progress.stdout.pending and progress.stdout.timer is None


def test_merge_keeps_head_and_tail_bounded_and_counts_omitted():
    first = ShellProgress(stdout="A" * 6000)
    merged, cut = first._merge(ShellProgress(stdout="B" * 6000, stderr="e"), 8192)
    assert cut == 1 and merged.batches == 2 and merged.stderr == "e"
    assert merged.omitted == 12000 - 8192
    head, tail = merged.stdout.split("progress omitted", 1)
    assert head.startswith("A" * 2048) and tail.endswith("B" * 6000)
    again, _ = merged._merge(ShellProgress(stdout="C" * 3000), 8192)
    assert again.omitted == merged.omitted + 3000
    assert again.stdout.count("progress omitted") == 1
    assert len(again.stdout.encode()) < 8192 + 40
    text, omitted = _keep("漢" * 5000, 100)
    assert omitted > 0 and "\ufffd" not in text


# ---- managed jobs -------------------------------------------------------------


@pytest.fixture
async def managed(tmp_path):
    channel = QueueManager().queue("jobs")
    (tmp_path / "scratch").mkdir()
    manager = WorkManager(channel, scratch=tmp_path / "scratch", max_items=3)
    yield manager, channel
    await manager.aclose()


async def until(predicate, timeout=10):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


NOISY_OUT = "O" * 20000
NOISY_ERR = "E" * 20000


def gated_command(tmp_path, *, prefix=NOISY_OUT, err=NOISY_ERR, tail="TAIL", name="gate"):
    gate = tmp_path / name
    ready = gate.with_suffix(".ready")
    program = (
        "import os,time,pathlib; "
        f"os.write(1,{prefix.encode()!r}); os.write(2,{err.encode()!r}); "
        f"pathlib.Path({str(ready)!r}).touch(); gate=pathlib.Path({str(gate)!r}); "
        "\nwhile not gate.exists(): time.sleep(.01)\n"
        f"os.write(1,{tail.encode()!r})"
    )
    return gate, shlex.join([sys.executable, "-c", program])


def stepped_command(tmp_path, steps):
    """Write each step's stdout text, then wait for ``<tmp>/stepN`` before the next."""
    lines = ["import os,time,pathlib"]
    for number, text in enumerate(steps):
        ready = tmp_path / f"step{number}.ready"
        gate = tmp_path / f"step{number}"
        lines += [
            f"os.write(1,{text.encode()!r})",
            f"pathlib.Path({str(ready)!r}).touch()",
            f"while not pathlib.Path({str(gate)!r}).exists(): time.sleep(.01)",
        ]
    return shlex.join([sys.executable, "-c", "\n".join(lines)])


async def ready(gate):
    await until(gate.with_suffix(".ready").exists)


async def past_old_preview_deadline(gate):
    # Both streams written; stay alive beyond 1s and the 8192-byte boundary.
    await ready(gate)
    await asyncio.sleep(1.1)
    assert not gate.exists()


def assert_full_capture(result, *, out=NOISY_OUT + "TAIL", err=NOISY_ERR):
    assert result.returncode == 0 and not result.timed_out
    read = "Full output: await self.files.read("
    assert read in result.stdout and read in result.stderr
    assert Path(result.stdout_path).read_text() == out
    assert Path(result.stderr_path).read_text() == err


def progress_events(channel):
    return [event for event in channel.snapshot() if event.kind == "item"]


async def test_default_launch_is_quiet_iterated_awaited_and_detached(managed, tmp_path):
    manager, channel = managed
    gate, command = gated_command(tmp_path)
    async with Shell(tmp_path) as shell:
        job = manager.launch(shell.run(command), name="observe")
        assert job.progress is None
        async with contextlib.aclosing(aiter(job)) as reader:
            pending = asyncio.create_task(anext(reader))
            try:
                await past_old_preview_deadline(gate)
                assert not pending.done() and job.status == "running"
                assert channel.snapshot() == [] and job._progress_buffered() == 0
                gate.touch()
                with pytest.raises(StopAsyncIteration):
                    await asyncio.wait_for(pending, 10)
            finally:
                gate.touch()
                if not pending.done():
                    pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
        assert_full_capture(await job)
        assert channel.snapshot() == [] and job.progress_dropped == 0
        assert not job._progress_sinks


async def test_detached_noisy_job_has_one_bounded_coalesced_item_then_final(managed, tmp_path):
    manager, channel = managed
    gate, command = gated_command(tmp_path)
    async with Shell(tmp_path) as shell:
        job = manager.launch(shell.run(command), name="noisy", progress=Throttle(seconds=0.2))
        try:
            await past_old_preview_deadline(gate)
            [event] = channel.snapshot()  # one unread preview per job, merged
            preview = event.value
            assert event.kind == "item" and isinstance(preview, ShellProgress)
            assert preview.batches >= 4 and preview.omitted > 0
            assert len(preview.stdout.encode()) < 8192 + 64
            assert len(preview.stderr.encode()) < 8192 + 64
            assert preview.stdout.startswith("O" * 2048) and preview.stderr.endswith("E" * 100)
            assert job.progress_dropped >= 1
            # Delivered (a turn took it): the next batch is a new item.
            channel.drain()
            gate.touch()
            await until(lambda: job.status == "done")
            [final] = channel.drain()  # final supersedes the pending "TAIL"
            assert final.kind == "done" and final.id == job.id
            assert_full_capture(final.value)
        finally:
            gate.touch()


async def test_final_supersedes_unread_progress_and_is_never_delayed(managed, tmp_path):
    manager, channel = managed
    gate, command = gated_command(tmp_path)
    async with Shell(tmp_path) as shell:
        job = manager.launch(shell.run(command), name="sup", progress=Throttle(seconds=0.2))
        try:
            await past_old_preview_deadline(gate)
            assert len(progress_events(channel)) == 1
            gate.touch()
            await until(lambda: job.status == "done")
            assert [event.kind for event in channel.snapshot()] == ["done"]
        finally:
            gate.touch()
        slow = manager.launch(shell.run("printf x"), name="slow", progress=Throttle(seconds=60))
        started = time.monotonic()
        result = await asyncio.wait_for(slow, 10)
        assert result.stdout == "x" and time.monotonic() - started < 5
        assert channel.snapshot()[-1].kind == "done"


async def test_iterator_gets_each_bounded_batch_in_stream_order(tmp_path):
    channel = QueueManager().queue("jobs")
    manager = WorkManager(channel, scratch=tmp_path, max_items=100)  # room for every batch
    gate, command = gated_command(tmp_path, prefix="o" * 20000, err="e" * 9000)
    async with Shell(tmp_path) as shell:
        job = manager.launch(shell.run(command), name="it", progress=Throttle(max_bytes=4096))
        got = []
        try:
            async with contextlib.aclosing(aiter(job)) as reader:
                await ready(gate)
                while sum(len(p.stdout) for p in got) < 20000 or (
                    sum(len(p.stderr) for p in got) < 9000
                ):
                    got.append(await asyncio.wait_for(anext(reader), 10))
                gate.touch()
                got += [item async for item in reader]
        finally:
            gate.touch()
        assert all(p.batches == 1 and not p.omitted for p in got)
        assert all(len(p.stdout) <= 4096 and len(p.stderr) <= 4096 for p in got)
        assert "".join(p.stdout for p in got).startswith("o" * 20000)
        assert "".join(p.stderr for p in got) == "e" * 9000
        assert (await job).returncode == 0
        assert channel.snapshot() == [] and job.progress_dropped == 0
    await manager.aclose()


async def test_set_progress_on_a_running_command(managed, tmp_path):
    manager, channel = managed
    command = stepped_command(tmp_path, ["quiet\n", "slow\n", "fast\n", "after-none\n"])
    async with Shell(tmp_path) as shell:
        job = manager.launch(shell.run(command), name="tune")
        try:
            await ready(tmp_path / "step0")
            await asyncio.sleep(0.3)
            assert channel.snapshot() == []
            with pytest.raises(TypeError):
                job.set_progress("loud")  # type: ignore[arg-type]
            with pytest.raises(ValueError):
                job.set_progress(Throttle(seconds=-1))
            assert job.progress is None  # invalid: nothing changed
            assert job.set_progress(Throttle(seconds=30)) is True
            assert job.set_progress(Throttle(seconds=30)) is True  # same: a no-op
            (tmp_path / "step0").touch()
            await ready(tmp_path / "step1")
            await until(lambda: job._progress_buffered() == 5)
            assert channel.snapshot() == []  # 30s: still batching
            # More chatty: the pending batch goes at once, under the old policy.
            assert job.set_progress(Throttle(seconds=0.1)) is True
            [event] = channel.drain()
            assert event.value.stdout == "slow\n"  # skipped quiet output is not replayed
            (tmp_path / "step1").touch()
            await ready(tmp_path / "step2")
            await until(lambda: bool(channel.snapshot()))
            assert [e.value.stdout for e in channel.drain()] == ["fast\n"]
            assert job.set_progress(None) is True
            (tmp_path / "step2").touch()
            await ready(tmp_path / "step3")
            await asyncio.sleep(0.4)
            assert channel.snapshot() == [] and job.status == "running"
            (tmp_path / "step3").touch()
            result = await asyncio.wait_for(job, 10)
            assert result.stdout == "quiet\nslow\nfast\nafter-none\n"  # final unaffected
        finally:
            for n in range(4):
                (tmp_path / f"step{n}").touch()
        assert job.set_progress(Throttle()) is False and job.progress is None


async def test_resource_default_is_for_future_requests_and_requests_never_mix(managed, tmp_path):
    manager, channel = managed
    resource = manager.launch(Shell(tmp_path), name="sh", progress=Throttle(seconds=0.1))
    assert resource.progress == Throttle(seconds=0.1)
    first_cmd = stepped_command(tmp_path, ["first-out\n"])
    first = resource.send(first_cmd)
    second = resource.send("printf second-out")  # queued behind the first
    try:
        await ready(tmp_path / "step0")
        assert first.progress == second.progress == Throttle(seconds=0.1)
        assert resource.set_progress(None) is True  # only future requests change
        third = resource.send("printf third-out; sleep .5")
        assert third.progress is None and first.progress == Throttle(seconds=0.1)
        await until(lambda: bool(channel.snapshot()))
        [event] = channel.drain()
        assert (event.name, event.value.stdout) == ("sh#1", "first-out\n")
        assert manager.get("sh#2") is second and second.set_progress(None) is True
        assert first.progress == Throttle(seconds=0.1)
        (tmp_path / "step0").touch()
        assert (await first).stdout == "first-out\n"
        assert (await second).stdout == "second-out"
        assert (await third).stdout == "third-out"
        for event in channel.snapshot():
            assert event.kind != "item"
    finally:
        (tmp_path / "step0").touch()


async def test_per_request_buffers_are_independent_under_one_policy(managed, tmp_path):
    manager, channel = managed
    resource = manager.launch(Shell(tmp_path), name="pair", progress=Throttle(seconds=0.1))
    one = resource.send("printf 'aaa\\n'; sleep .4")
    two = resource.send("printf 'bbb\\n'; sleep .4")
    await one
    await two
    items = [event for event in channel.snapshot() if event.kind == "item"]
    assert [(e.name, e.value.stdout) for e in items] == []  # awaited: completion-only
    three = resource.send("printf 'ccc\\n'; sleep .4")
    four = resource.send("printf 'ddd\\n'; sleep .4")
    await until(lambda: four.status == "done")
    items = [(e.name, e.value.stdout) for e in channel.snapshot() if e.kind == "item"]
    assert set(items) <= {("pair#3", "ccc\n"), ("pair#4", "ddd\n")}
    assert ("pair#3", "ccc\n") in items or three.status == "done"


async def test_completion_only_await_suppresses_previews_not_ordinary_emit(managed, tmp_path):
    manager, channel = managed
    gate, command = gated_command(tmp_path)
    async with Shell(tmp_path) as shell:

        async def work():
            emit("ordinary child/control message")
            return await shell.run(command)

        job = manager.launch(work(), name="final_only", progress=Throttle(seconds=0.1))
        waiter = asyncio.ensure_future(job)
        try:
            await past_old_preview_deadline(gate)
            assert [event.value for event in channel.snapshot()] == [
                "ordinary child/control message"
            ]
            assert job.progress_dropped > 0
            gate.touch()
            assert_full_capture(await asyncio.wait_for(waiter, 10))
        finally:
            gate.touch()
            await asyncio.wait_for(asyncio.gather(waiter, return_exceptions=True), 10)


async def test_timeout_and_lingering_background_writer_emit_no_late_items(managed, tmp_path):
    manager, channel = managed
    gate, command = gated_command(tmp_path)
    async with Shell(tmp_path) as shell:
        job = manager.launch(
            shell.run(command, timeout=3), name="timeout", progress=Throttle(seconds=0.2)
        )
        await past_old_preview_deadline(gate)
        assert len(progress_events(channel)) == 1
        await until(lambda: job.status == "done")
        [final] = channel.drain()
        assert final.kind == "done" and final.value.timed_out
        assert not job._progress_sinks
        lingering = manager.launch(
            shell.run("printf initial; (sleep 1.2; printf late; sleep .2) &"),
            name="background",
            progress=Throttle(seconds=0.1),
        )
        await until(lambda: lingering.status == "done")
        kinds = [event.kind for event in channel.drain()]
        assert kinds[-1] == "done" and kinds.count("done") == 1
        await asyncio.sleep(1.5)
        assert channel.snapshot() == [] and not lingering._progress_sinks


@pytest.mark.parametrize("owner_cancel", [False, True], ids=["task", "owner"])
async def test_cancel_detaches_timers_and_emits_no_late_items(managed, tmp_path, owner_cancel):
    manager, channel = managed
    gate, command = gated_command(tmp_path)
    async with Shell(tmp_path) as shell:
        job = manager.launch(shell.run(command), name="cancel", progress=Throttle(seconds=30))
        await ready(gate)
        await until(lambda: job._progress_buffered() > 0)
        sinks = list(job._progress_sinks)
        if owner_cancel:
            await job.cancel()
            assert channel.snapshot() == []
        else:
            job._task.cancel()
            await until(lambda: job.status == "cancelled")
            assert [event.kind for event in channel.drain()] == ["cancelled"]
        assert not job._progress_sinks and all(sink._job is None for sink in sinks)
        assert job._progress_buffered() == 0
        assert job.set_progress(Throttle()) is False
        assert (await manager.launch(shell.run("printf next"))).stdout == "next"
        await asyncio.sleep(0.3)
        assert [event.kind for event in channel.snapshot()] in ([], ["done"])


# ---- generic delivery (private sink) ---------------------------------------------


@pytest.mark.parametrize("reader_attached", [False, True])
async def test_switch_from_pending_preview_to_final_only_preserves_generic_items(
    managed, reader_attached
):
    manager, channel = managed
    started, release = asyncio.Event(), asyncio.Event()
    holder = []

    async def work():
        sink = _progress_sink()
        holder.append(sink)
        sink.emit(ShellProgress(stdout="preview"))
        emit("control")
        started.set()
        await release.wait()
        return "final"

    job = manager.launch(work(), name="transition")
    reader = aiter(job) if reader_attached else None
    await started.wait()
    if reader is not None:
        assert (await anext(reader)).stdout == "preview"
        holder[0].emit(ShellProgress(stdout="buffered"))
    waiter = asyncio.ensure_future(job)
    await until(lambda: bool(job._waiters))
    holder[0].emit(ShellProgress(stderr="suppressed"))
    release.set()
    assert await waiter == "final"
    if reader is not None:
        assert await anext(reader) == "control"
        with pytest.raises(StopAsyncIteration):
            await anext(reader)
    else:
        assert [event.value for event in channel.snapshot()] == ["control"]
    assert job.progress_dropped >= 2


async def test_unread_previews_coalesce_into_one_and_never_evict_generic_items(managed):
    manager, channel = managed
    started, release = asyncio.Event(), asyncio.Event()
    holder = []

    async def work():
        holder.append(_progress_sink())
        emit("keep")
        for i in range(20):
            holder[0].emit(ShellProgress(stdout=str(i)))
        started.set()
        await release.wait()
        return "done"

    job = manager.launch(work(), name="flood")
    await started.wait()
    keep, merged = channel.snapshot()
    assert keep.value == "keep" and merged.value.batches == 20
    assert merged.value.stdout == "".join(str(i) for i in range(20))
    assert job.progress_dropped == 0  # merged whole: nothing cut
    await job.cancel()
    holder[0].emit(ShellProgress(stdout="late"))
    assert channel.snapshot() == [] and not job._progress_sinks


async def test_reader_flood_is_bounded_and_never_evicts_generic_items(managed):
    manager, channel = managed
    started = asyncio.Event()
    holder = []

    async def work():
        holder.append(_progress_sink())
        emit("keep")
        for i in range(20):
            holder[0].emit(ShellProgress(stdout=str(i)))
        started.set()
        await asyncio.sleep(1000)

    job = manager.launch(work(), name="reader_flood")
    reader = aiter(job)
    await started.wait()
    assert len(reader._buffer) <= 3 and await anext(reader) == "keep"
    assert job.progress_dropped == 18
    await reader.aclose()
    await job.cancel()
    assert channel.snapshot() == []


async def test_cancel_with_attached_reader_withdraws_private_previews(managed):
    manager, channel = managed
    started = asyncio.Event()

    async def work():
        sink = _progress_sink()
        sink.emit(ShellProgress(stdout="stale"))
        emit("ordinary")
        started.set()
        await asyncio.sleep(1000)

    job = manager.launch(work(), name="cancel")
    reader = aiter(job)
    await started.wait()
    await job.cancel()
    assert await anext(reader) == "ordinary"
    with pytest.raises(RuntimeError, match="cancelled"):
        await anext(reader)
    assert channel.snapshot() == [] and job.progress_dropped == 1


async def test_pending_preview_then_terminal_queues_only_final_plus_generic(managed):
    manager, channel = managed

    async def work():
        sink = _progress_sink()
        sink.emit(ShellProgress(stdout="superseded"))
        emit("keep")
        return "final"

    job = manager.launch(work(), name="terminal")
    await until(lambda: job.status == "done")
    assert [(item.kind, item.value) for item in channel.snapshot()] == [
        ("item", "keep"),
        ("done", "final"),
    ]
    assert job.progress_dropped == 1


async def test_internal_task_cancellation_also_withdraws_preview_only(managed):
    manager, channel = managed
    started = asyncio.Event()

    async def work():
        sink = _progress_sink()
        sink.emit(ShellProgress(stdout="unread"))
        emit("ordinary")
        started.set()
        await asyncio.sleep(1000)

    job = manager.launch(work(), name="internal_cancel")
    await started.wait()
    job._task.cancel()
    await until(lambda: job.status == "cancelled")
    assert [(item.kind, item.value) for item in channel.snapshot()] == [
        ("item", "ordinary"),
        ("cancelled", None),
    ]
    assert job.progress_dropped == 1 and not job._progress_sinks


async def test_preview_delivery_failure_does_not_break_work_or_capture(managed, monkeypatch):
    manager, channel = managed
    started, release = asyncio.Event(), asyncio.Event()

    def broken_put(item):
        raise RuntimeError("outbox unavailable")

    monkeypatch.setattr(channel, "put", broken_put)

    async def work():
        sink = _progress_sink()
        progress = _ShellProgress(sink, Throttle())
        progress.stdout.feed(b"x" * 8192)
        assert progress.closed  # preview failure detaches it rather than killing a pump
        started.set()
        await release.wait()
        return "ordinary result"

    job = manager.launch(work(), name="delivery_failure", progress=Throttle())
    await started.wait()
    waiter = asyncio.ensure_future(job)
    await until(lambda: bool(job._waiters))
    release.set()
    assert await waiter == "ordinary result"
    assert job.progress_dropped == 1 and not job._progress_sinks


async def test_delivered_preview_frees_the_slot_and_cancel_counts_omissions(managed):
    manager, channel = managed
    started, resume = asyncio.Event(), asyncio.Event()
    holder = []

    async def work():
        for number in range(3):
            emit(number)
        holder.append(_progress_sink())
        started.set()
        await resume.wait()

    job = manager.launch(work(), name="history")
    await started.wait()
    assert [event.value for event in channel.drain()] == [0, 1, 2]
    holder[0].emit(ShellProgress(stdout="first"))
    [delivered] = channel.drain()
    holder[0].emit(ShellProgress(stdout="second"))
    holder[0].emit(ShellProgress(stdout="third"))
    assert delivered.value.stdout == "first"
    assert [event.value.stdout for event in channel.snapshot()] == ["secondthird"]
    await job.cancel()
    assert job.progress_dropped == 1 and channel.snapshot() == []


async def test_terminal_reader_detach_delivers_only_shell_result(managed, tmp_path):
    manager, channel = managed
    gate, command = gated_command(tmp_path)
    async with Shell(tmp_path) as shell:
        job = manager.launch(shell.run(command), name="detach_after_terminal")
        reader = aiter(job)
        try:
            await past_old_preview_deadline(gate)
            assert job.status == "running" and not reader._buffer
            gate.touch()
            await until(lambda: job.status != "running")
            assert job.status == "done", job.error
            assert len(reader._buffer) == 1  # only the terminal marker
            await reader.aclose()
            [final] = channel.snapshot()
            assert final.kind == "done" and final.id == job.id
            assert_full_capture(final.value)
            assert await job is final.value
        finally:
            gate.touch()
            await reader.aclose()
            if job.status == "running":
                await job.cancel()


@pytest.mark.parametrize("failed", [False, True], ids=["done", "failed"])
@pytest.mark.parametrize("collected", [False, True], ids=["close", "gc"])
async def test_terminal_reader_detach_counts_each_preview_once_and_keeps_generic_items(
    managed, failed, collected
):
    manager, channel = managed
    release = asyncio.Event()

    async def work():
        sink = _progress_sink()
        sink.emit(ShellProgress(stdout="one"))
        emit("ordinary")
        sink.emit(ShellProgress(stderr="two"))
        await release.wait()
        if failed:
            raise ValueError("work failed")
        return "final"

    job = manager.launch(work(), name="terminal_accounting")
    reader = aiter(job)
    await until(lambda: len(reader._buffer) == 3)
    release.set()
    await until(lambda: job.status != "running")
    if collected:
        ref = weakref.ref(reader)
        del reader
        gc.collect()
        assert ref() is None
    else:
        await reader.aclose()
        await reader.aclose()
    assert job.progress_dropped == 2
    [ordinary, final] = channel.snapshot()
    assert ordinary.kind == "item" and ordinary.value == "ordinary"
    assert final.kind == ("failed" if failed else "done")


@pytest.mark.parametrize("callback", ["equal", "raises"])
@pytest.mark.parametrize("attached", [False, True], ids=["outbox", "reader"])
async def test_progress_eviction_never_compares_an_ordinary_payload(managed, callback, attached):
    manager, channel = managed
    compared = []

    class Control:
        def __eq__(self, other):
            compared.append(True)
            if callback == "raises":
                raise AssertionError("ordinary payload equality must never run")
            return True

    control = Control()
    started, release = asyncio.Event(), asyncio.Event()

    async def work():
        emit(control)
        sink = _progress_sink()
        for text in ("one", "two", "three"):
            sink.emit(ShellProgress(stdout=text))
        started.set()
        await release.wait()
        return "final"

    job = manager.launch(work(), name="identity_eviction")
    reader = aiter(job) if attached else None
    try:
        await started.wait()
        assert compared == []
        if reader is not None:
            assert job.progress_dropped == 1
            assert await anext(reader) is control
            assert [(await anext(reader)).stdout for _ in range(2)] == ["two", "three"]
        else:
            ordinary, merged = channel.drain()
            assert ordinary.value is control and merged.value.stdout == "onetwothree"
        assert compared == []
        release.set()
        assert await job == "final"
    finally:
        release.set()
        if job.status == "running":
            await job.cancel()
        if reader is not None:
            await reader.aclose()


async def test_launch_rejects_a_non_throttle_and_other_work_accepts_one(managed):
    manager, _ = managed
    coro = asyncio.sleep(0, "x")
    with pytest.raises(TypeError, match="Throttle"):
        manager.launch(coro, progress={"seconds": 1})  # type: ignore[arg-type]
    job = manager.launch(asyncio.sleep(0, "plain"), progress=Throttle(items=5))
    assert job.progress == Throttle(items=5) and await job == "plain"
