# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private preview batching and real managed-Shell pre-exit delivery; no provider."""

import asyncio
import contextlib
import gc
import shlex
import sys
import weakref
from pathlib import Path

import pytest
from nooa_atom.agent.work_manager import WorkManager, _progress_sink, emit
from nooa_atom.agent.work_progress import ShellProgress, _ShellProgress
from nooa_atom.agent.work_shell import Shell

from nooa.runtime.channels import QueueManager


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


def test_time_starts_at_oldest_fragment_no_newline_or_empty_wakeups():
    clock, sink = Clock(), Sink()
    progress = _ShellProgress(sink, loop=clock)
    assert (progress.seconds, progress.bytes) == (1.0, 8192)
    progress.stdout.feed(b"a")
    clock.advance(0.75)
    progress.stdout.feed(b"b")
    clock.advance(0.25)
    assert sink.items == [ShellProgress("stdout", "ab", 1)]
    clock.advance(10)
    assert len(sink.items) == 1 and progress.stdout.timer is None
    progress.close()


def test_byte_boundary_utf8_invalid_and_per_stream_order():
    clock, sink = Clock(), Sink()
    progress = _ShellProgress(sink, loop=clock)
    text = "a" * 8190 + "🙂" + "漢" * 4000
    wire = text.encode()
    for part in (wire[:8191], wire[8191:8193], wire[8193:]):
        progress.stdout.feed(part)
    progress.stderr.feed(b"bad: \xff\xf0")
    progress.stderr.feed(b"", final=True)
    clock.advance(1)
    out = [item for item in sink.items if item.stream == "stdout"]
    assert "".join(item.text for item in out) == text
    assert [item.sequence for item in out] == list(range(1, len(out) + 1))
    assert all(len(item.text.encode()) <= 8192 for item in sink.items)
    assert [item.text for item in sink.items if item.stream == "stderr"] == ["bad: ��"]
    assert not progress.stdout.pending and not progress.stderr.pending
    progress.close()


def test_terminal_discards_tail_and_detaches_timers_and_sink():
    clock, sink = Clock(), Sink()
    progress = _ShellProgress(sink, loop=clock)
    progress.stdout.feed(b"short")
    progress.close()
    progress.stdout.feed(b"late" * 8192)
    clock.advance(100)
    assert not sink.items and progress.sink is None
    assert not progress.stdout.pending and progress.stdout.timer is None


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


def gated_command(tmp_path, *, prefix="OUT", err="ERR", tail="TAIL"):
    gate = tmp_path / "gate"
    program = (
        "import os,time,pathlib; "
        f"os.write(1,{prefix.encode()!r}); os.write(2,{err.encode()!r}); "
        f"gate=pathlib.Path({str(gate)!r}); "
        "\nwhile not gate.exists(): time.sleep(.01)\n"
        f"os.write(1,{tail.encode()!r})"
    )
    return gate, shlex.join([sys.executable, "-c", program])


async def test_real_process_iteration_sees_stdout_stderr_before_exit_same_job(managed, tmp_path):
    manager, channel = managed
    gate, command = gated_command(tmp_path)
    async with Shell(tmp_path) as shell:
        job = manager.launch(shell.run(command), name="observe")
        got = []
        async with contextlib.aclosing(aiter(job)) as reader:
            async with asyncio.timeout(10):
                async for item in reader:
                    assert isinstance(item, ShellProgress)
                    got.append(item)
                    if {part.stream for part in got} == {"stdout", "stderr"}:
                        assert job.status == "running" and not gate.exists()
                        gate.touch()
        result = await job
    assert (result.returncode, result.stdout, result.stderr) == (0, "OUTTAIL", "ERR")
    assert {part.stream: part.text for part in got} == {"stdout": "OUT", "stderr": "ERR"}
    assert channel.snapshot() == []


async def test_detached_notifications_before_exit_final_supersedes_unread_preview(
    managed, tmp_path
):
    manager, channel = managed
    gate, command = gated_command(tmp_path)
    async with Shell(tmp_path) as shell:
        job = manager.launch(shell.run(command), name="later")
        await until(lambda: len(channel.snapshot()) >= 2)
        first = channel.drain()
        assert all(event.kind == "item" and event.name == job.name for event in first)
        assert {event.value.stream for event in first} == {"stdout", "stderr"}
        assert job.status == "running"
        gate.touch()
        await until(lambda: job.status == "done")
        [final] = channel.drain()
        assert final.kind == "done" and final.value.stdout == "OUTTAIL"
        assert (await job).stderr == "ERR"


async def test_completion_only_await_does_not_notify_but_ordinary_emit_does(managed, tmp_path):
    manager, channel = managed
    gate, command = gated_command(tmp_path)
    async with Shell(tmp_path) as shell:

        async def work():
            emit("ordinary child/control message")
            return await shell.run(command)

        job = manager.launch(work(), name="final_only")
        waiter = asyncio.ensure_future(job)
        try:
            await until(lambda: job.progress_dropped == 2)
            assert [event.value for event in channel.snapshot()] == [
                "ordinary child/control message"
            ]
            assert job.status == "running"
            gate.touch()
            result = await asyncio.wait_for(waiter, 10)
            assert result.stdout == "OUTTAIL"
            assert [event.value for event in channel.snapshot()] == [
                "ordinary child/control message"
            ]
        finally:
            gate.touch()
            await asyncio.gather(waiter, return_exceptions=True)


async def test_short_command_has_only_final_and_never_a_pending_tail_timer(managed, tmp_path):
    manager, channel = managed
    job = manager.launch(Shell(tmp_path).run("printf tiny; printf err >&2"), name="short")
    await until(lambda: job.status == "done")
    assert [event.kind for event in channel.snapshot()] == ["done"]
    assert (await job).stdout == "tiny"
    await asyncio.sleep(1.1)
    assert channel.snapshot() == [] and not job._progress_sinks


@pytest.mark.parametrize("reader_attached", [False, True])
async def test_switch_from_pending_preview_to_final_only_preserves_generic_items(
    managed, reader_attached
):
    manager, channel = managed
    ready, release = asyncio.Event(), asyncio.Event()
    holder = []

    async def work():
        sink = _progress_sink()
        holder.append(sink)
        sink.emit(ShellProgress("stdout", "preview", 1))
        emit("control")
        ready.set()
        await release.wait()
        return "final"

    job = manager.launch(work(), name="transition")
    reader = aiter(job) if reader_attached else None
    await ready.wait()
    if reader is not None:
        # Leave this reader referenced, like breaking from a loop and then awaiting.
        assert (await anext(reader)).text == "preview"
        holder[0].emit(ShellProgress("stdout", "buffered", 2))
    waiter = asyncio.ensure_future(job)
    await until(lambda: bool(job._waiters))
    holder[0].emit(ShellProgress("stderr", "suppressed", 3))
    release.set()
    assert await waiter == "final"
    if reader is not None:
        assert await anext(reader) == "control"
        with pytest.raises(StopAsyncIteration):
            await anext(reader)
    else:
        assert [event.value for event in channel.snapshot()] == ["control"]
    assert job.progress_dropped >= 2


@pytest.mark.parametrize("attach", [False, True])
async def test_progress_flood_never_evicts_generic_items_and_is_bounded(managed, attach):
    manager, channel = managed
    ready, release = asyncio.Event(), asyncio.Event()
    holder = []

    async def work():
        holder.append(_progress_sink())
        emit("keep")
        for i in range(20):
            holder[0].emit(ShellProgress("stdout", str(i), i))
        ready.set()
        await release.wait()
        return "done"

    job = manager.launch(work(), name="flood")
    reader = aiter(job) if attach else None
    await ready.wait()
    if reader:
        assert len(reader._buffer) <= 3 and await anext(reader) == "keep"
        await reader.aclose()
    else:
        assert len(channel.snapshot()) <= 3 and channel.snapshot()[0].value == "keep"
    assert job.progress_dropped == 18
    await job.cancel()
    holder[0].emit(ShellProgress("stdout", "late", 100))
    assert channel.snapshot() == [] and not job._progress_sinks
    replacement = manager.launch(asyncio.sleep(0, "new"), name="flood")
    assert await replacement == "new"
    assert channel.snapshot() == []


async def test_cancel_with_attached_reader_withdraws_private_previews(managed):
    manager, channel = managed
    ready = asyncio.Event()

    async def work():
        sink = _progress_sink()
        sink.emit(ShellProgress("stdout", "stale", 1))
        emit("ordinary")
        ready.set()
        await asyncio.sleep(1000)

    job = manager.launch(work(), name="cancel")
    reader = aiter(job)
    await ready.wait()
    await job.cancel()
    assert await anext(reader) == "ordinary"  # not a global emit/reader behavior change
    with pytest.raises(RuntimeError, match="cancelled"):
        await anext(reader)
    assert channel.snapshot() == [] and job.progress_dropped == 1


async def test_pending_preview_then_terminal_queues_only_final_plus_generic(managed):
    manager, channel = managed

    async def work():
        sink = _progress_sink()
        sink.emit(ShellProgress("stdout", "superseded", 1))
        emit("keep")
        return "final"

    job = manager.launch(work(), name="terminal")
    await until(lambda: job.status == "done")
    assert [(item.kind, item.value) for item in channel.snapshot()] == [
        ("item", "keep"),
        ("done", "final"),
    ]
    assert job.progress_dropped == 1


async def test_timeout_and_lingering_background_writer_detach_preview(managed, tmp_path):
    manager, channel = managed
    async with Shell(tmp_path) as shell:
        job = manager.launch(shell.run("printf start; sleep 20", timeout=1.25), name="timeout")
        await until(lambda: bool(channel.snapshot()))
        assert channel.drain()[0].value.text == "start" and job.status == "running"
        result = await job
        assert result.timed_out and result.returncode == 124 and result.stdout == "start"
        assert not job._progress_sinks
        # The command ends while a writer keeps its FIFO open past the batch deadline.
        lingering = manager.launch(
            shell.run("printf initial; (sleep 1.2; printf late; sleep .2) &"), name="background"
        )
        await until(lambda: lingering.status == "done")
        assert [event.kind for event in channel.drain()] == ["done"]
        assert (await lingering).stdout == "initial"
        await asyncio.sleep(1.5)
        assert channel.snapshot() == [] and not lingering._progress_sinks


async def test_resource_requests_keep_progress_generation_and_capture_limits_separate(
    managed, tmp_path
):
    manager, channel = managed
    gate, command = gated_command(tmp_path, prefix="a" * 20000, err="", tail="z")
    resource = manager.launch(Shell(tmp_path, max_output=12000), name="resource")
    request = resource.send(command)
    reader = aiter(request)
    first = await asyncio.wait_for(anext(reader), 10)
    assert first.stream == "stdout" and len(first.text.encode()) == 8192
    assert request.status == "running" and not gate.exists()
    gate.touch()
    async for _ in reader:
        pass
    result = await request
    assert "output limit" in result.stdout and result.stdout_path
    assert len(Path(result.stdout_path).read_bytes()) == 12000
    second = resource.send("printf next")
    assert (await second).stdout == "next"
    assert second.id != request.id and not second._progress_sinks
    assert not channel.snapshot()


def test_policy_is_fixed_per_command_and_not_publicly_retunable():
    progress = _ShellProgress(Sink(), loop=Clock())
    with pytest.raises(AttributeError):
        progress.seconds = 0.01
    with pytest.raises(AttributeError):
        progress.bytes = 1
    progress.close()


async def test_internal_task_cancellation_also_withdraws_preview_only(managed):
    manager, channel = managed
    ready = asyncio.Event()

    async def work():
        sink = _progress_sink()
        sink.emit(ShellProgress("stdout", "unread", 1))
        emit("ordinary")
        ready.set()
        await asyncio.sleep(1000)

    job = manager.launch(work(), name="internal_cancel")
    await ready.wait()
    job._task.cancel()
    await until(lambda: job.status == "cancelled")
    assert [(item.kind, item.value) for item in channel.snapshot()] == [
        ("item", "ordinary"),
        ("cancelled", None),
    ]
    assert job.progress_dropped == 1 and not job._progress_sinks


async def test_preview_delivery_failure_does_not_break_work_or_capture(managed, monkeypatch):
    manager, channel = managed
    ready, release = asyncio.Event(), asyncio.Event()

    def broken_put(item):
        raise RuntimeError("outbox unavailable")

    monkeypatch.setattr(channel, "put", broken_put)

    async def work():
        sink = _progress_sink()
        progress = _ShellProgress(sink)
        progress.stdout.feed(b"x" * 8192)
        assert progress.closed  # preview failure detaches it rather than killing a pump
        ready.set()
        await release.wait()
        return "ordinary result"

    job = manager.launch(work(), name="delivery_failure")
    await ready.wait()
    waiter = asyncio.ensure_future(job)
    await until(lambda: bool(job._waiters))
    release.set()
    assert await waiter == "ordinary result"
    assert job.progress_dropped == 1 and not job._progress_sinks


async def test_queued_resource_requests_cannot_mix_live_previews(managed, tmp_path):
    manager, channel = managed
    resource = manager.launch(Shell(tmp_path), name="serial")
    first_gate, second_gate = tmp_path / "first", tmp_path / "second"

    def command(gate, prefix):
        return f"printf {prefix}; while [ ! -f {shlex.quote(str(gate))} ]; do sleep .01; done"

    first = resource.send(command(first_gate, "ONE"))
    second = resource.send(command(second_gate, "TWO"))
    async with contextlib.aclosing(aiter(first)) as a, contextlib.aclosing(aiter(second)) as b:
        assert (await asyncio.wait_for(anext(a), 10)).text == "ONE"
        assert not b._buffer and second.status == "running"
        first_gate.touch()
        async for _ in a:
            pass
        assert (await first).stdout == "ONE"
        part = await asyncio.wait_for(anext(b), 10)
        assert part.text == "TWO" and part.sequence == 1
        second_gate.touch()
        async for _ in b:
            pass
        assert (await second).stdout == "TWO"
    assert not channel.snapshot()


async def test_delivered_generic_history_frees_progress_slots_and_cancel_counts_omissions(managed):
    manager, channel = managed
    ready, resume = asyncio.Event(), asyncio.Event()
    holder = []

    async def work():
        for number in range(3):
            emit(number)
        holder.append(_progress_sink())
        ready.set()
        await resume.wait()

    job = manager.launch(work(), name="history")
    await ready.wait()
    assert [event.value for event in channel.drain()] == [0, 1, 2]
    holder[0].emit(ShellProgress("stdout", "first", 1))
    holder[0].emit(ShellProgress("stdout", "second", 2))
    assert [event.value.text for event in channel.snapshot()] == ["first", "second"]
    assert job.progress_dropped == 0
    await job.cancel()
    assert job.progress_dropped == 2 and channel.snapshot() == []


async def test_terminal_reader_detach_counts_unread_real_shell_progress(managed, tmp_path):
    manager, channel = managed
    gate = tmp_path / "terminal-gate"
    command = f"printf '%8192s' x; while [ ! -f {shlex.quote(str(gate))} ]; do sleep .01; done; printf TAIL"
    async with Shell(tmp_path) as shell:
        job = manager.launch(shell.run(command), name="detach_after_terminal")
        reader = aiter(job)
        try:
            await until(lambda: bool(reader._buffer))
            assert job.status == "running" and not gate.exists()
            assert job.progress_dropped == 0
            gate.touch()
            await until(lambda: job.status != "running")
            assert job.status == "done", job.error
            # Deliberately neither consume the preview nor await the final result:
            # both would exercise already-covered paths rather than late detach.
            assert len(reader._buffer) == 2  # one 8192-byte preview and terminal marker
            await reader.aclose()
            assert job.progress_dropped == 1
            [final] = channel.snapshot()
            assert final.kind == "done" and final.id == job.id
            assert final.value.stdout == " " * 8191 + "xTAIL"
            assert (await job).stdout == final.value.stdout
            assert channel.snapshot() == []
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
        sink.emit(ShellProgress("stdout", "one", 1))
        emit("ordinary")
        sink.emit(ShellProgress("stderr", "two", 1))
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
        await reader.aclose()  # repeated close does not count twice
    assert job.progress_dropped == 2
    [ordinary, final] = channel.snapshot()
    assert ordinary.kind == "item" and ordinary.value == "ordinary"
    assert final.kind == ("failed" if failed else "done")
    if failed:
        with pytest.raises(ValueError, match="work failed"):
            await job
    else:
        assert await job == "final"
    assert [event.value for event in channel.snapshot()] == ["ordinary"]


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
    ready, release = asyncio.Event(), asyncio.Event()

    async def work():
        emit(control)
        sink = _progress_sink()
        for sequence, text in enumerate(("one", "two", "three"), 1):
            sink.emit(ShellProgress("stdout", text, sequence))
        ready.set()
        await release.wait()
        return "final"

    job = manager.launch(work(), name="identity_eviction")
    reader = aiter(job) if attached else None
    try:
        await ready.wait()
        assert compared == []
        assert job.progress_dropped == 1
        if reader is not None:
            assert await anext(reader) is control
            assert [(await anext(reader)).text for _ in range(2)] == ["two", "three"]
        else:
            ordinary, first, second = channel.drain()
            assert ordinary.value is control
            assert [first.value.text, second.value.text] == ["two", "three"]
        assert compared == []
        release.set()
        assert await job == "final"
    finally:
        release.set()
        if job.status == "running":
            await job.cancel()
        if reader is not None:
            await reader.aclose()
