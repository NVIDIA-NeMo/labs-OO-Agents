# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private preview batching and completion-only managed Shells; no provider."""

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


async def past_old_preview_deadline(gate):
    # The subprocess has written both streams. Keep it alive beyond the former
    # 1s timer as well as the 8192-byte boundary, without relying on its exit.
    await until(gate.with_suffix(".ready").exists)
    await asyncio.sleep(1.1)
    assert not gate.exists()


def assert_full_capture(result, *, out=NOISY_OUT + "TAIL", err=NOISY_ERR):
    assert result.returncode == 0 and not result.timed_out
    assert "bytes omitted" in result.stdout and "bytes omitted" in result.stderr
    assert Path(result.stdout_path).read_text() == out
    assert Path(result.stderr_path).read_text() == err


async def test_real_process_iteration_has_no_items_before_or_after_exit(managed, tmp_path):
    manager, channel = managed
    gate, command = gated_command(tmp_path)
    async with Shell(tmp_path) as shell:
        job = manager.launch(shell.run(command), name="observe")
        async with contextlib.aclosing(aiter(job)) as reader:
            pending = asyncio.create_task(anext(reader))
            try:
                await past_old_preview_deadline(gate)
                assert not pending.done() and job.status == "running"
                assert channel.snapshot() == [] and not job._progress_sinks
                gate.touch()
                with pytest.raises(StopAsyncIteration):
                    await asyncio.wait_for(pending, 10)
            finally:
                gate.touch()
                if not pending.done():
                    pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
        result = await job
    assert_full_capture(result)
    assert result.cwd == str(tmp_path)
    assert channel.snapshot() == [] and job.progress_dropped == 0


async def test_detached_shell_notifies_once_only_after_exit(managed, tmp_path):
    manager, channel = managed
    gate, command = gated_command(tmp_path)
    async with Shell(tmp_path) as shell:
        job = manager.launch(shell.run(command), name="later")
        try:
            await past_old_preview_deadline(gate)
            assert channel.snapshot() == [] and job.status == "running"
            gate.touch()
            await until(lambda: job.status == "done")
            [final] = channel.drain()
            assert (final.kind, final.name, final.id) == ("done", job.name, job.id)
            assert_full_capture(final.value)
            assert await job is final.value
            await asyncio.sleep(1.1)
            assert channel.snapshot() == [] and job.progress_dropped == 0
        finally:
            gate.touch()


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
            await past_old_preview_deadline(gate)
            assert [event.value for event in channel.snapshot()] == [
                "ordinary child/control message"
            ]
            assert job.status == "running" and not waiter.done()
            assert job.progress_dropped == 0 and not job._progress_sinks
            gate.touch()
            assert_full_capture(await asyncio.wait_for(waiter, 10))
            assert [event.value for event in channel.snapshot()] == [
                "ordinary child/control message"
            ]
        finally:
            gate.touch()
            await asyncio.wait_for(asyncio.gather(waiter, return_exceptions=True), 10)


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


async def test_timeout_and_lingering_background_writer_are_completion_only(managed, tmp_path):
    manager, channel = managed
    gate, command = gated_command(tmp_path)
    async with Shell(tmp_path) as shell:
        job = manager.launch(shell.run(command, timeout=3), name="timeout")
        await past_old_preview_deadline(gate)
        assert channel.snapshot() == [] and job.status == "running"
        await until(lambda: job.status == "done")
        [final] = channel.drain()
        assert final.kind == "done" and final.id == job.id
        result = await job
        assert result is final.value and result.timed_out and result.returncode == 124
        assert Path(result.stdout_path).read_text() == NOISY_OUT
        assert Path(result.stderr_path).read_text().startswith(NOISY_ERR)
        assert not gate.exists() and not job._progress_sinks and job.progress_dropped == 0
        # The command ends while a writer keeps its FIFO open past the old deadline.
        lingering = manager.launch(
            shell.run("printf initial; (sleep 1.2; printf late; sleep .2) &"), name="background"
        )
        await until(lambda: lingering.status == "done")
        assert [event.kind for event in channel.drain()] == ["done"]
        assert (await lingering).stdout == "initial"
        await asyncio.sleep(1.5)
        assert channel.snapshot() == [] and not lingering._progress_sinks


@pytest.mark.parametrize("owner_cancel", [False, True], ids=["task", "owner"])
async def test_cancel_noisy_shell_never_emits_output_items(managed, tmp_path, owner_cancel):
    manager, channel = managed
    gate, command = gated_command(tmp_path)
    async with Shell(tmp_path) as shell:
        job = manager.launch(shell.run(command), name="cancel_shell")
        await past_old_preview_deadline(gate)
        assert channel.snapshot() == [] and job.status == "running"
        if owner_cancel:
            await job.cancel()  # owner cancellation withdraws its own terminal outcome
            assert channel.snapshot() == []
        else:
            job._task.cancel()
            await until(lambda: job.status == "cancelled")
            [final] = channel.drain()
            assert final.kind == "cancelled" and final.id == job.id
        assert job.status == "cancelled" and not gate.exists()
        assert job.progress_dropped == 0 and not job._progress_sinks
        assert (await manager.launch(shell.run("printf next"))).stdout == "next"
        await asyncio.sleep(1.1)
        assert channel.snapshot() == []


async def test_resource_requests_keep_completion_and_capture_limits_separate(managed, tmp_path):
    manager, channel = managed
    gate, command = gated_command(tmp_path, prefix="a" * 20000, tail="z")
    resource = manager.launch(Shell(tmp_path, max_output=12000), name="resource")
    request = resource.send(command)
    async with contextlib.aclosing(aiter(request)) as reader:
        try:
            await past_old_preview_deadline(gate)
            assert not reader._buffer and request.status == "running"
            assert channel.snapshot() == [] and not request._progress_sinks
            gate.touch()
            assert [item async for item in reader] == []
        finally:
            gate.touch()
    result = await request
    assert result.returncode == 0
    assert "output limit" in result.stdout and "output limit" in result.stderr
    assert Path(result.stdout_path).read_text() == "a" * 12000
    assert Path(result.stderr_path).read_text() == "E" * 12000
    assert result.stdout.endswith("z")  # summary tail survives the on-disk capture cap
    second = resource.send("printf next")
    assert (await second).stdout == "next"
    assert second.id != request.id and not second._progress_sinks
    assert request.progress_dropped == 0 and not channel.snapshot()


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


async def test_queued_resource_requests_complete_in_order_without_items(managed, tmp_path):
    manager, channel = managed
    resource = manager.launch(Shell(tmp_path), name="serial")
    first_gate, first_command = gated_command(tmp_path, name="first")
    second_gate, second_command = gated_command(tmp_path, name="second")
    first = resource.send(first_command)
    second = resource.send(second_command)
    async with contextlib.aclosing(aiter(first)) as a, contextlib.aclosing(aiter(second)) as b:
        try:
            await past_old_preview_deadline(first_gate)
            assert not a._buffer and not b._buffer and second.status == "running"
            assert not second_gate.with_suffix(".ready").exists()
            assert channel.snapshot() == []
            first_gate.touch()
            assert [item async for item in a] == []
            assert_full_capture(await first)
            await past_old_preview_deadline(second_gate)
            assert not b._buffer and channel.snapshot() == []
            second_gate.touch()
            assert [item async for item in b] == []
            assert_full_capture(await second)
        finally:
            first_gate.touch()
            second_gate.touch()
    assert first.progress_dropped == second.progress_dropped == 0
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


async def test_terminal_reader_detach_delivers_only_shell_result(managed, tmp_path):
    manager, channel = managed
    gate, command = gated_command(tmp_path)
    async with Shell(tmp_path) as shell:
        job = manager.launch(shell.run(command), name="detach_after_terminal")
        reader = aiter(job)
        try:
            await past_old_preview_deadline(gate)
            assert job.status == "running" and not reader._buffer
            assert channel.snapshot() == []
            gate.touch()
            await until(lambda: job.status != "running")
            assert job.status == "done", job.error
            # Deliberately neither consume the iterator nor await the final result.
            assert len(reader._buffer) == 1  # only the terminal marker
            await reader.aclose()
            assert job.progress_dropped == 0
            [final] = channel.snapshot()
            assert final.kind == "done" and final.id == job.id
            assert_full_capture(final.value)
            assert await job is final.value
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
