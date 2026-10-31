# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``WorkManager``: generic jobs, with no shell or agent code involved."""

import ast
import asyncio
import contextlib
import gc
import inspect
from pathlib import Path

import pytest
from nooa_atom.agent import work_manager
from nooa_atom.agent.work_manager import JobEvent, WorkManager, describe, emit, job_dir

from nooa.runtime.channels import QueueManager


@pytest.fixture
async def setup(tmp_path):
    channel = QueueManager().queue("jobs")
    manager = WorkManager(channel, scratch=tmp_path, max_items=5, max_requests=3)
    yield manager, channel
    await manager.aclose()


async def settle(n: int = 5) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


class Toy:
    """A toy resource: echoes inputs, records its lifecycle."""

    def __init__(self, fail_enter: bool = False, slow_enter: float = 0.0) -> None:
        self.log: list[str] = []
        self.fail_enter, self.slow_enter = fail_enter, slow_enter

    async def __aenter__(self):
        self.log.append("enter")
        await asyncio.sleep(self.slow_enter)
        if self.fail_enter:
            raise OSError("no device")
        return self

    async def __aexit__(self, *exc):
        self.log.append("exit")

    async def send(self, item):
        if item == "hang":
            try:
                await asyncio.sleep(3600)
            finally:
                self.log.append("hang cancelled")
        emit(f"got {item}")
        await asyncio.sleep(0.01)
        return f"echo {item}"


def test_manager_imports_no_work_kind():
    """Orthogonality: the manager imports no shell, session or agent code."""
    tree = ast.parse(Path(work_manager.__file__).read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not {m for m in imported if m and ("nooa" in m or "shell" in m or "session" in m)}


async def test_coroutine_result_error_and_cancel(setup):
    manager, channel = setup
    assert await manager.launch(asyncio.sleep(0, "v")) == "v"

    async def boom():
        raise ValueError("bad")

    with pytest.raises(ValueError, match="bad"):
        await manager.launch(boom(), name="boom")
    assert manager.get("boom").status == "failed"
    slow = manager.launch(asyncio.sleep(3600), name="slow")
    await slow.cancel()
    assert slow.status == "cancelled"
    with pytest.raises(RuntimeError, match="cancelled"):
        await slow
    assert channel.snapshot() == []  # awaited outcomes and owner-cancelled jobs never queue


async def test_timer_delivers_to_notification(setup):
    manager, channel = setup
    manager.launch(asyncio.sleep(0.01, "check CI"), name="ci")
    async with asyncio.timeout(5):  # wait for the observed event, not a fixed delay
        while not channel.snapshot():
            await asyncio.sleep(0.01)
    [event] = channel.snapshot()
    assert (event.name, event.kind, event.value) == ("ci", "done", "check CI")


async def test_finish_before_await_withdraws_the_event(setup):
    manager, channel = setup
    job = manager.launch(asyncio.sleep(0, 7), name="quick")
    await settle()
    assert [e.kind for e in channel.snapshot()] == ["done"]
    assert await job == 7
    assert channel.snapshot() == []
    assert await job == 7  # read back again; still nothing queued
    assert channel.snapshot() == []


async def test_claimed_completion_reads_back_without_duplicate(setup):
    manager, channel = setup
    job = manager.launch(asyncio.sleep(0, "r"), name="claimed")
    await settle()
    claimed = channel.drain()  # a turn took it
    assert [e.value for e in claimed] == ["r"]
    assert await job == "r"
    assert channel.snapshot() == []


async def test_cancelled_waiter_requeues_once(setup):
    manager, channel = setup
    gate = asyncio.Event()

    async def work():
        await gate.wait()
        return "late"

    job = manager.launch(work(), name="w")
    waiter = asyncio.create_task(job._wait())
    await settle()
    job._finish("done", "late")  # the outcome is handed to the waiter ...
    waiter.cancel()  # ... whose task is cancelled before it resumes
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert [(e.name, e.value) for e in channel.snapshot()] == [("w", "late")]
    assert await job == "late"  # withdrawn: exactly one receiver
    assert channel.snapshot() == []
    gate.set()
    await settle()
    assert channel.snapshot() == []  # the task's own end does not deliver again


async def test_cancelled_waiter_before_finish_leaves_the_channel(setup):
    manager, channel = setup
    gate = asyncio.Event()

    async def work():
        await gate.wait()
        return 1

    job = manager.launch(work(), name="w2")
    waiter = asyncio.create_task(job._wait())
    await settle()
    waiter.cancel()
    await asyncio.gather(waiter, return_exceptions=True)
    gate.set()
    await settle()
    assert [e.value for e in channel.snapshot()] == [1]


async def test_two_waiters_one_cancelled_no_requeue(setup):
    manager, channel = setup
    gate = asyncio.Event()

    async def work():
        await gate.wait()
        return 2

    job = manager.launch(work())
    a, b = asyncio.create_task(job._wait()), asyncio.create_task(job._wait())
    await settle()
    job._finish("done", 2)
    a.cancel()
    assert await b == 2
    await asyncio.gather(a, return_exceptions=True)
    assert channel.snapshot() == []


async def test_finite_stream_and_reader(setup):
    manager, channel = setup

    async def count(n):
        for i in range(n):
            yield i

    job = manager.launch(count(3), name="count")
    assert [i async for i in job] == [0, 1, 2]
    assert channel.snapshot() == []  # the reader took the items and the end
    queued = manager.launch(count(2), name="queued")
    await settle(10)
    assert [(e.kind, e.value) for e in channel.snapshot()] == [
        ("item", 0),
        ("item", 1),
        ("done", None),
    ]
    assert await queued is None


async def test_reader_early_exit_detaches_and_stream_continues(setup):
    manager, channel = setup
    step = asyncio.Event()

    async def ticks():
        for n in (1, 2, 3):
            yield n
        await step.wait()
        yield 4
        await asyncio.Event().wait()

    job = manager.launch(ticks(), name="ticks")
    async for n in job:
        if n == 2:
            break
    await settle()
    assert job._reader is None
    step.set()
    await settle(10)
    values = [e.value for e in channel.snapshot() if e.kind == "item"]
    assert values == [3, 4]  # unread (buffered) and later items reach the channel
    await job.cancel()


async def test_bounded_items_drop_oldest_keep_final(setup):
    manager, channel = setup

    async def many():
        for i in range(12):
            yield i

    job = manager.launch(many(), name="many")
    await settle(30)
    events = channel.snapshot()
    items = [e for e in events if e.kind == "item"]
    assert len(items) == 5  # max_items
    assert [e.value for e in items] == [7, 8, 9, 10, 11]
    assert sum(e.dropped for e in events) == 7
    assert events[-1].kind == "done"  # the terminal event is never dropped
    assert await job is None


async def test_file_watch_sees_appended_lines(setup, tmp_path):
    manager, _ = setup
    log = tmp_path / "server.log"
    log.write_text("old\n")

    async def follow(path):
        with open(path) as handle:
            handle.seek(0, 2)
            while True:
                line = handle.readline()
                if line:
                    yield line.rstrip("\n")
                else:
                    await asyncio.sleep(0.01)

    job = manager.launch(follow(log), name="log")
    seen = []

    async def append():
        await asyncio.sleep(0.05)
        with open(log, "a") as handle:
            handle.write("one\nREADY\n")

    writer = asyncio.create_task(append())
    async with asyncio.timeout(5):
        async for line in job:
            seen.append(line)
            if line == "READY":
                break
    await writer
    await job.cancel()
    assert seen == ["one", "READY"]


async def test_resource_requests_and_lifecycle(setup):
    manager, channel = setup
    toy = Toy()
    res = manager.launch(toy, name="toy")
    with pytest.raises(TypeError, match="resource"):
        await res
    first, second = res.send("a"), res.send("b")
    assert (first.name, second.name) == ("toy#1", "toy#2")  # not valid user names
    assert await first == "echo a" and await second == "echo b"
    assert [e.value for e in channel.snapshot()] == ["got a", "got b"]  # emitted items
    await res.cancel()
    assert toy.log == ["enter", "exit"]
    with pytest.raises(RuntimeError):
        res.send("c")


async def test_message_reaches_an_optional_resource_message_without_a_job(setup):
    manager, channel = setup

    class Noted(Toy):
        def __init__(self) -> None:
            super().__init__(slow_enter=0.02)
            self.notes: list[str] = []

        async def message(self, item):
            self.notes.append(item)

    noted = Noted()
    res = manager.launch(noted, name="noted")
    await res.message("n1")  # waits for entry, then hands it over
    assert noted.notes == ["n1"] and [j.name for j in manager.list()] == ["noted"]
    assert channel.qsize() == 0  # no request job, no event
    plain = manager.launch(Toy(), name="plain")
    with pytest.raises(TypeError, match="has no message"):
        await plain.message("x")
    with pytest.raises(TypeError, match="only a launched resource"):
        await manager.launch(asyncio.sleep(0), name="nap").message("x")
    await res.cancel()
    with pytest.raises(RuntimeError, match="noted is"):
        await res.message("late")


async def test_request_bound_and_cancel_keeps_resource(setup):
    manager, _ = setup
    toy = Toy()
    res = manager.launch(toy, name="toy")
    hang = res.send("hang")
    res.send("x"), res.send("y")
    with pytest.raises(RuntimeError, match="in flight"):
        res.send("z")
    await settle()
    await hang.cancel()
    assert "hang cancelled" in toy.log
    assert res.status == "running"
    assert await res.send("after") == "echo after"


async def test_cancel_resource_cancels_requests_before_exit(setup):
    manager, _ = setup
    toy = Toy()
    res = manager.launch(toy, name="toy")
    hang = res.send("hang")
    await settle()
    await res.cancel()
    assert toy.log == ["enter", "hang cancelled", "exit"]
    assert hang.status == "cancelled"


async def test_resource_entry_failure_fails_requests(setup):
    manager, channel = setup
    res = manager.launch(Toy(fail_enter=True), name="bad")
    request = res.send("a")
    with pytest.raises(RuntimeError, match="did not start: OSError: no device"):
        await asyncio.wait_for(request._wait(), 2)
    assert res.status == "failed"
    assert [(e.name, e.kind) for e in channel.snapshot()] == [("bad", "failed")]


async def test_cancel_resource_before_ready(setup):
    manager, _ = setup
    toy = Toy(slow_enter=3600)
    res = manager.launch(toy, name="slow")
    request = res.send("a")
    await settle()
    await asyncio.wait_for(res.cancel(), 2)
    assert res.status == "cancelled"
    assert request.status == "cancelled"
    assert "exit" not in toy.log  # never entered, never exited


async def test_old_generation_isolated(setup):
    manager, channel = setup
    gate = asyncio.Event()

    async def work(value):
        await gate.wait()
        return value

    old = manager.launch(work("old"), name="tests")
    await old.cancel()
    new = manager.launch(work("new"), name="tests")
    assert manager.get("tests") is new and new.id != old.id
    gate.set()
    assert await new == "new"
    with pytest.raises(RuntimeError, match="cancelled"):
        await old
    assert channel.snapshot() == []
    blocker = manager.launch(asyncio.sleep(3600), name="busy")
    with pytest.raises(ValueError, match="running"):
        manager.launch(asyncio.sleep(0), name="busy")
    await blocker.cancel()


async def test_name_validation_and_bad_work(setup):
    manager, _ = setup
    with pytest.raises(ValueError):
        manager.launch(asyncio.sleep(0), name="sh#1")
    with pytest.raises(TypeError, match="pass"):
        manager.launch(asyncio.sleep)  # type: ignore[arg-type]

    async def coro():
        return 1

    with pytest.raises(TypeError, match=r"coro\(\)"):
        manager.launch(coro)
    with pytest.raises(TypeError):
        manager.launch(42)


async def test_launch_errors_name_the_fix(setup):
    manager, _ = setup
    work = asyncio.sleep(0)
    with pytest.raises(TypeError, match=r"takes no timeout.*sh\.run\(cmd, timeout=600\)"):
        manager.launch(work, name="tests", timeout=600)  # type: ignore[call-arg]
    assert work.cr_frame is None  # closed, no "never awaited" warning
    with pytest.raises(TypeError, match="unexpected keyword argument.*: cwd"):
        manager.launch(asyncio.sleep(0), cwd="/")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="'run_tests'"):
        manager.launch(asyncio.sleep(0), name="run tests")
    with pytest.raises(ValueError, match="'run-tests'"):
        manager.launch(asyncio.sleep(0), name="run-tests.")  # hint keeps the hyphen
    with pytest.raises(ValueError, match="'job_2nd'"):
        manager.launch(asyncio.sleep(0), name="2nd")
    # The catch-all only improves the error: the documented signature has no **kwargs.
    assert "unexpected" not in str(inspect.signature(WorkManager.launch))


async def test_scratch_retention_and_owned_cleanup(tmp_path):
    channel = QueueManager().queue("jobs")
    manager = WorkManager(channel, scratch=tmp_path, keep=2)
    unrelated = tmp_path / "user.log"
    unrelated.write_text("keep me")

    async def write(n):
        directory = job_dir()
        (directory / "out.txt").write_text(str(n))
        return directory

    first = await manager.launch(write(1))
    assert (first / "out.txt").read_text() == "1"  # kept after the result was consumed
    second = await manager.launch(write(2))
    third = await manager.launch(write(3))
    assert not first.exists()  # evicted with its job (keep=2)
    assert second.exists() and third.exists()
    # A swapped directory is not removed: replace second's dir with a symlink elsewhere.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "precious").write_text("x")
    for child in second.iterdir():
        child.unlink()
    second.rmdir()
    second.symlink_to(elsewhere)
    await manager.aclose()
    assert (elsewhere / "precious").exists()
    assert unrelated.read_text() == "keep me"
    assert not third.exists()
    roots = [p for p in tmp_path.iterdir() if p.name.startswith("nooa-work-")]
    assert (
        roots == []
    )  # the manager's own root is gone; the swapped path inside it was not followed


async def test_records_keep_launch_info_and_the_latest_summary(tmp_path):
    saved: list[list[dict]] = []
    manager = WorkManager(
        QueueManager().queue("jobs"),
        scratch=tmp_path,
        launch_info=lambda work, kind: {"tag": "12", "what": kind, "empty": "", "long": "x" * 500},
    )
    manager._on_change = lambda: saved.append(manager.records())
    started = asyncio.Event()

    async def work() -> None:
        describe("$ pytest -q · cwd /repo")
        started.set()
        await asyncio.sleep(3600)

    job = manager.launch(work(), name="tests")
    await started.wait()
    [record] = saved[-1]  # describe() saved the summary at once
    assert record == {
        "id": 1,
        "name": "tests",
        "kind": "await",
        "tag": "12",
        "what": "await",
        "long": "x" * 79 + "…",
        "summary": "$ pytest -q · cwd /repo",
    }
    await job.cancel()
    await manager.aclose()


async def test_aclose_cancels_everything(tmp_path):
    channel = QueueManager().queue("jobs")
    manager = WorkManager(channel, scratch=tmp_path)
    toy = Toy()
    res = manager.launch(toy, name="toy")
    request = res.send("hang")
    task = manager.launch(asyncio.sleep(3600))
    await settle()
    assert len(manager.records()) == 3
    await manager.aclose()
    assert toy.log == ["enter", "hang cancelled", "exit"]
    assert request.status == task.status == "cancelled"
    assert manager.running() == []
    with pytest.raises(RuntimeError, match="closed"):
        manager.launch(asyncio.sleep(0))
    assert channel.snapshot() == []
    assert isinstance(JobEvent("a", 1, "done"), JobEvent)


async def test_summary_is_bounded_escaped_and_never_raises(setup):
    manager, _ = setup

    class Noisy(Toy):
        def summary(self):
            return "line1\nline2\x1b[31m" + "x" * 500

    class Broken(Toy):
        def summary(self):
            raise OSError("boom")

    noisy = manager.launch(Noisy(), name="noisy")
    broken = manager.launch(Broken(), name="broken")
    await settle()
    text = noisy.summary
    assert "\n" not in text and "\x1b" not in text and "\\n" in text and "\\x1b" in text
    assert len(text) <= 160 and text.endswith("…")
    assert broken.summary == "(summary failed: OSError)"

    async def work():
        describe("step\t2")
        await asyncio.sleep(3600)

    job = manager.launch(work(), name="described")
    await settle()
    assert job.summary == "step\\t2" and "step" in repr(job)
    await job.cancel()


# ---- hardening: consumption modes, explicit reader lifecycle, cancel, bounds, retention ----


async def test_reader_attached_late_takes_unread_items_in_order(setup):
    manager, channel = setup
    gate = asyncio.Event()

    async def stream():
        for i in range(4):
            yield i
        await gate.wait()
        yield 4

    job = manager.launch(stream(), name="late")
    await settle(10)
    taken = channel.get_nowait() if hasattr(channel, "get_nowait") else None
    # A turn claimed the first item (claim_batch pops from the head).
    first = channel._items.popleft() if taken is None else taken
    assert first.value == 0
    reader = aiter(job)
    assert channel.snapshot() == []  # the rest moved to the reader
    gate.set()
    assert [x async for x in reader] == [1, 2, 3, 4]
    assert channel.snapshot() == []


async def test_reader_attached_after_finish_gets_items_and_end(setup):
    manager, channel = setup

    async def stream():
        yield "a"
        yield "b"

    async def boom():
        yield "x"
        raise ValueError("late failure")

    ok = manager.launch(stream(), name="ok")
    bad = manager.launch(boom(), name="bad")
    await settle(10)
    assert [x async for x in ok] == ["a", "b"]
    got = []
    with pytest.raises(ValueError, match="late failure"):
        async for x in bad:
            got.append(x)
    assert got == ["x"]
    assert channel.snapshot() == []


async def test_awaiter_and_reader_together_both_end(setup):
    manager, channel = setup
    gate = asyncio.Event()

    async def stream():
        yield 1
        await gate.wait()
        yield 2

    job = manager.launch(stream(), name="both")
    reader = asyncio.create_task(_collect(job))
    waiter = asyncio.create_task(job._wait())
    await settle()
    gate.set()
    assert await asyncio.wait_for(reader, 2) == [1, 2]
    assert await asyncio.wait_for(waiter, 2) is None
    assert channel.snapshot() == []


async def _collect(job):
    return [x async for x in job]


async def test_reader_retained_after_break_resumes_then_aclose_detaches(setup):
    manager, channel = setup

    async def stream():
        for i in range(8):
            yield i

    job = manager.launch(stream(), name="kept")
    reader = aiter(job)
    async for x in reader:
        if x == 1:
            break
    assert job._reader is reader  # still referenced: still attached
    assert await anext(reader) == 2  # nothing lost while it was held
    await reader.aclose()  # explicit: no reliance on garbage collection
    assert job._reader is None
    await settle(20)
    values = [e.value for e in channel.snapshot() if e.kind == "item"]
    assert values == [3, 4, 5, 6, 7] and channel.snapshot()[-1].kind == "done"


async def test_aclosing_detaches_deterministically(setup):
    manager, channel = setup

    async def stream():
        for i in range(3):
            yield i

    job = manager.launch(stream(), name="ctx")
    gc.disable()
    try:
        async with contextlib.aclosing(aiter(job)) as items:
            first = await anext(items)
        assert job._reader is None
    finally:
        gc.enable()
    assert first == 0
    await settle(10)
    assert [e.value for e in channel.snapshot()] == [1, 2, None]


async def test_cancel_while_backpressured(setup):
    manager, channel = setup  # max_items=5

    async def forever():
        n = 0
        while True:
            n += 1
            yield n

    job = manager.launch(forever(), name="fast")
    reader = aiter(job)
    await settle(20)
    assert len(reader._buffer) == 5  # the producer waits for the reader
    await asyncio.wait_for(job.cancel(), 2)  # promptly, though backpressured
    assert [await anext(reader) for _ in range(5)] == [1, 2, 3, 4, 5]
    with pytest.raises(RuntimeError, match="cancelled"):
        await anext(reader)
    assert channel.snapshot() == []


async def test_cancelled_pending_anext_takes_nothing(setup):
    manager, _ = setup
    gate = asyncio.Event()

    async def stream():
        yield 1
        await gate.wait()
        yield 2

    job = manager.launch(stream(), name="gated")
    reader = aiter(job)
    assert await anext(reader) == 1
    pending = asyncio.create_task(anext(reader))
    await settle()
    assert not pending.done()
    pending.cancel()
    await asyncio.gather(pending, return_exceptions=True)
    gate.set()
    assert await anext(reader) == 2
    with pytest.raises(StopAsyncIteration):
        await anext(reader)


async def test_emit_into_full_reader_counts_loss(setup):
    manager, _ = setup
    gate = asyncio.Event()

    async def work():
        for i in range(8):
            emit(i)
        await gate.wait()
        return "ok"

    job = manager.launch(work(), name="emitter")
    reader = aiter(job)
    await settle()
    gate.set()
    assert [x async for x in reader] == [3, 4, 5, 6, 7]
    assert job.dropped == 3


async def test_queue_loss_counted_on_job(setup):
    manager, _ = setup

    async def many():
        for i in range(12):
            yield i

    job = manager.launch(many(), name="lossy")
    await job
    assert job.dropped == 7


async def test_owner_cancel_withdraws_only_its_generation(setup):
    manager, channel = setup
    gate = asyncio.Event()

    async def stream(tag):
        yield f"{tag}-1"
        await gate.wait()

    old = manager.launch(stream("old"), name="gen")
    await settle()
    await old.cancel()
    new = manager.launch(stream("new"), name="gen")
    other = manager.launch(stream("other"), name="other")
    await settle()
    await new.cancel()
    assert [e.value for e in channel.snapshot()] == ["other-1"]
    assert (old.status, new.status) == ("cancelled", "cancelled")
    await other.cancel()
    assert channel.snapshot() == []


async def test_resource_cancel_status_is_cancelled(setup):
    manager, channel = setup
    res = manager.launch(Toy(), name="toy")
    await res.send("a")
    channel._items.clear()
    await res.cancel()
    assert res.status == "cancelled"
    assert channel.snapshot() == []


@pytest.mark.parametrize("option", ["keep", "max_items", "max_requests"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
def test_bounds_are_validated(option, value):
    with pytest.raises(ValueError, match=option):
        WorkManager(QueueManager().queue("jobs"), **{option: value})


async def test_eviction_keeps_output_of_unread_notifications(tmp_path):
    channel = QueueManager().queue("jobs")
    manager = WorkManager(channel, scratch=tmp_path, keep=2)

    async def write():
        directory = job_dir()
        (directory / "out").write_text("x")
        return directory

    jobs = [manager.launch(write(), name=f"j{i}") for i in range(4)]
    await settle(10)
    dirs = [job.result for job in jobs]
    assert all(d.exists() for d in dirs)  # 4 > keep, but every event is unread
    channel._items.clear()  # a turn took them all
    extra = await manager.launch(write(), name="extra")
    assert [d.exists() for d in dirs] == [False, False, False, True]
    assert extra.exists()
    # The hard bound: at most 2 * keep finished jobs even if nothing is read.
    for i in range(6):
        manager.launch(write(), name=f"k{i}")
    await settle(20)
    assert len(manager.list()) == 4
    await manager.aclose()


# ---- terminal-state races (inspired by Codex's cell_actor tests) ----


class FlakyOutbox:
    """An outbox whose next put fails once."""

    def __init__(self):
        self.items, self.fail = [], 0

    def put(self, item):
        if self.fail:
            self.fail -= 1
            raise OSError("outbox unavailable")
        self.items.append(item)

    def remove(self, item):
        for i, queued in enumerate(self.items):
            if queued is item:
                del self.items[i]
                return True
        return False


async def test_failed_completion_delivery_leaves_result_readable(tmp_path):
    outbox = FlakyOutbox()
    manager = WorkManager(outbox, scratch=tmp_path)
    outbox.fail = 1
    job = manager.launch(asyncio.sleep(0, "kept"), name="flaky")
    await settle()
    assert outbox.items == [] and job._queued_final is None  # not falsely marked queued
    assert await job == "kept"  # the stored result is read back
    await manager.aclose()


async def test_dropped_reader_preserves_items_for_the_next_reader(setup):
    manager, channel = setup
    gate = asyncio.Event()

    async def stream():
        for i in range(3):
            yield i
        await gate.wait()
        yield 3

    job = manager.launch(stream(), name="resume")
    first = aiter(job)
    assert await anext(first) == 0
    await settle()
    await first.aclose()  # 1 and 2 were buffered, unread: back to the outbox
    assert [e.value for e in channel.snapshot()] == [1, 2]
    second = aiter(job)  # the next reader takes them back, in order
    gate.set()
    assert [x async for x in second] == [1, 2, 3]
    assert channel.snapshot() == []


async def test_only_one_await_claims_a_queued_completion(setup):
    manager, channel = setup
    job = manager.launch(asyncio.sleep(0, "once"), name="once")
    await settle()
    assert len(channel.snapshot()) == 1
    results = await asyncio.gather(job._wait(), job._wait(), job._wait())
    assert results == ["once"] * 3  # one withdraw, the others read back
    assert channel.snapshot() == []


async def test_natural_completion_wins_over_later_cancel(setup):
    manager, channel = setup
    job = manager.launch(asyncio.sleep(0, "first"), name="won")
    await settle()
    await job.cancel()
    assert job.status == "done" and job.result == "first"
    assert [e.kind for e in channel.snapshot()] == ["done"]  # its event is not withdrawn
    assert await job == "first"


async def test_second_concurrent_reader_is_refused(setup):
    manager, _ = setup
    job = manager.launch(asyncio.sleep(3600), name="busy_reader")
    reader = aiter(job)
    with pytest.raises(RuntimeError, match="already has a reader"):
        aiter(job)
    await reader.aclose()
    aiter(job)  # free again once the first detached
    await job.cancel()


async def test_work_that_swallows_cancel_ends_done_without_notice(setup):
    manager, channel = setup

    async def stubborn():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            return "finished anyway"

    job = manager.launch(stubborn(), name="stubborn")
    await settle()
    await job.cancel()
    assert job.status == "done" and await job == "finished anyway"
    assert channel.snapshot() == []  # cancelled by its owner: no event


# ---- built-in work: after / every -------------------------------------------------


async def _event(channel, timeout: float = 5.0):
    return await asyncio.wait_for(channel.get(), timeout)


async def test_after_delivers_done_after_the_delay(setup):
    manager, channel = setup
    loop = asyncio.get_running_loop()
    start = loop.time()
    job = manager.after(0.2, name="reading")
    assert job.name == "reading" and job.status == "running"
    assert "timer: 0.2 s" in job.summary
    event = await _event(channel)
    elapsed = loop.time() - start
    assert (event.name, event.kind, event.value) == ("reading", "done", None)
    assert 0.18 <= elapsed < 2.0
    assert job.status == "done"


async def test_after_can_be_awaited_and_cancelled(setup):
    manager, channel = setup
    assert await manager.after(0) is None
    job = manager.after(30, name="later")
    await job.cancel()
    assert job.status == "cancelled"
    await settle()
    assert channel.qsize() == 0  # the owner cancelled it: no event


async def test_timers_follow_the_launch_name_rules(setup):
    manager, _ = setup
    with pytest.raises(ValueError, match="not valid"):
        manager.after(1, name="bad name")
    manager.after(30, name="t")
    with pytest.raises(ValueError, match="is running"):
        manager.every(1, name="t")
    for bad in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            manager.after(bad)
    with pytest.raises(ValueError):
        manager.every(0)
    with pytest.raises(TypeError):
        manager.after("120")
    with pytest.raises(TypeError):
        manager.every(True)


async def test_every_ticks_until_cancelled(setup):
    manager, channel = setup
    job = manager.every(0.05, name="poll")
    first = await _event(channel)
    second = await _event(channel)
    assert (first.kind, first.value, second.value) == ("item", 1, 2)
    await job.cancel()
    await settle()
    assert job.status == "cancelled" and channel.qsize() == 0


async def test_unread_ticks_merge_into_one(setup):
    manager, channel = setup
    job = manager.every(0.02, name="poll")
    await asyncio.sleep(0.3)  # about 15 ticks, nobody reading
    assert channel.qsize() == 1
    event = await _event(channel)
    assert event.value >= 5 and event.dropped == event.value - 1
    await job.cancel()


async def test_every_late_wake_up_yields_the_latest_due_tick(monkeypatch):
    # The first sleep wakes 0.35 s in although only 0.1 s was asked for:
    # three ticks are due by then, so the timer says 3, not 1.
    clock = [100.0]

    async def late_sleep(seconds):
        clock[0] += 0.35 if clock[0] == 100.0 else seconds

    monkeypatch.setattr(work_manager.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(work_manager.asyncio, "sleep", late_sleep)
    ticks = work_manager._every(0.1)
    assert [await anext(ticks), await anext(ticks)] == [3, 4]
    await ticks.aclose()


async def test_every_can_be_read_with_async_for(setup):
    manager, _ = setup
    job = manager.every(0.01)
    seen = []
    async for tick in job:
        seen.append(tick)
        if len(seen) == 3:
            break
    assert seen == [1, 2, 3]
    await job.cancel()
