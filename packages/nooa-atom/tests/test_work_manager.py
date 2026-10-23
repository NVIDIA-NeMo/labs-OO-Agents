# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``WorkManager``: generic jobs, with no shell or agent code involved."""

import ast
import asyncio
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
    await asyncio.sleep(0.05)
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
