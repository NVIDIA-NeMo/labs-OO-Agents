# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``JobsApi`` (``self.jobs``): background jobs delivered on the ``jobs`` channel."""

import asyncio
import os
import time
from pathlib import Path

import pytest

from nooa.runtime.channels import QueueManager
from nooa.runtime.jobs import JobResult, JobsApi, JobUpdate
from nooa.tools._results import StreamDone, StreamEvent
from nooa.tools.shell_tools import ShellResult, ShellTools

TIMEOUT = 15.0


class _Agent:
    """The two attributes JobsApi uses: a queue manager and (optionally) a shell."""

    def __init__(self, cwd: Path, shell: bool = True):
        self.queue_manager = QueueManager()
        self.queue_manager.queue("jobs")
        self.shell = ShellTools(cwd=str(cwd)) if shell else None
        self.cwd = cwd


@pytest.fixture
async def agent(tmp_path):
    agent = _Agent(tmp_path)
    agent.jobs = JobsApi(agent)
    yield agent
    await agent.queue_manager.shutdown()
    await agent.shell.close()


async def _next(agent) -> object:
    return await asyncio.wait_for(agent.queue_manager.get_channel("jobs").get(), TIMEOUT)


async def _until(predicate, timeout: float = TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.02)


async def test_run_delivers_a_labelled_shell_result_and_a_log(agent):
    info = await agent.jobs.run("echo out; echo err >&2; exit 3", label="t")
    assert info.state == "running" and info.kind == "run"
    item = await _next(agent)
    assert isinstance(item, JobResult) and item.label == "t"
    assert isinstance(item.result, ShellResult)
    assert item.result.returncode == 3 and item.result.stdout == "out\n"
    assert item.result.stderr == "err\n"
    log = Path(item.log_path).read_text()
    assert "out\n" in log and "err\n" in log
    assert agent.jobs.get("t").state == "done"
    assert agent.jobs.get("t").finished_at is not None


async def test_run_starts_in_the_shell_directory_with_its_exports(agent, tmp_path):
    (tmp_path / "sub").mkdir()
    await agent.shell.run("export JOBS_TEST_VAR=from-shell; cd sub")
    await agent.jobs.run('echo "$JOBS_TEST_VAR"; pwd; cd /; export X=1', label="env")
    item = await _next(agent)
    assert item.result.stdout.splitlines() == ["from-shell", str(tmp_path / "sub")]
    # The job's cd/export do not leak into the agent's shell.
    assert (await agent.shell.run("pwd; echo ${X:-unset}")).splitlines() == [
        str(tmp_path / "sub"),
        "unset",
    ]


async def test_run_takes_stdin_and_cwd(agent, tmp_path):
    (tmp_path / "d").mkdir()
    await agent.jobs.run("cat; pwd", label="in", stdin="hello\n", cwd="d")
    item = await _next(agent)
    assert item.result.stdout == f"hello\n{tmp_path / 'd'}\n"


async def test_run_does_not_block_the_shell(agent):
    await agent.jobs.run("sleep 3; echo late", label="slow")
    started = time.monotonic()
    assert (await agent.shell.run("echo now")).strip() == "now"
    assert time.monotonic() - started < 2.0
    assert agent.jobs.get("slow").state == "running"


async def test_run_timeout_kills_and_reports_124(agent):
    await agent.jobs.run("echo start; sleep 30", label="to", timeout=0.5)
    item = await _next(agent)
    assert item.result.timed_out and item.result.returncode == 124
    assert item.result.stdout == "start\n"


async def test_cancel_kills_the_process(agent, tmp_path):
    pidfile = tmp_path / "pid"
    await agent.jobs.run(f"echo $$ > {pidfile}; exec sleep 60", label="long")
    await _until(lambda: pidfile.exists() and pidfile.read_text().strip())
    pid = int(pidfile.read_text())
    assert await agent.jobs.cancel("long") is True
    assert agent.jobs.get("long").state == "cancelled"
    await _until(lambda: not _alive(pid))
    assert await agent.jobs.cancel("long") is False
    assert await agent.jobs.cancel("nope") is False
    assert agent.queue_manager.get_channel("jobs").qsize() == 0


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie is dead for our purposes.
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return True


async def test_labels_are_unique_among_running_jobs(agent):
    await agent.jobs.run("sleep 5", label="x")
    with pytest.raises(ValueError, match="already running"):
        await agent.jobs.run("true", label="x")
    with pytest.raises(ValueError, match="already running"):
        agent.jobs.spawn(asyncio.sleep(0), label="x")
    with pytest.raises(ValueError, match="label"):
        await agent.jobs.run("true", label=" ")
    await agent.jobs.cancel("x")
    await agent.jobs.run("echo again", label="x")  # free once it stopped
    item = await _next(agent)
    assert item.result.stdout == "again\n"
    assert [job.label for job in agent.jobs.list()] == ["x", "x"]


async def test_run_stream_delivers_lines_then_a_final_done(agent):
    await agent.jobs.run_stream("echo a; echo b >&2; echo c", label="s")
    events = []
    while True:
        update = await _next(agent)
        assert isinstance(update, JobUpdate) and update.label == "s"
        events.extend(update.items)
        if update.final:
            break
        assert len(update.items) == 1  # unthrottled: one item per delivery
    assert isinstance(events[-1], StreamDone) and events[-1].returncode == 0
    lines = [(e.kind, e.text) for e in events[:-1]]
    assert sorted(lines) == [("stderr", "b\n"), ("stdout", "a\n"), ("stdout", "c\n")]
    assert all(isinstance(e, StreamEvent) for e in events[:-1])
    assert "a\n" in Path(update.log_path).read_text()


async def test_throttle_by_lines_batches_and_flushes_the_rest(agent):
    await agent.jobs.run_stream("for i in 1 2 3 4 5; do echo $i; done", label="b", lines=2)
    updates = [await _next(agent)]
    while not updates[-1].final:
        updates.append(await _next(agent))
    texts = [[getattr(e, "text", "done") for e in u.items] for u in updates]
    assert texts == [["1\n", "2\n"], ["3\n", "4\n"], ["5\n", "done"]]


async def test_throttle_by_seconds_batches_slow_output(agent):
    async def ticks():
        for i in range(6):
            yield i
            await asyncio.sleep(0.1)

    agent.jobs.spawn(ticks(), label="tick", seconds=0.35)
    updates = [await _next(agent)]
    while not updates[-1].final:
        updates.append(await _next(agent))
    flat = [v for u in updates for v in u.items]
    assert flat == list(range(6))
    assert len(updates[0].items) >= 3  # several items per delivery
    assert len(updates) <= 3


async def test_throttle_changes_a_running_job(agent):
    gate = asyncio.Event()

    async def numbers():
        for i in range(3):
            yield i
        await gate.wait()
        for i in range(3, 9):
            yield i

    agent.jobs.spawn(numbers(), label="n")
    assert [(await _next(agent)).items for _ in range(3)] == [[0], [1], [2]]
    info = agent.jobs.throttle("n", lines=3)
    assert info.throttle.lines == 3
    gate.set()
    assert (await _next(agent)).items == [3, 4, 5]
    assert (await _next(agent)).items == [6, 7, 8]
    final = await _next(agent)
    assert final.final and final.items == []


async def test_throttle_off_flushes_pending_items(agent):
    gate = asyncio.Event()

    async def two_then_wait():
        yield "a"
        yield "b"
        await gate.wait()

    agent.jobs.spawn(two_then_wait(), label="p", lines=10)
    await _until(lambda: agent.jobs.get("p").pending == 2)
    assert agent.jobs.throttle("p").throttle is None
    # Turning batching off delivers what is pending as soon as more arrives or the job ends.
    gate.set()
    items = []
    while True:
        update = await _next(agent)
        items.extend(update.items)
        if update.final:
            break
    assert items == ["a", "b"]


async def test_throttle_rejects_non_streaming_jobs(agent):
    await agent.jobs.run("sleep 5", label="r")
    with pytest.raises(ValueError, match="not a streaming job"):
        agent.jobs.throttle("r", lines=2)
    with pytest.raises(KeyError):
        agent.jobs.throttle("missing")
    with pytest.raises(ValueError):
        await agent.jobs.run_stream("true", label="bad", lines=0)


async def test_spawn_a_coroutine_delivers_one_result_or_its_error(agent):
    async def compute():
        await asyncio.sleep(0.01)
        return 42

    async def broken():
        raise RuntimeError("boom")

    agent.jobs.spawn(compute(), label="c")
    item = await _next(agent)
    assert isinstance(item, JobResult) and item.label == "c" and item.result == 42
    agent.jobs.spawn(broken(), label="e")
    item = await _next(agent)
    assert item.error == "RuntimeError: boom"
    assert agent.jobs.get("e").state == "failed"
    with pytest.raises(TypeError):
        agent.jobs.spawn(42, label="z")


async def test_get_shows_recent_items(agent):
    agent.jobs.spawn(asyncio.sleep(0, result="v"), label="g")
    await _next(agent)
    info = agent.jobs.get("g")
    assert info.state == "done" and info.last[-1].result == "v"
    assert agent.jobs.get("unknown") is None


async def test_status_names_self_jobs_not_the_queue_manager(agent):
    await agent.jobs.run("sleep 5", label="build")
    status = agent.queue_manager.status()
    assert "build" in status and "self.jobs.cancel('build')" in status
    assert "queue_manager" not in status


async def test_a_delivered_job_makes_the_queue_ready(agent):
    """The turn loop waits with ``wait_ready``/``claim_batch``: a delivery wakes it."""
    queues = agent.queue_manager
    waiter = asyncio.ensure_future(queues.wait_ready())
    await asyncio.sleep(0.05)
    assert not waiter.done()
    await agent.jobs.run("echo hi", label="w")
    await asyncio.wait_for(waiter, TIMEOUT)
    batch = queues.claim_batch()
    assert batch is not None and [item.label for item in batch["jobs"]] == ["w"]


async def test_without_a_shell_jobs_use_the_process_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("JOBS_TEST_ENV", "proc")
    agent = _Agent(tmp_path, shell=False)
    jobs = JobsApi(agent)
    await jobs.run('echo "$JOBS_TEST_ENV"', label="np")
    item = await asyncio.wait_for(agent.queue_manager.get_channel("jobs").get(), TIMEOUT)
    assert item.result.stdout == "proc\n"
    await agent.queue_manager.shutdown()


async def test_each_throttle_call_replaces_the_setting_and_flushes_the_batch(agent):
    gates = [asyncio.Event(), asyncio.Event()]

    async def numbers():
        for i in range(4):
            yield i  # batched under seconds=60: nothing delivered yet
        await gates[0].wait()
        for i in range(4, 10):
            yield i
        await gates[1].wait()
        for i in range(10, 13):
            yield i

    agent.jobs.spawn(numbers(), label="r", seconds=60)
    await _until(lambda: agent.jobs.get("r").pending == 4)
    # Second setting: lines only; seconds is dropped, the old batch is delivered first.
    info = agent.jobs.throttle("r", lines=3)
    assert info.throttle.lines == 3 and info.throttle.seconds is None
    assert agent.jobs.get("r").throttle == info.throttle
    assert (await _next(agent)).items == [0, 1, 2, 3]
    gates[0].set()
    assert (await _next(agent)).items == [4, 5, 6]
    assert (await _next(agent)).items == [7, 8, 9]
    # Third setting: lines=2 replaces lines=3.
    assert agent.jobs.throttle("r", lines=2).throttle.lines == 2
    gates[1].set()
    assert (await _next(agent)).items == [10, 11]
    final = await _next(agent)
    assert final.final and final.items == [12]
