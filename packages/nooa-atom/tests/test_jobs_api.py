# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``JobsApi`` (``self.jobs``): background jobs delivered on the ``jobs`` channel."""

import asyncio
import contextlib
import datetime
import gc
import os
import re
import time
import tracemalloc
import warnings
from pathlib import Path

import pytest
from nooa_atom.agent.jobs import JobsApi, JobUpdate

from nooa.runtime.channels import QueueManager
from nooa.tools._results import StreamEvent
from nooa.tools.shell_tools import ShellResult, ShellTools

TIMEOUT = 15.0


class _Agent:
    """The two attributes JobsApi uses: a queue manager and (optionally) a shell."""

    def __init__(self, cwd: Path, shell: bool = True):
        self.queue_manager = QueueManager()
        self.queue_manager.queue("jobs")
        self.shell = ShellTools(cwd=str(cwd)) if shell else None
        self.cwd = cwd
        self._jobs_running: dict = {}  # set by AtomAgent.__init__


@pytest.fixture
async def agent(tmp_path):
    agent = _Agent(tmp_path)
    agent.jobs = JobsApi(agent)
    yield agent
    await agent.queue_manager.shutdown()
    await agent.shell.close()


async def _next(agent) -> object:
    return await asyncio.wait_for(agent.queue_manager.get_channel("jobs").get(), TIMEOUT)


async def _stream_updates(agent, label: str) -> list[JobUpdate]:
    """Every JobUpdate a streaming job delivers, read until it has ended."""
    channel = agent.queue_manager.get_channel("jobs")
    updates = []
    while agent.jobs.get(label).state == "running" or channel.qsize():
        try:
            item = await asyncio.wait_for(channel.get(), 0.2)
        except TimeoutError:
            continue
        assert isinstance(item, JobUpdate) and item.label == label
        updates.append(item)
    return updates


async def _until(predicate, timeout: float = TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.02)


async def test_run_delivers_a_labelled_shell_result(agent):
    info = await agent.jobs.run("echo out; echo err >&2; exit 3", label="t")
    assert info.state == "running" and info.kind == "run"
    assert info.path == "/root/t"  # outside a session tree the agent is /root
    item = await _next(agent)
    assert (isinstance(item, JobUpdate) and item.done) and item.label == "t"
    assert item.path == "/root/t"
    assert isinstance(item.items[0], ShellResult)
    assert item.items[0].returncode == 3 and item.items[0].stdout == "out\n"
    assert item.items[0].stderr == "err\n"
    # No job-owned log file: the ShellResult is the whole delivered output.
    assert set(JobUpdate.model_fields) == {"label", "path", "items", "done", "error"}
    assert not [name for name in type(agent.jobs.get("t")).model_fields if "log" in name]
    assert agent.jobs.get("t").state == "done"
    assert agent.jobs.get("t").finished_at is not None


async def test_run_starts_in_the_shell_directory_with_its_exports(agent, tmp_path):
    (tmp_path / "sub").mkdir()
    await agent.shell.run("export JOBS_TEST_VAR=from-shell; cd sub")
    await agent.jobs.run('echo "$JOBS_TEST_VAR"; pwd; cd /; export X=1', label="env")
    item = await _next(agent)
    assert item.items[0].stdout.splitlines() == ["from-shell", str(tmp_path / "sub")]
    # The job's cd/export do not leak into the agent's shell.
    assert (await agent.shell.run("pwd; echo ${X:-unset}")).splitlines() == [
        str(tmp_path / "sub"),
        "unset",
    ]


async def test_run_takes_stdin_and_cwd(agent, tmp_path):
    (tmp_path / "d").mkdir()
    await agent.jobs.run("cat; pwd", label="stdin_job", stdin="hello\n", cwd="d")
    item = await _next(agent)
    assert item.items[0].stdout == f"hello\n{tmp_path / 'd'}\n"


async def test_run_does_not_block_the_shell(agent):
    await agent.jobs.run("sleep 3; echo late", label="slow")
    started = time.monotonic()
    assert (await agent.shell.run("echo now")).strip() == "now"
    assert time.monotonic() - started < 2.0
    assert agent.jobs.get("slow").state == "running"


async def test_run_timeout_kills_and_reports_124(agent):
    await agent.jobs.run("echo start; sleep 30", label="to", timeout=0.5)
    item = await _next(agent)
    assert item.items[0].timed_out and item.items[0].returncode == 124
    assert item.items[0].stdout == "start\n"


async def test_run_timeout_applies_when_the_output_is_redirected(agent):
    # The pipes close at once, so only the wait for the process can time out.
    started = time.monotonic()
    await agent.jobs.run("sleep 30 > /dev/null 2>&1", label="quiet", timeout=0.5)
    item = await _next(agent)
    assert item.done and item.items[0].timed_out and item.items[0].returncode == 124
    assert time.monotonic() - started < 5.0


async def test_cancel_kills_the_process_and_its_group(agent, tmp_path):
    pidfile, childfile = tmp_path / "pid", tmp_path / "child"
    await agent.jobs.run(
        f"sleep 999 & echo $! > {childfile}; echo $$ > {pidfile}; wait", label="long"
    )
    await _until(lambda: pidfile.exists() and pidfile.read_text().strip())
    pid, child = int(pidfile.read_text()), int(childfile.read_text())
    assert _alive(child)
    assert await agent.jobs.cancel("long") is True
    assert agent.jobs.get("long").state == "cancelled"
    await _until(lambda: not _alive(pid) and not _alive(child))
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


@pytest.mark.timeout(60)  # without the fix teardown hangs too: fail, do not stall the run
@pytest.mark.parametrize("throttle", [{}, {"items": 1}])
async def test_cancel_returns_while_a_chatty_jobs_output_is_buffered(agent, throttle):
    # ``yes`` prints far more than the 256-slot queues hold, so at the cancel
    # the line pumps are blocked on full queues. Their cleanup must not wait
    # for queue space nobody frees, or cancel() never returns.
    await agent.jobs.run("yes", label="chatty", **throttle)
    await asyncio.sleep(0.5)
    cancel = asyncio.ensure_future(agent.jobs.cancel("chatty"))
    # asyncio.wait, not wait_for: wait_for would wait for the hung cancel itself.
    done, _ = await asyncio.wait({cancel}, timeout=5.0)
    assert done, "cancel() did not return: the job's cleanup is blocked on a full queue"
    assert cancel.result() is True
    assert agent.jobs.get("chatty").state == "cancelled"


async def test_timeout_ends_a_chatty_job(agent):
    await agent.jobs.run("yes", label="chatty", timeout=0.5)
    item = await asyncio.wait_for(_next(agent), 5.0)
    assert isinstance(item, JobUpdate) and item.done
    assert item.items[0].returncode == 124 and item.items[0].timed_out


async def test_labels_are_unique_among_running_jobs(agent):
    await agent.jobs.run("sleep 5", label="x")
    hint = (
        "a job labelled 'x' is already running; pass replace=True to restart it, "
        "or await self.jobs.cancel('x')"
    )
    with pytest.raises(ValueError, match=re.escape(hint)):
        await agent.jobs.run("true", label="x")
    with pytest.raises(ValueError, match="already running"):
        agent.jobs.spawn(asyncio.sleep(0), label="x")
    for bad in (" ", "ci-check", "class", "1st"):
        with pytest.raises(ValueError, match="is not valid: use a Python identifier"):
            await agent.jobs.run("true", label=bad)
    for reserved in ("jobs", "delegates", "user", "user_messages", "system_messages"):
        with pytest.raises(ValueError, match=f"job name '{reserved}' is reserved: Waiting"):
            await agent.jobs.run("true", label=reserved)
    agent.queue_manager.queue("mesh")  # a queue-manager channel in use
    work = asyncio.sleep(0)
    with pytest.raises(ValueError, match="job name 'mesh' is reserved"):
        agent.jobs.spawn(work, label="mesh")
    work.close()  # refused before it was started
    await agent.jobs.cancel("x")
    await agent.jobs.run("echo again", label="x")  # free once it stopped
    item = await _next(agent)
    assert item.items[0].stdout == "again\n"
    assert [job.label for job in agent.jobs.list()] == ["x", "x"]


async def _collect_run(agent) -> tuple[list[JobUpdate], JobUpdate]:
    """Read a run job's progress updates up to and including its done update."""
    updates = []
    while True:
        item = await _next(agent)
        if item.done:
            return updates, item
        assert isinstance(item, JobUpdate) and not item.done
        updates.append(item)


async def test_a_quiet_run_ends_with_one_done_update_holding_the_shell_result(agent):
    await agent.jobs.run("for i in 1 2 3; do echo $i; done", label="q")
    updates, result = await _collect_run(agent)
    assert updates == [] and len(result.items) == 1 and isinstance(result.items[0], ShellResult)
    assert result.items[0].stdout == "1\n2\n3\n"
    assert result.error is None


async def test_run_with_progress_delivers_lines_then_its_result(agent):
    await agent.jobs.run("echo a; echo b >&2; echo c", label="s", items=1)
    updates, result = await _collect_run(agent)
    assert all(len(u.items) == 1 and isinstance(u.items[0], StreamEvent) for u in updates)
    lines = sorted((u.items[0].kind, u.items[0].text) for u in updates)
    assert lines == [("stderr", "b"), ("stdout", "a"), ("stdout", "c")]  # no trailing newline
    assert result.label == "s" and result.items[0].returncode == 0
    assert result.items[0].stdout == "a\nc\n" and result.items[0].stderr == "b\n"


async def test_throttle_by_items_batches_and_flushes_before_the_result(agent):
    await agent.jobs.run("for i in 1 2 3 4 5; do echo $i; done", label="b", items=2)
    updates, result = await _collect_run(agent)
    texts = [[e.text for e in u.items] for u in updates]
    # "5" is still pending at the end: not sent alone, the ShellResult has it.
    assert texts == [["1", "2"], ["3", "4"]]
    assert result.items == [result.items[0]] and result.items[0].stdout == "1\n2\n3\n4\n5\n"


async def test_run_progress_by_seconds(agent):
    # Bursts of lines separated by pauses longer than the window: each burst is
    # batched, and the window (not the end of the run) delivers it.
    await agent.jobs.run(
        "echo 1; echo 2; sleep 0.6; echo 3; echo 4; sleep 0.6; echo 5; echo 6",
        label="t",
        seconds=0.3,
    )
    updates, result = await _collect_run(agent)
    sent = [e.text for u in updates for e in u.items]
    # In order and without gaps; lines still batched at the end are in the ShellResult only.
    assert sent == [f"{i}" for i in range(1, len(sent) + 1)] and len(sent) >= 2
    assert any(len(u.items) > 1 for u in updates)  # batched, not one update per line
    assert result.items[0].returncode == 0
    assert result.items[0].stdout == "".join(f"{i}\n" for i in range(1, 7))


async def test_throttle_turns_run_progress_on_and_off(agent, tmp_path):
    # Every phase is gated by a file the test creates, so each throttle change
    # happens before the output it governs is printed (no timing assumptions).
    # The command touches `started` once running and waiting: a real signal from
    # the process itself. It prints nothing before go1, so no output can still be
    # in flight when progress is turned on.
    go, started = tmp_path / "go", tmp_path / "started"
    script = (
        f"touch {started}; "
        f"while [ ! -e {go}1 ]; do sleep 0.02; done; echo loud1; echo loud2; "
        f"while [ ! -e {go}2 ]; do sleep 0.02; done; echo quiet1; echo quiet2"
    )
    info = await agent.jobs.run(script, label="r")
    assert info.throttle is None  # quiet by default
    await _until(started.exists)
    assert agent.queue_manager.get_channel("jobs").qsize() == 0
    assert agent.jobs.throttle("r", items=2).throttle.items == 2  # progress on
    (tmp_path / "go1").touch()
    update = await _next(agent)
    assert isinstance(update, JobUpdate) and [e.text for e in update.items] == [
        "loud1",
        "loud2",
    ]
    assert agent.jobs.throttle("r").throttle is None  # progress off again
    (tmp_path / "go2").touch()
    # quiet1/quiet2 are printed only after progress was turned off: not delivered.
    updates, result = await _collect_run(agent)
    assert updates == []
    assert result.items[0].stdout == "loud1\nloud2\nquiet1\nquiet2\n"


async def test_throttle_by_seconds_batches_slow_output(agent):
    async def ticks():
        # Bursts separated by pauses longer than the window.
        for burst in ((0, 1, 2), (3, 4), (5,)):
            for i in burst:
                yield i
            await asyncio.sleep(0.6)

    agent.jobs.spawn(ticks(), label="tick", seconds=0.3)
    updates = await _stream_updates(agent, "tick")
    assert updates[-1].done and [u.done for u in updates].count(True) == 1
    flat = [v for u in updates for v in u.items]
    assert flat == list(range(6))  # complete and in order
    assert any(len(u.items) > 1 for u in updates)  # several items per delivery
    assert not updates[0].done  # the window delivered before the end


async def test_throttle_off_flushes_pending_items(agent):
    gate = asyncio.Event()

    async def two_then_wait():
        yield "a"
        yield "b"
        await gate.wait()

    agent.jobs.spawn(two_then_wait(), label="p", items=10)
    await _until(lambda: agent.jobs.get("p").pending == 2)
    assert agent.jobs.throttle("p").throttle is None
    # Turning batching off delivers what is pending as soon as more arrives or the job ends.
    gate.set()
    updates = await _stream_updates(agent, "p")
    assert [item for u in updates for item in u.items] == ["a", "b"]
    assert [u.done for u in updates] == [False, True]  # one done update ends it
    assert updates[-1].items == []  # nothing was left at the end


async def test_throttle_rejects_non_streaming_jobs(agent):
    agent.jobs.after(5, label="r")
    with pytest.raises(ValueError, match="not a streaming job"):
        agent.jobs.throttle("r", items=2)
    with pytest.raises(KeyError):
        agent.jobs.throttle("missing")
    with pytest.raises(ValueError):
        await agent.jobs.run("true", label="bad", items=0)


async def test_spawn_a_coroutine_delivers_one_result_or_its_error(agent):
    async def compute(value):
        await asyncio.sleep(0.01)
        return value

    async def broken():
        raise RuntimeError("boom")

    agent.jobs.spawn(compute(42), label="c")
    item = await _next(agent)
    assert (isinstance(item, JobUpdate) and item.done) and item.label == "c" and item.items == [42]
    # The result is always wrapped: None and lists are items[0] too.
    agent.jobs.spawn(compute(None), label="n")
    assert (await _next(agent)).items == [None]
    agent.jobs.spawn(compute([1, 2]), label="l")
    assert (await _next(agent)).items == [[1, 2]]
    agent.jobs.spawn(broken(), label="e")
    item = await _next(agent)
    assert item.done and item.items == [] and item.error == "RuntimeError: boom"
    assert agent.jobs.get("e").state == "failed"
    with pytest.raises(TypeError):
        agent.jobs.spawn(42, label="z")


async def test_get_shows_recent_items(agent):
    agent.jobs.spawn(asyncio.sleep(0, result="v"), label="g")
    await _next(agent)
    info = agent.jobs.get("g")
    assert info.state == "done" and info.last[-1].items[0] == "v"
    assert agent.jobs.get("unknown") is None


async def test_status_names_self_jobs_not_the_queue_manager(agent):
    await agent.jobs.run("sleep 5", label="build")
    status = agent.queue_manager.status()
    assert "build" in status
    assert status.endswith("\nCancel a job: await self.jobs.cancel('<name>')")
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
    assert item.items[0].stdout == "proc\n"
    await agent.queue_manager.shutdown()


async def test_each_throttle_call_replaces_the_setting_and_flushes_the_batch(agent):
    gates = [asyncio.Event(), asyncio.Event(), asyncio.Event()]

    async def numbers():
        for i in range(2):
            yield i  # not throttled: one update each
        await gates[0].wait()
        for i in range(2, 6):
            yield i  # batched under seconds=60: nothing delivered yet
        await gates[1].wait()
        for i in range(6, 12):
            yield i
        await gates[2].wait()
        for i in range(12, 16):
            yield i

    agent.jobs.spawn(numbers(), label="r")
    assert [(await _next(agent)).items for _ in range(2)] == [[0], [1]]
    assert agent.jobs.throttle("r", seconds=60).throttle.seconds == 60
    gates[0].set()
    await _until(lambda: agent.jobs.get("r").pending == 4)
    # Next setting: lines only; seconds is dropped, the old batch is delivered first.
    info = agent.jobs.throttle("r", items=3)
    assert info.throttle.items == 3 and info.throttle.seconds is None
    assert agent.jobs.get("r").throttle == info.throttle
    assert (await _next(agent)).items == [2, 3, 4, 5]
    gates[1].set()
    assert (await _next(agent)).items == [6, 7, 8]
    assert (await _next(agent)).items == [9, 10, 11]
    # items=2 replaces items=3.
    assert agent.jobs.throttle("r", items=2).throttle.items == 2
    gates[2].set()
    assert (await _next(agent)).items == [12, 13]
    assert (await _next(agent)).items == [14, 15]
    # It ended with nothing batched: the done update is empty.
    final = await _next(agent)
    assert final.done and final.items == [] and final.error is None


# ---- tail / after / every (ported from the removed nooa.runtime.producers tests) ----


async def test_tail_delivers_only_lines_appended_after_the_call(agent, tmp_path):
    log = tmp_path / "app.log"
    log.write_text("existing\n")
    info = agent.jobs.tail("app.log", label="log")  # relative to the shell's directory
    assert info.kind == "tail" and info.state == "running"
    with log.open("a") as fh:
        fh.write("line1\nline2\npart")
    assert (await _next(agent)).items == ["line1"]
    assert (await _next(agent)).items == ["line2"]
    with log.open("a") as fh:
        fh.write("ial\n")  # a line is delivered only once it is complete
    assert (await _next(agent)).items == ["partial"]
    assert await agent.jobs.cancel("log") is True
    assert agent.jobs.get("log").state == "cancelled"


async def test_tail_follows_a_file_that_is_truncated(agent, tmp_path):
    log = tmp_path / "out.log"
    log.write_text("old line 1\nold line 2\n")
    agent.jobs.tail("out.log", label="log")
    log.write_text("new\n")  # truncate and rewrite, as `cmd > out.log` does
    assert (await _next(agent)).items == ["new"]
    assert await agent.jobs.cancel("log") is True


async def test_tail_batches_and_rethrottles(agent, tmp_path):
    log = tmp_path / "batch.log"
    log.write_text("")
    agent.jobs.tail(str(log), label="b", items=3)
    with log.open("a") as fh:
        fh.write("".join(f"{i}\n" for i in range(5)))
    assert (await _next(agent)).items == ["0", "1", "2"]
    await _until(lambda: agent.jobs.get("b").pending == 2)
    agent.jobs.throttle("b")  # off: the pending two are delivered at once
    assert (await _next(agent)).items == ["3", "4"]
    await agent.jobs.cancel("b")


async def test_tail_of_a_missing_file_raises_at_once(agent):
    with pytest.raises(FileNotFoundError):
        agent.jobs.tail("missing.log", label="m")
    assert agent.jobs.get("m") is None


async def test_after_delivers_its_payload_once_none_by_default(agent):
    started = time.monotonic()
    info = agent.jobs.after(0.2, label="timer")
    assert info.kind == "after" and info.payload is None
    item = await _next(agent)
    assert (isinstance(item, JobUpdate) and item.done) and item.label == "timer"
    assert item.items == [None] and item.error is None
    assert time.monotonic() - started >= 0.2
    assert agent.jobs.get("timer").state == "done"
    task = "check CI on PR #462 and report"
    assert agent.jobs.after(0.05, label="check_ci", payload=task).payload == task
    assert (await _next(agent)).items == [task]
    assert agent.jobs.get("check_ci").payload == task
    # Other kinds keep payload None.
    assert agent.jobs.every(30, label="e").payload is None
    assert (await agent.jobs.run("true", label="r")).payload is None
    await agent.jobs.cancel("e")


async def test_after_can_be_cancelled(agent):
    agent.jobs.after(30, label="later")
    assert await agent.jobs.cancel("later") is True
    await asyncio.sleep(0.05)
    assert agent.queue_manager.get_channel("jobs").qsize() == 0


async def test_every_ticks_until_cancelled(agent):
    info = agent.jobs.every(0.05, label="tick")
    assert info.kind == "every"
    assert [(await _next(agent)).items for _ in range(3)] == [[1], [2], [3]]  # no payload
    assert await agent.jobs.cancel("tick") is True
    assert agent.jobs.get("tick").state == "cancelled"
    with pytest.raises(ValueError):
        agent.jobs.every(0, label="bad")


async def test_cancel_withdraws_the_jobs_queued_updates_only(agent):
    channel = agent.queue_manager.get_channel("jobs")
    agent.jobs.after(0, label="other", payload="keep")
    agent.jobs.every(0.01, label="tick")
    await _until(lambda: sum(u.label == "tick" for u in channel.snapshot()) >= 2)
    assert await agent.jobs.cancel("tick") is True
    await asyncio.sleep(0.05)
    assert [(u.label, u.items) for u in channel.snapshot()] == [("other", ["keep"])]


async def test_a_cancelled_job_has_nothing_pending(agent):
    agent.jobs.every(0.01, label="tick")
    agent.jobs.throttle("tick", items=1000)
    await _until(lambda: agent.jobs.get("tick").pending >= 2)
    assert await agent.jobs.cancel("tick") is True
    assert agent.jobs.get("tick").pending == 0


async def test_a_finished_job_leaves_no_stream_end_in_the_models_history(tmp_path):
    from nooa.runtime.channels import StreamEnd
    from nooa.runtime.event_manager import EventManager

    events = EventManager()
    ended: list = []
    events.on("StreamEnd", ended.append)
    agent = _Agent(tmp_path, shell=False)
    agent.queue_manager = QueueManager(event_manager=events)
    agent.queue_manager.queue("jobs")
    agent.jobs = JobsApi(agent)
    agent.jobs.after(0, label="t")
    assert (await _next(agent)).done
    await _until(lambda: ended)
    assert ended[0].channel_name == "jobs"  # subscribers still get it
    assert not [tag for tag in events.keys() if isinstance(events[tag], StreamEnd)]
    await agent.queue_manager.shutdown()


# ---- process isolation (ported from the removed monitor() tests) ----


async def test_commands_run_in_their_own_process_group(agent):
    await agent.jobs.run("ps -o pgid= -p $$", label="pg")
    item = await _next(agent)
    assert int(item.items[0].stdout.strip()) != os.getpgid(os.getpid())


async def test_concurrent_jobs_do_not_interfere(agent):
    await agent.jobs.run("echo A1; sleep 0.1; echo A2", label="a")
    await agent.jobs.run("echo B1; sleep 0.1; echo B2", label="b")
    items = {item.label: item for item in [await _next(agent), await _next(agent)]}
    assert items["a"].items[0].stdout == "A1\nA2\n"
    assert items["b"].items[0].stdout == "B1\nB2\n"


async def test_cancelling_one_job_leaves_the_other_running(agent):
    await agent.jobs.run("sleep 999", label="victim")
    await agent.jobs.run("sleep 0.3; echo survived", label="survivor")
    await agent.jobs.cancel("victim")
    item = await _next(agent)
    assert item.label == "survivor" and item.items[0].stdout == "survived\n"


async def test_undecodable_output_does_not_end_the_stream(agent):
    await agent.jobs.run(r"printf 'first\ncaf\xe9\nlast\n'", label="bytes", items=1)
    updates, result = await _collect_run(agent)
    texts = [e.text for u in updates for e in u.items]
    assert texts == ["first", "caf\ufffd", "last"]
    assert result.items[0].stdout == "first\ncaf\ufffd\nlast\n"


async def test_delivered_line_items_never_end_in_a_newline(agent, tmp_path):
    await agent.jobs.run(r"printf 'crlf\r\nlf\nlast'", label="nl", items=1)
    updates, result = await _collect_run(agent)
    assert [e.text for u in updates for e in u.items] == ["crlf", "lf", "last"]
    assert result.items[0].stdout == "crlf\r\nlf\nlast"  # the ShellResult keeps the output as is
    log = tmp_path / "nl.log"
    log.write_text("")
    agent.jobs.tail(str(log), label="t", items=2)
    log.write_bytes(b"one\r\ntwo\n")
    assert (await _next(agent)).items == ["one", "two"]
    await agent.jobs.cancel("t")


# ---- target and description ----


async def test_target_says_what_runs_for_each_kind(agent, tmp_path):
    log = tmp_path / "t.log"
    log.write_text("")

    async def fetch_report():
        await asyncio.sleep(30)

    async def lines_of_work():
        yield 1
        await asyncio.sleep(30)

    assert (await agent.jobs.run("sleep 30", label="r")).target == "sleep 30"
    assert agent.jobs.spawn(fetch_report(), label="c").target.endswith("fetch_report()")
    assert agent.jobs.spawn(lines_of_work(), label="g").target.endswith("lines_of_work()")
    assert agent.jobs.tail("t.log", label="rel").target == f"t.log ({log})"
    assert agent.jobs.tail(str(log), label="abs").target == str(log)
    assert agent.jobs.after(300, label="a").target == "after 300s"
    assert agent.jobs.every(0.5, label="e").target == "every 0.5s"
    assert {job.label: job.target for job in agent.jobs.list()}["a"] == "after 300s"
    for label in ("r", "c", "g", "rel", "abs", "a", "e"):
        await agent.jobs.cancel(label)


async def test_description_round_trips(agent):
    info = await agent.jobs.run("sleep 30", label="tests", description="full run before pushing")
    assert info.description == "full run before pushing"
    assert agent.jobs.get("tests").description == "full run before pushing"
    assert agent.jobs.list()[-1].description == "full run before pushing"
    assert agent.jobs.after(30, label="a").description == ""
    with pytest.raises(TypeError):
        agent.jobs.every(1, label="bad", description=3)
    await agent.jobs.cancel("tests")
    await agent.jobs.cancel("a")
    assert agent.jobs.get("tests").description == "full run before pushing"  # kept after it ends


async def test_long_targets_are_cut_to_one_short_line(agent):
    cmd = "echo start\n" + "echo " + "x" * 400
    info = await agent.jobs.run(cmd, label="long")
    assert len(info.target) == 200 and info.target.endswith("…")
    assert "\n" not in info.target and info.target.startswith("echo start echo xxx")
    await _next(agent)


async def test_the_status_line_shows_label_state_and_description_or_target(agent):
    await agent.jobs.run("sleep 30", label="tests", description="full run before pushing")
    agent.jobs.after(300, label="timer")
    await agent.jobs.run("sleep 30 # " + "y" * 300, label="noisy")
    status = agent.queue_manager.status()
    assert re.search(r"  • /root/tests · job · \d+s · full run before pushing", status)
    assert re.search(r"  • /root/timer · job · \d+s · after 300s", status)
    noisy = next(line for line in status.splitlines() if "noisy" in line)
    assert noisy.endswith("…") and len(noisy) <= 4 + 100
    assert "→ jobs" not in status
    for label in ("tests", "timer", "noisy"):
        await agent.jobs.cancel(label)


# ---- timer payloads ----


async def test_every_delivers_its_payload_each_tick(agent):
    payload = {"task": "poll CI", "pr": 462}
    agent.jobs.every(0.05, label="poll", payload=payload)
    assert [(await _next(agent)).items for _ in range(2)] == [[payload], [payload]]
    assert agent.jobs.get("poll").payload == payload
    assert agent.jobs.list()[-1].payload == payload
    await agent.jobs.cancel("poll")


# ---- replace=True ----


async def test_replace_restarts_a_running_run_job(agent, tmp_path):
    pidfile = tmp_path / "old.pid"
    await agent.jobs.run(f"echo $$ > {pidfile}; exec sleep 60", label="srv", items=1)
    await _until(lambda: pidfile.exists() and pidfile.read_text().strip())
    old_pid = int(pidfile.read_text())
    with pytest.raises(ValueError, match="already running"):
        await agent.jobs.run("true", label="srv")
    info = await agent.jobs.run("echo new", label="srv", replace=True)
    assert info.state == "running" and info.target == "echo new"
    await _until(lambda: not _alive(old_pid))
    item = await _next(agent)
    while not item.done:  # the new job's progress, if any
        item = await _next(agent)
    assert (isinstance(item, JobUpdate) and item.done) and item.items[0].stdout == "new\n"
    await asyncio.sleep(0.1)
    assert agent.queue_manager.get_channel("jobs").qsize() == 0  # nothing from the old job
    states = [(job.target, job.state) for job in agent.jobs.list()]
    assert states[-2:] == [
        (f"echo $$ > {pidfile}; exec sleep 60", "cancelled"),
        ("echo new", "done"),
    ]


async def test_replace_restarts_a_running_spawn_job_from_sync_code(agent):
    async def ticking(name):
        n = 0
        while True:
            n += 1
            yield f"{name}{n}"
            await asyncio.sleep(0.05)

    agent.jobs.spawn(ticking("old"), label="w")
    assert (await _next(agent)).items == ["old1"]
    agent.jobs.spawn(ticking("new"), label="w", replace=True)  # sync, no await
    seen = [(await _next(agent)).items[0] for _ in range(3)]
    assert seen == ["new1", "new2", "new3"]
    assert agent.jobs.list()[-2].state == "cancelled"
    await agent.jobs.cancel("w")


async def test_replace_without_a_running_job_just_starts(agent):
    info = agent.jobs.after(0.01, label="t", replace=True, payload="p")
    assert info.state == "running"
    assert (await _next(agent)).items[0] == "p"


# ---- cancel hints ----


async def test_status_has_no_queue_manager_cancel_hint_for_skill_channels():
    qm = QueueManager()
    qm.queue("mesh")

    async def pump():
        await asyncio.sleep(30)
        yield None

    qm.spawn(pump(), channel="mesh", label="inbox")
    await asyncio.sleep(0)
    status = qm.status()
    assert "inbox" in status and "cancel" not in status
    qm.set_cancel_hint("mesh", "Stop the inbox: await self.mesh.stop()")
    qm.spawn(pump(), channel="mesh", label="outbox")
    await asyncio.sleep(0)
    assert qm.status().count("Stop the inbox: await self.mesh.stop()") == 1
    await qm.shutdown()


# ---- lost on restart ----


async def test_jobs_running_at_a_snapshot_are_reported_once_after_restore(tmp_path):
    from nooa_atom.agent.jobs import JobsLost

    from nooa.interactive import InteractiveAgent
    from nooa.storage.snapshot import AgentSnapshot
    from nooa.unifiedllm import FakeLLMClient

    class Host(InteractiveAgent, llm=FakeLLMClient()):
        """A minimal interactive agent that installs JobsApi, as AtomAgent does."""

        jobs: JobsApi
        _jobs_running: dict

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._jobs_in = self.queue_manager.queue("jobs")
            self._jobs_running = {}
            self.jobs = JobsApi(self)

        def after_restore(self) -> None:
            super().after_restore()
            self.jobs.after_restore()

    old = Host(llm=FakeLLMClient())
    await old.jobs.run("sleep 30", label="tests", description="full run")
    old.jobs.after(0.01, label="quick")
    await asyncio.wait_for(old.queue_manager.get_channel("jobs").get(), TIMEOUT)
    await _until(lambda: "quick" not in old._jobs_running)  # a finished job is forgotten
    snapshot = AgentSnapshot.from_agent(old)
    await old.queue_manager.shutdown()  # the old process ends
    assert set(old._jobs_running) == {"tests"}  # a shutdown keeps the record

    new = Host(llm=FakeLLMClient())
    snapshot.restore(new)
    new.after_restore()
    [event] = new.event_manager.filter(type="JobsLost")
    assert isinstance(event, JobsLost)
    assert event.jobs == [
        {
            "label": "tests",
            "kind": "run",
            "target": "sleep 30",
            "description": "full run",
            "started_at": event.jobs[0]["started_at"],
        }
    ]
    assert new._jobs_running == {}
    new.after_restore()
    assert len(new.event_manager.filter(type="JobsLost")) == 1  # once
    await new.queue_manager.shutdown()


async def test_a_job_cancelled_by_the_model_is_not_reported_lost(agent):
    await agent.jobs.run("sleep 30", label="x")
    await agent.jobs.cancel("x")
    assert agent._jobs_running == {}


# ---- one done update ends every job ----


async def test_a_failing_stream_sends_its_pending_values_then_an_empty_done_update(agent):
    async def breaks():
        yield 1
        yield 2
        raise RuntimeError("boom")

    agent.jobs.spawn(breaks(), label="b", items=10)
    updates = await _stream_updates(agent, "b")
    assert [(u.items, u.done, u.error) for u in updates] == [
        ([1, 2], False, None),
        ([], True, "RuntimeError: boom"),
    ]
    assert agent.jobs.get("b").state == "failed"


async def test_a_stream_ends_with_its_last_pending_values(agent):
    async def three():
        for i in range(3):
            yield i

    agent.jobs.spawn(three(), label="s", items=2)
    updates = await _stream_updates(agent, "s")
    assert [(u.items, u.done) for u in updates] == [([0, 1], False), ([2], True)]


# ---- model-facing behaviour: result size, errors ------------------------


async def test_a_big_run_result_keeps_the_last_20000_chars_of_each_stream(agent):
    await agent.jobs.run(
        "python3 -c \"import sys; [print(f'line {i:05d}') for i in range(5000)];"
        " [sys.stderr.write(f'err {i:05d}\\n') for i in range(3000)]\"",
        label="big",
    )
    item = await _next(agent)
    result = item.items[0]
    out = "".join(f"line {i:05d}\n" for i in range(5000))  # 55,000 chars
    err = "".join(f"err {i:05d}\n" for i in range(3000))  # 30,000 chars
    # Each stream keeps exactly its own last 20,000 chars, after a one-line
    # notice that names no file (there is none).
    assert result.stdout == f"[... {55_000 - 20_000:,} chars omitted]\n" + out[-20_000:]
    assert result.stderr == f"[... {30_000 - 20_000:,} chars omitted]\n" + err[-20_000:]
    for stream in (result.stdout, result.stderr):
        notice, _, kept = stream.partition("\n")
        assert len(kept) == 20_000
        assert "full output" not in notice and "/" not in notice and ".log" not in notice


async def test_a_run_under_the_cap_is_not_truncated(agent):
    # 2,000 lines of 10 chars: exactly 20,000 chars, kept whole with no notice.
    await agent.jobs.run("for i in $(seq 2000); do echo 123456789; done", label="fits")
    item = await _next(agent)
    assert item.items[0].stdout == "123456789\n" * 2_000


@pytest.mark.parametrize("bad", ["10m", "10", True, None])
async def test_seconds_must_be_a_number(agent, bad):
    expected = TypeError
    with pytest.raises(expected, match=re.escape("seconds must be a number of seconds (e.g. 600")):
        agent.jobs.after(bad, label="a")
    with pytest.raises(expected, match="seconds must be a number of seconds"):
        agent.jobs.every(bad, label="e")
    if bad is not None:  # None turns throttle's seconds off
        agent.jobs.every(30, label="t")
        with pytest.raises(expected, match="seconds must be a number of seconds"):
            agent.jobs.throttle("t", seconds=bad)
        await agent.jobs.cancel("t")


@pytest.mark.parametrize("bad", ["200", 2.5, True])
async def test_items_must_be_a_whole_number(agent, bad):
    agent.jobs.every(30, label="t")
    with pytest.raises(TypeError, match="items must be a whole number"):
        agent.jobs.throttle("t", items=bad)
    with pytest.raises(TypeError, match="items must be a whole number"):
        await agent.jobs.run("true", label="r", items=bad)
    await agent.jobs.cancel("t")


async def test_after_rejects_negative_seconds_and_fires_at_once_on_zero(agent):
    with pytest.raises(ValueError, match="0 or more"):
        agent.jobs.after(-1, label="a")
    with pytest.raises(ValueError, match="positive"):
        agent.jobs.every(0, label="e")
    agent.jobs.after(0, label="now", payload="go")
    item = await _next(agent)
    assert item.done and item.items == ["go"]


async def test_spawn_of_an_uncalled_async_function_says_to_call_it(agent):
    async def compute():
        return 1

    async def numbers():
        yield 1

    with pytest.raises(TypeError, match=re.escape("pass compute(), not compute")):
        agent.jobs.spawn(compute, label="c")
    with pytest.raises(TypeError, match=re.escape("pass numbers(), not numbers")):
        agent.jobs.spawn(numbers, label="n")
    assert agent.jobs.list() == []


# ---- bounded memory: long lines, chatty output ---------------------------

_BIG_LINE = (
    "python3 -c \"import sys; sys.stdout.write('x' * 5_000_000); sys.stdout.flush()\";"
    " echo; echo after-1; echo after-2"
)


async def test_a_5mb_line_stays_bounded_and_later_lines_still_arrive(agent):
    tracemalloc.start()
    try:
        baseline = tracemalloc.get_traced_memory()[0]
        await agent.jobs.run(_BIG_LINE, label="big", items=1)
        updates = await _stream_updates(agent, "big")
        peak = tracemalloc.get_traced_memory()[1] - baseline
    finally:
        tracemalloc.stop()
    assert peak < 2_000_000, f"peak {peak:,} bytes above baseline"
    lines = [event.text for update in updates[:-1] for event in update.items]
    assert lines[0] == "x" * 8_000 + "[... 4,992,000 chars of this line omitted]"
    assert lines[1:] == ["after-1", "after-2"]
    done = updates[-1]
    assert done.done and len(done.items[0].stdout) <= 20_000
    assert done.items[0].stdout.endswith("after-1\nafter-2\n")


async def test_a_throttled_chatty_run_sends_batches_early_at_the_cap(agent):
    cmd = "python3 -c \"[print(f'line {i}') for i in range(10_000)]\""
    await agent.jobs.run(cmd, label="chatty", seconds=60)
    started = time.monotonic()
    updates = await _stream_updates(agent, "chatty")
    assert time.monotonic() - started < 30  # did not wait for the 60 s window
    progress = [len(update.items) for update in updates[:-1]]
    assert progress and max(progress) == 200
    assert sum(progress) >= 9_800  # the last partial batch goes to the ShellResult only
    assert updates[-1].done and updates[-1].items[0].stdout.endswith("line 9999\n")


async def test_tail_caps_a_5mb_line(agent, tmp_path):
    log = tmp_path / "app.log"
    log.write_text("")
    agent.jobs.tail(log, label="log")
    with log.open("a") as handle:
        handle.write("y" * 5_000_000 + "\nshort\n")
    first = await _next(agent)
    second = await _next(agent)
    assert first.items == ["y" * 8_000 + "[... 4,992,000 chars of this line omitted]"]
    assert second.items == ["short"]
    await agent.jobs.cancel("log")


def test_status_line_shows_age_while_running_and_pending_above_zero():
    from nooa_atom.agent import jobs as jobs_mod

    assert jobs_mod._age(jobs_mod._now()) in ("0s", "1s")
    old = (datetime.datetime.now() - datetime.timedelta(minutes=65)).isoformat(timespec="seconds")
    assert jobs_mod._age(old) == "1h05m"
    job = jobs_mod._Job("build", "run", None, True, target="make build")
    job.started_at = old
    job.pending = 2
    assert job.status_line() == "build · job · 1h05m · 2 pending · make build"
    job.pending = 0
    assert "pending" not in job.status_line()


@pytest.mark.parametrize(
    "start",
    [
        lambda jobs: jobs.run("true", label="old", description=3, replace=True),
        lambda jobs: jobs.run("true", label="old", items=0, replace=True),
        lambda jobs: jobs.run("true", label="old", seconds=-1, replace=True),
        lambda jobs: jobs.run("true", label="old", timeout="1", replace=True),
        lambda jobs: jobs.run(["true"], label="old", replace=True),
        lambda jobs: jobs.tail("missing.log", label="old", replace=True),
        lambda jobs: jobs.tail("f.log", label="old", description=None, replace=True),
        lambda jobs: jobs.tail("f.log", label="old", items="2", replace=True),
        lambda jobs: jobs.spawn(asyncio.sleep(0), label="old", items=2, replace=True),
        lambda jobs: jobs.spawn(asyncio.sleep(0), label="old", description=1, replace=True),
        lambda jobs: jobs.spawn(object(), label="old", replace=True),
        lambda jobs: jobs.after(-1, label="old", replace=True),
        lambda jobs: jobs.after(1, label="old", description=b"x", replace=True),
        lambda jobs: jobs.every(0, label="old", replace=True),
        lambda jobs: jobs.every(1, label="old", description=1, replace=True),
    ],
)
async def test_a_bad_argument_with_replace_leaves_the_old_job_running(agent, tmp_path, start):
    (tmp_path / "f.log").write_text("")
    await agent.jobs.run("sleep 30", label="old")
    old = agent.jobs.get("old")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # closed below, not the point here
        with pytest.raises((TypeError, ValueError, FileNotFoundError)):
            result = start(agent.jobs)
            if asyncio.iscoroutine(result):
                await result
    current = agent.jobs.get("old")
    assert current.state == "running" and current.started_at == old.started_at
    assert current.target == "sleep 30"


async def test_a_bad_label_with_replace_leaves_the_old_job_running(agent):
    await agent.jobs.run("sleep 30", label="old")
    with pytest.raises(ValueError, match="is not valid"):
        await agent.jobs.run("true", label="old-job", replace=True)
    assert agent.jobs.get("old").state == "running"


async def test_spawn_rejections_close_the_coroutine(agent):
    agent.jobs.after(30, label="busy")

    async def work():
        return 1

    rejections = [
        lambda: agent.jobs.spawn(work(), label="busy"),  # label taken
        lambda: agent.jobs.spawn(work(), label="not-valid"),  # bad label
        lambda: agent.jobs.spawn(work(), label="w", items=2),  # coroutine + items
        lambda: agent.jobs.spawn(work(), label="w", seconds=1),
        lambda: agent.jobs.spawn(work(), label="w", description=5),
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for reject in rejections:
            with pytest.raises((TypeError, ValueError)):
                reject()
        gc.collect()
    never_awaited = [w for w in caught if "was never awaited" in str(w.message)]
    assert never_awaited == []
    assert agent.jobs.get("w") is None


def _open_fds_on(path: Path) -> int:
    count = 0
    for fd in os.listdir("/proc/self/fd"):
        with contextlib.suppress(OSError):
            if os.readlink(f"/proc/self/fd/{fd}") == str(path):
                count += 1
    return count


@pytest.mark.skipif(not Path("/proc/self/fd").exists(), reason="needs /proc")
async def test_tail_cancelled_or_replaced_before_it_starts_holds_no_file(agent, tmp_path):
    log = tmp_path / "t.log"
    log.write_text("")
    agent.jobs.tail(log, label="tl")
    agent.jobs.tail(log, label="tl", replace=True)  # replaces the first before it ran
    assert await agent.jobs.cancel("tl") is True  # and cancels the second before it ran
    gc.collect()
    assert _open_fds_on(log) == 0
    # A tail that did run closes its file when cancelled.
    agent.jobs.tail(log, label="tl2")
    await _until(lambda: _open_fds_on(log) == 1)
    assert await agent.jobs.cancel("tl2") is True
    assert _open_fds_on(log) == 0


async def test_tail_delivers_lines_appended_after_the_call_before_it_ran(agent, tmp_path):
    log = tmp_path / "early.log"
    log.write_text("old\n")
    agent.jobs.tail(log, label="early")
    with log.open("a") as f:  # appended before the job's task first ran
        f.write("new\n")
    item = await _next(agent)
    assert item.items == ["new"]
    await agent.jobs.cancel("early")


def _update(kind, **fields):
    from nooa_atom.agent.jobs import JobUpdate

    update = JobUpdate(label="build", path="/root/build", **fields)
    update._kind = kind
    return update


def test_job_update_status_previews():
    from nooa.tools import ShellResult, StreamEvent

    def lines(*texts):
        return [StreamEvent(kind="stdout", text=text) for text in texts]

    def shell(code, timed_out=False):
        return [ShellResult(stdout="", stderr="", returncode=code, timed_out=timed_out)]

    progress = lines(*(f"line {i}" for i in range(601, 651)))
    assert _update("run", items=progress).status_preview() == (
        "/root/build: 50 lines, last: line 650"
    )
    assert _update("run", items=shell(0), done=True).status_preview() == "/root/build done: exit 0"
    assert _update("run", items=shell(3), done=True).status_preview() == "/root/build done: exit 3"
    assert _update("run", items=shell(124), done=True).status_preview() == (
        "/root/build done: timed out"
    )
    assert _update("run", items=shell(124, True), done=True).status_preview() == (
        "/root/build done: timed out"
    )
    failed = _update("spawn", done=True, error="ValueError: boom")
    assert failed.status_preview() == "/root/build failed: ValueError: boom"
    assert _update("tail", items=["a", "b", "c"]).status_preview() == (
        "/root/build: 3 lines, last: c"
    )
    assert _update("tail", items=["only"]).status_preview() == "/root/build: 1 line, last: only"
    assert _update("every", items=[3]).status_preview() == "/root/build: tick 3"
    assert _update("every", items=[4, 5]).status_preview() == "/root/build: 2 ticks, last: tick 5"
    assert _update("every", items=["check CI"]).status_preview() == "/root/build: check CI"
    assert _update("after", items=["check CI"], done=True).status_preview() == (
        "/root/build done: check CI"
    )
    assert _update("after", items=[None], done=True).status_preview() == "/root/build done: None"
    assert _update("spawn", items=[{"ok": 1}], done=True).status_preview() == (
        "/root/build done: {'ok': 1}"
    )
    assert _update("spawn", items=[1, 2, 3]).status_preview() == ("/root/build: 3 values, last: 3")
    assert _update("spawn", items=[1, 2], done=True).status_preview() == (
        "/root/build done: 2 values, last: 2"
    )
    assert _update("spawn", items=[], done=True).status_preview() == "/root/build done"
    long = _update("tail", items=["x" * 500]).status_preview()
    assert len(long) == 80 and long.endswith("…") and "\n" not in long


async def test_job_updates_carry_their_kind_into_the_unread_list(agent):
    agent.jobs.every(0.01, label="tick")
    await _until(lambda: agent.queue_manager.get_channel("jobs").qsize() >= 4)
    lines = agent.queue_manager.status().splitlines()
    await agent.jobs.cancel("tick")  # also withdraws its queued ticks
    assert lines[:4] == [
        "Unread:",
        "  • /root/tick: tick 1",
        "  • /root/tick: tick 2",
        "  • /root/tick: tick 3",
    ]
    assert re.fullmatch(r"  • … \d+ more", lines[4])
    assert re.fullmatch(r"  • /root/tick · job · \d+s · every 0.01s", lines[6])
    assert lines[5] == "Running:"
    agent.jobs.after(0, label="timer", payload="check CI")
    await _until(lambda: agent.jobs.get("timer").state == "done")
    assert agent.queue_manager.status() == "Unread:\n  • /root/timer done: check CI"
