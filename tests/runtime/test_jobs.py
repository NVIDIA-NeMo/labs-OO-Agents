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


async def _collect_run(agent) -> tuple[list[JobUpdate], JobResult]:
    """Read a run job's progress updates up to and including its JobResult."""
    updates = []
    while True:
        item = await _next(agent)
        if isinstance(item, JobResult):
            return updates, item
        assert isinstance(item, JobUpdate) and not item.final
        updates.append(item)


async def test_run_without_lines_or_seconds_is_quiet(agent):
    await agent.jobs.run("for i in 1 2 3; do echo $i; done", label="q")
    updates, result = await _collect_run(agent)
    assert updates == [] and result.result.stdout == "1\n2\n3\n"


async def test_run_with_progress_delivers_lines_then_its_result(agent):
    await agent.jobs.run("echo a; echo b >&2; echo c", label="s", lines=1)
    updates, result = await _collect_run(agent)
    assert all(len(u.items) == 1 and isinstance(u.items[0], StreamEvent) for u in updates)
    lines = sorted((u.items[0].kind, u.items[0].text) for u in updates)
    assert lines == [("stderr", "b\n"), ("stdout", "a\n"), ("stdout", "c\n")]
    assert result.label == "s" and result.result.returncode == 0
    assert result.result.stdout == "a\nc\n" and result.result.stderr == "b\n"
    assert "a\n" in Path(result.log_path).read_text()


async def test_throttle_by_lines_batches_and_flushes_before_the_result(agent):
    await agent.jobs.run("for i in 1 2 3 4 5; do echo $i; done", label="b", lines=2)
    updates, result = await _collect_run(agent)
    texts = [[e.text for e in u.items] for u in updates]
    assert texts == [["1\n", "2\n"], ["3\n", "4\n"], ["5\n"]]
    assert result.result.stdout == "1\n2\n3\n4\n5\n"


async def test_run_progress_by_seconds(agent):
    await agent.jobs.run(
        "for i in 1 2 3 4 5 6; do echo $i; sleep 0.1; done", label="t", seconds=0.35
    )
    updates, result = await _collect_run(agent)
    assert [e.text for u in updates for e in u.items] == [f"{i}\n" for i in range(1, 7)]
    assert 1 <= len(updates) <= 3 and len(updates[0].items) >= 2
    assert result.result.returncode == 0


async def test_throttle_turns_run_progress_on_and_off(agent, tmp_path):
    go = tmp_path / "go"
    script = (
        "echo quiet1; "
        f"while [ ! -e {go}1 ]; do sleep 0.02; done; echo loud1; echo loud2; "
        f"while [ ! -e {go}2 ]; do sleep 0.02; done; echo quiet2"
    )
    await agent.jobs.run(script, label="r")
    await asyncio.sleep(0.2)
    assert agent.queue_manager.get_channel("jobs").qsize() == 0  # quiet by default
    assert agent.jobs.throttle("r", lines=2).throttle.lines == 2  # progress on
    (tmp_path / "go1").touch()
    update = await _next(agent)
    assert isinstance(update, JobUpdate) and [e.text for e in update.items] == [
        "loud1\n",
        "loud2\n",
    ]
    assert agent.jobs.throttle("r").throttle is None  # progress off again
    (tmp_path / "go2").touch()
    updates, result = await _collect_run(agent)
    assert updates == []
    assert result.result.stdout == "quiet1\nloud1\nloud2\nquiet2\n"


async def test_throttle_by_seconds_batches_slow_output(agent):
    async def ticks():
        for i in range(6):
            yield i
            await asyncio.sleep(0.1)

    agent.jobs.spawn(ticks(), label="tick", seconds=0.35)
    updates = await _stream_updates(agent, "tick")
    assert updates[-1].final
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
    last = await _next(agent)
    assert last.items == [6, 7, 8] and not last.final
    # It ended with nothing batched: no extra empty update, the state says so.
    await _until(lambda: agent.jobs.get("n").state == "done")
    await asyncio.sleep(0.05)
    assert agent.queue_manager.get_channel("jobs").qsize() == 0


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
    updates = await _stream_updates(agent, "p")
    assert [item for u in updates for item in u.items] == ["a", "b"]
    assert not any(u.final for u in updates)  # nothing was left at the end


async def test_throttle_rejects_non_streaming_jobs(agent):
    agent.jobs.after(5, label="r")
    with pytest.raises(ValueError, match="not a streaming job"):
        agent.jobs.throttle("r", lines=2)
    with pytest.raises(KeyError):
        agent.jobs.throttle("missing")
    with pytest.raises(ValueError):
        await agent.jobs.run("true", label="bad", lines=0)


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


async def test_tail_batches_and_rethrottles(agent, tmp_path):
    log = tmp_path / "batch.log"
    log.write_text("")
    agent.jobs.tail(str(log), label="b", lines=3)
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


async def test_after_delivers_one_none_result(agent):
    started = time.monotonic()
    info = agent.jobs.after(0.2, label="timer")
    assert info.kind == "after"
    item = await _next(agent)
    assert isinstance(item, JobResult) and item.label == "timer" and item.result is None
    assert item.error is None and time.monotonic() - started >= 0.2
    assert agent.jobs.get("timer").state == "done"


async def test_after_can_be_cancelled(agent):
    agent.jobs.after(30, label="later")
    assert await agent.jobs.cancel("later") is True
    await asyncio.sleep(0.05)
    assert agent.queue_manager.get_channel("jobs").qsize() == 0


async def test_every_ticks_until_cancelled(agent):
    info = agent.jobs.every(0.05, label="tick")
    assert info.kind == "every"
    assert [(await _next(agent)).items for _ in range(3)] == [[1], [2], [3]]
    assert await agent.jobs.cancel("tick") is True
    assert agent.jobs.get("tick").state == "cancelled"
    with pytest.raises(ValueError):
        agent.jobs.every(0, label="bad")


async def test_every_supports_throttle(agent):
    agent.jobs.every(0.03, label="t")
    first = await _next(agent)
    agent.jobs.throttle("t", lines=3)
    batched = await _next(agent)
    while not batched.items or len(batched.items) < 3:  # a flush of the old batch may come first
        batched = await _next(agent)
    assert len(batched.items) == 3
    assert batched.items == list(range(batched.items[0], batched.items[0] + 3))
    assert batched.items[0] > first.items[0]
    await agent.jobs.cancel("t")


# ---- process isolation (ported from the removed monitor() tests) ----


async def test_commands_run_in_their_own_process_group(agent):
    await agent.jobs.run("ps -o pgid= -p $$", label="pg")
    item = await _next(agent)
    assert int(item.result.stdout.strip()) != os.getpgid(os.getpid())


async def test_cancel_kills_the_whole_process_group(agent, tmp_path):
    pidfile = tmp_path / "child"
    await agent.jobs.run(f"sleep 999 & echo $! > {pidfile}; echo ready; wait", label="grp")
    await _until(lambda: pidfile.exists() and pidfile.read_text().strip())
    child = int(pidfile.read_text())
    assert _alive(child)
    await agent.jobs.cancel("grp")
    await _until(lambda: not _alive(child))


async def test_concurrent_jobs_do_not_interfere(agent):
    await agent.jobs.run("echo A1; sleep 0.1; echo A2", label="a")
    await agent.jobs.run("echo B1; sleep 0.1; echo B2", label="b")
    items = {item.label: item for item in [await _next(agent), await _next(agent)]}
    assert items["a"].result.stdout == "A1\nA2\n"
    assert items["b"].result.stdout == "B1\nB2\n"


async def test_cancelling_one_job_leaves_the_other_running(agent):
    await agent.jobs.run("sleep 999", label="victim")
    await agent.jobs.run("sleep 0.3; echo survived", label="survivor")
    await agent.jobs.cancel("victim")
    item = await _next(agent)
    assert item.label == "survivor" and item.result.stdout == "survived\n"


async def test_undecodable_output_does_not_end_the_stream(agent):
    await agent.jobs.run(r"printf 'first\ncaf\xe9\nlast\n'", label="bytes", lines=1)
    updates, result = await _collect_run(agent)
    texts = [e.text for u in updates for e in u.items]
    assert texts == ["first\n", "caf\ufffd\n", "last\n"]
    assert result.result.stdout == "first\ncaf\ufffd\nlast\n"


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
    assert "  • tests · running · full run before pushing" in status
    assert "  • timer · running · after 300s" in status
    noisy = next(line for line in status.splitlines() if "noisy" in line)
    assert noisy.endswith("…") and len(noisy) <= 4 + 100
    assert "→ jobs" not in status
    for label in ("tests", "timer", "noisy"):
        await agent.jobs.cancel(label)


# ---- timer payloads ----


async def test_after_delivers_its_payload(agent):
    info = agent.jobs.after(0.05, label="check-ci", payload="check CI on PR #462 and report")
    assert info.payload == "check CI on PR #462 and report"
    item = await _next(agent)
    assert isinstance(item, JobResult) and item.result == "check CI on PR #462 and report"
    assert agent.jobs.get("check-ci").payload == "check CI on PR #462 and report"


async def test_every_keeps_ticks_and_exposes_its_payload(agent):
    payload = {"task": "poll CI", "pr": 462}
    agent.jobs.every(0.05, label="poll", payload=payload)
    assert [(await _next(agent)).items for _ in range(2)] == [[1], [2]]
    assert agent.jobs.get("poll").payload == payload
    assert agent.jobs.list()[-1].payload == payload
    await agent.jobs.cancel("poll")


async def test_payload_defaults_to_none(agent):
    assert agent.jobs.every(30, label="e").payload is None
    assert (await agent.jobs.run("true", label="r")).payload is None
    agent.jobs.after(0.01, label="a")
    items = [await _next(agent), await _next(agent)]
    assert next(i for i in items if i.label == "a").result is None
    await agent.jobs.cancel("e")


# ---- replace=True ----


async def test_replace_restarts_a_running_run_job(agent, tmp_path):
    pidfile = tmp_path / "old.pid"
    await agent.jobs.run(f"echo $$ > {pidfile}; exec sleep 60", label="srv", lines=1)
    await _until(lambda: pidfile.exists() and pidfile.read_text().strip())
    old_pid = int(pidfile.read_text())
    with pytest.raises(ValueError, match="already running"):
        await agent.jobs.run("true", label="srv")
    info = await agent.jobs.run("echo new", label="srv", replace=True)
    assert info.state == "running" and info.target == "echo new"
    await _until(lambda: not _alive(old_pid))
    item = await _next(agent)
    while isinstance(item, JobUpdate):  # the new job's progress, if any
        item = await _next(agent)
    assert isinstance(item, JobResult) and item.result.stdout == "new\n"
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
    assert (await _next(agent)).result == "p"


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
    qm.set_cancel_hint("mesh", "await self.mesh.stop()  # {label}")
    assert "💡 await self.mesh.stop()  # inbox" in qm.status()
    await qm.shutdown()


# ---- lost on restart ----


async def test_jobs_running_at_a_snapshot_are_reported_once_after_restore(tmp_path):
    from nooa.interactive import InteractiveAgent
    from nooa.runtime.jobs import JobsLost
    from nooa.storage.snapshot import AgentSnapshot
    from nooa.unifiedllm import FakeLLMClient

    class Host(InteractiveAgent, llm=FakeLLMClient()):
        pass

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
