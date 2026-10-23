# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``Shell`` work under the ``WorkManager``: real bash, one-shot and persistent."""

import asyncio
import os
from pathlib import Path

import pytest
from nooa_atom.agent.work_manager import WorkManager
from nooa_atom.agent.work_shell import CommandResult, Shell

from nooa.runtime.channels import QueueManager


@pytest.fixture
async def manager(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    manager = WorkManager(QueueManager().queue("jobs"), scratch=scratch)
    yield manager
    await manager.aclose()


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return (
        Path(f"/proc/{pid}").exists()
        and "Z" not in Path(f"/proc/{pid}/stat").read_text().split()[2]
    )


async def test_one_shot_result_and_streams(manager, tmp_path):
    r = await manager.launch(Shell(tmp_path).run("echo out; echo err >&2; exit 3"))
    assert isinstance(r, CommandResult)
    assert (r.returncode, r.timed_out, r.stdout, r.stderr) == (3, False, "out\n", "err\n")
    assert r.stdout_path is None and r.stderr_path is None  # nothing was cut
    r = await Shell(tmp_path).run("cat; pwd", stdin="piped\n")  # no manager needed
    assert r.stdout == f"piped\n{tmp_path}\n"


async def test_one_shot_timeout(manager):
    r = await manager.launch(Shell(timeout=0.3).run("sleep 30; echo never"))
    assert (r.returncode, r.timed_out, r.stdout) == (124, True, "")


async def test_long_output_bounded_with_retained_file(manager):
    r = await manager.launch(Shell().run("seq 1 200000"))
    assert len(r.stdout) < 11_000 and "bytes omitted; full output:" in r.stdout
    assert r.stdout.startswith("1\n2\n") and r.stdout.endswith("199999\n200000\n")
    assert r.stdout_path and Path(r.stdout_path).read_text().count("\n") == 200000
    # Kept after the result was consumed; removed with the manager.
    await asyncio.sleep(0)
    assert Path(r.stdout_path).exists()
    await manager.aclose()
    assert not Path(r.stdout_path).exists()


async def test_one_shot_without_manager_keeps_no_file():
    r = await Shell().run("seq 1 100000")
    assert r.stdout_path is None and "full output not kept" in r.stdout


async def test_one_shot_cancel_kills_process_group(manager, tmp_path):
    marker = tmp_path / "pid"
    job = manager.launch(Shell().run(f"sleep 60 & echo $! > {marker}; wait"))
    for _ in range(100):
        if marker.exists() and marker.read_text().strip():
            break
        await asyncio.sleep(0.05)
    pid = int(marker.read_text())
    await job.cancel()
    await asyncio.sleep(0.1)
    assert not alive(pid)


async def test_persistent_state_across_sends(manager, tmp_path):
    (tmp_path / "sub").mkdir()
    sh = manager.launch(Shell(tmp_path), name="sh")
    await sh.send("cd sub && export EXPORTED=1 && LOCAL=2")
    r = await sh.send('echo "$EXPORTED $LOCAL $PWD"; echo e >&2; false')
    assert r.stdout == f"1 2 {tmp_path / 'sub'}\n"
    assert (r.returncode, r.stderr) == (1, "e\n")
    with pytest.raises(TypeError, match="resource"):
        await sh


async def test_persistent_requests_run_in_order(manager, tmp_path):
    sh = manager.launch(Shell(tmp_path), name="sh")
    requests = [sh.send(f"sleep 0.0{5 - i}; echo {i} >> order.txt; X={i}") for i in range(5)]
    sh.send("cd /tmp")  # state from one request is seen by the next
    results = await asyncio.gather(*(asyncio.ensure_future(r._wait()) for r in requests))
    assert all(r.returncode == 0 for r in results)
    r = await sh.send("echo $X; pwd")
    assert r.stdout == "4\n/tmp\n"
    assert (tmp_path / "order.txt").read_text() == "0\n1\n2\n3\n4\n"  # FIFO despite sleeps


async def test_independent_shells(manager, tmp_path):
    a = manager.launch(Shell(tmp_path), name="a")
    b = manager.launch(Shell(tmp_path), name="b")
    await a.send("export V=a; cd /")
    r = await b.send('echo "[$V] $PWD"')
    assert r.stdout == f"[] {tmp_path}\n"


async def test_cancel_request_keeps_session_state(manager, tmp_path):
    sh = manager.launch(Shell(tmp_path), name="sh")
    await sh.send("cd / && KEEP=yes")
    slow = sh.send("sleep 60")
    await asyncio.sleep(0.3)
    await asyncio.wait_for(slow.cancel(), 10)
    assert slow.status == "cancelled" and slow.error is None
    r = await sh.send('echo "$KEEP $PWD"')
    assert r.stdout == "yes /\n" and "restarted" not in r.stderr


async def test_cancel_builtin_loop_restarts_with_explicit_note(manager, tmp_path):
    sh = manager.launch(Shell(tmp_path), name="sh")
    await sh.send("cd / && LOST=1")
    spin = sh.send("while true; do :; done")
    await asyncio.sleep(0.3)
    await asyncio.wait_for(spin.cancel(), 10)
    assert "restarted" in (spin.error or "")
    r = await sh.send('echo "[$LOST] $PWD"')
    assert r.stdout == "[] /\n"  # cwd kept, variables lost
    assert "the shell restarted" in r.stderr  # said once, on the next result
    assert "restarted" not in (await sh.send("true")).stderr


async def test_persistent_timeout_recovers(manager, tmp_path):
    sh = manager.launch(Shell(tmp_path, timeout=0.5), name="sh")
    await sh.send("KEEP=1")
    r = await sh.send("sleep 30")
    assert (r.returncode, r.timed_out) == (124, True)
    r = await sh.send("echo $KEEP")
    assert r.stdout == "1\n"  # interrupted, not restarted


async def test_entry_failure_and_cancel_before_ready(manager, tmp_path):
    bad = manager.launch(Shell(tmp_path / "missing"), name="bad")
    request = bad.send("true")
    with pytest.raises(RuntimeError, match="did not start"):
        await asyncio.wait_for(request._wait(), 10)
    assert bad.status == "failed"
    early = manager.launch(Shell(tmp_path), name="early")
    await early.cancel()  # before bash finished starting
    assert early.status in ("cancelled", "done")


async def test_aclose_ends_bash_and_the_shell_is_reusable(manager, tmp_path):
    shell = Shell(tmp_path)
    sh = manager.launch(shell, name="sh")
    pid = int((await sh.send("echo $$")).stdout)
    assert alive(pid)
    await manager.aclose()
    await asyncio.sleep(0.1)
    assert not alive(pid)
    async with shell:  # the inert config can be entered again
        assert (await shell.send("echo again")).stdout == "again\n"


async def test_shell_reports_confirmed_cwd_and_busy_state(manager, tmp_path):
    shell = Shell(tmp_path / "later")  # constructor cwd is config, not a claim about bash
    assert shell.summary() == "not started" and shell.current_dir is None
    (tmp_path / "later").mkdir()
    sh = manager.launch(shell, name="sh")
    r = await sh.send("mkdir -p d && cd d")
    assert r.cwd == shell.current_dir == str(tmp_path / "later" / "d")
    assert sh.summary == f"idle · in {tmp_path / 'later' / 'd'}"
    slow = sh.send("cd / && sleep 60")
    await asyncio.sleep(0.3)
    assert sh.summary.startswith("running $ cd / && sleep 60 · was in ")  # pending, not guessed
    assert slow.summary == "$ cd / && sleep 60"
    await slow.cancel()
    assert shell.current_dir == "/"  # confirmed by bash once the command ended
    spin = sh.send("while :; do :; done")
    await asyncio.sleep(0.3)
    await spin.cancel()
    assert "restarted, variables lost" in sh.summary
    await sh.send("true")
    assert "restarted" not in sh.summary
    one = await manager.launch(Shell(tmp_path).run("cd /; pwd"))
    assert one.stdout == "/\n" and one.cwd == str(tmp_path)  # a one-shot cd does not last
    await manager.aclose()
    assert shell.summary() == "closed"
