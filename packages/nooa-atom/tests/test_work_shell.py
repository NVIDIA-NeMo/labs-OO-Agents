# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``Shell`` work under the ``WorkManager``: real bash, one-shot and persistent."""

import asyncio
import contextlib
import json
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
    """Whether ``pid`` is a live (non-zombie) process; safe if it exits meanwhile."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return False  # exited (and was reaped) between the two checks
    return "Z" not in stat.split()[2]


async def until_dead(*pids: int, timeout: float = 10.0) -> None:
    """Wait, bounded, until none of ``pids`` is alive (killing and reaping are asynchronous)."""
    async with asyncio.timeout(timeout):
        while any(alive(pid) for pid in pids):
            await asyncio.sleep(0.02)


async def test_one_shot_result_and_streams(manager, tmp_path):
    r = await manager.launch(Shell(tmp_path).run("echo out; echo err >&2; (exit 3)"))
    assert isinstance(r, CommandResult)
    assert (r.returncode, r.timed_out, r.stdout, r.stderr) == (3, False, "out\n", "err\n")
    assert r.stdout_path is None and r.stderr_path is None  # nothing was cut
    r = await Shell(tmp_path).run("cat; pwd", stdin="piped\n")  # no manager needed
    assert r.stdout == f"piped\n{tmp_path}\n"


async def test_one_shot_timeout(manager):
    r = await manager.launch(Shell().run("sleep 30; echo never", timeout=0.3))
    assert (r.returncode, r.timed_out, r.stdout) == (124, True, "")
    # The result says how to go on: a larger timeout=, or the background.
    assert "[timed out after 0.3 s: pass a larger timeout= for a longer command" in r.stderr
    assert "self.jobs.launch(sh.run(cmd, timeout=...), name=...)]" in r.stderr


NOTICE = "Full output: await self.files.read({})"


def cut_notice(path: str, total: int) -> str:
    """The first lines of a text cut at 1 MiB (256 Ki head, 768 Ki tail characters)."""
    return (
        "<truncated-output>\n"
        f"Output too large ({total:,} chars). Showing first 262,144 and last 786,432 chars.\n"
        + NOTICE.format(json.dumps(path))
        + "\n\n"
    )


async def test_long_output_bounded_with_retained_file(manager):
    r = await manager.launch(Shell().run("seq 1 200000"))  # 1,288,895 bytes
    assert r.stdout_path and r.stdout.startswith(cut_notice(r.stdout_path, 1_288_895) + "1\n2\n")
    assert len(r.stdout) < 1_049_000 and r.stdout.endswith("199999\n200000\n\n</truncated-output>")
    assert Path(r.stdout_path).read_text().count("\n") == 200000
    # Kept after the result was consumed; removed with the manager.
    await asyncio.sleep(0)
    assert Path(r.stdout_path).exists()
    await manager.aclose()
    assert not Path(r.stdout_path).exists()


async def test_output_up_to_1_mib_is_whole_in_the_result(tmp_path):
    out = "".join(f"line {i:05d} " + "y" * 25 + "\n" for i in range(2500))  # 92,500 bytes
    (tmp_path / "o").write_text(out)
    r = await Shell(tmp_path).run(f"cat {tmp_path / 'o'}")
    assert len(out.encode()) == 92_500 and r.stdout == out
    assert len(r.stdout.splitlines()) == 2500 and r.stdout_path is None
    exact = "z" * (1024 * 1024 - 1) + "\n"  # exactly 1 MiB: still whole
    (tmp_path / "o").write_text(exact)
    r = await Shell(tmp_path).run(f"cat {tmp_path / 'o'}")
    assert r.stdout == exact and r.stdout_path is None


async def test_multibyte_text_split_across_reads_stays_whole(tmp_path):
    # 3-byte characters: the 64 KiB reads split many of them.
    out = "".join(f"{i:04d} 漢字テスト🚀\n" for i in range(9000))  # ~230 KB
    (tmp_path / "o").write_text(out)
    r = await Shell(tmp_path).run(f"cat {tmp_path / 'o'}")
    assert r.stdout == out and "\ufffd" not in r.stdout
    big = out * 12  # > 1 Mi characters: cut, the file byte-exact
    (tmp_path / "o").write_text(big)
    r = await Shell(tmp_path).run(f"cat {tmp_path / 'o'}")
    assert r.stdout_path and Path(r.stdout_path).read_bytes() == big.encode()
    assert "\ufffd" not in r.stdout


async def test_one_line_over_the_limit_is_cut_and_kept_byte_exact(tmp_path):
    data = bytes(range(256)) * 8192  # 2 MiB without a newline, invalid UTF-8 included
    (tmp_path / "o").write_bytes(data)
    r = await Shell(tmp_path).run(f"cat {tmp_path / 'o'}")
    assert r.stdout_path and Path(r.stdout_path).read_bytes() == data
    assert (
        r.stdout.startswith("<truncated-output>\n")
        and NOTICE.format(json.dumps(r.stdout_path)) in r.stdout.split("\n", 3)[2]
    )


async def test_direct_run_keeps_long_output_until_aclose(tmp_path):
    sh = Shell(tmp_path)
    out = "".join(f"line {i:05d} " + "x" * 18 + "\n" for i in range(40_000))  # 1.2 MB
    cmd = f"cat {tmp_path / 'o'}; cat {tmp_path / 'o'} >&2"
    (tmp_path / "o").write_text(out)
    r = await sh.run(cmd)
    for text, path in ((r.stdout, r.stdout_path), (r.stderr, r.stderr_path)):
        assert path and text.startswith(cut_notice(path, 1_200_000) + "line 00000 ")
        assert text.endswith("line 39999 " + "x" * 18 + "\n\n</truncated-output>")
        assert Path(path).read_text() == out
    short = await sh.run("echo hi")
    assert (short.stdout, short.stdout_path, short.stderr_path) == ("hi\n", None, None)
    assert sorted(p.name for p in Path(r.stdout_path).parent.iterdir()) == [
        "cmd-1.stderr",
        "cmd-1.stdout",
    ]  # short output keeps no file
    await sh.aclose()
    assert not Path(r.stdout_path).exists() and not Path(r.stderr_path).exists()


async def test_direct_run_output_limit_marker():
    sh = Shell(max_output=1000)
    r = await sh.run("head -c 2000000 /dev/zero | tr '\\0' a")
    assert r.stdout_path and Path(r.stdout_path).stat().st_size == 1000
    assert (
        NOTICE.format(json.dumps(r.stdout_path)) + "\nThe file is incomplete: it holds the first "
        "1,000 bytes; 1,999,000 more bytes were over its 1,000-byte limit.\n"
    ) in r.stdout
    await sh.aclose()
    assert not Path(r.stdout_path).exists()


async def test_dropped_unowned_shell_keeps_files_while_the_result_lives():
    import gc

    r = await Shell().run("seq 1 300000")  # the temporary Shell is dropped
    gc.collect()
    path = Path(r.stdout_path)
    assert path.read_text().count("\n") == 300000
    del r
    gc.collect()
    assert not path.exists()


async def test_dropped_owned_shell_keeps_files_until_the_owner_closes():
    import gc

    from nooa_atom.agent.work_shell import ShellOwner, shell_owner

    owner = ShellOwner()
    token = shell_owner.set(owner)
    try:
        r = await Shell().run("seq 1 300000")  # a cell's one-liner
        pid = int((await Shell().run("echo $$")).stdout)
    finally:
        shell_owner.reset(token)
    gc.collect()
    await until_dead(pid)  # the dropped Shells ended their bash
    path = Path(r.stdout_path)
    del r
    gc.collect()  # neither the Shell nor the result is needed: the owner keeps it
    assert path.read_text().count("\n") == 300000
    await owner.aclose()
    assert not path.exists()


async def test_owned_shell_aclose_removes_its_files():
    from nooa_atom.agent.work_shell import ShellOwner, shell_owner

    owner = ShellOwner()
    token = shell_owner.set(owner)
    try:
        sh = Shell()
        r = await sh.run("seq 1 300000")
    finally:
        shell_owner.reset(token)
    assert Path(r.stdout_path).exists()
    await sh.aclose()
    assert not Path(r.stdout_path).exists()
    await owner.aclose()


async def test_one_shot_cancel_kills_process_group(manager, tmp_path):
    marker = tmp_path / "pid"
    job = manager.launch(Shell().run(f"sleep 60 & echo $! > {marker}; wait"))
    for _ in range(100):
        if marker.exists() and marker.read_text().strip():
            break
        await asyncio.sleep(0.05)
    pid = int(marker.read_text())
    await job.cancel()
    await until_dead(pid)


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
    shell = Shell(tmp_path)
    sh = manager.launch(shell, name="sh")
    await sh.send("KEEP=1")
    r = await manager.launch(
        shell.run("sleep 30", timeout=0.5)
    )  # a launched call: its own deadline
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
    await until_dead(pid)
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
    assert slow.summary == f"$ cd / && sleep 60 · cwd {tmp_path / 'later' / 'd'}"
    await slow.cancel()
    assert shell.current_dir == "/"  # confirmed by bash once the command ended
    spin = sh.send("while :; do :; done")
    await asyncio.sleep(0.3)
    await spin.cancel()
    assert "restarted, variables lost" in sh.summary
    await sh.send("true")
    assert "restarted" not in sh.summary
    one = await manager.launch(Shell(tmp_path).run("cd /; pwd"))
    assert one.stdout == "/\n" and one.cwd == "/"  # the shell's real directory after it
    await manager.aclose()
    assert shell.summary() == "closed"


# ---- regressions from the independent review of 83ae65c77 ----


async def test_stdin_feed_shares_the_command_deadline():
    started = asyncio.get_running_loop().time()
    r = await asyncio.wait_for(Shell().run("sleep 30", stdin="x" * (8 << 20), timeout=1), 10)
    assert r.timed_out and r.returncode == 124
    assert asyncio.get_running_loop().time() - started < 8


async def test_bash_exit_is_reported_and_cancel_still_interrupts(manager, tmp_path):
    shell = Shell(tmp_path)
    sh = manager.launch(shell, name="sh")
    await sh.send("export FOO=1")
    ended = await sh.send("exit 0")
    assert "bash ended during this command" in ended.stderr
    assert "restarted, variables lost" in sh.summary
    r = await sh.send('echo "FOO=$FOO"')
    assert r.stdout == "FOO=\n"
    slow = sh.send("sleep 30")
    await asyncio.sleep(0.5)
    await asyncio.wait_for(slow.cancel(), 10)
    assert slow.error is None  # interrupted in the new bash, not restarted
    assert "restarted" not in (await sh.send("true")).stderr
    pid = int((await sh.send("echo $$")).stdout)
    assert shell._bash is not None and shell._bash[0] == pid


async def test_same_shell_cannot_be_entered_twice(manager, tmp_path):
    shell = Shell(tmp_path)
    first = manager.launch(shell, name="first")
    second = manager.launch(shell, name="second")
    request = second.send("true")
    with pytest.raises(RuntimeError, match="already running|did not start"):
        await asyncio.wait_for(request._wait(), 10)
    assert second.status == "failed"
    pid = int((await first.send("echo $$")).stdout)
    await manager.aclose()
    await until_dead(pid)


async def test_output_limit_is_enforced_while_writing(manager, tmp_path):
    cmd = "head -c 2000000 /dev/zero | tr '\\0' a"
    r = await manager.launch(Shell(max_output=1000).run(cmd))
    assert Path(r.stdout_path).stat().st_size == 1000
    assert "it holds the first 1,000 bytes; 1,999,000 more bytes were over" in r.stdout
    assert r.stdout.endswith("a" * 100 + "\n</truncated-output>")  # the tail is still shown
    sh = manager.launch(Shell(tmp_path, max_output=1000), name="sh")
    r = await sh.send("head -c 2000000 /dev/zero | tr '\\0' b; echo done >&2")
    assert Path(r.stdout_path).stat().st_size == 1000 and r.stderr == "done\n"
    r = await manager.launch(Shell(max_output=10_000).run("yes", timeout=1))
    assert r.timed_out and Path(r.stdout_path).stat().st_size == 10_000


async def test_background_job_output_does_not_block_the_session(manager, tmp_path):
    sh = manager.launch(Shell(tmp_path, max_output=2000), name="sh")
    r = await asyncio.wait_for(sh.send("(sleep 1; yes | head -c 100000) & echo started"), 10)
    assert r.stdout == "started\n"  # returned before the background job wrote anything
    await asyncio.sleep(2)  # the background writer is drained (and capped), not blocked
    r = await asyncio.wait_for(sh.send("jobs -r | wc -l"), 10)
    assert r.stdout.strip() == "0"


async def test_background_process_does_not_hold_the_command_and_ends_with_the_shell(tmp_path):
    """A background process that inherits stdout/stderr does not keep run() waiting."""
    loop = asyncio.get_running_loop()
    started = loop.time()
    shell = Shell(tmp_path)
    r = await asyncio.wait_for(shell.run("sleep 30 & echo $! >bg.pid; echo hi", timeout=2), 10)
    assert (r.returncode, r.timed_out, r.stdout) == (0, False, "hi\n")
    assert loop.time() - started < 1.5
    pid = int((tmp_path / "bg.pid").read_text())
    try:
        assert alive(pid)  # run() did not kill it: it lives as long as the shell
        await shell.aclose()  # deterministic: bash and its background processes end
        await until_dead(pid)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, 9)
    # No transport or pump is left behind: nothing still reads its pipe.
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    assert not pending


# ---- one bash per Shell object: run() is persistent ----


async def test_run_reuses_the_objects_bash(tmp_path):
    (tmp_path / "sub").mkdir()
    async with Shell(tmp_path) as sh:
        await sh.run("cd sub")
        await sh.run("export EXPORTED=1")
        await sh.run("UNEXPORTED=2; f() { echo fn; }")
        r = await sh.run('echo "$EXPORTED $UNEXPORTED $PWD $(f)"')
        assert r.stdout == f"1 2 {tmp_path / 'sub'} fn\n" and r.cwd == str(tmp_path / "sub")
        assert sh.current_dir == str(tmp_path / "sub")
        pids = {(await sh.run("echo $$")).stdout for _ in range(3)}
        assert len(pids) == 1  # one bash, not one per call
        r = await sh.run("cat; echo [$?]", stdin="in\n")
        assert r.stdout == "in\n[0]\n"
        r = await asyncio.wait_for(sh.run("cat; echo none"), 10)  # no stdin: reads nothing
        assert r.stdout == "none\n"


async def test_constructor_is_inert_and_objects_are_isolated(tmp_path):
    a, b = Shell(tmp_path), Shell(tmp_path)
    assert a.summary() == "not started" and a._session is None
    try:
        await a.run("export V=a; W=a; cd /")
        r = await b.run('echo "[$V][$W] $PWD"')
        assert r.stdout == f"[][] {tmp_path}\n"
        assert (await a.run("echo $V$W")).stdout == "aa\n"
    finally:
        await a.aclose()
        await b.aclose()


async def test_concurrent_runs_are_fifo(tmp_path):
    async with Shell(tmp_path) as sh:
        calls = [sh.run(f"sleep 0.0{5 - i}; echo {i} >> order.txt; X={i}") for i in range(5)]
        results = await asyncio.gather(*calls, sh.run("echo $X"))
        assert results[-1].stdout == "4\n"  # queued last, saw the state of the others
        assert (tmp_path / "order.txt").read_text() == "0\n1\n2\n3\n4\n"


async def test_run_and_send_share_one_bash(manager, tmp_path):
    shell = Shell(tmp_path)
    await shell.run("cd / && A=1")  # started directly
    sh = manager.launch(shell, name="sh")
    r = await sh.send('echo "$A $PWD"; B=2')
    assert r.stdout == "1 /\n"
    r = await shell.run('echo "$B"')  # direct use while launched: the same bash, in order
    assert r.stdout == "2\n"
    pid = int((await shell.run("echo $$")).stdout)
    await sh.cancel()  # the job's end closes the shell it entered
    await until_dead(pid)
    assert shell.summary() == "closed"


async def test_timeout_stops_the_whole_command_and_keeps_state(tmp_path):
    async with Shell(tmp_path) as sh:
        await sh.run("KEEP=1")
        r = await sh.run("sleep 30; touch after; echo never", timeout=0.5)
        assert (r.returncode, r.timed_out, r.stdout) == (124, True, "")
        assert not (tmp_path / "after").exists()  # the rest of the command was skipped
        r = await sh.run('echo "$KEEP"; echo $?; echo "[$__nooa_stop]"')
        assert r.stdout == "1\n0\n[]\n"  # state kept; the next command runs whole


async def test_cancelled_run_keeps_state_and_the_next_command_runs(tmp_path):
    async with Shell(tmp_path) as sh:
        await sh.run("cd / && KEEP=yes")
        task = asyncio.create_task(sh.run(f"sleep 60; touch {tmp_path}/after"))
        await asyncio.sleep(0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        r = await sh.run('echo "$KEEP $PWD"')
        assert r.stdout == "yes /\n" and "restarted" not in r.stderr
        assert not (tmp_path / "after").exists()


async def test_exit_ends_the_bash_and_says_so(tmp_path):
    async with Shell(tmp_path) as sh:
        await sh.run("export FOO=1; cd /")
        r = await sh.run("exit 3")
        assert "bash ended during this command" in r.stderr
        r = await sh.run('echo "[$FOO] $PWD"')
        assert r.stdout == "[] /\n"  # a new bash: cwd kept, variables lost


async def test_named_job_of_run_result_and_event(manager, tmp_path):
    shell = Shell(tmp_path)
    try:
        await shell.run("X=kept")
        job = manager.launch(shell.run("sleep 0.2; echo $X; Y=new"), name="tests")
        r = await job
        assert job.status == "done" and r.stdout == "kept\n"
        assert (await shell.run("echo $Y")).stdout == "new\n"
        manager.launch(shell.run("echo later"), name="later")  # nobody awaits: an event
        outbox = manager._outbox
        event = await asyncio.wait_for(outbox.get(), 10)
        assert (event.name, event.kind, event.value.stdout) == ("later", "done", "later\n")
    finally:
        await shell.aclose()


async def test_coroutine_job_summary_uses_cwd_when_command_starts(manager, tmp_path, monkeypatch):
    """A queued command must not inherit the cwd from before the preceding cd."""
    sub = tmp_path / "sub"
    sub.mkdir()
    async with Shell(tmp_path) as shell:
        await shell.run("true")
        original = shell._session.run_with_timeout_flag
        started = [asyncio.Event(), asyncio.Event()]
        release = [asyncio.Event(), asyncio.Event()]
        calls = 0

        async def gated(*args, **kwargs):
            nonlocal calls
            index = calls
            calls += 1
            started[index].set()
            await release[index].wait()
            return await original(*args, **kwargs)

        monkeypatch.setattr(shell._session, "run_with_timeout_flag", gated)
        first = manager.launch(shell.run("cd sub"), name="change_directory")
        second = None
        try:
            await asyncio.wait_for(started[0].wait(), 5)
            queued = asyncio.Event()

            async def queued_run():
                queued.set()
                return await shell.run("pwd")

            second = manager.launch(queued_run(), name="probe")
            await asyncio.wait_for(queued.wait(), 5)
            # No guessed directory while queued; it names what it waits behind.
            assert second.summary == "$ pwd · queued behind `cd sub` (job change_directory)"
            release[0].set()
            await first
            await asyncio.wait_for(started[1].wait(), 5)
            assert second.status == "running"
            assert second.summary == f"$ pwd · cwd {sub}"
            assert manager.get("probe").summary == second.summary
            release[1].set()
            assert (await second).stdout == f"{sub}\n"
        finally:
            for gate in release:
                gate.set()
            await asyncio.wait_for(
                asyncio.gather(
                    *[job for job in (first, second) if job is not None], return_exceptions=True
                ),
                10,
            )


async def test_dropped_shell_ends_its_bash(tmp_path):
    pid = int((await Shell(tmp_path).run("echo $$")).stdout)  # a temporary
    await until_dead(pid)


async def test_owner_closes_its_shells_and_their_background_processes(manager, tmp_path):
    from nooa_atom.agent.work_shell import ShellOwner, shell_owner

    owner = ShellOwner()
    token = shell_owner.set(owner)
    try:
        kept = Shell(tmp_path)  # held, as a cell local would be
        bg = int((await kept.run("sleep 60 & echo $!")).stdout)
        bash = int((await kept.run("echo $$")).stdout)
        launched = manager.launch(Shell(tmp_path), name="sh")  # the job's context: same owner
        other = int((await launched.send("echo $$")).stdout)
        assert len(owner) == 2
    finally:
        shell_owner.reset(token)
    await owner.aclose()
    await until_dead(bg, bash, other)
    assert kept.summary() == "closed"
    assert (await kept.run("echo again")).stdout == "again\n"  # usable again: a new bash
    await kept.aclose()


# ---- the wrapper under the user's shell options and traps ----


async def test_set_e_persists_and_ordinary_commands_still_succeed(tmp_path):
    async with Shell(tmp_path) as sh:
        assert (await sh.run("set -e")).returncode == 0
        r = await sh.run("printf ok")
        assert (r.returncode, r.stdout, r.stderr) == (0, "ok", "")
        r = await sh.run("false && true; echo [$?]")  # not an errexit failure
        assert (r.returncode, r.stdout) == (0, "[1]\n")
        r = await sh.run("! true")  # a nonzero status that errexit ignores: bash stays
        assert r.returncode == 1 and "bash ended" not in r.stderr
        r = await sh.run("[ -f missing ] && . missing")  # a common last line under set -e
        assert r.returncode == 1 and "bash ended" not in r.stderr
        r = await sh.run("case $- in *e*) echo on;; esac; X=1")
        assert r.stdout == "on\n"  # the option is the user's, in force again
        r = await sh.run("false; echo never")  # errexit ends the shell, as in a terminal
        assert r.returncode == 1 and r.stdout == ""
        assert "bash ended during this command (exit 1)" in r.stderr
        r = await sh.run('echo "[$X]"; case $- in *e*) echo on;; esac')
        assert (r.returncode, r.stdout) == (0, "[]\n")  # a new bash: options, variables gone


async def test_nonzero_statuses_are_reported_as_is(tmp_path):
    async with Shell(tmp_path) as sh:
        assert (await sh.run("false")).returncode == 1
        assert (await sh.run("(exit 7)")).returncode == 7
        assert (await sh.run("sh -c 'kill -TERM $$'")).returncode == 143
        r = await sh.run("exit 7")
        assert r.returncode == 7 and "bash ended during this command (exit 7)" in r.stderr
        r = await sh.run("exit 0")
        assert r.returncode == 0 and "bash ended during this command (exit 0)" in r.stderr
        assert (await sh.run("echo back")).stdout == "back\n"


async def test_nounset_and_pipefail(tmp_path):
    async with Shell(tmp_path) as sh:
        await sh.run("set -u -o pipefail")
        r = await sh.run("false | true; echo [$?]; echo ${DEFINED:-d}")
        assert (r.returncode, r.stdout, r.stderr) == (0, "[1]\nd\n", "")
        r = await sh.run("set -o | grep -E '^(nounset|pipefail)' | tr -s ' \t' ' '")
        assert r.stdout == "nounset on\npipefail on\n"
        r = await sh.run("set -eu -o pipefail; printf ok; Y=2")
        assert (r.returncode, r.stdout) == (0, "ok")
        assert (await sh.run('echo "$Y"')).stdout == "2\n"
        r = await sh.run("echo $NOT_SET")  # nounset ends a non-interactive bash
        assert r.returncode != 0 and "bash ended during this command" in r.stderr


async def test_user_traps_persist_and_work(tmp_path):
    async with Shell(tmp_path) as sh:
        await sh.run("trap 'echo got >>usr1.log' USR1; trap ':' DEBUG; trap 'echo bye' EXIT")
        r = await sh.run("trap -p USR1 DEBUG EXIT")
        assert r.stdout == (
            "trap -- 'echo got >>usr1.log' SIGUSR1\ntrap -- ':' DEBUG\ntrap -- 'echo bye' EXIT\n"
        )
        await sh.run("kill -USR1 $$; true")
        assert (tmp_path / "usr1.log").read_text() == "got\n"  # the user's trap, not ours
        await sh.run("trap - DEBUG")
        r = await sh.run("trap -p DEBUG; trap -p USR1")
        assert r.stdout == "trap -- 'echo got >>usr1.log' SIGUSR1\n"


async def test_user_usr1_trap_makes_an_interrupt_restart_explicitly(tmp_path):
    async with Shell(tmp_path) as sh:
        await sh.run("trap ':' USR1; KEEP=1")
        r = await sh.run(f"sleep 30; touch {tmp_path}/after", timeout=0.5)
        assert r.timed_out and "the shell restarted" in r.stderr  # no silent half-stop
        assert not (tmp_path / "after").exists()
        r = await sh.run('echo "[$KEEP]"; trap -p USR1')
        assert r.stdout.startswith("[]\n") and "':'" not in r.stdout  # a new bash: user trap gone


async def test_timeout_leaves_no_helper_state_and_keeps_the_users(tmp_path):
    async with Shell(tmp_path) as sh:
        await sh.run("set -u -o pipefail; trap ':' DEBUG; KEEP=1")
        r = await sh.run(f"sleep 30; touch {tmp_path}/after; echo never", timeout=0.5)
        assert (r.returncode, r.timed_out, r.stdout) == (124, True, "")
        assert not (tmp_path / "after").exists()
        r = await sh.run(
            "compgen -v __nooa; shopt -p extdebug; trap -p DEBUG; "
            "set -o | grep -E '^(nounset|pipefail|errtrace|functrace)' | tr -s ' \t' ' '; "
            'echo "$KEEP"'
        )
        assert r.stdout == (
            "shopt -u extdebug\ntrap -- ':' DEBUG\n"
            "errtrace off\nfunctrace off\nnounset on\npipefail on\n1\n"
        )


async def test_timeout_under_errexit_reports_the_lost_shell(tmp_path):
    async with Shell(tmp_path) as sh:
        await sh.run("set -e; KEEP=1")
        r = await sh.run("sleep 30; echo never", timeout=0.5)
        assert (r.returncode, r.timed_out, r.stdout) == (124, True, "")
        assert "bash ended during this command" in r.stderr  # errexit on the stopped sleep
        r = await sh.run('echo "[$KEEP]"')
        assert r.stdout == "[]\n" and r.returncode == 0


async def test_unparsable_command_runs_nothing_and_the_shell_goes_on(tmp_path):
    async with Shell(tmp_path) as sh:
        await sh.run("set -e; K=1")
        r = await sh.run("touch before; echo 'open quote")
        assert r.returncode == 2 and "unexpected" in r.stderr and not (tmp_path / "before").exists()
        r = await sh.run("cat <<EOF\nno end")
        assert r.returncode != 0 and r.stdout == ""
        r = await sh.run('echo "$K"; case $- in *e*) echo e;; esac; compgen -v __nooa; trap -p')
        assert r.stdout.startswith("1\ne\n") and "__nooa_" not in r.stdout.split("trap --")[0]


async def test_xtrace_shows_the_users_commands_not_the_wrapper(tmp_path):
    async with Shell(tmp_path) as sh:
        await sh.run("set -x")
        r = await sh.run("echo hi")
        assert r.stdout == "hi\n" and "+ echo hi" in r.stderr
        assert "__nooa" not in r.stderr and "_nemo_ec" not in r.stderr
        await sh.run("set +x")
        assert (await sh.run("echo quiet")).stderr == ""


# ---- each run has its own deadline ----


async def test_run_takes_a_timeout_per_call(tmp_path):
    """The reported call shape: ``Shell().run(command, timeout=30)``."""
    r = await Shell(tmp_path).run("echo ok", timeout=30)
    assert (r.returncode, r.timed_out, r.stdout) == (0, False, "ok\n")
    r = await Shell(tmp_path).run("cat", stdin="in", timeout=30.5)
    assert r.stdout == "in"


def test_timeout_is_a_run_parameter_not_shell_configuration():
    import inspect

    params = inspect.signature(Shell.run).parameters
    assert params["timeout"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["timeout"].default == 600.0  # omitted: 10 minutes
    assert "timeout" not in inspect.signature(Shell).parameters
    assert not hasattr(Shell(), "timeout")
    with pytest.raises(TypeError):
        Shell(timeout=30)  # type: ignore[call-arg]
    # The docs keep a raw Shell (sh) apart from its Job: a Job has no run().
    docs = (Shell.__doc__ or "") + (inspect.getmodule(Shell).__doc__ or "")
    assert "sh = self.jobs.launch(" not in docs and "sh = manager.launch(" not in docs
    assert "sh = Shell()" in docs
    assert 'r = await sh.run("pytest -q", timeout=600)' in docs  # direct, with the deadline
    run_docs = inspect.getdoc(Shell.run)
    assert "timeout" in run_docs and "deadline" in run_docs


async def test_the_documented_launched_shell_and_timed_run_variant(manager, tmp_path):
    """``job = launch(sh)``; ``job.send(...)``; ``launch(sh.run(cmd, timeout=...))``."""
    sh = Shell(tmp_path)
    job = manager.launch(sh, name="sh")
    await job.send("KEEP=1")
    r = await manager.launch(sh.run("sleep 30; echo never", timeout=0.3))
    assert (r.returncode, r.timed_out) == (124, True)
    assert (await job.send('echo "$KEEP"')).stdout == "1\n"  # one bash, state kept


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (0, ValueError),
        (-1, ValueError),
        (float("inf"), ValueError),
        (float("nan"), ValueError),
        (None, TypeError),
        (True, TypeError),
        ("30", TypeError),
    ],
)
async def test_run_rejects_a_timeout_that_is_not_a_finite_positive_number(tmp_path, value, error):
    shell = Shell(tmp_path)
    with pytest.raises(error, match="timeout"):
        await shell.run("touch ran", timeout=value)
    assert not (tmp_path / "ran").exists() and shell.summary() == "not started"


async def test_successive_runs_have_independent_deadlines(tmp_path):
    async with Shell(tmp_path) as sh:
        await sh.run("KEEP=1")
        r = await sh.run("sleep 30; echo never", timeout=0.3)
        assert (r.returncode, r.timed_out) == (124, True)
        r = await sh.run("sleep 0.6; echo slow", timeout=5)  # longer than the last deadline
        assert (r.returncode, r.timed_out, r.stdout) == (0, False, "slow\n")
        r = await sh.run('sleep 0.6; echo "$KEEP"')  # omitted: the default, not the last one
        assert (r.returncode, r.timed_out, r.stdout) == (0, False, "1\n")


async def test_concurrent_runs_queue_and_keep_their_own_deadlines(tmp_path):
    async with Shell(tmp_path) as sh:
        loop = asyncio.get_running_loop()
        started = loop.time()
        short, long_, default = await asyncio.wait_for(
            asyncio.gather(
                sh.run("sleep 30; echo never", timeout=0.3),
                sh.run("sleep 0.6; echo long", timeout=5),
                sh.run("sleep 0.6; echo default"),
            ),
            20,
        )
        assert (short.returncode, short.timed_out) == (124, True)
        assert (long_.returncode, long_.timed_out, long_.stdout) == (0, False, "long\n")
        assert (default.timed_out, default.stdout) == (False, "default\n")
        assert loop.time() - started < 10
        # A queued short deadline counts from its own start, not from when it was asked.
        first, second = await asyncio.wait_for(
            asyncio.gather(
                sh.run("sleep 0.6; echo first", timeout=5),
                sh.run("sleep 30; echo never", timeout=0.3),
            ),
            20,
        )
        assert (first.timed_out, first.stdout) == (False, "first\n")
        assert (second.returncode, second.timed_out) == (124, True)


async def test_a_launched_run_with_a_timeout_ends_its_job(manager, tmp_path):
    shell = Shell(tmp_path)
    job = manager.launch(shell.run("sleep 30; echo never", timeout=0.3), name="slow")
    r = await asyncio.wait_for(job, 10)
    assert (r.returncode, r.timed_out, r.stdout) == (124, True, "")
    assert (await shell.run("echo after")).stdout == "after\n"  # the shell goes on


# ---- closing races: aclose stops work, rejects queued work, nothing comes back ----


async def until(predicate, timeout: float = 10.0) -> None:
    """Wait, bounded, until ``predicate()`` holds (an observed state, not a fixed sleep)."""
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


def running(sh: Shell) -> bool:
    """The command's own bash wrote its pid file: the command is running now."""
    return sh._pidfile is not None and sh._pidfile.exists()


def queued(sh: Shell, n: int = 1) -> bool:
    return len(sh._lock._waiters or ()) >= n


@pytest.fixture
def gated_start(monkeypatch):
    """Hold ``BashSession.start`` at a gate (before or after bash spawns) until released."""
    from nooa.tools import BashSession

    state = {"where": "after", "armed": True}
    entered, gate, spawned = asyncio.Event(), asyncio.Event(), []
    original = BashSession.start

    async def start(self):
        if not state["armed"]:
            return await original(self)
        if state["where"] == "before":
            entered.set()
            await gate.wait()
        await original(self)
        spawned.append(self._process.pid)
        if state["where"] == "after":
            entered.set()
            await gate.wait()

    monkeypatch.setattr(BashSession, "start", start)
    state.update(entered=entered, gate=gate, spawned=spawned)
    yield state
    gate.set()  # never needed for cleanup; only frees a broken (pre-fix) run


async def test_aclose_stops_the_running_command_promptly(tmp_path):
    sh = Shell(tmp_path)
    bash = int((await sh.run("echo $$")).stdout)
    task = asyncio.create_task(sh.run(f"sleep 1800; touch {tmp_path}/after"))  # default deadline
    await until(lambda: running(sh))
    await asyncio.wait_for(sh.aclose(), 10)  # not the command's 1800 s
    with pytest.raises(RuntimeError, match="closed while this command ran"):
        await asyncio.wait_for(task, 10)
    await until_dead(bash)
    assert sh._session is None and sh.summary() == "closed"
    assert not (tmp_path / "after").exists()


async def test_aclose_rejects_queued_commands_and_starts_no_new_bash(tmp_path):
    sh = Shell(tmp_path)
    bash = int((await sh.run("echo $$")).stdout)
    active = asyncio.create_task(sh.run("sleep 1800"))
    await until(lambda: running(sh))
    stale = asyncio.create_task(sh.run(f"echo $$ > {tmp_path}/stale"))
    await until(lambda: queued(sh))
    await asyncio.wait_for(sh.aclose(), 10)
    results = await asyncio.wait_for(asyncio.gather(active, stale, return_exceptions=True), 10)
    assert all(isinstance(r, RuntimeError) for r in results), results
    assert "closed before this command started" in str(results[1])
    assert not (tmp_path / "stale").exists()  # it never ran
    assert sh._session is None  # no bash came back for it
    await until_dead(bash)
    # Explicit reuse after the close is a new bash, as before.
    again = await sh.run("echo $$")
    assert again.returncode == 0 and int(again.stdout) != bash
    await sh.aclose()


@pytest.mark.parametrize("where", ["before", "after"])
async def test_aclose_during_lazy_start_leaves_no_bash(tmp_path, gated_start, where):
    gated_start["where"] = where
    sh = Shell(tmp_path)
    task = asyncio.create_task(sh.run(f"touch {tmp_path}/ran"))
    await asyncio.wait_for(gated_start["entered"].wait(), 10)
    await asyncio.wait_for(sh.aclose(), 10)  # the gate is never released
    with pytest.raises(RuntimeError, match="closed"):
        await asyncio.wait_for(task, 10)
    await until_dead(*gated_start["spawned"])
    assert sh._session is None and not (tmp_path / "ran").exists()


async def test_owner_close_during_start_is_terminal(tmp_path, gated_start):
    from nooa_atom.agent.work_shell import ShellOwner, shell_owner

    owner = ShellOwner()
    token = shell_owner.set(owner)
    try:
        sh = Shell(tmp_path)
        task = asyncio.create_task(sh.run("true"))
        await asyncio.wait_for(gated_start["entered"].wait(), 10)
        await asyncio.wait_for(owner.aclose(), 10)  # the gate is never released
        with pytest.raises(RuntimeError, match="closed"):
            await asyncio.wait_for(task, 10)
        await until_dead(*gated_start["spawned"])
        assert len(owner) == 0 and sh._session is None
        gated_start["armed"] = False
        spawned = len(gated_start["spawned"])
        with pytest.raises(RuntimeError, match="closed"):  # a closed owner adopts nothing more
            await asyncio.wait_for(Shell(tmp_path).run("true"), 10)
        assert len(gated_start["spawned"]) == spawned and len(owner) == 0
    finally:
        shell_owner.reset(token)
    r = await sh.run("echo again")  # outside the closed owner the raw Shell is reusable
    assert r.stdout == "again\n"
    await sh.aclose()


@pytest.mark.parametrize("how", ["cancel", "timeout"])
async def test_aclose_during_interrupt_restart_stays_closed(tmp_path, gated_start, how):
    gated_start["armed"] = False
    sh = Shell(tmp_path)
    bash = int((await sh.run("echo $$")).stdout)
    gated_start["armed"] = True  # only the restart's start is held
    timeout = 0.5 if how == "timeout" else 1800
    task = asyncio.create_task(sh.run("while :; do :; done", timeout=timeout))  # a builtin
    await until(lambda: running(sh))
    if how == "cancel":
        task.cancel()
    await asyncio.wait_for(gated_start["entered"].wait(), 10)  # restarting, as it will not stop
    await asyncio.wait_for(sh.aclose(), 10)
    expected = asyncio.CancelledError if how == "cancel" else RuntimeError
    with pytest.raises(expected):
        await asyncio.wait_for(task, 10)
    await until_dead(bash, *gated_start["spawned"])
    assert sh._session is None


async def test_queued_cancel_is_isolated_and_others_run_in_order(tmp_path):
    async with Shell(tmp_path) as sh:
        await sh.run("X=1")
        gate = tmp_path / "gate"
        first = asyncio.create_task(sh.run(f"while [ ! -e {gate} ]; do sleep 0.01; done; echo A"))
        await until(lambda: running(sh))
        dropped = asyncio.create_task(sh.run(f"touch {tmp_path}/dropped"))
        last = asyncio.create_task(sh.run("echo C $X"))
        await until(lambda: queued(sh, 2))
        dropped.cancel()
        gate.touch()
        a, b, c = await asyncio.wait_for(
            asyncio.gather(first, dropped, last, return_exceptions=True), 10
        )
        assert a.stdout == "A\n" and isinstance(b, asyncio.CancelledError) and c.stdout == "C 1\n"
        assert not (tmp_path / "dropped").exists()


async def test_separate_shells_run_concurrently(tmp_path):
    a, b = Shell(tmp_path), Shell(tmp_path)
    try:
        gate = tmp_path / "gate"
        waiting = asyncio.create_task(a.run(f"while [ ! -e {gate} ]; do sleep 0.01; done; echo a"))
        await until(lambda: running(a))
        assert (await asyncio.wait_for(b.run(f"touch {gate}; echo b"), 10)).stdout == "b\n"
        assert (await asyncio.wait_for(waiting, 10)).stdout == "a\n"
    finally:
        await a.aclose()
        await b.aclose()


async def test_earlier_background_job_survives_timeout_and_cancel_until_close(tmp_path):
    sh = Shell(tmp_path)
    bg = int((await sh.run("sleep 300 & echo $!")).stdout)
    try:
        assert (await sh.run("sleep 30", timeout=0.3)).timed_out
        task = asyncio.create_task(sh.run("sleep 30"))
        await until(lambda: running(sh))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        assert alive(bg)
        await sh.aclose()
        await until_dead(bg)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(bg, 9)


async def test_inherited_preserve_scope_keeps_background_past_shell_close(tmp_path):
    """Pinned inherited limit: a harness's preserve scope outranks Shell.aclose."""
    from nooa.tools.shell_lifecycle import preserve_background_services

    bg = None
    try:
        async with preserve_background_services():
            sh = Shell(tmp_path)
            bg = int((await sh.run("sleep 300 & echo $!")).stdout)
            bash = int((await sh.run("echo $$")).stdout)
            await sh.aclose()
            await until_dead(bash)  # bash itself ends
            assert alive(bg)  # its service is left to the scope's owner (the harness)
        assert alive(bg)
    finally:
        if bg is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(bg, 9)
            await until_dead(bg)


@pytest.fixture
def gated_close(monkeypatch):
    """Hold the first armed ``BashSession.close`` at a gate; record every bash started."""
    from nooa.tools import BashSession

    state = {"armed": False, "entered": asyncio.Event(), "gate": asyncio.Event(), "spawned": []}
    original_close, original_start = BashSession.close, BashSession.start

    async def close(self):
        if state["armed"]:
            state["armed"] = False
            state["entered"].set()
            await state["gate"].wait()
        return await original_close(self)

    async def start(self):
        await original_start(self)
        state["spawned"].append(self._process.pid)

    monkeypatch.setattr(BashSession, "close", close)
    monkeypatch.setattr(BashSession, "start", start)
    yield state
    state["gate"].set()  # only frees a broken run; cleanup never relies on it


async def still_pending(task: asyncio.Future, seconds: float = 0.2) -> bool:
    """Bounded negative check: ``task`` has not finished within ``seconds``."""
    done, _ = await asyncio.wait({task}, timeout=seconds)
    return not done


@pytest.mark.parametrize("how", ["cancel", "timeout"])
async def test_aclose_while_an_interrupt_closes_the_old_bash_starts_none(
    tmp_path, gated_close, how
):
    sh = Shell(tmp_path)
    bash = int((await sh.run("echo $$")).stdout)
    gated_close["spawned"].clear()
    gated_close["armed"] = True  # the interrupt's own close of the old bash is held
    timeout = 0.5 if how == "timeout" else 1800
    task = asyncio.create_task(sh.run("while :; do :; done", timeout=timeout))  # a builtin
    await until(lambda: running(sh))
    if how == "cancel":
        task.cancel()
    await asyncio.wait_for(gated_close["entered"].wait(), 10)
    await asyncio.wait_for(sh.aclose(), 10)
    await until_dead(bash)
    gated_close["gate"].set()  # the interrupt goes on after aclose has finished
    expected = asyncio.CancelledError if how == "cancel" else RuntimeError
    with pytest.raises(expected):
        await asyncio.wait_for(task, 10)
    assert gated_close["spawned"] == []  # no restart after the close
    assert sh._session is None


async def test_enter_queued_before_aclose_starts_no_bash(tmp_path):
    sh = Shell(tmp_path)
    bash = int((await sh.run("echo $$")).stdout)
    active = asyncio.create_task(sh.run("sleep 1800"))
    await until(lambda: running(sh))
    enter = asyncio.create_task(sh.__aenter__())
    await until(lambda: queued(sh))
    await asyncio.wait_for(sh.aclose(), 10)
    results = await asyncio.wait_for(asyncio.gather(active, enter, return_exceptions=True), 10)
    assert all(isinstance(r, RuntimeError) for r in results), results
    assert sh._session is None and not sh._entered
    await until_dead(bash)
    async with sh:  # a later entry is a new bash
        assert (await sh.run("echo ok")).stdout == "ok\n"


async def test_concurrent_acloses_all_wait_for_the_one_cleanup(tmp_path, gated_close):
    sh = Shell(tmp_path)
    bash = int((await sh.run("echo $$")).stdout)
    gated_close["armed"] = True
    first = asyncio.create_task(sh.aclose())
    await asyncio.wait_for(gated_close["entered"].wait(), 10)
    second = asyncio.create_task(sh.aclose())
    third = asyncio.create_task(sh.aclose())
    assert await still_pending(second) and not third.done()  # they join the cleanup
    first.cancel()  # one closer giving up does not stop the cleanup
    with pytest.raises(asyncio.CancelledError):
        await first
    assert alive(bash) and await still_pending(second)
    gated_close["gate"].set()
    await asyncio.wait_for(asyncio.gather(second, third), 10)
    assert not alive(bash)  # bash had ended when they returned


async def test_owner_close_waits_for_a_users_close_in_progress(tmp_path, gated_close):
    from nooa_atom.agent.work_shell import ShellOwner, shell_owner

    owner = ShellOwner()
    token = shell_owner.set(owner)
    try:
        sh = Shell(tmp_path)
        bash = int((await sh.run("echo $$")).stdout)
    finally:
        shell_owner.reset(token)
    gated_close["armed"] = True
    user = asyncio.create_task(sh.aclose())
    await asyncio.wait_for(gated_close["entered"].wait(), 10)
    closing = asyncio.create_task(owner.aclose())
    assert await still_pending(closing)
    gated_close["gate"].set()
    await asyncio.wait_for(asyncio.gather(user, closing), 10)
    assert not alive(bash) and len(owner) == 0


async def test_run_during_aclose_waits_and_keeps_its_scratch_files(tmp_path, gated_close):
    sh = Shell(tmp_path)
    old = int((await sh.run("echo $$")).stdout)
    await sh.run("sleep 300 & echo $! >bg.pid")  # output pumps that the close ends too
    bg = int((tmp_path / "bg.pid").read_text())
    try:
        gated_close["armed"] = True
        closing = asyncio.create_task(sh.aclose())
        await asyncio.wait_for(gated_close["entered"].wait(), 10)
        later = asyncio.create_task(sh.run("cat; echo $$", stdin="in\n"))  # stdin: a scratch file
        assert await still_pending(later)  # it waits for the cleanup, not beside it
        gated_close["gate"].set()
        await asyncio.wait_for(closing, 10)
        r = await asyncio.wait_for(later, 10)
        lines = r.stdout.split()
        assert r.returncode == 0 and lines[0] == "in" and int(lines[1]) != old
        assert alive(int(lines[1])) and not alive(old)
        assert (await sh.run("cat", stdin="again\n")).stdout == "again\n"
        await sh.aclose()
        await until_dead(int(lines[1]), bg)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(bg, 9)


async def test_a_python_script_runs_on_stdin_without_nested_quotes(tmp_path):
    # The prompt advice: sh.run("python3 -", stdin=script), and "bash -s" for bash.
    script = 'print("double", \'single\', """triple""")\nprint(\'ends with "\')\n'
    r = await Shell(tmp_path).run("python3 -", stdin=script)
    assert r.returncode == 0, r.stderr
    assert r.stdout == 'double single triple\nends with "\n'
    r = await Shell(tmp_path).run("bash -s", stdin='echo "it\'s $((1 + 2))"\n')
    assert (r.returncode, r.stdout) == (0, "it's 3\n")


async def test_an_inline_command_behind_a_job_fails_at_once_naming_it(manager, tmp_path):
    from nooa.runtime.harness_metrics import harness_metrics_session

    with harness_metrics_session() as metrics:
        sh = Shell(tmp_path)
        log = manager.launch(sh.run("sleep 30; echo never", timeout=60), name="log")
        async with asyncio.timeout(10):
            while "running" not in sh.summary():
                await asyncio.sleep(0.02)
        # Awaited directly (not in a job): no queueing, an error at once.
        with pytest.raises(RuntimeError) as caught:
            await asyncio.wait_for(sh.run("echo polled", timeout=0.5), 2)
        message = str(caught.value)
        assert "busy running `sleep 30; echo never` (job log)" in message
        assert "launched in the background" in message and "another Shell()" in message
        attrs = metrics.to_span_attributes()
        assert attrs["harness.shell_busy_rejected.count"] == 1
        assert "harness.shell_queued_behind_running.count" not in attrs
        assert log.status == "running"  # the job goes on
        await log.cancel()
        after = await asyncio.wait_for(sh.run("echo next", timeout=10), 20)
        assert after.stdout == "next\n"  # the refused command never ran; the Shell goes on
        await sh.aclose()


async def test_an_inline_command_behind_a_job_starting_bash_fails_at_once(
    manager, tmp_path, monkeypatch
):
    # The job holds the lock while bash first starts: still busy for a cell's run.
    sh = Shell(tmp_path)
    boot = Shell._boot
    booting = asyncio.Event()

    async def slow_boot(self, session):
        booting.set()
        await asyncio.sleep(0.5)
        return await boot(self, session)

    monkeypatch.setattr(Shell, "_boot", slow_boot)
    log = manager.launch(sh.run("sleep 30", timeout=60), name="log")
    await asyncio.wait_for(booting.wait(), 10)
    with pytest.raises(RuntimeError) as caught:
        await asyncio.wait_for(sh.run("echo polled", timeout=5), 2)
    assert "busy running `sleep 30` (job log)" in str(caught.value)
    await log.cancel()
    await sh.aclose()


async def test_a_job_behind_a_job_waits_and_its_timeout_counts_from_its_start(manager, tmp_path):
    from nooa.runtime.harness_metrics import harness_metrics_session

    with harness_metrics_session() as metrics:
        async with Shell(tmp_path) as sh:
            make = manager.launch(sh.run("sleep 1.5; echo make", timeout=10), name="make")
            async with asyncio.timeout(10):
                while "running" not in sh.summary():
                    await asyncio.sleep(0.02)
            test = manager.launch(sh.run("echo pytest", timeout=0.5), name="pytest")
            async with asyncio.timeout(10):
                while "1 queued behind it" not in sh.summary():
                    await asyncio.sleep(0.02)
            assert test.summary == "$ echo pytest · queued behind `sleep 1.5; echo make` (job make)"
            first, second = await asyncio.wait_for(asyncio.gather(make, test), 20)
            # Its 0.5 s deadline counts from its own start, not from the call: it ran.
            assert (first.stdout, second.stdout) == ("make\n", "pytest\n")
            assert (second.returncode, second.timed_out) == (0, False)
            assert "waited" not in second.stderr  # under the 5 s note threshold
        attrs = metrics.to_span_attributes()
        assert attrs["harness.shell_queued_behind_running.count"] == 1
        assert "harness.shell_busy_rejected.count" not in attrs


async def test_a_long_wait_behind_a_job_is_noted_on_the_result(manager, tmp_path, monkeypatch):
    import nooa_atom.agent.work_shell as work_shell

    monkeypatch.setattr(work_shell, "_QUEUE_NOTE_SECONDS", 0.2)
    async with Shell(tmp_path) as sh:
        make = manager.launch(sh.run("sleep 0.6; echo make"), name="make")
        async with asyncio.timeout(10):
            while "running" not in sh.summary():
                await asyncio.sleep(0.02)
        test = manager.launch(sh.run("echo pytest", timeout=0.3), name="pytest")
        _, second = await asyncio.wait_for(asyncio.gather(make, test), 20)
        assert (second.stdout, second.timed_out) == ("pytest\n", False)
        assert "waited" in second.stderr and "`sleep 0.6; echo make` (job make)" in second.stderr


async def test_a_long_wait_behind_a_direct_run_is_noted_on_the_result(tmp_path, monkeypatch):
    import nooa_atom.agent.work_shell as work_shell

    monkeypatch.setattr(work_shell, "_QUEUE_NOTE_SECONDS", 0.2)
    async with Shell(tmp_path) as sh:
        first, second = await asyncio.wait_for(
            asyncio.gather(sh.run("sleep 0.6; echo first"), sh.run("echo second", timeout=0.3)),
            20,
        )
        # Behind a direct run the deadline still counts from its own start: it ran.
        assert (first.stdout, second.stdout, second.timed_out) == ("first\n", "second\n", False)
        assert "waited" in second.stderr and "`sleep 0.6; echo first`" in second.stderr
        assert "(job" not in second.stderr
