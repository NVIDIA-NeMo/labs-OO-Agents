# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``Shell`` work under the ``WorkManager``: real bash, one-shot and persistent."""

import asyncio
import contextlib
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
    r = await manager.launch(Shell(tmp_path).run("echo out; echo err >&2; (exit 3)"))
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
    assert one.stdout == "/\n" and one.cwd == "/"  # the shell's real directory after it
    await manager.aclose()
    assert shell.summary() == "closed"


# ---- regressions from the independent review of 83ae65c77 ----


async def test_stdin_feed_shares_the_command_deadline():
    started = asyncio.get_running_loop().time()
    r = await asyncio.wait_for(Shell(timeout=1).run("sleep 30", stdin="x" * (8 << 20)), 10)
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
    await asyncio.sleep(0.1)
    assert not alive(pid)


async def test_output_limit_is_enforced_while_writing(manager, tmp_path):
    r = await manager.launch(Shell(max_output=1000).run("head -c 100000 /dev/zero | tr '\\0' a"))
    assert Path(r.stdout_path).stat().st_size == 1000
    assert "over the 1,000-byte output limit" in r.stdout and "first 1,000 of 100,000" in r.stdout
    assert r.stdout.endswith("a" * 100)  # the tail is still shown
    sh = manager.launch(Shell(tmp_path, max_output=1000), name="sh")
    r = await sh.send("head -c 50000 /dev/zero | tr '\\0' b; echo done >&2")
    assert Path(r.stdout_path).stat().st_size == 1000 and r.stderr == "done\n"
    r = await manager.launch(Shell(timeout=1, max_output=10_000).run("yes"))
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
    shell = Shell(tmp_path, timeout=2)
    r = await asyncio.wait_for(shell.run("sleep 30 & echo $! >bg.pid; echo hi"), 10)
    assert (r.returncode, r.timed_out, r.stdout) == (0, False, "hi\n")
    assert loop.time() - started < 1.5
    pid = int((tmp_path / "bg.pid").read_text())
    try:
        assert alive(pid)  # run() did not kill it: it lives as long as the shell
        await shell.aclose()  # deterministic: bash and its background processes end
        await asyncio.sleep(0.1)
        assert not alive(pid)
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
    await asyncio.sleep(0.1)
    assert not alive(pid) and shell.summary() == "closed"


async def test_timeout_stops_the_whole_command_and_keeps_state(tmp_path):
    async with Shell(tmp_path, timeout=0.5) as sh:
        await sh.run("KEEP=1")
        r = await sh.run("sleep 30; touch after; echo never")
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


async def test_dropped_shell_ends_its_bash(tmp_path):
    pid = int((await Shell(tmp_path).run("echo $$")).stdout)  # a temporary
    await asyncio.sleep(0.1)
    assert not alive(pid)


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
    await asyncio.sleep(0.1)
    assert not alive(bg) and not alive(bash) and not alive(other)
    assert kept.summary() == "closed"
    assert (await kept.run("echo again")).stdout == "again\n"  # usable again: a new bash
    await kept.aclose()


# ---- the wrapper under the user's shell options and traps ----


async def test_set_e_persists_and_ordinary_commands_still_succeed(tmp_path):
    async with Shell(tmp_path, timeout=10) as sh:
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
    async with Shell(tmp_path, timeout=10) as sh:
        assert (await sh.run("false")).returncode == 1
        assert (await sh.run("(exit 7)")).returncode == 7
        assert (await sh.run("sh -c 'kill -TERM $$'")).returncode == 143
        r = await sh.run("exit 7")
        assert r.returncode == 7 and "bash ended during this command (exit 7)" in r.stderr
        r = await sh.run("exit 0")
        assert r.returncode == 0 and "bash ended during this command (exit 0)" in r.stderr
        assert (await sh.run("echo back")).stdout == "back\n"


async def test_nounset_and_pipefail(tmp_path):
    async with Shell(tmp_path, timeout=10) as sh:
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
    async with Shell(tmp_path, timeout=10) as sh:
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
    async with Shell(tmp_path, timeout=0.5) as sh:
        await sh.run("trap ':' USR1; KEEP=1")
        r = await sh.run(f"sleep 30; touch {tmp_path}/after")
        assert r.timed_out and "the shell restarted" in r.stderr  # no silent half-stop
        assert not (tmp_path / "after").exists()
        r = await sh.run('echo "[$KEEP]"; trap -p USR1')
        assert r.stdout.startswith("[]\n") and "':'" not in r.stdout  # a new bash: user trap gone


async def test_timeout_leaves_no_helper_state_and_keeps_the_users(tmp_path):
    async with Shell(tmp_path, timeout=0.5) as sh:
        await sh.run("set -u -o pipefail; trap ':' DEBUG; KEEP=1")
        r = await sh.run(f"sleep 30; touch {tmp_path}/after; echo never")
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
    async with Shell(tmp_path, timeout=0.5) as sh:
        await sh.run("set -e; KEEP=1")
        r = await sh.run("sleep 30; echo never")
        assert (r.returncode, r.timed_out, r.stdout) == (124, True, "")
        assert "bash ended during this command" in r.stderr  # errexit on the stopped sleep
        r = await sh.run('echo "[$KEEP]"')
        assert r.stdout == "[]\n" and r.returncode == 0


async def test_unparsable_command_runs_nothing_and_the_shell_goes_on(tmp_path):
    async with Shell(tmp_path, timeout=10) as sh:
        await sh.run("set -e; K=1")
        r = await sh.run("touch before; echo 'open quote")
        assert r.returncode == 2 and "unexpected" in r.stderr and not (tmp_path / "before").exists()
        r = await sh.run("cat <<EOF\nno end")
        assert r.returncode != 0 and r.stdout == ""
        r = await sh.run('echo "$K"; case $- in *e*) echo e;; esac; compgen -v __nooa; trap -p')
        assert r.stdout.startswith("1\ne\n") and "__nooa_" not in r.stdout.split("trap --")[0]


async def test_xtrace_shows_the_users_commands_not_the_wrapper(tmp_path):
    async with Shell(tmp_path, timeout=10) as sh:
        await sh.run("set -x")
        r = await sh.run("echo hi")
        assert r.stdout == "hi\n" and "+ echo hi" in r.stderr
        assert "__nooa" not in r.stderr and "_nemo_ec" not in r.stderr
        await sh.run("set +x")
        assert (await sh.run("echo quiet")).stderr == ""
