# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real Shells in scripted Atom root/children: quiet by default, Throttle opt-in; no provider."""

import asyncio
import shlex
import sys

import pytest
from atom_test_agents import AtomModels, cell
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa.interactive import Done


def gated_command(name, workspace):
    gate = workspace / (name + ".gate")
    ready = workspace / (name + ".ready")
    program = (
        "import os,time,pathlib; "
        f"os.write(1, {name.encode()!r} + b'O'*20000); os.write(2, b'E'*20000); "
        f"pathlib.Path({str(ready)!r}).touch(); gate=pathlib.Path({str(gate)!r}); "
        "\nwhile not gate.exists(): time.sleep(.01)\n"
        "os.write(1,b'END')"
    )
    return shlex.join([sys.executable, "-c", program])


def capture_assertions(name):
    return (
        "from pathlib import Path\n"
        "assert result.returncode == 0 and not result.timed_out\n"
        f"assert Path(result.stdout_path).read_text() == {name!r} + 'O'*20000 + 'END'\n"
        "assert Path(result.stderr_path).read_text() == 'E'*20000\n"
        "assert 'bytes omitted' in result.stdout and 'bytes omitted' in result.stderr\n"
    )


def completion_code(name, workspace, observation):
    command = gated_command(name, workspace)
    if observation == "throttled":
        return (
            f"work=self.jobs.launch(Shell().run({command!r}), name='shell', "
            "progress=Throttle(seconds=0.2))\n"
            "parts=[part async for part in work]\n"
            "assert parts and {type(p).__name__ for p in parts} == {'ShellProgress'}\n"
            "assert all(len(p.stdout) <= 8192 and len(p.stderr) <= 8192 for p in parts)\n"
            f"assert ''.join(p.stdout for p in parts).startswith({name!r} + 'O'*100)\n"
            "result=await work\n" + capture_assertions(name) + "self.v.completion_seen=True\n"
        )
    return (
        f"work=self.jobs.launch(Shell().run({command!r}), name='shell')\n"
        + ("assert [part async for part in work] == []\n" if observation == "iterator" else "")
        + "result=await work\n"
        + capture_assertions(name)
        + "assert work.progress_dropped == 0\n"
        + "self.v.completion_seen=True\n"
    )


async def hold_then_release(workspace, name, models, pending):
    ready, gate = workspace / (name + ".ready"), workspace / (name + ".gate")
    async with asyncio.timeout(20):
        while not ready.exists():
            if pending.done():
                await pending  # surface script errors instead of timing out
                pytest.fail("session ended before the gated command started")
            await asyncio.sleep(0.01)
    # Both streams have exceeded 8192 bytes. Observe beyond the former 1s timer
    # before the test (not any generated turn) releases the command.
    await asyncio.sleep(1.1)
    assert not pending.done() and not gate.exists()
    assert all(len(llm.calls) == 1 for llm in models.llms.values())
    gate.touch()


def done(report):
    return (
        "return Done(explanation='completed', result=TaskResult("
        f"solution_description='completion only', evidence='gate', how_to_verify='test', report={report!r}))"
    )


@pytest.mark.parametrize("observation", ["await", "iterator", "throttled"])
async def test_root_child_grandchild_complete_without_output_items_or_extra_turns(
    tmp_path, observation
):
    models = AtomModels()
    models.scripts[None] = [
        cell(
            completion_code("root", tmp_path, observation)
            + "child=await self.jobs.launch(Subagent('child').run('observe'), name='child')\n"
            + "assert child.report == 'child+grand'\n"
            + done("root+child+grand")
        )
    ]
    models.scripts["child"] = [
        cell(
            completion_code("child", tmp_path, observation)
            + "grand=await self.jobs.launch(Subagent('grand').run('observe'), name='grand')\n"
            + "assert grand.report == 'grand'\n"
            + done("child+grand")
        )
    ]
    models.scripts["grand"] = [
        cell(completion_code("grand", tmp_path, observation) + done("grand"))
    ]
    registry = SessionRegistry(SessionStore(tmp_path / "sessions"), agent_factory=models)
    pending = None
    try:
        root = await registry.create(
            SessionOptions(workspace=tmp_path, agent_spec="nooa_atom.agent:AtomAgent")
        )
        pending = asyncio.create_task(root.prompt("observe completion-only output"))
        for name in ("root", "child", "grand"):
            await hold_then_release(tmp_path, name, models, pending)
        outcome = await asyncio.wait_for(pending, 45)
        assert outcome.result.report == "root+child+grand"
        assert root._agent.v.completion_seen is True
        await asyncio.sleep(0.1)
        for llm in models.llms.values():
            assert len(llm.calls) == 1
            prompt = str(llm.calls[0].messages)
            assert "Completion-only unless launched with ``progress=Throttle(...)``" in prompt
            assert "class Throttle:" in prompt and "set_progress(None)" in prompt
            assert "1s OR 8192 UTF-8 bytes" not in prompt
    finally:
        for name in ("root", "child", "grand"):
            (tmp_path / (name + ".gate")).touch()
        if pending is not None and not pending.done():
            pending.cancel()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)
        await registry.close_all()


async def test_waiting_receives_one_final_without_pre_exit_output_turns(tmp_path):
    models = AtomModels()
    models.scripts[None] = [
        cell(
            f"self.jobs.launch(Shell().run({gated_command('root', tmp_path)!r}), name='shell')\n"
            "return Waiting(explanation='running', on=['shell'])"
        ),
        cell(
            "[event]=notification['jobs']\n"
            "assert event.kind == 'done' and event.name == 'shell'\n"
            "result=event.value\n"
            + capture_assertions("root")
            + "assert self.jobs.get('shell').status == 'done'\n"
            + "assert self.jobs.get('shell').progress_dropped == 0\n"
            + "return Done(explanation='observed only completion')"
        ),
    ]
    registry = SessionRegistry(SessionStore(tmp_path / "sessions"), agent_factory=models)
    pending = None
    try:
        root = await registry.create(
            SessionOptions(workspace=tmp_path, agent_spec="nooa_atom.agent:AtomAgent")
        )
        pending = asyncio.create_task(root.prompt("observe"))
        await hold_then_release(tmp_path, "root", models, pending)
        assert await asyncio.wait_for(pending, 30) == Done(explanation="observed only completion")
        await asyncio.sleep(1.1)
        assert len(models.llms[None].calls) == 2
    finally:
        (tmp_path / "root.gate").touch()
        if pending is not None and not pending.done():
            pending.cancel()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)
        await registry.close_all()


async def test_direct_one_liner_output_file_lives_until_the_session_closes(tmp_path):
    import gc
    from pathlib import Path

    models = AtomModels()
    models.scripts[None] = [
        cell(
            "r = await Shell().run('seq 1 100000')\n"  # the temporary Shell is dropped
            "self.v.out = r.stdout_path\n"
            "assert 'bytes omitted; full output:' in r.stdout\n"
            "del r\n"
            "return Done(explanation='ran')"
        )
    ]
    registry = SessionRegistry(SessionStore(tmp_path / "sessions"), agent_factory=models)
    try:
        root = await registry.create(
            SessionOptions(workspace=tmp_path, agent_spec="nooa_atom.agent:AtomAgent")
        )
        assert await asyncio.wait_for(root.prompt("run"), 30) == Done(explanation="ran")
        gc.collect()
        path = Path(root._agent.v.out)
        assert path.read_text().count("\n") == 100000  # outlives the dropped Shell
    finally:
        await registry.close_all()
    assert not path.exists()  # removed with the session


def burst_command(workspace, size=30_000):
    """Jazz's case: a ~30 KB burst well before completion, then a short tail of work."""
    program = (
        "import os,time,pathlib; "
        f"os.write(1, b'J'*{size}); time.sleep(.3); "
        f"pathlib.Path({str(workspace / 'burst.done')!r}).touch(); os.write(1, b'END')"
    )
    return shlex.join([sys.executable, "-c", program])


WAIT_LOOP = (
    "events = notification.get('jobs', [])\n"
    "seen += [len(e.value.stdout.encode()) + len(e.value.stderr.encode()) "
    "for e in events if e.kind == 'item']\n"
    "final = [e for e in events if e.kind == 'done']\n"
    "if final:\n"
    "    assert final[0].value.stdout.endswith('END')\n"
    "    self.v.seen = seen\n"
    "    return Done(explanation='burst done')\n"
    "else:\n"
    "    return Waiting(explanation='still running', on=['burst'])\n"
)


async def test_noisy_burst_with_waiting_wakes_for_progress_at_most_once(tmp_path):
    models = AtomModels()
    models.scripts[None] = [
        cell(
            "seen = []\n"
            f"self.jobs.launch(Shell().run({burst_command(tmp_path)!r}), name='burst', "
            "progress=Throttle())\n"
            "return Waiting(explanation='running', on=['burst'])"
        ),
        *[cell(WAIT_LOOP) for _ in range(6)],
    ]
    registry = SessionRegistry(SessionStore(tmp_path / "sessions"), agent_factory=models)
    try:
        root = await registry.create(
            SessionOptions(workspace=tmp_path, agent_spec="nooa_atom.agent:AtomAgent")
        )
        outcome = await asyncio.wait_for(root.prompt("watch the burst"), 30)
        assert outcome == Done(explanation="burst done")
        seen = root._agent.v.seen
        # Before: several raw 8 KB previews (~24 KB) each woke a turn. Now the burst
        # merges into one bounded preview and the final result supersedes the rest.
        assert len(seen) <= 1 and all(size <= 8192 + 64 for size in seen)
        assert len(models.llms[None].calls) <= 3
    finally:
        await registry.close_all()


async def test_status_block_shows_policies_and_counts_never_output(tmp_path):
    from nooa_atom.agent.agent import AtomAgent
    from nooa_atom.agent.work_progress import Throttle
    from nooa_atom.agent.work_shell import Shell

    from nooa.unifiedllm import FakeLLMClient

    agent = AtomAgent(cwd=tmp_path, llm=FakeLLMClient())
    try:
        build = agent.jobs.launch(
            Shell(tmp_path).run("printf %s $((6*7))xyz; sleep 30"),
            name="build",
            progress=Throttle(items=20, seconds=30),
        )
        quiet = agent.jobs.launch(Shell(tmp_path).run("sleep 30"), name="quiet")
        sh = agent.jobs.launch(Shell(tmp_path), name="sh", progress=Throttle(seconds=5))
        async with asyncio.timeout(10):
            while build._progress_buffered() == 0 or not sh._entered:
                await asyncio.sleep(0.02)
        lines = agent._status().splitlines()
        [build_line] = [line for line in lines if line.lstrip(" •-").startswith("build ·")]
        [quiet_line] = [line for line in lines if line.lstrip(" •-").startswith("quiet ·")]
        [idle_line] = [line for line in lines if "sh · " in line and "new requests" in line]
        assert "progress Throttle(items=20, seconds=30)" in build_line
        assert "5 B buffered" in build_line
        assert "progress" not in quiet_line
        assert "new requests: progress Throttle(seconds=5)" in idle_line
        assert "42xyz" not in "\n".join(lines)  # the output itself is never shown
        request = sh.send("sleep 30")
        assert request.set_progress(None) is True
        async with asyncio.timeout(10):
            while not agent.jobs.get("sh#1").summary:
                await asyncio.sleep(0.02)
        text = agent._status()
        [request_line] = [line for line in text.splitlines() if "sh#1 ·" in line]
        assert "new sh requests: progress Throttle(seconds=5)" in request_line
        assert request_line.count("progress") == 1  # its own policy is None: nothing more
        assert build.set_progress(None) is True
        assert (
            "progress Throttle"
            not in [line for line in agent._status().splitlines() if "build ·" in line][0]
        )
        await quiet.cancel()
    finally:
        await agent.aclose()
