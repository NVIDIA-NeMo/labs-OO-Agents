# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real Shells are completion-only in scripted Atom root/children; no provider."""

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
    return (
        f"work=self.jobs.launch(Shell().run({gated_command(name, workspace)!r}), name='shell')\n"
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
        "return_result(Done(explanation='completed', result=TaskResult("
        f"solution_description='completion only', evidence='gate', how_to_verify='test', report={report!r})))"
    )


@pytest.mark.parametrize("observation", ["await", "iterator"])
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
            assert "Shell output is completion-only" in prompt
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
            "return_result(Waiting(explanation='running', on=['shell']))"
        ),
        cell(
            "[event]=notification['jobs']\n"
            "assert event.kind == 'done' and event.name == 'shell'\n"
            "result=event.value\n"
            + capture_assertions("root")
            + "assert self.jobs.get('shell').status == 'done'\n"
            + "assert self.jobs.get('shell').progress_dropped == 0\n"
            + "return_result(Done(explanation='observed only completion'))"
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
            "return_result(Done(explanation='ran'))"
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
