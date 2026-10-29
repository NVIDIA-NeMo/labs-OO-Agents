# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real Shell output seen by scripted Atom root/children before process exit."""

import asyncio
import shlex

from atom_test_agents import AtomModels, cell
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa.interactive import Done


def preview_code(name, workspace):
    gate = str(workspace / (name + ".gate"))
    command = f"printf {name}; while [ ! -f {shlex.quote(gate)} ]; do sleep .01; done; printf END"
    return (
        "from pathlib import Path\n"
        f"gate=Path({gate!r})\n"
        f"work=self.jobs.launch(Shell().run({command!r}), name='preview')\n"
        "seen=[]\n"
        "async for part in work:\n"
        f" assert part.stream == 'stdout' and part.text == {name!r}\n"
        " assert work.status == 'running' and not gate.exists()\n"
        " seen.append(part.text)\n"
        " gate.touch()\n"
        "result=await work\n"
        f"assert result.returncode == 0 and result.stdout == {name + 'END'!r}\n"
        f"assert seen == [{name!r}]\n"
        "self.v.preview_seen=seen\n"
    )


def done(report):
    return (
        "return_result(Done(explanation='observed', result=TaskResult("
        f"solution_description='live output', evidence='gate', how_to_verify='test', report={report!r})))"
    )


async def test_root_child_grandchild_observe_same_fixed_policy_in_one_handle(tmp_path):
    models = AtomModels()
    models.scripts[None] = [
        cell(
            preview_code("root", tmp_path)
            + "child=await self.jobs.launch(Subagent('child').run('observe'), name='child')\n"
            + "assert child.report == 'child+grand'\n"
            + done("root+child+grand")
        )
    ]
    models.scripts["child"] = [
        cell(
            preview_code("child", tmp_path)
            + "grand=await self.jobs.launch(Subagent('grand').run('observe'), name='grand')\n"
            + "assert grand.report == 'grand'\n"
            + done("child+grand")
        )
    ]
    models.scripts["grand"] = [cell(preview_code("grand", tmp_path) + done("grand"))]
    registry = SessionRegistry(SessionStore(tmp_path / "sessions"), agent_factory=models)
    try:
        root = await registry.create(
            SessionOptions(
                workspace=tmp_path,
                agent_spec="nooa_atom.agent:AtomAgent",
            )
        )
        outcome = await asyncio.wait_for(root.prompt("observe the live output"), 45)
        assert outcome.result.report == "root+child+grand"
        assert root._agent.v.preview_seen == ["root"]
        assert all((tmp_path / (name + ".gate")).exists() for name in ("root", "child", "grand"))
        await asyncio.sleep(0.1)
        for llm in models.llms.values():
            assert len(llm.calls) == 1  # previews were observed, not extra generation turns
            prompt = str(llm.calls[0].messages)
            assert "1s OR 8192 UTF-8 bytes" in prompt
            assert "await job`` is final-only" in prompt
    finally:
        await registry.close_all()


async def test_waiting_receives_live_preview_then_one_final_without_releasing_gate_early(tmp_path):
    models = AtomModels()
    gate = str(tmp_path / "gate")
    command = f"printf READY; while [ ! -f {shlex.quote(gate)} ]; do sleep .01; done; printf END"
    models.scripts[None] = [
        cell(
            f"self.jobs.launch(Shell().run({command!r}), name='live')\n"
            "return_result(Waiting(explanation='running', on=['live']))"
        ),
        cell(
            "[event]=notification['jobs']\n"
            "assert event.kind == 'item' and event.value.stream == 'stdout'\n"
            "assert event.value.text == 'READY' and event.value.sequence == 1\n"
            "assert self.jobs.get('live').status == 'running'\n"
            f"from pathlib import Path\nassert not Path({gate!r}).exists()\nPath({gate!r}).touch()\n"
            "return_result(Waiting(explanation='released', on=['live']))"
        ),
        cell(
            "[event]=notification['jobs']\n"
            "assert event.kind == 'done' and event.value.stdout == 'READYEND'\n"
            "return_result(Done(explanation='observed before exit'))"
        ),
    ]
    registry = SessionRegistry(SessionStore(tmp_path / "sessions"), agent_factory=models)
    try:
        root = await registry.create(
            SessionOptions(
                workspace=tmp_path,
                agent_spec="nooa_atom.agent:AtomAgent",
            )
        )
        assert await asyncio.wait_for(root.prompt("observe"), 30) == Done(
            explanation="observed before exit"
        )
        await asyncio.sleep(0.1)
        assert len(models.llms[None].calls) == 3
    finally:
        await registry.close_all()
