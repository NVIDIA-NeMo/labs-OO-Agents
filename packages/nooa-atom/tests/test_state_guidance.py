# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt-only guidance: real session/worker boundaries and a cross-arm receipt probe.

Run as a script with --expect-guidance present/absent and --output to capture
actual root/kept-child/one-shot-child model prompts against either checkout.
"""

import asyncio
import hashlib
import inspect
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

from atom_test_agents import AtomModels, CellLLM, cell, work_result
from nooa_atom.agent.agent import AtomAgent
from nooa_atom.gym import AtomHarness
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa.interactive import Done
from nooa.storage.json_snapshot import snapshot_to_json

START = "State across turns: Python locals end when handle() returns, including Waiting."
END = "A subagent sees only its brief:"
GUIDANCE_SHA256 = "56b930014e9c7b846aa642b06af921580c5cdc68179e167fb0abf14869c69224"
PATTERN = (
    'sh = self.jobs.get("workspace_shell")\n'
    'if sh is None or sh.status != "running":\n'
    '    sh = self.jobs.launch(Shell(), name="workspace_shell")\n'
    'request = sh.send("pwd")'
)


def _guidance(text):
    if START not in text:
        return None
    return inspect.cleandoc(text[text.index(START) :].split(END, 1)[0].rstrip())


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def test_guidance_is_exact_static_text_and_contains_executable_pattern():
    block = _guidance(AtomAgent.__doc__)
    assert block is not None
    assert hashlib.sha256(block.encode()).hexdigest() == GUIDANCE_SHA256
    assert 150 <= len(block.split()) <= 250
    assert "{" not in block and "}" not in block  # no accidental prompt template expressions
    assert block.split("```python\n", 1)[1].split("\n```", 1)[0] == PATTERN
    compile(PATTERN, "guidance-example", "exec")


async def capture_prompts(workspace, expect_guidance):
    workspace.mkdir(parents=True, exist_ok=True)
    scripts = [
        "review = self.jobs.launch(Subagent(), name='review')\n"
        "assert (await review.send('review')).report == 'kept'\n"
        "assert (await Subagent().run('one-shot')).report == 'one-shot'\n"
        + work_result(report="root"),
        work_result(report="kept"),
        work_result(report="one-shot"),
    ]
    llm = CellLLM([cell(code) for code in scripts])
    harness = AtomHarness(llm, workdir=workspace, sessions_dir=workspace / "sessions")
    assert (await asyncio.wait_for(harness.solve("Prompt visibility probe"), 60)).report == "root"
    assert len(llm.calls) == 3
    prompts = []
    for label, call in zip(("root", "kept_child", "one_shot_child"), llm.calls, strict=True):
        messages = _plain(call.messages)
        # Find the actual rendered instruction text, not a configured constructor value.
        text_parts = [m.get("content", "") for m in messages if isinstance(m.get("content"), str)]
        matching = [part for part in text_parts if START in part]
        assert bool(matching) is expect_guidance, label
        block = _guidance(matching[0]) if matching else None
        if block is not None:
            assert hashlib.sha256(block.encode()).hexdigest() == GUIDANCE_SHA256
            assert sum(part.count(START) for part in text_parts) == 1, label
        prompts.append(
            {
                "session_kind": label,
                "guidance_present": bool(matching),
                "guidance_text": block,
                "guidance_sha256": hashlib.sha256(block.encode()).hexdigest() if block else None,
                "messages": messages,
            }
        )
    return prompts


async def test_root_and_both_child_kinds_receive_rendered_guidance(tmp_path):
    await capture_prompts(tmp_path, expect_guidance=True)


async def exercise_named_shell(workspace):
    workspace.mkdir(parents=True, exist_ok=True)
    models = AtomModels()
    models.scripts[None] = [
        cell(
            PATTERN + "\n"
            "assert (await request).returncode == 0\n"
            "todo = self.todo.add('keep state by stable identifiers')\n"
            "self.v.todo_id = todo.id\n"
            "self.v.shell_name = 'workspace_shell'\n"
            "result = await sh.send('mkdir -p work; cd work; export GUIDANCE_MARKER=kept; echo $$')\n"
            "self.v.shell_pid = int(result.stdout)\n"
            'request = sh.send(\'sleep 0.05; printf "%s\\n" "$$" "$PWD" "$GUIDANCE_MARKER"\')\n'
            "self.v.request_name = request.name\n"
            "return_result(Waiting(explanation='command running', on=[request.name]))"
        ),
        cell(
            "try:\n    sh\nexcept NameError:\n    pass\n"
            "else:\n    raise AssertionError('locals unexpectedly survived handle')\n"
            "sh = self.jobs.get(self.v.shell_name)\n"
            "assert sh is not None and sh.status == 'running'\n"
            "request = self.jobs.get(self.v.request_name)\n"
            "assert request.status == 'done'\n"
            "pid, cwd, marker = request.result.stdout.strip().splitlines()\n"
            "assert int(pid) == self.v.shell_pid and marker == 'kept'\n"
            "self.v.cwd = cwd\n"
            "assert (await sh.send('pwd')).stdout.strip() == cwd\n"
            "assert int((await sh.send('echo $$')).stdout) == self.v.shell_pid\n"
            "todo = self.todo.get(self.v.todo_id)\n"
            "todo.comments.append('recovered by id after Waiting')\n"
            "self.todo.complete(todo)\nreturn_result(Done(explanation='reused'))"
        ),
        cell(
            PATTERN + "\n"
            "assert (await request).stdout.strip() != self.v.cwd\n"
            "assert (await sh.send('printf %s \"$GUIDANCE_MARKER\"')).stdout == ''\n"
            "self.v.recreated_pid = int((await sh.send('echo $$')).stdout)\n"
            "assert self.v.recreated_pid != self.v.shell_pid\n"
            "return_result(Done(explanation='recreated'))"
        ),
    ]
    registry = SessionRegistry(SessionStore(workspace / "sessions"), agent_factory=models)
    try:
        options = SessionOptions(workspace=workspace, agent_spec="nooa_atom.agent:AtomAgent")
        root = await registry.create(options)
        assert await asyncio.wait_for(root.prompt("run"), 60) == Done(explanation="reused")
        agent = root._agent
        assert len(models.llms[None].calls) == 2  # an actual Waiting -> new handle boundary
        assert agent.todo.get(agent.v.todo_id).status == "done"
        assert agent.v.cwd == str(workspace / "work")
        facts = dict(agent.vars)
        assert set(facts) == {"todo_id", "shell_name", "shell_pid", "request_name", "cwd"}
        assert all(isinstance(value, (str, int)) for value in facts.values())
        json.dumps(facts)
        snapshot = snapshot_to_json(agent)
        json.dumps(snapshot)  # no Shell/Job/module/function serialization needed
        assert "work_manager" not in str(snapshot) and "work_shell" not in str(snapshot)
        first_pid = agent.v.shell_pid
        await agent.jobs.get("workspace_shell").cancel()
        assert await asyncio.wait_for(root.prompt("recreate"), 60) == Done(explanation="recreated")
        return {
            "facts_after_waiting": facts,
            "first_pid": first_pid,
            "new_pid": agent.v.recreated_pid,
            "model_calls": len(models.llms[None].calls),
        }
    finally:
        await registry.close_all()


async def test_guidance_pattern_relooks_up_shell_and_serializable_facts_across_waiting(tmp_path):
    await exercise_named_shell(tmp_path)


if __name__ == "__main__":
    import argparse
    import tempfile

    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-guidance", choices=("present", "absent"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    async def receipt():
        with tempfile.TemporaryDirectory(prefix="atom-state-guidance-") as temp:
            workspace = Path(temp)
            os.environ["NEMO_OO_USER_DIR"] = str(workspace / "user")
            prompts = await capture_prompts(
                workspace / "prompts", args.expect_guidance == "present"
            )
            shell = await exercise_named_shell(workspace / "shell")
            source = Path(inspect.getfile(AtomAgent)).resolve()
            revision = subprocess.check_output(
                ["git", "-C", str(source.parent), "rev-parse", "HEAD"], text=True
            ).strip()
            args.output.write_text(
                json.dumps(
                    {
                        "source": str(source),
                        "revision": revision,
                        "guidance_expected": args.expect_guidance,
                        "prompts": prompts,
                        "named_shell": shell,
                    },
                    indent=2,
                )
            )

    asyncio.run(receipt())
