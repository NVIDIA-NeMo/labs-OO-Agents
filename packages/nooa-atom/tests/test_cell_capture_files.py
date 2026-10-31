# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atom cells print file-backed: past the print limit the text names a session-owned file."""

import asyncio
import json
from pathlib import Path

from atom_test_agents import AtomModels, cell
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa.config import CaptureConfig, TruncationConfig
from nooa.interactive import Done

# 60,000 characters: over python_cell's 50,000-character print limit.
PRINT = "print('HEAD' + 'p' * 59_992 + 'TAIL', end='')\n"
FIND = (
    "import os\n"
    "d = self._shells.cell_output_dir()\n"
    "files = [os.path.join(d, n) for n in os.listdir(d)]\n"
    "cut = [f for f in files if open(f).read(4) == 'HEAD']\n"
    "assert len(cut) == 1, files  # the cut stdout; cell 1's short stderr kept no file\n"
    "assert len(files) <= 3, files  # at most this cell's own two besides\n"
    "self.v.files = cut + [d]\n"
)


def test_atom_turns_on_file_backed_cells_only_in_its_own_config():
    from nooa_atom.agent.agent import AtomAgent

    from nooa.unifiedllm import FakeLLMClient

    assert CaptureConfig().file_backed is False  # the core default is unchanged
    agent = AtomAgent(llm=FakeLLMClient())
    try:
        capture = agent.runtime.truncation_config.capture
        assert capture.file_backed is True
        assert (capture.max_stdout, capture.max_stderr) == (50_000, 2_000)  # limits unchanged
        assert capture.file_dir == agent._shells.cell_output_dir()
        assert capture.file_notice == "Full output: await self.files.read({quoted_path})"
        assert TruncationConfig().capture.file_backed is False
    finally:
        asyncio.run(agent.aclose())


async def test_cut_print_names_its_file_and_the_file_goes_with_the_session(tmp_path):
    models = AtomModels()
    models.scripts[None] = [
        cell("import sys\n" + PRINT + "print('short stderr', file=sys.stderr)"),
        cell(FIND + "return_result(Done(explanation='printed'))"),
    ]
    registry = SessionRegistry(SessionStore(tmp_path / "sessions"), agent_factory=models)
    try:
        root = await registry.create(
            SessionOptions(workspace=tmp_path, agent_spec="nooa_atom.agent:AtomAgent")
        )
        assert await asyncio.wait_for(root.prompt("print"), 30) == Done(explanation="printed")
        path, directory = root._agent.v.files
        full = "HEAD" + "p" * 59_992 + "TAIL"
        assert Path(path).read_text() == full  # the whole print, not the cut view
        seen = str(models.llms[None].calls[1].messages)
        notice = f"Full output: await self.files.read({json.dumps(path)})"
        assert json.dumps(notice)[1:-1] in seen or notice in seen
        assert "Output too large (60,000 chars)" in seen
    finally:
        await registry.close_all()
    assert not Path(path).exists() and not Path(directory).exists()


async def test_child_sessions_keep_their_own_files_until_they_close(tmp_path):
    models = AtomModels()
    models.scripts[None] = [
        cell(
            "r = await self.jobs.launch(Subagent('child').run('print'), name='child')\n"
            "self.v.child = r.report\n"
            "return_result(Done(explanation='child ran'))"
        )
    ]
    models.scripts["child"] = [
        cell(PRINT),
        cell(
            FIND + "from nooa_atom.session.items import TaskResult\n"
            "return_result(Done(explanation='ok', result=TaskResult(solution_description='p', "
            "evidence='e', how_to_verify='h', report=cut[0])))"
        ),
    ]
    registry = SessionRegistry(SessionStore(tmp_path / "sessions"), agent_factory=models)
    try:
        root = await registry.create(
            SessionOptions(workspace=tmp_path, agent_spec="nooa_atom.agent:AtomAgent")
        )
        assert await asyncio.wait_for(root.prompt("go"), 45) == Done(explanation="child ran")
        child_file = root._agent.v.child
        assert not child_file.startswith(root._agent._shells.cell_output_dir())
        await asyncio.sleep(0.2)
    finally:
        await registry.close_all()
    assert not Path(child_file).exists()
