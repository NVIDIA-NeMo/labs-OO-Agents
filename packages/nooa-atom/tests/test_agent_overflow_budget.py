# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One overflow folder per agent: Shell runs, the commands of its jobs and cell output share 20 files."""

import asyncio
import json
import re
from pathlib import Path

import pytest
from atom_test_agents import AtomModels, cell
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa.interactive import Done

_BIG = "head -c 1100000 /dev/zero | tr '\\0' a; head -c 1100000 /dev/zero | tr '\\0' b >&2"

_WORK = f"""
import json
from pathlib import Path

async def many():
    paths = []
    async with Shell(max_output=1000) as sh:
        for _ in range(12):
            r = await sh.run({_BIG!r})
            paths += [r.stdout_path, r.stderr_path]
    return paths

job_paths = await self.jobs.launch(many(), name="many")
direct = []
sh = Shell()
for _ in range(2):
    r = await sh.run({_BIG!r})
    direct += [r.stdout_path, r.stderr_path]
Path(OUT).write_text(json.dumps({{"job": job_paths, "direct": direct}}))
"""


async def test_one_agent_keeps_at_most_20_overflow_files_across_jobs_runs_and_cells(
    tmp_path, sessions_dir
):
    models = AtomModels()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    out = workspace / "paths.json"
    work = f"OUT = {str(out)!r}\n" + _WORK
    models.scripts[None] = [
        cell(
            "print('h' * 30_000 + 'm' * 10_000 + 't' * 30_000)"
        ),  # the agent's first overflow file
        cell(work),  # then 28 Shell files: 24 from 12 commands in one job, 4 from direct runs
        cell("return_result(Done(explanation='ok'))"),
    ]
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=models)
    try:
        root = await registry.create(
            SessionOptions(workspace=workspace, agent_spec="nooa_atom.agent:AtomAgent")
        )
        assert await asyncio.wait_for(root.prompt("go"), 120) == Done(explanation="ok")
        seen = str(models.llms[None].calls[1].messages)
        found = re.search(r"\\n\[\.\.\. 20,001 chars omitted; full output: ([^\]]+)\]\\n", seen)
        assert found, seen[-3000:]
        cell_path = found.group(1)
        assert out.exists(), seen[-3000:]
        paths = json.loads(out.read_text())
        job, direct = paths["job"], paths["direct"]
        assert len(job) == 24 and len(direct) == 4 and all(job + direct)
        made = [cell_path, *job, *direct]  # 29 overflow files, oldest first
        folder = Path(cell_path).parent
        assert {Path(p).parent for p in made} == {folder}  # one folder for the whole agent
        kept = sorted(folder.glob("nooa-output-*"))
        # 20 files at most for the whole agent (cell 2's own captures, never cut,
        # opened none: a stream opens its file only once it is cut).
        assert len(kept) == 20
        # The oldest closed files went first, whatever made them: the cell's print,
        # then the job's first 8 command outputs.
        for path in made[:9]:
            with pytest.raises(FileNotFoundError):
                open(path, "rb")  # noqa: SIM115
        assert sorted(Path(p) for p in made[9:]) == kept
        assert Path(job[-1]).stat().st_size == 1000  # each Shell keeps its own max_output
        assert Path(direct[-1]).stat().st_size == 1_100_000
    finally:
        await registry.close_all()
    assert not any(Path(p).exists() for p in made)  # all removed with the session
