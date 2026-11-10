# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A cell print over the 50,000-character limit is cut and spilled to the session's folder."""

import asyncio
import re
from pathlib import Path

from atom_test_agents import AtomModels, cell
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa.interactive import Done


async def test_a_long_cell_print_is_cut_and_spilled_until_the_session_closes(
    tmp_path, sessions_dir
):
    models = AtomModels()
    models.scripts[None] = [
        cell("print('h' * 30_000 + 'm' * 10_000 + 't' * 30_000)"),
        cell("return_result(Done(explanation='ok'))"),
    ]
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=models)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    try:
        root = await registry.create(
            SessionOptions(workspace=workspace, agent_spec="nooa_atom.agent:AtomAgent")
        )
        assert await asyncio.wait_for(root.prompt("go"), 60) == Done(explanation="ok")
        seen = str(models.llms[None].calls[1].messages)
        # 70,001 chars (with the newline) into 25,000 + 25,000: render_output's one cut line.
        found = re.search(r"\\n\[\.\.\. 20,001 chars omitted; full output: ([^\]]+)\]\\n", seen)
        assert found, seen[-3000:]
        path = Path(found.group(1))
        assert "nooa-shells-" in str(path)  # the session's spill folder
        assert path.read_text() == "h" * 30_000 + "m" * 10_000 + "t" * 30_000 + "\n"
        assert "<truncated-output>" not in seen and "not recoverable" not in seen
    finally:
        await registry.close_all()
    assert not path.exists()  # removed with the session
