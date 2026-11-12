# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A cell print over the 50,000-character limit is cut in memory; the notice says what to do."""

import asyncio

from atom_test_agents import AtomModels, cell
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa.interactive import Done


async def test_a_long_cell_print_is_cut_with_the_file_hint_and_no_file(tmp_path, sessions_dir):
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
    finally:
        await registry.close_all()
    seen = str(models.llms[None].calls[1].messages)
    assert "<truncated-output>" in seen
    assert (
        "The 20,001 chars in the middle are not recoverable. Write large output to a file "
        "and read it with self.files.read(path)."
    ) in seen
    assert "Full output" not in seen and "nooa-shells-" not in seen  # no file
