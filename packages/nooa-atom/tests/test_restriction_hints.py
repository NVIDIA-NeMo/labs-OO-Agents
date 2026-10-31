# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A blocked import or call says what to use instead in an Atom cell."""

import asyncio

from atom_test_agents import PUBLIC_ATOM_SPEC, AtomModels, cell
from nooa_atom import SessionOptions, SessionStore
from nooa_atom.agent.agent import AtomAgent
from nooa_atom.session.registry import SessionRegistry

from nooa.runtime.code_validator import _module_fix_hint, _shell_hint

SHELL = "use a Shell: `sh = Shell()` then `r = await sh.run(cmd)`"


def test_each_blocked_module_or_call_names_its_alternative():
    shell = _shell_hint(AtomAgent)
    assert shell == SHELL
    assert _module_fix_hint("subprocess", shell) == SHELL
    assert _module_fix_hint("os", shell, "system") == SHELL
    assert _module_fix_hint("os", shell, "popen") == SHELL
    for module in ("urllib.request", "http.client", "socket", "telnetlib", "xmlrpc.client"):
        hint = _module_fix_hint(module, shell)
        assert "httpx.AsyncClient" in hint and hint.endswith(f"or curl: {SHELL}")
    assert "httpx.AsyncClient" in _module_fix_hint("socket", None)  # no shell: httpx only
    assert "await coro" in _module_fix_hint("asyncio", shell, "run")
    assert "await coro" in _module_fix_hint("asyncio", shell, "run_until_complete")
    assert _module_fix_hint("time", shell, "sleep") == "`await asyncio.sleep(seconds)`"
    assert "await" in _module_fix_hint("threading", shell, "Event.wait")


async def test_an_atom_cell_importing_subprocess_is_told_to_use_a_shell(tmp_path, sessions_dir):
    models = AtomModels()
    models.scripts[None] = [
        cell("import subprocess"),
        cell("import urllib.request"),
        cell("return Done(explanation='done')"),
    ]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    options = SessionOptions(
        workspace=workspace, agent_spec=PUBLIC_ATOM_SPEC, sessions_dir=sessions_dir
    )
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=models)
    try:
        root = await registry.create(options)
        assert (await asyncio.wait_for(root.prompt("hi"), 30)).explanation == "done"
    finally:
        await registry.close_all()
    calls = models.llms[None].calls
    text = "\n".join(str(message.get("content") or "") for message in calls[-1].messages)
    assert "import of 'subprocess' is blocked" in text and f"Fix: {SHELL}" in text
    assert "Fix: use `httpx.AsyncClient`" in text and f"or curl: {SHELL}" in text
