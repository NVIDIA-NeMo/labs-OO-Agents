# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``self.jobs.after/every`` in a real Atom turn: they wake a ``Waiting`` turn."""

import asyncio

from atom_test_agents import PUBLIC_ATOM_SPEC, AtomModels, cell
from nooa_atom import SessionOptions, SessionStore
from nooa_atom.session.registry import SessionRegistry

TIMEOUT = 30
DONE = "return_result(Done(explanation='done'))"


async def _run(tmp_path, sessions_dir, cells):
    """Run one interactive turn to its Done; return each model call's new text."""
    models = AtomModels()
    models.scripts[None] = cells
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    options = SessionOptions(
        workspace=workspace, agent_spec=PUBLIC_ATOM_SPEC, sessions_dir=sessions_dir
    )
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=models)
    try:
        root = await registry.create(options)
        outcome = await asyncio.wait_for(root.prompt("hi"), TIMEOUT)
        assert outcome.explanation == "done"
    finally:
        await registry.close_all()
    calls = models.llms[None].calls
    return [
        "\n".join(
            str(message.get("content") or "")
            for message in call.messages[len(calls[index - 1].messages) if index else 0 :]
        )
        for index, call in enumerate(calls)
    ]


async def test_after_wakes_a_waiting_turn(tmp_path, sessions_dir):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell(
                "import time\n"
                "self.v.start = time.monotonic()\n"
                "self.jobs.after(0.3, name='reading')\n"
                "return_result(Waiting(explanation='w', on=['reading']))"
            ),
            cell(
                "import time\n"
                "[event] = notification['jobs']\n"
                "assert (event.name, event.kind) == ('reading', 'done'), event\n"
                "assert time.monotonic() - self.v.start >= 0.25\n"
                "assert not hasattr(self, 'producers')\n" + DONE
            ),
        ],
    )
    assert len(calls) == 2


async def test_every_in_a_turn(tmp_path, sessions_dir):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell(
                "self.jobs.every(0.1, name='poll')\n"
                "return_result(Waiting(explanation='w', on=['poll']))"
            ),
            cell(
                "assert notification['jobs'][0].kind == 'item'\n"
                "await self.jobs.get('poll').cancel()\n" + DONE
            ),
        ],
    )
    assert len(calls) == 2
    for text in calls[1:]:
        assert "AssertionError" not in text, text


def test_the_tools_doc_shows_after_every_and_the_tail_idiom():
    from nooa_atom.agent.agent import _WORK_EXAMPLES
    from nooa_atom.agent.work_manager import WorkManager
    from nooa_atom.agent.work_shell import Shell

    from nooa.agentdoc import doc

    text = doc(WorkManager, inline_depth=0)
    for wanted in ("def after(", "def every("):
        assert wanted in text, wanted
    assert "progress=Throttle(" in _WORK_EXAMPLES
    assert "def tail(" not in text and "self.jobs.tail(" not in _WORK_EXAMPLES
    # after() says the delay counts from now; no elapsed-time advice is repeated.
    assert "subtract" not in text.lower() and "subtract" not in _WORK_EXAMPLES
    for call in ("self.jobs.after(", "self.jobs.every("):
        assert sum(call in line for line in _WORK_EXAMPLES.splitlines()) == 1
    # The one-command-at-a-time fact is said once, in the Shell doc; no tail example.
    assert "tail -F" not in text and "tail -F" not in _WORK_EXAMPLES
    assert "one command at a time" not in text and "one command at a time" not in _WORK_EXAMPLES
    shell_doc = " ".join(doc(Shell, inline_depth=0).split())
    assert (
        "A Shell runs one command at a time: use another Shell for a second concurrent command."
    ) in shell_doc
