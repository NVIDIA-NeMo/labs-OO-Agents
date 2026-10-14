# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every call advertised in Atom's footers and context blocks must work as written.

The footers are rendered from real objects; each advertised call is then
executed against a fake owner whose collaborators are real (``vars``, ``v``,
``todo``) or autospecced from the real classes (``skills``, ``events``), so a
wrong name, a wrong signature or a wrong ``await`` fails here.
"""

import re
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
from nooa_atom.agent import AtomAgent
from nooa_atom.skills.manager import SkillManager

from nooa.context_blocks.models import COLLAPSE_HINT, ContextWindowStats
from nooa.interactive import AgentVars
from nooa.runtime.events import EventsApi
from nooa.storage.snapshot_vars import SnapshotVars
from nooa.tools.todo import TodoManager
from nooa.unifiedllm import FakeLLMClient

_CALL = re.compile(r"((?:await )?self\.[\w.]+(?:\([^()]*\))?)")


def _backticked(text: str) -> list[str]:
    return re.findall(r"`([^`]+)`", text)


def _calls(text: str) -> list[str]:
    """Advertised calls; bare names such as ``doc(self.events)`` arguments are skipped."""
    return [match.group(1) for match in _CALL.finditer(text) if "(" in match.group(1)]


def _owner() -> SimpleNamespace:
    owner = SimpleNamespace(vars=SnapshotVars())
    owner.v = AgentVars(owner)
    owner.v.note = "kept"
    owner.todo = TodoManager()
    owner.skills = create_autospec(SkillManager, instance=True)
    owner.events = create_autospec(EventsApi, instance=True)
    return owner


async def _run(snippet: str, owner: SimpleNamespace, todo_id: str = "") -> None:
    code = (
        snippet.replace("<name>", "note")
        .replace("(id,", f'("{todo_id}",')
        .replace("(id)", f'("{todo_id}")')
        .replace(", ...)", ', "progress")')
    )
    namespace: dict = {}
    body = "\n".join(f"    {line}" for line in code.splitlines())
    exec(f"async def _cell(self):\n{body}\n", namespace)  # noqa: S102 - test-owned footer text
    await namespace["_cell"](owner)


@pytest.fixture
async def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    skills_dir = tmp_path / "skills"
    (skills_dir / "review").mkdir(parents=True)
    (skills_dir / "review" / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review a change\n---\nRead the diff.\n"
    )
    made = AtomAgent(llm=FakeLLMClient(), cwd=tmp_path, skills_dirs=[skills_dir])
    yield made
    await made.aclose()


async def test_workspace_state_examples_run(agent):
    rendered = agent._workspace_state_context()
    snippets = [s for s in _backticked(rendered) if "self." in s and s != "self.v"]
    snippets = [s for s in snippets if s not in ("self.shell", "self.repo")]
    assert len(snippets) >= 3, rendered
    for snippet in snippets:
        owner = _owner()
        await _run(snippet, owner)
    # Each example does what the footer says.
    owner = _owner()
    await _run("del self.v.<name>", owner)
    assert "note" not in owner.vars
    owner = _owner()
    await _run("self.vars.clear()", owner)
    assert len(owner.vars) == 0


async def test_skills_block_examples_run(agent):
    rendered = agent.skills.status()
    calls = _calls(rendered)
    assert any(call.startswith("await self.skills.activate(") for call in calls), rendered
    assert len(calls) == 3, calls
    owner = _owner()
    for call in calls:
        await _run(call, owner)


@pytest.mark.parametrize("with_comment", [False, True])
async def test_active_todo_hints_run(with_comment):
    owner = _owner()
    todo = owner.todo.add("Write the fix")
    owner.todo.add("Other task")
    if with_comment:
        owner.todo.comment(todo.id, "started")
    owner.todo.activate(todo)
    calls = _calls(owner.todo.status())
    assert calls, owner.todo.status()
    for call in calls:
        owner.todo.activate(todo)
        await _run(call, owner, todo.id)


async def test_todo_list_hints_run():
    owner = _owner()
    todos = [owner.todo.add(f"task {n}", description="d") for n in range(15)]
    owner.todo.complete(todos[0])
    calls = _calls(owner.todo.status(max_items=3))
    assert {"self.todo.list_todos()", "self.todo.clear_done()"} <= set(calls), calls
    for call in calls:
        await _run(call, owner, todos[1].id)


async def test_context_usage_examples_run():
    tight = ContextWindowStats(
        context_blocks_count=1,
        events_count=1,
        context_blocks_chars=1,
        events_chars=1,
        prompt_tokens=90_000,
        model_context_window=100_000,
    ).format()
    assert COLLAPSE_HINT in tight
    calls = _calls(tight)
    assert calls == ['self.events.collapse("2", "40", "summary of events 2-40")'], calls
    owner = _owner()
    for call in calls:
        await _run(call, owner)
    owner.events.collapse.assert_called_once_with("2", "40", "summary of events 2-40")
