# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atom's ``TodoList``: four calls for the model, the full ``TodoManager`` underneath."""

import asyncio

import pytest
from atom_test_agents import AtomModels, cell
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore
from nooa_atom.tools.todo_list import Todo, TodoList

from nooa.agentdoc import doc
from nooa.interactive import Done

UNIFIED = "nooa_atom.agent:AtomAgent"
TIMEOUT = 60
WORK_RESULT = (
    "return_result(Done(explanation='{e}', result=TaskResult(solution_description='s', "
    "evidence='e', how_to_verify='v', report='{r}')))"
)


def test_the_documented_surface_is_four_calls_and_four_fields():
    text = doc(TodoList, concise=True)
    for wanted in (
        "def add(self, title: str, description: str = '') -> Todo",
        "def activate(self, todo: Todo | str) -> Todo",
        "def comment(self, text: str) -> None",
        "def complete(self) -> None",
        "id: str",
        "title: str",
        "status: Literal[open, done]",
        "description: str",
    ):
        assert wanted in text, wanted
    for gone in (
        "def get(",
        "def update(",
        "def set_var(",
        "def get_var(",
        "def comments(",
        "def list_todos(",
        "def status(",
        "deps",
        "created_at",
        "TodoComment",
        "**vars",
        "source_dir",
    ):
        assert gone not in text, gone


def test_ids_count_up_and_calls_act_on_the_active_item():
    todos = TodoList()
    first = todos.add("Read filter.py")
    second = todos.add("Write out.html", description="GOAL: survives html.parser")
    assert (first.id, second.id) == ("1", "2")
    assert isinstance(first, Todo)
    todos.activate(1)
    todos.comment("filter strips on* attributes")
    todos.complete()
    assert first.status == "done" and todos.active() is None
    todos.activate("2")
    todos.comment("svg survives")
    assert [c.body for c in second.comments] == ["svg survives"]
    assert todos.add("Verify").id == "3"


def test_one_open_item_needs_no_activate_and_ambiguity_says_what_to_do():
    todos = TodoList()
    only = todos.add("Fix it")
    todos.comment("found it")
    todos.complete()
    assert only.status == "done" and [c.body for c in only.comments] == ["found it"]
    todos.add("A")
    todos.add("B")
    with pytest.raises(ValueError, match=r"activate\(id\) first, then comment\(\)"):
        todos.comment("which one?")
    with pytest.raises(ValueError, match="no item '9'"):
        todos.complete("9")


def test_todo_manager_calls_keep_working():
    todos = TodoList()
    explore = todos.add("Explore")
    fix = todos.add("Fix", deps=[explore], owner="me")  # hidden keywords
    todos.comment(fix, "legacy argument order")  # comment(todo, text)
    todos.complete(explore)  # complete(todo)
    assert fix.deps == ["1"] and fix.v.owner == "me"
    assert [c.body for c in fix.comments] == ["legacy argument order"]
    assert explore.status == "done"
    assert todos.get(2) is fix and todos.update("2", title="Fix parser").title == "Fix parser"
    assert [t.id for t in todos.list_todos("open")] == ["2"]


def test_status_shows_the_plan_then_the_active_item_without_call_hints():
    todos = TodoList()
    assert todos.status() == "(empty)"
    for i in range(6):
        todos.add(f"step {i + 1}")
    for i in range(1, 5):
        todos.activate(str(i))
        todos.complete()
    todos.activate("5")
    todos.update("5", description="GOAL: the whole brief, shown in full")
    for i in range(8):
        todos.comment(f"note {i}")
    text = todos.status()
    assert text.splitlines()[0] == "(+1 done)"  # older done items fold away
    assert "✓ 4  step 4" in text and "▸ 5  step 5" in text and "· 6  step 6" in text
    assert "Active: 5  step 5\nGOAL: the whole brief, shown in full" in text
    assert "Notes: (+2 earlier)\n- note 2" in text and "- note 7" in text
    for hint in ("self.todo", "deactivate", "list_todos", "get("):
        assert hint not in text


def test_snapshot_round_trip_keeps_ids_notes_and_the_active_item():
    todos = TodoList()
    todos.add("A")
    todos.add("B", description="brief")
    todos.activate("2")
    todos.comment("n1")
    restored = TodoList(todos.to_dict())
    assert restored.status() == todos.status()
    assert restored.add("C").id == "3"


def test_subclass_hides_what_its_base_documents():
    from nooa import hidden

    class Base:
        def shown(self) -> None:
            """Base doc."""

    class Child(Base):
        @hidden
        def shown(self) -> None:
            pass

    assert "shown" not in doc(Child, concise=True)


@pytest.fixture
def models():
    return AtomModels()


@pytest.fixture
async def registry(sessions_dir, models):
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=models)
    yield registry
    await registry.close_all()


@pytest.fixture
def options(tmp_path, sessions_dir):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return SessionOptions(workspace=workspace, agent_spec=UNIFIED, sessions_dir=sessions_dir)


async def test_unified_agent_shows_the_short_list_and_a_brief_arrives_active(
    registry, options, models
):
    models.scripts[None] = [
        cell(
            "todo = self.todo.add('fix the parser', description='CHILD-BRIEF: fix it')\n"
            "result = await self.jobs.launch(Subagent('fix parser').run(todo), name='fix')\n"
            "self.todo.activate(todo)\n"
            "self.todo.comment('child reported: ' + result.report)\n"
            "self.todo.complete()\n"
            "return_result(Done(explanation=result.report))"
        ),
    ]
    # The child's copy of its brief is already active: no id, no activate.
    models.scripts["fix"] = [
        cell("self.todo.comment('patched parser.py')\nself.todo.complete()\n" + WORK_RESULT.format(e="ok", r="FIXED"))
    ]
    root = await registry.create(options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="FIXED")
    call = models.llms[None].calls[0]
    rendered = "\n".join(str(m.get("content", "")) for m in call.messages)
    assert "class TodoList" in rendered and "def complete(self) -> None" in rendered
    for gone in ("def set_var(", "def list_todos(", "deps: list[str]", "class TodoComment", "TodoManager"):
        assert gone not in rendered, gone
    todos = root._agent.todo
    assert isinstance(todos, TodoList)
    [item] = todos.list_todos()
    assert item.status == "done" and [c.body for c in item.comments] == ["child reported: FIXED"]
