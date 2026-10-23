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


def test_the_list_keeps_items_and_progress_and_an_item_keeps_its_content():
    text = doc(TodoList, concise=True)
    for wanted in (
        "def add(self, title: str, description: str = '') -> Todo",
        "def get(self, todo: Todo | str) -> Todo",
        "def delete(self, todo: Todo | str) -> None",
        "def activate(self, todo: Todo | str) -> Todo",
        "def complete(self, todo: Todo | str | None = None) -> None",
        "id: str",
        "title: str",
        "status: Literal[open, done]",
        "description: str",
        "comments: list[str]",
    ):
        assert wanted in text, wanted
    for gone in (
        "def comment(",
        "def update(",
        "def set_var(",
        "def get_var(",
        "def comments(",
        "def list_todos(",
        "def status(",
        "deps",
        "created_at",
        "TodoComment",
        "**",
        "source_dir",
    ):
        assert gone not in text, gone


def test_items_are_edited_directly_and_the_list_tracks_progress():
    todos = TodoList()
    first = todos.add("Read filter.py")
    second = todos.add("Write out.html", description="GOAL: survives html.parser")
    assert len(first.id) == 8 and first.id != second.id and isinstance(first, Todo)
    assert todos.activate(first.id) is first and todos.get(first.id) is first
    first.comments.append("filter strips on* attributes")
    todos.complete()  # the active item
    assert first.status == "done" and todos.active() is None
    second.title = "Write /app/out.html"
    second.description += "; CHECK: python3 test_outputs.py"
    todos.complete(second)
    assert second.status == "done" and second.comments == []
    third = todos.add("Verify")
    todos.delete(third.id)
    assert todos.list_todos() == [first, second]
    with pytest.raises(ValueError, match="no item 'deadbeef'"):
        todos.get("deadbeef")
    with pytest.raises(ValueError, match=f"no item '{third.id}'"):
        todos.delete(third)


def test_complete_without_an_item_needs_one_active_or_one_open():
    todos = TodoList()
    only = todos.add("Fix it")
    todos.complete()
    assert only.status == "done"
    todos.add("A")
    todos.add("B")
    with pytest.raises(
        ValueError, match=r"pass the item, or activate one first, then complete\(\)"
    ):
        todos.complete()


def test_todo_manager_calls_keep_working():
    todos = TodoList()
    explore = todos.add("Explore")
    fix = todos.add("Fix", deps=[explore], owner="me")  # hidden keywords
    todos.comment(fix, "legacy argument order")  # self.todo.comment(todo, text), hidden
    todos.complete(explore)  # complete(todo)
    assert fix.deps == [explore.id] and fix.v.owner == "me"
    assert fix.comments == ["legacy argument order"]
    assert explore.status == "done"
    assert (
        todos.get(fix.id) is fix and todos.update(fix.id, title="Fix parser").title == "Fix parser"
    )
    assert todos.list_todos("open") == [fix]


def test_status_shows_progress_the_plan_with_results_and_the_active_card():
    todos = TodoList()
    assert todos.status() == "No items yet."
    steps = [todos.add(f"step {i + 1}") for i in range(7)]
    for step in steps[:5]:
        step.comments.append(f"result of {step.title}")
        todos.complete(step)
    ids = [step.id for step in steps]
    text = todos.status()
    assert text.splitlines()[0] == "2 open · 5 done"
    assert text.splitlines()[1] == "✓ (+1 earlier done)"  # older done rows fold away
    assert f"✓ {ids[4]}  step 5 — result of step 5" in text  # a done row carries its latest note
    assert text.endswith(
        "Nothing active: activate the item you work on to see its description and notes here."
    )

    sixth = todos.activate(steps[5])
    sixth.description = "GOAL: the whole brief\nCHECK: shown in full, line breaks kept"
    for i in range(10):
        sixth.comments.append(f"note {i}")
    text = todos.status()
    assert text.splitlines()[0] == f"2 open · 5 done · working on {ids[5]}"
    assert f"▸ {ids[5]}  step 6  (10 notes)" in text and f"· {ids[6]}  step 7" in text
    assert (
        f"▸ {ids[5]}  step 6\nGOAL: the whole brief\nCHECK: shown in full, line breaks kept" in text
    )
    assert "Notes: (2 earlier not shown)\n3. note 2" in text and text.endswith("10. note 9")
    for hint in ("self.todo", "deactivate", "list_todos", "get("):
        assert hint not in text

    todos.complete()
    todos.complete(steps[6])
    assert todos.status().endswith("All items are done.")


def test_snapshot_round_trip_keeps_ids_notes_and_the_active_item():
    todos = TodoList()
    todos.add("A")
    b = todos.add("B", description="brief")
    todos.activate(b).comments.append("n1")
    restored = TodoList(todos.to_dict())
    assert restored.status() == todos.status()
    assert restored.active().id == b.id and restored.get(b.id).comments == ["n1"]


def test_hand_built_and_added_todos_get_distinct_hex_ids():
    brief = Todo(title="Hand-built brief")
    assert len(brief.id) == 8 and brief.id != Todo(title="another").id
    todos = TodoList()
    ids = {todos.add(f"item {i}").id for i in range(50)}
    assert len(ids) == 50 and all(len(i) == 8 and int(i, 16) >= 0 for i in ids)


def test_comment_takes_a_todo_object_first_and_rejects_ambiguous_ids():
    todos = TodoList()
    first = todos.add("one")
    second = todos.add("two")
    todos.comment(second, "1")  # the object is the target, "1" is the text
    assert second.comments == ["1"] and first.comments == []
    todos.comment("on one", first)  # text, then the target
    assert first.comments == ["on one"]
    with pytest.raises(ValueError, match="ambiguous"):
        todos.comment(first.id, second.id)
    with pytest.raises(TypeError, match="two todos"):
        todos.comment(first, second)
    assert second.comments == ["1"]


def test_legacy_notes_keyword_is_the_description():
    todos = TodoList()
    assert todos.add("x", notes="old-style brief").description == "old-style brief"
    with pytest.raises(ValueError, match="either description or legacy notes"):
        todos.add("y", description="a", notes="b")


def test_the_manager_is_held_not_inherited():
    from nooa.tools import TodoManager

    todos = TodoList()
    assert not isinstance(todos, TodoManager)
    legacy = TodoManager()
    kept = legacy.add("saved before")
    wrapped = TodoList(manager=legacy)
    [item] = wrapped.list_todos()
    assert isinstance(item, Todo) and (item.id, item.title) == (kept.id, kept.title)
    legacy_state = {
        "todos": [{**kept.model_dump(), "notes": "old brief", "description": None}],
        "active_id": kept.id,
    }
    del legacy_state["todos"][0]["description"]
    legacy_state["todos"][0]["comments"] = [
        {"id": "c1", "body": "core note", "created_at": "2026-10-01 10:00"}
    ]
    restored = TodoList(legacy_state)
    assert (restored.active().description, restored.active().comments) == (
        "old brief",
        ["core note"],
    )


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
            "todo.comments.append('child reported: ' + result.report)\n"
            "self.todo.complete(todo)\n"
            "return_result(Done(explanation=result.report))"
        ),
    ]
    # The child's copy of its brief is already active: no id, no activate.
    models.scripts["fix"] = [
        cell(
            "self.todo.active().comments.append('patched parser.py')\nself.todo.complete()\n"
            + WORK_RESULT.format(e="ok", r="FIXED")
        )
    ]
    root = await registry.create(options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="FIXED")
    call = models.llms[None].calls[0]
    rendered = "\n".join(str(m.get("content", "")) for m in call.messages)
    assert "class TodoList" in rendered and "comments: list[str]" in rendered
    for gone in (
        "def set_var(",
        "def list_todos(",
        "deps: list[str]",
        "class TodoComment",
        "TodoManager",
    ):
        assert gone not in rendered, gone
    todos = root._agent.todo
    assert isinstance(todos, TodoList)
    [item] = todos.list_todos()
    assert item.status == "done" and item.comments == ["child reported: FIXED"]


async def _saved(sessions_dir, options, models, build) -> str:
    """Save a session after ``build(agent)`` has filled its to-do list; return its id."""
    models.scripts[None] = [cell("return_result(Done(explanation='ok'))")]
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=models)
    try:
        root = await registry.create(options)
        await asyncio.wait_for(root.prompt("go"), TIMEOUT)
        build(root._agent)
        root._checkpoint()
        await root.wait_for_checkpoint()
        return root.id
    finally:
        await registry.close_all()


async def _loaded_todo(sessions_dir, session_id):
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=AtomModels())
    try:
        loaded = await registry.load(session_id)
        return loaded._agent.todo, loaded._agent.plan()
    finally:
        await registry.close_all()


async def test_a_saved_todo_list_reloads_with_ids_notes_and_the_active_item(
    sessions_dir, options, models
):
    def build(agent):
        agent.todo.add("Read the code")
        agent.todo.activate(agent.todo.add("Fix it", description="the brief")).comments.append(
            "found it"
        )

    todos, plan = await _loaded_todo(
        sessions_dir, await _saved(sessions_dir, options, models, build)
    )
    assert isinstance(todos, TodoList) and all(isinstance(t, Todo) for t in todos.list_todos())
    read, fix = todos.list_todos()
    assert (read.title, fix.title) == ("Read the code", "Fix it") and todos.active().id == fix.id
    assert todos.get(fix.id).comments == ["found it"]
    assert [(p.content, p.status) for p in plan] == [
        ("Read the code", "pending"),
        ("Fix it", "in_progress"),
    ]
    todos.complete()
    assert len(todos.add("Verify").id) == 8


async def test_an_earlier_todo_manager_reloads_as_a_todo_list(sessions_dir, options):
    from atom_test_agents import ATOM_SPEC

    class EarlierAtom(AtomModels):
        def __call__(self, options, storage):
            return super().__call__(options.model_copy(update={"agent_spec": ATOM_SPEC}), storage)

    def build(agent):
        assert not isinstance(agent.todo, TodoList)  # the core TodoManager, as saved before
        legacy = agent.todo.add("Keep the parser fix", description="carried over")
        agent.todo.activate(legacy)
        agent.todo.comment(legacy, "half done")

    todos, plan = await _loaded_todo(
        sessions_dir, await _saved(sessions_dir, options, EarlierAtom(), build)
    )
    assert isinstance(todos, TodoList)
    [item] = todos.list_todos()
    assert len(item.id) == 8 and todos.active().id == item.id  # the saved hex id is kept
    assert (item.title, item.description, item.comments) == (
        "Keep the parser fix",
        "carried over",
        ["half done"],
    )
    todos.get(item.id).comments.append("resumed")  # core TodoComments came back as their text
    todos.complete()
    assert item.comments == ["half done", "resumed"] and item.status == "done"
    assert todos.add("Next").id != item.id
    assert [p.status for p in plan] == ["in_progress"]


def test_a_core_todo_brief_arrives_active_without_colliding_with_new_ids():
    from nooa.tools import Todo as CoreTodo

    child = TodoList()
    child.insert_copy(CoreTodo(id="2", title="Brief from the parent", description="do X"))
    child.get("2").comments.append("started")
    assert child.active().id == "2" and isinstance(child.get("2"), Todo)
    assert child.add("My own step").id != "2"
    child.complete()
    assert child.get("2").status == "done"
