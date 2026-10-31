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
from nooa.tools.todo import Todo as CoreTodo

UNIFIED = "nooa_atom.agent:AtomAgent"
TIMEOUT = 60
WORK_RESULT = (
    "return Done(explanation='{e}', result=TaskResult(solution_description='s', "
    "evidence='e', how_to_verify='v', report='{r}'))"
)


def test_the_list_keeps_items_and_progress_and_an_item_keeps_its_content():
    text = doc(TodoList, concise=True)
    for wanted in (
        "def add(self, title: str, description: str = '') -> Todo",
        "def get(self, id_or_title: str) -> Todo",
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


@pytest.mark.parametrize("concise", [False, True])
def test_title_lookup_rules_are_in_generated_docs(concise):
    text = doc(TodoList, concise=concise)
    assert "Look up an exact ID or unique exact title" in text
    assert "get/activate/complete/delete" not in text
    assert "IDs win; use an ID for ambiguous titles" in text
    assert "todo_status context block" in text
    if not concise:
        full = " ".join(text.split())
        assert "current titles across open and done items" in full
        assert "no trimming or fuzzy matching" in full
        assert "ambiguous titles raise ValueError" in full
        assert "Todo objects and legacy integer references" not in full


def test_get_advertises_and_accepts_the_actual_string_lookup_keyword():
    import inspect

    signature = inspect.signature(TodoList.get)
    assert list(signature.parameters) == ["self", "id_or_title"]
    todos = TodoList()
    item = todos.add("Current title")
    assert todos.get(id_or_title=item.id) is item
    assert todos.get(id_or_title=item.title) is item
    # No public **kwargs bag or misleading renamed signature. Positional old
    # Todo/int references are still ID-only, as the compatibility cases below check.
    with pytest.raises(TypeError, match="unexpected keyword argument 'todo'"):
        todos.get(todo=item.id)


def test_public_operations_accept_a_unique_exact_title():
    todos = TodoList()
    first = todos.add("Read the code")
    second = todos.add("Fix the parser", deps=[first])
    assert todos.get(first.title) is first
    assert todos.activate(first.title) is first
    assert f"working on {first.id}" in todos.status()
    todos.complete(first.title)
    assert first.status == "done" and todos.active() is None
    assert todos.get(first.title) is first  # done items still participate
    todos.delete(first.title)
    assert todos.list_todos() == [second] and second.deps == []


@pytest.mark.parametrize("method", ["get", "activate", "complete", "delete"])
def test_exact_ids_win_even_over_duplicate_titles(method):
    todos = TodoList()
    target = todos.add("By ID")
    others = [todos.add(target.id), todos.add(target.id)]
    result = getattr(todos, method)(target.id)
    if method in ("get", "activate"):
        assert result is target
    elif method == "complete":
        assert target.status == "done"
    else:
        assert todos.list_todos() == others
    assert all(t.status == "open" for t in others)


@pytest.mark.parametrize("method", ["get", "activate", "complete", "delete"])
@pytest.mark.parametrize("done_count", [0, 1, 2])
def test_ambiguous_titles_fail_without_side_effects(method, done_count):
    todos = TodoList()
    matches = [todos.add("Duplicate"), todos.add("Duplicate")]
    for item in matches[:done_count]:
        todos.complete(item)
    active = todos.add("Keep active", deps=matches)
    todos.activate(active)
    active.comments.append("keep this note")
    before = todos.to_dict()
    with pytest.raises(ValueError, match="ambiguous item title") as error:
        getattr(todos, method)("Duplicate")
    assert all(item.id in str(error.value) for item in matches)
    assert "use an id" in str(error.value)
    assert todos.to_dict() == before
    assert todos.active() is active and todos.get(active) is active


@pytest.mark.parametrize("method", ["get", "activate", "complete", "delete"])
def test_ambiguous_titles_do_not_normalize_a_wrapped_legacy_manager(method):
    from nooa.tools.todo import TodoManager

    manager = TodoManager()
    first = manager.add("Duplicate")
    second = manager.add("Duplicate", deps=[first])
    manager.activate(second)
    manager.comment(first, "legacy note")
    before = manager.to_dict()
    todos = TodoList(manager=manager)
    with pytest.raises(ValueError, match="ambiguous item title"):
        getattr(todos, method)("Duplicate")
    assert manager.to_dict() == before
    assert manager.get(first.id) is first and manager.get(second.id) is second


def test_ambiguous_title_error_bounds_candidate_ids():
    todos = TodoList()
    matches = [todos.insert_copy(Todo(id=f"candidate-{i}", title="Duplicate")) for i in range(8)]
    with pytest.raises(ValueError, match="ambiguous item title") as error:
        todos.get("Duplicate")
    text = str(error.value)
    assert all(item.id in text for item in matches[:5])
    assert all(item.id not in text for item in matches[5:])
    assert "+3 more" in text


@pytest.mark.parametrize("title", ["", " ", "  ", "\t\n"])
def test_empty_and_whitespace_titles_use_literal_exact_equality(title):
    todos = TodoList()
    item = todos.add(title)
    assert todos.get(title) is item
    todos.add(title)
    with pytest.raises(ValueError, match="ambiguous item title"):
        todos.get(title)
    assert todos.get(item.id) is item


@pytest.mark.parametrize("method", ["get", "activate", "complete", "delete"])
def test_unknown_or_inexact_titles_keep_the_missing_item_error(method):
    todos = TodoList()
    todos.add("Fix the parser")
    before = todos.to_dict()
    for query in ("fix the parser", "Fix", "parser", " Fix the parser", "Fix  the parser", "", " "):
        with pytest.raises(ValueError) as error:
            getattr(todos, method)(query)
        assert (
            str(error.value)
            == f"no item {query!r}; valid ids: {todos.to_dict()['todos'][0]['id']} (Fix the parser)"
        )
        assert todos.to_dict() == before


def test_lookup_errors_list_the_valid_ids_bounded():
    todos = TodoList()
    with pytest.raises(ValueError, match="no item 'x'; the list is empty: add one with"):
        todos.get("x")
    items = [todos.add(f"Item {n} " + "x" * 40) for n in range(10)]
    with pytest.raises(ValueError) as error:
        todos.get("missing")
    text = str(error.value)
    assert f"{items[0].id} (Item 0 " in text and "…)" in text
    assert items[8].id not in text and text.endswith("+2 more (see todo_status)")


def test_misuses_name_what_to_use_instead():
    todos = TodoList()
    with pytest.raises(TypeError, match=r"add\(\) takes add\(title, description=''\), got 3"):
        todos.add("Fix the parser", "the brief", "more")
    assert todos.to_dict()["todos"] == []
    item = todos.add("Fix the parser")
    with pytest.raises(
        TypeError, match=r"self.todo is not subscriptable; use self.todo.get\('1'\)"
    ):
        todos["1"]
    with pytest.raises(TypeError, match=r"write v.commit = value or v.set\('commit', value\)"):
        item.v["commit"] = "abc"
    with pytest.raises(TypeError, match=r"read v.commit or v.get\('commit'\)"):
        _ = item.v["commit"]


@pytest.mark.parametrize("todo_type", [Todo, CoreTodo])
@pytest.mark.parametrize("method", ["get", "activate", "complete", "delete"])
def test_todo_object_references_resolve_only_by_id(method, todo_type):
    todos = TodoList()
    target = todos.add("Managed")
    other = todos.add("Other")
    reference = todo_type(id=target.id, title=other.title)
    result = getattr(todos, method)(reference)
    if method in ("get", "activate"):
        assert result is target
    elif method == "complete":
        assert target.status == "done"
    else:
        assert todos.list_todos() == [other]
    assert other.status == "open"


@pytest.mark.parametrize("todo_type", [Todo, CoreTodo])
@pytest.mark.parametrize("method", ["get", "activate", "complete", "delete"])
def test_stale_todo_objects_never_retarget_by_title_or_stringified_id(method, todo_type):
    todos = TodoList()
    reference = todo_type(id="stale-id", title="Reused title")
    todos.insert_copy(reference)
    todos.delete(reference)
    todos.add(reference.title)
    todos.add(reference.id)
    before = todos.to_dict()
    with pytest.raises(ValueError, match="no item 'stale-id'"):
        getattr(todos, method)(reference)
    assert todos.to_dict() == before


@pytest.mark.parametrize("method", ["get", "activate", "complete", "delete"])
def test_legacy_integer_references_remain_id_only(method):
    todos = TodoList()
    target = todos.insert_copy(Todo(id="7", title="By integer ID"))
    other = todos.add("7")
    result = getattr(todos, method)(7)
    if method in ("get", "activate"):
        assert result is target
    elif method == "complete":
        assert target.status == "done"
    if method != "delete":
        todos.delete(target)
    before = todos.to_dict()
    with pytest.raises(ValueError, match="no item '7'"):
        getattr(todos, method)(7)
    assert todos.to_dict() == before
    assert todos.get("7") is other  # only explicit strings get the title fallback


def test_titles_follow_rename_deletion_and_snapshot_state_without_a_cache():
    todos = TodoList()
    first = todos.add("Original")
    second = todos.add("Other")
    assert todos.get("Original") is first
    first.title = "Renamed"
    with pytest.raises(ValueError, match="no item 'Original'"):
        todos.get("Original")
    assert todos.get("Renamed") is first
    todos.update(second.id, title="Renamed")
    with pytest.raises(ValueError, match="ambiguous item title"):
        todos.get("Renamed")
    todos.delete(first)
    assert todos.get("Renamed") is second
    todos.activate("Renamed").comments.append("current note")
    restored = TodoList(todos.to_dict())
    assert restored.status() == todos.status()
    assert restored.get("Renamed") is restored.active()
    assert restored.get("Renamed").id == second.id
    restored.complete("Renamed")
    assert restored.get("Renamed").status == "done"
    with pytest.raises(ValueError, match="already done"):
        restored.activate("Renamed")
    restored.delete("Renamed")
    replacement = restored.add("Renamed")
    assert restored.get("Renamed") is replacement
    restored.from_dict(todos.to_dict())
    assert restored.get("Renamed").id == second.id


def test_legacy_comment_order_and_passthroughs_still_use_only_ids():
    todos = TodoList()
    first = todos.add("First")
    second = todos.add("Second")
    todos.activate(first)
    todos.comment(second.title)  # title-looking note still goes on the active item
    todos.comment(first.id, second.title)  # id first, text second
    todos.comment(second.title, first.id)  # text first, id second
    assert first.comments == [second.title] * 3 and second.comments == []
    for args in (("note", second.title), (first.title, "note")):
        with pytest.raises(ValueError, match="no item"):
            todos.comment(*args)
    assert todos.update(first.title, title="changed") is None
    assert todos.done(first.title) is None
    assert todos.remove(first.title) is False
    assert first.title == "First" and first.status == "open"
    assert first.comments == [second.title] * 3 and second.comments == []


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
            "return Done(explanation=result.report)"
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
    assert "unique exact title" in rendered and "IDs win" in rendered
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
    models.scripts[None] = [cell("return Done(explanation='ok')")]
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
    assert todos.get("Fix it") is fix and todos.get("Read the code") is read
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


@pytest.mark.parametrize(
    ("note", "text"),
    [
        (123, "123"),
        (None, "None"),
        ({"x": 1}, "{'x': 1}"),
        ({"body": "dict note"}, "dict note"),
        ("plain", "plain"),
    ],
)
async def test_any_appended_note_survives_a_real_save_and_reload(
    sessions_dir, options, models, note, text
):
    from nooa.tools.todo import TodoComment

    ids = {}

    def build(agent):
        first = agent.todo.add("Read the code")
        second = agent.todo.add("Fix it", description="the brief")
        first.comments.append("ok")
        second.comments.append(note)  # comments is a plain list: nothing validates the append
        second.comments.append(TodoComment(body="core note"))
        agent.todo.complete(first)
        agent.todo.activate(second)
        ids.update(first=first.id, second=second.id)

    todos, plan = await _loaded_todo(
        sessions_dir, await _saved(sessions_dir, options, models, build)
    )
    # The whole list and the active item come back, not an older snapshot.
    assert [t.id for t in todos.list_todos()] == [ids["first"], ids["second"]]
    assert todos.active().id == ids["second"]
    assert todos.get(ids["second"]).comments == [text, "core note"]
    assert todos.get(ids["first"]).comments == ["ok"]
    assert [(p.content, p.status) for p in plan] == [
        ("Read the code", "completed"),
        ("Fix it", "in_progress"),
    ]
    assert text in todos.status()
