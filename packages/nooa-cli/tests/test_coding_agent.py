# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared coding-agent construction and repository instructions."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from nooa_cli.coding import CodingAgent, discover_agent_instruction_files
from nooa_cli.tui.bootstrap import _instantiate_custom_agent

from nooa.agentdoc import doc
from nooa.skill import Skill, get_slash_commands, slash_command
from nooa.unifiedllm import FakeLLMClient


def test_agent_instructions_follow_repository_hierarchy(tmp_path):
    (tmp_path / ".git").mkdir()
    nested = tmp_path / "packages" / "example"
    nested.mkdir(parents=True)
    root_instructions = tmp_path / "AGENTS.md"
    package_instructions = tmp_path / "packages" / "AGENTS.md"
    root_instructions.write_text("root rule")
    package_instructions.write_text("package rule")

    assert discover_agent_instruction_files(nested) == (
        root_instructions,
        package_instructions,
    )


async def test_coding_agent_uses_observed_shell_and_instruction_context(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text("run the focused tests")
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        assert agent.shell.session is agent._base_shell.session
        assert "run the focused tests" in str(agent.context["repository_instructions"])
        assert "nemo.shell" in agent.skills.activated()
        assert "nemo.repo" in agent.skills.activated()
        assert "nemo.methodwriting" in agent.skills.activated()
        assert "asyncio.gather" in doc(agent.methodwriting)
    finally:
        await agent.close()


async def test_directory_workflow_skills_are_loaded_but_opt_in(tmp_path):
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "root-cause"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: root-cause\ndescription: Diagnose a defect\n---\nFind the cause.\n"
    )

    agent = CodingAgent(
        llm=FakeLLMClient(),
        cwd=tmp_path,
        skills_dirs=[skills_dir],
    )
    try:
        assert "cmd.root-cause" in agent.skills.loaded()
        assert "cmd.root-cause" not in agent.skills.activated()
    finally:
        await agent.close()


async def test_installed_skill_commands_load_without_automatic_activation(tmp_path, monkeypatch):
    class WorkflowSkill(Skill):
        @slash_command("root-cause")
        def root_cause(self) -> str:
            return "diagnose"

    entry_point = SimpleNamespace(
        name="nemo.workflow",
        load=lambda: WorkflowSkill,
    )
    monkeypatch.setattr(
        "nooa.skill_registry.entry_points",
        lambda *, group: [entry_point],
    )

    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        assert "nemo.workflow" in agent.skills.loaded()
        assert "nemo.workflow" not in agent.skills.activated()
        assert [meta.name for meta, _ in get_slash_commands(agent.workflow)] == ["root-cause"]
    finally:
        await agent.close()


async def test_installed_memory_skill_is_left_for_host_configuration(tmp_path, monkeypatch):
    class InstalledMemory(Skill):
        pass

    entry_point = SimpleNamespace(name="nemo.memory", load=lambda: InstalledMemory)
    monkeypatch.setattr(
        "nooa.skill_registry.entry_points",
        lambda *, group: [entry_point],
    )

    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        assert not hasattr(agent, "memory")
        assert "nemo.memory" not in agent.skills.loaded()
    finally:
        await agent.close()


async def test_library_directory_can_be_scoped_by_the_host(tmp_path):
    """Hosts that run several workspaces in one process must be able to
    separate the libs directory.

    SkillWriting puts it on sys.path and imports from it, so a shared one
    leaks agent-authored code between concurrent sessions. The default is
    unchanged for single-workspace hosts like the TUI.
    """
    libs_dir = tmp_path / "scoped" / "libs"
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path, libs_dir=libs_dir)
    try:
        assert agent.libs._path == libs_dir
    finally:
        await agent.close()


async def test_coding_agent_declares_the_host_input_channels(tmp_path):
    """slash_commands and system_messages belong to the coding host.

    InteractiveAgent only declares user_messages: being dispatcher-driven does
    not imply slash commands (a UI affordance whose registry is in this
    package) or host continuations such as keep-going.
    """
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        channels = agent.queue_manager.channels()
        assert {"user_messages", "slash_commands", "system_messages"} <= channels.keys()
        assert agent.slash_commands is agent._slash_commands_in.reader
        assert agent.system_messages is agent._system_messages_in.reader
    finally:
        await agent.close()


async def test_coding_agent_requests_session_title_through_system_channel(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        agent.request_session_title("Fix the TUI input box")

        queued = agent._system_messages_in.snapshot()
        assert len(queued) == 1
        assert "self.rename_session" in queued[0]
        assert "Fix the TUI input box" in queued[0]
    finally:
        await agent.close()


async def test_coding_agent_owns_session_naming(tmp_path):
    """name_session sits with the session model it feeds.

    SessionHandle and SessionTitleUpdated live in nooa_cli.sessions, so the
    generator belongs at this layer rather than in core, which has no notion
    of a session at all.
    """
    from nooa.interactive import InteractiveAgent

    assert hasattr(CodingAgent, "name_session")
    assert not hasattr(InteractiveAgent, "name_session")


def test_repository_instructions_are_read_boundedly(tmp_path, monkeypatch):
    """The cap must bound the read, not just what is kept.

    Truncating after read_text() still pulls a workspace-controlled file into
    memory in full. The budget also has to cover the rendered text — headers,
    separators, truncation markers — or the declared total is not the real one.
    """
    from nooa_cli.coding import instructions

    (tmp_path / ".git").mkdir()
    reads: list[int | None] = []
    real_open = Path.open

    def spying_open(self, *args, **kwargs):
        stream = real_open(self, *args, **kwargs)
        real_read = stream.read

        def read(size=-1):
            reads.append(size)
            return real_read(size)

        stream.read = read  # type: ignore[method-assign]
        return stream

    monkeypatch.setattr(Path, "open", spying_open)
    monkeypatch.setattr(instructions, "_MAX_INSTRUCTION_FILE_CHARS", 100)
    (tmp_path / "AGENTS.md").write_text("x" * 10_000)

    rendered = instructions.render_agent_instructions(tmp_path)

    # Positive sizes only: an unbounded .read() records -1, which satisfies
    # any `<= limit` assertion and made this test pass against the very
    # regression it names.
    assert reads == [101], reads
    assert "[... truncated ...]" in rendered
    assert len(rendered) < 1_000


async def test_a_directly_assigned_protected_attribute_is_still_protected(tmp_path):
    """Protection must not depend on the attribute being skill-owned.

    _protected_owner only found attributes some skill had registered, so a
    protected attribute the agent assigns directly — `self.skills` — had no
    entry and was left unguarded. Registering `mcp.skills` replaced the
    registry itself.
    """
    from nooa.skill import Skill

    class _Evil(Skill):
        pass

    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        registry = agent.skills
        with pytest.raises(ValueError, match="skills"):
            agent.skills.register("mcp.skills", _Evil())
        assert agent.skills is registry

        # A skill-owned protected attr stays protected too.
        shell = agent.shell
        with pytest.raises(ValueError, match="shell"):
            agent.skills.register("mcp.shell", _Evil())
        assert agent.shell is shell

        # Re-binding the same object under its owning name is still allowed.
        agent.skills.register("nemo.shell", shell)
    finally:
        await agent.close()


async def test_coding_agent_delegates_with_same_model_and_workspace(tmp_path, monkeypatch):
    """TUI/ACP coding hosts expose the tested context-isolated worker primitive."""
    observed = {}

    class FakeWorker:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        async def investigate(self, objective: str, supplied_context=None) -> str:
            observed.update(objective=objective, supplied_context=supplied_context)
            return "review complete"

        async def close(self) -> None:
            observed["closed"] = True

    monkeypatch.setattr(CodingAgent, "_worker_type", FakeWorker)
    llm = FakeLLMClient()
    agent = CodingAgent(llm=llm, cwd=tmp_path)
    try:
        todos = [agent.todo.add("Review empty-input handling")]
        result = await agent.delegate("review parser", todos)
        assert result == "review complete"
        assert observed.pop("supplied_context") is todos
        assert observed == {
            "llm": llm,
            "cwd": agent.shell.cwd,
            "init_command": None,
            "objective": objective,
            "closed": True,
        }
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_coding_agent_delegate_merges_todo_notes_and_vars(tmp_path, monkeypatch):
    """A Todo delegation is isolated while running and merged before return."""
    observed = {}

    class FakeWorker:
        def __init__(self, **kwargs):
            observed.update(kwargs)
            self.todo = kwargs["todo"]

        async def investigate(self, objective: str, supplied_context=None) -> str:
            delegated = self.todo.list_todos()[0]
            observed.update(
                objective=objective, supplied_context=supplied_context, delegated=delegated
            )
            self.todo.comment(delegated, "worker finding")
            self.todo.set_var(delegated, "path", "parser.py")
            return "review complete"

        async def close(self) -> None:
            observed["closed"] = True

    monkeypatch.setattr(CodingAgent, "_worker_type", FakeWorker)
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        task = agent.todo.add("review parser", notes="focus on errors")
        original_identity = id(task)

        report = await agent.delegate(task)

        assert report == "review complete"
        assert observed["objective"] == task.title
        assert observed["supplied_context"] is observed["delegated"]
        assert observed["delegated"] is not task
        assert id(agent.todo.get(task)) == original_identity
        assert [comment.body for comment in task.comments] == ["worker finding"]
        assert task.v.path == "parser.py"
        assert observed["closed"] is True
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_coding_agent_delegate_does_not_merge_failed_todo(tmp_path, monkeypatch):
    class FakeWorker:
        def __init__(self, **kwargs):
            self.todo = kwargs["todo"]

        async def investigate(self, objective: str, supplied_context=None) -> str:
            delegated = self.todo.list_todos()[0]
            self.todo.comment(delegated, "partial finding")
            raise RuntimeError("worker failed")

        async def close(self) -> None:
            pass

    monkeypatch.setattr(CodingAgent, "_worker_type", FakeWorker)
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        task = agent.todo.add("review parser")
        with pytest.raises(RuntimeError, match="worker failed"):
            await agent.delegate(task)
        assert task.comments == []
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_coding_agent_delegate_does_not_merge_when_close_fails(tmp_path, monkeypatch):
    class FakeWorker:
        def __init__(self, **kwargs):
            self.todo = kwargs["todo"]

        async def investigate(self, objective: str, supplied_context=None) -> str:
            self.todo.comment(self.todo.list_todos()[0], "worker finding")
            return "review complete"

        async def close(self) -> None:
            raise RuntimeError("close failed")

    monkeypatch.setattr(CodingAgent, "_worker_type", FakeWorker)
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        task = agent.todo.add("review parser")
        with pytest.raises(RuntimeError, match="close failed"):
            await agent.delegate(task)
        assert task.comments == []
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_coding_agent_spawn_merges_todo_before_notification(tmp_path, monkeypatch):
    class FakeWorker:
        def __init__(self, **kwargs):
            self.todo = kwargs["todo"]

        async def investigate(self, objective: str, supplied_context=None) -> str:
            self.todo.comment(self.todo.list_todos()[0], "background finding")
            return "done"

        async def close(self) -> None:
            pass

    monkeypatch.setattr(CodingAgent, "_worker_type", FakeWorker)
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        task = agent.todo.add("review parser. Inspect errors")
        handle = agent.spawn(task)
        notification = await agent.delegates.get()

        assert notification == {
            "objective": "review parser. Inspect errors",
            "report": "done",
            "todo_id": task.id,
        }
        assert [comment.body for comment in task.comments] == ["background finding"]
        assert handle.label == "review parser."
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_coding_agent_rejects_unmanaged_todo(tmp_path):
    from nooa.tools import Todo

    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        with pytest.raises(ValueError, match="not managed"):
            await agent.delegate(Todo(title="foreign"))
    finally:
        await agent.close()


def test_coding_agent_delegation_labels_are_concise() -> None:
    objective = (
        "Architecture/simplicity review of PR #189. Do not edit. "
        "Review every abstraction and report exact file and line references."
    )

    assert CodingAgent._delegation_label(objective) == "Architecture/simplicity review of PR #189."
    assert CodingAgent._delegation_label("first line\nsecond line") == "first line"
    assert CodingAgent._delegation_label(objective, "  API   review  ") == "API review"
    assert CodingAgent._delegation_label("x" * 100) == f"{'x' * 79}…"
    assert CodingAgent._delegation_label("   ") == "Delegated task"


async def test_coding_agent_spawns_delegation_in_background(tmp_path):
    """Background delegation keeps its full objective but displays a short label."""
    started = asyncio.Event()
    release = asyncio.Event()
    objective = "review parser. Do not edit. Report every finding with exact lines."

    class TestAgent(CodingAgent):
        async def delegate(self, received_objective: str, supplied_context=None) -> str:
            assert received_objective == objective
            assert supplied_context == {"path": "parser.py"}
            started.set()
            await release.wait()
            return "review complete"

    agent = TestAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        handle = agent.spawn(objective, {"path": "parser.py"})
        assert handle.label == "review parser."
        assert handle.state == "running"
        await started.wait()
        assert agent.delegates.status() == ""

        release.set()
        assert await agent.delegates.get() == {
            "objective": objective,
            "report": "review complete",
        }
        await asyncio.sleep(0)
        assert handle.state == "done"
    finally:
        await agent.close()


def test_coding_agent_prompt_exposes_bounded_delegation(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        rendered = doc(agent)
        assert (CodingAgent.__doc__ or "").startswith(
            "You are a careful software-development agent working in one local repository."
        )
        assert "spawn" in rendered
        assert "prefer it over awaiting ``delegate()``" in (CodingAgent.__doc__ or "")
        assert "Reports arrive in later ``delegates``" in (CodingAgent.__doc__ or "")
    finally:
        # This sync test does not start shell work; close is covered elsewhere.
        pass


def test_custom_coding_agent_receives_workspace_extension_arguments(tmp_path):
    captured = {}

    class CustomAgent:
        def __init__(self, *, llm, storage, cwd, skills_dirs, summarization):
            captured.update(
                llm=llm,
                storage=storage,
                cwd=cwd,
                skills_dirs=skills_dirs,
                summarization=summarization,
            )

    llm = object()
    storage = object()
    skills_dirs = [tmp_path / "skills"]
    summarization = object()
    agent = _instantiate_custom_agent(
        CustomAgent,
        llm=llm,
        storage=storage,
        working_directory=tmp_path,
        skills_dirs=skills_dirs,
        summarization=summarization,
    )

    assert isinstance(agent, CustomAgent)
    assert captured == {
        "llm": llm,
        "storage": storage,
        "cwd": tmp_path,
        "skills_dirs": skills_dirs,
        "summarization": summarization,
    }


def test_coding_agent_delegation_labels_are_concise() -> None:
    objective = (
        "Architecture/simplicity review of PR #189. Do not edit. "
        "Review every abstraction and report exact file and line references."
    )

    assert CodingAgent._delegation_label(objective) == "Architecture/simplicity review of PR #189."
    assert CodingAgent._delegation_label("first line\nsecond line") == "first line"
    assert CodingAgent._delegation_label(objective, "  API   review  ") == "API review"
    assert CodingAgent._delegation_label("x" * 100) == f"{'x' * 79}…"
    assert CodingAgent._delegation_label("   ") == "Delegated task"


async def test_coding_agent_spawns_delegation_in_background(tmp_path):
    """Background delegation keeps its full objective but displays a short label."""
    started = asyncio.Event()
    release = asyncio.Event()
    objective = "review parser. Do not edit. Report every finding with exact lines."

    class TestAgent(CodingAgent):
        async def delegate(self, received_objective: str, supplied_context=None) -> str:
            assert received_objective == objective
            assert supplied_context == {"path": "parser.py"}
            started.set()
            await release.wait()
            return "review complete"

    agent = TestAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        handle = agent.spawn(objective, {"path": "parser.py"})
        assert handle.label == "review parser."
        assert handle.state == "running"
        await started.wait()
        assert agent.delegates.status() == ""

        release.set()
        assert await agent.delegates.get() == {
            "objective": objective,
            "report": "review complete",
        }
        await asyncio.sleep(0)
        assert handle.state == "done"
    finally:
        await agent.close()
