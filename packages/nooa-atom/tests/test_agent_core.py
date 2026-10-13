# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atom agent construction and repository instructions."""

from types import SimpleNamespace

import pytest
from nooa_atom.agent import (
    AtomAgent,
    SlashCommandRegistry,
    discover_agent_instruction_files,
)

from nooa.skill import Skill, get_slash_commands, slash_command
from nooa.unifiedllm import FakeLLMClient


async def test_aclose_awaits_background_components_and_leaves_the_client_open(tmp_path):
    """The agent does not own its model client: the session closes one it created."""
    from unittest.mock import AsyncMock

    agent = AtomAgent(llm=FakeLLMClient(), cwd=tmp_path)
    calls = []
    agent.event_manager.on_close(AsyncMock(side_effect=lambda: calls.append("component")))
    agent.llm.aclose = AsyncMock(side_effect=lambda: calls.append("client"))
    await agent.aclose()
    assert calls == ["component"]


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


async def test_atom_agent_uses_observed_shell_and_instruction_context(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text("run the focused tests")
    agent = AtomAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        assert agent.shell.session is agent._base_shell.session
        assert "run the focused tests" in str(agent.context["repository_instructions"])
        assert "nemo.shell" in agent.skills.activated()
        assert "nemo.repo" in agent.skills.activated()
    finally:
        await agent.aclose()


async def test_directory_workflow_skills_are_loaded_but_opt_in(tmp_path):
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "root-cause"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: root-cause\ndescription: Diagnose a defect\n---\nFind the cause.\n"
    )

    agent = AtomAgent(
        llm=FakeLLMClient(),
        cwd=tmp_path,
        skills_dirs=[skills_dir],
    )
    try:
        assert "cmd.root-cause" in agent.skills.loaded()
        assert "cmd.root-cause" not in agent.skills.activated()
    finally:
        await agent.aclose()


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

    agent = AtomAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        assert "nemo.workflow" in agent.skills.loaded()
        assert "nemo.workflow" not in agent.skills.activated()
        assert [meta.name for meta, _ in get_slash_commands(agent.workflow)] == ["root-cause"]
    finally:
        await agent.aclose()


async def test_installed_memory_skill_is_not_automatically_attached(tmp_path, monkeypatch):
    class InstalledMemory(Skill):
        pass

    entry_point = SimpleNamespace(name="nemo.memory", load=lambda: InstalledMemory)
    monkeypatch.setattr(
        "nooa.skill_registry.entry_points",
        lambda *, group: [entry_point],
    )

    agent = AtomAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        assert not hasattr(agent, "memory")
        assert "nemo.memory" not in agent.skills.loaded()
    finally:
        await agent.aclose()


async def test_library_directory_can_be_scoped_by_the_host(tmp_path):
    """Hosts that run several workspaces in one process must be able to
    separate the libs directory.

    SkillWriting puts it on sys.path and imports from it, so a shared one
    leaks agent-authored code between concurrent sessions. The default is
    unchanged for single-workspace hosts like the TUI.
    """
    libs_dir = tmp_path / "scoped" / "libs"
    agent = AtomAgent(llm=FakeLLMClient(), cwd=tmp_path, libs_dir=libs_dir)
    try:
        assert agent.libs._path == libs_dir
    finally:
        await agent.aclose()


async def test_atom_agent_declares_the_host_input_channels(tmp_path):
    """slash_commands and system_messages belong to the agent host.

    InteractiveAgent only declares user_messages: being dispatcher-driven does
    not imply slash commands (a UI affordance whose registry is in this
    package) or host continuations such as keep-going.
    """
    agent = AtomAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        channels = agent.queue_manager.channels()
        assert {"user_messages", "slash_commands", "system_messages"} <= channels.keys()
        # The public name is the command registry a Session lists and runs.
        assert isinstance(agent.slash_commands, SlashCommandRegistry)
        assert agent.system_messages is agent._system_messages_in.reader
    finally:
        await agent.aclose()


async def test_atom_agent_owns_session_naming(tmp_path):
    """name_session sits with the session model it feeds.

    Sessions live in nooa_atom.session, so the generator belongs at this
    layer rather than in core, which has no notion of a session at all.
    """
    from nooa.interactive import InteractiveAgent

    assert hasattr(AtomAgent, "name_session")
    assert not hasattr(InteractiveAgent, "name_session")


def test_repository_instructions_are_read_boundedly(tmp_path, monkeypatch):
    """The cap must bound the read, not just what is kept.

    Truncating after read_text() still pulls a workspace-controlled file into
    memory in full. The budget also has to cover the rendered text — headers,
    separators, truncation markers — or the declared total is not the real one.
    """
    from nooa_atom.agent import instructions

    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text("x" * 1000)
    reads: list[int] = []
    real_fdopen = instructions.os.fdopen

    class BoundedStream:
        def __init__(self, *args, **kwargs):
            self.stream = real_fdopen(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, size=-1):
            reads.append(size)
            assert 0 <= size <= 101
            return self.stream.read(size)

    monkeypatch.setattr(instructions, "_MAX_INSTRUCTION_FILE_CHARS", 100)
    monkeypatch.setattr(instructions.os, "fdopen", BoundedStream)

    rendered = instructions.render_agent_instructions(tmp_path)

    assert reads == [101]
    assert "[... truncated ...]" in rendered
    assert len(rendered) <= instructions._MAX_INSTRUCTION_TOTAL_CHARS


def test_repository_instructions_allow_a_symlinked_workspace(tmp_path):
    from nooa_atom.agent.instructions import render_agent_instructions

    root = tmp_path / "real"
    root.mkdir()
    (root / "AGENTS.md").write_text("reachable repository instructions")
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    assert "reachable repository instructions" in render_agent_instructions(alias)
    (root / "AGENTS.md").unlink()
    outside = tmp_path / "outside.md"
    outside.write_text("outside instructions")
    (root / "AGENTS.md").symlink_to(outside)
    assert not render_agent_instructions(alias)


def test_repository_instructions_allow_a_symlink_within_the_boundary(tmp_path):
    """A symlinked AGENTS.md whose target stays inside the repo (the common
    AGENTS.md -> CLAUDE.md pattern) must not be rejected outright -- only a
    symlink that actually escapes the boundary should be. _is_safe_path
    already rejects any symlinked ANCESTOR DIRECTORY on the path to the
    file; the file itself being a symlink to ordinary text content in the
    same tree carries no additional risk.
    """
    from nooa_atom.agent.instructions import render_agent_instructions

    root = tmp_path / "repo"
    root.mkdir()
    (root / "CLAUDE.md").write_text("claude-specific instructions")
    (root / "AGENTS.md").symlink_to(root / "CLAUDE.md")

    rendered = render_agent_instructions(root)
    assert "claude-specific instructions" in rendered


async def test_atom_agent_owns_bounded_application_state_context(tmp_path):
    agent = AtomAgent(cwd=tmp_path, llm=FakeLLMClient())
    try:
        agent.vars["token"] = "private-value"
        agent.shell.cwd = "</workspace_state>\n" + "x" * 500
        rendered = agent._workspace_state_context()
        assert "1 persistent vars" in rendered
        assert "print(self.v.items())" in rendered
        assert "private-value" not in rendered
        assert "</workspace_state>" not in rendered
        assert len(rendered) < 600
    finally:
        await agent.aclose()


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

    agent = AtomAgent(llm=FakeLLMClient(), cwd=tmp_path)
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
        await agent.aclose()


async def test_summarization_status_reports_installed_token_budget(tmp_path):
    from nooa.interactive import SummarizationConfig

    agent = AtomAgent(
        llm=FakeLLMClient(),
        cwd=tmp_path,
        summarization=SummarizationConfig(threshold_fraction=0.60),
    )
    try:
        summarizer = agent._summarizers[0]
        status = agent.get_summarization_status()

        assert status["has_summarizer"] is True
        assert status["policy"] == "token_budget"
        assert status["max_tokens"] == summarizer.config.max_tokens
        assert status["threshold_fraction"] == 0.60
        assert status["preserve_recent"] == summarizer.config.preserve_recent
        assert status["compaction_pending"] is False
        assert status["compaction_ready"] is False
    finally:
        await agent.aclose()


async def test_summarization_status_reports_disabled_policy(tmp_path):
    from nooa.interactive import SummarizationConfig

    agent = AtomAgent(
        llm=FakeLLMClient(),
        cwd=tmp_path,
        summarization=SummarizationConfig(policy="none"),
    )
    try:
        assert agent.get_summarization_status() == {
            "active_events": 0,
            "summary_count": 0,
            "summary_tags": [],
            "has_summarizer": False,
            "policy": "none",
            "current_tokens": 0,
            "max_tokens": 0,
            "threshold_fraction": None,
            "preserve_recent": 0,
            "compaction_pending": False,
            "compaction_ready": False,
        }
    finally:
        await agent.aclose()


def test_the_session_title_request_asks_for_an_awaited_rename():
    """The title request is text a host submits; it names the awaited rename."""
    from nooa_atom.agent.agent import session_title_request

    prompt = session_title_request("  fix the flaky parser test  ")
    assert prompt.startswith("[session-title]")
    assert 'await self.rename_session("your title")' in prompt
    assert "<opening_user_message>\nfix the flaky parser test\n</opening_user_message>" in prompt
    assert not hasattr(AtomAgent, "request_session_title")


async def test_rename_session_needs_a_session(tmp_path):
    agent = AtomAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        with pytest.raises(RuntimeError, match="not running in a session"):
            await agent.rename_session("Parser test fix")
    finally:
        await agent.aclose()


@pytest.mark.parametrize("module", ["agent", "experimental_agent"])
def test_cells_see_the_turn_types_but_not_the_helpers(module):
    from importlib import import_module

    from nooa.agentdoc._visibility import filter_mro_module_globals

    cls = getattr(
        import_module(f"nooa_atom.agent.{module}"),
        "AtomAgent" if module == "agent" else "ExperimentalAtomAgent",
    )
    names = set(filter_mro_module_globals(cls))
    assert {"Done", "NeedInput", "Waiting", "TaskResult", "ChildResult"} <= names
    assert {"ChildFailedError", "DepthLimitError"} <= names
    hidden = {
        "_todo_prompt",
        "_report_text",
        "_V2_CONTEXT",
        "require_result",
        "session_title_request",
        "SessionPort",
    }
    assert names & hidden == set()


def test_the_handle_prompt_names_every_input_channel():
    from nooa_atom.agent.experimental_agent import ExperimentalAtomAgent

    for cls in (AtomAgent, ExperimentalAtomAgent):
        text = cls.handle.__doc__ or ""
        for channel in ("user_messages", "system_messages", "slash_commands", "delegates"):
            assert f'"{channel}"' in text, (cls.__name__, channel)
    assert "SessionInfo" not in (AtomAgent.get_summarization_status.__doc__ or "")


async def test_cd_moves_the_repo_tools_with_the_shell(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "x.py").write_text("def only_in_sub():\n    pass\n")
    agent = AtomAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        await agent.shell.run("cd sub")
        assert agent.shell.cwd == sub.resolve()
        assert agent.repo.cwd == agent.shell.cwd
        assert agent.repo.root == tmp_path.resolve()
        result = await agent.repo.symbols("x.py")
        assert result.diagnostic is None
        assert "only_in_sub" in str(result)
        assert f"cwd={str(sub.resolve())!r}" in repr(agent.repo)
        assert f"root={str(tmp_path.resolve())!r}" in repr(agent.repo)
    finally:
        await agent.aclose()


async def test_the_state_block_shows_the_shell_directory_and_the_repo_root(tmp_path, monkeypatch):
    elsewhere = tmp_path / "process-cwd"
    elsewhere.mkdir()
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    monkeypatch.chdir(elsewhere)
    agent = AtomAgent(llm=FakeLLMClient(), cwd=repo)
    try:
        await agent.shell.run("cd pkg")
        rendered = agent._workspace_state_context()
        assert str((repo / "pkg").resolve()) in rendered
        assert f"Repository root (the boundary for repo searches): {repo.resolve()}" in rendered
        assert str(elsewhere) not in rendered
    finally:
        await agent.aclose()


@pytest.mark.parametrize("method", ["handle", "handle_batch"])
async def test_the_turn_prompt_says_locals_last_one_call(tmp_path, method):
    from nooa import build_prompt_data
    from nooa.prompts import render_prompt_data

    agent = AtomAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        data = await build_prompt_data(getattr(agent, method), {"user_messages": ["hi"]})
        rendered = " ".join(render_prompt_data(data).split())
        assert (
            "Python locals live for one method call; when the call returns they are gone. "
            "Anything you need later goes in ``self.v`` (durable, snapshot-backed) or the todo "
            "list. Do not rely on a variable from an earlier call."
        ) in rendered
    finally:
        await agent.aclose()


async def test_repo_tools_take_a_cwd_for_one_call(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "x.py").write_text("def only_in_sub():\n    pass\n")
    (tmp_path / "caller.py").write_text("from sub.x import only_in_sub\nonly_in_sub()\n")
    agent = AtomAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        relative = await agent.repo.symbols("x.py", cwd="sub")
        assert relative.diagnostic is None and "only_in_sub" in str(relative)
        assert "sub/x.py" in str(relative)  # still shown relative to the root
        absolute = await agent.repo.symbols("x.py", cwd=str(sub))
        assert "only_in_sub" in str(absolute)
        assert agent.shell.cwd == tmp_path.resolve()  # the shell did not move
        assert agent.repo.cwd == tmp_path.resolve()
        missing = await agent.repo.symbols("x.py")
        assert missing.diagnostic is not None
        refs = await agent.repo.refs("only_in_sub", cwd=str(tmp_path))
        assert "caller.py" in str(refs)
        refs_sub = await agent.repo.refs("only_in_sub", ".", cwd="sub")
        assert "caller.py" not in str(refs_sub)
    finally:
        await agent.aclose()


def test_the_repo_tool_docs_offer_cwd():
    from nooa_atom.tools.repo_tools import RepoTools

    from nooa.agentdoc import doc

    rendered = str(doc(RepoTools, concise=True))
    assert rendered.count("cwd") >= 2
    assert "defaults to the shell's current directory" in rendered


async def test_replace_contract_survives_the_default_concise_tool_context(tmp_path):
    from nooa_atom.agent.activity import ActivityShellTools
    from nooa_atom.tools.repo_tools import RepoTools

    from nooa.agentdoc import doc
    from nooa.tools.todo import TodoManager

    # Construct only: strict exhaustion forbids even an accidental fake model call.
    agent = AtomAgent(llm=FakeLLMClient(strict_exhaustion=True), cwd=tmp_path)
    try:
        tools = doc(RepoTools, ActivityShellTools, TodoManager, concise=True)
        assert str(agent.context["python_cell_tools"]) == tools
        for rendered in (tools, doc(ActivityShellTools, concise=True)):
            assert "replace(match: Match, new_text: str)" in rendered
            assert "replace(file_path: str, old_text: str, new_text: str)" in rendered
            assert "line numbers are not handles" in rendered
            assert (
                "A Match replaces its entire line region, not a substring within it."
                not in rendered
            )
    finally:
        await agent.aclose()


def test_context_block_helpers_are_not_traced():
    """Evaluating a dynamic context block is prompt rendering, not agent work: no span."""
    assert getattr(AtomAgent._workspace_state_context, "_no_trace", False) is True


@pytest.mark.parametrize("method", ["handle", "handle_batch"])
def test_the_turn_methods_run_on_the_single_tool_strategy(method):
    from nooa.strategies import CodeActV2

    assert isinstance(getattr(AtomAgent, method)._plan_strategy, CodeActV2)
    assert not isinstance(AtomAgent.name_session._plan_strategy, CodeActV2)


async def _first_call_messages(tmp_path, method: str) -> list[dict]:
    """The messages of the first model call of a ``method`` turn."""
    from test_experimental_agent import python_cell

    code = (
        "return_result(Done(explanation='x', result=TaskResult("
        "solution_description='a', evidence='b', how_to_verify='c')))"
    )
    llm = FakeLLMClient([python_cell(code, "call_1")], strict_exhaustion=True)
    agent = AtomAgent(llm=llm, cwd=tmp_path)
    try:
        await getattr(agent, method)({"user_messages": ["hi"]})
    finally:
        await agent.aclose()
    return llm.calls[0].messages


@pytest.mark.parametrize("method", ["handle", "handle_batch"])
async def test_the_turn_prompt_has_cell_state_and_no_state_dump(tmp_path, method):
    messages = await _first_call_messages(tmp_path, method)
    rendered = "\n".join(str(m.get("content", "")) for m in messages)
    assert "<python_cell_state" in rendered
    assert "<state " not in rendered and "<state>" not in rendered
    # The stable agent keeps the context-usage block; it drives compaction.
    assert "<context_usage" in rendered


async def test_the_tool_docs_are_in_the_cached_prefix(tmp_path):
    """Stable tool docs sit in the leading system message, not the trailing context."""
    messages = await _first_call_messages(tmp_path, "handle")
    assert messages[0]["role"] == "system"
    assert "<python_cell_tools>" in str(messages[0]["content"])
    envelope = str(messages[-1]["content"])
    assert envelope.lstrip().startswith("<context>")
    assert "<python_cell_tools" not in envelope


class _CallRecorder:
    """Instrumentation hooks that record which agent methods open a span."""

    def __init__(self):
        self.methods: list[str] = []

    def before_agent_call(self, agent, method_name, *args, **kwargs):
        self.methods.append(method_name)
        return {}

    def __getattr__(self, name):
        return lambda *args, **kwargs: {}


def test_host_reads_record_no_span(tmp_path):
    """``plan()`` and ``get_summarization_status()`` are host reads, not agent work."""
    from nooa.runtime.hooks import get_hooks, set_hooks

    class ProbedAgent(AtomAgent):
        def probe(self) -> int:
            return 1

    agent = ProbedAgent(llm=FakeLLMClient(), cwd=tmp_path)
    recorder = _CallRecorder()
    previous = get_hooks()
    set_hooks(recorder)
    try:
        agent.probe()  # a traced method, so the recorder is known to work
        assert agent.plan() == []
        agent.get_summarization_status()
    finally:
        set_hooks(previous)
    assert recorder.methods == ["probe"]
