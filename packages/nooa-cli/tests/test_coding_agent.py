# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared coding-agent construction and repository instructions."""

from types import SimpleNamespace

import pytest
from nooa_cli.coding import CodingAgent, discover_agent_instruction_files

from nooa.skill import Skill, get_slash_commands, slash_command
from nooa.unifiedllm import FakeLLMClient


async def test_close_awaits_background_components_before_closing_shared_client(tmp_path):
    from unittest.mock import AsyncMock

    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    calls = []
    agent.event_manager.on_close(AsyncMock(side_effect=lambda: calls.append("component")))
    agent.llm.aclose = AsyncMock(side_effect=lambda: calls.append("client"))
    await agent.close()
    assert calls == ["component", "client"]


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
    """The real reader asks the stream for at most budget + 1 characters."""
    from nooa_cli.coding import instructions

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
    from nooa_cli.coding.instructions import render_agent_instructions

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


async def test_coding_agent_owns_bounded_application_state_context(tmp_path):
    from nooa_cli.coding.experimental_agent import ExperimentalCodingAgent

    from nooa.unifiedllm import FakeLLMClient

    agent = ExperimentalCodingAgent(cwd=tmp_path, llm=FakeLLMClient())
    try:
        agent.vars["token"] = "private-value"
        agent.shell.cwd = "</coding_state>\n" + "x" * 500
        rendered = agent._coding_state_context()
        assert "1 persistent vars" in rendered
        assert "print(self.v.items())" in rendered
        assert "private-value" not in rendered
        assert "</coding_state>" not in rendered
        assert len(rendered) < 600
    finally:
        await agent.close()


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
