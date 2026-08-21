# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared coding-agent construction and repository instructions."""

import asyncio
from types import SimpleNamespace

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
            "objective": "review parser",
            "closed": True,
        }
    finally:
        await agent.close()


async def test_coding_agent_spawns_delegation_in_background(tmp_path):
    """Background delegation is the preferred path and reports through a queue."""
    started = asyncio.Event()
    release = asyncio.Event()

    class TestAgent(CodingAgent):
        async def delegate(self, objective: str, supplied_context=None) -> str:
            assert objective == "review parser"
            assert supplied_context == {"path": "parser.py"}
            started.set()
            await release.wait()
            return "review complete"

    agent = TestAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        handle = agent.spawn("review parser", {"path": "parser.py"})
        assert handle.label == "review parser"
        assert handle.state == "running"
        await started.wait()
        assert agent.delegates.status() == ""

        release.set()
        assert await agent.delegates.get() == {
            "objective": "review parser",
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
