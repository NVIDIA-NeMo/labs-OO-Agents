# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared coding-agent construction and repository instructions."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from nooa_cli.coding import (
    CodingAgent,
    discover_agent_instruction_files,
    resolve_developer_overlay,
    resolve_instruction_profile,
    resolve_instruction_stack,
)

from nooa.skill import Skill, get_slash_commands, slash_command
from nooa.unifiedllm import FakeLLMClient

@pytest.fixture(autouse=True)
def isolated_instruction_settings(tmp_path, monkeypatch):
    """Keep instruction tests independent of developer user settings."""
    user_config = tmp_path / "user-config"
    user_config.mkdir()
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_config))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    monkeypatch.delenv("NEMO_OO_DEVELOPER_INSTRUCTIONS", raising=False)
    return user_config


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


def test_instruction_profile_prefers_exact_model_over_family_glob(tmp_path):
    (tmp_path / ".git").mkdir()
    config_dir = tmp_path / ".nooa"
    overlays = tmp_path / "overlays"
    config_dir.mkdir()
    overlays.mkdir()
    family = overlays / "family.md"
    exact = overlays / "exact.md"
    family.write_text("family guidance")
    exact.write_text("exact guidance")
    (config_dir / "settings.yaml").write_text(
        "instructions:\n"
        "  models:\n"
        '    "openai/gpt-6-*":\n'
        "      - overlays/family.md\n"
        '    "openai/gpt-6-astra":\n'
        "      - overlays/exact.md\n"
    )

    resolved = resolve_instruction_profile(tmp_path, model="openai/gpt-6-astra")

    assert resolved.name == "model:openai/gpt-6-astra"
    assert resolved.model_pattern == "openai/gpt-6-astra"
    assert resolved.files == (exact,)


def test_instruction_profile_uses_most_specific_matching_glob(tmp_path):
    (tmp_path / ".git").mkdir()
    config_dir = tmp_path / ".nooa"
    config_dir.mkdir()
    broad = tmp_path / "broad.md"
    specific = tmp_path / "specific.md"
    broad.write_text("broad")
    specific.write_text("specific")
    (config_dir / "settings.yaml").write_text(
        "instructions:\n"
        "  models:\n"
        '    "openai/*": [broad.md]\n'
        '    "openai/gpt-5.6-*": [specific.md]\n'
    )

    resolved = resolve_instruction_profile(tmp_path, model="openai/gpt-5.6-sol")

    assert resolved.model_pattern == "openai/gpt-5.6-*"
    assert resolved.files == (specific,)


def test_explicit_named_profile_overrides_model_match(tmp_path):
    (tmp_path / ".git").mkdir()
    config_dir = tmp_path / ".nooa"
    config_dir.mkdir()
    model_overlay = tmp_path / "model.md"
    explicit_overlay = tmp_path / "explicit.md"
    model_overlay.write_text("model")
    explicit_overlay.write_text("explicit")
    (config_dir / "settings.yaml").write_text(
        "instructions:\n"
        "  profiles:\n"
        "    review: [explicit.md]\n"
        "  models:\n"
        '    "openai/*": [model.md]\n'
    )

    resolved = resolve_instruction_profile(
        tmp_path,
        model="openai/gpt-6-astra",
        explicit_profile="review",
    )

    assert resolved.name == "review"
    assert resolved.selection == "explicit"
    assert resolved.model_pattern is None
    assert resolved.files == (explicit_overlay,)


def test_model_mapping_can_reference_named_profile(tmp_path):
    (tmp_path / ".git").mkdir()
    config_dir = tmp_path / ".nooa"
    config_dir.mkdir()
    overlay = tmp_path / "astra.md"
    overlay.write_text("astra")
    (config_dir / "settings.yaml").write_text(
        "instructions:\n"
        "  profiles:\n"
        "    astra: [astra.md]\n"
        "  models:\n"
        '    "openai/gpt-6-astra": astra\n'
    )

    resolved = resolve_instruction_profile(tmp_path, model="openai/gpt-6-astra")

    assert resolved.name == "astra"
    assert resolved.selection == "model"
    assert resolved.files == (overlay,)


def test_instruction_stack_is_inspectable_in_effective_precedence_order(tmp_path):
    (tmp_path / ".git").mkdir()
    nested = tmp_path / "package"
    config_dir = tmp_path / ".nooa"
    nested.mkdir()
    config_dir.mkdir()
    root = tmp_path / "AGENTS.md"
    local = nested / "AGENTS.md"
    overlay = tmp_path / "overlay.md"
    root.write_text("root")
    local.write_text("local")
    overlay.write_text("overlay")
    (config_dir / "settings.yaml").write_text(
        'instructions:\n  models:\n    "test/*": [overlay.md]\n'
    )

    stack = resolve_instruction_stack(nested, model="test/model")

    assert stack.files == (root, local, overlay)
    assert "matched model pattern: test/*" in stack.format_debug()
    assert stack.format_debug().index(str(local)) < stack.format_debug().index(str(overlay))


def test_private_developer_overlay_is_additive_and_repository_scoped(
    tmp_path,
    isolated_instruction_settings,
):
    (tmp_path / ".git").mkdir()
    nested = tmp_path / "package"
    nested.mkdir()
    repository_instructions = tmp_path / "AGENTS.md"
    developer_instructions = isolated_instruction_settings / "nooa.md"
    developer_config = isolated_instruction_settings / "developer-instructions.yaml"
    repository_instructions.write_text("shared repository rule")
    developer_instructions.write_text("create a Linear ticket")
    developer_config.write_text(f'repositories:\n  "{tmp_path}":\n    - nooa.md\n')

    stack = resolve_instruction_stack(nested, model="test/model")

    assert stack.files == (repository_instructions, developer_instructions)
    assert stack.developer.selection == "repository"
    assert stack.developer.repository == tmp_path
    assert stack.developer.repository_pattern == str(tmp_path)
    debug = stack.format_debug()
    assert debug.index(str(repository_instructions)) < debug.index(str(developer_instructions))
    assert f"config: {developer_config}" in debug


def test_developer_overlay_uses_most_specific_repository_glob(tmp_path):
    workspace = tmp_path / "worktrees" / "123" / "project"
    workspace.mkdir(parents=True)
    (workspace / ".git").mkdir()
    private = tmp_path / "private"
    private.mkdir()
    broad = private / "broad.md"
    specific = private / "specific.md"
    config = private / "developer-instructions.yaml"
    broad.write_text("broad workflow")
    specific.write_text("specific workflow")
    config.write_text(
        "repositories:\n"
        f'  "{tmp_path}/worktrees/*/*": [broad.md]\n'
        f'  "{tmp_path}/worktrees/*/project": [specific.md]\n'
    )

    resolved = resolve_developer_overlay(workspace, config_file=config)

    assert resolved.repository_pattern == f"{tmp_path}/worktrees/*/project"
    assert resolved.files == (specific,)


def test_different_private_configs_can_overlay_the_same_repository(tmp_path):
    (tmp_path / ".git").mkdir()
    first_dir = tmp_path / "first-developer"
    second_dir = tmp_path / "second-developer"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "workflow.md").write_text("first workflow")
    (second_dir / "workflow.md").write_text("second workflow")
    first_config = first_dir / "developer-instructions.yaml"
    second_config = second_dir / "developer-instructions.yaml"
    first_config.write_text(f'repositories:\n  "{tmp_path}": [workflow.md]\n')
    second_config.write_text(f'repositories:\n  "{tmp_path}": [workflow.md]\n')

    first = resolve_developer_overlay(tmp_path, config_file=first_config)
    second = resolve_developer_overlay(tmp_path, config_file=second_config)

    assert first.files == (first_dir / "workflow.md",)
    assert second.files == (second_dir / "workflow.md",)
    assert not (tmp_path / "developer-instructions.yaml").exists()


def test_repository_cannot_configure_the_private_developer_layer(tmp_path):
    (tmp_path / ".git").mkdir()
    project_config = tmp_path / ".nooa"
    project_config.mkdir()
    private = tmp_path / "private.md"
    private.write_text("private workflow")
    (project_config / "developer-instructions.yaml").write_text(
        f'repositories:\n  "{tmp_path}": [../private.md]\n'
    )

    resolved = resolve_developer_overlay(tmp_path)

    assert resolved.selection == "none"
    assert resolved.files == ()
    assert resolved.config_file is None


def test_explicit_missing_developer_instruction_file_fails_loudly(tmp_path):
    (tmp_path / ".git").mkdir()
    config = tmp_path / "developer-instructions.yaml"
    config.write_text(f'repositories:\n  "{tmp_path}": [missing.md]\n')

    with pytest.raises(ValueError, match="missing.md.*does not exist"):
        resolve_developer_overlay(tmp_path, config_file=config)


def test_developer_repository_selector_must_be_absolute(tmp_path):
    config = tmp_path / "developer-instructions.yaml"
    config.write_text('repositories:\n  "relative/repository": [workflow.md]\n')

    with pytest.raises(ValueError, match="must be an absolute path"):
        resolve_developer_overlay(tmp_path, config_file=config)


def test_no_model_instruction_config_preserves_empty_overlay(tmp_path):
    (tmp_path / ".git").mkdir()

    resolved = resolve_instruction_profile(tmp_path, model="openai/gpt-6-astra")

    assert resolved.name is None
    assert resolved.files == ()
    assert resolved.selection == "none"


def test_configured_missing_instruction_file_fails_loudly(tmp_path):
    (tmp_path / ".git").mkdir()
    config_dir = tmp_path / ".nooa"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text(
        'instructions:\n  models:\n    "openai/*": [missing.md]\n'
    )

    with pytest.raises(ValueError, match="missing.md.*does not exist"):
        resolve_instruction_profile(tmp_path, model="openai/gpt-6-astra")


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


async def test_coding_agent_composes_repository_then_model_overlay(tmp_path):
    (tmp_path / ".git").mkdir()
    config_dir = tmp_path / ".nooa"
    config_dir.mkdir()
    (tmp_path / "AGENTS.md").write_text("repository rule")
    (tmp_path / "astra.md").write_text("astra-specific guidance")
    (config_dir / "settings.yaml").write_text(
        'instructions:\n  models:\n    "openai/gpt-6-astra": [astra.md]\n'
    )
    llm = FakeLLMClient()
    llm.model = "openai/gpt-6-astra"

    agent = CodingAgent(llm=llm, cwd=tmp_path)
    try:
        keys = list(agent.context)
        assert keys.index("repository_instructions") < keys.index("instruction_profile")
        assert "repository rule" in str(agent.context["repository_instructions"])
        assert "astra-specific guidance" in str(agent.context["instruction_profile"])
        assert agent.instruction_stack.profile.name == "model:openai/gpt-6-astra"

        messages = await agent.runtime._build_messages(agent.handle)
        system_prompt = "\n".join(
            str(message["content"]) for message in messages if message["role"] == "system"
        )
        assert system_prompt.index("repository rule") < system_prompt.index(
            "astra-specific guidance"
        )
    finally:
        await agent.close()


async def test_coding_agent_composes_developer_overlay_before_model_profile(tmp_path):
    (tmp_path / ".git").mkdir()
    project_config = tmp_path / ".nooa"
    private_config = tmp_path / "private"
    project_config.mkdir()
    private_config.mkdir()
    (tmp_path / "AGENTS.md").write_text("repository rule")
    (private_config / "workflow.md").write_text("developer workflow")
    (tmp_path / "model.md").write_text("model guidance")
    developer_config = private_config / "developer-instructions.yaml"
    developer_config.write_text(f'repositories:\n  "{tmp_path}": [workflow.md]\n')
    (project_config / "settings.yaml").write_text(
        'instructions:\n  models:\n    "test/*": [model.md]\n'
    )
    first_llm = FakeLLMClient()
    first_llm.model = "test/first"
    second_llm = FakeLLMClient()
    second_llm.model = "test/second"

    agent = CodingAgent(
        llm=first_llm,
        cwd=tmp_path,
        developer_instruction_config=developer_config,
    )
    try:
        keys = list(agent.context)
        assert keys.index("repository_instructions") < keys.index("developer_instructions")
        assert keys.index("developer_instructions") < keys.index("instruction_profile")
        assert "developer workflow" in str(agent.context["developer_instructions"])

        messages = await agent.runtime._build_messages(agent.handle)
        system_prompt = "\n".join(
            str(message["content"]) for message in messages if message["role"] == "system"
        )
        assert system_prompt.index("repository rule") < system_prompt.index("developer workflow")
        assert system_prompt.index("developer workflow") < system_prompt.index("model guidance")

        agent.set_llm(second_llm)
        assert agent.instruction_stack.developer.files == (private_config / "workflow.md",)
        assert "developer workflow" in str(agent.context["developer_instructions"])
    finally:
        await agent.close()


async def test_coding_agent_refreshes_overlay_when_host_switches_models(tmp_path):
    (tmp_path / ".git").mkdir()
    config_dir = tmp_path / ".nooa"
    config_dir.mkdir()
    (tmp_path / "sol.md").write_text("sol guidance")
    (tmp_path / "astra.md").write_text("astra guidance")
    (config_dir / "settings.yaml").write_text(
        "instructions:\n"
        "  profiles:\n"
        "    detailed: [sol.md]\n"
        "  models:\n"
        '    "openai/gpt-5.6-*": [sol.md]\n'
        '    "openai/gpt-6-astra": [astra.md]\n'
    )
    sol = FakeLLMClient()
    sol.model = "openai/gpt-5.6-sol"
    astra = FakeLLMClient()
    astra.model = "openai/gpt-6-astra"
    agent = CodingAgent(llm=sol, cwd=tmp_path)
    try:
        assert "sol guidance" in str(agent.context["instruction_profile"])

        agent.set_llm(astra)
        assert "astra guidance" in str(agent.context["instruction_profile"])

        agent.set_instruction_profile("detailed")
        assert agent.instruction_stack.profile.selection == "explicit"
        assert "sol guidance" in str(agent.context["instruction_profile"])

        agent.set_instruction_profile(None)
        assert "astra guidance" in str(agent.context["instruction_profile"])
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
