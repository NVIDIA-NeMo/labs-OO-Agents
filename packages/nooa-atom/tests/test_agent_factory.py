# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""create_session_agent() as the registry's default factory, and the model factory."""

import asyncio

import pytest
from coder_test_agents import CODER_SPEC, CellLLM, ModelFactory, cell
from nooa_coder.coding.agent import CodingAgent
from nooa_coder.coding.factory import create_session_agent, default_llm_factory
from nooa_coder.session.options import SessionOptions
from nooa_coder.session.registry import SessionRegistry
from nooa_coder.session.store import SessionStore

from nooa.interactive import Done
from nooa.storage import InMemoryStorageManager
from nooa.unifiedllm import FakeLLMClient

TIMEOUT = 30

CHILD_RESULT = (
    "return_result(Done(explanation='fixed', result=TaskResult("
    "solution_description='a', evidence='b', how_to_verify='c', report='child report')))"
)

_FORWARDING_AGENT = """\
from nooa_coder.coding.agent import CodingAgent


class ForwardingCoder(CodingAgent):
    '''The normal subclass-extension pattern: forwards **kwargs to super().'''

    def __init__(self, llm=None, storage=None, **kwargs):
        super().__init__(llm=llm, storage=storage, **kwargs)
"""

_UNRELATED_AGENT = """\
from nooa.interactive import InteractiveAgent


class Unrelated(InteractiveAgent):
    '''Declares **kwargs but understands none of the coding keywords.'''

    def __init__(self, llm=None, storage=None, **kwargs):
        if kwargs:
            raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")
        super().__init__(llm=llm, storage=storage)
"""


@pytest.fixture(autouse=True)
def _no_settings_env(monkeypatch):
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "workspace"
    path.mkdir()
    return path


def _options(workspace, sessions_dir, **values):
    return SessionOptions(
        workspace=workspace, agent_spec=CODER_SPEC, sessions_dir=sessions_dir, **values
    )


async def _close(agent):
    await agent.queue_manager.shutdown()
    await agent.aclose()


async def test_the_registry_builds_a_workspace_coding_agent_by_default(workspace, sessions_dir):
    llm = FakeLLMClient()
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=create_session_agent)
    try:
        root = await registry.create(_options(workspace, sessions_dir, llm=llm))
        agent = root._agent
        assert isinstance(agent, CodingAgent)
        assert agent.cwd == workspace.resolve()
        assert agent.llm is llm
        assert agent.libs._path == workspace / ".nooa" / "libs"
        # configure_session_skills ran: the MCP servers and workspace settings.
        assert "nooa.workspace_settings" in agent.skills.activated()
        assert agent.skills.mcp.project_dir == workspace.resolve() / ".nooa"
    finally:
        await registry.close_all()


async def test_the_mcp_and_skills_controls_are_installed_without_a_host(workspace, sessions_dir):
    """/mcp approve is what MCPApprovalRequired tells the user to run: it must exist."""
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=create_session_agent)
    try:
        root = await registry.create(_options(workspace, sessions_dir, llm=FakeLLMClient()))
        names = {command.name for command in root.commands()}
        assert {"mcp", "skills"} <= names
        result = await root._agent.slash_commands.invoke("mcp", "status")
        assert "MCP servers" in str(result.value)
    finally:
        await registry.close_all()


async def test_the_factory_installs_the_skills_and_mcp_controls(workspace):
    options = SessionOptions(workspace=workspace, agent_spec=CODER_SPEC, llm=FakeLLMClient())
    agent = create_session_agent(options, InMemoryStorageManager())
    try:
        assert {"mcp", "skills"} <= {c.name for c in agent.slash_commands.commands()}
    finally:
        await _close(agent)


async def test_a_settings_file_from_before_legacy_agent_was_removed_still_loads(workspace):
    from nooa_coder.workspace.options import CoderOptions

    (workspace / ".nooa").mkdir()
    (workspace / ".nooa" / "settings.yaml").write_text(
        "tui:\n  legacy_agent: true\ncoding:\n  legacy_agent: true\n  default_model: m\n"
    )
    options = CoderOptions.load(workspace)
    assert options.default_model == "m"
    assert not hasattr(options, "legacy_agent")


async def test_a_mistyped_setting_does_not_abort_session_creation(workspace, sessions_dir, caplog):
    settings = workspace / ".nooa" / "settings.yaml"
    settings.parent.mkdir()
    settings.write_text('coding:\n  active_skills: "just-one"\n')
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=create_session_agent)
    try:
        with caplog.at_level("WARNING"):
            root = await registry.create(_options(workspace, sessions_dir, llm=FakeLLMClient()))
        assert isinstance(root._agent, CodingAgent)
        assert root._agent.cwd == workspace.resolve()
        [warning] = [r.getMessage() for r in caplog.records if str(settings) in r.getMessage()]
        assert "active_skills" in warning
    finally:
        await registry.close_all()


async def test_workspace_settings_reach_the_agent(workspace):
    (workspace / ".nooa").mkdir()
    (workspace / ".nooa" / "settings.yaml").write_text(
        "coding:\n  summarization:\n    policy: none\n"
    )
    options = SessionOptions(workspace=workspace, agent_spec=CODER_SPEC, llm=FakeLLMClient())
    agent = create_session_agent(options, InMemoryStorageManager())
    try:
        assert agent.get_summarization_status()["policy"] == "none"
    finally:
        await _close(agent)


async def test_a_kwargs_forwarding_subclass_gets_the_real_workspace(workspace):
    """A CodingAgent subclass using **kwargs still gets cwd and libs_dir.

    A literal-name-only check would drop them and fall back to cwd='.',
    the process's own directory. (From coder/3-engine's test_coding_factory.)
    """
    (workspace / "forwarding.py").write_text(_FORWARDING_AGENT)
    options = SessionOptions(
        workspace=workspace, agent_spec="./forwarding.py:ForwardingCoder", llm=FakeLLMClient()
    )
    agent = create_session_agent(options, InMemoryStorageManager())
    try:
        assert type(agent).__name__ == "ForwardingCoder"
        assert agent.cwd == workspace.resolve()
        assert agent.libs._path == workspace / ".nooa" / "libs"
    finally:
        await _close(agent)


_NARROW_AGENT = """\
from nooa_coder.coding.agent import CodingAgent


class NarrowCoder(CodingAgent):
    '''Overrides __init__ with explicit keywords and no **kwargs.'''

    def __init__(self, llm=None, *, storage=None):
        super().__init__(llm=llm, storage=storage)
"""

_OWN_CWD_AGENT = """\
from nooa.interactive import InteractiveAgent


class OwnCwd(InteractiveAgent):
    '''An unrelated agent with its own cwd and summarization parameters.'''

    def __init__(self, llm=None, storage=None, cwd="/opt/mydata", summarization="off"):
        assert isinstance(summarization, str)
        super().__init__(llm=llm, storage=storage)
        self.my_cwd = cwd
"""


async def test_a_narrow_subclass_keeps_the_workspace_wiring_and_is_warned(workspace, caplog):
    (workspace / "narrow.py").write_text(_NARROW_AGENT)
    options = SessionOptions(
        workspace=workspace, agent_spec="./narrow.py:NarrowCoder", llm=FakeLLMClient()
    )
    with caplog.at_level("WARNING", logger="nooa_coder.coding.factory"):
        agent = create_session_agent(options, InMemoryStorageManager())
    try:
        assert "nooa.workspace_settings" in agent.skills.activated()
        assert agent.skills.mcp is not None
        assert "mcp" in {c.name for c in agent.slash_commands.commands()}
        warnings = [r.getMessage() for r in caplog.records if "NarrowCoder" in r.getMessage()]
        assert len(warnings) == 1
        assert "cwd" in warnings[0] and "libs_dir" in warnings[0]
    finally:
        await _close(agent)
    # Once per class: a second session does not repeat it.
    caplog.clear()
    with caplog.at_level("WARNING", logger="nooa_coder.coding.factory"):
        agent = create_session_agent(options, InMemoryStorageManager())
    try:
        assert not [r for r in caplog.records if "NarrowCoder" in r.getMessage()]
    finally:
        await _close(agent)


def test_an_unrelated_agent_keeps_its_own_cwd_and_summarization(workspace):
    (workspace / "own.py").write_text(_OWN_CWD_AGENT)
    options = SessionOptions(workspace=workspace, agent_spec="./own.py:OwnCwd", llm=FakeLLMClient())
    agent = create_session_agent(options, InMemoryStorageManager())
    assert agent.my_cwd == "/opt/mydata"


def test_kwargs_are_not_forced_on_an_unrelated_agent(workspace):
    (workspace / "unrelated.py").write_text(_UNRELATED_AGENT)
    llm = FakeLLMClient()
    options = SessionOptions(workspace=workspace, agent_spec="./unrelated.py:Unrelated", llm=llm)
    agent = create_session_agent(options, InMemoryStorageManager())
    assert type(agent).__name__ == "Unrelated"
    assert agent.llm is llm


@pytest.mark.parametrize(
    "spec",
    [
        "nooa_cli.tui.agent:TUIAgent",
        "nooa_cli.coding.legacy_agent:TUIAgent",
        "nooa_cli.coding.agent:CodingAgent",
    ],
)
def test_legacy_coding_agent_specs_load_the_moved_class(spec):
    """Saved specs from nooa_cli keep loading after the move to nooa_coder."""
    from nooa_coder.session.loader import load_agent_class

    assert load_agent_class(spec) is CodingAgent


async def test_a_child_with_another_model_gets_its_own_client(workspace, sessions_dir):
    models = ModelFactory({"other": [[cell(CHILD_RESULT)]]})
    registry = SessionRegistry(
        SessionStore(sessions_dir), agent_factory=create_session_agent, llm_factory=models
    )
    parent_llm = CellLLM(
        [
            cell(
                "done = await self.delegate('Other', 'use the other model', model='other')\n"
                "return_result(Done(explanation=done.result.report))"
            )
        ]
    )
    try:
        root = await registry.create(_options(workspace, sessions_dir, llm=parent_llm))
        outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
        assert outcome == Done(explanation="child report")
        assert [alias for alias, _ in models.calls] == ["other"]
        [child] = registry.children(root.id)
        assert child.model == "other"

        # The child's session created that client, so it closes it; the
        # throwaway child is closed in the background after its result.
        async def closed():
            while not models.made[0].closed:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(closed(), TIMEOUT)
    finally:
        await registry.close_all()


async def test_stale_memory_context_is_dropped_after_a_reload(workspace, sessions_dir):
    """Keys an older snapshot carries are gone after load (restore is additive)."""
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=create_session_agent)
    try:
        root = await registry.create(_options(workspace, sessions_dir, llm=FakeLLMClient()))
        session_id = root.id
        root._agent.context["memory_system"] = "stale memory prompt"
        root._agent.context["recalled_memories"] = "stale recall"
        root._checkpoint()
        await root.wait_for_checkpoint()
        await registry.close(session_id)
        loaded = await registry.load(session_id, llm=FakeLLMClient())
        assert "memory_system" not in loaded._agent.context
        assert "recalled_memories" not in loaded._agent.context
    finally:
        await registry.close_all()


async def test_skills_are_configured_before_and_stale_context_dropped_after_a_restore(
    workspace, sessions_dir, monkeypatch
):
    """Skills are configured before the snapshot is restored; the stale-context
    cleanup runs after it, since restoring is additive and would re-add the keys."""
    import nooa_coder.coding.factory as factory
    import nooa_coder.workspace.options as coder_options

    from nooa.storage.sqlite import SQLiteStorageManager

    order: list[str] = []
    configure = factory.configure_session_skills
    restore = SQLiteStorageManager.restore_latest_snapshot
    cleanup = coder_options.drop_stale_memory_context

    def spy_configure(agent, options):
        order.append("configure")
        return configure(agent, options)

    def spy_restore(self, agent):
        order.append("restore")
        return restore(self, agent)

    def spy_cleanup(agent):
        order.append("cleanup")
        return cleanup(agent)

    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=create_session_agent)
    try:
        root = await registry.create(_options(workspace, sessions_dir, llm=FakeLLMClient()))
        session_id = root.id
        root._checkpoint()
        await root.wait_for_checkpoint()
        await registry.close(session_id)
        monkeypatch.setattr(factory, "configure_session_skills", spy_configure)
        monkeypatch.setattr(SQLiteStorageManager, "restore_latest_snapshot", spy_restore)
        monkeypatch.setattr(coder_options, "drop_stale_memory_context", spy_cleanup)
        await registry.load(session_id, llm=FakeLLMClient())
        assert order == ["configure", "restore", "cleanup"]
    finally:
        await registry.close_all()


def test_the_default_llm_factory_uses_the_workspace_default_model(workspace, monkeypatch):
    import nooa_coder.coding.factory as factory

    (workspace / ".nooa").mkdir()
    (workspace / ".nooa" / "settings.yaml").write_text("coding:\n  default_model: ws-model\n")
    built: list[str] = []
    monkeypatch.setattr(
        factory, "workspace_llm_client", lambda alias, workspace: built.append(alias) or alias
    )
    make = default_llm_factory()
    assert make(None, workspace) == "ws-model"
    assert make("named", workspace) == "named"
    assert default_llm_factory(workspace_default="fixed")(None, workspace) == "fixed"
    assert built == ["ws-model", "named", "fixed"]


async def test_a_host_registry_builds_the_workspace_default_model(
    workspace, sessions_dir, monkeypatch
):
    """default_llm_factory() as the registry's llm_factory, for a session without a model."""
    import nooa_coder.coding.factory as factory
    from coder_test_agents import TrackedLLM

    (workspace / ".nooa").mkdir()
    (workspace / ".nooa" / "settings.yaml").write_text("coding:\n  default_model: ws-model\n")
    built: list[TrackedLLM] = []

    def fake_client(alias, workspace):
        built.append(TrackedLLM(alias, []))
        return built[-1]

    monkeypatch.setattr(factory, "workspace_llm_client", fake_client)
    make = default_llm_factory()
    aliases: list[str | None] = []

    def spy(alias, path):
        aliases.append(alias)
        return make(alias, path)

    registry = SessionRegistry(
        SessionStore(sessions_dir), agent_factory=create_session_agent, llm_factory=spy
    )
    try:
        root = await registry.create(_options(workspace, sessions_dir))
        assert aliases == [None]
        [client] = built
        assert client.alias == "ws-model"
        assert root._agent.llm is client
        assert root.info.model == "ws-model"
        await registry.close(root.id)
        assert client.closed is True
    finally:
        await registry.close_all()
