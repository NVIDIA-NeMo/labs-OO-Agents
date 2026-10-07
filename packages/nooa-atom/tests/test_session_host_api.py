# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Session's data API for hosts: channels, model info, reasoning, commands changed.

A host (the ACP adapter) reads and changes these through the Session and
never holds the agent.
"""

import pytest
from atom_test_agents import LeveledModelFactory, plain_agent_factory
from nooa_atom.session.items import CommandsChangedUpdate, ModelInfo, ReasoningChangedUpdate
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa.unifiedllm.reasoning import ReasoningConfig


def _declare_levels(llm):
    llm._reasoning_config = ReasoningConfig(
        levels={"low": {"reasoning_effort": "low"}, "high": {"reasoning_effort": "high"}},
        default="low",
    )


async def test_channels_names_the_agents_queue_channels(make_session):
    session, _ = make_session(start=False)
    channels = session.channels()
    assert "user_messages" in channels
    assert channels == list(session._agent.queue_manager.channels())


async def test_model_info_reads_the_client(make_session):
    session, llm = make_session(start=False)
    _declare_levels(llm)
    session.info.model = "alias-a"
    assert session.model_info() == ModelInfo(
        alias="alias-a",
        context_window=llm.context_window,
        reasoning_level=None,
        reasoning_levels=["low", "high"],
        reasoning_default="low",
    )


async def test_model_info_without_declared_levels(make_session):
    session, _ = make_session(start=False)
    info = session.model_info()
    assert info.reasoning_levels == []
    assert info.reasoning_default is None


async def test_set_reasoning_applies_records_and_says_so(make_session, sessions_dir):
    session, llm = make_session(start=False)
    _declare_levels(llm)
    seen = []
    session.subscribe(seen.append)
    await session.set_reasoning("high")
    assert llm.reasoning_level == "high"
    assert session.model_info().reasoning_level == "high"
    assert session.info.reasoning == "high"
    assert [u for u in seen if isinstance(u, ReasoningChangedUpdate)] == [
        ReasoningChangedUpdate(session_id=session.id, level="high")
    ]
    assert SessionStore(sessions_dir).get(session.id).reasoning == "high"


async def test_set_reasoning_rejects_undeclared_levels(make_session):
    session, llm = make_session(start=False)
    with pytest.raises(ValueError, match="none for this model"):
        await session.set_reasoning("high")
    _declare_levels(llm)
    with pytest.raises(ValueError, match="low, high"):
        await session.set_reasoning("max")
    assert llm.reasoning_level is None


async def test_a_load_restores_the_reasoning_level_and_a_model_switch_resets_it(
    root_options, sessions_dir
):
    factory = LeveledModelFactory()
    registry = SessionRegistry(
        SessionStore(sessions_dir), agent_factory=plain_agent_factory, llm_factory=factory
    )
    try:
        root = await registry.create(root_options.model_copy(update={"model": "alias-a"}))
        await root.set_reasoning("high")
        session_id = root.id
        await registry.close(session_id)
        loaded = await registry.load(session_id)
        assert loaded.model_info().reasoning_level == "high"
        assert factory.made[-1].reasoning_level == "high"

        # A new model starts from its own default; the level is not carried over.
        await loaded.set_model("alias-b")
        assert loaded.info.reasoning is None
        assert loaded.model_info().reasoning_level is None
        await loaded.set_reasoning("high")  # applies to the client the next turn uses
        assert factory.made[-1].reasoning_level == "high"
        await registry.close(session_id)
        assert registry.store.get(session_id).reasoning == "high"
    finally:
        await registry.close_all()


async def test_a_model_switch_is_recorded_as_resetting_the_level(root_options, sessions_dir):
    registry = SessionRegistry(
        SessionStore(sessions_dir),
        agent_factory=plain_agent_factory,
        llm_factory=LeveledModelFactory(),
    )
    try:
        root = await registry.create(root_options.model_copy(update={"model": "alias-a"}))
        await root.set_reasoning("high")
        await root.set_model("alias-b")
        assert registry.store.get(root.id).reasoning is None
    finally:
        await registry.close_all()


async def test_commands_changed_is_emitted_when_the_registry_changes(make_session):
    session, _ = make_session(agent_spec="atom_test_agents:CommandAgent", start=False)
    seen = []
    session.subscribe(seen.append)
    session._agent.slash_commands.add("review", "Review the diff")
    [update] = [u for u in seen if isinstance(u, CommandsChangedUpdate)]
    assert update.session_id == session.id
    assert [command.name for command in update.commands] == ["model", "clear", "review"]
    assert update.commands == session.commands()


# ---- the agent's events, as session updates ---------------------------------


async def test_agent_events_reach_subscribers_as_agent_event_updates(make_session):
    from nooa_atom.agent.activity import TerminalCommandStarted
    from nooa_atom.session.items import AgentEventUpdate

    from nooa.interactive import AgentMessage

    session, _ = make_session(start=False)
    seen = []
    session.subscribe(seen.append)
    message = AgentMessage(content="hello")
    session._agent.event_manager.add(message)
    # Runtime events (never recorded for the model) are forwarded too.
    started = TerminalCommandStarted(command_id="c1", command="ls", working_directory="/")
    session._agent.event_manager.add(started)
    updates = [u for u in seen if isinstance(u, AgentEventUpdate)]
    assert [u.event for u in updates] == [message, started]
    assert updates[0].event is message  # the event itself, not a copy
    assert all(u.session_id == session.id for u in updates)
    assert updates[0].model_dump(mode="json")["event"]["content"] == "hello"


async def test_a_model_response_arrives_before_the_usage_it_changes(make_session):
    from nooa_atom.session.items import AgentEventUpdate, UsageChangedUpdate

    from nooa.events import LLMResponse
    from nooa.llm_types import LLMUsage

    session, _ = make_session(start=False)
    seen = []
    session.subscribe(seen.append)
    session._agent.event_manager.add(LLMResponse(usage=LLMUsage(input_tokens=10)))
    kinds = [type(u) for u in seen]
    assert kinds == [AgentEventUpdate, UsageChangedUpdate]


# ---- host status ----------------------------------------------------------------


async def test_the_plan_is_empty_for_an_agent_without_one(make_session):
    session, _ = make_session(start=False)
    assert session.plan() == []


async def test_the_plan_is_the_agents_as_entries(make_session):
    from nooa_atom.session.items import PlanEntry

    session, _ = make_session(agent_spec="atom_test_agents:PlanAgent", start=False)
    assert session.plan() == [
        PlanEntry(content="write the test", status="in_progress"),
        PlanEntry(content="run it"),
    ]


async def test_a_failing_plan_is_empty(make_session, caplog):
    session, _ = make_session(agent_spec="atom_test_agents:BrokenPlanAgent", start=False)
    assert session.plan() == []
    assert "plan" in caplog.text


async def test_the_atom_agent_plans_from_its_todos(tmp_path):
    from nooa_atom.agent.agent import AtomAgent
    from nooa_atom.session.items import PlanEntry

    from nooa.unifiedllm import FakeLLMClient

    agent = AtomAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        first = agent.todo.add("Write the test")
        second = agent.todo.add("Make it pass")
        agent.todo.activate(first.id)
        agent.todo.complete(second.id)
        assert agent.plan() == [
            PlanEntry(content="Write the test", status="in_progress"),
            PlanEntry(content="Make it pass", status="completed"),
        ]
    finally:
        await agent.aclose()


async def test_prepare_tools_runs_the_agents_hook_and_returns_its_warnings(make_session):
    session, _ = make_session(agent_spec="atom_test_agents:ToolPrepAgent", start=False)
    assert await session.prepare_tools() == ["server 'x' was not connected"]
    assert session._agent.prepared == 1


async def test_prepare_tools_without_a_hook_does_nothing(make_session):
    session, _ = make_session(start=False)
    assert await session.prepare_tools() == []


class _Tool:
    """A tool a host hands to the agent."""

    def ping(self) -> str:
        return "pong"


async def test_register_tools_registers_and_activates_each_tool(make_session):
    session, _ = make_session(agent_spec="nooa_atom.agent:AtomAgent", start=False)
    tool = _Tool()
    assert await session.register_tools({"mcp.remote": tool, "repo": _Tool()}) == {
        "repo": "Cannot register skill 'repo' as agent attr 'repo': already provided by 'nemo.repo'"
    }
    assert "mcp.remote" in session._agent.skills.activated()
    assert session._agent.skills["mcp.remote"] is tool


async def test_register_tools_on_an_agent_without_skills(make_session):
    session, _ = make_session(start=False)
    assert await session.register_tools({"mcp.remote": _Tool()}) == {
        "mcp.remote": "the agent has no skills"
    }


async def test_the_atom_agent_connects_the_servers_its_workspace_remembers(
    root_options, sessions_dir, tmp_path_factory, monkeypatch
):
    import yaml
    from nooa_atom.agent.factory import create_session_agent

    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path_factory.mktemp("user-config")))
    workspace = root_options.workspace
    (workspace / ".nooa").mkdir(exist_ok=True)
    (workspace / ".nooa" / "settings.yaml").write_text(
        yaml.safe_dump({"atom": {"mcp_auto_connect": ["docs", "nowhere"]}})
    )
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=create_session_agent)
    try:
        options = root_options.model_copy(
            update={"agent_spec": "nooa_atom.agent:AtomAgent", "llm": _fake_llm()}
        )
        session = await registry.create(options)
        asked = []

        async def connect(names):
            asked.extend(names)
            if names == ["nowhere"]:
                raise RuntimeError("not approved")
            return names

        session._agent.skills.mcp.connect = connect
        assert await session.prepare_tools() == [
            "MCP server 'nowhere' was not connected: not approved"
        ]
        assert asked == ["docs", "nowhere"]
    finally:
        await registry.close_all()


def _fake_llm():
    from nooa.unifiedllm import FakeLLMClient

    return FakeLLMClient()
