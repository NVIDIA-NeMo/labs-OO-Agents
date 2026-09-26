# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Session's data API for hosts: channels, model info, reasoning, commands changed.

A host (the ACP adapter) reads and changes these through the Session and
never holds the agent.
"""

import pytest
from coder_test_agents import LeveledModelFactory, plain_agent_factory
from nooa_coder.session.items import CommandsChangedUpdate, ModelInfo, ReasoningChangedUpdate
from nooa_coder.session.registry import SessionRegistry
from nooa_coder.session.store import SessionStore

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
    assert channels == list(session.agent.queue_manager.channels())


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
    session, _ = make_session(agent_spec="coder_test_agents:CommandAgent", start=False)
    seen = []
    session.subscribe(seen.append)
    session.agent.slash_commands.add("review", "Review the diff")
    [update] = [u for u in seen if isinstance(u, CommandsChangedUpdate)]
    assert update.session_id == session.id
    assert [command.name for command in update.commands] == ["model", "clear", "review"]
    assert update.commands == session.commands()
