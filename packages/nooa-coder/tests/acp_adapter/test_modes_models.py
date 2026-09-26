# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Permission modes and the model select option over ACP."""

import asyncio

import pytest
from acp import RequestError, text_block
from acp.schema import AgentMessageChunk, CurrentModeUpdate
from coder_test_agents import ModelFactory, ScriptedModels, reply

TIMEOUT = 30


@pytest.fixture(autouse=True)
def aliases(monkeypatch):
    from nooa_coder.acp import server

    monkeypatch.setattr(server, "model_aliases", lambda: ["fast", "smart"])


async def test_set_mode_auto_is_accepted_and_reported(make_adapter, workspace, client):
    adapter = await make_adapter(ScriptedModels())
    session_id = (await adapter.new_session(str(workspace))).session_id
    await adapter.set_session_mode(session_id, "auto")
    await adapter.bridge(session_id).flush()
    assert [u.current_mode_id for u in client.updates(session_id, CurrentModeUpdate)] == ["auto"]


async def test_ask_mode_is_invalid_until_tools_can_ask(make_adapter, workspace):
    adapter = await make_adapter(ScriptedModels())
    session_id = (await adapter.new_session(str(workspace))).session_id
    for mode in ("ask", "nonsense"):
        with pytest.raises(RequestError) as caught:
            await adapter.set_session_mode(session_id, mode)
        assert caught.value.code == -32602
    with pytest.raises(RequestError):
        await adapter.set_session_mode("no-such-session", "auto")


async def test_the_model_option_lists_the_aliases_and_the_current_model(make_adapter, workspace):
    factory = ModelFactory()
    adapter = await make_adapter(ScriptedModels(), llm_factory=factory, model="fast")
    response = await adapter.new_session(str(workspace))
    [option] = response.config_options or []
    assert option.id == "model" and option.type == "select"
    assert option.current_value == "fast"
    assert [choice.value for choice in option.options] == ["fast", "smart"]


async def test_an_unlisted_current_model_is_offered_too(make_adapter, workspace):
    adapter = await make_adapter(ScriptedModels(), llm_factory=ModelFactory(), model="custom/x")
    response = await adapter.new_session(str(workspace))
    [option] = response.config_options or []
    assert [choice.value for choice in option.options] == ["custom/x", "fast", "smart"]


async def test_load_reports_modes_and_the_model_option(make_adapter, workspace):
    adapter = await make_adapter(ScriptedModels(), llm_factory=ModelFactory(), model="fast")
    session_id = (await adapter.new_session(str(workspace))).session_id
    loaded = await adapter.load_session(str(workspace), session_id)
    assert loaded.config_options is not None and loaded.config_options[0].current_value == "fast"


async def test_choosing_a_model_switches_the_next_turn(make_adapter, workspace, client):
    factory = ModelFactory({"smart": [[reply("From the smart model.")]]})
    adapter = await make_adapter(ScriptedModels(), llm_factory=factory, model="fast")
    session_id = (await adapter.new_session(str(workspace))).session_id
    response = await adapter.set_config_option("model", session_id, "smart")
    [option] = response.config_options
    assert option.current_value == "smart"
    assert [alias for alias, _ in factory.calls][-1] == "smart"
    result = await asyncio.wait_for(adapter.prompt(session_id, [text_block("hi")]), TIMEOUT)
    assert result.stop_reason == "end_turn"
    assert client.texts(AgentMessageChunk, session_id)[-1] == "From the smart model.\n"


async def test_the_model_option_reports_the_chosen_model_at_once_and_after_a_load(
    make_adapter, workspace
):
    adapter = await make_adapter(ScriptedModels(), llm_factory=ModelFactory(), model="fast")
    session_id = (await adapter.new_session(str(workspace))).session_id
    await adapter.set_config_option("model", session_id, "smart")
    session = adapter.registry_for(workspace).get(session_id)
    [option] = adapter._config_options(session)
    assert option.current_value == "smart"  # before any turn used it
    await adapter.close_session(session_id)
    loaded = await adapter.load_session(str(workspace), session_id)
    [option] = loaded.config_options or []
    assert option.current_value == "smart"


async def test_a_model_that_cannot_be_built_is_invalid(make_adapter, workspace):
    def factory(alias, workspace):
        if alias == "broken":
            raise ValueError("unknown model alias 'broken'")
        return ModelFactory()(alias, workspace)

    adapter = await make_adapter(ScriptedModels(), llm_factory=factory, model="fast")
    session_id = (await adapter.new_session(str(workspace))).session_id
    with pytest.raises(RequestError) as caught:
        await adapter.set_config_option("model", session_id, "broken")
    assert caught.value.code == -32602
    assert "broken" in str(caught.value.data)


async def test_unknown_config_options_are_invalid(make_adapter, workspace):
    adapter = await make_adapter(ScriptedModels(), llm_factory=ModelFactory(), model="fast")
    session_id = (await adapter.new_session(str(workspace))).session_id
    with pytest.raises(RequestError):
        await adapter.set_config_option("temperature", session_id, "hot")


class _LeveledModels(ScriptedModels):
    """Scripted agents whose client declares reasoning levels (low by default)."""

    def __call__(self, options, storage):
        from nooa.unifiedllm.reasoning import ReasoningConfig

        agent = super().__call__(options, storage)
        agent.llm._reasoning_config = ReasoningConfig(
            levels={"low": {"reasoning_effort": "low"}, "high": {"reasoning_effort": "high"}},
            default="low",
        )
        return agent


async def test_the_reasoning_option_lists_the_client_levels(make_adapter, workspace):
    adapter = await make_adapter(_LeveledModels(), model="fast")
    response = await adapter.new_session(str(workspace))
    by_id = {option.id: option for option in response.config_options or []}
    assert set(by_id) == {"model", "reasoning"}
    reasoning = by_id["reasoning"]
    assert reasoning.current_value == "low"
    assert [choice.value for choice in reasoning.options] == ["low", "high"]


async def test_choosing_a_reasoning_level_applies_to_the_client(make_adapter, workspace):
    adapter = await make_adapter(_LeveledModels(), model="fast")
    session_id = (await adapter.new_session(str(workspace))).session_id
    response = await adapter.set_config_option("reasoning", session_id, "high")
    reasoning = next(option for option in response.config_options if option.id == "reasoning")
    assert reasoning.current_value == "high"
    assert adapter.session(session_id).agent.llm.reasoning_level == "high"
    with pytest.raises(RequestError) as excinfo:
        await adapter.set_config_option("reasoning", session_id, "max")
    assert excinfo.value.code == -32602
    assert "allowed: low, high" in str(excinfo.value.data)


async def test_no_reasoning_option_without_declared_levels(make_adapter, workspace):
    adapter = await make_adapter(ScriptedModels(), model="fast")
    response = await adapter.new_session(str(workspace))
    assert [option.id for option in response.config_options or []] == ["model"]
    with pytest.raises(RequestError):
        await adapter.set_config_option("reasoning", response.session_id, "high")
