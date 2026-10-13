# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verbosity uses session settings, not an elicitation form."""

import pytest
from acp import text_block
from acp.schema import (
    AgentMessageChunk,
    ClientCapabilities,
    ConfigOptionUpdate,
    ElicitationCapabilities,
    ElicitationFormCapabilities,
    Implementation,
    UserMessageChunk,
)
from atom_test_agents import ScriptedModels


@pytest.mark.parametrize("pool", [False, True])
async def test_verbosity_query_uses_settings_without_elicitation(
    make_adapter, workspace, client, pool
):
    adapter = await make_adapter(
        ScriptedModels(),
        client_info=Implementation(name="pool", version="1.0.16") if pool else None,
        capabilities=ClientCapabilities(
            elicitation=ElicitationCapabilities(form=ElicitationFormCapabilities())
        ),
    )
    response = await adapter.new_session(str(workspace))
    session_id = response.session_id
    option = next(o for o in response.config_options if o.id == "verbosity")
    assert option.type == "select" and option.current_value == "verbose"
    assert [o.value for o in option.options] == ["silent", "concise", "verbose"]
    model_before = adapter.session(session_id).model_info()
    mode_before = adapter.session(session_id).info.mode
    result = await adapter.prompt(session_id, [text_block("/verbosity")])
    assert result.stop_reason == "end_turn"
    assert "session settings" in client.texts(AgentMessageChunk, session_id)[-1]
    [update] = client.updates(session_id, ConfigOptionUpdate)
    assert next(o for o in update.config_options if o.id == "verbosity").current_value == "verbose"
    for level in ("silent", "concise", "verbose"):
        selected = await adapter.set_config_option("verbosity", session_id, level)
        assert (
            next(o for o in selected.config_options if o.id == "verbosity").current_value == level
        )
        assert adapter.bridge(session_id).verbosity == level
    assert not any(entry[0] in {"ext", "elicitation", "permission"} for entry in client.log)
    assert adapter.session(session_id).info.reply_count == 0
    assert adapter.session(session_id).transcript() == []
    assert client.texts(UserMessageChunk, session_id) == []
    assert adapter.session(session_id).model_info() == model_before
    assert adapter.session(session_id).info.mode == mode_before
