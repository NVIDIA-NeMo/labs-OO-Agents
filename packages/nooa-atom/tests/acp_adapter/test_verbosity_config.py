# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pool's config-option picker controls ACP verbosity without an agent turn."""

import pytest
from acp import RequestError, text_block
from acp.schema import ConfigOptionUpdate, Implementation
from atom_test_agents import ScriptedModels


def verbosity(options):
    return next(o for o in options if o.id == "verbosity")


async def test_pool_verbosity_picker_and_text_command_share_state(make_adapter, workspace, client):
    adapter = await make_adapter(
        ScriptedModels(), client_info=Implementation(name="pool", version="1.0.16")
    )
    response = await adapter.new_session(str(workspace))
    session_id = response.session_id
    option = verbosity(response.config_options)
    assert option.name == "Verbosity" and option.current_value == "verbose"
    assert [(o.value, o.name) for o in option.options] == [
        ("silent", "Silent"),
        ("concise", "Concise"),
        ("verbose", "Verbose"),
    ]
    for level in ("silent", "concise", "verbose"):
        selected = await adapter.set_config_option("verbosity", session_id, level)
        assert verbosity(selected.config_options).current_value == level
        assert adapter.bridge(session_id).verbosity == level
    await adapter.prompt(session_id, [text_block("/verbosity concise")])
    assert (
        verbosity(client.updates(session_id, ConfigOptionUpdate)[-1].config_options).current_value
        == "concise"
    )
    assert adapter.session(session_id).info.reply_count == 0
    assert not client.ext_answers  # no elicitation is needed
    for invalid in ("other", True, "SILENT"):
        with pytest.raises(RequestError) as exc:
            await adapter.set_config_option("verbosity", session_id, invalid)
        assert exc.value.code == -32602
    assert adapter.bridge(session_id).verbosity == "concise"
    with pytest.raises(RequestError):
        await adapter.set_config_option("verbosity", "missing", "silent")
    # New attachments use the saved workspace verbosity.
    await adapter.close_session(session_id)
    loaded = await adapter.load_session(str(workspace), session_id)
    assert verbosity(loaded.config_options).current_value == "concise"
