# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The model option offers the aliases of the session's own workspace."""

import logging
import os
import time
from pathlib import Path

import pytest
from atom_test_agents import ScriptedModels
from nooa_atom.agent.factory import default_llm_factory


def write_models(workspace: Path, **aliases: str) -> Path:
    path = workspace / ".nooa" / "llm_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["models:"]
    for alias, model_name in aliases.items():
        lines += [f"  {alias}:", f"    model_name: {model_name}"]
    path.write_text("\n".join(lines) + "\n")
    # A same-size rewrite within one clock tick must still be seen.
    stamp = time.time_ns() + 2_000_000_000
    os.utime(path, ns=(stamp, stamp))
    return path


@pytest.fixture(autouse=True)
def _no_outside_config(tmp_path, monkeypatch):
    monkeypatch.delenv("NEMO_OO_LLM_CONFIG", raising=False)
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "package-project"))


def offered(response) -> list[str]:
    [option] = [o for o in response.config_options or [] if o.id == "model"]
    return [choice.value for choice in option.options]


async def test_a_workspace_alias_is_offered_and_one_added_later_appears(make_adapter, workspace):
    write_models(workspace, mine="openai/mine-model")
    adapter = await make_adapter(ScriptedModels(), llm_factory=default_llm_factory(), model="m")
    response = await adapter.new_session(str(workspace))
    assert "mine" in offered(response)

    write_models(workspace, mine="openai/mine-model", later="openai/later-model")
    response = await adapter.set_config_option("model", response.session_id, "later")
    assert {"mine", "later"} <= set(offered(response))


async def test_one_process_keeps_the_aliases_of_each_workspace_apart(make_adapter, tmp_path):
    one, two = tmp_path / "a", tmp_path / "b"
    write_models(one, a_only="openai/a-model")
    two.mkdir()
    adapter = await make_adapter(ScriptedModels(), llm_factory=default_llm_factory(), model="m")
    first = await adapter.new_session(str(one))
    second = await adapter.new_session(str(two))
    assert "a_only" in offered(first)
    assert "a_only" not in offered(second)

    await adapter.set_config_option("model", first.session_id, "a_only")
    await adapter.set_config_option("model", second.session_id, "a_only")
    # Built from A's entry in A; in B the name is not an alias and goes to litellm as is.
    assert adapter.session(first.session_id)._next_llm().model == "openai/a-model"
    assert adapter.session(second.session_id)._next_llm().model == "a_only"


async def test_the_first_session_in_a_workspace_logs_its_configuration(
    make_adapter, workspace, caplog
):
    own = write_models(workspace, mine="openai/mine-model")
    adapter = await make_adapter(ScriptedModels(), llm_factory=default_llm_factory(), model="m")
    with caplog.at_level(logging.INFO, logger="nooa_atom.acp"):
        await adapter.new_session(str(workspace))
        await adapter.new_session(str(workspace))
    lines = [r.getMessage() for r in caplog.records if "LLM configuration" in r.getMessage()]
    assert len(lines) == 1
    assert str(own.resolve()) in lines[0]
    assert str(workspace) in lines[0]
