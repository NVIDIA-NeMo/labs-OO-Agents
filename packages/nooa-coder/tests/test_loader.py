# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Loading agent classes from a module:Class spec, and the default agent factory."""

import pytest
from coder_test_agents import EchoAgent
from nooa_coder.session.loader import AgentSpecError, default_agent_factory, load_agent_class
from nooa_coder.session.options import SessionOptions

from nooa.storage import InMemoryStorageManager
from nooa.unifiedllm import FakeLLMClient


def test_loads_an_interactive_agent_class():
    assert load_agent_class("coder_test_agents:EchoAgent") is EchoAgent


@pytest.mark.parametrize(
    "spec",
    [
        "coder_test_agents",  # no class
        "coder_test_agents:",  # empty class
        ":EchoAgent",  # empty module
        "no_such_module_for_nooa_coder:Agent",
        "coder_test_agents:NoSuchAgent",
        "coder_test_agents:NotAnAgent",  # not an InteractiveAgent
    ],
)
def test_bad_specs_are_rejected(spec):
    with pytest.raises(AgentSpecError):
        load_agent_class(spec)


def test_default_factory_passes_storage_and_llm(tmp_path):
    storage = InMemoryStorageManager()
    llm = FakeLLMClient()
    options = SessionOptions(workspace=tmp_path, agent_spec="coder_test_agents:EchoAgent", llm=llm)
    agent = default_agent_factory(options, storage)
    assert isinstance(agent, EchoAgent)
    assert agent.llm is llm
    assert agent._storage is storage
