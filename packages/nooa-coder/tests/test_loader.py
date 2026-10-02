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


_ANNOTATED_AGENT_FILE = """\
from __future__ import annotations

from nooa.interactive import InteractiveAgent
from nooa.unifiedllm import FakeLLMClient


class {helper}:
    pass


class {name}(InteractiveAgent, llm=FakeLLMClient()):
    helper: {helper} | None = None
"""


def _resolve_own_annotation(cls: type, name: str) -> str:
    """Resolve one string annotation the way typing does: via sys.modules[cls.__module__]."""
    import sys
    import typing

    hint = eval(cls.__dict__["__annotations__"][name], vars(sys.modules[cls.__module__]))
    (helper,) = [arg for arg in typing.get_args(hint) if arg is not type(None)]
    return f"{helper.__module__}.{helper.__qualname__}"


def test_file_agents_get_distinct_modules_and_keep_resolving_annotations(tmp_path):
    """A second file-based agent must not replace the first one's sys.modules entry.

    Postponed annotations resolve through ``sys.modules[cls.__module__]``; with
    one shared module name the first class's annotations pointed at the second
    file's namespace and ``get_type_hints`` failed. (From coder/3-engine's
    test_coding_factory.py, now against the one loader.)
    """
    import sys

    first = tmp_path / "one" / "agent.py"
    second = tmp_path / "two" / "agent.py"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text(_ANNOTATED_AGENT_FILE.format(name="FirstAgent", helper="FirstHelper"))
    second.write_text(_ANNOTATED_AGENT_FILE.format(name="SecondAgent", helper="SecondHelper"))

    first_cls = load_agent_class(f"{first}:FirstAgent")
    second_cls = load_agent_class(f"{second}:SecondAgent")

    assert first_cls.__module__ != second_cls.__module__
    assert sys.modules[first_cls.__module__].FirstAgent is first_cls
    assert _resolve_own_annotation(first_cls, "helper") == first_cls.__module__ + ".FirstHelper"
    assert _resolve_own_annotation(second_cls, "helper") == second_cls.__module__ + ".SecondHelper"
    # The same unchanged file loads once, so repeated specs share one class.
    assert load_agent_class(f"{first}:FirstAgent") is first_cls


def test_a_relative_file_spec_resolves_against_the_given_base(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "agents").mkdir(parents=True)
    (workspace / "agents" / "mine.py").write_text(
        _ANNOTATED_AGENT_FILE.format(name="MyAgent", helper="MyHelper")
    )
    cls = load_agent_class("./agents/mine.py:MyAgent", base=workspace)
    assert cls.__name__ == "MyAgent"


@pytest.mark.parametrize(
    "body",
    [
        None,  # the file does not exist
        "class Other:\n    pass\n",  # no such class
        "from nooa import Agent\n\nclass Plain(Agent):\n    pass\n",  # not interactive
    ],
)
def test_bad_file_specs_are_rejected(tmp_path, body):
    path = tmp_path / "agent_file.py"
    if body is not None:
        path.write_text(body)
    with pytest.raises(AgentSpecError):
        load_agent_class(f"{path}:Plain")
