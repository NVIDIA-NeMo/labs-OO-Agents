# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Loading agent classes from a module:Class spec, and building one for a session."""

import pytest
from atom_test_agents import EchoAgent
from nooa_atom.agent.factory import create_session_agent
from nooa_atom.session.loader import AgentSpecError, load_agent_class
from nooa_atom.session.options import SessionOptions

from nooa.storage import InMemoryStorageManager
from nooa.unifiedllm import FakeLLMClient


def test_loads_an_interactive_agent_class():
    assert load_agent_class("atom_test_agents:EchoAgent") is EchoAgent


@pytest.mark.parametrize(
    "spec",
    [
        "atom_test_agents",  # no class
        "atom_test_agents:",  # empty class
        ":EchoAgent",  # empty module
        "no_such_module_for_nooa_atom:Agent",
        "atom_test_agents:NoSuchAgent",
        "atom_test_agents:NotAnAgent",  # not an InteractiveAgent
    ],
)
def test_bad_specs_are_rejected(spec):
    with pytest.raises(AgentSpecError):
        load_agent_class(spec)


def test_the_session_factory_passes_storage_and_llm(tmp_path):
    storage = InMemoryStorageManager()
    llm = FakeLLMClient()
    options = SessionOptions(workspace=tmp_path, agent_spec="atom_test_agents:EchoAgent", llm=llm)
    agent = create_session_agent(options, storage)
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
    file's namespace and ``get_type_hints`` failed. (From atom/3-engine's
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


@pytest.mark.parametrize(
    "spec",
    [
        "nooa_atom.agent:ExperimentalAtomAgent",
        "nooa_cli.coding.experimental_agent:ExperimentalTUIAgent",
    ],
)
def test_removed_experimental_agent_specs_load_the_atom_agent(spec):
    """ExperimentalAtomAgent and the experimental agents before it are gone; AtomAgent loads."""
    from nooa_atom.agent.agent import AtomAgent

    assert load_agent_class(spec) is AtomAgent


@pytest.mark.parametrize(
    "spec",
    [
        "nooa_atom.agent:ExperimentalAtomAgent",
        "nooa_coder.coding.experimental_agent:ExperimentalCodingAgent",
    ],
)
async def test_a_saved_experimental_session_loads_as_an_atom_agent(spec, tmp_path, sessions_dir):
    """A session saved under a removed experimental agent spec resumes with its state."""
    import asyncio

    from atom_test_agents import python_cell
    from nooa_atom.agent.agent import AtomAgent
    from nooa_atom.session.registry import SessionRegistry
    from nooa_atom.session.store import SessionStore

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    code = "self.todo.add('carry the plan over'); self.v.note = 'kept'\n" + (
        "return_result(Done(explanation='saved'))"
    )
    llm = FakeLLMClient([python_cell(code, "c1")], strict_exhaustion=True)
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=create_session_agent)
    try:
        session = await registry.create(
            SessionOptions(workspace=workspace, agent_spec=spec, llm=llm, sessions_dir=sessions_dir)
        )
        await asyncio.wait_for(session.prompt("remember this"), 30)
        session_id = session.id
    finally:
        await registry.close_all()

    store = SessionStore(sessions_dir)
    [info] = store.list(workspace=workspace)
    assert info.agent == spec
    registry = SessionRegistry(store, agent_factory=create_session_agent)
    try:
        session = await registry.load(session_id, llm=FakeLLMClient([]))
        assert type(session._agent) is AtomAgent
        assert session._agent.v.note == "kept"
        assert "carry the plan over" in session._agent.todo.status()
    finally:
        await registry.close_all()
