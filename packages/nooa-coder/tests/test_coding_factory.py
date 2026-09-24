# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""create_session_agent() must forward workspace-scoping kwargs correctly."""

from nooa_coder.coding.agent import CodingAgent
from nooa_coder.coding.factory import create_session_agent
from nooa_coder.interactive.options import SessionOptions

from nooa import Agent
from nooa.storage import InMemoryStorageManager
from nooa.unifiedllm import FakeLLMClient


class _KwargsForwardingCodingAgent(CodingAgent):
    """The normal subclass-extension pattern: forwards **kwargs to super()."""

    def __init__(self, llm=None, storage=None, **kwargs):
        super().__init__(llm=llm, storage=storage, **kwargs)


class _KwargsAcceptingUnrelatedAgent(Agent):
    """An unrelated custom Agent that happens to declare **kwargs too, but
    does not descend from CodingAgent and does not understand cwd/
    skills_dirs/config/libs_dir at all.
    """

    def __init__(self, llm=None, storage=None, **kwargs):
        if kwargs:
            raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")
        super().__init__(llm=llm, storage=storage)


def test_kwargs_forwarding_coding_agent_subclass_gets_the_real_workspace(tmp_path):
    """A CodingAgent subclass using **kwargs must still get cwd/skills_dirs/
    etc. -- a literal-name-only check silently drops them, falling back to
    cwd='.' (this process's own directory) instead of the session's actual
    workspace, exactly the cross-workspace code exposure CodingAgent's own
    libs_dir comment warns about.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    options = SessionOptions(working_dir=str(workspace))
    agent = create_session_agent(
        llm=FakeLLMClient(),
        storage=InMemoryStorageManager(),
        options=options,
        agent_cls=_KwargsForwardingCodingAgent,
    )
    try:
        assert agent.cwd == workspace.resolve()
    finally:
        pass


def test_kwargs_forwarding_does_not_break_an_unrelated_custom_agent(tmp_path):
    """A **kwargs-declaring Agent that is NOT a CodingAgent must not have
    CodingAgent-specific kwargs (cwd, skills_dirs, config, libs_dir) forced
    on it -- those would raise from code that never asked for them.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    options = SessionOptions(working_dir=str(workspace))
    agent = create_session_agent(
        llm=FakeLLMClient(),
        storage=InMemoryStorageManager(),
        options=options,
        agent_cls=_KwargsAcceptingUnrelatedAgent,
    )
    assert isinstance(agent, _KwargsAcceptingUnrelatedAgent)


def test_named_parameters_still_work_without_kwargs(tmp_path):
    """The original name-matching path (no **kwargs involved) is unaffected."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    options = SessionOptions(working_dir=str(workspace))
    agent = create_session_agent(
        llm=FakeLLMClient(),
        storage=InMemoryStorageManager(),
        options=options,
        agent_cls=CodingAgent,
    )
    assert agent.cwd == workspace.resolve()


_ANNOTATED_AGENT_FILE = """\
from __future__ import annotations

from nooa import Agent


class {helper}:
    pass


class {name}(Agent):
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
    file's namespace and ``get_type_hints`` failed.
    """
    import sys

    from nooa_coder.coding.factory import load_agent_class

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
