# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An Atom cell sees the same framework names after the namespace-provider registry (#558 L9)."""

import asyncio

from atom_test_agents import CellLLM, cell
from nooa_atom.agent.factory import create_session_agent
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa.runtime.execution_namespace import ExecutionNamespace

# The framework names an Atom handle cell saw before the registry: the core
# cell names plus the CodeActV2 builtins.
FRAMEWORK_NAMES = frozenset(
    {
        "self", "asyncio", "typing",
        "Annotated", "Any", "Literal", "Optional", "Union",
        "doc", "methods", "variables", "help", "pprint",
        "show", "Media", "Image", "Audio", "Video", "File",
        "strategy", "PurePythonStrategy", "CodeActStrategy", "ReflexionStrategy",
        "PredictStrategy", "TemplateStrategy", "CompositeStrategy",
        "notification", "python_cell_state", "return_result",
    }
)  # fmt: skip


async def test_an_atom_cell_sees_the_same_framework_names(tmp_path, monkeypatch):
    import nooa.runtime.actor as actor

    seen: list[set[str]] = []
    original = actor._strip_blocked_modules

    def spy(ns, blocked):
        seen.append(set(ns))
        return original(ns, blocked)

    monkeypatch.setattr(actor, "_strip_blocked_modules", spy)
    static: list[dict] = []
    initialize = ExecutionNamespace.initialize

    def spy_initialize(self, exec_globals, *, static_names):
        static.append(dict(static_names))
        return initialize(self, exec_globals, static_names=static_names)

    monkeypatch.setattr(ExecutionNamespace, "initialize", spy_initialize)
    registry = SessionRegistry(
        SessionStore(tmp_path / "sessions"), agent_factory=create_session_agent
    )
    options = SessionOptions(
        workspace=tmp_path,
        agent_spec="nooa_atom.agent:AtomAgent",
        llm=CellLLM([cell("return_result(Done(explanation='x'))")]),
    )
    root = await registry.create(options)
    try:
        await asyncio.wait_for(root.prompt("hi"), 30)
    finally:
        await registry.close_all()
    [cell_names] = [names for names in seen if "self" in names][-1:]
    assert FRAMEWORK_NAMES <= cell_names
    # The framework names the owned namespace keeps static: the registry's names.
    assert set(static[-1]) == FRAMEWORK_NAMES - {
        "notification",
        "python_cell_state",
        "return_result",
    }
