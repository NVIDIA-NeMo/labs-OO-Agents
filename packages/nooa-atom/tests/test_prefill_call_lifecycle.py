# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The real prefill receives this call, never a session's first CurrentCall."""

import asyncio

import pytest
from atom_test_agents import CellLLM, cell
from nooa_atom.agent.factory import create_session_agent
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa.interactive import Done
from nooa.strategies.prefill import InspectInputsPrefill


@pytest.mark.parametrize("mode", [None, "legacy", "call", "session"])
async def test_repeated_session_prefills_use_current_call_and_keep_user_globals(
    tmp_path, monkeypatch, mode
):
    original = InspectInputsPrefill.get_code

    def get_code(self, call, config=None):
        # Exercise the actual prefill injection, not a synthetic execute_code
        # environment. _call is intentionally injected here; ordinary model
        # cells are not promised this private runtime name.
        return (original(self, call, config=config) or "") + (
            "\nself.v.prefill_seen = str(_call.kwargs['notification'])\n"
            "self.v.prefill_matches = _call is self.runtime.current_call\n"
        )

    monkeypatch.setattr(InspectInputsPrefill, "get_code", get_code)
    retained = mode == "session"
    llm = CellLLM(
        [
            cell("ordinary = ['retained']\nreturn_result(Done(explanation='first'))"),
            cell(
                (
                    "assert ordinary == ['retained']\n"
                    if retained
                    else "try:\n ordinary\nexcept NameError:\n pass\nelse:\n raise AssertionError('leaked user global')\n"
                )
                + "return_result(Done(explanation='second'))"
            ),
        ]
    )
    made = []

    def factory(options, storage):
        agent = create_session_agent(options.model_copy(update={"llm": llm}), storage)
        made.append(agent)
        return agent

    registry = SessionRegistry(SessionStore(tmp_path / "sessions"), agent_factory=factory)
    options = {"experimental_namespace": mode} if mode is not None else {}
    try:
        root = await registry.create(
            SessionOptions(
                workspace=tmp_path,
                agent_spec="nooa_atom.agent:AtomAgent",
                experimental_done_reflection=False,
                **options,
            )
        )
        assert await asyncio.wait_for(root.prompt("OLD-NOTIFICATION"), 30) == Done(
            explanation="first"
        )
        agent = made[0]
        assert "OLD-NOTIFICATION" in agent.v.prefill_seen and agent.v.prefill_matches
        owner = agent._handle_namespace
        ordinary = owner.globals["ordinary"] if retained else None
        assert await asyncio.wait_for(root.prompt("NEW-NOTIFICATION"), 30) == Done(
            explanation="second"
        )
        assert "NEW-NOTIFICATION" in agent.v.prefill_seen and agent.v.prefill_matches
        assert "OLD-NOTIFICATION" not in agent.v.prefill_seen
        assert len(llm.calls) == 2
        if retained:
            assert agent._handle_namespace is owner
            assert owner.globals["ordinary"] is ordinary
            assert "_call" not in owner.globals
        else:
            assert agent._handle_namespace is None
    finally:
        await registry.close_all()
