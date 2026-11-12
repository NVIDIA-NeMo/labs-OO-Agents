# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Terminal ledger vs accepted response sources at actual Atom delivery/reload."""

import asyncio

import pytest
from atom_test_agents import CellLLM, ScriptedModels, cell, done, work_done
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa.unifiedllm import LLMUsage
from nooa.unifiedllm._accounting import _logical_account_scope
from nooa.unifiedllm._observations import _NumericUsage


@pytest.fixture
def owned_models(monkeypatch):
    original = CellLLM.acall
    snapshots = []

    async def owned(self, *args, **kwargs):
        with _logical_account_scope() as account:
            # Deliberately participating fake, not automatic duck ownership.
            # A paid rejected attempt is not present on the final response.
            index, owner = account.attempt()
            owner.writer.entering_provider()
            owner.writer.delivered(
                _NumericUsage(input_tokens=7, output_tokens=2, total_tokens=9, cost_usd=0.03)
            )
            account.fold(index, owner)
            response = await original(self, *args, **kwargs)
            index, owner = account.attempt()
            owner.writer.entering_provider()
            owner.writer.delivered(
                _NumericUsage(input_tokens=3, output_tokens=1, total_tokens=4, cost_usd=0.01)
            )
            account.fold(index, owner)
            account.accepted_response = response
        snapshots.append(account.snapshot)
        return response

    monkeypatch.setattr(CellLLM, "acall", owned)
    return snapshots


async def test_owned_canonical_receipt_charges_terminal_once_and_reload(
    registry, root_options, models, sessions_dir, owned_models
):
    models.scripts[None] = [
        done("one", usage=LLMUsage(input_tokens=3, output_tokens=1, total_tokens=4, cost_usd=0.01))
    ]
    root = await registry.create(root_options)
    await asyncio.wait_for(root.prompt("one"), 20)
    assert len(owned_models) == 1
    usage = root.info.usage
    assert (
        usage.input_tokens,
        usage.output_tokens,
        usage.total_tokens,
        usage.cost_usd,
        usage.last_input_tokens,
    ) == (10, 3, 13, 0.04, 3)
    await registry.close_all()
    fresh = SessionRegistry(SessionStore(sessions_dir), agent_factory=ScriptedModels())
    try:
        loaded = await fresh.load(root.id)
        assert (
            loaded.info.usage.input_tokens,
            loaded.info.usage.output_tokens,
            loaded.info.usage.cost_usd,
        ) == (10, 3, 0.04)
    finally:
        await fresh.close_all()


async def test_owned_child_terminal_attributed_once_and_reload(
    registry, root_options, models, sessions_dir, owned_models
):
    final_usage = LLMUsage(input_tokens=3, output_tokens=1, total_tokens=4, cost_usd=0.01)
    models.scripts[None] = [
        cell(
            "a = await self.session.delegate('A', 'a', kept=True, name='a')\nawait a.wait()\nreturn_result(Done(explanation='ok'))",
            usage=final_usage,
        )
    ]
    models.scripts["a"] = [work_done("a done", usage=final_usage)]
    root = await registry.create(root_options)
    await asyncio.wait_for(root.prompt("go"), 20)
    assert len(owned_models) == 2 and len({s.logical_id for s in owned_models}) == 2
    usage = root.info.usage
    assert (
        usage.input_tokens,
        usage.output_tokens,
        usage.attributed_input_tokens,
        usage.attributed_output_tokens,
    ) == (10, 3, 10, 3)
    assert usage.attributed_cost_usd == 0.04
    await registry.close_all()
    fresh = SessionRegistry(SessionStore(sessions_dir), agent_factory=ScriptedModels())
    try:
        loaded = await fresh.load(root.id)
        assert loaded.info.usage == usage.model_copy(update={"last_input_tokens": 0})
    finally:
        await fresh.close_all()


async def test_owned_reflection_uses_fresh_account_and_counts_terminal(tmp_path, owned_models):
    from test_reflection_gate import Clock, idle, make_root, prediction

    clock = Clock()
    registry, root, llm = await make_root(
        tmp_path,
        [done("accepted"), prediction()],
        clock,
        experimental_reflection_min_turns=1,
        experimental_reflection_min_seconds=0,
    )
    try:
        await root.prompt("go")
        await idle(root)
        assert len(owned_models) == 2
        assert len({snapshot.logical_id for snapshot in owned_models}) == 2
        assert sum(call.output_model is not None for call in llm.calls) == 1
        usage = root.info.usage
        assert (usage.input_tokens, usage.output_tokens, usage.cost_usd) == (20, 6, 0.08)
    finally:
        await registry.close_all()


@pytest.mark.parametrize("cancelled", [False, True])
async def test_owned_failed_cancelled_turn_settles_once_and_reload(
    registry, root_options, models, sessions_dir, monkeypatch, cancelled
):
    entered = asyncio.Event()
    snapshots = []

    async def failure(self, *args, **kwargs):
        try:
            with _logical_account_scope() as account:
                index, owner = account.attempt()
                owner.writer.entering_provider()
                owner.writer.delivered(
                    _NumericUsage(input_tokens=7, output_tokens=2, cost_usd=0.03)
                )
                account.fold(index, owner)
                entered.set()
                if cancelled:
                    await asyncio.Event().wait()
                from nooa.unifiedllm import FailureKind, LLMFailure

                raise LLMFailure(FailureKind.BAD_REQUEST, "paid parse failure")
        finally:
            snapshots.append(account.snapshot)

    monkeypatch.setattr(CellLLM, "acall", failure)
    root = await registry.create(root_options)
    prompt = asyncio.create_task(root.prompt("go"))
    await asyncio.wait_for(entered.wait(), 5)
    if cancelled:
        assert await root.cancel()
    if cancelled:
        await asyncio.wait_for(prompt, 20)
    else:
        from nooa_atom.session.session import TurnFailedError

        with pytest.raises(TurnFailedError):
            await asyncio.wait_for(prompt, 20)
    # CodeAct's existing strategy-level failure recovery makes three fresh
    # logical calls; cancellation ends the first. Each is settled exactly once.
    calls = 1 if cancelled else 3
    assert len(snapshots) == calls
    assert len({snapshot.logical_id for snapshot in snapshots}) == calls
    assert all(
        snapshot.outcome == ("cancelled" if cancelled else "error") for snapshot in snapshots
    )
    assert (root.info.usage.input_tokens, root.info.usage.output_tokens) == (7 * calls, 2 * calls)
    assert root.info.usage.cost_usd == pytest.approx(0.03 * calls)
    await registry.close_all()
    fresh = SessionRegistry(SessionStore(sessions_dir), agent_factory=ScriptedModels())
    try:
        loaded = await fresh.load(root.id)
        assert (loaded.info.usage.input_tokens, loaded.info.usage.output_tokens) == (
            7 * calls,
            2 * calls,
        )
        assert loaded.info.usage.cost_usd == pytest.approx(0.03 * calls)
    finally:
        await fresh.close_all()
