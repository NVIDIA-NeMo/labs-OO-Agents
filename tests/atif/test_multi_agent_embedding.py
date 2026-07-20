# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-agent embedding: a pure-Python orchestrator driving distinct sub-Agent
instances (plus standalones) produces ONE trajectory.

Each sub-agent run embeds into the orchestrator's ``subagent_trajectories[]``
(no separate file). A reused sub-agent instance shares its event history, so it
appears as ONE accumulating sub-trajectory; each handoff emits a reference step
carrying the step-range it produced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from nooa import Agent, strategy
from nooa.atif import Trajectory, enable_atif
from nooa.standalone import _atif_exporter_var
from nooa.strategies import PredictStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse
from tests.atif.normative import assert_atif_normative

_DEFAULT = FakeLLMClient()


def _resp(content: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": content},
        usage={"prompt_tokens": 5, "completion_tokens": 1},
    )


@pytest.fixture
def _isolated_agent_init():
    from nooa import atif as atif_module
    from nooa.agent import Agent

    orig_init = Agent.__init__
    orig_patched = atif_module.install._AGENT_INIT_PATCHED
    orig_cfg = dict(atif_module.install._ENABLE_ATIF_CONFIG)
    try:
        yield
    finally:
        Agent.__init__ = orig_init  # type: ignore[method-assign]
        atif_module.install._AGENT_INIT_PATCHED = orig_patched
        atif_module.install._ENABLE_ATIF_CONFIG.clear()
        atif_module.install._ENABLE_ATIF_CONFIG.update(orig_cfg)


def _only(output_dir: Path, name: str) -> Trajectory:
    files = list((output_dir / name).glob("*.json"))
    assert len(files) == 1, f"expected one {name} trajectory, got {files}"
    return Trajectory.model_validate_json(files[0].read_text())


@strategy(PredictStrategy())
async def classify(text: str) -> Literal["ok", "revise"]:
    """Reply 'ok' if {text} is a complete sentence, else 'revise'."""
    ...


class _Producer(Agent, llm=_DEFAULT):
    @strategy(PredictStrategy())
    async def produce(self, spec: str) -> str:
        """Write a one-sentence draft for {spec}."""
        ...


class _Verifier(Agent, llm=_DEFAULT):
    @strategy(PredictStrategy())
    async def verify(self, draft: str) -> str:
        """Reply 'ok' if {draft} is a complete sentence, else 'revise'."""
        ...


class _Orchestrator(Agent, llm=_DEFAULT):
    async def run(self, spec: str, rounds: int = 1) -> str:
        producer = _Producer(llm=self._llm)
        verifier = _Verifier(llm=self._llm)
        draft = ""
        for _ in range(rounds):
            draft = await producer.produce(spec)
            verdict = await verifier.verify(draft)
            tag = await classify(draft)  # standalone
            if "ok" in verdict.lower() and tag == "ok":
                break
        return draft


@pytest.mark.asyncio
async def test_orchestrator_embeds_subagents_and_standalone(
    tmp_path: Path, _isolated_agent_init
) -> None:
    output_dir = tmp_path / "atif"
    enable_atif(output_dir=output_dir)
    fake = FakeLLMClient(
        scripted_responses=[
            _resp(content='{"value": "a complete sentence."}'),  # produce
            _resp(content='{"value": "ok"}'),  # verify
            _resp(content='{"value": "ok"}'),  # classify
        ]
    )
    assert await _Orchestrator(llm=fake).run("spec") == "a complete sentence."

    # Single combined trajectory; sub-agents are NOT separate files.
    assert not (output_dir / "_Producer").exists()
    assert not (output_dir / "_Verifier").exists()

    traj = _only(output_dir, "_Orchestrator")
    assert_atif_normative(traj)
    assert traj.final_metrics is not None  # B2: pure-Python orchestrator finalized

    subs = sorted(c.agent.name for c in (traj.subagent_trajectories or []))
    assert subs == ["_Producer", "_Verifier", "classify"]

    kinds = [(s.extra or {}).get("event_kind") for s in traj.steps]
    assert kinds == ["subagent_handoff", "subagent_handoff", "standalone_dispatch"]

    # Each handoff ref resolves and carries a step-range offset.
    child_ids = {c.trajectory_id for c in traj.subagent_trajectories}
    for s in traj.steps:
        for r in s.observation.results if s.observation else []:
            for ref in r.subagent_trajectory_ref or []:
                assert ref.trajectory_id in child_ids
                if (s.extra or {}).get("event_kind") == "subagent_handoff":
                    assert (ref.extra or {}).get("subagent_step_range") is not None

    assert _atif_exporter_var.get() is None


@pytest.mark.asyncio
async def test_reused_subagent_accumulates_with_per_handoff_ranges(
    tmp_path: Path, _isolated_agent_init
) -> None:
    """A sub-agent instance reused across loop rounds is ONE accumulating
    sub-trajectory; each handoff emits a ref step with that round's step-range."""
    output_dir = tmp_path / "atif"
    enable_atif(output_dir=output_dir)
    # Round 1: verify says revise → loop again. Round 2: ok → break.
    fake = FakeLLMClient(
        scripted_responses=[
            _resp(content='{"value": "draft one"}'),  # produce r1
            _resp(content='{"value": "revise"}'),  # verify r1
            _resp(content='{"value": "revise"}'),  # classify r1
            _resp(content='{"value": "draft two."}'),  # produce r2
            _resp(content='{"value": "ok"}'),  # verify r2
            _resp(content='{"value": "ok"}'),  # classify r2
        ]
    )
    await _Orchestrator(llm=fake).run("spec", rounds=2)

    traj = _only(output_dir, "_Orchestrator")
    assert_atif_normative(traj)

    # ONE Producer sub-trajectory (accumulating both rounds), not two.
    producers = [c for c in (traj.subagent_trajectories or []) if c.agent.name == "_Producer"]
    assert len(producers) == 1, "reused sub-agent must be a single accumulating sub-trajectory"
    # Shared history: the system step is emitted once; round 1 = system+user+agent
    # (3), round 2 = user+agent (2) → 5 accumulated steps.
    assert len(producers[0].steps) == 5

    # Two Producer handoff steps, with non-overlapping increasing ranges.
    ranges = []
    for s in traj.steps:
        if (s.extra or {}).get("subagent_name") == "_Producer":
            ref = s.observation.results[0].subagent_trajectory_ref[0]
            ranges.append(tuple(ref.extra["subagent_step_range"]))
    assert ranges == [(0, 3), (3, 5)], f"per-handoff step-ranges; got {ranges}"


@pytest.mark.asyncio
async def test_subagent_run_top_level_after_nested_writes_own_file(
    tmp_path: Path, _isolated_agent_init
) -> None:
    """File-write suppression is scoped to the nested call: an instance run
    nested (embedded, no file) and *later* run top-level must write its own
    file."""
    output_dir = tmp_path / "atif"
    enable_atif(output_dir=output_dir)
    fake = FakeLLMClient(
        scripted_responses=[_resp(content='{"value": "a"}'), _resp(content='{"value": "b"}')]
    )

    class _Worker(Agent, llm=_DEFAULT):
        @strategy(PredictStrategy())
        async def work(self, x: str) -> str:
            """Do {x}."""
            ...

    class _Orch(Agent, llm=_DEFAULT):
        def __init__(self, worker: _Worker, **kw: object) -> None:
            super().__init__(**kw)
            self._worker = worker

        async def run(self, x: str) -> str:
            return await self._worker.work(x)  # nested ⇒ embedded, no own file

    worker = _Worker(llm=fake)
    await _Orch(worker, llm=fake).run("first")
    assert not (output_dir / "_Worker").exists(), "nested run must not write its own file"

    # Same instance, now top-level: it must write its own file again.
    assert await worker.work("second") == "b"
    assert (output_dir / "_Worker").exists(), (
        "file suppression leaked past the nested call (instance never writes again)"
    )


@pytest.mark.asyncio
async def test_pure_orchestrator_standalone_only_finalizes(
    tmp_path: Path, _isolated_agent_init
) -> None:
    """An orchestrator that calls ONLY a standalone (no generation method, no
    sub-agent) still flushes its dispatch step and finalizes (bug B2)."""
    output_dir = tmp_path / "atif"
    enable_atif(output_dir=output_dir)

    class _OnlyStandalone(Agent, llm=_DEFAULT):
        async def run(self) -> str:
            return await classify("a complete sentence.")

    fake = FakeLLMClient(scripted_responses=[_resp(content='{"value": "ok"}')])
    assert await _OnlyStandalone(llm=fake).run() == "ok"

    traj = _only(output_dir, "_OnlyStandalone")
    assert_atif_normative(traj)
    assert traj.final_metrics is not None
    assert [(s.extra or {}).get("event_kind") for s in traj.steps] == ["standalone_dispatch"]
    assert (traj.subagent_trajectories or [])[0].agent.name == "classify"
    assert _atif_exporter_var.get() is None
