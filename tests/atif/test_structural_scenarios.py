# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end ATIF scenarios, asserted by structure (not byte-pinned).

Each scenario runs a representative agent (real runtime via FakeLLMClient,
or synthetic events for crash/multimodal/compaction) and asserts the
*structural* facts that define the exporter's contract: step sources and
ordering, tool_call/observation joinability, subagent embedding, copied-
context propagation, crash markers, and metrics aggregation.

Asserting structure (rather than byte-pinning the whole serialized
trajectory) keeps these tests focused on the exporter's behavior: they
don't churn on framework system-prompt wording or unrelated module-symbol
changes. The MUST-invariants are still enforced on every trajectory via
``assert_atif_normative`` plus schema validation in :func:`_loaded`; these
tests layer scenario-specific structural checks on top.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest

from nooa import Agent, strategy
from nooa.atif import Trajectory, atif_scope
from nooa.atif.exporter import AtifExporter
from nooa.atif.schema import StepObject
from nooa.config import CodeActConfig
from nooa.context_blocks import ResultStatus
from nooa.events import (
    AfterTurn,
    BeforeTurn,
    LLMComplete,
    LLMOutput,
    PythonOutput,
    Summary,
    SystemPrompt,
    Task,
)
from nooa.strategies import CodeActStrategy, PredictStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from tests.atif.normative import assert_atif_normative

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exec_tool_call(code: str, call_id: str = "call_exec") -> ToolCall:
    return ToolCall(id=call_id, name="execute_python", arguments=json.dumps({"code": code}))


def _ret_tool_call(result: object, call_id: str = "call_ret") -> ToolCall:
    return ToolCall(id=call_id, name="return_result", arguments=json.dumps({"result": result}))


def _resp(
    content: str = "", tool_calls: list | None = None, *, usage: dict | None = None
) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=tool_calls or [],
        finish_reason="tool_calls" if tool_calls else "stop",
        assistant_message={"role": "assistant", "content": content},
        usage=usage or {"prompt_tokens": 50, "completion_tokens": 10},
    )


_TEST_LLM_DEFAULT = FakeLLMClient()


def _loaded(path: Path) -> Trajectory:
    """Read the on-disk trajectory, validating schema + normative invariants."""
    traj = Trajectory.model_validate_json(path.read_text())
    assert_atif_normative(traj)
    return traj


def _by_source(traj: Trajectory, source: str) -> list[StepObject]:
    return [s for s in traj.steps if s.source == source]


def _all_tool_calls(step: StepObject) -> dict[str, str]:
    """Map tool_call_id -> function_name for a step."""
    return {tc.tool_call_id: tc.function_name for tc in (step.tool_calls or [])}


def _observation_content(step: StepObject, tool_call_id: str) -> str | None:
    if step.observation is None:
        return None
    for r in step.observation.results:
        if r.source_call_id == tool_call_id:
            return r.content if isinstance(r.content, str) else None
    return None


def _subagent_refs(step: StepObject) -> list[str]:
    """trajectory_ids referenced from a step's observation."""
    if step.observation is None:
        return []
    return [
        ref.trajectory_id
        for r in step.observation.results
        for ref in (r.subagent_trajectory_ref or [])
        if ref.trajectory_id is not None
    ]


# ---------------------------------------------------------------------------
# Agents used in scenarios
# ---------------------------------------------------------------------------


class _CodeActAgent(Agent, llm=_TEST_LLM_DEFAULT):
    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=5)))
    async def run(self, prompt: str) -> int:
        """Solve {prompt}."""
        ...


class _ClassifierAgent(Agent, llm=_TEST_LLM_DEFAULT):
    @strategy(PredictStrategy())
    async def classify(self, text: str) -> Literal["pos", "neg"]:
        """Classify {text}."""
        ...


class _NestedAgent(Agent, llm=_TEST_LLM_DEFAULT):
    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=5)))
    async def outer(self, x: str) -> str:
        """{x}"""
        ...

    @strategy(PredictStrategy())
    async def inner_classify(self, text: str) -> Literal["a", "b"]:
        """Classify {text}."""
        ...


@strategy(PredictStrategy())
async def _standalone_classify(text: str) -> Literal["pos", "neg"]:
    """Classify {text}."""
    ...


@strategy(PredictStrategy())
async def _standalone_gather_one(text: str) -> Literal["pos", "neg", "neu"]:
    """Classify {text}."""
    ...


class _StandaloneCaller(Agent, llm=_TEST_LLM_DEFAULT):
    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=5)))
    async def run(self, prompt: str) -> str:
        """{prompt}"""
        ...


# ---------------------------------------------------------------------------
# End-to-end scenarios — real agent runtime via FakeLLMClient
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_codeact_single_turn(tmp_path: Path) -> None:
    """One CodeAct turn: system + user + agent; the return_result tool_call
    and its observation are present and joinable; metrics aggregate."""
    fake = FakeLLMClient(
        scripted_responses=[_resp(tool_calls=[_ret_tool_call(2, call_id="call_ret")])]
    )
    agent = _CodeActAgent(llm=fake)
    out = tmp_path / "t.json"
    async with atif_scope(
        agent, path=out, session_id="s1", agent_name="_CodeActAgent", agent_version="0.1.0"
    ):
        assert await agent.run("hi") == 2

    traj = _loaded(out)
    assert [s.source for s in traj.steps] == ["system", "user", "agent"]
    agent_step = _by_source(traj, "agent")[0]
    assert agent_step.llm_call_count == 1
    assert "return_result" in _all_tool_calls(agent_step).values()
    # system prompt captured and non-empty.
    assert _by_source(traj, "system")[0].message
    # metrics aggregated.
    assert traj.final_metrics is not None
    assert traj.final_metrics.total_steps == 3
    assert traj.final_metrics.total_prompt_tokens and traj.final_metrics.total_prompt_tokens > 0


@pytest.mark.asyncio
async def test_codeact_multi_turn_with_error(tmp_path: Path) -> None:
    """Two CodeAct turns (first errors, second recovers): two agent steps,
    an execute_python and a return_result both appear."""
    fake = FakeLLMClient(
        scripted_responses=[
            _resp(
                tool_calls=[
                    _exec_tool_call("raise RuntimeError('first attempt fails')", call_id="call_e1")
                ]
            ),
            _resp(
                tool_calls=[
                    _exec_tool_call("x = 7", call_id="call_e2"),
                    _ret_tool_call(7, call_id="call_ret"),
                ]
            ),
        ]
    )
    agent = _CodeActAgent(llm=fake)
    out = tmp_path / "t.json"
    async with atif_scope(
        agent, path=out, session_id="s2", agent_name="_CodeActAgent", agent_version="0.1.0"
    ):
        assert await agent.run("recover from error") == 7

    traj = _loaded(out)
    agent_steps = _by_source(traj, "agent")
    assert len(agent_steps) == 2
    fn_names = {fn for s in agent_steps for fn in _all_tool_calls(s).values()}
    assert "execute_python" in fn_names
    assert "return_result" in fn_names


@pytest.mark.asyncio
async def test_predict_strategy_isolated(tmp_path: Path) -> None:
    """PredictStrategy alone: a single agent step, no tool_calls, the
    structured return value lands in the step message."""
    fake = FakeLLMClient(scripted_responses=[_resp(content='{"value": "pos"}')])
    agent = _ClassifierAgent(llm=fake)
    out = tmp_path / "t.json"
    async with atif_scope(
        agent, path=out, session_id="s3", agent_name="_ClassifierAgent", agent_version="0.1.0"
    ):
        assert await agent.classify("great") == "pos"

    traj = _loaded(out)
    agent_steps = _by_source(traj, "agent")
    assert len(agent_steps) == 1
    assert not agent_steps[0].tool_calls
    assert "pos" in (agent_steps[0].message or "")
    assert not (traj.subagent_trajectories or [])


@pytest.mark.asyncio
async def test_codeact_calls_predict_flattened(tmp_path: Path) -> None:
    """Case B: same-agent nested generation is flattened into the parent
    trajectory (no subagent_trajectories); the inner step carries
    parent_generation_id in extra."""
    fake = FakeLLMClient(
        scripted_responses=[
            _resp(
                tool_calls=[
                    _exec_tool_call("r = await self.inner_classify('hi')", call_id="call_e1"),
                    _ret_tool_call("a", call_id="call_ret1"),
                ]
            ),
            _resp(content='{"value": "a"}'),
        ]
    )
    agent = _NestedAgent(llm=fake)
    out = tmp_path / "t.json"
    async with atif_scope(
        agent, path=out, session_id="s4", agent_name="_NestedAgent", agent_version="0.1.0"
    ):
        assert await agent.outer("hi") == "a"

    traj = _loaded(out)
    # Flattened, not embedded.
    assert not (traj.subagent_trajectories or [])
    # The nested generation appears as its own step linked by parent_generation_id.
    nested = [s for s in traj.steps if s.extra and s.extra.get("parent_generation_id")]
    assert nested, "expected a flattened nested step carrying parent_generation_id"


@pytest.mark.asyncio
async def test_standalone_generation_subagent(tmp_path: Path) -> None:
    """Case C: a standalone @strategy function called from CodeAct is
    embedded under subagent_trajectories[], with a resolvable ref on the
    parent step's observation."""
    fake = FakeLLMClient(
        scripted_responses=[
            _resp(
                tool_calls=[
                    _exec_tool_call("r = await _standalone_classify('great')", call_id="call_s"),
                    _ret_tool_call("pos", call_id="call_ret_s"),
                ]
            ),
            _resp(content='{"value": "pos"}'),
        ]
    )
    agent = _StandaloneCaller(llm=fake)
    out = tmp_path / "t.json"
    async with atif_scope(
        agent, path=out, session_id="s5", agent_name="_StandaloneCaller", agent_version="0.1.0"
    ):
        assert await agent.run("classify") == "pos"

    traj = _loaded(out)
    children = traj.subagent_trajectories or []
    assert len(children) == 1
    child_ids = {c.trajectory_id for c in children}
    # Every ref on the parent resolves to an embedded child.
    refs = [ref for s in traj.steps for ref in _subagent_refs(s)]
    assert refs and set(refs) <= child_ids
    # The child is the standalone function's own trajectory.
    assert children[0].agent.name == "_standalone_classify"
    assert _by_source(children[0], "agent")


@pytest.mark.asyncio
async def test_async_gather_standalones(tmp_path: Path) -> None:
    """Case D.2: asyncio.gather over a standalone fans out to N embedded
    child trajectories, all referenced from the parent observation."""
    fake = FakeLLMClient(
        scripted_responses=[
            _resp(
                tool_calls=[
                    _exec_tool_call(
                        "rs = await asyncio.gather("
                        "*(_standalone_gather_one(t) for t in ['x','y','z']))",
                        call_id="call_g",
                    ),
                    _ret_tool_call(["pos", "neg", "neu"], call_id="call_ret_g"),
                ]
            ),
            _resp(content='{"value": "pos"}'),
            _resp(content='{"value": "neg"}'),
            _resp(content='{"value": "neu"}'),
        ]
    )

    class _GatherCaller(Agent, llm=fake):
        @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=5)))
        async def run(self, prompts: list[str]) -> list[str]:
            """gather over {prompts}"""
            ...

    agent = _GatherCaller(llm=fake)
    out = tmp_path / "t.json"
    async with atif_scope(
        agent, path=out, session_id="s6", agent_name="_GatherCaller", agent_version="0.1.0"
    ):
        assert await agent.run(["x", "y", "z"]) == ["pos", "neg", "neu"]

    traj = _loaded(out)
    children = traj.subagent_trajectories or []
    assert len(children) == 3
    child_ids = {c.trajectory_id for c in children}
    refs = [ref for s in traj.steps for ref in _subagent_refs(s)]
    assert len(refs) == 3
    assert set(refs) <= child_ids


@pytest.mark.asyncio
async def test_codeact_inline_return_result(tmp_path: Path) -> None:
    """Inline return_result(...) from generated Python surfaces as a
    synthetic tool_call (extra.synthetic=True) with a paired observation —
    the inline-return visibility fix."""
    fake = FakeLLMClient(
        scripted_responses=[
            _resp(
                tool_calls=[
                    _exec_tool_call(
                        "total = 0.75 + 0.50\nreturn_result(total)", call_id="call_inline"
                    )
                ]
            )
        ]
    )

    class _InlineAgent(Agent, llm=_TEST_LLM_DEFAULT):
        @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=5)))
        async def run(self, prompt: str) -> float:
            """{prompt}"""
            ...

    agent = _InlineAgent(llm=fake)
    out = tmp_path / "t.json"
    async with atif_scope(
        agent, path=out, session_id="s-inline", agent_name="_InlineAgent", agent_version="0.1.0"
    ):
        assert await agent.run("compute total") == 1.25

    traj = _loaded(out)
    agent_step = _by_source(traj, "agent")[0]
    synth = [
        tc
        for tc in (agent_step.tool_calls or [])
        if tc.function_name == "return_result" and (tc.extra or {}).get("synthetic") is True
    ]
    assert synth, "inline return_result should appear as a synthetic tool_call"
    # The synthetic tool_call has a paired observation (visibility, not an orphan).
    assert _observation_content(agent_step, synth[0].tool_call_id) is not None


@pytest.mark.asyncio
async def test_nested_subagent(tmp_path: Path) -> None:
    """Closest in-repo 'agent-as-tool' analog: a standalone @strategy
    function embeds as a subagent (Case C)."""
    fake = FakeLLMClient(
        scripted_responses=[
            _resp(
                tool_calls=[
                    _exec_tool_call("r = await _standalone_classify('great')", call_id="call_n"),
                    _ret_tool_call("pos", call_id="call_n_ret"),
                ]
            ),
            _resp(content='{"value": "pos"}'),
        ]
    )
    agent = _StandaloneCaller(llm=fake)
    out = tmp_path / "t.json"
    async with atif_scope(
        agent, path=out, session_id="s10", agent_name="_StandaloneCaller", agent_version="0.1.0"
    ):
        assert await agent.run("classify nested") == "pos"

    traj = _loaded(out)
    children = traj.subagent_trajectories or []
    assert len(children) == 1
    refs = {ref for s in traj.steps for ref in _subagent_refs(s)}
    assert refs <= {c.trajectory_id for c in children}


# ---------------------------------------------------------------------------
# Synthetic-event scenarios (multimodal / compaction / crash)
# ---------------------------------------------------------------------------


_TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M8AAAICAQCRz3qXAAAAAElFTkSuQmCC"
)


def test_multimodal_input(tmp_path: Path) -> None:
    """Task.images render as a ContentPart[] message with an image part, and
    the image bytes are written next to the trajectory."""
    exporter = AtifExporter(
        path=tmp_path / "t.json",
        session_id="s7",
        agent_name="MmAgent",
        agent_version="0.1.0",
    )
    exporter.on_system_prompt(SystemPrompt(content="You are a vision agent.", generation_id=""))
    exporter.on_task(
        Task(
            prompt="Describe this picture.",
            images=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{_TINY_PNG_BASE64}"},
                }
            ],
        )
    )
    exporter.on_before_turn(
        BeforeTurn(
            method_name="run",
            strategy="CodeActStrategy",
            generation_id="gen-1",
            parent_generation_id=None,
            turn_number=1,
        )
    )
    exporter.on_llm_complete(
        LLMComplete(
            model_name="fake-model", prompt_tokens=10, completion_tokens=4, generation_id="gen-1"
        )
    )
    exporter.on_llm_output(LLMOutput(content="A 1x1 transparent pixel."))
    exporter.on_after_turn(
        AfterTurn(
            method_name="run",
            strategy="CodeActStrategy",
            generation_id="gen-1",
            parent_generation_id=None,
            turn_number=1,
            is_final=True,
            success=True,
        )
    )

    traj = _loaded(exporter.path)
    user_step = _by_source(traj, "user")[0]
    assert isinstance(user_step.message, list)
    image_parts = [cp for cp in user_step.message if cp.type == "image"]
    assert image_parts and image_parts[0].source is not None
    img_path = tmp_path / image_parts[0].source.path
    assert img_path.exists()
    assert img_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_compaction_boundary(tmp_path: Path) -> None:
    """A Summary event emits a system compaction step with boundary=replace
    and marks every prior step is_copied_context=True; post-boundary steps
    are not marked."""
    exporter = AtifExporter(
        path=tmp_path / "t.json",
        session_id="s8",
        agent_name="CompactAgent",
        agent_version="0.1.0",
    )
    exporter.on_system_prompt(SystemPrompt(content="You are a compaction agent.", generation_id=""))
    exporter.on_task(Task(prompt="long task"))
    exporter.on_before_turn(
        BeforeTurn(
            method_name="run",
            strategy="CodeActStrategy",
            generation_id="gen-1",
            parent_generation_id=None,
            turn_number=1,
        )
    )
    exporter.on_llm_complete(
        LLMComplete(
            model_name="fake-model",
            prompt_tokens=100,
            completion_tokens=10,
            cost_usd=0.0001,
            tool_calls=[
                {
                    "tool_call_id": "call_a",
                    "function_name": "execute_python",
                    "arguments": json.dumps({"code": "x = 1"}),
                }
            ],
            generation_id="gen-1",
        )
    )
    exporter.on_llm_output(LLMOutput(content=""))
    exporter.on_python_output(
        PythonOutput(
            tool_call_id="call_a",
            execution_count=1,
            stdout="",
            stderr="",
            execution_status=ResultStatus.COMPLETE,
        )
    )
    exporter.on_after_turn(
        AfterTurn(
            method_name="run",
            strategy="CodeActStrategy",
            generation_id="gen-1",
            parent_generation_id=None,
            turn_number=1,
            is_final=False,
            success=True,
        )
    )
    exporter.on_summary(
        Summary(
            summary_tag="2..3",
            replaced_range=(2, 3),
            summary_text="Earlier steps computed x=1.",
            children_tags=["2", "3"],
        )
    )
    exporter.on_before_turn(
        BeforeTurn(
            method_name="run",
            strategy="CodeActStrategy",
            generation_id="gen-2",
            parent_generation_id=None,
            turn_number=2,
        )
    )
    exporter.on_llm_complete(
        LLMComplete(
            model_name="fake-model",
            prompt_tokens=50,
            completion_tokens=5,
            cost_usd=0.00005,
            tool_calls=[
                {
                    "tool_call_id": "call_b",
                    "function_name": "return_result",
                    "arguments": json.dumps({"result": 1}),
                }
            ],
            generation_id="gen-2",
        )
    )
    exporter.on_llm_output(LLMOutput(content=""))
    exporter.on_after_turn(
        AfterTurn(
            method_name="run",
            strategy="CodeActStrategy",
            generation_id="gen-2",
            parent_generation_id=None,
            turn_number=2,
            is_final=True,
            success=True,
        )
    )

    traj = _loaded(exporter.path)
    compaction = next(
        s
        for s in traj.steps
        if s.source == "system" and s.extra and "context_management" in s.extra
    )
    assert compaction.extra["context_management"]["boundary"] == "replace"
    for s in traj.steps:
        if s.step_id < compaction.step_id:
            assert s.is_copied_context is True
    post = [s for s in traj.steps if s.step_id > compaction.step_id]
    assert post and post[0].is_copied_context is None


def test_crashed_mid_turn(tmp_path: Path) -> None:
    """A crash before completion writes the crashed marker, omits
    final_metrics, and keeps the (bounded) steps captured so far."""
    exporter = AtifExporter(
        path=tmp_path / "t.json",
        session_id="s9",
        agent_name="CrashAgent",
        agent_version="0.1.0",
    )
    exporter.on_task(Task(prompt="risky"))
    exporter.on_before_turn(
        BeforeTurn(
            method_name="run",
            strategy="CodeActStrategy",
            generation_id="gen-1",
            parent_generation_id=None,
            turn_number=1,
        )
    )
    exporter.finalize_on_exception(RuntimeError("boom"))

    traj = _loaded(exporter.path)
    assert traj.extra is not None
    assert traj.extra["crashed"] is True
    assert traj.extra["exception_type"] == "RuntimeError"
    assert traj.final_metrics is None
    # Bounded loss: the user task was flushed; the unfinished agent turn is not on disk.
    assert [s.source for s in traj.steps] == ["user"]
