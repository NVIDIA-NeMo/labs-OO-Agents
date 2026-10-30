# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A plain Gym adapter (no benchmark keys) runs AtomHarness without the policy guard."""

from __future__ import annotations

import json

import pytest
from atom_test_agents import CellLLM, cell
from nooa_atom.benchmark_policy import POLICY
from nooa_atom.gym import AtomHarness

from nooa.runtime.hooks import hooks_scope

FINISH = (
    "open('hello.txt', 'w').write('hello gym')\n"
    "return_result(Done(explanation='wrote the file', result=TaskResult("
    "solution_description='created hello.txt', evidence='file exists', "
    "how_to_verify='cat hello.txt', report='done')))"
)


@pytest.fixture(autouse=True)
def no_optional_viewer_probe(monkeypatch):
    monkeypatch.setattr("nooa.agent._auto_tracing_attempted", True)


class _Hooks:
    """Records code executions, as a Gym observer would."""

    def __init__(self) -> None:
        self.cells: list[str] = []

    def before_code_execution(self, *, code: str, **_: object) -> dict:
        self.cells.append(code)
        return {}

    def after_code_execution(self, **_: object) -> None: ...

    def before_agent_call(self, **_: object) -> None: ...

    def after_agent_call(self, **_: object) -> None: ...

    def before_tool_execution(self, **_: object) -> None: ...

    def after_tool_execution(self, **_: object) -> None: ...

    def before_generation(self, **_: object) -> None: ...

    def after_generation(self, **_: object) -> None: ...

    def before_method_invocation(self, **_: object) -> None: ...


async def test_plain_client_without_policy_hooks_solves_with_defaults(tmp_path):
    target = repr(str(tmp_path / "hello.txt"))
    llm = CellLLM([cell(FINISH.replace("'hello.txt'", target))])
    assert not hasattr(llm, "bind_episode_policy")
    harness = AtomHarness(llm, turn_method="handle_batch", workdir=tmp_path)
    assert harness.benchmark_guard is None
    assert harness.options.experimental_done_reflection is False
    assert harness.options.benchmark_policy is None
    hooks = _Hooks()
    with hooks_scope(hooks):
        result = await harness.solve("write hello.txt")
    assert result.report == "done"
    assert (tmp_path / "hello.txt").read_text() == "hello gym"
    assert harness.benchmark_policy_receipt() is None
    assert len(llm.calls) == 1
    # The outer hooks still see Atom's code executions in the session loop.
    assert any("hello.txt" in code for code in hooks.cells)
    assert not list(tmp_path.glob("*.atom-policy.json"))


async def test_plain_harness_runs_one_episode(tmp_path):
    target = repr(str(tmp_path / "hello.txt"))
    harness = AtomHarness(CellLLM([cell(FINISH.replace("'hello.txt'", target))]), workdir=tmp_path)
    await harness.solve("once")
    with pytest.raises(RuntimeError, match="one episode"):
        await harness.solve("twice")


async def test_plain_harness_traces_the_root_session_under_the_harness_call(tmp_path):
    """The root's turn, generation and LLM-call spans join the harness's trace and session."""
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import nooa.tracing as tracing
    from nooa.runtime.hooks import set_hooks
    from nooa.tracing._hooks_impl import OpenInferenceHooks
    from nooa.tracing._session_processor import SessionSpanProcessor

    exporter = InMemorySpanExporter()
    provider = trace.get_tracer_provider()
    if not hasattr(provider, "add_span_processor"):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    provider.add_span_processor(SessionSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    saved = tracing._hooks
    tracing._hooks = OpenInferenceHooks(tracer=provider.get_tracer(__name__))
    try:
        tracing.set_session("episode-1")
        tracing.register_hooks_in_current_context()
        target = repr(str(tmp_path / "hello.txt"))
        harness = AtomHarness(
            CellLLM([cell(FINISH.replace("'hello.txt'", target))]), workdir=tmp_path
        )
        await harness.solve("write hello.txt")
        finished = exporter.get_finished_spans()
    finally:
        tracing._hooks = saved
        set_hooks(None)
        tracing.set_session(None)
        exporter.clear()
    by_id = {s.context.span_id: s for s in finished}
    [solve] = [s for s in finished if s.name == "method.solve"]
    handles = [s for s in finished if s.name.startswith("method.handle")]
    assert handles, sorted({s.name for s in finished})
    handle = handles[0]
    [llm_call] = [s for s in finished if s.name == "llm_call"]
    for span in (handle, llm_call):
        assert span.context.trace_id == solve.context.trace_id, span.name
        assert span.attributes.get("session.id") == "episode-1", span.name
    ancestors = []
    parent = handle.parent
    while parent is not None and parent.span_id in by_id:
        ancestors.append(by_id[parent.span_id].name)
        parent = by_id[parent.span_id].parent
    assert "method.solve" in ancestors
    assert tuple(llm_call.attributes["nooa.llm.tool_calls.names"]) == ("python_cell",)


@pytest.mark.parametrize(
    "keys",
    [
        {"benchmark_policy": POLICY},
        {"experimental_done_reflection": False},
        {"benchmark_allow_session_titles": False},
    ],
)
def test_partial_benchmark_configuration_without_receipt_path_fails_closed(tmp_path, keys):
    with pytest.raises(ValueError):
        AtomHarness(CellLLM([]), workdir=tmp_path, **keys)


def test_receipt_path_with_plain_client_fails_closed_and_records_failure(tmp_path):
    path = tmp_path / "episode.atom-policy.json"
    with pytest.raises(ValueError, match="instrumented Gym policy client"):
        AtomHarness(
            CellLLM([]),
            workdir=tmp_path,
            benchmark_policy=POLICY,
            experimental_done_reflection=False,
            benchmark_allow_session_titles=False,
            benchmark_policy_path=path,
        )
    receipt = json.loads(path.read_text())
    assert receipt["phase"] == "final" and receipt["verdict"] == "fail"


@pytest.mark.parametrize(
    "key", ["experimental_namespace", "experimental_unknown", "benchmark_typo"]
)
def test_plain_harness_rejects_unknown_experiment_keys(tmp_path, key):
    with pytest.raises(ValueError, match="unknown benchmark/experiment keys"):
        AtomHarness(CellLLM([]), workdir=tmp_path, **{key: "session"})


def test_plain_harness_ignores_other_harness_supplied_arguments(tmp_path):
    harness = AtomHarness(CellLLM([]), workdir=tmp_path, run_command=object())
    assert harness.benchmark_guard is None
