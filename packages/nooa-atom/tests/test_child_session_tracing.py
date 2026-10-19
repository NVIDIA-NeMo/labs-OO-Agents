# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A child session's spans land in its parent's trace, under the cell that created it.

Each session's turn loop runs in a fresh context, so without the registry's
``trace_under_caller`` hook a child's turns emitted no spans at all. The
tracer provider is a real one with an in-memory exporter, set up as the
nooa tracing tests do it; the models are scripted fakes.
"""

import asyncio

import pytest
from atom_test_agents import ATOM_SPEC, AtomModels, cell
from nooa_atom.hosts.headless import _trace_turns
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore
from nooa_atom.session.tracing import (
    DEPTH_ATTR,
    PARENT_ID_ATTR,
    SESSION_ID_ATTR,
    SESSION_NAME_ATTR,
)

from nooa.interactive import Done

pytest.importorskip("opentelemetry.sdk.trace")

TIMEOUT = 30

CHILD_RESULT = (
    "return_result(Done(explanation='fixed', result=TaskResult("
    "solution_description='s', evidence='e', how_to_verify='h', report='REPORT')))"
)


@pytest.fixture
def spans():
    """A real tracer provider with an in-memory exporter and the tracing hooks enabled."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import nooa.tracing as tracing
    from nooa.runtime.hooks import set_hooks
    from nooa.tracing._hooks_impl import OpenInferenceHooks
    from nooa.tracing._session_processor import SessionSpanProcessor

    exporter = InMemorySpanExporter()
    # OTel's global TracerProvider cannot be reset once set: reuse it when present.
    provider = trace.get_tracer_provider()
    if not hasattr(provider, "add_span_processor"):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    provider.add_span_processor(SessionSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    saved = tracing._hooks
    # What enable_tracing keeps for register_hooks_in_current_context.
    tracing._hooks = OpenInferenceHooks(tracer=provider.get_tracer(__name__))
    try:
        yield exporter
    finally:
        tracing._hooks = saved
        set_hooks(None)
        tracing.set_session(None)
        exporter.clear()


@pytest.fixture
def atom_models():
    return AtomModels()


@pytest.fixture
async def atom_registry(sessions_dir, atom_models):
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=atom_models)
    yield registry
    await registry.close_all()


@pytest.fixture
def atom_options(tmp_path, sessions_dir):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return SessionOptions(workspace=workspace, agent_spec=ATOM_SPEC, sessions_dir=sessions_dir)


async def _run_root(registry, options):
    """Create a root traced the way the headless host traces it and run one prompt."""
    root = await registry.create(options, prepare=_trace_turns)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    return root, outcome


def _cell_span(finished, code_fragment):
    """The one code_execution span whose code contains ``code_fragment``."""
    [span] = [
        s
        for s in finished
        if s.name == "code_execution" and code_fragment in str(s.attributes.get("input.value"))
    ]
    return span


def _of_session(finished, name):
    return [s for s in finished if s.attributes.get(SESSION_NAME_ATTR) == name]


def _ancestors(span, by_id):
    seen = []
    parent = span.parent
    while parent is not None and parent.span_id in by_id:
        seen.append(parent.span_id)
        parent = by_id[parent.span_id].parent
    return seen


def _assert_nested(child_spans, under, root_id, finished):
    by_id = {s.context.span_id: s for s in finished}
    names = {s.name for s in child_spans}
    # The child's AGENT (method), GENERATION and code_execution spans all exist.
    assert {"generation", "code_execution"} <= names, names
    assert any(n.startswith("method.") for n in names), names
    for span in child_spans:
        assert span.context.trace_id == under.context.trace_id, span.name
        assert span.attributes.get("session.id") == root_id, span.name
        assert under.context.span_id in _ancestors(span, by_id), span.name


async def test_a_delegated_childs_spans_nest_under_the_parents_cell(
    spans, atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        cell("done = await self.delegate('Fix', 'fix it')\nreturn_result(Done(explanation='ok'))")
    ]
    atom_models.scripts["Fix"] = [cell(CHILD_RESULT)]
    root, outcome = await _run_root(atom_registry, atom_options)
    assert outcome == Done(explanation="ok")
    finished = spans.get_finished_spans()
    under = _cell_span(finished, "self.delegate('Fix'")
    child_spans = _of_session(finished, "Fix")
    _assert_nested(child_spans, under, root.id, finished)
    [child] = atom_registry.children(root.id)
    for span in child_spans:
        assert span.attributes[SESSION_ID_ATTR] == child.id
        assert span.attributes[PARENT_ID_ATTR] == root.id
        assert span.attributes[DEPTH_ATTR] == 1
    # The root's own spans are not marked as a child's.
    assert under.attributes.get(SESSION_ID_ATTR) is None
    assert under.attributes.get("session.id") == root.id


async def test_a_kept_childs_follow_up_turn_stays_in_the_roots_trace(
    spans, atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        cell(
            "self.v.ref = await self.spawn('Kept', 'first task')\n"
            "return_result(Waiting(explanation='child working', on=['delegates']))"
        ),
        cell(
            "[item] = notification['delegates']\n"
            "await self.v.ref.send('follow-up')\n"
            "return_result(Waiting(explanation='follow-up', on=['delegates']))"
        ),
        cell("[item] = notification['delegates']\nreturn_result(Done(explanation='both'))"),
    ]
    atom_models.scripts["Kept"] = [cell(CHILD_RESULT), cell(CHILD_RESULT)]
    root, outcome = await _run_root(atom_registry, atom_options)
    assert outcome == Done(explanation="both")
    finished = spans.get_finished_spans()
    spawn_cell = _cell_span(finished, "self.spawn('Kept'")
    child_spans = _of_session(finished, "Kept")
    # Two turns of the kept child, both in the root's trace, both under the
    # spawn cell: its loop context was set up once, when it was created.
    assert len([s for s in child_spans if s.name == "method.handle_work"]) == 2
    _assert_nested(child_spans, spawn_cell, root.id, finished)


async def test_a_grandchilds_spans_nest_under_the_childs_cell(
    spans, atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        cell("done = await self.delegate('Mid', 'go on')\nreturn_result(Done(explanation='ok'))")
    ]
    atom_models.scripts["Mid"] = [
        cell(
            "done = await self.delegate('Leaf', 'leaf work')\n"
            "return_result(Done(explanation='mid', result=TaskResult("
            "solution_description='s', evidence='e', how_to_verify='h', report='MID')))"
        )
    ]
    atom_models.scripts["Leaf"] = [cell(CHILD_RESULT)]
    root, outcome = await _run_root(atom_registry, atom_options)
    assert outcome == Done(explanation="ok")
    finished = spans.get_finished_spans()
    root_cell = _cell_span(finished, "self.delegate('Mid'")
    mid_cell = _cell_span(finished, "self.delegate('Leaf'")
    leaf_spans = _of_session(finished, "Leaf")
    _assert_nested(_of_session(finished, "Mid"), root_cell, root.id, finished)
    _assert_nested(leaf_spans, mid_cell, root.id, finished)
    _assert_nested(leaf_spans, root_cell, root.id, finished)
    [mid] = atom_registry.children(root.id)
    for span in leaf_spans:
        assert span.attributes[DEPTH_ATTR] == 2
        assert span.attributes[PARENT_ID_ATTR] == mid.id


async def test_with_tracing_off_children_run_and_emit_no_spans(
    atom_registry, atom_options, atom_models, monkeypatch
):
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import nooa.tracing as tracing

    # Tracing off: enable_tracing never ran, so there are no hooks to register.
    monkeypatch.setattr(tracing, "_hooks", None)
    exporter = InMemorySpanExporter()
    provider = trace.get_tracer_provider()
    if hasattr(provider, "add_span_processor"):
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    atom_models.scripts[None] = [
        cell("done = await self.delegate('Fix', 'fix it')\nreturn_result(Done(explanation='ok'))")
    ]
    atom_models.scripts["Fix"] = [cell(CHILD_RESULT)]
    root, outcome = await _run_root(atom_registry, atom_options)
    assert outcome == Done(explanation="ok")
    assert exporter.get_finished_spans() == ()


async def test_without_nooa_tracing_children_still_run(
    atom_registry, atom_options, atom_models, monkeypatch
):
    import sys

    # An import of nooa.tracing fails as it does when OpenTelemetry is not installed.
    monkeypatch.setitem(sys.modules, "nooa.tracing", None)
    atom_models.scripts[None] = [
        cell("done = await self.delegate('Fix', 'fix it')\nreturn_result(Done(explanation='ok'))")
    ]
    atom_models.scripts["Fix"] = [cell(CHILD_RESULT)]
    root, outcome = await _run_root(atom_registry, atom_options)
    assert outcome == Done(explanation="ok")
