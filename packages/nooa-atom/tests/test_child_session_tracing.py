# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A child session's spans land in its parent's trace, as a separate agent.

The child's turns nest under the ``delegate()``/``spawn()`` call that
created it, and its agent's spans are named after the session
(``Fix.handle_work``), as a sub-agent called from a cell would be.

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


def _methods(finished, method, session_name=None):
    """The ``method.<method>`` spans of one session (``None``: the root's own)."""
    return [
        s
        for s in finished
        if s.name == f"method.{method}" and s.attributes.get(SESSION_NAME_ATTR) == session_name
    ]


def _parent(span, finished):
    [parent] = [s for s in finished if s.context.span_id == span.parent.span_id]
    return parent


def _assert_turns_under(finished, session_name, call, count=1):
    """``session_name``'s turns (and its close) have ``call``'s span as direct parent."""
    turns = _methods(finished, "handle_work", session_name)
    assert len(turns) == count, [s.name for s in _of_session(finished, session_name)]
    for span in turns + _methods(finished, "aclose", session_name):
        assert span.parent.span_id == call.context.span_id, span.name


def _assert_named(child_spans, name):
    """The child's method, generation, code and tool spans name the child as their agent."""
    kinds = ("method.", "generation", "code_execution", "tool_execution")
    named = [s for s in child_spans if s.name.startswith(kinds)]
    assert {s.name.split(".")[0] for s in named} >= {"method", "generation", "code_execution"}
    for span in named:
        assert span.attributes.get("agent.name") == name, span.name


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


async def test_a_delegated_child_is_its_own_agent_under_the_delegate_call(
    spans, atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        cell(
            "done = await self.delegate('fix', 'fix it', name='fix')\nreturn_result(Done(explanation='ok'))"
        )
    ]
    atom_models.scripts["fix"] = [cell(CHILD_RESULT)]
    root, outcome = await _run_root(atom_registry, atom_options)
    assert outcome == Done(explanation="ok")
    finished = spans.get_finished_spans()
    under = _cell_span(finished, "self.delegate('fix'")
    child_spans = _of_session(finished, "fix")
    _assert_nested(child_spans, under, root.id, finished)
    [delegate] = _methods(finished, "delegate")
    assert _parent(delegate, finished) is under
    _assert_turns_under(finished, "fix", delegate)
    _assert_named(child_spans, "fix")
    # The root keeps its class name.
    root_spans = [s for s in finished if s.attributes.get(SESSION_NAME_ATTR) is None]
    assert {s.attributes.get("agent.name") for s in root_spans} - {None} == {"AtomAgent"}
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
            "self.v.ref = await self.spawn('kept', 'first task', name='kept')\n"
            "return_result(Waiting(explanation='child working', on=['delegates']))"
        ),
        cell(
            "[item] = notification['delegates']\n"
            "await self.v.ref.send('follow-up')\n"
            "return_result(Waiting(explanation='follow-up', on=['delegates']))"
        ),
        cell("[item] = notification['delegates']\nreturn_result(Done(explanation='both'))"),
    ]
    atom_models.scripts["kept"] = [cell(CHILD_RESULT), cell(CHILD_RESULT)]
    root, outcome = await _run_root(atom_registry, atom_options)
    assert outcome == Done(explanation="both")
    finished = spans.get_finished_spans()
    spawn_cell = _cell_span(finished, "self.spawn('kept'")
    child_spans = _of_session(finished, "kept")
    # Two turns of the kept child, both in the root's trace, both under the
    # spawn cell: its loop context was set up once, when it was created.
    assert len([s for s in child_spans if s.name == "method.handle_work"]) == 2
    _assert_nested(child_spans, spawn_cell, root.id, finished)
    # Directly under the spawn call, the follow-up turn too (it ran after
    # that call returned, in the loop context set up at creation).
    [spawn] = _methods(finished, "spawn")
    assert _parent(spawn, finished) is spawn_cell
    _assert_turns_under(finished, "kept", spawn, count=2)
    _assert_named(child_spans, "kept")


async def test_a_grandchild_nests_under_the_childs_delegate_call(
    spans, atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        cell(
            "done = await self.delegate('mid', 'go on', name='mid')\nreturn_result(Done(explanation='ok'))"
        )
    ]
    atom_models.scripts["mid"] = [
        cell(
            "done = await self.delegate('leaf', 'leaf work', name='leaf')\n"
            "return_result(Done(explanation='mid', result=TaskResult("
            "solution_description='s', evidence='e', how_to_verify='h', report='MID')))"
        )
    ]
    atom_models.scripts["leaf"] = [cell(CHILD_RESULT)]
    root, outcome = await _run_root(atom_registry, atom_options)
    assert outcome == Done(explanation="ok")
    finished = spans.get_finished_spans()
    root_cell = _cell_span(finished, "self.delegate('mid'")
    mid_cell = _cell_span(finished, "self.delegate('leaf'")
    leaf_spans = _of_session(finished, "leaf")
    _assert_nested(_of_session(finished, "mid"), root_cell, root.id, finished)
    _assert_nested(leaf_spans, mid_cell, root.id, finished)
    _assert_nested(leaf_spans, root_cell, root.id, finished)
    [root_delegate] = _methods(finished, "delegate")
    [mid_delegate] = _methods(finished, "delegate", "mid")
    assert _parent(mid_delegate, finished) is mid_cell
    assert mid_delegate.attributes["agent.name"] == "mid"
    _assert_turns_under(finished, "mid", root_delegate)
    _assert_turns_under(finished, "leaf", mid_delegate)
    _assert_named(_of_session(finished, "mid"), "mid")
    _assert_named(leaf_spans, "leaf")
    [mid] = atom_registry.children(root.id)
    for span in leaf_spans:
        assert span.attributes[DEPTH_ATTR] == 2
        assert span.attributes[PARENT_ID_ATTR] == mid.id


async def test_an_ordinary_agent_called_in_a_childs_cell_keeps_its_class_name(
    spans, atom_registry, atom_options, atom_models
):
    atom_models.scripts[None] = [
        cell(
            "done = await self.delegate('fix', 'fix it', name='fix')\nreturn_result(Done(explanation='ok'))"
        )
    ]
    atom_models.scripts["fix"] = [
        cell(
            "from nooa import Agent\n"
            "class Helper(Agent):\n"
            "    async def ping(self) -> int:\n"
            "        return 1\n"
            "assert await Helper().ping() == 1\n" + CHILD_RESULT
        )
    ]
    root, outcome = await _run_root(atom_registry, atom_options)
    assert outcome == Done(explanation="ok")
    finished = spans.get_finished_spans()
    [ping] = _methods(finished, "ping", "fix")
    assert ping.attributes["agent.name"] == "Helper"
    assert _parent(ping, finished).name == "code_execution"
    assert _parent(ping, finished).attributes["agent.name"] == "fix"


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
        cell(
            "done = await self.delegate('fix', 'fix it', name='fix')\nreturn_result(Done(explanation='ok'))"
        )
    ]
    atom_models.scripts["fix"] = [cell(CHILD_RESULT)]
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
        cell(
            "done = await self.delegate('fix', 'fix it', name='fix')\nreturn_result(Done(explanation='ok'))"
        )
    ]
    atom_models.scripts["fix"] = [cell(CHILD_RESULT)]
    root, outcome = await _run_root(atom_registry, atom_options)
    assert outcome == Done(explanation="ok")


async def _root_and_closed_child(registry, options, models, spans, name="Kid"):
    # The root's later turns read the child's results on ``delegates``.
    models.scripts[None] = [cell("return_result(Done(explanation='root'))")] * 3
    models.scripts[name] = [cell(CHILD_RESULT)]
    root, _ = await _run_root(registry, options)
    child = await registry.create(root.options.inherit(name=name, kept=True), parent_id=root.id)
    await registry.close(child.id)
    assert registry.get(child.id) is None and registry.get(root.id) is root
    spans.clear()
    return root, child.id


async def _run_child(child):
    outcome = await asyncio.wait_for(child.prompt("work"), TIMEOUT)
    assert outcome is not None
    await child.close()


async def test_a_direct_load_of_a_child_with_a_live_parent_is_traced_in_the_roots_session(
    spans, atom_registry, atom_options, atom_models
):
    from opentelemetry import trace

    import nooa.tracing as tracing

    root, child_id = await _root_and_closed_child(atom_registry, atom_options, atom_models, spans)
    # The loading host has its own span and trace session: neither is captured.
    tracer = trace.get_tracer_provider().get_tracer(__name__)
    tracing.set_session("host-session")
    with tracer.start_as_current_span("host-call") as host_call:
        child = await atom_registry.load(child_id)
    await _run_child(child)
    finished = spans.get_finished_spans()
    child_spans = _of_session(finished, "Kid")
    names = {s.name for s in child_spans}
    # ``prompt`` runs the ``handle`` turn method.
    assert {"method.handle", "generation", "code_execution"} <= names, names
    by_id = {s.context.span_id: s for s in finished}
    for span in child_spans:
        assert span.attributes.get("session.id") == root.id, span.name
        assert span.attributes[SESSION_ID_ATTR] == child_id
        assert span.attributes[PARENT_ID_ATTR] == root.id
        assert span.attributes[DEPTH_ATTR] == 1
        assert span.context.trace_id != host_call.context.trace_id, span.name
        assert host_call.context.span_id not in _ancestors(span, by_id), span.name
    # Its own top-level trace: the turn has no parent span.
    [turn] = _methods(finished, "handle", "Kid")
    assert turn.parent is None
    _assert_named(child_spans, "Kid")


async def test_a_detached_load_of_a_child_is_not_traced(
    spans, atom_registry, atom_options, atom_models
):
    root, child_id = await _root_and_closed_child(atom_registry, atom_options, atom_models, spans)
    await atom_registry.close(root.id)
    child = await atom_registry.load(child_id)
    await _run_child(child)
    assert _of_session(spans.get_finished_spans(), "Kid") == []


async def test_open_child_still_nests_a_reopened_child_under_the_reopening_call(
    spans, atom_registry, atom_options, atom_models
):
    from opentelemetry import trace

    root, child_id = await _root_and_closed_child(atom_registry, atom_options, atom_models, spans)
    tracer = trace.get_tracer_provider().get_tracer(__name__)
    with tracer.start_as_current_span("reopening-call") as call:
        child = await atom_registry.open_child(root, child_id)
    await _run_child(child)
    finished = spans.get_finished_spans()
    [turn] = _methods(finished, "handle", "Kid")
    assert turn.parent.span_id == call.context.span_id
    assert turn.context.trace_id == call.context.trace_id
