# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default session namespace integration with strict fake models, no namespace option."""

import asyncio

import pytest
from atom_test_agents import CellLLM, cell
from nooa_atom.agent.agent import AtomAgent
from nooa_atom.agent.factory import create_session_agent
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa.interactive import Done, NeedInput


async def make_root(tmp_path, responses, *, reflection=False):
    llm = CellLLM(responses)
    registry = SessionRegistry(
        SessionStore(tmp_path / "sessions"), agent_factory=create_session_agent
    )
    options = SessionOptions(
        workspace=tmp_path,
        agent_spec="nooa_atom.agent:AtomAgent",
        llm=llm,
        experimental_done_reflection=reflection,
        experimental_reflection_min_turns=1,
        experimental_reflection_min_seconds=0,
    )
    root = await registry.create(options)
    return registry, root, llm


async def test_kept_session_retains_names_across_done_and_need_input(tmp_path):
    registry, root, llm = await make_root(
        tmp_path,
        [
            cell(
                "counter=0\ndef bump():\n global counter\n counter+=1\n return counter\nreturn_result(Done(explanation=str(bump())))"
            ),
            cell("return_result(NeedInput(question=str(bump())))"),
            cell("return_result(Done(explanation=str(counter)))"),
        ],
    )
    try:
        assert await asyncio.wait_for(root.prompt("one"), 30) == Done(explanation="1")
        assert await asyncio.wait_for(root.prompt("two"), 30) == NeedInput(question="2")
        assert await asyncio.wait_for(root.prompt("three"), 30) == Done(explanation="2")
        ns = root._agent._handle_namespace
        assert ns.globals["bump"].__globals__ is ns.globals
        assert len(llm.calls) == 3
    finally:
        await registry.close_all()
    assert root._agent._handle_namespace is None


@pytest.mark.parametrize("obsolete", ["legacy", "call", "session", None])
def test_namespace_selector_is_rejected_not_ignored(tmp_path, obsolete):
    with pytest.raises(TypeError, match="experimental_namespace.*removed"):
        AtomAgent(cwd=tmp_path, llm=CellLLM(), experimental_namespace=obsolete)
    values = {"workspace": tmp_path, "agent_spec": "nooa_atom.agent:AtomAgent"}
    with pytest.raises(TypeError, match="experimental_namespace.*removed"):
        SessionOptions(**values, experimental_namespace=obsolete)
    options = SessionOptions(**values)
    assert "experimental_namespace" not in options.model_dump()
    with pytest.raises(TypeError, match="experimental_namespace.*removed"):
        options.inherit(experimental_namespace=obsolete)


async def test_managed_background_barrier_inputs_completion_and_output_are_owned(tmp_path):
    registry, root, llm = await make_root(
        tmp_path,
        [
            cell("""
gate = asyncio.Event()
alias = return_result
async def worker():
    await gate.wait()
    import sys
    print('job stdout')
    sys.stderr.write('job stderr')
    show(Image.from_bytes(b'job image', media_type='image/png'))
    try:
        alias(Done(explanation='WRONG'))
    except RuntimeError:
        pass
    else:
        raise AssertionError('completion was not rejected')
    return notification['user_messages'][0]
job = self.jobs.launch(worker(), name='late')
print('cell A')
return_result(Waiting(explanation='working', on=['late']))
"""),
            cell(r"""
gate.set()
value = await job
assert 'A-input' in str(value), value
assert job.output.stdout.startswith('job stdout\n'), job.output.stdout
assert job.output.stderr == 'job stderr', job.output.stderr
assert len(job.output.images) == 1
print('cell B')
return_result(Done(explanation='B-finished'))
"""),
        ],
    )
    try:
        receipt_a = asyncio.create_task(root.prompt("A-input"))
        # Waiting does not settle the receipt; wait until first turn finishes.
        for _ in range(200):
            if root._agent.jobs.get("late") is not None and not root._agent.turns.running:
                break
            await asyncio.sleep(0.01)
        receipt_b = root.prompt("B-input")
        try:
            answer = await asyncio.wait_for(receipt_b, 30)
        except Exception:
            await asyncio.gather(receipt_a, return_exceptions=True)
            pytest.fail(
                str([event.error for event in root._agent.events.query(type="PythonOutput")])
            )
        assert answer == Done(explanation="B-finished")
        assert await asyncio.wait_for(receipt_a, 30) == Done(explanation="B-finished")
        outputs = list(root._agent.events.query(type="PythonOutput"))
        assert all(not output.images for output in outputs)
        text = [output.stdout for output in outputs if not output.metadata.get("prefill")]
        assert text == ["cell A\n", "cell B\n"]
        assert len(llm.calls) == 2
    finally:
        await registry.close_all()


async def test_inventory_is_bounded_without_repr_or_metaclass_callbacks(tmp_path):
    llm = CellLLM(
        [
            cell(
                """
class Hostile:
    def __repr__(self): raise AssertionError('repr')
hostile = Hostile()
"""
                + "\n".join(f"name_{i}=1" for i in range(50))
                + "\nreturn_result(Done(explanation='ok'))"
            )
        ]
    )
    agent = AtomAgent(cwd=tmp_path, llm=llm)
    try:
        try:
            outcome = await agent.handle({"user_messages": ["go"]})
        except Exception:
            pytest.fail(str([event.error for event in agent.events.query(type="PythonOutput")]))
        assert outcome == Done(explanation="ok")

        class HostileMeta(type):
            def __getattribute__(cls, name):
                raise AssertionError("metaclass callback must not run")

        ns = agent._handle_namespace
        ns.globals["aaa_hostile_meta"] = HostileMeta("HostileType", (), {})()
        ns.user_names.add("aaa_hostile_meta")
        inventory = ns.inventory()
        assert len(inventory["names"]) == 20 and inventory["omitted"] >= 30
        assert len(str(inventory)) < 2000
    finally:
        await agent.aclose()


async def test_del_does_not_close_resource_and_session_close_owns_shell(tmp_path):
    registry, root, _ = await make_root(
        tmp_path,
        [
            cell("""
class Resource:
    def close(self): raise AssertionError('del must not close')
resource = Resource()
del resource
shell = Shell()
result = await self.jobs.launch(shell.run('printf shell-owned'), name='command')
assert result.stdout == 'shell-owned'
return_result(Done(explanation='ok'))
""")
        ],
    )
    try:
        assert await asyncio.wait_for(root.prompt("run"), 30) == Done(explanation="ok")
        ns = root._agent._handle_namespace
        assert "resource" not in ns.globals
        shell = ns.globals["shell"]
    finally:
        await registry.close_all()
    assert root._agent._handle_namespace is None
    # The owner has closed its persistent shell, not merely dropped the variable.
    assert shell._session is None
    assert not shell._bash_alive()


async def wait_idle(root):
    async def poll():
        while root._agent.turns.running:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), 10)


def reflection(**fields):
    import json

    from nooa.unifiedllm import LLMResponse, LLMUsage

    return LLMResponse(
        content=json.dumps(fields),
        finish_reason="stop",
        usage=LLMUsage(input_tokens=11, output_tokens=7),
    )


async def test_accepted_done_reflects_archives_and_drops_only_proposed_variable(tmp_path):
    registry, root, llm = await make_root(
        tmp_path,
        [
            cell(
                "scratch=42\nstable=7\ndef helper():\n return stable\nself.context['user_decision']='KEEP VERBATIM'\nprint('TOOL_EVIDENCE_' + str(1700 * 43))\nreturn_result(Done(explanation='accepted result'))"
            ),
            reflection(
                summary="Completed request; result accepted. Evidence is in the original cells.",
                useful_findings="helper returns the stable value",
                decisions="KEEP VERBATIM",
                outstanding_work="none",
                drop_names=["scratch"],
            ),
            cell(
                "assert helper() == 7\nassert self.context['user_decision'] == 'KEEP VERBATIM'\ntry:\n scratch\nexcept NameError:\n return_result(NeedInput(question='retained helper works'))"
            ),
        ],
        reflection=True,
    )
    try:
        outcome = await asyncio.wait_for(root.prompt("complete this"), 30)
        assert outcome == Done(explanation="accepted result")
        await wait_idle(root)
        ns = root._agent._handle_namespace
        assert "scratch" not in ns.globals, (
            ns.user_names,
            ns.cleanup_candidates(ns.cleanup_snapshot()),
            len(llm.calls),
            [event.event_type for event in root._agent.events.query()],
        )
        assert ns.globals["stable"] == 7 and ns.globals["helper"]() == 7
        summaries = list(root._agent.events.query(type="Summary"))
        assert len(summaries) == 1
        summary = summaries[0]
        assert "Completed request" in summary.summary_text
        assert all(root._agent.events[tag] is not None for tag in summary.children_tags)
        assert "KEEP VERBATIM" in root._agent.context["done_reflection_notes"]
        assert root._agent.context["user_decision"] == "KEEP VERBATIM"
        await root.wait_for_checkpoint()
        saved = AtomAgent(cwd=tmp_path, llm=CellLLM())
        try:
            assert root.handle.storage.restore_latest_snapshot(saved)
            assert "KEEP VERBATIM" in saved.context["done_reflection_notes"]
        finally:
            await saved.aclose()
        assert len(llm.calls) == 2 and llm.calls[1].output_model.__name__ == "DoneReflectionPlan"
        assert llm.calls[1].kwargs["max_tokens"] == 2048
        assert "TOOL_EVIDENCE_73100" in str(llm.calls[1].messages)
        assert root.info.usage.input_tokens >= 11 and root.info.usage.output_tokens >= 7
        assert await root.prompt("next") == NeedInput(question="retained helper works")
        await wait_idle(root)
        assert len(llm.calls) == 3  # NeedInput never reflects; no recursion
        await root.wait_for_checkpoint()
        restored = AtomAgent(cwd=tmp_path, llm=CellLLM())
        try:
            assert root.handle.storage.restore_latest_snapshot(restored)
            assert "KEEP VERBATIM" in restored.context["done_reflection_notes"]
            assert restored._handle_namespace is None
        finally:
            await restored.aclose()
    finally:
        await registry.close_all()


@pytest.mark.parametrize(
    "fields",
    [
        {"unexpected": "invalid plan"},
        {"drop_names": ["unobserved"], "summary": "must reject"},
        {"summary": "x" * 4001},
    ],
)
async def test_reflection_failure_rejection_budget_keeps_original_done(tmp_path, fields):
    registry, root, llm = await make_root(
        tmp_path,
        [
            cell("scratch=42\nreturn_result(Done(explanation='original'))"),
            reflection(**fields),
        ],
        reflection=True,
    )
    try:
        accepted = await root.prompt("go")
        assert accepted == Done(explanation="original")
        await wait_idle(root)
        assert root.last_outcome_kind == "done" and root.info.reply_count == 1
        assert root._agent._handle_namespace.globals["scratch"] == 42
        assert len(llm.calls) == 2
        assert not list(root._agent.events.query(type="Summary"))
    finally:
        await registry.close_all()


async def test_reflection_timeout_is_owned_and_original_receipt_precedes_it(tmp_path, monkeypatch):
    from nooa_atom.agent import done_reflection

    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocked(self, request):
        entered.set()
        try:
            await asyncio.sleep(100)
        finally:
            cancelled.set()

    monkeypatch.setattr(AtomAgent, "_reflect_done_plan", blocked)
    monkeypatch.setattr(done_reflection, "REFLECTION_TIMEOUT", 0.05)
    registry, root, llm = await make_root(
        tmp_path,
        [
            cell("scratch=42\nreturn_result(Done(explanation='original'))"),
        ],
        reflection=True,
    )
    try:
        accepted = await root.prompt("go")
        assert accepted == Done(explanation="original")
        await asyncio.wait_for(entered.wait(), 2)
        assert root._agent.turns.running  # maintenance, not another user turn
        await wait_idle(root)
        assert cancelled.is_set()
        assert root.info.reply_count == 1 and len(llm.calls) == 1
        assert root._agent._handle_namespace.globals["scratch"] == 42
    finally:
        await registry.close_all()


async def test_close_during_reflection_cancels_awaits_without_second_settlement(
    tmp_path, monkeypatch
):
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def blocked(self, request):
        entered.set()
        try:
            await asyncio.sleep(100)
        finally:
            cancelled.set()

    monkeypatch.setattr(AtomAgent, "_reflect_done_plan", blocked)
    registry, root, _ = await make_root(
        tmp_path,
        [
            cell("return_result(Done(explanation='delivered'))"),
        ],
        reflection=True,
    )
    assert await root.prompt("go") == Done(explanation="delivered")
    await asyncio.wait_for(entered.wait(), 2)
    await asyncio.wait_for(registry.close_all(), 5)
    assert cancelled.is_set() and root.info.reply_count == 1
    assert root._agent._handle_namespace is None


@pytest.mark.parametrize("change", ["rebind", "active_work", "overlapping_summary", "notes"])
async def test_stale_cleanup_and_context_plans_fail_safe(tmp_path, monkeypatch, change):
    from nooa_atom.agent.done_reflection import DoneReflectionPlan

    async def propose(self, request):
        if change == "rebind":
            self._handle_namespace.globals["scratch"] = 999
        elif change == "active_work":
            self.jobs.launch(asyncio.sleep(100), name="became_active")
        elif change == "overlapping_summary":
            tags = request["completed_event_tags"]
            self.events.collapse(tags[0], tags[-1], "external summary")
        else:
            self.context["done_reflection_notes"] = "newer decision"
        return DoneReflectionPlan(
            summary="maintenance summary", drop_names=["scratch"], useful_findings="stale plan"
        )

    monkeypatch.setattr(AtomAgent, "_reflect_done_plan", propose)
    registry, root, _ = await make_root(
        tmp_path,
        [
            cell("scratch=42\nreturn_result(Done(explanation='original'))"),
        ],
        reflection=True,
    )
    try:
        assert await root.prompt("go") == Done(explanation="original")
        await wait_idle(root)
        assert root._agent._handle_namespace.globals["scratch"] == (
            999 if change == "rebind" else 42
        )
        if change == "notes":
            assert root._agent.context["done_reflection_notes"] == "newer decision"
    finally:
        await registry.close_all()


async def test_arrival_after_reflection_watermark_is_not_collapsed(tmp_path, monkeypatch):
    from nooa_atom.agent.done_reflection import DoneReflectionPlan

    from nooa.events import Error

    arrived = []

    async def propose(self, request):
        arrived.append(self.event_manager.add(Error(content="LATER EVENT MUST REMAIN")))
        return DoneReflectionPlan(summary="completed work", drop_names=["scratch"])

    monkeypatch.setattr(AtomAgent, "_reflect_done_plan", propose)
    registry, root, _ = await make_root(
        tmp_path,
        [
            cell("scratch=42\nreturn_result(Done(explanation='original'))"),
        ],
        reflection=True,
    )
    try:
        assert await root.prompt("go") == Done(explanation="original")
        await wait_idle(root)
        assert arrived[0] in root._agent.events.keys()
        assert root._agent.events[arrived[0]].content == "LATER EVENT MUST REMAIN"
    finally:
        await registry.close_all()


async def test_namespace_nosnapshot_restart_notice_once_and_self_v_survives(tmp_path):
    from nooa.storage.json_snapshot import snapshot_from_json, snapshot_to_json

    agent = AtomAgent(
        cwd=tmp_path,
        llm=CellLLM(
            [
                cell(
                    "transient=object()\nself.v.fact='durable'\nreturn_result(Done(explanation='saved'))"
                ),
            ]
        ),
    )
    restored = None
    try:
        await agent.handle({"user_messages": ["go"]})
        snapshot = snapshot_to_json(agent)
        assert "_handle_namespace" not in snapshot["attributes"]
        assert "_namespace_lifetime" not in snapshot["attributes"]
        restored = AtomAgent(cwd=tmp_path, llm=CellLLM())
        snapshot_from_json(snapshot, restored)
        restored.after_restore()
        assert restored._handle_namespace is None and restored.v.fact == "durable"
        notes = restored._system_messages_in.snapshot()
        assert (
            sum("Python cell variables and helpers were lost" in str(note) for note in notes) == 1
        )
        restored.after_restore()
        assert len(restored._system_messages_in.snapshot()) == len(notes)
    finally:
        await agent.aclose()
        if restored is not None:
            await restored.aclose()


async def test_real_root_and_kept_child_own_distinct_persistent_namespaces(tmp_path):
    scripts = {
        None: CellLLM(
            [
                cell("""
parent_only = 11
child = self.jobs.launch(Subagent('child'), name='kid')
one = await child.send('first')
two = await child.send('second')
assert one.report == '22' and two.report == '23'
assert parent_only == 11
return_result(Done(explanation='isolated'))
""")
            ]
        ),
        "kid": CellLLM(
            [
                cell(
                    "try:\n parent_only\nexcept NameError:\n pass\nelse:\n raise AssertionError('parent leaked')\nchild_only=22\nreturn_result(Done(explanation='child1', result=TaskResult(solution_description='s', evidence='e', how_to_verify='v', report=str(child_only))))"
                ),
                cell(
                    "child_only+=1\nreturn_result(Done(explanation='child2', result=TaskResult(solution_description='s', evidence='e', how_to_verify='v', report=str(child_only))))"
                ),
            ]
        ),
    }

    def factory(options, storage):
        return create_session_agent(
            options.model_copy(update={"llm": scripts[options.name]}), storage
        )

    registry = SessionRegistry(SessionStore(tmp_path / "sessions"), agent_factory=factory)
    try:
        root = await registry.create(
            SessionOptions(
                workspace=tmp_path,
                agent_spec="nooa_atom.agent:AtomAgent",
            )
        )
        assert await asyncio.wait_for(root.prompt("go"), 30) == Done(explanation="isolated")
        [child_info] = registry.children(root.id)
        child = registry.sessions[child_info.id]
        root_ns, child_ns = root._agent._handle_namespace, child._agent._handle_namespace
        assert root_ns is not child_ns and root_ns.globals is not child_ns.globals
        assert "child_only" not in root_ns.globals and "parent_only" not in child_ns.globals
        assert child._agent._namespace_lifetime == "session"
    finally:
        await registry.close_all()


async def test_non_done_never_reflects_and_active_work_defers_deletion(tmp_path):
    registry, root, llm = await make_root(
        tmp_path,
        [
            cell("return_result(NeedInput(question='q'))"),
            cell(
                "scratch=42\njob=self.jobs.launch(asyncio.sleep(100), name='live')\nreturn_result(Waiting(explanation='working', on=['live']))"
            ),
            cell("return_result(Done(explanation='interactive done with live work'))"),
            reflection(summary="done", drop_names=["scratch"]),
        ],
        reflection=True,
    )
    pending = None
    try:
        assert await root.prompt("one") == NeedInput(question="q")
        await wait_idle(root)
        assert len(llm.calls) == 1
        pending = asyncio.create_task(root.prompt("two"))
        while root._agent.jobs.get("live") is None:
            await asyncio.sleep(0.01)
        await wait_idle(root)
        assert len(llm.calls) == 2
        accepted = await root.prompt("three")
        assert accepted == Done(explanation="interactive done with live work")
        assert await pending == accepted
        await wait_idle(root)
        assert len(llm.calls) == 4
        assert root._agent._handle_namespace.globals["scratch"] == 42
    finally:
        await registry.close_all()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)


async def test_reflection_can_summarize_context_without_dropping_bindings(tmp_path):
    registry, root, llm = await make_root(
        tmp_path,
        [
            cell("return_result(Done(explanation='context only'))"),
            reflection(summary="Completed context compacted", useful_findings="context only"),
        ],
        reflection=True,
    )
    try:
        assert await root.prompt("go") == Done(explanation="context only")
        await wait_idle(root)
        assert root._agent._handle_namespace is not None
        assert len(list(root._agent.events.query(type="Summary"))) == 1
        assert len(llm.calls) == 2
    finally:
        await registry.close_all()


async def test_managed_job_output_when_no_cell_is_active(tmp_path):
    agent = AtomAgent(
        cwd=tmp_path,
        llm=CellLLM(
            [
                cell("""
gate=asyncio.Event()
async def worker():
    await gate.wait()
    print('between cells')
    show(Image.from_bytes(b'image', media_type='image/png'))
    try:
        return_result(Done(explanation='NOT AUTHORIZED'))
    except RuntimeError:
        return notification['user_messages'][0]
job=self.jobs.launch(worker(), name='between')
return_result(Done(explanation='foreground done'))
""")
            ]
        ),
    )
    try:
        assert await agent.handle({"user_messages": ["A-origin"]}) == Done(
            explanation="foreground done"
        )
        ns = agent._handle_namespace
        outputs = list(agent.events.query(type="PythonOutput"))
        before = [(output.stdout, list(output.images)) for output in outputs]
        ns.globals["gate"].set()
        job = ns.globals["job"]
        assert await job == "A-origin"
        assert job.output.stdout.startswith("between cells\n")
        assert len(job.output.images) == 1
        assert [(output.stdout, output.images) for output in outputs] == before
    finally:
        await agent.aclose()


def test_reflection_evidence_has_a_hard_budget_and_skips_oversized_events():
    from nooa_atom.agent.done_reflection import EVIDENCE_CHARS, _bounded_evidence

    from nooa.events import Error

    one, big, later = (
        Error(content="small", metadata={"call_id": "completed"}),
        Error(content="x" * EVIDENCE_CHARS, metadata={"call_id": "completed"}),
        Error(content="later", metadata={"call_id": "completed"}),
    )
    selected, evidence = _bounded_evidence([("1", one), ("2", big), ("3", later)])
    assert len(selected) == 1 and selected[0].tag == "1"
    assert selected[0].id == one.id and selected[0].call_id == "completed"
    assert len(str(evidence)) < EVIDENCE_CHARS
    assert "later" not in str(evidence)


async def test_rejected_done_is_not_a_reflection_boundary(tmp_path):
    registry, root, llm = await make_root(
        tmp_path,
        [
            cell("scratch=42\nreturn_result(explanation='')"),
            cell("return_result(Done(explanation='accepted after correction'))"),
            reflection(summary="corrected result accepted", drop_names=["scratch"]),
        ],
        reflection=True,
    )
    try:
        assert await root.prompt("go") == Done(explanation="accepted after correction")
        await wait_idle(root)
        assert len(llm.calls) == 3
        assert llm.calls[0].output_model is None and llm.calls[1].output_model is None
        assert llm.calls[2].output_model.__name__ == "DoneReflectionPlan"
        assert root.info.reply_count == 1
        assert "scratch" not in root._agent._handle_namespace.globals
    finally:
        await registry.close_all()


@pytest.mark.parametrize("summary", ["", "archive"])
@pytest.mark.parametrize("change", ["content", "call_id", "id"])
async def test_every_reflection_effect_requires_frozen_evidence_and_provenance(
    tmp_path,
    monkeypatch,
    summary,
    change,
):
    from uuid import uuid4

    from nooa_atom.agent.done_reflection import DoneReflectionPlan

    backends = []

    async def propose(self, request):
        backends.append(type(self.event_manager._backend).__name__)
        target = next(
            tag
            for tag in request["completed_event_tags"]
            if self.events[tag].event_type == "PythonOutput"
        )
        if change == "content":
            self.event_manager.update(target, stdout="NEW CORRECTED EVIDENCE")
        elif change == "call_id":
            self.event_manager.update(target, metadata={"call_id": "different-call"})
        else:
            self.event_manager.update(target, id=str(uuid4()))
        return DoneReflectionPlan(
            summary=summary, useful_findings="STALE NOTES", drop_names=["scratch"]
        )

    monkeypatch.setattr(AtomAgent, "_reflect_done_plan", propose)
    registry, root, llm = await make_root(
        tmp_path,
        [cell("scratch=42\nprint('OLD EVIDENCE')\nreturn_result(Done(explanation='accepted'))")],
        reflection=True,
    )
    try:
        assert await root.prompt("go") == Done(explanation="accepted")
        await wait_idle(root)
        assert backends == ["SQLiteEventBackend"]
        assert root.info.reply_count == 1 and len(llm.calls) == 1
        assert root._agent._handle_namespace.globals["scratch"] == 42
        assert root._agent.context.get("done_reflection_notes") is None
        assert not list(root._agent.events.query(type="Summary"))
    finally:
        await registry.close_all()


@pytest.mark.parametrize("effect", ["notes", "cleanup", "noop"])
async def test_notes_only_cleanup_only_and_noop_stale_plans(tmp_path, monkeypatch, effect):
    from nooa_atom.agent.done_reflection import DoneReflectionPlan

    async def propose(self, request):
        target = request["completed_event_tags"][0]
        self.event_manager.update(target, metadata={"call_id": "new owner"})
        return DoneReflectionPlan(
            useful_findings="STALE" if effect == "notes" else "",
            drop_names=["scratch"] if effect == "cleanup" else [],
        )

    monkeypatch.setattr(AtomAgent, "_reflect_done_plan", propose)
    registry, root, _ = await make_root(
        tmp_path, [cell("scratch=42\nreturn_result(Done(explanation='accepted'))")], reflection=True
    )
    try:
        result = await root.prompt("go")
        await wait_idle(root)
        assert result == Done(explanation="accepted") and root.info.reply_count == 1
        assert root._agent._handle_namespace.globals["scratch"] == 42
        assert root._agent.context.get("done_reflection_notes") is None
        assert not list(root._agent.events.query(type="Summary"))
    finally:
        await registry.close_all()


async def test_summary_observer_newer_notes_are_not_overwritten(tmp_path, monkeypatch):
    from nooa_atom.agent.done_reflection import DoneReflectionPlan

    observed = []

    async def propose(self, request):
        def observer(event):
            self.context["done_reflection_notes"] = "NEWER NOTES"
            observed.append(event.tag)

        self.event_manager.on("Summary", observer)
        return DoneReflectionPlan(
            summary="completed", useful_findings="STALE NOTES", drop_names=["scratch"]
        )

    monkeypatch.setattr(AtomAgent, "_reflect_done_plan", propose)
    registry, root, llm = await make_root(
        tmp_path, [cell("scratch=42\nreturn_result(Done(explanation='accepted'))")], reflection=True
    )
    try:
        assert await root.prompt("go") == Done(explanation="accepted")
        await wait_idle(root)
        assert root.info.reply_count == 1 and len(llm.calls) == 1
        assert len(observed) == 1
        assert root._agent.context["done_reflection_notes"] == "NEWER NOTES"
        assert "scratch" not in root._agent._handle_namespace.globals
        await root.wait_for_checkpoint()
        restored = AtomAgent(cwd=tmp_path, llm=CellLLM())
        try:
            assert root.handle.storage.restore_latest_snapshot(restored)
            assert restored.context["done_reflection_notes"] == "NEWER NOTES"
        finally:
            await restored.aclose()
    finally:
        await registry.close_all()


async def test_in_memory_event_id_is_frozen_not_a_live_event_alias(monkeypatch):
    from types import SimpleNamespace
    from uuid import uuid4

    from nooa_atom.agent.done_reflection import DoneReflectionPlan, reflect_accepted_done

    from nooa.events import Error
    from nooa.runtime.event_manager import EventManager

    manager = EventManager()
    event = Error(content="captured", metadata={"call_id": "completed"})
    tag = manager.add(event)
    original_id = event.id
    outcome = Done(explanation="accepted")
    agent = SimpleNamespace(
        _reflection_candidate=(outcome, "completed", tag),
        _done_reflection=True,
        events=manager,
        _handle_namespace=None,
        jobs=SimpleNamespace(running=lambda: []),
        context={},
    )

    async def propose(request):
        manager.update(tag, id=str(uuid4()))
        assert manager[tag] is event and event.id != original_id
        return DoneReflectionPlan(summary="MUST REJECT", useful_findings="STALE")

    agent._reflect_done_plan = propose
    await reflect_accepted_done(agent, outcome)
    assert manager.keys() == [tag] and agent.context == {}


async def test_registry_reload_uses_default_namespace_and_replaces_old_guidance(tmp_path):
    registry, root, _ = await make_root(
        tmp_path,
        [
            cell(
                "transient=object()\nself.v.fact='durable'\n"
                "self.context['experimental_namespace']='obsolete opt-in guidance'\n"
                "return_result(Done(explanation='checkpointed'))"
            )
        ],
    )
    store, identity = registry.store, root.id
    try:
        assert await root.prompt("save") == Done(explanation="checkpointed")
        await root.wait_for_checkpoint()
    finally:
        await registry.close_all()

    client = CellLLM(
        [
            cell(
                "assert self.v.fact == 'durable'\n"
                "try:\n transient\nexcept NameError:\n pass\n"
                "else:\n raise AssertionError('live Python value was restored')\n"
                "counter=40\ndef bump():\n global counter\n counter+=1\n return counter\n"
                "return_result(Done(explanation=str(bump())))"
            ),
            cell("return_result(Done(explanation=str(bump())))"),
        ]
    )
    registry = SessionRegistry(store, agent_factory=create_session_agent)
    try:
        root = await registry.load(identity, llm=client)
        assert root._agent._namespace_lifetime == "session"
        assert root._agent._handle_namespace is None
        assert "experimental_namespace" not in root._agent.context
        assert "live session" in root._agent.context["python_cell_lifetime"]
        assert await root.prompt("first after restart") == Done(explanation="41")
        assert await root.prompt("second after restart") == Done(explanation="42")
        assert len(client.calls) == 2
    finally:
        await registry.close_all()
