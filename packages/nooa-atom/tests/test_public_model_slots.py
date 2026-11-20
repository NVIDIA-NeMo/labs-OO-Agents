# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json

import httpx
import pytest
from public_model_fixtures import native_models as native_models
from public_model_fixtures import public_agent_factory as public_agent_factory
from public_model_fixtures import public_registry as public_registry

from nooa import Agent
from nooa.unifiedllm import Need
from nooa.unifiedllm.limits import context_limits_for
from tests.runtime.test_public_llm_factory_calls import generate
from tests.unifiedllm.test_native_projection import payload


async def test_actual_registry_parent_child_reasoning_model_and_close(public_registry):
    registry, options, made = public_registry
    root = await registry.create(options)
    child = await registry.create(root.options.inherit(kept=True), parent_id=root.id)
    binding = root._agent.llms["main"]
    assert child._agent.llms["main"] is binding
    await root.set_reasoning("high")
    assert child.info.reasoning == root.info.reasoning == "high"
    assert child._agent.llm is root._agent.llm
    await generate(child._agent)
    await binding._join_cleanup()
    assert made[0]._closed is True
    assert not binding.client._closed
    await root.set_model("replacement")
    newest = await registry.create(root.options.inherit(kept=True), parent_id=root.id)
    assert root._agent.llms["main"] is binding
    assert child._agent.llms["main"] is binding
    assert newest._agent.llms["main"] is root._pending_model[1]
    assert newest._agent.llm is made[1]
    await root._apply_pending_model()
    assert root._agent.llm is newest._agent.llm is made[1]
    assert child._agent.llms["main"] is binding
    assert root.model_info().reasoning_level is None
    await registry.close_all()
    assert all(view._closed for view in made)


async def test_public_borrowed_root_and_same_alias_independent(public_registry, native_models):
    registry, options, made = public_registry
    borrowed = native_models.make()
    root = await registry.create(options.model_copy(update={"llm": borrowed}))
    assert root._agent.llm is borrowed
    other = await registry.create(options)
    another = await registry.create(options)
    assert other._agent.llms["main"] is not another._agent.llms["main"]
    await registry.close_all()
    assert not borrowed._closed
    assert all(view._closed for view in made)


async def test_inflight_native_call_retains_preselection_view(public_registry, native_models):
    registry, options, made = public_registry
    entered, release = asyncio.Event(), asyncio.Event()
    bodies = []

    async def handler(request):
        bodies.append(json.loads(request.content))
        entered.set()
        await release.wait()
        return httpx.Response(200, json=payload("openai-chat"))

    borrowed = native_models.make(
        handler=handler, reasoning_levels={"high": {"reasoning_effort": "high"}}
    )
    root = await registry.create(options.model_copy(update={"llm": borrowed}))
    task = asyncio.create_task(generate(root._agent, max_tokens=200))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        await root.set_reasoning("high")
        assert root._agent.llm is not borrowed
        assert "reasoning_effort" not in bodies[0]
        release.set()
        await task
        await generate(root._agent)
        assert bodies[-1]["reasoning_effort"] == "high"
        assert context_limits_for(borrowed).reserved_output_tokens == 100
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_explicit_none_rejected_and_lazy_vision_does_not_deny_main(native_models):
    root = native_models.make()
    with pytest.raises(ValueError, match="llm=None"):
        Agent(llm=None)

    class VisionAgent(Agent):
        llms = {"vision": Need.IMAGES}

    agent = VisionAgent(llm=root)
    await generate(agent)
    assert agent.llm is root
    with pytest.raises(RuntimeError, match="unbound"):
        _ = agent.llms["vision"].client
    await agent.aclose()


async def test_registry_reload_replays_explicit_clear(public_registry):
    from nooa_atom.session.registry import SessionRegistry

    registry, options, made = public_registry
    root = await registry.create(options)
    await root.set_reasoning("high")
    await root.set_reasoning(None)
    session_id = root.id
    await registry.close_all()
    fresh = SessionRegistry(
        registry.store, agent_factory=public_agent_factory, llm_factory=registry.llm_factory
    )
    try:
        restored = await fresh.load(session_id)
        assert restored.info.reasoning_selected is True
        assert restored.info.reasoning is None
        assert restored.model_info().reasoning_level is None
        await generate(restored._agent)
    finally:
        await fresh.close_all()


async def test_actual_atom_reflection_predict_2048_native_request(native_models, tmp_path):
    from nooa_atom.agent.agent import AtomAgent
    from nooa_atom.agent.done_reflection import DoneReflectionPlan

    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=payload("openai-chat", "{}"))

    root = native_models.make(handler=handler)
    agent = AtomAgent(cwd=tmp_path, llm=root)
    try:
        result = await agent._reflect_done_plan({"completed_event_tags": []})
        assert isinstance(result, DoneReflectionPlan)
        assert result.summary == ""
        assert len(seen) == 1
        assert seen[0]["max_completion_tokens"] == 2048
        assert seen[0]["response_format"]["type"] == "json_schema"
        assert "output_model" not in seen[0]
        assert context_limits_for(root).reserved_output_tokens == 100
    finally:
        await agent.aclose()


async def test_child_same_alias_reopen_and_reload_explicit_clear(public_registry, native_models):
    from nooa_atom.session.registry import SessionRegistry

    registry, options, made = public_registry
    parent = await registry.create(options)
    child = await registry.create(parent.options.inherit(kept=True), parent_id=parent.id)
    child_id = child.id
    binding = parent._agent.llms["main"]
    await child.set_reasoning("high")
    assert parent._agent.llms["main"] is child._agent.llms["main"] is binding
    await registry.close(child.id)
    reopened = await registry.load(child_id)
    assert reopened._agent.llms["main"] is binding
    assert reopened.info.reasoning == parent.info.reasoning == "high"
    await reopened.set_reasoning(None)
    assert parent.info.reasoning_selected and parent.info.reasoning is None
    await registry.close_all()
    fresh = SessionRegistry(
        registry.store, agent_factory=public_agent_factory, llm_factory=registry.llm_factory
    )
    try:
        restored_parent = await fresh.load(parent.id)
        restored_child = await fresh.load(child_id)
        assert restored_child._agent.llms["main"] is restored_parent._agent.llms["main"]
        assert restored_child.info.reasoning_selected and restored_child.info.reasoning is None
        await generate(restored_child._agent)
        assert "reasoning_effort" not in native_models.seen[-1][1]
    finally:
        await fresh.close_all()


async def test_options_explicit_client_none_inherit_and_coherent_binding(
    public_registry, native_models
):
    from nooa_atom.session.options import SessionOptions

    registry, options, made = public_registry
    parent = await registry.create(options)
    binding = parent._agent.llms["main"]
    replacement = native_models.make()
    inherited = parent.options.inherit(llm=None)
    assert inherited.model_binding is binding
    explicit = parent.options.inherit(llm=replacement)
    assert explicit.llm is replacement and explicit.model_binding is None
    child = await registry.create(explicit, parent_id=parent.id)
    assert child._agent.llm is replacement and child._agent.llms["main"] is not binding
    changed = parent.options.inherit(llm=None, model="different")
    assert changed.model_binding is None and changed.llm is None
    data = options.model_dump()
    with pytest.raises(ValueError, match="same client"):
        SessionOptions(**data, llm=replacement, model_binding=binding)
    coherent = SessionOptions(**data, llm=binding.client, model_binding=binding)
    assert coherent.llm is binding.client
    assert len(made) == 1


@pytest.mark.parametrize("failure_site", ["restore_note", "install_port"])
async def test_actual_registry_failed_restore_or_install_closes_every_constructed_agent(
    public_registry, monkeypatch, failure_site
):
    import nooa_atom.session.registry as registry_module
    from nooa_atom.session.events import SnapshotRestoreFailed

    from nooa.runtime.event_manager import EventManager
    from nooa.storage.sqlite import SQLiteStorageManager

    registry, options, made = public_registry
    created = []

    def factory(options, storage):
        agent = public_agent_factory(options, storage)
        created.append(agent)
        return agent

    registry._agent_factory = factory
    failure = RuntimeError("failed publication boundary")
    if failure_site == "install_port":
        monkeypatch.setattr(
            registry_module, "install_port", lambda *args: (_ for _ in ()).throw(failure)
        )
        with pytest.raises(RuntimeError) as caught:
            await registry.create(options)
    else:
        root = await registry.create(options)
        session_id = root.id
        await registry.close(session_id)
        created.clear()
        monkeypatch.setattr(
            SQLiteStorageManager,
            "restore_latest_snapshot",
            lambda *args: (_ for _ in ()).throw(RuntimeError("bad snapshot")),
        )
        original = EventManager.add

        def add(self, event):
            if isinstance(event, SnapshotRestoreFailed):
                raise failure
            return original(self, event)

        monkeypatch.setattr(EventManager, "add", add)
        with pytest.raises(RuntimeError) as caught:
            await registry.load(session_id)
    assert caught.value is failure
    assert len(created) == (2 if failure_site == "restore_note" else 1)
    assert all(agent._models_close_task.done() for agent in created)
    assert all(client._closed for client in made)
    assert not registry._reserved
    assert all(
        binding._closed and not binding._cleanup for binding in registry._failed_model_bindings
    )


@pytest.mark.parametrize("op", ["load", "open_child"])
@pytest.mark.parametrize("selection", ["high", None, "default"])
async def test_live_child_adoption_exact_view_and_persisted_selection(
    public_registry, native_models, op, selection
):
    from nooa.runtime.model_binding import public_base

    registry, options, made = public_registry
    parent = await registry.create(options)
    child = await registry.create(parent.options.inherit(kept=True), parent_id=parent.id)
    child_id = child.id
    await child.set_reasoning("high")
    await registry.close(child_id)
    if selection == "default":
        await parent.set_model("initial")
    else:
        await parent.set_reasoning(selection)
    binding = parent._selected_binding()
    view = binding.client
    pool = public_base(view)._lease
    await binding._join_cleanup()
    leases = pool.leases
    from nooa.unifiedllm import LLMRequest

    await view.acall(LLMRequest(messages=[{"role": "user", "content": "baseline"}]))
    source_wire = native_models.seen[-1][1]
    parent_info = parent.info.model_copy(deep=True)
    reopened = await (
        registry.load(child_id) if op == "load" else registry.open_child(parent, child_id)
    )
    assert reopened._agent.llms["main"] is binding
    assert reopened._agent.llm is binding.client is view
    assert pool.leases == leases
    assert parent.info == parent_info
    assert reopened.info.reasoning == reopened.handle.info.reasoning == parent.info.reasoning
    assert (
        reopened.info.reasoning_selected
        == reopened.handle.info.reasoning_selected
        == parent.info.reasoning_selected
    )
    await generate(reopened._agent)
    wire = native_models.seen[-1][1]
    assert wire.get("reasoning_effort") == source_wire.get("reasoning_effort")
    if selection is None:
        assert "reasoning_effort" not in wire
    elif selection == "high":
        assert wire["reasoning_effort"] == "high"


@pytest.mark.parametrize("selection", [None, "default"])
async def test_child_adoption_metadata_failure_preserves_parent_and_original_error(
    public_registry, monkeypatch, selection
):
    from nooa_atom.session.store import SessionHandle

    registry, options, made = public_registry
    parent = await registry.create(options)
    child = await registry.create(parent.options.inherit(kept=True), parent_id=parent.id)
    child_id = child.id
    await child.set_reasoning("high")
    await registry.close(child_id)
    if selection == "default":
        await parent.set_model("initial")
    else:
        await parent.set_reasoning(None)
    binding = parent._selected_binding()
    view = binding.client
    refs = binding._references
    before = parent.info.model_copy(deep=True)
    failure = RuntimeError("child adoption record failed")
    original = SessionHandle.set_model if selection == "default" else SessionHandle.set_reasoning

    def fail(self, value):
        if self.id == child_id:
            raise failure
        return original(self, value)

    monkeypatch.setattr(
        SessionHandle, "set_model" if selection == "default" else "set_reasoning", fail
    )
    with pytest.raises(RuntimeError) as caught:
        await registry.load(child_id)
    assert caught.value is failure
    assert child_id not in registry.sessions and child_id not in registry._reserved
    assert parent.info == before and binding.client is view
    assert binding._references == refs
    assert not binding._closed


async def test_reopened_child_override_detached_and_pending_alias_precedence(
    public_registry, native_models
):
    registry, options, made = public_registry
    parent = await registry.create(options)
    child = await registry.create(parent.options.inherit(kept=True), parent_id=parent.id)
    child_id = child.id
    old_binding = child._agent.llms["main"]
    await registry.close(child_id)
    await parent.set_model("other")
    pending = parent._selected_binding()
    assert pending is not old_binding
    reopened = await registry.open_child(parent, child_id)
    assert reopened._agent.llms["main"] is not pending
    assert reopened.info.model == "initial"
    await registry.close(child_id)
    replacement = native_models.make()
    explicit = await registry.load(child_id, llm=replacement)
    assert explicit._agent.llm is replacement and explicit._agent.llms["main"] is not pending
    await registry.close(child_id)
    different = await registry.load(child_id, model="third")
    assert different.info.model == "initial"  # overrides do not rewrite the stored alias
    assert different._agent.llms["main"] is not pending
    await registry.close(child_id)
    await registry.close(parent.id)
    detached = await registry.load(child_id)
    assert detached._agent.llms["main"] is not old_binding
    assert detached._agent.llms["main"] is not pending


@pytest.mark.parametrize("op", ["load", "open_child"])
@pytest.mark.parametrize("explicit_clear", [False, True])
@pytest.mark.parametrize("configured_kind", ["selected-loader-level", "raw-provider-default"])
async def test_live_child_adoption_selected_default_and_raw_restoration_control(
    public_registry, native_models, monkeypatch, op, explicit_clear, configured_kind
):
    from nooa.runtime.model_binding import public_base
    from nooa.unifiedllm import LLMRequest

    registry, options, made = public_registry

    def configured_loader(alias, workspace):
        configured = (
            {"reasoning_level": "low"}
            if configured_kind == "selected-loader-level"
            else {"reasoning_effort": "low"}
        )
        view = native_models.make(
            model=alias or "initial",
            reasoning_levels={
                "low": {"reasoning_effort": "low"},
                "high": {"reasoning_effort": "high"},
            },
            **configured,
        )
        made.append(view)
        return view

    monkeypatch.setattr(registry, "llm_factory", configured_loader)
    parent = await registry.create(options)
    assert parent.info.reasoning_selected is False
    await generate(parent._agent)
    assert native_models.seen[-1][1]["reasoning_effort"] == "low"
    child = await registry.create(parent.options.inherit(kept=True), parent_id=parent.id)
    await child.set_reasoning("high")
    child_id = child.id
    await registry.close(child_id)
    await parent.set_model("initial")
    binding = parent._selected_binding()
    assert parent.info.reasoning_selected is False
    await binding.client.acall(LLMRequest(messages=[{"role": "user", "content": "configured"}]))
    assert native_models.seen[-1][1]["reasoning_effort"] == "low"
    if explicit_clear:
        await parent.set_reasoning(None)
    await binding.client.acall(LLMRequest(messages=[{"role": "user", "content": "selected"}]))
    selected_wire = native_models.seen[-1][1]
    view = binding.client
    pool = public_base(view)._lease
    leases = pool.leases
    before = parent.info.model_copy(deep=True)
    reopened = await (
        registry.load(child_id) if op == "load" else registry.open_child(parent, child_id)
    )
    assert reopened._agent.llms["main"] is binding
    assert reopened._agent.llm is binding.client is view
    assert pool.leases == leases
    assert parent.info == before
    assert reopened.info.reasoning is None
    assert reopened.info.reasoning_selected is explicit_clear
    assert reopened.handle.info.reasoning_selected is explicit_clear
    await generate(reopened._agent)
    wire = native_models.seen[-1][1]
    assert wire.get("reasoning_effort") == selected_wire.get("reasoning_effort")
    if explicit_clear and configured_kind == "selected-loader-level":
        assert "reasoning_effort" not in wire
    else:
        # Clearing a selected declaration restores genuine raw provider values;
        # it does not erase raw defaults or force metadata reasoning_default.
        assert wire["reasoning_effort"] == "low"


@pytest.mark.parametrize("op", ["load", "open_child"])
@pytest.mark.parametrize("wrapped", [False, True])
async def test_live_child_adopts_injected_public_and_exact_controller(
    public_registry, native_models, op, wrapped
):
    from nooa.unifiedllm.admission import AdmissionControlConfig
    from nooa.unifiedllm.controlled import AdmissionControl

    registry, options, made = public_registry
    root = native_models.make()
    client = AdmissionControl(root, AdmissionControlConfig(max_in_flight=1)) if wrapped else root
    parent = await registry.create(options.model_copy(update={"llm": client}))
    child = await registry.create(parent.options.inherit(kept=True), parent_id=parent.id)
    child_id = child.id
    await registry.close(child_id)
    binding = parent._selected_binding()
    reopened = await (
        registry.load(child_id) if op == "load" else registry.open_child(parent, child_id)
    )
    assert reopened._agent.llms["main"] is binding
    assert reopened._agent.llm is client
    await generate(reopened._agent)
    assert native_models.seen[-1][1]["model"] == "fixture-model"
    if wrapped:
        assert reopened._agent.llm.controller is client.controller
    assert not made
    await registry.close_all()
    assert not root._closed


@pytest.mark.parametrize(
    "bad_kind", ["llm-subclass", "controller-subclass", "facade", "mutated-controller"]
)
async def test_live_child_rejects_unsupported_public_before_factory_or_wire(
    public_registry, native_models, monkeypatch, bad_kind
):
    from types import SimpleNamespace

    from nooa.unifiedllm import LLM
    from nooa.unifiedllm.admission import AdmissionControlConfig
    from nooa.unifiedllm.controlled import AdmissionControl

    registry, options, made = public_registry
    parent = await registry.create(options)
    child = await registry.create(parent.options.inherit(kept=True), parent_id=parent.id)
    child_id = child.id
    await registry.close(child_id)
    binding = parent._selected_binding()
    original = binding.client
    if bad_kind == "llm-subclass":

        class Subclass(LLM):
            pass

        bad = object.__new__(Subclass)
    elif bad_kind == "controller-subclass":

        class Subclass(AdmissionControl):
            pass

        bad = object.__new__(Subclass)
        bad.base_llm = original
    elif bad_kind == "facade":
        bad = SimpleNamespace(base_llm=original, acall=original.acall)
    else:
        bad = AdmissionControl(original, AdmissionControlConfig(max_in_flight=1))
        bad.acall = original.acall
    before = len(native_models.seen)
    factory_calls = []

    def factory(options, storage):
        factory_calls.append(options)
        return public_agent_factory(options, storage)

    monkeypatch.setattr(registry, "_agent_factory", factory)
    binding._current = bad
    try:
        with pytest.raises(TypeError):
            await registry.load(child_id)
        assert not factory_calls
        assert len(native_models.seen) == before
        assert not registry._reserved
    finally:
        binding._current = original


async def test_slot_based_legacy_child_reload_keeps_per_session_factory(public_registry):
    from public_model_fixtures import PublicAgent

    registry, options, made = public_registry
    clients = []

    class LegacySlotClient:
        __slots__ = ()

        async def acall(self, messages, **params):
            raise AssertionError("This structural reload control must not dispatch")

    def factory(options, storage):
        client = LegacySlotClient()
        clients.append(client)
        return PublicAgent(storage=storage, llm=client)

    registry._agent_factory = factory
    registry.llm_factory = None  # Per-session injection, not an alias-producing factory.
    options = options.model_copy(update={"model": None})
    parent = await registry.create(options)
    child = await registry.create(parent.options.inherit(kept=True), parent_id=parent.id)
    child_id = child.id
    await registry.close(child_id)
    reopened = await registry.load(child_id)
    assert reopened._agent.llm is clients[-1]
    assert reopened._agent.llm is not parent._agent.llm
    assert len(clients) == 3 and not made
    await registry.close(child_id)
    await registry.close(parent.id)
    detached = await registry.load(child_id)
    assert detached._agent.llm is clients[-1]
    assert len(clients) == 4 and not made


@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("close_failure", [False, True])
async def test_idle_reasoning_publication_disposes_owned_previous_view_after_commit(
    public_registry, native_models, monkeypatch, borrowed, close_failure
):
    registry, options, made = public_registry
    injected = native_models.make(reasoning_levels={"high": {"reasoning_effort": "high"}})
    parent = await registry.create(
        options.model_copy(update={"llm": injected}) if borrowed else options
    )
    binding = parent._agent.llms["main"]
    original = binding.client
    await parent.set_reasoning("high")
    first = binding.client
    await binding._join_cleanup()
    assert first is not original and not first._closed
    assert original._closed is (not borrowed)
    failure = RuntimeError("post-commit idle view disposal")
    if close_failure:
        prepare = binding._prepare_publication
        close = first.aclose

        async def failing_close():
            await close()
            raise failure

        def prepared_then_fault(view):
            publication = prepare(view)
            monkeypatch.setattr(first, "aclose", failing_close)
            return publication

        monkeypatch.setattr(binding, "_prepare_publication", prepared_then_fault)
    await parent.set_reasoning(None)
    selected = binding.client
    assert selected is not first and not selected._closed
    assert id(first) not in binding._owned and binding._cleanup
    assert parent.info.reasoning_selected and parent.info.reasoning is None
    assert parent.handle.info.reasoning_selected and parent.handle.info.reasoning is None
    if close_failure:
        with pytest.raises(RuntimeError) as caught:
            await binding._join_cleanup()
        assert caught.value is failure
    else:
        await binding._join_cleanup()
    assert first._closed and not binding._cleanup
    assert original._closed is (not borrowed)
    assert selected._lease.leases == 1 + int(borrowed)
    await generate(parent._agent)
    assert "reasoning_effort" not in native_models.seen[-1][1]
    await registry.close_all()
    assert selected._closed and original._closed is (not borrowed)


@pytest.mark.parametrize("level", ["high", None])
async def test_create_after_selection_persists_current_binding_and_reloads_alone(
    public_registry, native_models, level
):
    from nooa_atom.session.registry import SessionRegistry

    registry, options, made = public_registry
    parent = await registry.create(options)
    await parent.set_reasoning("high")
    await parent.set_reasoning(level)
    selected = parent._selected_binding()
    assert parent.options.llm is selected.client
    inherited = parent.options.inherit(kept=True)
    assert inherited.llm is selected.client and inherited.model_binding is selected
    child = await registry.create(inherited, parent_id=parent.id)
    assert child._selected_binding() is selected
    assert child.info.reasoning_selected and child.info.reasoning == level
    assert registry.store.get(child.id).reasoning_selected
    await generate(child._agent)
    assert native_models.seen[-1][1].get("reasoning_effort") == level
    await registry.close_all()
    fresh = SessionRegistry(
        registry.store, agent_factory=registry._agent_factory, llm_factory=registry.llm_factory
    )
    try:
        restored = await fresh.load(child.id)
        assert restored.options.llm is restored._selected_binding().client
        assert restored.info.reasoning_selected and restored.info.reasoning == level
        await generate(restored._agent)
        assert native_models.seen[-1][1].get("reasoning_effort") == level
    finally:
        await fresh.close_all()
