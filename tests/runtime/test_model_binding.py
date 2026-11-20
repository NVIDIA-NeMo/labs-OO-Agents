# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio

import pytest

from nooa import Agent, ModelBinding
from nooa.unifiedllm import FakeLLMClient, Need, Requirements
from nooa.unifiedllm.errors import LLMFailure
from tests.runtime.public_llm_fixtures import native_models as native_models


def test_slot_mro_snapshot_and_readonly():
    declaration = {"main": Need.NONE, "vision": Requirements(Need.IMAGES)}

    class Root(Agent):
        llms = declaration

    declaration.clear()

    class Child(Root):
        llms = {"main": Need.NONE}

    agent = Child(llm=FakeLLMClient())
    assert agent._llm_requirements["vision"].flags == Need.IMAGES
    with pytest.raises(TypeError):
        agent.llms["main"] = None
    assert agent.llm is agent.llms["main"].client
    with pytest.raises(RuntimeError, match="unbound"):
        _ = agent.llms["vision"].client


@pytest.mark.parametrize("declaration", [{"bad-name": Need.NONE}, {"main": 7}, {"main": "alias"}])
def test_bad_declarations(declaration):
    with pytest.raises((TypeError, ValueError, LLMFailure)):

        class Invalid(Agent):
            llms = declaration


def test_main_collision_and_borrowed_custom():
    client = FakeLLMClient()
    with pytest.raises(TypeError, match="main"):
        Agent(llm=client, llms={"main": client})
    agent = Agent(llm=client)
    assert agent.llm is client


async def test_capture_retired_view_sharer_and_idempotent_creator(native_models):
    root = native_models.make()
    binding = ModelBinding(root)
    sharer = binding.retain()
    async with binding.capture() as old:
        candidate = binding.prepare(max_tokens=200)
        binding.publish(candidate)
        assert old is root and binding.client is candidate
        await binding.aclose()
        await binding.aclose()
        assert binding.client is candidate
        assert not candidate._closed
    await sharer._release()
    assert candidate._closed and not root._closed
    assert not native_models.http[0].is_closed


async def test_owned_root_retirement_waits_for_capture(native_models):
    root = native_models.make()
    binding = ModelBinding(root, owned=True)
    async with binding.capture():
        await binding.aclose()
        assert not root._closed
    assert root._closed


async def test_agent_same_binding_reasoning_and_child_model_isolation(native_models):
    root = native_models.make()
    other = native_models.make()
    parent = Agent(llm=root)
    # Inheritance comes from the executing parent call, not a constructor API.
    child = Agent(llm=parent.llms["main"])
    assert child.llms["main"] is parent.llms["main"]
    binding = parent.llms["main"]
    replacement = binding.prepare(temperature=0)
    binding.publish(replacement)
    assert parent.llm is child.llm is replacement
    parent.set_llm(other)
    assert parent.llm is other and child.llm is replacement
    await parent.aclose()
    assert not replacement._closed
    await child.aclose()
    assert replacement._closed and not root._closed and not other._closed


async def test_cancelled_close_keeps_cleanup_task():
    started, release = asyncio.Event(), asyncio.Event()

    class Owned:
        async def acall(self, **kwargs):
            pass

        async def aclose(self):
            started.set()
            await release.wait()

    binding = ModelBinding(Owned(), owned=True)
    task = asyncio.create_task(binding.aclose())
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()  # cancellation is deferred until owned cleanup ends
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await binding.aclose()
    assert not binding._cleanup


async def test_delegated_publication_is_local_and_preserves_old_capture(native_models):
    parent = ModelBinding(native_models.make())
    delegated = ModelBinding(parent, require=Need.NONE)
    async with delegated.capture() as old:
        selected = delegated.prepare(max_tokens=300)
        delegated.publish(selected)
        assert delegated.client is selected and parent.client is old
        async with delegated.capture() as new:
            assert new is selected
        with pytest.raises(TypeError):
            delegated._prepare_publication(object())
        assert delegated.client is selected
        await parent.aclose()
        assert not old._closed
    await delegated.aclose()
    assert selected._closed and parent._references == 0


async def test_failed_main_and_later_slot_drop_all_retained_references(native_models):
    source = ModelBinding(native_models.make())

    class RequiresImages(Agent):
        llms = {"main": Need.IMAGES}

    from nooa.unifiedllm import LLMFailure

    with pytest.raises(LLMFailure, match="image capability"):
        RequiresImages(llm=source)
    assert source._references == 1
    closed = ModelBinding(native_models.make())
    await closed.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        Agent(llm=source, llms={"vision": closed})
    assert source._references == 1
    await source.aclose()


async def test_all_model_closers_join_after_first_failure_and_repeated_cancel():
    entered, release = asyncio.Event(), asyncio.Event()
    closed = []
    failure = RuntimeError("first closer failed")

    class Owned:
        def __init__(self, number):
            self.number = number

        async def acall(self, *args, **kwargs):
            pass

        async def aclose(self):
            closed.append(self.number)
            if self.number == 1:
                raise failure
            entered.set()
            await release.wait()

    binding = ModelBinding(Owned(1), owned=True)
    binding.publish(Owned(2))
    task = asyncio.create_task(binding.aclose())
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and sorted(closed) == [1, 2]
    release.set()
    with pytest.raises((RuntimeError, asyncio.CancelledError)):
        await task
    assert not binding._cleanup
    await binding.aclose()


@pytest.mark.parametrize("unsubscribe_failure", [False, True])
async def test_summary_close_drains_pending_native_task_and_owned_binding(
    native_models, monkeypatch, unsubscribe_failure
):
    import asyncio

    import httpx

    from nooa.agents import MethodSummarizer
    from tests.runtime.test_public_llm_factory_calls import generate
    from tests.unifiedllm.test_native_projection import payload

    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def handler(request):
        entered.set()
        asyncio.current_task().add_done_callback(lambda task: finished.set())
        await release.wait()
        return httpx.Response(200, json=payload("openai-chat"))

    root = native_models.make(handler=handler)
    binding = ModelBinding(root, owned=True)
    parent = Agent(llm=native_models.make())
    summary = MethodSummarizer.install(parent, llm=binding)
    pending = asyncio.create_task(generate(summary, max_tokens=250))
    summary._pending_task = pending
    sentinel = RuntimeError("unsubscribe failure")
    if unsubscribe_failure:
        original = summary._uninstall

        def fail():
            original()
            raise sentinel

        monkeypatch.setattr(summary, "_uninstall", fail)
    try:
        await asyncio.wait_for(entered.wait(), 3)
        close_entered = asyncio.Event()

        async def close_summary_owner():
            close_entered.set()
            await summary.aclose()

        close_task = asyncio.create_task(close_summary_owner())
        try:
            await asyncio.wait_for(close_entered.wait(), 3)
            await asyncio.wait_for(asyncio.gather(pending, return_exceptions=True), 3)
            assert pending.cancelled() and summary._pending_task is None
            done, _ = await asyncio.wait({close_task}, timeout=0.15)
            assert not done and not close_task.done() and not finished.is_set()
            assert not root._closed
            release.set()
            if unsubscribe_failure:
                with pytest.raises(RuntimeError) as caught:
                    await asyncio.wait_for(close_task, 3)
                assert caught.value is sentinel
            else:
                await asyncio.wait_for(close_task, 3)
            assert summary._unsub_before is None
            assert summary._unsub_after is None
            assert summary._unsub_close is None and binding._references == 1
            assert finished.is_set() and not root._closed and root._lease.leases == 1
            await binding.aclose()
            assert binding._references == 0 and root._closed and root._lease.leases == 0
            assert root._lease._disposal is not None and root._lease._disposal.done()
            assert not native_models.http[0].is_closed
        finally:
            release.set()
            await asyncio.gather(close_task, return_exceptions=True)
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        # Retained close failures are deliberately re-raised on repeated close.
        try:
            await parent.aclose()
        except RuntimeError as exc:
            assert exc is sentinel
        await binding.aclose()


async def test_late_constructor_retained_binding_drops_with_held_traceback(native_models):
    from nooa.config.truncation_config import TruncationConfig

    root = native_models.make()
    binding = ModelBinding(root, owned=True)
    held = None
    try:
        Agent(llm=binding, truncation=TruncationConfig())
    except ValueError as exc:
        held = exc
    assert held is not None and held.__traceback__ is not None
    assert binding._references == 1
    await binding.aclose()
    assert binding._references == 0 and root._closed and root._lease.leases == 0
    assert not native_models.http[0].is_closed


async def test_failed_constructor_alias_disposal_is_strong_and_cancel_safe(
    native_models, monkeypatch
):
    from nooa.agent import _failed_model_cleanups, _join_failed_model_cleanups
    from nooa.unifiedllm import registry

    fixture_root = native_models.make()
    monkeypatch.setitem(
        registry.MODELS,
        "constructor-alias",
        {
            "adapter": "openai-chat",
            "model": "fixture-model",
            "api_key": "fixture-key",
            "client": native_models.sdks[0],
            "base_url": "https://fixture.invalid/v1",
        },
    )
    entered, release = asyncio.Event(), asyncio.Event()
    sentinel = RuntimeError("late constructor sentinel")
    aliases = []

    def fail(agent, config):
        alias = agent.llm
        aliases.append(alias)
        original = alias.aclose

        async def gated():
            entered.set()
            await release.wait()
            await original()

        monkeypatch.setattr(alias, "aclose", gated)
        raise sentinel

    monkeypatch.setattr(Agent, "_resolve_truncation", fail)
    with pytest.raises(RuntimeError) as caught:
        Agent(llm="constructor-alias")
    assert caught.value is sentinel and caught.value.__traceback__ is not None
    await asyncio.wait_for(entered.wait(), 3)
    assert _failed_model_cleanups and not aliases[0]._closed
    joining = asyncio.create_task(_join_failed_model_cleanups())
    await asyncio.sleep(0)  # Start the joiner before testing in-flight cancellation.
    try:
        for _ in range(2):
            joining.cancel()
            await asyncio.sleep(0)
        assert not joining.done() and not aliases[0]._closed
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await joining
        await _join_failed_model_cleanups()
        assert not _failed_model_cleanups
        assert aliases[0]._closed and aliases[0]._lease.leases == 0
        assert fixture_root._lease is not aliases[0]._lease and fixture_root._lease.leases == 1
        assert not fixture_root._closed and not native_models.http[0].is_closed
    finally:
        release.set()
        await asyncio.gather(joining, return_exceptions=True)
        await _join_failed_model_cleanups()


async def test_constructor_drops_every_holder_despite_secondary_baseexception(
    native_models, monkeypatch
):
    from nooa.agent import _join_failed_model_cleanups

    class Secondary(BaseException):
        pass

    first, second = ModelBinding(native_models.make()), ModelBinding(native_models.make())
    original = first._drop_reference
    secondary = Secondary("secondary drop")
    sentinel = RuntimeError("constructor error wins")

    def failed_drop(**kwargs):
        original(**kwargs)
        raise secondary

    monkeypatch.setattr(first, "_drop_reference", failed_drop)

    def fail(agent, config):
        raise sentinel

    monkeypatch.setattr(Agent, "_resolve_truncation", fail)
    with pytest.raises(RuntimeError) as caught:
        Agent(llm=first, llms={"aux": second})
    assert caught.value is sentinel
    assert first._references == second._references == 1
    await _join_failed_model_cleanups()
    monkeypatch.setattr(first, "_drop_reference", original)
    await first.aclose()
    await second.aclose()


async def test_fifth_shared_raw_owners_join_only_their_exact_native_receipts(native_models):
    import asyncio

    import httpx

    from tests.runtime.test_public_llm_factory_calls import generate
    from tests.unifiedllm.test_native_projection import payload

    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    finished = [asyncio.Event(), asyncio.Event()]
    calls = []

    async def handler(request):
        index = len(calls)
        calls.append(request)
        asyncio.current_task().add_done_callback(lambda _, index=index: finished[index].set())
        entered[index].set()
        await release[index].wait()
        return httpx.Response(200, json=payload("openai-chat"))

    root = native_models.make(handler=handler)
    creator = ModelBinding(root, owned=True)
    agents = [Agent(llm=creator), Agent(llm=creator)]
    logical, closing = [], []
    try:
        for index, agent in enumerate(agents):
            logical.append(asyncio.create_task(generate(agent, max_tokens=240)))
            await asyncio.wait_for(entered[index].wait(), 3)
        for task in logical:
            task.cancel("prompt logical exit")
        await asyncio.wait_for(asyncio.gather(*logical, return_exceptions=True), 3)
        assert all(task.cancelled() for task in logical)
        closing = [asyncio.create_task(agent.aclose()) for agent in agents]
        done, _ = await asyncio.wait(set(closing), timeout=0.1)
        assert not done and not root._closed
        release[0].set()
        await asyncio.wait_for(closing[0], 3)
        assert finished[0].is_set() and not finished[1].is_set()
        assert not closing[1].done() and not root._closed
        release[1].set()
        await asyncio.wait_for(closing[1], 3)
        assert finished[1].is_set() and creator._references == 1
        assert all(not agent._retired_bindings for agent in agents)
        await creator.aclose()
        assert root._closed and root._lease.leases == 0 and len(calls) == 2
    finally:
        for gate in release:
            gate.set()
        await asyncio.gather(*logical, *closing, return_exceptions=True)
        for agent in agents:
            await agent.aclose()
        await creator.aclose()


async def test_fifth_actual_dispatch_preentry_cancel_receipt_releases_pool(native_models):
    import asyncio

    from nooa.unifiedllm._transport_lifetime import _dispatch_registrar
    from nooa.unifiedllm.request import LLMRequest

    root = native_models.make()
    binding = ModelBinding(root)
    dispatched = []
    async with binding.capture() as selected:
        original = _dispatch_registrar.get()

        def cancel_before_entry(task, pool):
            receipt = original(task, pool)
            dispatched.append(task)
            task.cancel("before coroutine entry")
            return receipt

        token = _dispatch_registrar.set(cancel_before_entry)
        try:
            with pytest.raises(asyncio.CancelledError):
                await selected.acall(LLMRequest(messages=[{"role": "user", "content": "x"}]))
        finally:
            _dispatch_registrar.reset(token)
    await binding.aclose()
    assert len(dispatched) == 1 and dispatched[0].cancelled()
    assert not native_models.seen and root._lease.leases == 1 and not root._closed
    assert not binding._records


async def test_fifth_actual_dispatch_registrar_failure_does_not_rollback_task(native_models):
    import asyncio

    import httpx

    from nooa.unifiedllm._transport_lifetime import _dispatch_registrar
    from nooa.unifiedllm.request import LLMRequest
    from tests.unifiedllm.test_native_projection import payload

    entered, release = asyncio.Event(), asyncio.Event()
    control = BaseException("registration control")
    dispatched, seen = [], []

    async def handler(request):
        seen.append(request)
        entered.set()
        await release.wait()
        return httpx.Response(200, json=payload("openai-chat"))

    root = native_models.make(handler=handler)

    def fail(task, pool):
        assert pool is root._lease
        dispatched.append(task)
        raise control

    token = _dispatch_registrar.set(fail)
    try:
        with pytest.raises(BaseException) as caught:
            await root.acall(LLMRequest(messages=[{"role": "user", "content": "x"}]))
        assert caught.value is control
        await asyncio.wait_for(entered.wait(), 3)
        assert len(dispatched) == 1 and not dispatched[0].done()
        assert root._lease.leases == 2 and not root._closed
        release.set()
        await asyncio.wait_for(asyncio.gather(*dispatched), 3)
        await asyncio.sleep(0)
        assert root._lease.leases == 1 and len(seen) == 1
    finally:
        _dispatch_registrar.reset(token)
        release.set()
        await asyncio.gather(*dispatched, return_exceptions=True)


@pytest.mark.parametrize("control_type", [BaseException, SystemExit, KeyboardInterrupt])
async def test_fifth_resource_callback_control_is_joined_data(
    native_models, monkeypatch, control_type
):
    import asyncio

    import httpx

    from nooa.unifiedllm.request import LLMRequest
    from tests.unifiedllm.test_native_projection import payload

    entered, release = asyncio.Event(), asyncio.Event()
    control = control_type("resource callback control")
    agent = Agent(llm=native_models.make())

    async def handler(request):
        entered.set()
        await release.wait()
        return httpx.Response(200, json=payload("openai-chat"))

    root = native_models.make(handler=handler)
    binding = ModelBinding(root)
    original_release = root._lease.release
    injected = []

    def failing_release():
        original_release()
        if not injected:
            injected.append(True)
            raise control

    async def invoke():
        async with binding.capture(owner=agent) as selected:
            return await selected.acall(LLMRequest(messages=[{"role": "user", "content": "x"}]))

    logical = asyncio.create_task(invoke())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        logical.cancel("caller")
        await asyncio.gather(logical, return_exceptions=True)
        # Inject only the actual provider-task callback, not LLM.acall's
        # independent prompt logical-exit lease release.
        monkeypatch.setattr(root._lease, "release", failing_release)
        from nooa.runtime.model_binding import _cleanup_result

        joining = asyncio.create_task(_cleanup_result(lambda: binding._join_owner_cleanup(agent)))
        done, _ = await asyncio.wait({joining}, timeout=0.1)
        assert not done
        release.set()
        outcome = await asyncio.wait_for(joining, 3)
        assert outcome is control and root._lease.leases == 1
        assert not root._closed
    finally:
        release.set()
        await asyncio.gather(logical, return_exceptions=True)
        await agent.aclose()
        await binding.aclose()


@pytest.fixture
async def sixth_retiring_native(native_models, monkeypatch):
    from nooa.unifiedllm.request import LLMRequest

    root = native_models.make()
    binding = ModelBinding(root, owned=True)
    owners = [Agent(llm=binding), Agent(llm=binding)]
    foreign = Agent(llm=binding)
    entered, release = asyncio.Event(), asyncio.Event()
    transport = root._lease.client
    original = transport.aclose

    async def disposal():
        entered.set()
        await release.wait()
        await original()

    monkeypatch.setattr(transport, "aclose", disposal)

    async def invoke(owner):
        async with binding.capture(owner=owner) as selected:
            await selected.acall(LLMRequest(messages=[{"role": "user", "content": "x"}]))

    yield binding, root, owners, foreign, entered, release, invoke
    release.set()
    for owner in [*owners, foreign]:
        await owner.aclose()
    await binding.aclose()


async def test_sixth_foreign_partial_and_all_responsible_retirement(
    sixth_retiring_native, native_models
):
    binding, root, owners, foreign, entered, release, invoke = sixth_retiring_native
    for owner in owners:
        await invoke(owner)
    assert not binding._records and not binding._cleanup
    binding.publish(native_models.make(max_tokens=310))
    await asyncio.wait_for(entered.wait(), 3)
    assert root._closed and not binding._records
    task = next(iter(binding._cleanup))
    assert all(
        any(saved is owner for saved in binding._retirement_owners[task]) for owner in owners
    )
    # Actual Agent close invokes the unchanged legacy hook BEFORE its owner hook.
    await asyncio.wait_for(foreign.aclose(), 3)
    assert not task.done()
    closing = [asyncio.create_task(owner.aclose()) for owner in owners]
    try:
        done, _ = await asyncio.wait(set(closing), timeout=0.1)
        assert not done and binding._references == 1
        release.set()
        await asyncio.wait_for(asyncio.gather(*closing), 3)
        assert not binding._cleanup and not binding._retirement_owners
    finally:
        release.set()
        await asyncio.gather(*closing, return_exceptions=True)


async def test_sixth_current_provenance_unique_weak_identity_and_no_history(native_models):
    import gc
    import weakref

    from nooa.unifiedllm.request import LLMRequest

    class EqualOwner(Agent):
        def __eq__(self, other):
            return True

        __hash__ = None

    binding = ModelBinding(native_models.make(), owned=True)
    owner = EqualOwner(llm=binding)
    other = EqualOwner(llm=binding)
    request = LLMRequest(messages=[{"role": "user", "content": "x"}])

    async def invoke(scope_owner):
        async with binding.capture(owner=scope_owner) as selected:
            await selected.acall(request)

    try:
        baseline = set(asyncio.all_tasks())
        for _ in range(20):
            await invoke(owner)
        await invoke(other)
        provenance = binding._view_owners[id(binding.client)]
        assert len(provenance.weak) == 2 and provenance.strong is None
        assert provenance.weak[id(owner)]() is owner
        assert provenance.weak[id(other)]() is other
        references = []
        for _ in range(20):
            temporary = Agent(llm=binding)
            await invoke(temporary)
            references.append(weakref.ref(temporary))
            await temporary.aclose()
            del temporary
        await asyncio.sleep(0)
        gc.collect()
        assert all(reference() is None for reference in references)
        assert len(provenance.weak) == 2
        assert not binding._records and not binding._native
        assert not binding._cleanup and not binding._retirement_owners
        assert set(asyncio.all_tasks()) == baseline
    finally:
        await owner.aclose()
        await other.aclose()
        await binding.aclose()
    assert not binding._view_owners


async def test_sixth_final_owned_empty_ownerless_retirement_and_repeated_cancel(
    native_models, monkeypatch
):
    from nooa.runtime.model_binding import _cleanup_result
    from nooa.unifiedllm.request import LLMRequest

    root = native_models.make()
    binding = ModelBinding(root, owned=True)
    async with binding.capture() as selected:
        await selected.acall(LLMRequest(messages=[{"role": "user", "content": "x"}]))
    entered, release = asyncio.Event(), asyncio.Event()
    original = root._lease.client.aclose

    async def disposal():
        entered.set()
        await release.wait()
        await original()

    monkeypatch.setattr(root._lease.client, "aclose", disposal)
    closing = asyncio.create_task(_cleanup_result(binding.aclose))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        assert binding._closed and not binding._owned and not binding._records
        assert list(binding._retirement_owners.values()) == [()]
        for _ in range(2):
            closing.cancel("join cancelled")
            await asyncio.sleep(0)
        assert not closing.done()
        release.set()
        result = await asyncio.wait_for(closing, 3)
        assert isinstance(result, asyncio.CancelledError)
        await binding.aclose()
        assert not binding._cleanup and not binding._retirement_owners
    finally:
        release.set()
        await asyncio.gather(closing, return_exceptions=True)
        await binding.aclose()


async def test_sixth_final_owned_empty_still_drains_borrowed_native_receipts(native_models):
    import httpx

    from nooa.unifiedllm.request import LLMRequest
    from tests.unifiedllm.test_native_projection import payload

    entered, release = asyncio.Event(), asyncio.Event()

    async def handler(request):
        entered.set()
        await release.wait()
        return httpx.Response(200, json=payload("openai-chat"))

    root = native_models.make(handler=handler)
    binding = ModelBinding(root)

    async def invoke():
        async with binding.capture() as selected:
            await selected.acall(LLMRequest(messages=[{"role": "user", "content": "x"}]))

    logical = asyncio.create_task(invoke())
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        logical.cancel("prompt logical exit")
        await asyncio.gather(logical, return_exceptions=True)
        closing = asyncio.create_task(binding.aclose())
        done, _ = await asyncio.wait({closing}, timeout=0.1)
        assert not done and binding._closed and not binding._owned
        assert binding._records and binding._native and not root._closed
        release.set()
        await asyncio.wait_for(closing, 3)
        assert not binding._records and not binding._native and not binding._view_owners
        assert not root._closed and root._lease.leases == 1
    finally:
        release.set()
        await asyncio.gather(logical, *([closing] if closing else []), return_exceptions=True)
        await binding.aclose()


@pytest.mark.parametrize("control_type", [BaseException, SystemExit, KeyboardInterrupt])
async def test_sixth_retired_data_failure_keeps_responsible_identity(
    native_models, monkeypatch, control_type
):
    from nooa.runtime.model_binding import _cleanup_result
    from nooa.unifiedllm.request import LLMRequest

    root = native_models.make()
    binding = ModelBinding(root, owned=True)
    owner, foreign = Agent(llm=binding), Agent(llm=binding)
    async with binding.capture(owner=owner) as selected:
        await selected.acall(LLMRequest(messages=[{"role": "user", "content": "x"}]))
    control = control_type("retired disposal DATA")
    original = root.aclose

    async def disposal():
        await original()
        raise control

    monkeypatch.setattr(root, "aclose", disposal)
    binding.publish(native_models.make())
    tasks = tuple(binding._cleanup)
    try:
        assert len(tasks) == 1
        await asyncio.wait_for(asyncio.gather(*tasks), 3)
        assert tasks[0].exception() is None and tasks[0].result() is control
        assert tasks[0] in binding._cleanup and not binding._records
        assert binding._retirement_owners[tasks[0]][0] is owner
        await foreign.aclose()
        assert tasks[0] in binding._cleanup
        outcome = await _cleanup_result(owner.aclose)
        assert outcome is control
        assert not binding._cleanup and not binding._retirement_owners
        assert await _cleanup_result(owner.aclose) is control
    finally:
        monkeypatch.setattr(root, "aclose", original)
        await _cleanup_result(owner.aclose)
        await foreign.aclose()
        await binding.aclose()


@pytest.mark.parametrize("delegated", [False, True])
async def test_sixth_cancelled_native_retirement_promotes_before_receipt_disappears(
    native_models, monkeypatch, delegated
):
    import httpx

    from nooa.unifiedllm.request import LLMRequest
    from tests.unifiedllm.test_native_projection import payload

    entered, release = asyncio.Event(), asyncio.Event()
    disposing, disposed = asyncio.Event(), asyncio.Event()

    async def handler(request):
        entered.set()
        await release.wait()
        return httpx.Response(200, json=payload("openai-chat"))

    root = native_models.make(handler=handler)
    binding = ModelBinding(root, owned=True)
    delegate = ModelBinding(binding) if delegated else None
    owner = Agent(llm=delegate if delegated else binding)
    foreign = Agent(llm=binding)
    original = root._lease.client.aclose

    async def disposal():
        disposing.set()
        await disposed.wait()
        await original()

    monkeypatch.setattr(root._lease.client, "aclose", disposal)

    async def invoke():
        async with owner.llms["main"].capture(owner=owner) as selected:
            await selected.acall(LLMRequest(messages=[{"role": "user", "content": "x"}]))

    logical = asyncio.create_task(invoke())
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        logical.cancel("prompt logical cancellation")
        await asyncio.wait_for(asyncio.gather(logical, return_exceptions=True), 3)
        assert logical.cancelled() and binding._records and binding._native
        binding.publish(native_models.make())
        assert binding._view_owners[id(root)].strong[id(owner)] is owner
        release.set()
        await asyncio.wait_for(disposing.wait(), 3)
        assert not binding._records and not binding._native and id(root) not in binding._view_owners
        task = next(iter(binding._cleanup))
        assert any(saved is owner for saved in binding._retirement_owners[task])
        await asyncio.wait_for(foreign.aclose(), 3)
        closing = asyncio.create_task(owner.aclose())
        done, _ = await asyncio.wait({closing}, timeout=0.1)
        assert not done and not task.done()
        disposed.set()
        await asyncio.wait_for(closing, 3)
        assert not binding._cleanup and not binding._retirement_owners
    finally:
        release.set()
        disposed.set()
        await asyncio.gather(logical, *([closing] if closing else []), return_exceptions=True)
        await owner.aclose()
        await foreign.aclose()
        if delegate is not None:
            await delegate.aclose()
        await binding.aclose()


@pytest.mark.parametrize(
    "outcome", [None, RuntimeError, BaseException, SystemExit, KeyboardInterrupt]
)
async def test_sixth_ownerless_idle_join_is_independent_of_concurrent_foreign_disposal(
    native_models, monkeypatch, outcome
):
    from nooa.runtime.model_binding import _cleanup_result
    from nooa.unifiedllm.request import LLMRequest

    roots = [native_models.make() for _ in range(3)]
    binding = ModelBinding(roots[0], owned=True)
    owner = Agent(llm=binding)
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    failure = outcome("idle retirement original") if outcome else None
    originals = [root.aclose for root in roots[:2]]

    def gate_disposal(index):
        async def dispose():
            entered[index].set()
            await release[index].wait()
            await originals[index]()
            if index == 0 and failure is not None:
                raise failure

        monkeypatch.setattr(roots[index], "aclose", dispose)

    request = LLMRequest(messages=[{"role": "user", "content": "native idle scope"}])
    joining = responsible = None
    try:
        async with binding.capture() as selected:
            await selected.acall(request)
        gate_disposal(0)
        binding.publish(roots[1])
        async with binding.capture(owner=owner) as selected:
            await selected.acall(request)
        gate_disposal(1)
        binding.publish(roots[2])
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered)), 3)
        assert not binding._records and not binding._native
        idle = next(task for task in binding._cleanup if binding._retirement_owners[task] == ())
        foreign = next(task for task in binding._cleanup if task is not idle)
        assert idle in binding._ownerless_retirements
        assert foreign not in binding._ownerless_retirements
        assert binding._retirement_owners[foreign] == (owner,)
        joining = asyncio.create_task(_cleanup_result(binding._join_cleanup))
        done, _ = await asyncio.wait({joining}, timeout=0.1)
        assert not done
        release[0].set()
        assert await asyncio.wait_for(joining, 3) is failure
        assert not foreign.done() and foreign in binding._cleanup
        assert idle not in binding._cleanup and idle not in binding._ownerless_retirements
        await binding._join_cleanup()  # observed idle failure is not replayed
        responsible = asyncio.create_task(owner.aclose())
        done, _ = await asyncio.wait({responsible}, timeout=0.1)
        assert not done and not binding._records
        release[1].set()
        await asyncio.wait_for(responsible, 3)
        await binding.aclose()
        assert not binding._cleanup and not binding._retirement_owners
        assert not binding._ownerless_retirements
        assert len(native_models.seen) == 2
    finally:
        for gate in release:
            gate.set()
        await asyncio.gather(
            *[task for task in (joining, responsible) if task], return_exceptions=True
        )
        await _cleanup_result(owner.aclose)
        await _cleanup_result(binding.aclose)
        for root, original in zip(roots[:2], originals, strict=True):
            monkeypatch.setattr(root, "aclose", original)


@pytest.mark.parametrize("classification", ["mixed", "dead-owner", "missing-view", "missing-task"])
async def test_sixth_ownerless_eligibility_requires_explicit_never_owned_provenance(
    native_models, monkeypatch, classification
):
    import gc
    import weakref

    from nooa.unifiedllm.request import LLMRequest

    class Owner:
        pass

    root = native_models.make()
    binding = ModelBinding(root, owned=True)
    binding.retain()
    owner = Owner()
    reference = weakref.ref(owner)
    request = LLMRequest(messages=[{"role": "user", "content": "native provenance"}])
    if classification in {"mixed", "dead-owner"}:
        async with binding.capture(owner=owner) as selected:
            await selected.acall(request)
    async with binding.capture() as selected:
        await selected.acall(request)
    provenance = binding._view_owners[id(root)]
    if classification == "dead-owner":
        del owner
        await asyncio.sleep(0)
        gc.collect()
        assert reference() is None and not provenance.weak
        assert provenance.had_runtime_owner
    if classification == "missing-view":
        del binding._view_owners[id(root)]
    entered, release = asyncio.Event(), asyncio.Event()
    original = root.aclose

    async def dispose():
        entered.set()
        await release.wait()
        await original()

    monkeypatch.setattr(root, "aclose", dispose)
    binding.publish(native_models.make())
    await asyncio.wait_for(entered.wait(), 3)
    task = next(iter(binding._cleanup))
    if classification == "missing-task":
        assert task in binding._ownerless_retirements
        del binding._retirement_owners[task]
    else:
        assert task not in binding._ownerless_retirements
    try:
        await asyncio.wait_for(binding._join_cleanup(), 3)
        assert not task.done() and task in binding._cleanup
        if classification == "mixed":
            assert binding._retirement_owners[task] == (owner,)
        # Final release joins ALL even if no owned view remains.
        binding._drop_reference(owner=True)
        binding._drop_reference()
        assert binding._closed and not binding._owned
        final = asyncio.create_task(binding._join_cleanup())
        done, _ = await asyncio.wait({final}, timeout=0.1)
        assert not done
        release.set()
        await asyncio.wait_for(final, 3)
        assert not binding._cleanup and not binding._ownerless_retirements
        assert not binding._retirement_owners
    finally:
        release.set()
        await binding.aclose()
        if binding._references:
            await binding._release()


async def test_sixth_ownerless_generic_join_repeated_cancel_is_data_and_pruned(
    native_models, monkeypatch
):
    from nooa.runtime.model_binding import _cleanup_result
    from nooa.unifiedllm.request import LLMRequest

    root = native_models.make()
    binding = ModelBinding(root, owned=True)
    async with binding.capture() as selected:
        await selected.acall(LLMRequest(messages=[{"role": "user", "content": "cancel idle"}]))
    entered, release = asyncio.Event(), asyncio.Event()
    original = root.aclose

    async def dispose():
        entered.set()
        await release.wait()
        await original()

    monkeypatch.setattr(root, "aclose", dispose)
    binding.publish(native_models.make())
    await asyncio.wait_for(entered.wait(), 3)
    joining = asyncio.create_task(_cleanup_result(binding._join_cleanup))
    try:
        await asyncio.sleep(0)
        for _ in range(2):
            joining.cancel("repeat idle join cancel")
            await asyncio.sleep(0)
        assert not joining.done() and binding._ownerless_retirements
        release.set()
        result = await asyncio.wait_for(joining, 3)
        assert isinstance(result, asyncio.CancelledError) and joining.exception() is None
        await binding._join_cleanup()
        assert not binding._cleanup and not binding._ownerless_retirements
    finally:
        release.set()
        await asyncio.gather(joining, return_exceptions=True)
        await binding.aclose()


async def test_sixth_publication_preallocates_explicit_idle_metadata_without_live_mutation(
    native_models, monkeypatch
):
    import nooa.runtime.model_binding as module
    from nooa.unifiedllm import registry
    from nooa.unifiedllm.request import LLMRequest

    root = native_models.make()
    binding = ModelBinding(root, owned=True)
    assert not binding._view_owners[id(root)].had_runtime_owner
    candidate = binding.prepare(max_tokens=330)
    before = binding._view_owners
    owned_before = binding._owned
    publication = binding._prepare_publication(candidate)
    assert binding._view_owners is before and id(candidate) not in before
    assert binding._owned is owned_before and binding.client is root
    assert id(candidate) in publication[2]
    allocator = module._ViewOwners

    def fail():
        raise RuntimeError("provenance allocation before commit")

    monkeypatch.setattr(module, "_ViewOwners", fail)
    with pytest.raises(RuntimeError, match="before commit"):
        binding._prepare_publication(candidate)
    assert binding._view_owners is before and binding._owned is owned_before
    assert binding.client is root and not binding._cleanup
    binding._publish_prepared(publication)  # no allocator called after commit
    assert binding.client is candidate and binding._view_owners is publication[2]
    monkeypatch.setattr(module, "_ViewOwners", allocator)
    binding._collect()
    await binding._join_cleanup()
    assert root._closed
    monkeypatch.setitem(
        registry.MODELS,
        "explicit-idle-alias",
        {
            "adapter": "openai-chat",
            "model": "fixture-model",
            "client": native_models.sdks[0],
            "api_key": "fixture-key",
            "base_url": "https://fixture.invalid/v1",
        },
    )
    lazy = ModelBinding("explicit-idle-alias")
    try:
        selected = lazy.client
        assert id(selected) in lazy._view_owners
        assert not lazy._view_owners[id(selected)].had_runtime_owner
        await selected.acall(LLMRequest(messages=[{"role": "user", "content": "native lazy"}]))
        await candidate.acall(LLMRequest(messages=[{"role": "user", "content": "native prepared"}]))
    finally:
        await lazy.aclose()
        await binding.aclose()
