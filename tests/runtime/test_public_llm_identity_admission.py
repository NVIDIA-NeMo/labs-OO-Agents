# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Identity and admission cross the actual public SDK task boundary."""

import asyncio
import json

import httpx
import pytest

from nooa import Agent
from nooa.unifiedllm import AdmissionControl, AdmissionControlConfig
from nooa.unifiedllm._call_identity import current_call_identity
from tests.runtime.public_llm_fixtures import native_models as native_models
from tests.runtime.test_public_llm_factory_calls import generate
from tests.unifiedllm.test_native_projection import payload


@pytest.mark.parametrize("family", ["openai-chat", "openai-responses", "anthropic-messages"])
@pytest.mark.parametrize("configured", [False, True])
async def test_identity_config_precedence_and_no_leak(native_models, family, configured):
    seen = []

    def handler(request):
        seen.append((current_call_identity(), json.loads(request.content)))
        return httpx.Response(200, json=payload(family))

    settings = (
        {"prompt_cache_key": "configured"} if configured and family != "anthropic-messages" else {}
    )
    root = native_models.make(family, handler=handler, **settings)
    agents = [Agent(llm=root), Agent(llm=root)]
    try:
        await asyncio.gather(*(generate(agent, max_tokens=250) for agent in agents))
        assert all(identity for identity, _ in seen)
        assert seen[0][0] != seen[1][0]
        for identity, body in seen:
            if family == "anthropic-messages":
                assert "prompt_cache_key" not in body
            else:
                assert body["prompt_cache_key"] == ("configured" if configured else identity)
        assert current_call_identity() is None
    finally:
        for agent in agents:
            await agent.aclose()


async def test_actual_queued_cancel_has_no_http_and_next_view_same_controller(native_models):
    entered, release = asyncio.Event(), asyncio.Event()
    attempts = []

    async def handler(request):
        attempts.append(json.loads(request.content))
        entered.set()
        await release.wait()
        return httpx.Response(200, json=payload("openai-chat"))

    root = native_models.make(handler=handler)
    wrapped = AdmissionControl(
        root, AdmissionControlConfig(max_in_flight=1, concurrency_group="d2-native-queue")
    )
    first, second = Agent(llm=wrapped), Agent(llm=wrapped)
    tasks = []
    try:
        tasks.append(asyncio.create_task(generate(first, max_tokens=200)))
        await asyncio.wait_for(entered.wait(), 2)
        tasks.append(asyncio.create_task(generate(second, max_tokens=300)))
        for _ in range(10):
            await asyncio.sleep(0)
        assert len(attempts) == 1
        tasks[-1].cancel()
        with pytest.raises(asyncio.CancelledError):
            await tasks[-1]
        assert len(attempts) == 1
        release.set()
        await tasks[0]
        await generate(second, max_tokens=350)
        assert len(attempts) == 2 and attempts[-1]["max_completion_tokens"] == 350
        assert first.llm is wrapped and second.llm is wrapped
        assert current_call_identity() is None
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await first.aclose()
        await second.aclose()
        await wrapped.aclose()
    assert not root._closed


async def test_actual_retry_attempts_reacquire_release_and_keep_identity(native_models):
    events, identities = [], []

    class Permit:
        def release(self):
            events.append("release")

    class Controller:
        async def acquire(self, observer):
            events.append("acquire")
            return Permit()

    def handler(request):
        events.append("http")
        identities.append(current_call_identity())
        if len(identities) == 1:
            return httpx.Response(500, json={"error": {"message": "retry", "type": "server_error"}})
        return httpx.Response(200, json=payload("openai-chat"))

    root = native_models.make(handler=handler)
    controller = Controller()
    wrapped = AdmissionControl(root, AdmissionControlConfig(controller=controller))
    agent = Agent(llm=wrapped)
    try:
        await generate(agent, max_tokens=250)
        assert identities[0] == identities[1] and identities[0]
        assert events == ["acquire", "http", "release", "acquire", "http", "release"]
        assert wrapped.controller is controller
        assert current_call_identity() is None
    finally:
        await agent.aclose()
        await wrapped.aclose()
    assert root._closed is False


@pytest.mark.parametrize("replacement_kind", ["custom", "duck", "subclass", "facade"])
async def test_loaded_generation_rejects_nonloaded_middleware_replacement(
    native_models, replacement_kind, monkeypatch
):
    from nooa.unifiedllm import LLM, FakeLLMClient

    root = native_models.make()

    class Duck:
        async def acall(self, *args, **kwargs):
            raise AssertionError("must reject before custom dispatch")

    class Facade:
        def __getattr__(self, name):
            return getattr(root, name)

    class Subclass(LLM):
        pass

    replacement = {
        "custom": FakeLLMClient(),
        "duck": Duck(),
        "subclass": Subclass(root.adapter),
        "facade": Facade(),
    }[replacement_kind]
    agent = Agent(llm=root)
    renders = []
    original = agent.runtime._build_messages

    async def build(*args, **kwargs):
        renders.append(kwargs["llm_client"])
        return await original(*args, **kwargs)

    monkeypatch.setattr(agent.runtime, "_build_messages", build)

    async def replace(ctx, nxt):
        return await nxt(ctx.model_copy(update={"client": replacement}))

    agent.event_manager.intercept("llm_call", replace)
    try:
        with pytest.raises(TypeError):
            await generate(agent)
        assert renders == [root] and not native_models.seen and root._lease.leases == 1
    finally:
        await agent.aclose()


@pytest.mark.parametrize(
    "slot", ["_with_public_view", "acall", "_acall_once", "call", "close", "count_tokens"]
)
@pytest.mark.parametrize("on_class", [False, True])
async def test_admission_behavior_rejected_before_settings_or_http(
    native_models, monkeypatch, slot, on_class
):
    root = native_models.make()
    wrapped = AdmissionControl(root, AdmissionControlConfig(max_in_flight=1))
    target = AdmissionControl if on_class else wrapped

    def broken(*args, **kwargs):
        raise AssertionError("must not call edited bridge behavior")

    with monkeypatch.context() as edits:
        edits.setattr(target, slot, broken)
        from nooa.runtime.model_binding import derive_client

        with pytest.raises(TypeError):
            derive_client(wrapped, max_tokens=200)
        assert root._lease.leases == 1 and not native_models.seen
    await wrapped.aclose()


async def test_admission_constructor_rejects_class_edit_before_controller_lookup(
    native_models, monkeypatch
):
    root = native_models.make()

    def forbidden(*args, **kwargs):
        raise AssertionError("controller lookup must not occur")

    monkeypatch.setattr(AdmissionControlConfig, "_controller_for", forbidden)
    monkeypatch.setattr(AdmissionControl, "_with_public_view", forbidden)
    with pytest.raises(TypeError):
        AdmissionControl(root, AdmissionControlConfig(max_in_flight=1))
    assert root._lease.leases == 1 and not native_models.seen


async def test_bridge_construction_error_releases_prepared_view(native_models, monkeypatch):
    import nooa.runtime.model_binding as bindings

    root = native_models.make()
    wrapped = AdmissionControl(root, AdmissionControlConfig(max_in_flight=1))
    original = bindings.load_llm
    failure = RuntimeError("bridge construction failed")
    prepared = []

    def prepare(*args, **kwargs):
        view = original(*args, **kwargs)
        prepared.append(view)
        # Fault injection AFTER the original static proof/derivation succeeds.
        monkeypatch.setattr(
            AdmissionControl, "_with_public_view", lambda *a: (_ for _ in ()).throw(failure)
        )
        return view

    monkeypatch.setattr(bindings, "load_llm", prepare)
    with pytest.raises(RuntimeError) as caught:
        bindings.derive_client(wrapped, max_tokens=200)
    assert caught.value is failure
    assert len(prepared) == 1 and prepared[0]._closed
    assert root._lease.leases == 1 and not root._closed


async def test_repeated_cancel_keeps_actual_sdk_task_and_transport_lease(native_models):
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    sdk_tasks = []

    async def handler(request):
        sdk_tasks.append(asyncio.current_task())
        sdk_tasks[-1].add_done_callback(lambda task: finished.set())
        entered.set()
        await release.wait()
        return httpx.Response(200, json=payload("openai-chat"))

    root = native_models.make(handler=handler)
    agent = Agent(llm=root)
    task = asyncio.create_task(generate(agent, max_tokens=200))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not finished.is_set() and not sdk_tasks[0].cancelled()
        assert root._lease.leases >= 2
        release.set()
        await asyncio.wait_for(finished.wait(), 3)
        with pytest.raises(asyncio.CancelledError):
            await task
        assert root._lease.leases == 1 and not root._closed
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await agent.aclose()


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_explicit_legacy_close_delegates_but_agent_borrows_wrapper(asynchronous):
    from nooa.unifiedllm import FakeLLMClient

    class Legacy(FakeLLMClient):
        closed = 0

        def close(self):
            self.closed += 1

        async def aclose(self):
            self.closed += 1

    base = Legacy()
    wrapper = AdmissionControl(
        base, AdmissionControlConfig(max_in_flight=1, concurrency_group="legacy-close")
    )
    agent = Agent(llm=wrapper)
    await agent.aclose()
    assert base.closed == 0
    if asynchronous:
        await wrapper.aclose()
    else:
        wrapper.close()
    assert base.closed == 1


@pytest.mark.parametrize("shape", ["subclass", "changed-hook", "facade"])
async def test_explicit_wrapper_constructor_rejects_recognizable_malformed_public(
    native_models, monkeypatch, shape
):
    from nooa.unifiedllm import LLM

    root = native_models.make()
    if shape == "subclass":

        class Changed(LLM):
            pass

        client = Changed(root.adapter)
    elif shape == "changed-hook":
        client = root
        monkeypatch.setattr(client, "close", lambda: None)
    else:

        class Facade:
            def __init__(self):
                self.base_llm = root

        client = Facade()
    try:
        with pytest.raises(TypeError):
            AdmissionControl(client, AdmissionControlConfig(max_in_flight=1))
        assert not native_models.seen and not root._closed
    finally:
        if shape == "subclass":
            await client.aclose()
        elif shape == "changed-hook":
            monkeypatch.undo()


@pytest.mark.parametrize("invalid", ["none", "deleted", "noncallable"])
@pytest.mark.parametrize("direct", [False, True])
async def test_installed_controller_revalidated_before_strict_dispatch(
    native_models, invalid, direct
):
    from nooa.unifiedllm.request import LLMRequest

    root = native_models.make()
    wrapped = AdmissionControl(root, AdmissionControlConfig(max_in_flight=1))
    agent = Agent(llm=wrapped)
    controller = wrapped.controller
    if invalid == "deleted":
        del wrapped.controller
    elif invalid == "none":
        wrapped.controller = None
    else:

        class Invalid:
            acquire = None

        wrapped.controller = Invalid()
    try:
        with pytest.raises(TypeError, match="controller"):
            if direct:
                await wrapped.acall(LLMRequest(messages=[{"role": "user", "content": "test"}]))
            else:
                await generate(agent)
        assert not native_models.seen and root._lease.leases == 1
        assert not root._closed
    finally:
        wrapped.controller = controller
        await agent.aclose()


async def test_controller_removed_by_postfreeze_hook_never_sends(native_models):
    from nooa.runtime.hooks import hooks_scope

    root = native_models.make()
    wrapped = AdmissionControl(root, AdmissionControlConfig(max_in_flight=1))
    agent = Agent(llm=wrapped)
    controller = wrapped.controller

    class Hook:
        def on_messages_built(self, **kwargs):
            wrapped.controller = None

    try:
        with hooks_scope(Hook(), compose=False), pytest.raises(TypeError, match="controller"):
            await generate(agent)
        assert not native_models.seen and root._lease.leases == 1
    finally:
        wrapped.controller = controller
        await agent.aclose()


async def test_external_controller_replacement_accepts_awaitable_method(native_models):
    from nooa.unifiedllm.request import LLMRequest

    class Permit:
        def __init__(self, controller):
            self.controller = controller

        def release(self):
            self.controller.released += 1

    class External:
        def __init__(self):
            self.acquired = self.released = 0

        def acquire(self, observer):
            async def result():
                self.acquired += 1
                return Permit(self)

            return result()

    first, second = External(), External()
    root = native_models.make()
    wrapped = AdmissionControl(root, AdmissionControlConfig(controller=first))
    agent = Agent(llm=wrapped)
    try:
        await generate(agent)
        wrapped.controller = second
        await wrapped.acall(LLMRequest(messages=[{"role": "user", "content": "direct"}]))
        await generate(agent, max_tokens=200)
        assert first.acquired == first.released == 1
        assert second.acquired == second.released == 2
        assert len(native_models.seen) == 3 and root._lease.leases == 1
    finally:
        await agent.aclose()
