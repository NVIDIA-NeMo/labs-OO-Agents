# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise both real Actor paths through unmodified public native clients."""

import copy

import pytest
from pydantic import BaseModel

from nooa import Agent
from nooa.runtime.actor import _current_llm_var, _current_method_var
from nooa.unifiedllm.limits import context_limits_for
from tests.runtime.public_llm_fixtures import native_models as native_models


async def generate(agent, **params):
    async def answer():
        pass

    lt = _current_llm_var.set(agent.llm)
    mt = _current_method_var.set(answer)
    try:
        return await agent.runtime.generate(**params)
    finally:
        _current_llm_var.reset(lt)
        _current_method_var.reset(mt)


@pytest.mark.parametrize("family", ["openai-chat", "openai-responses", "anthropic-messages"])
@pytest.mark.parametrize("middleware", [False, True])
async def test_real_actor_strict_request_native_http(native_models, family, middleware):
    root = native_models.make(family)
    agent = Agent(llm=root)
    if middleware:

        async def passthrough(ctx, nxt):
            return await nxt(ctx)

        agent.event_manager.intercept("llm_call", passthrough)
    response, _ = await generate(agent, temperature=0, max_tokens=200)
    assert response.usage.input_tokens > 0
    assert agent.context_stats.reserved_output_tokens == 200
    assert agent.context_stats.model_context_window == 12800
    wire = native_models.seen[-1][1]
    assert wire["temperature"] == 0
    cap = {
        "openai-chat": "max_completion_tokens",
        "openai-responses": "max_output_tokens",
        "anthropic-messages": "max_tokens",
    }[family]
    assert wire[cap] == 200
    assert context_limits_for(root).reserved_output_tokens == 100
    assert "output_model" not in wire and "output_type" not in wire
    await agent.aclose()
    assert not root._closed


async def test_noop_middleware_does_not_render_twice(native_models, monkeypatch):
    root = native_models.make()
    agent = Agent(llm=root)
    original = agent.runtime._build_messages
    renders = []

    async def counted(*args, **kwargs):
        renders.append(kwargs["llm_client"])
        return await original(*args, **kwargs)

    monkeypatch.setattr(agent.runtime, "_build_messages", counted)

    async def passthrough(ctx, nxt):
        return await nxt(ctx)

    agent.event_manager.intercept("llm_call", passthrough)
    await generate(agent, temperature=0, max_tokens=200)
    assert len(renders) == 1
    await agent.aclose()


async def test_middleware_final_view_rebudget_and_snapshot(native_models, monkeypatch):
    root = native_models.make()
    replacement = native_models.make(context_window=24000)
    agent = Agent(llm=root)
    original = agent.runtime._build_messages
    reserves = []

    async def counted(*args, **kwargs):
        reserves.append(context_limits_for(kwargs["llm_client"]).reserved_output_tokens)
        return await original(*args, **kwargs)

    monkeypatch.setattr(agent.runtime, "_build_messages", counted)

    async def route(ctx, nxt):
        ctx.params["max_tokens"] = 300
        ctx.params["temperature"] = None
        return await nxt(ctx.model_copy(update={"client": replacement}))

    agent.event_manager.intercept("llm_call", route)
    response, _ = await generate(agent, temperature=0, max_tokens=200)
    assert reserves == [200, 300]
    assert agent.context_stats.model_context_window == 24000
    assert agent.context_stats.reserved_output_tokens == 300
    assert native_models.seen[-1][1]["max_completion_tokens"] == 300
    assert "temperature" not in native_models.seen[-1][1]
    assert response.dynamic_context is not None
    assert context_limits_for(root).reserved_output_tokens == 100
    assert context_limits_for(replacement).reserved_output_tokens == 100
    await agent.aclose()


@pytest.mark.parametrize("nested", [False, True])
async def test_middleware_changed_payload_preserved_without_stale_stats(native_models, nested):
    root = native_models.make()
    agent = Agent(llm=root)
    expected = []

    async def edit(ctx, nxt):
        if nested:
            ctx.messages.append(
                {"role": "user", "content": [{"type": "text", "text": "middleware-owned prompt"}]}
            )
        else:
            ctx = ctx.model_copy(
                update={"messages": [{"role": "user", "content": "middleware-owned prompt"}]}
            )
        expected[:] = copy.deepcopy(ctx.messages)
        ctx.params["max_tokens"] = 350
        return await nxt(ctx)

    agent.event_manager.intercept("llm_call", edit)
    response, _ = await generate(agent)
    assert native_models.seen[-1][1]["messages"] == [
        item for item in expected if isinstance(item, dict)
    ]
    assert agent.runtime._last_context_stats is None
    assert agent.runtime._last_prompt_tokens_actual == response.usage.input_tokens
    await agent.aclose()


class Answer(BaseModel):
    value: int


async def test_actual_predict_strategy_output_model_translation(native_models):
    from nooa import strategy
    from nooa.strategies.predict import PredictStrategy

    root = native_models.make()

    class AnswerAgent(Agent):
        @strategy(PredictStrategy())
        async def answer(self) -> Answer:
            """Return the answer."""
            ...

    agent = AnswerAgent(llm=root)
    result = await agent.answer()
    assert result.value == 42
    wire = native_models.seen[-1][1]
    assert wire["response_format"]["type"] == "json_schema"
    assert "output_model" not in wire
    await agent.aclose()


async def test_actual_parent_constructor_inherits_binding_and_vision_is_independent(native_models):
    from nooa import strategy
    from nooa.strategies.predict import PredictStrategy
    from nooa.unifiedllm import Need

    root, vision = native_models.make(), native_models.make()
    children = []

    class Child(Agent):
        llms = {"vision": Need.IMAGES}

    class Parent(Agent):
        @strategy(PredictStrategy())
        async def answer(self) -> Answer:
            """Answer while constructing the child in the executing parent scope."""
            child = Child(llms={"vision": vision})
            children.append(child)
            assert child.llms["main"] is self.llms["main"]
            assert child.llms["vision"] is not self.llms["main"]
            assert child.llm is self.llm
            ...

    agent = Parent(llm=root)
    try:
        assert (await agent.answer()).value == 42
        assert native_models.seen[-1][1]["model"] == "fixture-model"
    finally:
        for child in children:
            await child.aclose()
        await agent.aclose()


async def test_actual_standalone_alias_cache_owner_and_borrowed_method_override(
    native_models, monkeypatch
):
    from nooa import strategy
    from nooa.strategies.predict import PredictStrategy

    root = native_models.make()
    loads = []

    def loader(alias):
        loads.append(alias)
        return root

    monkeypatch.setattr("nooa.unifiedllm.load_llm", loader)

    @strategy(PredictStrategy(), llm="owned")
    async def answer() -> Answer:
        """Return an answer through the cached creator-owned alias."""
        ...

    assert (await answer()).value == (await answer()).value == 42
    assert loads == ["owned"]
    assert len(native_models.seen) == 2 and not root._closed
    await answer.aclose()
    await answer.aclose()
    assert root._closed
    with pytest.raises(RuntimeError, match="closed"):
        await answer()


async def test_generation_settings_warn_only_once_per_runtime(native_models):
    import warnings

    agent = Agent(llm=native_models.make())
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            await generate(agent, temperature=0)
            await generate(agent, temperature=0, max_tokens=200)
        assert len([w for w in caught if "Generation settings" in str(w.message)]) == 1
    finally:
        await agent.aclose()


@pytest.mark.parametrize("middleware", [False, True])
@pytest.mark.parametrize("failure", ["context_length_exceeded", "image_count_exceeded"])
async def test_real_native_typed_recovery_rebinds_without_legacy_kwargs(
    native_models, failure, middleware
):
    import json

    import httpx

    from nooa.events import PythonOutput, ResultStatus
    from tests.unifiedllm.test_native_projection import payload

    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        if len(seen) == 1:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "code": failure,
                        "type": "invalid_request_error",
                        "message": "refused",
                        "prompt_tokens": 0,
                        "context_window": 8192,
                        "media_kind": "image",
                        "media_count": 1,
                        "media_count_limit": 0,
                    }
                },
            )
        return httpx.Response(200, json=payload("openai-chat"))

    root = native_models.make(handler=handler, context_window=200000, media_inputs=["image"])
    agent = Agent(llm=root)
    if failure == "image_count_exceeded":
        agent.event_manager.add(
            PythonOutput(
                tool_call_id="image",
                execution_count=1,
                execution_status=ResultStatus.COMPLETE,
                stdout="captured image",
                images=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}],
            )
        )
    if middleware:

        async def passthrough(ctx, nxt):
            return await nxt(ctx)

        agent.event_manager.intercept("llm_call", passthrough)
    try:
        await generate(agent, max_tokens=64000 if failure == "context_length_exceeded" else 200)
        assert len(seen) == 2
        caps = [wire["max_completion_tokens"] for wire in seen]
        assert caps == ([64000, 7936] if failure == "context_length_exceeded" else [200, 200])
        if failure == "image_count_exceeded":

            def image_parts(wire):
                return [
                    part
                    for message in wire["messages"]
                    if isinstance(message.get("content"), list)
                    for part in message["content"]
                    if part.get("type") == "image_url"
                ]

            assert len(image_parts(seen[0])) == 1
            assert image_parts(seen[1]) == []
        assert all("output_model" not in wire for wire in seen)
        assert context_limits_for(root).reserved_output_tokens == 100
    finally:
        await agent.aclose()


@pytest.mark.parametrize("standalone", [False, True])
async def test_cached_alias_owner_close_preserves_active_native_capture(
    native_models, monkeypatch, standalone
):
    import asyncio

    import httpx

    from nooa import strategy
    from nooa.strategies.predict import PredictStrategy
    from tests.unifiedllm.test_native_projection import payload

    started, release = asyncio.Event(), asyncio.Event()

    async def handler(request):
        started.set()
        await release.wait()
        return httpx.Response(200, json=payload("openai-chat"))

    root = native_models.make(handler=handler)
    monkeypatch.setattr("nooa.unifiedllm.load_llm", lambda alias: root)
    if standalone:

        @strategy(PredictStrategy(), llm="cached")
        async def answer() -> Answer: ...

        close = answer.aclose
    else:

        class A(Agent):
            @strategy(PredictStrategy(), llm="cached")
            async def answer(self) -> Answer: ...

        agent = A(llm=native_models.make())
        answer, close = agent.answer, agent.aclose
    task = asyncio.create_task(answer())
    close_entered = asyncio.Event()

    async def close_alias_owner():
        close_entered.set()
        await close()

    close_task = None
    try:
        await asyncio.wait_for(started.wait(), 2)
        close_task = asyncio.create_task(close_alias_owner())
        try:
            await asyncio.wait_for(close_entered.wait(), 3)
            done, _ = await asyncio.wait({close_task}, timeout=0.15)
            assert not done and not close_task.done() and not task.done() and not root._closed
            release.set()
            assert (await asyncio.wait_for(task, 3)).value == 42
            await asyncio.wait_for(close_task, 3)
            assert root._closed and root._lease.leases == 0
            assert root._lease._disposal is not None and root._lease._disposal.done()
            assert all(not http.is_closed for http in native_models.http)
        finally:
            release.set()
            await asyncio.gather(close_task, return_exceptions=True)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await close()


@pytest.mark.parametrize("cancel", [False, True])
async def test_actual_native_summary_fork_final_view_identity_receipt_and_cleanup(
    native_models, monkeypatch, cancel
):
    import asyncio
    import json

    import httpx

    from nooa.agents import TokenBudgetSummarizer
    from nooa.config.summarizer_config import TokenBudgetConfig
    from nooa.context_blocks.events import UserEvent
    from nooa.runtime.model_binding import public_base
    from nooa.unifiedllm._accounting import _accounting_receiver, _owns_delivered_response
    from tests.unifiedllm.test_native_projection import payload

    entered, release, wire_done = asyncio.Event(), asyncio.Event(), asyncio.Event()
    wire_tasks = []
    seen, visits, receipts, canonical = [], [], [], []

    async def handler(request):
        seen.append(json.loads(request.content))
        if len(seen) == 2:
            entered.set()
            task = asyncio.current_task()
            wire_tasks.append(task)
            task.add_done_callback(lambda completed: wire_done.set())
            await release.wait()
        data = payload("openai-chat", '{"value":7}')
        data["usage"]["prompt_tokens"] = 1000 if len(seen) == 1 else 40
        return httpx.Response(200, json=data)

    root = native_models.make(handler=handler)
    agent = Agent(llm=root)
    for text in ("stable fact", "another fact", "recent fact"):
        agent.event_manager.add(UserEvent(content=text))
    summarizer = TokenBudgetSummarizer.install(
        agent, config=TokenBudgetConfig(max_tokens=100, preserve_recent=1)
    )

    async def policy(ctx, nxt):
        result = await nxt(ctx)
        visits.append(result.client)
        return result

    agent.event_manager.intercept("llm_call", policy)
    original_add = agent.event_manager.add

    def add(event, *args, **kwargs):
        from nooa.unifiedllm import LLMResponse

        if isinstance(event, LLMResponse):
            canonical.append(_owns_delivered_response(event))
        return original_add(event, *args, **kwargs)

    monkeypatch.setattr(agent.event_manager, "add", add)
    from nooa.unifiedllm import Tool

    def forbidden() -> str:
        raise AssertionError("A summary must never execute tools")

    tools = [Tool(name="forbidden", callable=forbidden, description="Never execute")]

    class Item(BaseModel):
        value: int

    with _accounting_receiver(receipts.append):
        parent = asyncio.create_task(
            generate(
                agent, tools=tools, output_model=Item, max_tokens=250, prompt_cache_key="fork-shard"
            )
        )
        try:
            waiting = asyncio.create_task(entered.wait())
            await asyncio.wait((parent, waiting), timeout=3, return_when=asyncio.FIRST_COMPLETED)
            if parent.done():
                await parent
            if not waiting.done():
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)
            assert entered.is_set()
            assert not parent.done()
            assert len(seen) == 2
            parent_wire, fork_wire = seen
            assert fork_wire["messages"][:-1] == parent_wire["messages"]
            assert "Background memory compaction" in fork_wire["messages"][-1]["content"]
            assert {k: v for k, v in parent_wire.items() if k != "messages"} == {
                k: v for k, v in fork_wire.items() if k != "messages"
            }
            assert fork_wire["max_completion_tokens"] == 250
            assert fork_wire["prompt_cache_key"] == "fork-shard"
            assert fork_wire["tools"] == parent_wire["tools"]
            assert fork_wire["response_format"]["type"] == "json_schema"
            assert context_limits_for(root).reserved_output_tokens == 100
            if cancel:
                # Logical fork cancellation is prompt; explicit owner cleanup
                # separately joins this owner's actual shielded provider exit.
                close_entered = asyncio.Event()

                async def close_fork_owner():
                    close_entered.set()
                    await agent.aclose()

                close_task = asyncio.create_task(close_fork_owner())
                try:
                    await asyncio.wait_for(close_entered.wait(), 3)
                    await asyncio.wait_for(asyncio.gather(parent, return_exceptions=True), 3)
                    with pytest.raises(asyncio.CancelledError):
                        await parent
                    pool = public_base(root)._lease
                    done, _ = await asyncio.wait({close_task}, timeout=0.15)
                    assert not done and not close_task.done() and not wire_done.is_set()
                    assert not wire_tasks[0].cancelled() and pool.leases > 1
                    assert not root._closed
                    release.set()
                    await asyncio.wait_for(close_task, 3)
                    assert wire_done.is_set() and not wire_tasks[0].cancelled()
                    assert pool.leases == 1
                    assert canonical == []
                    assert summarizer._pending_summary is None
                    assert [r.outcome for r in receipts] == ["success", "cancelled"]
                    assert not root._closed
                    assert all(not http.is_closed for http in native_models.http)
                finally:
                    release.set()
                    await asyncio.gather(close_task, return_exceptions=True)
            else:
                release.set()
                response, _ = await asyncio.wait_for(parent, 3)
                assert response.content == '{"value":7}'
                assert summarizer._pending_summary == '{"value":7}'
                assert canonical == [True]  # Fork cannot overwrite the parent's receipt.
                assert len(receipts) == 2 and all(r.outcome == "success" for r in receipts)
                assert sorted(r.subtotal("input_tokens").known for r in receipts) == [40, 1000]
                assert len(visits) == 2
                assert visits[0] is visits[1]  # Exact captured final public view, no rederive.
                assert public_base(visits[0])._closed
                summarizer._apply_pending_summary()
                assert agent.event_manager["1..2"].summary_text == '{"value":7}'
        finally:
            release.set()
            if not parent.done():
                parent.cancel()
            await asyncio.gather(parent, return_exceptions=True)
            await agent.aclose()


async def test_warning_failure_releases_view_and_does_not_mark_warning_emitted(native_models):
    import warnings

    root = native_models.make()
    agent = Agent(llm=root)
    pool = root._lease
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            with pytest.raises(DeprecationWarning):
                await generate(agent, max_tokens=200)
        assert pool.leases == 1 and not native_models.seen
        assert not getattr(agent.runtime, "_runtime_settings_warned", False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            await generate(agent, max_tokens=200)
        assert len(caught) == 1 and pool.leases == 1
    finally:
        await agent.aclose()


async def test_actual_standalone_constructor_failure_releases_alias(native_models, monkeypatch):
    import nooa.standalone as standalone
    from nooa import strategy
    from nooa.strategies.predict import PredictStrategy

    root = native_models.make()
    failure = RuntimeError("stub constructor failed")
    monkeypatch.setattr("nooa.unifiedllm.load_llm", lambda alias: root)

    class Broken:
        def __init__(self, **kwargs):
            raise failure

    monkeypatch.setattr(standalone, "_get_agent_cls", lambda module: Broken)

    @strategy(PredictStrategy(), llm="owned")
    async def answer() -> Answer: ...

    try:
        with pytest.raises(RuntimeError) as caught:
            await answer()
        assert caught.value is failure
    finally:
        await answer.aclose()
    assert root._closed and root._lease.leases == 0


@pytest.mark.parametrize("unwind_order", [False, True])
async def test_summary_provenance_survives_hook_and_copied_unwind(native_models, unwind_order):
    import json

    import httpx

    from nooa.agents import TokenBudgetSummarizer
    from nooa.config.summarizer_config import TokenBudgetConfig
    from nooa.context_blocks.events import UserEvent
    from nooa.runtime.hooks import hooks_scope
    from nooa.unifiedllm import Tool
    from tests.unifiedllm.test_native_projection import payload

    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        data = payload("openai-chat", '{"value":7}')
        data["usage"]["prompt_tokens"] = 1000 if len(seen) == 1 else 40
        return httpx.Response(200, json=data)

    root, alien = native_models.make(handler=handler), native_models.make(max_tokens=700)
    agent = Agent(llm=root)
    for text in ("stable", "another", "recent"):
        agent.event_manager.add(UserEvent(content=text))

    async def policy(ctx, nxt):
        result = await nxt(ctx)
        return result.model_copy(
            update={
                "client": alien,
                "messages": [{"role": "user", "content": "UNWIND_ONLY"}],
                "params": {"max_tokens": 700, "prompt_cache_key": "UNWIND_ONLY"},
            }
        )

    if unwind_order:
        agent.event_manager.intercept("llm_call", policy)
    summarizer = TokenBudgetSummarizer.install(
        agent, config=TokenBudgetConfig(max_tokens=100, preserve_recent=1)
    )
    if not unwind_order:
        agent.event_manager.intercept("llm_call", policy)

    class Hook:
        def on_messages_built(self, *, messages, **kwargs):
            messages.append({"role": "user", "content": "HOOK_ONLY"})

    try:
        tools = [Tool(name="forbidden", callable=lambda: "never", description="Never execute")]
        with hooks_scope(Hook(), compose=False):
            await generate(agent, max_tokens=250, tools=tools, output_model=Answer)
        assert len(seen) == 2
        assert seen[1]["messages"][:-1] == seen[0]["messages"]
        assert "HOOK_ONLY" not in json.dumps(seen)
        assert "UNWIND_ONLY" not in json.dumps(seen)
        assert {k: v for k, v in seen[0].items() if k != "messages"} == {
            k: v for k, v in seen[1].items() if k != "messages"
        }
        assert seen[0]["prompt_cache_key"] and seen[0]["max_completion_tokens"] == 250
        assert context_limits_for(root).reserved_output_tokens == 100
    finally:
        await summarizer.aclose()
        await agent.aclose()


async def test_summary_shortcut_without_actual_dispatch_does_not_fork(native_models):
    from nooa.agents import TokenBudgetSummarizer
    from nooa.config.summarizer_config import TokenBudgetConfig
    from nooa.context_blocks.events import UserEvent
    from nooa.llm_types import LLMUsage
    from nooa.unifiedllm import LLMResponse

    agent = Agent(llm=native_models.make())
    for text in ("stable", "another", "recent"):
        agent.event_manager.add(UserEvent(content=text))
    summarizer = TokenBudgetSummarizer.install(
        agent, config=TokenBudgetConfig(max_tokens=100, preserve_recent=1)
    )

    async def shortcut(ctx, nxt):
        ctx.response = LLMResponse(content="shortcut", usage=LLMUsage(input_tokens=1000))
        return ctx

    agent.event_manager.intercept("llm_call", shortcut)
    try:
        response, _ = await generate(agent)
        assert response.content == "shortcut" and not native_models.seen
        assert summarizer._pending_summary is None
    finally:
        await agent.aclose()


@pytest.mark.parametrize("kind", ["coding-omitted", "interactive-omitted", "interactive-none"])
async def test_wrapper_default_choice_uses_actual_native_sdk(native_models, monkeypatch, kind):
    from nooa_cli.coding.agent import CodingAgent

    from nooa.agent import INHERIT
    from nooa.interactive import InteractiveAgent

    root = native_models.make(max_tokens=250)
    monkeypatch.setattr("nooa.runtime.model_binding.load_llm", lambda alias, **kw: root)
    with pytest.raises(ValueError, match="llm=None"):
        Agent(llm=None)
    if kind == "coding-omitted":
        agent = CodingAgent()
    elif kind == "interactive-none":
        agent = InteractiveAgent(llm=None)
    else:
        agent = InteractiveAgent()
    try:
        assert agent.llm is root
        await generate(agent, max_tokens=250)
        assert any(wire["max_completion_tokens"] == 250 for _, wire in native_models.seen)
        assert INHERIT is not None
    finally:
        await agent.aclose()
    assert root._closed


@pytest.mark.parametrize(
    "label",
    [
        "A.valid_/model-1",
        "bad\nlabel",
        "bad\x1blabel",
        "éclair",
        "https://secret",
        "-bad",
        "x" * 257,
    ],
)
async def test_display_label_ascii_boundary_does_not_change_native_wire(native_models, label):
    from nooa.runtime.llm_call import model_label
    from nooa.unifiedllm.adapters._view_limits import _loaded_runtime_metadata

    root = native_models.make(model=label)
    agent = Agent(llm=root)
    try:
        await generate(agent, max_tokens=250)
        assert native_models.seen[-1][1]["model"] == label
        expected = label if label == "A.valid_/model-1" else "model"
        assert _loaded_runtime_metadata(root)["label"] == model_label(root) == expected
        assert root.adapter._settings.model == label
    finally:
        await agent.aclose()
    assert not root._closed


@pytest.mark.parametrize("middleware", [False, True])
@pytest.mark.parametrize("configured_model", ["fixture-model", "bad label ü"])
async def test_canonical_model_metadata_is_not_a_display_label(
    native_models, middleware, configured_model
):
    import httpx

    from nooa.unifiedllm.request import LLMRequest
    from tests.unifiedllm.test_native_projection import payload

    def handler(request):
        data = payload("openai-chat")
        data["model"] = "provider-resolved"
        return httpx.Response(200, json=data)

    root = native_models.make(handler=handler, model=configured_model)
    direct = await root.acall(LLMRequest(messages=[{"role": "user", "content": "control"}]))
    agent = Agent(llm=root)
    if middleware:

        async def noop(ctx, nxt):
            return await nxt(ctx)

        agent.event_manager.intercept("llm_call", noop)
    try:
        response, event_id = await generate(agent)
        assert response.model_name == direct.model_name == configured_model
        assert response.raw_response.model == direct.raw_response.model == "provider-resolved"
        assert root.adapter._settings.model == configured_model
        event = agent.event_manager[event_id]
        assert event.model_name == configured_model
    finally:
        await agent.aclose()


@pytest.mark.parametrize("middleware", [False, True])
async def test_strict_custom_blank_canonical_name_stays_blank(middleware):
    from nooa.llm_types import LLMResponse
    from nooa.unifiedllm import LLM
    from nooa.unifiedllm.adapters import Capabilities

    class BlankAdapter:
        capabilities = Capabilities(context_window=12800, max_output_tokens=100)

        async def send(self, request):
            return LLMResponse(content="strict custom", model_name="")

        def send_sync(self, request):
            raise AssertionError("sync send is not used")

    root = LLM(BlankAdapter())
    agent = Agent(llm=root)
    if middleware:

        async def noop(ctx, nxt):
            return await nxt(ctx)

        agent.event_manager.intercept("llm_call", noop)
    try:
        response, event_id = await generate(agent)
        assert response.model_name == ""
        assert agent.event_manager[event_id].model_name == ""
        assert not root._closed
    finally:
        await agent.aclose()
        await root.aclose()


@pytest.mark.parametrize("canonical", ["canonical-provider", ""])
async def test_legacy_only_missing_canonical_name_is_enriched(canonical):
    from nooa.unifiedllm import FakeLLMClient, LLMResponse

    response = LLMResponse(content="legacy", model_name=canonical)
    client = FakeLLMClient(scripted_responses=[response])
    client.model = "legacy-display"
    agent = Agent(llm=client)
    try:
        result, _ = await generate(agent)
        assert result is response
        assert result.model_name == (canonical or "legacy-display")
    finally:
        await agent.aclose()


@pytest.mark.parametrize("middleware", [False, True])
async def test_recovery_final_hook_is_detached_and_identity_stable(native_models, middleware):
    import json

    import httpx

    from nooa.runtime.hooks import hooks_scope
    from nooa.unifiedllm._call_identity import current_call_identity
    from tests.unifiedllm.test_native_projection import payload

    wires, identities, snapshots = [], [], []

    def handler(request):
        wires.append(json.loads(request.content))
        identities.append(current_call_identity())
        if len(wires) == 1:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "code": "context_length_exceeded",
                        "type": "invalid_request_error",
                        "message": "refused",
                        "prompt_tokens": 0,
                        "context_window": 8192,
                    }
                },
            )
        return httpx.Response(200, json=payload("openai-chat"))

    class Hook:
        def on_messages_built(self, *, messages, context_stats, **kwargs):
            snapshots.append((copy.deepcopy(messages), context_stats.reserved_output_tokens))
            messages.append({"role": "user", "content": "hook-must-not-reach-wire"})

    root = native_models.make(handler=handler, context_window=200000)
    agent = Agent(llm=root)
    if middleware:

        async def noop(ctx, nxt):
            return await nxt(ctx)

        agent.event_manager.intercept("llm_call", noop)
    try:
        with hooks_scope(Hook(), compose=False):
            await generate(agent, max_tokens=64000)
        assert len(wires) == len(snapshots) == 2
        assert [stats for _, stats in snapshots] == [64000, 7936]
        assert [wire["max_completion_tokens"] for wire in wires] == [64000, 7936]
        assert identities[0] == identities[1] and identities[0]
        from nooa.llm_types import CacheBoundary

        assert [wire["messages"] for wire in wires] == [
            [message for message in messages if not isinstance(message, CacheBoundary)]
            for messages, _ in snapshots
        ]
        assert all("hook-must-not-reach-wire" not in json.dumps(wire) for wire in wires)
        assert root._lease.leases == 1 and not root._closed
    finally:
        await agent.aclose()


class _FourthCodingControl(BaseException):
    pass


def _fourth_configured_alias(native_models, monkeypatch, name):
    from nooa.unifiedllm import registry

    fixture_root = native_models.make()
    monkeypatch.setitem(
        registry.MODELS,
        name,
        {
            "adapter": "openai-chat",
            "model": "fourth-owned-model",
            "api_key": "fixture-key",
            "base_url": "https://fixture.invalid/v1",
            "client": native_models.sdks[-1],
            "context_window": 12800,
            "max_tokens": 100,
        },
    )
    return fixture_root


@pytest.mark.parametrize(
    "injection", ["borrowed", "creator-binding", "configured-alias", "unused-alias"]
)
async def test_coding_fourth_real_loader_ownership_and_shared_native_use(
    native_models, monkeypatch, tmp_path, injection
):
    from nooa_cli.coding.agent import CodingAgent

    from nooa.interactive import SummarizationConfig
    from nooa.runtime.model_binding import ModelBinding

    fixture_root = _fourth_configured_alias(native_models, monkeypatch, "fourth-coding")
    creator = ModelBinding(fixture_root, owned=True) if injection == "creator-binding" else None
    source = (
        creator
        if creator is not None
        else (fixture_root if injection in {"borrowed", "unused-alias"} else "fourth-coding")
    )
    coding = CodingAgent(
        llm=source,
        llms={"spare": "fourth-coding"} if injection == "unused-alias" else None,
        cwd=tmp_path,
        libs_dir=tmp_path / "skills",
        summarization=SummarizationConfig(policy="none") if injection == "unused-alias" else None,
    )
    sharer = Agent(llm=creator or fixture_root)
    selected = None if injection == "unused-alias" else coding.llm
    try:
        if selected is not None:
            await generate(coding, max_tokens=250)
            assert native_models.seen[-1][1]["max_completion_tokens"] == 250
        await generate(sharer, max_tokens=200)
        assert native_models.seen[-1][1]["max_completion_tokens"] == 200
        await coding.close()
        await coding.close()
        assert not fixture_root._closed and fixture_root._lease.leases == 1
        await generate(sharer, max_tokens=300)
        assert native_models.seen[-1][1]["max_completion_tokens"] == 300
        if injection == "configured-alias":
            assert selected._closed and selected._lease.leases == 0
            assert selected._lease is not fixture_root._lease
        elif injection == "unused-alias":
            assert coding.llms["spare"]._current is None
        else:
            assert not selected._closed
        if creator is not None:
            assert creator.client is fixture_root and creator._references == 2
        assert all(sdk.max_retries == 9 for sdk in native_models.sdks)
        assert all(not http.is_closed for http in native_models.http)
    finally:
        await coding.close()
        await sharer.aclose()
        if creator is not None:
            await creator.aclose()
            assert fixture_root._closed and fixture_root._lease.leases == 0


@pytest.mark.parametrize("stage", ["agent", "skills", "queues", "shell"])
@pytest.mark.parametrize("kind", ["ordinary", "cancel", "custom", "systemexit"])
async def test_coding_fourth_all_stages_same_first_terminal_failure(
    native_models, monkeypatch, tmp_path, stage, kind
):
    import asyncio

    from nooa_cli.coding.agent import CodingAgent

    fixture_root = _fourth_configured_alias(native_models, monkeypatch, "fourth-stages")
    coding = CodingAgent(llm="fourth-stages", cwd=tmp_path, libs_dir=tmp_path / "skills")
    selected = coding.llm
    await generate(coding, max_tokens=250)
    failure = {
        "ordinary": RuntimeError,
        "cancel": asyncio.CancelledError,
        "custom": _FourthCodingControl,
        "systemexit": SystemExit,
    }[kind]("resource close")
    attempts = []
    agent_settled = False
    for label, owner, method in (
        ("agent", coding, "aclose"),
        ("skills", coding.skills, "aclose"),
        ("queues", coding.queue_manager, "shutdown"),
        ("shell", coding.shell, "close"),
    ):
        actual = getattr(owner, method)

        async def close(*args, actual=actual, label=label, owner=owner, method=method):
            nonlocal agent_settled
            # Agent.aclose also shuts its event queue. Inject the queued
            # stage control only at CodingAgent's later standalone guard.
            if label == "queues" and not agent_settled:
                await actual(*args)
                return
            if label == "agent":
                vars(owner).pop(method, None)
            else:
                monkeypatch.setattr(owner, method, actual)
            attempts.append(label)
            await actual(*args)
            if label == "agent":
                agent_settled = True
            if label == stage:
                raise failure

        if label == "agent":
            vars(owner)[method] = close
        else:
            monkeypatch.setattr(owner, method, close)
    for _ in range(2):
        with pytest.raises(BaseException) as caught:
            await coding.close()
        assert caught.value is failure
    assert attempts[-1] == "shell" and all(
        label in attempts for label in ("agent", "skills", "queues", "shell")
    )
    before = list(attempts)
    with pytest.raises(BaseException) as caught:
        await coding.close()
    assert caught.value is failure and attempts == before
    assert coding._coding_close_task.done() and coding._coding_close_task.result() is failure
    assert selected._closed and selected._lease.leases == 0
    assert not fixture_root._closed and fixture_root._lease.leases == 1


async def test_coding_fourth_repeated_caller_cancel_defers_and_wins_resource_error(
    native_models, monkeypatch, tmp_path
):
    import asyncio

    from nooa_cli.coding.agent import CodingAgent

    fixture_root = _fourth_configured_alias(native_models, monkeypatch, "fourth-cancel")
    coding = CodingAgent(llm="fourth-cancel", cwd=tmp_path, libs_dir=tmp_path / "skills")
    selected = coding.llm
    await generate(coding, max_tokens=250)
    actual = coding.aclose
    entered, release = asyncio.Event(), asyncio.Event()
    failure = _FourthCodingControl("secondary disposer")

    async def gated():
        entered.set()
        await release.wait()
        vars(coding).pop("aclose", None)
        await actual()
        raise failure

    vars(coding)["aclose"] = gated
    caller = asyncio.create_task(coding.close())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        caller.cancel("first standalone cancellation")
        await asyncio.sleep(0)
        caller.cancel("second standalone cancellation")
        await asyncio.sleep(0)
        assert not caller.done() and not selected._closed
        release.set()
        with pytest.raises(asyncio.CancelledError, match="first standalone cancellation"):
            await caller
        assert coding._coding_close_task.done() and coding._coding_close_task.result() is failure
        assert selected._closed and selected._lease.leases == 0
        assert not fixture_root._closed
        with pytest.raises(_FourthCodingControl) as caught:
            await coding.close()
        assert caught.value is failure
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)


async def test_coding_fourth_self_turn_guard_precedes_worker(native_models, tmp_path):
    import asyncio

    from nooa_cli.coding.agent import CodingAgent

    root = native_models.make()
    coding = CodingAgent(llm=root, cwd=tmp_path, libs_dir=tmp_path / "skills")
    # Exercise the actual task-identity guard, not a replaced property.
    coding.turns._turn = asyncio.current_task()
    try:
        with pytest.raises(RuntimeError, match="executing turn cannot"):
            await coding.close()
        assert coding._coding_close_task is None and not root._closed
    finally:
        coding.turns._turn = None
        await coding.close()


@pytest.mark.parametrize("settings", [False, True])
async def test_fifth_successful_raw_calls_prune_ephemeral_retirement(native_models, settings):
    import asyncio

    root = native_models.make()
    agent = Agent(llm=root)
    try:
        for _ in range(12):
            await generate(agent, **({"max_tokens": 230} if settings else {}))
            await asyncio.sleep(0)
            assert not agent._retired_bindings
        assert len(native_models.seen) == 12
        await agent.aclose()
        assert not root._closed and root._lease.leases == 1
    finally:
        await agent.aclose()


@pytest.mark.parametrize("settings", [False, True])
async def test_fifth_raw_final_replacement_native_close_joins_exact_client(native_models, settings):
    import asyncio
    import json

    import httpx

    from tests.unifiedllm.test_native_projection import payload

    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        asyncio.current_task().add_done_callback(lambda _: finished.set())
        entered.set()
        await release.wait()
        return httpx.Response(200, json=payload("openai-chat"))

    root = native_models.make()
    replacement = native_models.make(handler=handler)
    agent = Agent(llm=root)

    async def replace(ctx, nxt):
        return await nxt(ctx.model_copy(update={"client": replacement}))

    agent.event_manager.intercept("llm_call", replace)
    logical = asyncio.create_task(generate(agent, **({"max_tokens": 230} if settings else {})))
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        logical.cancel("exact logical cancellation")
        with pytest.raises(asyncio.CancelledError, match="exact logical cancellation"):
            await asyncio.wait_for(logical, 3)
        assert not finished.is_set() and not native_models.seen
        closing = asyncio.create_task(agent.aclose())
        done, _ = await asyncio.wait({closing}, timeout=0.1)
        assert not done and not replacement._closed and not root._closed
        release.set()
        await asyncio.wait_for(closing, 3)
        assert finished.is_set() and len(requests) == 1
        assert requests[0]["max_completion_tokens"] == (230 if settings else 100)
        assert not replacement._closed and not root._closed
        assert not agent._retired_bindings
    finally:
        release.set()
        await asyncio.gather(logical, *([closing] if closing else []), return_exceptions=True)
        await agent.aclose()


@pytest.mark.parametrize("control_type", [BaseException, SystemExit, KeyboardInterrupt])
async def test_fifth_event_drain_control_data_all_batches_and_reuse(control_type):
    import asyncio

    from nooa.runtime.event_manager import EventManager

    manager = EventManager()
    control = control_type("same drain control")
    entered, release = asyncio.Event(), asyncio.Event()
    order = []

    async def first():
        order.append("first")

    async def failing():
        order.append("control")
        raise control

    async def late():
        order.append("late")

    async def last():
        order.append("last")
        entered.set()
        await release.wait()
        manager.on_close(late)

    manager.on_close(first)
    manager.on_close(failing)
    manager.on_close(last)
    from nooa.runtime.model_binding import _cleanup_result

    a = asyncio.create_task(_cleanup_result(manager.aclose))
    b = None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        b = asyncio.create_task(_cleanup_result(manager.aclose))
        release.set()
        outcomes = await asyncio.wait_for(asyncio.gather(a, b, return_exceptions=True), 3)
        assert outcomes[0] is outcomes[1] is control
        assert order == ["last", "control", "first", "late"]
        assert manager._close_task is None and not manager._close_callbacks
        await manager.aclose()
        assert order == ["last", "control", "first", "late"]
    finally:
        release.set()
        await asyncio.gather(a, *([b] if b else []), return_exceptions=True)


async def test_fifth_event_drain_first_caller_cancel_identity_after_all_callbacks():
    import asyncio

    from nooa.runtime.event_manager import EventManager

    manager = EventManager()
    entered, release = asyncio.Event(), asyncio.Event()
    seen, caught = [], []
    control = BaseException("callback control")

    async def tail():
        seen.append("tail")
        raise control

    async def gate():
        entered.set()
        await release.wait()
        seen.append("gate")

    manager.on_close(tail)
    manager.on_close(gate)

    async def close():
        try:
            await manager.aclose()
        except BaseException as error:
            caught.append(error)
            raise

    caller = asyncio.create_task(close())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        caller.cancel("first identity")
        await asyncio.sleep(0)
        caller.cancel("second identity")
        await asyncio.sleep(0)
        assert not caller.done()
        release.set()
        result = (await asyncio.wait_for(asyncio.gather(caller, return_exceptions=True), 3))[0]
        # gather wraps cancelled tasks; the original deferred object is observed
        # by the caller itself, before task cancellation packaging.
        assert isinstance(result, asyncio.CancelledError)
        assert len(caught) == 1 and caught[0].args == ("first identity",)
        assert seen == ["gate", "tail"] and manager._close_task is None
        await manager.aclose()
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)


# Sixth repair: actual outer public-boundary DATA workers, no SDK replacements.
@pytest.fixture
async def sixth_registry(native_models, tmp_path):
    from nooa_atom.agent.factory import create_session_agent
    from nooa_atom.session.options import SessionOptions
    from nooa_atom.session.registry import SessionRegistry
    from nooa_atom.session.store import SessionStore

    made = []

    def factory(alias, workspace):
        model = native_models.make(model=alias or "initial")
        made.append(model)
        return model

    registry = SessionRegistry(
        SessionStore(tmp_path / "sessions"),
        agent_factory=create_session_agent,
        llm_factory=factory,
    )
    options = SessionOptions(
        workspace=tmp_path, agent_spec="atom_test_agents:EchoAgent", model="initial"
    )
    try:
        yield registry, options, made
    finally:
        await registry.close_all()


async def _sixth_dispatch(client):
    from nooa.unifiedllm.request import LLMRequest

    response = await client.acall(LLMRequest(messages=[{"role": "user", "content": "probe"}]))
    assert response.usage.input_tokens > 0


@pytest.mark.parametrize("stage", ["half-built", "constructed"])
@pytest.mark.parametrize("control_type", [BaseException, SystemExit, KeyboardInterrupt])
async def test_sixth_registry_outer_cleanup_workers_return_control_as_data(
    sixth_registry, stage, control_type
):
    import asyncio

    registry, options, made = sixth_registry
    original = RuntimeError("original preparation failure")
    from nooa_atom.agent.factory import _constructed_agent, create_session_agent

    control = control_type("secondary cleanup control")
    workers = []
    loop = asyncio.get_running_loop()
    prior = loop.get_task_factory()

    # Identify actual public boundaries and their DATA wrapper by callable/code
    # identity, not qualname text. Also require this registry's lexical producer.
    import inspect

    from nooa_atom.session.registry import SessionRegistry
    from nooa_atom.session.session import Session

    from nooa import Agent
    from nooa.interactive import InteractiveAgent
    from nooa.runtime.model_binding import _cleanup_result

    public_calls = (Session.close, Agent.aclose, InteractiveAgent.aclose)
    public_codes = {call.__code__ for call in public_calls}
    producer_codes = {SessionRegistry._close_half_built.__code__, SessionRegistry._build.__code__}

    def track(loop, coro, **kwargs):
        code = getattr(coro, "cr_code", None)
        frame = getattr(coro, "cr_frame", None)
        wrapped_call = frame.f_locals.get("call") if frame is not None else None
        wrapped_public = (
            code is _cleanup_result.__code__
            and getattr(wrapped_call, "__func__", None) in public_calls
        )
        public_worker = code in public_codes or wrapped_public
        producer = False
        cursor = inspect.currentframe()
        try:
            cursor = cursor.f_back
            while cursor is not None:
                if cursor.f_code in producer_codes and cursor.f_locals.get("self") is registry:
                    producer = True
                    break
                cursor = cursor.f_back
        finally:
            del cursor
        task = (
            prior(loop, coro, **kwargs)
            if prior is not None
            else asyncio.Task(coro, loop=loop, **kwargs)
        )
        if public_worker and producer:
            workers.append(task)
        return task

    async def closing():
        raise control

    async def prepare(session):
        await _sixth_dispatch(session._agent.llm)
        session._agent.event_manager.on_close(closing)
        raise original

    previous_factory = registry._agent_factory
    if stage == "constructed":

        def factory(opts, storage):
            # Run the actual default construction path and its owner hook.
            for _ in range(2):
                agent = create_session_agent(opts, storage)
                _constructed_agent.get()(agent)
                agent.event_manager.on_close(closing)
            raise original

        registry._agent_factory = factory
        # Native dispatch is real even though this case fails synchronously
        # before a Session can be returned to prepare().
        await _sixth_dispatch(registry.llm_factory("native-preflight", options.workspace))
    loop.set_task_factory(track)
    try:
        with pytest.raises(RuntimeError) as caught:
            await registry.create(options, prepare=prepare)
        assert caught.value is original
        assert not registry._reserved and not registry.sessions
        assert workers, "probe did not observe the actual outer cleanup Task"
        failures = [task.exception() for task in workers if task.done() and not task.cancelled()]
        print("stage", stage, "outer_tasks", len(workers), "raw_controls", failures)
        assert all(error is None for error in failures), "registry outer Task leaked BaseException"
        assert any(task.result() is control for task in workers), "control not retained as DATA"
    finally:
        loop.set_task_factory(prior)
        registry._agent_factory = previous_factory
