# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from pydantic import ValidationError

from nooa import (
    Agent,
    Block,
    CacheBoundary,
    CodeActView,
    ContextView,
    DynamicContext,
    Skill,
    apply_context_budget,
    collect_context_items,
    context_text,
    evaluate_context_expression,
    is_model_visible,
    spec,
    strategy,
)
from nooa.config.truncation_config import FormatConfig
from nooa.context_blocks import (
    OpenAIProviderFormatter,
    Role,
    XMLBlockFormatter,
    render_context,
)
from nooa.context_blocks.events import UserEvent
from nooa.default_context_view import ManagedContextView, PredictView, collect_skill_blocks
from nooa.events import DebugTrace, LLMResponse, Task
from nooa.strategies import CodeActStrategy, PredictStrategy
from nooa.strategies.current_call import CurrentCall


class NamedView:
    def __init__(self, name: str, *, role: Role = Role.SYSTEM):
        self.name = name
        self.role = role

    async def assemble(self, owner: Any, call: CurrentCall):
        yield Block(key=self.name, content=self.name, role=self.role)


async def test_assembly_is_stable_and_immutable():
    call = CurrentCall(id="1", method_name="run", decorator="plan")
    result = await collect_context_items(NamedView("one"), object(), call)
    assert isinstance(result, tuple)
    assert [item.key for item in result] == ["one"]
    with pytest.raises(ValidationError):
        result[0].content = "changed"


async def test_collection_accepts_structural_cache_boundary():
    class BoundaryView:
        async def assemble(self, owner, call):
            yield Block(key="one", content="one")
            yield CacheBoundary()

    call = CurrentCall(id="1", method_name="run", decorator="plan")
    assert await collect_context_items(BoundaryView(), object(), call) == (
        Block(key="one", content="one"),
        CacheBoundary(),
    )


async def test_collection_rejects_unknown_item_type():
    class InvalidView:
        async def assemble(self, owner, call):
            yield "not context"

    call = CurrentCall(id="1", method_name="run", decorator="plan")
    with pytest.raises(TypeError, match="Block, EventBase, or CacheBoundary"):
        await collect_context_items(InvalidView(), object(), call)


def test_context_text_uses_call_format():
    call = CurrentCall(
        id="1",
        method_name="run",
        decorator="plan",
        _context_format=FormatConfig(max_string=5, max_length=5, max_depth=2),
    )
    assert "str(len=10" in context_text({"value": "abcdefghij"}, call=call)


async def test_expression_helper_binds_the_requested_owner():
    class Example(Agent, llm=object()):
        pass

    class Owner:
        value = "owner"

    agent = Example()
    call = CurrentCall(id="1", method_name="run", decorator="plan", agent=agent)
    assert await evaluate_context_expression("self.value", owner=Owner(), call=call) == "owner"


def test_resolution_call_method_strategy_default():
    class Example(Agent, llm=object()):
        @strategy(PredictStrategy(), context_view=NamedView("method"))
        async def run(self) -> str: ...

        @strategy(PredictStrategy())
        async def predict(self) -> str: ...

    agent = Example()
    assert agent.runtime._select_context_view(agent.run, NamedView("call")).name == "call"
    assert agent.runtime._select_context_view(agent.run).name == "method"
    assert isinstance(agent.runtime._select_context_view(agent.predict), PredictView)
    assert isinstance(
        agent.runtime._select_context_view(agent.predict, strategy=CodeActStrategy()),
        CodeActView,
    )


async def test_context_view_named_method_parameter_is_not_consumed():
    from nooa.strategies import PredictStrategy
    from nooa.unifiedllm import FakeLLMClient

    client = FakeLLMClient.simple_message('"ordinary input"')

    class Example(Agent, llm=client):
        @strategy(PredictStrategy())
        async def run(self, context_view: str) -> str:
            """Return the input unchanged."""
            ...

    assert await Example().run(context_view="ordinary input") == "ordinary input"
    assert "ordinary input" in str(client.last_messages)


async def test_call_context_view_override_survives_argument_validation():
    from nooa.strategies import PredictStrategy
    from nooa.unifiedllm import FakeLLMClient

    client = FakeLLMClient.simple_message('"ok"')

    class Example(Agent, llm=client):
        @strategy(PredictStrategy(), context_view=NamedView("method"))
        async def run(self) -> str:
            """Return ok."""
            ...

    assert await Example().run(context_view=NamedView("call")) == "ok"
    rendered = str(client.last_messages)
    assert "<call>" in rendered
    assert "<method>" not in rendered


def test_agents_and_skills_do_not_register_views():
    assert not hasattr(Agent, "__context_view__")
    assert not hasattr(Skill, "__context_view__")
    with pytest.raises(TypeError):
        Agent(llm=object(), context_view=NamedView("unsupported"))
    with pytest.raises(TypeError):
        Skill(content="instructions", context_view=NamedView("unsupported"))
    with pytest.raises(TypeError):

        class InvalidSkill(Skill, context_view=NamedView("unsupported")):
            pass


def test_skill_constructor_argument_is_not_intercepted():
    class CustomSkill(Skill):
        def __init__(self, context_view):
            self.received = context_view

    value = object()
    assert CustomSkill(context_view=value).received is value


async def test_custom_agent_view_bypasses_managers_and_skills():
    class DeclaredSkill(Skill):
        context_block = ("skill", "'skill'")

    class Example(Agent, llm=object()):
        skill = DeclaredSkill()

        @strategy(context_view=NamedView("only"))
        async def run(self): ...

    agent = Example(context={"manager": "manager"})
    items = await agent.runtime._prepare_context(Example.run)
    assert [item.key for item in items] == ["only"]


async def test_method_view_composes_explicit_boundary_contribution():
    from nooa.unifiedllm import FakeLLMClient

    class BoundaryView:
        async def assemble(self, owner, call):
            yield Block(key="policy", content="policy")
            yield CacheBoundary()

    class ComposedView:
        async def assemble(self, owner, call):
            async for item in BoundaryView().assemble(owner, call):
                yield item
            yield Block(key="request", content="Return ok.", role=Role.USER)

    client = FakeLLMClient.simple_message('"ok"')

    class Example(Agent, llm=client):
        @strategy(PredictStrategy(), context_view=ComposedView())
        async def run(self) -> str: ...

    agent = Example()
    items = await agent.runtime._prepare_context(Example.run)
    assert items[1] == CacheBoundary()
    assert sum(isinstance(item, CacheBoundary) for item in items) == 1
    assert await agent.run() == "ok"


async def test_hidden_skill_contributes_nothing():
    class DeclaredSkill(Skill):
        context_block = ("hidden_skill", "'secret'")

    class Example(Agent, llm=object()):
        def __init__(self):
            super().__init__()
            self.hidden_skill = DeclaredSkill()
            spec(self, "hidden_skill", hidden=True)

        async def run(self): ...

    items = await Example().runtime._prepare_context(Example.run)
    assert "hidden_skill" not in [getattr(item, "key", None) for item in items]


async def test_registry_skill_contributes_only_while_active():
    from unittest.mock import patch

    from nooa.skill_registry import SkillRegistry

    class RegisteredSkill(Skill):
        context_block = ("registered", "'active'")

    class Example(Agent, llm=object()):
        def __init__(self):
            super().__init__()
            with patch("nooa.skill_registry.entry_points", return_value=[]):
                self.skills = SkillRegistry(self)
            self.skills.register("test.registered", RegisteredSkill())

        async def run(self): ...

    agent = Example()
    inactive = await agent.runtime._prepare_context(Example.run)
    assert "registered" not in [getattr(item, "key", None) for item in inactive]

    agent.skills.activate(["test.registered"])
    active = await agent.runtime._prepare_context(Example.run)
    assert next(item for item in active if getattr(item, "key", None) == "registered").content == (
        "active"
    )

    agent.skills.deactivate(["test.registered"])
    inactive_again = await agent.runtime._prepare_context(Example.run)
    assert "registered" not in [getattr(item, "key", None) for item in inactive_again]


async def test_hidden_active_registry_skill_contributes_nothing():
    from unittest.mock import patch

    from nooa.skill_registry import SkillRegistry

    class RegisteredSkill(Skill):
        context_block = ("registered", "'active'")

    class Example(Agent, llm=object()):
        def __init__(self):
            super().__init__()
            with patch("nooa.skill_registry.entry_points", return_value=[]):
                self.skills = SkillRegistry(self)
            self.skills.register("test.registered", RegisteredSkill())
            self.skills.activate(["test.registered"])
            spec(self, "registered", hidden=True)

        async def run(self): ...

    items = await Example().runtime._prepare_context(Example.run)
    assert "registered" not in [getattr(item, "key", None) for item in items]


async def test_skill_declaration_is_materialized_by_strategy_view():
    class DeclaredSkill(Skill):
        context_block = ("skill_state", "self.value")

    class Example(Agent, llm=object()):
        value = "ready"
        skill = DeclaredSkill()

        async def run(self): ...

    agent = Example()
    assert "skill_state" not in agent.context_manager
    items = await agent.runtime._prepare_context(Example.run)
    block = next(item for item in items if getattr(item, "key", None) == "skill_state")
    assert block.content == "ready"
    assert block.role == Role.USER
    assert block.metadata is not None
    assert block.metadata.expr == "self.value"
    assert block.metadata.source_dynamic is True


async def test_disabled_key_suppresses_default_skill_shorthand():
    class DeclaredSkill(Skill):
        context_block = ("skill_state", "self.read_skill_state()")

    class Example(Agent, llm=object()):
        skill = DeclaredSkill()

        def __init__(self):
            super().__init__()
            self.reads = 0

        def read_skill_state(self):
            self.reads += 1
            return "visible"

        async def run(self): ...

    agent = Example()
    agent.context["skill_state"] = None
    items = await agent.runtime._prepare_context(Example.run)
    assert "skill_state" not in [getattr(item, "key", None) for item in items]
    assert agent.reads == 0


@pytest.mark.parametrize("method", [False, True])
async def test_skill_discovery_does_not_reserve_active_skills_name(method):
    class DeclaredSkill(Skill):
        context_block = ("skill_state", "'visible'")

    class Example(Agent, llm=object()):
        skill = DeclaredSkill()

        async def active_skills(self) -> str:
            """A user-defined generation method, not a framework hook."""
            ...

        async def run(self): ...

    agent = Example()
    if not method:
        agent.active_skills = "a user field"
    items = await agent.runtime._prepare_context(Example.run)
    assert (
        next(item for item in items if getattr(item, "key", None) == "skill_state").content
        == "visible"
    )


async def test_protected_key_suppresses_skill_fallback_without_evaluation():
    class DeclaredSkill(Skill):
        context_block = ("state", "self.read_skill_state()")

    class Example(Agent, llm=object()):
        skill = DeclaredSkill()

        def __init__(self):
            super().__init__()
            self.reads = 0

        def read_skill_state(self):
            self.reads += 1
            return "skill state"

        async def run(self): ...

    agent = Example()
    items = await agent.runtime._prepare_context(Example.run)
    assert len([item for item in items if getattr(item, "key", None) == "state"]) == 1
    assert agent.reads == 0


async def test_manager_declaration_wins_without_evaluating_skill_fallback():
    class DeclaredSkill(Skill):
        context_block = ("skill_state", "self.read_skill_state()")

    class Example(Agent, llm=object()):
        skill = DeclaredSkill()

        def __init__(self):
            super().__init__(context={"skill_state": "explicit"})
            self.reads = 0

        def read_skill_state(self):
            self.reads += 1
            return "fallback"

        async def run(self): ...

    agent = Example()
    items = await agent.runtime._prepare_context(Example.run)
    block = next(item for item in items if getattr(item, "key", None) == "skill_state")
    assert block.content == "explicit"
    assert agent.reads == 0


async def test_default_view_partitions_and_evicts_manager_blocks():
    class Example(Agent, llm=object(), context={"tail": DynamicContext("'tail'")}):
        async def run(self): ...

    agent = Example()
    call = CurrentCall(
        id="1",
        method_name="run",
        decorator="plan",
        agent=agent,
        _method=Example.run,
        _context_format=agent._truncation.context_block_format,
        context_budget=0,
        _context_token_counter=len,
    )
    items = await collect_context_items(ManagedContextView(), agent, call)
    tail = next(block for block in items if getattr(block, "key", None) == "tail")
    assert tail.role == Role.USER
    assert tail.metadata is not None
    assert tail.metadata.static is False
    assert tail.metadata.truncated is True


async def test_default_skill_shorthand_keeps_legacy_trailing_order():
    class DeclaredSkill(Skill):
        context_block = ("skill", "'skill'")

    class Example(Agent, llm=object(), context={"tail": DynamicContext("'tail'")}):
        skill = DeclaredSkill()

        async def run(self): ...

    agent = Example()
    event = UserEvent(content="event")
    agent.event_manager.add(event)
    items = await agent.runtime._prepare_context(Example.run)
    skill_index = next(i for i, item in enumerate(items) if getattr(item, "key", None) == "skill")
    event_index = items.index(event)
    boundary_index = items.index(CacheBoundary())
    tail_index = next(i for i, item in enumerate(items) if getattr(item, "key", None) == "tail")
    assert event_index < boundary_index < tail_index < skill_index


async def test_default_view_places_one_boundary_before_trailing_context():
    class Example(Agent, llm=object(), context={"tail": DynamicContext("'tail'")}):
        async def run(self): ...

    agent = Example()
    event = UserEvent(content="event")
    agent.event_manager.add(event)
    items = await agent.runtime._prepare_context(Example.run)

    boundaries = [i for i, item in enumerate(items) if isinstance(item, CacheBoundary)]
    tail = next(i for i, item in enumerate(items) if getattr(item, "key", None) == "tail")
    assert boundaries == [items.index(event) + 1]
    assert boundaries[0] < tail


async def test_empty_llm_output_is_persisted_but_not_provider_visible():
    class Example(Agent, llm=object()):
        async def run(self): ...

    agent = Example()
    empty = LLMResponse(content="", tag="empty")
    visible = LLMResponse(content="answer", tag="visible")
    agent.event_manager.add(empty)
    agent.event_manager.add(visible)

    items = await agent.runtime._prepare_context(Example.run)

    assert empty not in items
    assert visible in items
    assert empty in agent.event_manager.values()


def test_event_helper_uses_active_history_and_filters_non_model_events():
    class Example(Agent, llm=object()):
        pass

    agent = Example()
    agent.event_manager.add(Task(prompt="archived one"))
    agent.event_manager.add(Task(prompt="archived two"))
    summary_tag = agent.events.collapse("1", "2", "active summary")
    metadata = DebugTrace(content="diagnostic")
    empty = LLMResponse(content="")
    visible = LLMResponse(content="answer")
    agent.event_manager.add(metadata)
    agent.event_manager.add(empty)
    agent.event_manager.add(visible)

    selected = tuple(e for e in agent.event_manager.values() if is_model_visible(e))

    assert [event.tag for event in selected] == [summary_tag, visible.tag]
    assert metadata not in selected
    assert empty not in selected
    assert all(
        event.prompt not in {"archived one", "archived two"}
        for event in selected
        if isinstance(event, Task)
    )


def test_event_helper_preserves_native_only_assistant_replay():
    from nooa.unifiedllm import AssistantReasoning

    class Example(Agent, llm=object()):
        pass

    native = {
        "type": "reasoning",
        "id": "rs_1",
        "encrypted_content": "opaque",
        "summary": [],
    }
    response = LLMResponse(
        parts=(AssistantReasoning(native=native),),
        replay_scope="responses:openai:test",
    )
    agent = Example()
    agent.event_manager.add(response)

    selected = tuple(e for e in agent.event_manager.values() if is_model_visible(e))
    assert selected == (response,)

    rendered = render_context(
        selected,
        block_formatter=XMLBlockFormatter(),
        provider_formatter=OpenAIProviderFormatter(),
    ).output
    assert rendered == [response]
    assert rendered[0] is response
    assert rendered[0].parts[0].native["encrypted_content"] == "opaque"


async def test_default_view_boundary_with_empty_history_and_no_trailing_context():
    class Example(Agent, llm=object()):
        async def run(self): ...

    agent = Example()
    agent.context["state"] = None
    items = await agent.runtime._prepare_context(Example.run)
    assert isinstance(items[-1], CacheBoundary)
    assert sum(isinstance(item, CacheBoundary) for item in items) == 1


async def test_custom_view_without_boundary_gets_no_implicit_boundary():
    class Example(Agent, llm=object()):
        @strategy(context_view=NamedView("only"))
        async def run(self): ...

    items = await Example().runtime._prepare_context(Example.run)
    assert not any(isinstance(item, CacheBoundary) for item in items)


async def test_agent_view_controls_exact_skill_placement():
    class DeclaredSkill(Skill):
        context_block = ("skill", "'skill'")

    class PlacementView:
        async def assemble(self, agent, call):
            yield Block(key="before", content="before")
            for item in await collect_skill_blocks(
                agent, call, disabled=set(), declaration_keys=frozenset()
            ):
                yield item
            yield Block(key="after", content="after")

    class Example(Agent, llm=object()):
        skill = DeclaredSkill()

        @strategy(context_view=PlacementView())
        async def run(self): ...

    items = await Example().runtime._prepare_context(Example.run)
    assert [item.key for item in items] == ["before", "skill", "after"]


def test_event_expansion_preserves_position():
    items = (
        Block(key="before", content="before"),
        UserEvent(content="event", tag="1"),
        Block(key="after", content="after", role=Role.USER),
    )
    output = render_context(
        items,
        block_formatter=XMLBlockFormatter(),
        provider_formatter=OpenAIProviderFormatter(),
    ).output
    assert [message["role"] for message in output] == ["system", "user", "user"]
    assert "before" in output[0]["content"]
    assert "event" in output[1]["content"]
    assert "after" in output[2]["content"]


def test_provider_formatter_preserves_layout_for_unifiedllm():
    items = (
        Block(key="first", content="first", role=Role.USER),
        Block(key="late_system", content="late", role=Role.SYSTEM),
    )
    openai = render_context(
        items,
        block_formatter=XMLBlockFormatter(),
        provider_formatter=OpenAIProviderFormatter(),
    ).output
    assert [message["role"] for message in openai] == ["user", "system"]


def test_default_agent_view_satisfies_protocol():
    assert isinstance(CodeActView(), ContextView)


def test_budget_helper_allows_adaptation_before_budgeting():
    call = CurrentCall(
        id="1",
        method_name="run",
        decorator="plan",
        model="special/model",
        context_budget=3,
        _context_token_counter=len,
    )
    original = Block(key="prompt", content="too long")
    adapted = original.model_copy(update={"content": "ok"})
    assert apply_context_budget((adapted,), call=call, evictable=(adapted,)) == (adapted,)


def test_budget_counts_replacement_notice_and_ignores_boundary():
    counted: list[str] = []

    def counter(value: str) -> int:
        counted.append(value)
        return len(value)

    call = CurrentCall(
        id="1",
        method_name="run",
        decorator="plan",
        context_budget=0,
        _context_token_counter=counter,
    )
    block = Block(key="large", content="large")
    result = apply_context_budget((block, CacheBoundary()), call=call, evictable=(block,))
    assert result[1] == CacheBoundary()
    assert result[0].content.startswith("EVICTED:")
    assert result[0].content in counted
    assert result[0].metadata.truncated is True
    assert block.metadata is None


@pytest.mark.parametrize("with_metadata", [False, True])
async def test_budget_evictions_are_counted_with_or_without_metadata(monkeypatch, with_metadata):
    from nooa.context_blocks import BlockMetadata

    metadata = BlockMetadata(expr="source", user_block=True) if with_metadata else None
    block = Block(key="large", content="x" * 1000, metadata=metadata)
    call = CurrentCall(
        id="eviction",
        method_name="run",
        decorator="plan",
        context_budget=100,
        _context_token_counter=len,
    )
    items = apply_context_budget((block,), call=call, evictable=(block,))
    assert items[0].metadata.truncated is True
    assert block.metadata is metadata
    if metadata is not None:
        assert items[0].metadata.expr == "source"
        assert items[0].metadata.user_block is True
        assert not metadata.truncated

    async def prepared(*args, **kwargs):
        return items

    agent = Agent(llm=object())
    monkeypatch.setattr(agent.runtime, "_prepare_context", prepared)
    try:
        await agent.runtime._build_messages(lambda: None)
        assert agent.runtime._last_context_stats.context_blocks_dropped == 1
    finally:
        await agent.aclose()


async def test_current_call_exposes_call_overridden_model_and_budget():
    from nooa.runtime.actor import (
        _current_call_var,
        _current_context_view_var,
        _current_llm_var,
    )

    captured: list[CurrentCall] = []

    class CaptureView:
        async def assemble(self, owner, call):
            captured.append(call)
            yield Block(key="capture", content="capture")

    class LLM:
        def __init__(self, model, window):
            self.model = model
            self.context_window = window

    class Example(Agent, llm=LLM("agent/model", 100)):
        async def run(self) -> str: ...

    agent = Example()
    resolved_call = CurrentCall(
        id="1",
        method_name="run",
        decorator="plan",
        agent=agent,
        model="call/model",
        context_window=300,
    )
    tokens = (
        (_current_call_var, _current_call_var.set(resolved_call)),
        (_current_context_view_var, _current_context_view_var.set(CaptureView())),
        (_current_llm_var, _current_llm_var.set(LLM("call/model", 300))),
    )
    try:
        await agent.runtime._prepare_context(Example.run, context_limit=17, count_tokens=len)
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)

    call = captured[-1]
    assert (call.model, call.context_window, call.context_budget) == (
        "call/model",
        300,
        17,
    )


async def test_prepare_context_refreshes_mutable_scoped_state():
    from nooa.context_blocks import ScopedContext
    from nooa.runtime.actor import _current_call_var, _current_context_view_var

    captured: list[CurrentCall] = []

    class CaptureView:
        async def assemble(self, owner, call):
            captured.append(call)
            yield Block(key="capture", content="capture")

    class Example(Agent, llm=object()):
        async def run(self) -> str: ...

    agent = Example()
    base = CurrentCall(
        id="1",
        method_name="run",
        decorator="plan",
        _scoped_context={"stale": "stale"},
    )
    call_token = _current_call_var.set(base)
    view_token = _current_context_view_var.set(CaptureView())
    try:
        with ScopedContext(context={"current": "current"}):
            await agent.runtime._prepare_context(Example.run)
    finally:
        _current_context_view_var.reset(view_token)
        _current_call_var.reset(call_token)
    assert captured[-1].scoped_context == {"current": "current"}


def test_default_view_has_no_runtime_assembly_dependency():
    import inspect

    from nooa.runtime.actor import ActorRuntime

    source = inspect.getsource(CodeActView)
    assert "owner.runtime" not in source
    assert "_assemble_default_context" not in vars(ActorRuntime)
