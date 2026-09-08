# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from pydantic import ValidationError

from nooa import (
    Agent,
    Block,
    ContextView,
    DefaultAgentView,
    DefaultSkillView,
    DynamicContext,
    Skill,
    apply_context_budget,
    collect_context,
    context_text,
    evaluate_context_expression,
    resolve_context_view,
    spec,
    strategy,
)
from nooa.config.truncation_config import FormatConfig
from nooa.context_blocks import (
    OpenAIProviderFormatter,
    Role,
    UnsupportedContextLayout,
    XMLBlockFormatter,
    render_context,
)
from nooa.context_blocks.events import UserEvent
from nooa.strategies.current_call import CurrentCall


class NamedView:
    def __init__(self, name: str, *, role: Role = Role.SYSTEM):
        self.name = name
        self.role = role

    async def assemble(self, owner: Any, call: CurrentCall):
        yield Block(key=self.name, content=self.name, role=self.role)


async def test_assembly_is_stable_and_immutable():
    call = CurrentCall(id="1", method_name="run", decorator="plan")
    result = await collect_context(NamedView("one"), object(), call)
    assert isinstance(result, tuple)
    assert [item.key for item in result] == ["one"]
    with pytest.raises(ValidationError):
        result[0].content = "changed"


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


def test_agent_resolution_call_method_instance_class():
    class Example(Agent, llm=object(), context_view=NamedView("class")):
        @strategy(context_view=NamedView("method"))
        async def run(self) -> str: ...

    agent = Example(context_view=NamedView("instance"))
    assert agent.runtime._select_context_view(agent.run, NamedView("call")).name == "call"
    assert agent.runtime._select_context_view(agent.run).name == "method"

    class Undecorated(Agent, llm=object(), context_view=NamedView("class")):
        async def run(self) -> str: ...

    instance = Undecorated(context_view=NamedView("instance"))
    assert instance.runtime._select_context_view(instance.run).name == "instance"
    assert Undecorated().runtime._select_context_view(Undecorated.run).name == "class"


def test_agent_instance_replaces_class_view():
    class Example(Agent, llm=object(), context_view=NamedView("class")):
        pass

    class_view = resolve_context_view(Example(), default=DefaultAgentView())
    instance_view = resolve_context_view(
        Example(context_view=NamedView("instance")), default=DefaultAgentView()
    )
    assert class_view.name == "class"
    assert instance_view.name == "instance"


async def test_custom_agent_view_bypasses_managers_and_skills():
    class DeclaredSkill(Skill):
        context_block = ("skill", "'skill'")

    class Example(Agent, llm=object()):
        skill = DeclaredSkill()

        async def run(self): ...

    agent = Example(context={"manager": "manager"}, context_view=NamedView("only"))
    items = await agent.runtime._prepare_context(Example.run)
    assert [item.key for item in items] == ["only"]


async def test_skill_class_and_instance_view_resolution():
    class ClassSkill(Skill, context_view=NamedView("skill_class")):
        pass

    class Example(Agent, llm=object()):
        class_skill = ClassSkill()
        instance_skill = ClassSkill(context_view=NamedView("skill_instance"))

        async def run(self): ...

    agent = Example()
    event = UserEvent(content="event")
    agent.event_manager.add(event)
    items = await agent.runtime._prepare_context(Example.run)
    keys = [getattr(item, "key", None) for item in items]
    assert keys.index("skill_class") < keys.index("skill_instance") < items.index(event)


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
    assert "hidden_skill" not in [item.key for item in items]


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
    assert "registered" not in [item.key for item in inactive]

    agent.skills.activate(["test.registered"])
    active = await agent.runtime._prepare_context(Example.run)
    assert next(item for item in active if getattr(item, "key", None) == "registered").content == (
        "active"
    )

    agent.skills.deactivate(["test.registered"])
    inactive_again = await agent.runtime._prepare_context(Example.run)
    assert "registered" not in [item.key for item in inactive_again]


async def test_legacy_skill_block_is_materialized_by_default_skill_view():
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
    items = await collect_context(DefaultAgentView(), agent, call)
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
    tail_index = next(i for i, item in enumerate(items) if getattr(item, "key", None) == "tail")
    assert event_index < tail_index < skill_index


async def test_agent_view_controls_exact_skill_placement():
    class DeclaredSkill(Skill):
        context_block = ("skill", "'skill'")

    class PlacementView:
        async def assemble(self, agent, call):
            yield Block(key="before", content="before")
            for skill in agent.active_skills():
                view = resolve_context_view(skill, default=DefaultSkillView())
                async for item in view.assemble(skill, call):
                    yield item
            yield Block(key="after", content="after")

    class Example(Agent, llm=object(), context_view=PlacementView()):
        skill = DeclaredSkill()

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


def test_provider_preserves_or_rejects_layout():
    from nooa.context_blocks import AnthropicProviderFormatter

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

    with pytest.raises(UnsupportedContextLayout):
        render_context(
            items,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=AnthropicProviderFormatter(),
        )


def test_default_agent_view_satisfies_protocol():
    assert isinstance(DefaultAgentView(), ContextView)


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
        provider="call",
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
    assert (call.model, call.provider, call.context_window, call.context_budget) == (
        "call/model",
        "call",
        300,
        17,
    )


def test_default_view_has_no_runtime_assembly_dependency():
    import inspect

    from nooa.runtime.actor import ActorRuntime

    source = inspect.getsource(DefaultAgentView)
    assert "owner.runtime" not in source
    assert "_assemble_default_context" not in vars(ActorRuntime)
