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
    Skill,
    resolve_context_view,
    spec,
    strategy,
)
from nooa.context_blocks import (
    OpenAIProviderFormatter,
    Role,
    UnsupportedContextLayout,
    XMLBlockFormatter,
    render_context,
)
from nooa.context_blocks.events import UserEvent
from nooa.context_view import assemble_context
from nooa.strategies.current_call import CurrentCall


class NamedView:
    def __init__(self, name: str, *, role: Role = Role.SYSTEM):
        self.name = name
        self.role = role

    async def assemble(self, owner: Any, call: CurrentCall):
        yield Block(key=self.name, content=self.name, role=self.role)


async def test_assembly_is_stable_and_immutable():
    call = CurrentCall(id="1", method_name="run", decorator="plan")
    result = await assemble_context(NamedView("one"), object(), call)
    assert isinstance(result, tuple)
    assert [item.key for item in result] == ["one"]
    with pytest.raises(ValidationError):
        result[0].content = "changed"


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

    items = await Example().runtime._prepare_context(Example.run)
    keys = [item.key for item in items]
    assert keys.index("skill_class") < keys.index("skill_instance")


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
