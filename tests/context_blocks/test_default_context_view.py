# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral contract for the standalone default context policy."""

from types import SimpleNamespace

import pytest

from nooa import Agent, Context, DefaultAgentView, DynamicContext, collect_context
from nooa.context_blocks import Role
from nooa.context_blocks.exceptions import DynamicNotResolvedError
from nooa.default_context_view import (
    agent_interface_block,
    agent_state_block,
    apply_context_overrides,
    order_blocks,
    stored_context_blocks,
    system_prompt_block,
    visible_events,
)
from nooa.events import Task
from nooa.strategies.current_call import CurrentCall


def _call(agent, *, strategy=None, decorator=None, scoped=None, invocation_id=None):
    return CurrentCall(
        id="view-call",
        method_name="run",
        decorator="plan",
        agent=agent,
        strategy=strategy,
        _method=getattr(type(agent), "run", None),
        _context_format=agent._truncation.context_block_format,
        _decorator_context=decorator,
        _scoped_context=scoped,
        _context_call_id=invocation_id,
    )


class ExampleAgent(Agent, llm=object()):
    value = "dynamic value"

    async def run(self): ...


class ExampleStrategy:
    def __init__(self, overrides=None, order=None, static_keys=None):
        self.overrides = overrides or {}
        self.order = order
        self.static_keys = static_keys or set()

    def get_block_overrides(self):
        return self.overrides

    def get_block_order(self):
        return self.order

    def get_static_block_keys(self):
        return self.static_keys


async def test_named_helpers_make_framework_sources_explicit():
    agent = ExampleAgent()
    call = _call(agent)

    system = await system_prompt_block(agent, call)
    interface = await agent_interface_block(agent, call)
    state = await agent_state_block(agent, call)
    stored = await stored_context_blocks(
        agent.context_manager,
        agent,
        call,
        exclude=frozenset({"system_prompt", "self", "state"}),
    )

    assert [system.key, interface.key, state.key] == ["system_prompt", "self", "state"]
    assert all(block.metadata.source_dynamic for block in (system, interface, state))
    assert stored == ()


async def test_framework_source_helpers_do_not_read_context_manager():
    agent = ExampleAgent()
    call = _call(agent)
    expected = (
        await system_prompt_block(agent, call),
        await agent_interface_block(agent, call),
        await agent_state_block(agent, call),
    )

    class ForbiddenManager:
        def __getattribute__(self, name):
            raise AssertionError(f"context manager access: {name}")

    object.__setattr__(agent, "context_manager", ForbiddenManager())
    assert (
        await system_prompt_block(agent, call),
        await agent_interface_block(agent, call),
        await agent_state_block(agent, call),
    ) == expected


async def test_default_materialization_preserves_protected_named_reads():
    agent = ExampleAgent()
    with pytest.raises(DynamicNotResolvedError):
        agent.context_manager["system_prompt"]

    items = await collect_context(DefaultAgentView(), agent, _call(agent))
    by_key = {item.key: item for item in items if hasattr(item, "key")}
    assert agent.context_manager["system_prompt"] == by_key["system_prompt"].content
    assert agent.context_manager["self"] == by_key["self"].content
    assert agent.context_manager["state"] == by_key["state"].content


async def test_unregistered_framework_defaults_are_not_built():
    class MinimalManager:
        def disabled(self):
            return set()

        def declarations(self):
            return ()

        def update_resolved(self, resolved):
            assert resolved == {}

        def is_protected(self, key):
            return False

    owner = SimpleNamespace(
        context_manager=MinimalManager(),
        active_skills=lambda: (),
        events=SimpleNamespace(keys=lambda: [], get=lambda key: None),
    )
    assert await collect_context(DefaultAgentView(), owner, _call(ExampleAgent())) == ()


async def test_unprotected_framework_declaration_is_still_materialized():
    from nooa.runtime.context_manager import ContextManager

    manager = ContextManager()
    manager["system_prompt"] = "standalone override"
    owner = SimpleNamespace(
        context_manager=manager,
        active_skills=lambda: (),
        events=SimpleNamespace(keys=lambda: [], get=lambda key: None),
    )
    items = await collect_context(DefaultAgentView(), owner, _call(ExampleAgent()))
    assert next(
        item for item in items if getattr(item, "key", None) == "system_prompt"
    ).content == ("standalone override")


@pytest.mark.parametrize("policy", ["disabled", "override"])
async def test_unselected_system_prompt_is_not_evaluated(policy):
    class ExplodingAgent(Agent, llm=object()):
        """{self.mark_called()}"""

        calls = 0

        def mark_called(self):
            self.calls += 1
            return "unexpected"

        async def run(self): ...

    agent = ExplodingAgent()
    if policy == "disabled":
        agent.context["system_prompt"] = None
    else:
        agent.context_manager.apply_override("system_prompt", Context("replacement", prefix=True))

    items = await collect_context(DefaultAgentView(), agent, _call(agent))
    prompts = [item.content for item in items if getattr(item, "key", None) == "system_prompt"]
    assert prompts == ([] if policy == "disabled" else ["replacement"])
    assert agent.calls == 0


async def test_framework_sources_publish_named_reads_in_source_order_each_turn():
    class DependentAgent(
        Agent,
        llm=object(),
        context={"state": Context(expr='self.context["system_prompt"]')},
    ):
        """prompt-{self.version}"""

        version = "one"

        async def run(self): ...

    agent = DependentAgent()
    first = await collect_context(DefaultAgentView(), agent, _call(agent))
    first_by_key = {item.key: item for item in first if hasattr(item, "key")}
    assert first_by_key["state"].content == "prompt-one"

    agent.version = "two"
    second = await collect_context(DefaultAgentView(), agent, _call(agent))
    second_by_key = {item.key: item for item in second if hasattr(item, "key")}
    assert second_by_key["state"].content == "prompt-two"


async def test_malformed_system_prompt_is_materialized_as_an_error():
    class MalformedAgent(Agent, llm=object()):
        """malformed {"""

        async def run(self): ...

    agent = MalformedAgent()
    items = await collect_context(DefaultAgentView(), agent, _call(agent))
    prompt = next(item for item in items if getattr(item, "key", None) == "system_prompt")
    assert prompt.content.startswith("ValueError:")


async def test_interface_and_state_failures_are_materialized_as_errors():
    class BrokenInterfaceAgent(ExampleAgent):
        @classmethod
        def __type_info__(cls):
            raise RuntimeError("broken interface")

    class BrokenStateAgent(ExampleAgent):
        def __instance_values__(self):
            raise RuntimeError("broken state")

    interface = await agent_interface_block(BrokenInterfaceAgent(), _call(ExampleAgent()))
    state = await agent_state_block(BrokenStateAgent(), _call(ExampleAgent()))
    assert interface.content == "RuntimeError: broken interface"
    assert state.content == "RuntimeError: broken state"


async def test_default_source_precedence_and_removal_are_visible_policy():
    agent = ExampleAgent(context={"shared": "stored", "removed": "stored"})
    strategy = ExampleStrategy({"shared": "strategy", "removed": "strategy"})
    call = _call(
        agent,
        strategy=strategy,
        decorator={"shared": "decorator"},
        scoped={"shared": "scoped", "removed": None},
    )

    blocks = await collect_context(DefaultAgentView(), agent, call)
    by_key = {block.key: block for block in blocks if hasattr(block, "key")}
    assert by_key["shared"].content == "scoped"
    assert "removed" not in by_key


async def test_disabled_key_suppresses_every_source():
    agent = ExampleAgent(context={"shared": "stored"})
    agent.context_manager.disable("shared")
    call = _call(
        agent,
        strategy=ExampleStrategy({"shared": "strategy"}),
        decorator={"shared": "decorator"},
        scoped={"shared": "scoped"},
    )

    blocks = await collect_context(DefaultAgentView(), agent, call)
    assert "shared" not in {block.key for block in blocks if hasattr(block, "key")}


@pytest.mark.parametrize("source", ["strategy", "decorator", "scoped"])
async def test_each_override_source_shares_dynamic_error_and_none_semantics(source):
    agent = ExampleAgent(context={"broken": "stored", "removed": "stored"})
    overrides = {"broken": DynamicContext("1 / 0"), "removed": None}
    call = _call(
        agent,
        strategy=ExampleStrategy(overrides) if source == "strategy" else None,
        decorator=overrides if source == "decorator" else None,
        scoped=overrides if source == "scoped" else None,
    )

    blocks = await collect_context(DefaultAgentView(), agent, call)
    by_key = {block.key: block for block in blocks if hasattr(block, "key")}
    assert by_key["broken"].content == "ZeroDivisionError: division by zero"
    assert "removed" not in by_key


async def test_override_placement_inherits_or_uses_explicit_policy():
    agent = ExampleAgent()
    call = _call(agent)
    fixed = await system_prompt_block(agent, call)

    blocks = await apply_context_overrides(
        (fixed,),
        {
            "system_prompt": DynamicContext("'replacement'"),
            "new_dynamic": DynamicContext("'dynamic'"),
            "new_literal": "literal",
            "forced_suffix": Context("suffix", prefix=False),
        },
        agent=agent,
        call=call,
        static_expr=lambda key: key,
    )
    by_key = {block.key: block for block in blocks}

    assert by_key["system_prompt"].role is Role.SYSTEM
    assert by_key["new_dynamic"].role is Role.USER
    assert by_key["new_literal"].role is Role.SYSTEM
    assert by_key["forced_suffix"].role is Role.USER


async def test_dynamic_manager_value_updates_public_cache():
    agent = ExampleAgent(context={"live": DynamicContext("self.value")})
    call = _call(agent)

    await collect_context(DefaultAgentView(), agent, call)
    assert agent.context_manager["live"] == "dynamic value"
    blocks = await stored_context_blocks(agent.context_manager, agent, call)
    assert next(block for block in blocks if block.key == "live").metadata.source_dynamic


async def test_strategy_order_lists_keys_first_and_keeps_remainder_stable():
    agent = ExampleAgent(context={"a": "a", "b": "b", "c": "c"})
    blocks = await stored_context_blocks(agent.context_manager, agent, _call(agent))

    ordered = order_blocks(blocks, ["c", "a"])
    keys = [block.key for block in ordered]
    assert keys[:2] == ["c", "a"]
    assert keys[2:] == [key for key in [block.key for block in blocks] if key not in {"c", "a"}]


def test_visible_events_uses_public_stable_invocation_id():
    event = Task(prompt="task", metadata={"call_id": "outer"}, tag="1")
    events = SimpleNamespace(keys=lambda: ["1"], get=lambda key: event if key == "1" else None)
    query = SimpleNamespace(
        apply=lambda events, *, current_call_id: events if current_call_id == "outer" else []
    )
    agent = SimpleNamespace(events=events)
    call = CurrentCall(
        id="strategy-mutated",
        method_name="run",
        decorator="plan",
        event_query=query,
        _context_call_id="outer",
    )

    assert visible_events(agent, call) == (event,)


def test_default_module_imports_no_runtime_modules():
    import ast
    import inspect

    import nooa.default_context_view as module

    tree = ast.parse(inspect.getsource(module))
    imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    ]
    assert not any(name == "nooa.runtime" or name.startswith("nooa.runtime.") for name in imports)
