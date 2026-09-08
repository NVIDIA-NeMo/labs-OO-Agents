"""Behavioral contract for the standalone default context policy."""

from types import SimpleNamespace

import pytest

from nooa import Agent, Context, DefaultAgentView, DynamicContext, collect_context
from nooa.context_blocks import Role
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
    base = await stored_context_blocks(agent.context_manager, agent, call)
    fixed = next(block for block in base if block.key == "system_prompt")

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
    event = Task(prompt="task", call_id="outer")
    manager = SimpleNamespace(values=lambda: [event])
    query = SimpleNamespace(
        apply=lambda events, *, current_call_id: events if current_call_id == "outer" else []
    )
    agent = SimpleNamespace(event_manager=manager)
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
