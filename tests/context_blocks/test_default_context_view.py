# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral contract for the standalone default context policy."""

from types import SimpleNamespace

import pytest

from nooa import Agent, Block, Context, DynamicContext, collect_context_items
from nooa.context_blocks import Role
from nooa.default_context_view import (
    ManagedContextView,
    agent_interface_block,
    agent_state_block,
    apply_block_overrides,
    materialize_manager_blocks,
    order_blocks,
    system_prompt_block,
    visible_events,
)
from nooa.events import Task
from nooa.strategies.current_call import CurrentCall


def _call(agent, *, strategy=None, decorator=None, scoped=None):
    return CurrentCall(
        id="view-call",
        method_name="run",
        decorator="plan",
        agent=agent,
        _method=getattr(type(agent), "run", None),
        _context_format=agent._truncation.context_block_format,
        _decorator_context=decorator,
        _scoped_context=scoped,
    )


class ExampleAgent(Agent, llm=object()):
    value = "dynamic value"

    async def run(self): ...


class ExampleView(ManagedContextView):
    def __init__(self, overrides=None, order=None):
        self.overrides = overrides or {}
        self.block_order = order

    async def instruction_blocks(self, owner, call):
        return self.overrides


async def test_named_helpers_make_framework_sources_explicit():
    agent = ExampleAgent()
    call = _call(agent)

    system = await system_prompt_block(agent, call)
    interface = await agent_interface_block(agent, call)
    state = await agent_state_block(agent, call)
    stored = await materialize_manager_blocks(agent.context_manager, agent, call)

    assert [system.key, interface.key, state.key] == ["system_prompt", "self", "state"]
    assert system.metadata.expr is None
    assert interface.metadata.expr == "doc(type(self))"
    assert state.metadata.expr.startswith("pformat(self")
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


async def test_assembly_does_not_write_to_the_context_manager():
    agent = ExampleAgent(context={"live": Context(expr="self.value")})
    await collect_context_items(ManagedContextView(), agent, _call(agent))

    assert agent.context_manager["live"] == Context(expr="self.value")
    with pytest.raises(KeyError):
        agent.context_manager["system_prompt"]


async def test_owner_that_is_not_an_agent_gets_no_agent_blocks():
    class MinimalManager:
        def disabled(self):
            return set()

        def declarations(self):
            return ()

    owner = SimpleNamespace(
        context_manager=MinimalManager(),
        __instance_values__=lambda: {},
        event_manager=SimpleNamespace(values=lambda: []),
    )
    assert await collect_context_items(ManagedContextView(), owner, _call(ExampleAgent())) == ()


async def test_stored_block_named_like_a_view_block_is_materialized_for_any_owner():
    from nooa.runtime.context_manager import ContextManager

    manager = ContextManager()
    manager["system_prompt"] = "standalone override"
    owner = SimpleNamespace(
        context_manager=manager,
        __instance_values__=lambda: {},
        event_manager=SimpleNamespace(values=lambda: []),
    )
    items = await collect_context_items(ManagedContextView(), owner, _call(ExampleAgent()))
    assert next(
        item for item in items if getattr(item, "key", None) == "system_prompt"
    ).content == ("standalone override")


async def test_self_context_cannot_replace_or_hide_view_blocks():
    class PromptAgent(Agent, llm=object()):
        """Real prompt."""

        async def run(self): ...

    agent = PromptAgent()
    agent.context["system_prompt"] = "pirate"
    agent.context["state"] = None
    call = _call(agent, scoped={"self": None})

    items = await collect_context_items(ManagedContextView(), agent, call)
    keys = [item.key for item in items if hasattr(item, "key")]
    prompts = [item.content for item in items if getattr(item, "key", None) == "system_prompt"]
    assert prompts == ["Real prompt.", "pirate"]
    assert {"self", "state"} <= set(keys)


async def test_state_option_omits_the_state_block():
    agent = ExampleAgent()
    items = await collect_context_items(ManagedContextView(state=False), agent, _call(agent))
    keys = {item.key for item in items if hasattr(item, "key")}
    assert "state" not in keys
    assert {"system_prompt", "self"} <= keys


async def test_malformed_system_prompt_is_materialized_as_an_error():
    class MalformedAgent(Agent, llm=object()):
        """malformed {"""

        async def run(self): ...

    agent = MalformedAgent()
    items = await collect_context_items(ManagedContextView(), agent, _call(agent))
    prompt = next(item for item in items if getattr(item, "key", None) == "system_prompt")
    assert prompt.content.startswith("ValueError:")


async def test_custom_view_replaces_the_system_prompt():
    class CustomPromptView(ManagedContextView):
        async def view_blocks(self, owner, call):
            return [
                Block(key="system_prompt", content="custom prompt")
                if block.key == "system_prompt"
                else block
                for block in await super().view_blocks(owner, call)
            ]

    agent = ExampleAgent()
    items = await collect_context_items(CustomPromptView(), agent, _call(agent))
    prompts = [item for item in items if getattr(item, "key", None) == "system_prompt"]
    assert [prompt.content for prompt in prompts] == ["custom prompt"]


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
    view = ExampleView({"strategy_prompt": "instructions"})
    call = _call(
        agent,
        decorator={"shared": "decorator"},
        scoped={"shared": "scoped", "removed": None, "strategy_prompt": None},
    )

    blocks = await collect_context_items(view, agent, call)
    by_key = {block.key: block for block in blocks if hasattr(block, "key")}
    assert by_key["shared"].content == "scoped"
    assert "removed" not in by_key
    assert by_key["strategy_prompt"].content == "instructions"


async def test_disabled_key_suppresses_user_skill_and_call_blocks():
    agent = ExampleAgent(context={"shared": "stored"})
    agent.context_manager.disable("shared")
    call = _call(
        agent,
        decorator={"shared": "decorator"},
        scoped={"shared": "scoped"},
    )

    blocks = await collect_context_items(ManagedContextView(), agent, call)
    assert "shared" not in {block.key for block in blocks if hasattr(block, "key")}


@pytest.mark.parametrize("source", ["decorator", "scoped"])
async def test_each_override_source_shares_dynamic_error_and_none_semantics(source):
    agent = ExampleAgent(context={"broken": "stored", "removed": "stored"})
    overrides = {"broken": DynamicContext("1 / 0"), "removed": None}
    call = _call(
        agent,
        decorator=overrides if source == "decorator" else None,
        scoped=overrides if source == "scoped" else None,
    )

    blocks = await collect_context_items(ManagedContextView(), agent, call)
    by_key = {block.key: block for block in blocks if hasattr(block, "key")}
    assert by_key["broken"].content == "ZeroDivisionError: division by zero"
    assert "removed" not in by_key


async def test_override_placement_inherits_or_uses_explicit_policy():
    agent = ExampleAgent()
    call = _call(agent)
    fixed = await system_prompt_block(agent, call)

    blocks = await apply_block_overrides(
        (fixed,),
        {
            "system_prompt": DynamicContext("'replacement'"),
            "new_dynamic": DynamicContext("'dynamic'"),
            "new_literal": "literal",
            "forced_suffix": Context("suffix", prefix=False),
        },
        agent=agent,
        call=call,
    )
    by_key = {block.key: block for block in blocks}

    assert by_key["system_prompt"].role is Role.SYSTEM
    assert by_key["new_dynamic"].role is Role.USER
    # A new bare value lands in the suffix, as with self.context["key"] = value.
    assert by_key["new_literal"].role is Role.USER
    assert by_key["forced_suffix"].role is Role.USER
    assert by_key["new_dynamic"].metadata.expr == "'dynamic'"
    assert by_key["new_literal"].metadata.expr is None


async def test_dynamic_manager_value_is_labelled_with_its_expression():
    agent = ExampleAgent(context={"live": DynamicContext("self.value")})
    blocks = await materialize_manager_blocks(agent.context_manager, agent, _call(agent))
    live = next(block for block in blocks if block.key == "live")
    assert live.content == "dynamic value"
    assert live.metadata.expr == "self.value"


async def test_strategy_order_lists_keys_first_and_keeps_remainder_stable():
    agent = ExampleAgent(context={"a": "a", "b": "b", "c": "c"})
    blocks = await materialize_manager_blocks(agent.context_manager, agent, _call(agent))

    ordered = order_blocks(blocks, ["c", "a"])
    keys = [block.key for block in ordered]
    assert keys[:2] == ["c", "a"]
    assert keys[2:] == [key for key in [block.key for block in blocks] if key not in {"c", "a"}]


def test_visible_events_includes_active_history_without_call_filter():
    event = Task(prompt="task", call_id="outer", tag="1")
    agent = SimpleNamespace(event_manager=SimpleNamespace(values=lambda: [event]))

    assert visible_events(agent) == (event,)


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


async def test_history_override_changes_only_replayed_events():
    import json

    from nooa import strategy
    from nooa.default_context_view import CodeActView
    from nooa.llm_types import LLMResponse, ToolCall
    from nooa.unifiedllm import FakeLLMClient

    class CurrentCallView(CodeActView):
        def history(self, agent, call):
            return (e for e in visible_events(agent) if call.id in e.related_call_ids)

    def finish(value: str, call_id: str) -> LLMResponse:
        code = f"return_result({value!r})"
        return LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id=call_id, name="execute_python", arguments=json.dumps({"code": code}))
            ],
            finish_reason="tool_calls",
        )

    llm = FakeLLMClient([finish("one", "c1"), finish("two", "c2")])

    class Solver(Agent, llm=llm):
        @strategy(context_view=CurrentCallView())
        async def solve(self, problem: str) -> str:
            """Solve the problem."""
            ...

    agent = Solver()
    await agent.solve("first problem")
    await agent.solve("second problem")

    request = str(llm.calls[1].messages)
    assert llm.calls[1].messages[0]["role"] == "system"
    assert "system_prompt" in request
    assert "<state" in request
    assert "second problem" in request
    assert "first problem" not in request


async def test_history_override_still_passes_through_replayable():
    from nooa.events import LLMResponse
    from nooa.unifiedllm import ToolCall

    incomplete = LLMResponse(
        tool_calls=[ToolCall(id="missing", name="execute_python", arguments="{}")],
        finish_reason="tool_calls",
    )

    class GivenHistory(ManagedContextView):
        def history(self, agent, call):
            return [incomplete]

    agent = ExampleAgent()
    items = await collect_context_items(GivenHistory(), agent, _call(agent))
    assert incomplete not in items
