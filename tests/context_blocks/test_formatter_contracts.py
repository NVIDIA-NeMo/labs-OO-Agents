# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Formatting changes presentation without changing view or native replay policy."""

import pytest

from nooa import Agent, Block, CacheBoundary, DefaultAgentView, collect_context_items, context_text
from nooa.config.truncation_config import FormatConfig
from nooa.context_blocks import (
    BlockFormatter,
    Context,
    MarkdownBlockFormatter,
    RenderConfig,
    Role,
    XMLBlockFormatter,
    render_context,
)
from nooa.context_blocks.events import EventBase, ToolCallEvent, ToolResult
from nooa.context_blocks.exceptions import UnsupportedContextLayout
from nooa.context_blocks.models import BlockPart, RenderedMessage, ResolvedBlock
from nooa.context_blocks.renderers import CachedBlockFormatter
from nooa.llm_types import AssistantReasoning, AssistantText, LLMResponse, ToolCall
from nooa.plain_formatter import PlainBlockFormatter
from nooa.strategies.current_call import CurrentCall

FORMATTERS = [XMLBlockFormatter, MarkdownBlockFormatter, CachedBlockFormatter]


class CustomWrapper(BlockFormatter):
    @property
    def format_type(self):
        return XMLBlockFormatter().format_type

    def wrap_system_block(self, block):
        return f"[{block.key}] {block.content}"

    def wrap_message_block(self, block):
        return block.content


def call_for(agent, **kwargs):
    return CurrentCall(id="format-test", method_name="run", decorator="plan", agent=agent, **kwargs)


@pytest.mark.parametrize("formatter_type", FORMATTERS)
async def test_default_guidance_matches_render_config(formatter_type):
    formatter = formatter_type()
    agent = Agent(llm=object(), render_config=RenderConfig(block_formatter=formatter))
    try:
        items = await collect_context_items(DefaultAgentView(), agent, call_for(agent))
        prompt = next(
            item.content for item in items if getattr(item, "key", None) == "system_prompt"
        )
        assert formatter.format_description() in prompt
        assert 'self.events["N"]' in formatter.format_description()
        assert prompt.count(formatter.format_description()) == 1
        if isinstance(formatter, MarkdownBlockFormatter):
            assert "XML context blocks" not in prompt
        assert "XML context blocks" not in Agent.__doc__
    finally:
        await agent.aclose()


@pytest.mark.parametrize(
    "source", ["manager", "strategy", "decorator", "scoped", "disable", "class", "resolver"]
)
async def test_default_guidance_respects_prompt_ownership(source):
    class CustomPrompt(Agent):
        """user prompt"""

    class CustomResolver(Agent):
        def _resolve_system_prompt(self):
            return "user prompt"

    class CustomStrategy:
        def get_block_overrides(self):
            return {"system_prompt": "user prompt"}

    cls = CustomPrompt if source == "class" else CustomResolver if source == "resolver" else Agent
    agent = cls(llm=object())
    kwargs = {}
    if source == "manager":
        agent.context_manager.apply_override("system_prompt", "user prompt")
    elif source == "disable":
        agent.context_manager.disable("system_prompt")
    elif source == "strategy":
        kwargs["strategy"] = CustomStrategy()
    elif source in {"decorator", "scoped"}:
        kwargs[f"_{source}_context"] = {"system_prompt": "user prompt"}
    try:
        items = await collect_context_items(DefaultAgentView(), agent, call_for(agent, **kwargs))
        prompts = [item.content for item in items if getattr(item, "key", None) == "system_prompt"]
        assert prompts == ([] if source == "disable" else ["user prompt"])
    finally:
        await agent.aclose()


@pytest.mark.parametrize("formatter_type", FORMATTERS)
async def test_custom_view_receives_no_implicit_guidance(formatter_type):
    class CustomView:
        async def assemble(self, owner, call):
            yield Block(key="custom", content="only my content", role=Role.USER)

    config = RenderConfig(block_formatter=formatter_type())
    agent = Agent(llm=object(), render_config=config, context_view=CustomView())
    try:
        items = await collect_context_items(agent.__context_view__(), agent, call_for(agent))
        assert items == (Block(key="custom", content="only my content", role=Role.USER),)
        output = render_context(
            items,
            block_formatter=config.block_formatter,
            provider_formatter=config.provider_formatter,
        ).output
        assert len(output) == 1
        assert config.block_formatter.format_description() not in output[0]["content"]
    finally:
        await agent.aclose()


@pytest.mark.parametrize("formatter_type", [*FORMATTERS, PlainBlockFormatter, CustomWrapper])
@pytest.mark.parametrize("complete", [True, False])
def test_format_changes_retain_atomic_native_replay_and_boundary(formatter_type, complete):
    arguments = ('{ "x" : 1, "x":2 }', '{"unicode":"\\u0061", "n":1.00}')
    response = LLMResponse(
        parts=(
            AssistantReasoning(
                text="retained reasoning",
                native={"encrypted_content": "opaque", "signature": "signed"},
            ),
            AssistantText(text="original assistant text"),
            *(
                ToolCall(
                    id=f"call-{i}", name="run", arguments=arg, native={"provider_id": f"native-{i}"}
                )
                for i, arg in enumerate(arguments)
            ),
        ),
        replay_scope="responses:openai:test",
        finish_reason="tool_calls",
    )
    archived = response.model_dump()
    results = [
        ToolCallEvent(
            tool_call_id=call.id,
            name=call.name,
            arguments={"execution": i},
            llm_response_id=response.id,
            result=ToolResult(tool_call_id=call.id, content=f"result-{i}"),
        )
        for i, call in enumerate(response.tool_calls)
    ]
    # Results may arrive out of call order, with unrelated content in between.
    items = [
        Block(key="fixed", content="fixed"),
        response,
        results[1],
        Block(key="middle", content="middle", role=Role.USER),
    ]
    if complete:
        items.append(results[0])
    boundary = CacheBoundary()
    items += [boundary, Block(key="live", content="live", role=Role.USER)]
    formatter = formatter_type()
    config = RenderConfig(block_formatter=formatter)
    rendered = render_context(
        items, block_formatter=formatter, provider_formatter=config.provider_formatter
    )
    output = rendered.output
    assert [item["role"] for item in output] == (
        ["system", "assistant", "tool", "tool", "user", "metadata", "user"]
        if complete
        else ["system", "user", "metadata", "user"]
    )
    assert isinstance(output[-2], CacheBoundary)
    assert output[-2] is rendered.messages[-2].replay_message
    if complete:
        assert output[1] is response
        assert [output[i]["tool_call_id"] for i in (2, 3)] == ["call-0", "call-1"]
        assert [call.arguments for call in output[1].tool_calls] == list(arguments)
        assert output[1].parts[0].native == response.parts[0].native
    else:
        assert not any(isinstance(item, LLMResponse) for item in output)
        assert not any(item["role"] == "tool" for item in output)
    assert response.model_dump() == archived
    if formatter_type is CachedBlockFormatter:
        assert formatter.format_description() == XMLBlockFormatter().format_description()
        xml_output = render_context(
            items, block_formatter=XMLBlockFormatter(), provider_formatter=config.provider_formatter
        ).output
        assert output == xml_output


def test_context_text_bounds_non_strings_but_passes_strings_unchanged():
    text = "unbounded string" * 100
    call = CurrentCall(
        id="bounded",
        method_name="run",
        decorator="plan",
        _context_format=FormatConfig(max_string=5),
    )
    assert context_text(text, call=call) is text
    assert len(context_text({"value": text}, call=call)) < len(text)


@pytest.mark.parametrize("formatter_type", [XMLBlockFormatter, MarkdownBlockFormatter])
async def test_runtime_uses_render_config_without_custom_view(formatter_type, monkeypatch):
    import nooa.agent as agent_module
    import nooa.runtime.method_wrapper as method_wrapper
    from nooa.unifiedllm import FakeLLMClient

    async def no_flush():
        pass

    monkeypatch.setattr(agent_module, "_auto_tracing_attempted", True)
    monkeypatch.setattr(method_wrapper, "_flush_litellm_journal", no_flush)
    llm = FakeLLMClient.with_tool_call("return_result", {"result": "done"})

    class ConfiguredAgent(Agent, llm=llm):
        async def run(self) -> str: ...

    formatter = formatter_type()
    agent = ConfiguredAgent(render_config=RenderConfig(block_formatter=formatter))
    try:
        assert await agent.run() == "done"
        request_text = "\n".join(message.get("content", "") for message in llm.last_messages)
        assert formatter.format_description() in request_text
        assert (
            "# System Prompt" if formatter_type is MarkdownBlockFormatter else "<system_prompt"
        ) in request_text
    finally:
        await agent.aclose()


async def test_default_view_membership_and_boundary_do_not_depend_on_format():
    agent = Agent(llm=object(), context={"notes": "live notes"})
    try:
        snapshots = []
        for formatter_type in FORMATTERS:
            agent.render_config = RenderConfig(block_formatter=formatter_type())
            items = await collect_context_items(DefaultAgentView(), agent, call_for(agent))
            snapshots.append(
                [(item.key, item.role) if isinstance(item, Block) else type(item) for item in items]
            )
        assert snapshots[0] == snapshots[1] == snapshots[2]
        assert CacheBoundary in snapshots[0]
    finally:
        await agent.aclose()


async def test_empty_description_and_explicit_source_helper_do_not_inject_guidance():
    from nooa.default_context_view import system_prompt_block

    class SilentFormatter(XMLBlockFormatter):
        def format_description(self):
            return ""

    agent = Agent(llm=object(), render_config=RenderConfig(block_formatter=SilentFormatter()))
    try:
        items = await collect_context_items(DefaultAgentView(), agent, call_for(agent))
        prompt = next(
            item.content for item in items if getattr(item, "key", None) == "system_prompt"
        )
        assert "## Context format" not in prompt
        agent.render_config = RenderConfig(block_formatter=MarkdownBlockFormatter())
        # A custom view can explicitly select the source without default-view policy.
        source = await system_prompt_block(agent, call_for(agent))
        assert agent.render_config.block_formatter.format_description() not in source.content
    finally:
        await agent.aclose()


def test_plain_formatter_preserves_ordinary_assistant_text():
    class AssistantEvent(EventBase):
        _role = Role.ASSISTANT
        content: str

    event = AssistantEvent(content="original assistant text")
    assert PlainBlockFormatter().format_event(event) == event.content


async def test_repeated_default_assembly_uses_current_formatter_once():
    agent = Agent(llm=object())
    try:
        for formatter_type in [XMLBlockFormatter, MarkdownBlockFormatter, XMLBlockFormatter]:
            formatter = formatter_type()
            agent.render_config = RenderConfig(block_formatter=formatter)
            items = await collect_context_items(DefaultAgentView(), agent, call_for(agent))
            prompt = next(
                item.content for item in items if getattr(item, "key", None) == "system_prompt"
            )
            assert prompt.count("## Context format") == 1
            assert prompt.count(formatter.format_description()) == 1
            assert ("XML context blocks" in prompt) == (formatter_type is XMLBlockFormatter)
    finally:
        await agent.aclose()


def test_provider_formatter_preserves_original_cache_marker():
    boundary = CacheBoundary()
    message = RenderedMessage(role=Role.METADATA, replay_message=boundary)
    assert RenderConfig().provider_formatter.format([message])[0] is boundary


async def test_whitespace_only_description_does_not_inject_guidance():
    class WhitespaceFormatter(XMLBlockFormatter):
        def format_description(self):
            return "   \n\t  "

    agent = Agent(llm=object(), render_config=RenderConfig(block_formatter=WhitespaceFormatter()))
    try:
        items = await collect_context_items(DefaultAgentView(), agent, call_for(agent))
        prompt = next(
            item.content for item in items if getattr(item, "key", None) == "system_prompt"
        )
        assert "## Context format" not in prompt
    finally:
        await agent.aclose()


async def test_subclass_hierarchy_prompt_guidance_ownership():
    class SubDefault(Agent):
        pass

    class CustomBase(Agent):
        """custom base prompt"""

    class SubCustom(CustomBase):
        pass

    formatter = MarkdownBlockFormatter()
    sub_default = SubDefault(llm=object(), render_config=RenderConfig(block_formatter=formatter))
    sub_custom = SubCustom(llm=object(), render_config=RenderConfig(block_formatter=formatter))
    try:
        default_items = await collect_context_items(
            DefaultAgentView(), sub_default, call_for(sub_default)
        )
        default_prompt = next(
            item.content for item in default_items if getattr(item, "key", None) == "system_prompt"
        )
        assert formatter.format_description() in default_prompt

        custom_items = await collect_context_items(
            DefaultAgentView(), sub_custom, call_for(sub_custom)
        )
        custom_prompt = next(
            item.content for item in custom_items if getattr(item, "key", None) == "system_prompt"
        )
        assert custom_prompt == "custom base prompt"
        assert formatter.format_description() not in custom_prompt
    finally:
        await sub_default.aclose()
        await sub_custom.aclose()


@pytest.mark.parametrize("formatter_type", [*FORMATTERS, PlainBlockFormatter, CustomWrapper])
def test_duplicate_tool_call_ids_in_llm_response_omits_group_atomically(formatter_type):
    response = LLMResponse(
        parts=(
            AssistantText(text="duplicate calls"),
            ToolCall(id="dup", name="run", arguments='{"a": 1}'),
            ToolCall(id="dup", name="run", arguments='{"a": 2}'),
        ),
        finish_reason="tool_calls",
    )
    execution = ToolCallEvent(
        tool_call_id="dup",
        name="run",
        arguments={"a": 1},
        llm_response_id=response.id,
        result=ToolResult(tool_call_id="dup", content="done"),
    )
    formatter = formatter_type()
    config = RenderConfig(block_formatter=formatter)
    rendered = render_context(
        [Block(key="sys", content="rules"), response, execution],
        block_formatter=formatter,
        provider_formatter=config.provider_formatter,
    )
    assert [item["role"] for item in rendered.output] == ["system"]


@pytest.mark.parametrize("formatter_type", FORMATTERS)
def test_cache_boundary_splitting_response_and_linked_result_raises(formatter_type):
    response = LLMResponse(
        parts=(ToolCall(id="call-0", name="run", arguments='{"x": 1}'),),
        finish_reason="tool_calls",
    )
    execution = ToolCallEvent(
        tool_call_id="call-0",
        name="run",
        arguments={"x": 1},
        llm_response_id=response.id,
        result=ToolResult(tool_call_id="call-0", content="ok"),
    )
    formatter = formatter_type()
    config = RenderConfig(block_formatter=formatter)
    with pytest.raises(UnsupportedContextLayout, match="CacheBoundary splits tool execution"):
        render_context(
            [response, CacheBoundary(), execution],
            block_formatter=formatter,
            provider_formatter=config.provider_formatter,
        )


def test_adjacent_materialized_blocks_coalesce_by_role_without_reordering():
    blocks = [
        ResolvedBlock(key="sys_a", content="A", role=Role.SYSTEM),
        ResolvedBlock(key="sys_b", content="B", role=Role.SYSTEM),
        ResolvedBlock(key="usr_c", content="C", role=Role.USER),
        ResolvedBlock(key="usr_d", content="D", role=Role.USER),
        ResolvedBlock(key="sys_e", content="E", role=Role.SYSTEM),
    ]
    for formatter in (XMLBlockFormatter(), MarkdownBlockFormatter(), CustomWrapper()):
        messages = formatter.format(blocks)
        assert [m.role for m in messages] == [Role.SYSTEM, Role.USER, Role.SYSTEM]
        assert [p.key for p in messages[0].parts if isinstance(p, BlockPart)] == ["sys_a", "sys_b"]
        assert [p.key for p in messages[1].parts if isinstance(p, BlockPart)] == ["usr_c", "usr_d"]
        assert [p.key for p in messages[2].parts if isinstance(p, BlockPart)] == ["sys_e"]


async def test_instance_prompt_resolver_override_owns_guidance():
    formatter = MarkdownBlockFormatter()
    agent = Agent(llm=object(), render_config=RenderConfig(block_formatter=formatter))
    # Exercise the supported instance resolver override without method binding.
    agent.__dict__["_resolve_system_prompt"] = lambda: "instance-owned prompt"
    try:
        items = await collect_context_items(DefaultAgentView(), agent, call_for(agent))
        prompt = next(
            item.content for item in items if getattr(item, "key", None) == "system_prompt"
        )
        assert prompt == "instance-owned prompt"
        assert formatter.format_description() not in prompt
    finally:
        await agent.aclose()


@pytest.mark.parametrize("source", ["strategy", "decorator", "scoped"])
@pytest.mark.parametrize(
    "replacement",
    [None, "owner-selected prompt", Context(expr="self._resolve_system_prompt()")],
    ids=["removed", "replaced", "same-text-expression"],
)
async def test_replaced_prompt_does_not_evaluate_unused_guidance(source, replacement):
    class UnusedFormatter(XMLBlockFormatter):
        def format_description(self):
            raise AssertionError("An overridden prompt must not evaluate default guidance")

    class OverrideStrategy:
        def get_block_overrides(self):
            return {"system_prompt": replacement}

    kwargs = (
        {"strategy": OverrideStrategy()}
        if source == "strategy"
        else {f"_{source}_context": {"system_prompt": replacement}}
    )
    agent = Agent(llm=object(), render_config=RenderConfig(block_formatter=UnusedFormatter()))
    try:
        items = await collect_context_items(DefaultAgentView(), agent, call_for(agent, **kwargs))
        prompts = [item.content for item in items if getattr(item, "key", None) == "system_prompt"]
        expected = (
            agent._resolve_system_prompt() if isinstance(replacement, Context) else replacement
        )
        assert prompts == ([] if expected is None else [expected])
    finally:
        await agent.aclose()
