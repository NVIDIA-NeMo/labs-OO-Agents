# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live-context policies: how the cached renderer sends the live ``<context>``.

``dynamic`` is the existing behaviour (covered by test_cached_renderer.py).
``off`` drops the live context after the boundary. ``append`` replays each
response's stamped snapshot before it and sends a fresh one every N turns.
"""

import pytest
from pydantic import ValidationError

from nooa.context_blocks import (
    AppendedLiveContext,
    DynamicLiveContext,
    LiveContextPolicy,
    NoLiveContext,
    RenderConfig,
    coerce_live_context,
    snapshot_preface,
)
from nooa.context_blocks.events import LiveContextSnapshot, UserEvent
from nooa.context_blocks.formatter import OpenAIProviderFormatter
from nooa.context_blocks.live_context import snapshot_due
from nooa.context_blocks.models import BlockMetadata, ResolvedBlock, Role
from nooa.context_blocks.renderer import render_context
from nooa.context_blocks.renderers.cached import CachedBlockFormatter, live_blocks, snapshot_event
from nooa.llm_types import AssistantText, LLMResponse
from nooa.unifiedllm import CacheBoundary


def _static(key: str, content: str) -> ResolvedBlock:
    return ResolvedBlock(
        key=key, content=content, role=Role.SYSTEM, metadata=BlockMetadata(static=True)
    )


def _dynamic(key: str, content: str) -> ResolvedBlock:
    """A fixed user block in the volatile suffix, with the metadata the context builder
    gives it (``expr`` names how to read it back; it is not ``source_dynamic``)."""
    return ResolvedBlock(
        key=key,
        content=content,
        role=Role.SYSTEM,
        metadata=BlockMetadata(expr=f'self.context["{key}"]', static=False, user_block=True),
    )


def _expression(key: str, content: str) -> ResolvedBlock:
    """A ``set_dynamic`` / ``Context(expr=...)`` block: re-evaluated every turn."""
    return ResolvedBlock(
        key=key,
        content=content,
        role=Role.SYSTEM,
        metadata=BlockMetadata(
            expr=f"self.{key}", static=False, user_block=True, source_dynamic=True
        ),
    )


def _user(tag: str, text: str) -> ResolvedBlock:
    event = UserEvent(content=text)
    event.tag = tag
    return ResolvedBlock(
        key=f"event_{tag}", content="", role=Role.USER, metadata=BlockMetadata(tag=tag), event=event
    )


def _response(tag: str, text: str, *, dynamic_context: str = "") -> ResolvedBlock:
    event = LLMResponse(
        parts=(AssistantText(text=text),),
        tag=tag,
        dynamic_context=dynamic_context,
    )
    return ResolvedBlock(
        key=f"event_{tag}",
        content=text,
        role=Role.ASSISTANT,
        metadata=BlockMetadata(tag=tag),
        event=event,
    )


def _roles(rendered) -> list[str]:
    """Role labels of a rendered request: dicts, replayed LLMResponse objects, boundaries."""
    labels = []
    for message in rendered:
        if isinstance(message, dict):
            labels.append(message["role"])
        elif isinstance(message, LLMResponse):
            labels.append("assistant")
        else:
            assert message == CacheBoundary()
            labels.append("boundary")
    return labels


def _render_default(blocks) -> list:
    return render_context(
        blocks, block_formatter=CachedBlockFormatter(), provider_formatter=OpenAIProviderFormatter()
    ).output


def _render(blocks, policy: LiveContextPolicy) -> list:
    return render_context(
        blocks,
        block_formatter=CachedBlockFormatter(live_context=policy),
        provider_formatter=OpenAIProviderFormatter(),
    ).output


def _trailing(policy: LiveContextPolicy, blocks) -> str:
    """The trailing user message the renderer sends, as the runtime stamps it."""
    rendered = _render(blocks, policy)
    last = rendered[-1]
    assert last["role"] == "user" and "<context>" in last["content"]
    return last["content"]


BASE = [_static("sys", "You are X."), _user("1", "do the task"), _dynamic("state", "s0")]


class TestPolicy:
    def test_default_is_append_every_five_turns(self):
        assert AppendedLiveContext().every_n_turns == 5
        assert CachedBlockFormatter().live_context == AppendedLiveContext()
        assert RenderConfig().block_formatter.live_context == AppendedLiveContext()
        assert RenderConfig().live_context == AppendedLiveContext()

    def test_each_kind_carries_only_its_own_parameters(self):
        assert DynamicLiveContext().model_dump() == {"kind": "dynamic"}
        assert NoLiveContext().model_dump() == {"kind": "off"}
        assert AppendedLiveContext(every_n_turns=3).model_dump() == {
            "kind": "append",
            "every_n_turns": 3,
        }
        with pytest.raises(ValidationError):
            DynamicLiveContext(every_n_turns=3)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("dynamic", DynamicLiveContext()),
            ("off", NoLiveContext()),
            (False, NoLiveContext()),  # YAML reads a bare ``off`` as false
            ("append", AppendedLiveContext()),
            ({"kind": "append", "every_n_turns": 2}, AppendedLiveContext(every_n_turns=2)),
            ({"kind": False}, NoLiveContext()),
            (NoLiveContext(), NoLiveContext()),
        ],
    )
    def test_coercion_accepts_objects_dicts_and_kind_names(self, value, expected):
        assert coerce_live_context(value) == expected

    def test_a_bare_formatter_renders_history_then_the_boundary(self):
        """Under append the renderer adds nothing after the boundary: the live context is history."""
        rendered = _render_default(BASE)
        assert _roles(rendered) == ["system", "user", "boundary"]

    @pytest.mark.parametrize(
        "bad",
        [
            "sometimes",
            {"kind": "sometimes"},
            {"kind": "append", "every_n_turns": 0},
            {"kind": "append", "extra": 1},
            {"kind": "dynamic", "every_n_turns": 2},
            {"every_n_turns": 2},
        ],
    )
    def test_rejects_unknown_values(self, bad):
        with pytest.raises((ValidationError, ValueError)):
            coerce_live_context(bad)

    def test_render_config_builds_the_formatter_with_the_policy(self):
        policy = AppendedLiveContext(every_n_turns=4)
        config = RenderConfig(live_context=policy)
        assert isinstance(config.block_formatter, CachedBlockFormatter)
        assert config.block_formatter.live_context == policy
        assert RenderConfig(live_context={"kind": "off"}).block_formatter.live_context.kind == "off"

    def test_render_config_reads_the_policy_from_any_mapping(self):
        from collections import UserDict

        config = RenderConfig.model_validate(UserDict({"live_context": {"kind": "off"}}))
        assert config.live_context == NoLiveContext()

    def test_a_policy_and_a_formatter_together_are_refused(self):
        formatter = CachedBlockFormatter(live_context=DynamicLiveContext())
        with pytest.raises(ValidationError, match="not both"):
            RenderConfig(block_formatter=formatter, live_context=NoLiveContext())
        assert RenderConfig(block_formatter=formatter).live_context == DynamicLiveContext()


class TestOff:
    def test_history_then_boundary_and_nothing_else(self):
        rendered = _render(BASE, NoLiveContext())
        assert _roles(rendered) == ["system", "user", "boundary"]
        assert rendered[-1] == CacheBoundary()
        assert "<state>" not in rendered[0]["content"]
        assert "s0" not in str(rendered)

    def test_static_less_agents_keep_fixed_text_and_drop_expression_blocks(self):
        """Without static blocks, fixed text stays in SYSTEM; live expressions follow ``off``."""
        blocks = [_dynamic("plan", "P"), _expression("status", "busy"), _user("1", "go")]
        rendered = _render(blocks, NoLiveContext())
        assert rendered[0]["role"] == "system" and "<plan>" in rendered[0]["content"]
        assert "<status" not in str(rendered)
        assert rendered[-1] == CacheBoundary()

    def test_a_fixed_block_set_through_context_is_kept_without_static_blocks(self):
        """``context["plan"] = "..."`` gets ``expr='self.context["plan"]'`` from the builder;
        it is fixed content and ``off`` must not drop it."""
        plan = ResolvedBlock(
            key="plan",
            content="write tests",
            role=Role.SYSTEM,
            metadata=BlockMetadata(expr='self.context["plan"]', static=False, user_block=True),
        )
        rendered = _render([plan, _user("1", "go")], NoLiveContext())
        assert rendered[0]["role"] == "system" and "write tests" in rendered[0]["content"]

    def test_description_drops_the_expr_sentence(self):
        assert (
            "expr=" in CachedBlockFormatter(live_context=DynamicLiveContext()).format_description()
        )
        assert "expr=" in CachedBlockFormatter().format_description()
        assert (
            "expr=" not in CachedBlockFormatter(live_context=NoLiveContext()).format_description()
        )


class TestStaticLessFallback:
    """Agents with no static block: ``dynamic`` keeps today's all-SYSTEM shape, the
    other policies apply only to expression-backed blocks."""

    BLOCKS = [_dynamic("plan", "P"), _expression("status", "busy"), _user("1", "go")]

    def test_dynamic_puts_every_system_block_in_the_head(self):
        rendered = _render(self.BLOCKS, DynamicLiveContext())
        assert _roles(rendered) == ["system", "user"]
        assert "<plan>" in rendered[0]["content"] and "<status" in rendered[0]["content"]

    def test_append_keeps_fixed_text_and_treats_expression_blocks_as_live(self):
        rendered = _render(self.BLOCKS, AppendedLiveContext(every_n_turns=1))
        assert _roles(rendered) == ["system", "user", "boundary"]
        assert "<plan>" in rendered[0]["content"] and "<status" not in rendered[0]["content"]
        assert [b.key for b in live_blocks(self.BLOCKS)] == ["status"]


class TestAppend:
    def test_append_renders_exactly_like_off(self):
        blocks = [*BASE[:-1], _response("2", "reply"), BASE[-1]]
        assert _render(blocks, AppendedLiveContext()) == _render(blocks, NoLiveContext())
        assert _roles(_render(blocks, AppendedLiveContext())) == [
            "system",
            "user",
            "assistant",
            "boundary",
        ]

    def test_live_blocks_are_the_formatters_dynamic_partition(self):
        assert [b.key for b in live_blocks(BASE)] == ["state"]
        assert live_blocks([_static("sys", "S"), _user("1", "go")]) == []

    def test_snapshot_event_carries_the_envelope_and_each_block(self):
        event = snapshot_event(live_blocks(BASE), 5)
        assert event.content.startswith(snapshot_preface(5) + "\n<context>\n")
        assert event.content.endswith("\n</context>")
        assert event.comparable == {"state": "<state>\ns0\n</state>"}

    def test_the_snapshot_event_renders_raw_as_the_user_message_the_model_saw(self):
        event = snapshot_event(live_blocks(BASE), 5)
        event.tag = "2"
        block = ResolvedBlock(
            key="event_2", content="", role=Role.USER, metadata=BlockMetadata(tag="2"), event=event
        )
        rendered = _render([*BASE[:-1], block, BASE[-1]], AppendedLiveContext())
        assert _roles(rendered) == ["system", "user", "user", "boundary"]
        assert rendered[2]["content"] == event.content  # no <sys tag> wrapper, no repr

    def test_dynamic_stamps_from_an_old_session_are_not_replayed(self):
        """A session saved under dynamic (stamps on every response) rendered under append."""
        stamp = _trailing(DynamicLiveContext(), BASE)
        responses = [_response(str(i + 2), f"r{i}", dynamic_context=stamp) for i in range(6)]
        rendered = _render([*BASE[:-1], *responses, BASE[-1]], AppendedLiveContext())
        assert sum(1 for m in rendered if isinstance(m, dict) and m.get("content") == stamp) == 0
        assert _roles(rendered) == ["system", "user", *["assistant"] * 6, "boundary"]

    def test_dynamic_mode_still_regenerates_on_every_request(self):
        dynamic = DynamicLiveContext()
        stamp = _trailing(dynamic, BASE)
        blocks = [
            *BASE[:-1],
            _response("2", "reply", dynamic_context=stamp),
            _dynamic("state", "s1"),
        ]
        rendered = _render(blocks, dynamic)
        assert _roles(rendered) == ["system", "user", "assistant", "boundary", "user"]
        assert "s1" in rendered[-1]["content"] and stamp not in str(rendered[:-1])


class TestSnapshotDue:
    """The cadence over the visible history, as the runtime evaluates it before a request."""

    @staticmethod
    def _snapshot():
        return LiveContextSnapshot(content="snap")

    @staticmethod
    def _response(text="r"):
        return LLMResponse(parts=(AssistantText(text=text),))

    def test_the_first_request_is_due(self):
        assert snapshot_due([], 5) is True
        assert snapshot_due([self._response()], 5) is True  # no snapshot yet

    @pytest.mark.parametrize("every_n", [1, 3])
    def test_due_once_every_n_responses_follow_the_last_snapshot(self, every_n):
        events = [self._snapshot()]
        for count in range(1, every_n + 2):
            events.append(self._response())
            # Not before N responses; from N on it stays due until a snapshot is recorded.
            assert snapshot_due(events, every_n) is (count >= every_n)

    def test_the_count_restarts_at_the_latest_snapshot(self):
        events = [self._snapshot(), self._response(), self._response(), self._snapshot()]
        assert snapshot_due(events, 2) is False
        events.append(self._response())
        assert snapshot_due(events, 2) is False
        events.append(self._response())
        assert snapshot_due(events, 2) is True

    def test_an_archived_snapshot_makes_the_next_request_due(self):
        """Only visible events are passed in; a collapsed snapshot is simply absent."""
        assert snapshot_due([self._response(), self._response()], 5) is True
