# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-one contract tests. All dispatch is mocked; no inference spend."""

import copy
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from nooa.context_blocks.events import ToolCallEvent, ToolResult
from nooa.context_blocks.formatter import ResponsesProviderFormatter
from nooa.context_blocks.models import BlockMetadata, ResolvedBlock, Role
from nooa.context_blocks.renderer import render_context
from nooa.context_blocks.renderers.cached import CachedBlockFormatter
from nooa.llm_types import AssistantText, LLMResponse, ToolCall
from nooa.runtime.middleware import LLMCallContext
from nooa.storage.sqlite import SQLiteStorageManager
from nooa.tracing._journal_builder import build_journal_payload
from nooa.tracing._secret_scrubber import scrub_value
from nooa.unifiedllm import ResponsesClient
from nooa.unifiedllm.replay_state import ReasoningReplayError, prepare_chat_messages, replay_scope
from nooa.unifiedllm.response_parts import capture_parts, project_turn

SCOPE = replay_scope("openai/gpt-5.6", "responses", {})


def output_items():
    return [
        {
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "opaque-one",
            "summary": [{"type": "summary_text", "text": "think first"}],
        },
        {
            "type": "message",
            "role": "assistant",
            "phase": "commentary",
            "content": [{"type": "output_text", "text": "Before.", "annotations": []}],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "execute_python",
            "arguments": '{"code": "a()"}',
        },
        {
            "type": "reasoning",
            "id": "rs_2",
            "encrypted_content": "opaque-two",
            "summary": [{"type": "summary_text", "text": "then think"}],
        },
        {
            "type": "message",
            "role": "assistant",
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": "Between.", "annotations": []}],
        },
        {
            "type": "function_call",
            "call_id": "call_2",
            "name": "execute_python",
            "arguments": '{ "code" : "b()" }',
        },
    ]


def turn(items=None):
    return LLMResponse(
        parts=capture_parts(items or output_items(), SCOPE),
        replay_scope=SCOPE,
        finish_reason="tool_calls",
    )


def blocks(response, state="live 1"):
    result = [
        ResolvedBlock(
            key="instructions",
            content="Stable instructions " * 100,
            metadata=BlockMetadata(static=True),
        ),
        ResolvedBlock(key="turn", content="", role=Role.ASSISTANT, event=response),
    ]
    for call in response.tool_calls:
        event = ToolCallEvent(
            tool_call_id=call.id,
            name=call.name,
            arguments=json.loads(call.arguments),
            llm_response_id=response.id,
            result=ToolResult(tool_call_id=call.id, content="done"),
        )
        result.append(ResolvedBlock(key=call.id, content="", role=Role.ASSISTANT, event=event))
    result.append(
        ResolvedBlock(
            key="live", content=state, metadata=BlockMetadata(user_block=True, static=False)
        )
    )
    return result


def render(response, state="live 1", formatter=None):
    return render_context(
        blocks(response, state),
        block_formatter=formatter or CachedBlockFormatter(),
        provider_formatter=ResponsesProviderFormatter(),
    )


def test_exact_order_and_derived_views():
    response = turn()
    assert project_turn(response, SCOPE) == output_items()
    assert response.content == "Before.Between."
    assert response.reasoning == "think first\nthen think"
    assert [call.id for call in response.tool_calls] == ["call_1", "call_2"]
    data = response.model_dump()
    assert not {"content", "reasoning", "tool_calls", "llm_state"} & data.keys()
    for part in response.parts:
        if isinstance(part, ToolCall):
            assert "arguments" not in part.native
        else:
            assert part.text not in json.dumps(part.model_dump()["native"])


def test_nested_text_block_metadata_round_trips_without_repeating_public_text():
    items = output_items()
    items[0]["summary"].append({"type": "summary_text", "text": "second summary"})
    items[1]["content"].append({"type": "output_text", "text": "More.", "annotations": []})
    response = turn(items)
    assert project_turn(response, SCOPE) == items


def test_native_is_deeply_immutable_and_projection_borrows_large_scalar_leaves():
    original = turn()
    native = original.parts[0].native
    with pytest.raises(TypeError):
        native["encrypted_content"] = "edited"
    with pytest.raises(TypeError):
        native["summary"][0]["_text_length"] = 0
    assert project_turn(original, SCOPE)[0]["encrypted_content"] is native["encrypted_content"]
    assert original.model_copy(deep=True).parts[0].native is native


@pytest.mark.parametrize("bad", [[], "json string", 42])
def test_native_slot_requires_a_json_object(bad):
    with pytest.raises(ValidationError, match="JSON object"):
        AssistantText(text="answer", native=bad)


@pytest.mark.parametrize(
    "model",
    ["anthropic/claude-sonnet-4-5", "gemini/gemini-2.5-pro", "openai/gpt-5.6", "unknown-route"],
)
def test_text_reasoning_crosses_to_chat_without_native_state(model):
    messages = prepare_chat_messages([turn()], replay_scope(model, "chat", {}))
    assert "think first" in messages[0]["content"]
    assert "then think" in messages[0]["content"]
    assert messages[0]["content"] == "think first\n\nBefore.\n\nthen think\n\nBetween."
    assert "opaque" not in json.dumps(messages)
    assert len(messages[0]["tool_calls"]) == 2


@pytest.mark.parametrize(
    "scope", [None, "responses:azure:other", "chat:openai:other", "responses:openai:other"]
)
def test_incompatible_scope_replays_portable_reasoning_in_order(scope):
    public = project_turn(turn(), scope)
    assert "opaque" not in json.dumps(public)
    assert [item.get("content") for item in public if item.get("role") == "assistant"] == [
        "think first",
        "Before.",
        "then think",
        "Between.",
    ]
    assert len([item for item in public if item.get("type") == "function_call"]) == 2


@pytest.mark.parametrize(
    "field,value", [("parts", ()), ("replay_scope", "different"), ("content", "edited")]
)
def test_public_response_fields_cannot_be_mutated(field, value):
    with pytest.raises((ValidationError, AttributeError)):
        setattr(turn(), field, value)


@pytest.mark.parametrize(
    "index,field,value",
    [(0, "text", "edited"), (1, "text", "edited"), (2, "arguments", "{}"), (0, "native", "{}")],
)
def test_parts_are_immutable(index, field, value):
    with pytest.raises(ValidationError):
        setattr(turn().parts[index], field, value)


def test_replacements_strip_all_native_slots_and_leave_original_untouched():
    original = turn()
    replacement = original.replace_text("short")
    assert replacement.replay_scope is None
    assert all(part.native is None for part in replacement.parts)
    assert replacement.content == "short"
    assert project_turn(original, SCOPE) == output_items()
    assert original.model_copy(update={"tag": "3"}).parts is original.parts
    assert all(
        part.native is None for part in original.model_copy(update={"parts": original.parts}).parts
    )


@pytest.mark.parametrize("edited", [False, True])
def test_relay_json_round_trip_preserves_only_unchanged_turn_references(edited):
    original = turn()
    ctx = LLMCallContext(
        messages=[
            {"role": "system", "content": "stable"},
            original,
        ]
    )
    public = json.loads(json.dumps([dict(m) for m in ctx.messages]))
    assert "opaque" not in json.dumps(public)
    assert public[1]["reasoning_content"] == original.reasoning
    if edited:
        public[1]["content"] = "changed"
    from nooa.nemo_relay_middleware import _reconcile_messages

    resolved = _reconcile_messages(ctx.messages, public)
    assert (resolved[1] is original) is not edited
    if edited:
        assert resolved[1] == {**original.public_message(), "content": "changed"}


def test_relay_insertion_does_not_attach_state_to_new_neighbors():
    original = turn()
    ctx = LLMCallContext(messages=[original])
    from nooa.nemo_relay_middleware import _reconcile_messages

    public = [{"role": "user", "content": "inserted"}, dict(original)]
    resolved = _reconcile_messages(ctx.messages, public)
    assert type(resolved[1]) is dict
    assert resolved[0] == {"role": "user", "content": "inserted"}


def test_renderer_keeps_reference_and_truncation_replaces_without_native_state():
    original = turn()
    assert render(original).output[1] == original

    class ShortFormatter(CachedBlockFormatter):
        def format_event(self, event, event_format=None):
            return super().format_event(event, event_format)[:3]

    replacement = render(original, formatter=ShortFormatter()).output[1]
    assert replacement["content"] == "Bef"
    resolved = [replacement]
    assert isinstance(resolved[0], dict)
    assert "nooa_turn" not in resolved[0]
    assert original.content == "Before.Between."


def test_renderer_drops_incomplete_calls_as_a_state_stripping_edit():
    original = turn()
    original.finish_reason = "length"
    result = render(original)
    replacement = result.output[1]
    assert "tool_calls" not in replacement
    resolved = [replacement]
    assert isinstance(resolved[0], dict)
    assert "nooa_turn" not in resolved[0]
    assert original.tool_calls


def test_dispatch_maps_ordinary_images_without_changing_caller_input():
    client = ResponsesClient("openai/gpt-5.6", api_key="test")
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/img.png", "detail": "high"},
                    },
                ],
            }
        ]
        original = copy.deepcopy(messages)
        wire, _ = client._transform_messages(messages, SCOPE)
        assert wire[0]["content"] == [
            {"type": "input_text", "text": "look"},
            {"type": "input_image", "image_url": "https://example.com/img.png", "detail": "high"},
        ]
        assert original == messages
    finally:
        client.close()


def test_sqlite_close_reopen_keeps_exact_replay(tmp_path):
    path = tmp_path / "session.db"
    original = turn()
    storage = SQLiteStorageManager(path)
    storage.event_backend.store("1", original)
    storage.close()
    resumed = SQLiteStorageManager(path)
    try:
        loaded = resumed.event_backend.get("1")
        assert loaded is not original
        assert loaded.parts == original.parts
        assert project_turn(loaded, SCOPE) == output_items()
        assert loaded.raw_response is None
    finally:
        resumed.close()


@pytest.mark.parametrize("event_type", ["LLMOutput", "LLMResponse"])
def test_legacy_flat_archives_are_portable_only(event_type):
    storage = SQLiteStorageManager()
    try:
        old = {
            "event_type": event_type,
            "content": "answer",
            "reasoning": "thinking",
            "tool_calls": [{"id": "c", "name": "f", "arguments": "{}"}],
            "llm_state": {"secret": "old opaque state"},
        }
        loaded = storage.event_backend._deserialize(json.dumps(old))
        assert [part.kind for part in loaded.parts] == ["reasoning", "text", "tool_call"]
        assert loaded.replay_scope is None
        assert all(part.native is None for part in loaded.parts)
        assert loaded.reasoning == "thinking"
    finally:
        storage.close()


def test_journal_and_scrubber_keep_readable_reasoning_not_native_state():
    response = turn()
    journal = build_journal_payload(render(response).messages)
    assert journal.skeleton[1]["reasoning_content"] == response.reasoning
    assert "opaque" not in json.dumps(journal.skeleton)
    scrubbed, count = scrub_value(response.model_dump(mode="json"))
    assert count > 0
    assert "opaque-one" not in json.dumps(scrubbed)
    assert "think first" in json.dumps(scrubbed)


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_mocked_dispatch_resume_and_changing_live_suffix(monkeypatch, tmp_path, is_async):
    captured = []

    def respond(**kwargs):
        captured.append(copy.deepcopy(kwargs["input"]))
        return SimpleNamespace(
            output=output_items(), status="completed", model="gpt-5.6", usage=None
        )

    async def arespond(**kwargs):
        return respond(**kwargs)

    monkeypatch.setattr("litellm.responses", respond)
    monkeypatch.setattr("litellm.aresponses", arespond)
    client = ResponsesClient("openai/gpt-5.6", api_key="test")
    try:
        call = client.acall if is_async else client.call

        async def invoke(messages):
            result = call(messages)
            return await result if is_async else result

        original = await invoke([{"role": "user", "content": "start"}])
        storage = SQLiteStorageManager(tmp_path / "dispatch.db")
        storage.event_backend.store("1", original)
        storage.close()
        resumed = SQLiteStorageManager(tmp_path / "dispatch.db")
        loaded = resumed.event_backend.get("1")
        resumed.close()
        for state in ("live 1", "live 2"):
            ctx = LLMCallContext(messages=render(loaded, state).output)
            assert ctx.messages[1] is loaded
            await invoke(ctx.messages)
        assert captured[-2][:-1] == captured[-1][:-1]
        assert captured[-2][-1] != captured[-1][-1]
        # Tool-result cache markers are policy, not a turn-model change.
        assert captured[-1][: len(output_items())] == output_items()
        assert "live 2" in captured[-1][-1]["content"]
    finally:
        await client.aclose()


def test_unsupported_provider_parts_fail_instead_of_disappearing():
    with pytest.raises(ReasoningReplayError, match="Unsupported"):
        capture_parts([{"type": "future_item"}], SCOPE)


def test_unknown_route_cannot_capture_encrypted_state():
    parts = capture_parts(output_items(), None)
    assert all(part.native is None for part in parts)
    assert LLMResponse(parts=parts).reasoning == "think first\nthen think"


def test_boundary_beside_turn_preserves_reference_and_state():
    from nooa._llm_state import carried_cache_boundary

    original = turn()
    rendered = ResponsesProviderFormatter().format(
        [
            render(original).messages[1].model_copy(update={"cache_boundary_before": True}),
        ]
    )
    assert len(rendered) == 2
    assert rendered[0] == {"nooa_cache_boundary": True}
    assert rendered[1] == original
    client = ResponsesClient("openai/gpt-5.6", api_key="test")
    try:
        wire, _ = client._transform_messages(rendered, SCOPE)
        assert carried_cache_boundary(wire[0])
        assert wire[1:] == output_items()
    finally:
        client.close()


def test_rendered_public_values_share_strings_but_no_native_objects():
    original = turn()
    message = render(original).messages[1]
    assert message.content == original.content
    assert message.reasoning == original.reasoning
    assert message.tool_calls[0].arguments is original.tool_calls[0].arguments
    assert not hasattr(message.tool_calls[0], "native")
    replacement = message.model_copy(update={"content": "edited"})
    public = ResponsesProviderFormatter().format([replacement])
    assert isinstance(public[0], dict)


def test_live_flat_constructor_rejects_removed_native_state():
    with pytest.raises(ValidationError, match="llm_state is removed"):
        LLMResponse(content="text", llm_state={"opaque": "must not disappear"})


def test_generic_snapshot_round_trip_and_flat_archive_migration():
    from nooa.storage.serialization import deserialize, serialize

    original = turn()
    original.raw_response = object()  # Neither raw SDK values nor parsed values belong in archives.
    original.parsed = object()
    blob, allowed = serialize(original)
    loaded = deserialize(json.loads(json.dumps(blob)), allowed)
    assert project_turn(loaded, SCOPE) == output_items()
    blob["data"] = {
        "content": "legacy answer",
        "reasoning": "legacy thought",
        "llm_state": {"opaque": "old state"},
        "event_type": "LLMResponse",
    }
    migrated = deserialize(blob, allowed)
    assert migrated.content == "legacy answer"
    assert migrated.reasoning == "legacy thought"
    assert migrated.replay_scope is None


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_cache_helpers_and_calibration_receive_projected_dicts(monkeypatch, is_async):
    from nooa.llm_types import LLMUsage

    response = SimpleNamespace(
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            }
        ],
        model="test",
        status="completed",
    )
    sent = []

    def respond(**params):
        sent.append(params["input"])
        return response

    async def arespond(**params):
        return respond(**params)

    calibrated = []
    monkeypatch.setattr("litellm.responses", respond)
    monkeypatch.setattr("litellm.aresponses", arespond)
    monkeypatch.setattr(
        "nooa.unifiedllm.unifiedllm._extract_usage",
        lambda _: LLMUsage(input_tokens=10, output_tokens=1),
    )
    monkeypatch.setattr(
        "nooa.unifiedllm.unifiedllm._update_token_calibration",
        lambda model, messages, usage, **kwargs: calibrated.append(messages),
    )
    client = ResponsesClient("anthropic/claude-sonnet-4-5", api_key="test")
    try:
        messages = [LLMResponse(content="previous"), {"role": "user", "content": "go"}]
        params = {"cache_control_injection_points": [{"role": "user", "position": "last"}]}
        result = (
            await client.acall(messages, **params) if is_async else client.call(messages, **params)
        )
        assert result.content == "done"
        assert len(calibrated) == 1
        assert calibrated[0] is sent[0]
        assert all(isinstance(message, dict) for message in sent[0])
        assert sent[0][-1]["content"][0]["type"] == "input_text"
        assert sent[0][-1]["content"][0]["cache_control"] == {"type": "ephemeral"}
    finally:
        await client.aclose()


def test_list_content_text_mapping_respects_role():
    client = ResponsesClient("openai/gpt-5.6", api_key="test")
    try:
        wire, _ = client._transform_messages(
            [
                {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
                {"role": "user", "content": [{"type": "text", "text": "question"}]},
            ]
        )
        assert wire[0]["content"][0]["type"] == "output_text"
        assert wire[1]["content"][0]["type"] == "input_text"
    finally:
        client.close()


@pytest.mark.parametrize(
    "message",
    [
        {"type": "reasoning", "encrypted_content": "untrusted"},
        {"role": "assistant", "content": "public", "reasoning_items": []},
    ],
)
def test_raw_reasoning_wire_input_requires_explicit_migration(message):
    client = ResponsesClient("openai/gpt-5.6", api_key="test")
    try:
        with pytest.raises(ReasoningReplayError, match="canonical LLMResponse"):
            client._transform_messages([message], SCOPE)
    finally:
        client.close()


def test_search_retains_response_metadata_without_native_parts():
    from nooa.runtime.event_manager import EventManager

    response = turn()
    response.model_name = "searchable-model"
    text = EventManager()._get_searchable_text(response)
    assert "searchable-model" in text
    assert "tool_calls" in text
    assert "think first" in text
    assert "Before.Between." in text
    assert "opaque-one" not in text


def test_relay_deletion_conservatively_discards_shifted_native_state():
    original = turn()
    ctx = LLMCallContext(
        messages=[
            {"role": "user", "content": "removed"},
            original,
        ]
    )
    from nooa.nemo_relay_middleware import _reconcile_messages

    public = [dict(original)]
    assert type(_reconcile_messages(ctx.messages, public)[0]) is dict


def test_native_mapping_is_intentionally_unhashable():
    with pytest.raises(TypeError):
        hash(turn().parts[0].native)
