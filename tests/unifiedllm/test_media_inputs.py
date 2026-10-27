# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit declarations and request ownership; every provider is mocked."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import litellm
import pytest
import yaml

from nooa.config.model_config import ModelConfig
from nooa.llm_types import AssistantReasoning, AssistantText, LLMResponse, ToolCall
from nooa.media import Audio, File, Image, Video
from nooa.runtime.media_capture import media_to_content_block
from nooa.unifiedllm import CompletionClient, ResponsesClient
from nooa.unifiedllm.connect import configure_entry, plan, refresh_plan, write
from nooa.unifiedllm.media_inputs import attachment_kind, filter_media_messages
from nooa.unifiedllm.registry import client_from_config, get_llm_client, reload_registry
from nooa.unifiedllm.replay_state import replay_scope

ALL = ["image", "audio", "video", "pdf"]


def response(responses):
    if not responses:
        return litellm.ModelResponse(choices=[{"message": {"content": "ok"}}])
    return SimpleNamespace(
        output=[{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
        output_text="ok",
        status="completed",
        usage=None,
    )


def media_objects():
    return [
        Image.from_bytes(b"png", media_type="image/png"),
        Audio.from_bytes(b"wav", media_type="audio/wav"),
        Video.from_bytes(b"mp4", media_type="video/mp4"),
        File.from_bytes(b"pdf", media_type="application/pdf"),
    ]


@pytest.mark.parametrize("client_type", [CompletionClient, ResponsesClient])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("enabled", [None, [], ["image"], ["audio"], ["video"], ["pdf"], ALL])
async def test_fresh_history_and_python_media_owned(client_type, asynchronous, enabled):
    responses = client_type is ResponsesClient
    objects = media_objects()
    object_state = [(obj.data_url, repr(obj)) for obj in objects]
    blocks = [media_to_content_block(obj) for obj in objects]
    history = {"role": "user", "content": [{"type": "text", "text": "historical"}, *blocks]}
    fresh = {"role": "user", "content": deepcopy(blocks)}
    messages = [history, {"role": "assistant", "content": "previous answer"}, fresh]
    before = deepcopy(messages)
    method = ("a" if asynchronous else "") + ("responses" if responses else "completion")
    with client_type("openai/test", media_inputs=enabled, cache_breakpoint=None) as client:
        with patch(f"litellm.{method}", new_callable=AsyncMock if asynchronous else None) as wire:
            wire.return_value = response(responses)
            if asynchronous:
                await client.acall(messages)
            else:
                client.call(messages)
            payload = wire.call_args.kwargs
            assert "media_inputs" not in payload and "media_inputs" not in client.config
            sent = payload["input" if responses else "messages"]
            for index in (0, 2):
                kinds = [attachment_kind(block) for block in sent[index]["content"]]
                assert {kind for kind in kinds if kind} == set(enabled or [])
                text = " ".join(block.get("text", "") for block in sent[index]["content"])
                assert text.count("was removed here") == 4 - len(enabled or [])
            assert sent[1]["role"] == "assistant"
            # Even a provider mutating its payload cannot alter future history.
            sent[0]["content"][0]["text"] = "provider edit"
    assert messages == before
    assert [(obj.data_url, repr(obj)) for obj in objects] == object_state


@pytest.mark.parametrize("invalid", ["image", {}, True, ["file"], ["PDF"], [1], ["image", "image"]])
def test_invalid_declarations_match_every_boundary(invalid):
    for make in (
        lambda: ModelConfig(media_inputs=invalid),
        lambda: CompletionClient("openai/test", media_inputs=invalid),
        lambda: ResponsesClient("openai/test", media_inputs=invalid),
        lambda: configure_entry({"media_inputs": invalid}),
        lambda: plan("local", "test", "chat", "https://example.test/v1", "", media_inputs=invalid),
    ):
        with pytest.raises(ValueError, match="media_inputs"):
            make()


@pytest.mark.parametrize(
    "block,expected",
    [
        ({"type": "file", "file": {"file_data": "data:application/pdf;base64,eA=="}}, "pdf"),
        ({"type": "input_file", "mime_type": "application/pdf", "file_id": "abc"}, "pdf"),
        ({"type": "document", "source": {"media_type": "application/pdf", "data": "x"}}, "pdf"),
        ({"type": "input_file", "file_id": "pdf-test"}, "file"),
        (
            {"type": "file", "file": {"filename": "test.pdf", "file_data": "https://x/test.pdf"}},
            "file",
        ),
        ({"type": "file", "file": {"file_data": "data:application/msword;base64,eA=="}}, "file"),
        ({"type": "file", "mime_type": "application/pdf-extra"}, "file"),
        ({"type": "file", "mime_type": "application/octet-stream"}, "file"),
        ({"type": "image_url", "image_url": {"url": "data:application/pdf;base64,x"}}, "file"),
    ],
)
def test_pdf_is_exact_mime_not_every_file(block, expected):
    assert attachment_kind(block) == expected
    filtered = filter_media_messages([{"role": "user", "content": [block]}], ("pdf",))
    assert (filtered[0]["content"][0] == block) == (expected == "pdf")


@pytest.mark.parametrize("client_type", [CompletionClient, ResponsesClient])
def test_call_overrides_are_constructor_only_and_routes_default_empty(client_type):
    with client_type("openai/test", media_inputs=ALL, api_base="https://one.test/v1") as client:
        for overrides in (
            {"model": "openai/other"},
            {"api_base": "https://two.test/v1"},
            {"base_url": "https://two.test/v1"},
            {"client": object()},
            {"custom_llm_provider": "azure"},
        ):
            params = client._prepare_call_config(overrides)
            assert client._request_media_inputs(params) == ()
        assert client._request_media_inputs(
            client._prepare_call_config({"api_key": "other"})
        ) == tuple(ALL)
        for overrides in ({"media_inputs": ["image"]}, {"extra_body": {"media_inputs": ["image"]}}):
            with pytest.raises(ValueError, match="media_inputs.*client setting"):
                client.call([], **overrides)
        assert client.media_inputs == tuple(ALL)


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": "openai/other"},
        {"api_base": "https://two.test"},
        {"base_url": "https://two.test"},
        {"client": object()},
        {"custom_llm_provider": "azure"},
        {"client_type": "responses"},
    ],
)
def test_factory_replacement_clears_inherited_but_explicit_wins(overrides):
    config = {
        "model_name": "openai/test",
        "media_inputs": ["image"],
        "api_base": "https://one.test",
    }
    with client_from_config("local", config, **overrides) as client:
        assert client.media_inputs == ()
    with client_from_config("local", config, media_inputs=["audio"], **overrides) as client:
        assert client.media_inputs == ("audio",)


def test_yaml_constructor_connect_and_explicit_empty_parity(tmp_path):
    proposal = plan(
        "local",
        "test",
        "responses",
        "https://example.test/v1",
        "",
        media_inputs=ALL,
        catalogue={"id": "x", "architecture": {"input_modalities": ["image"]}},
    )
    proposal = refresh_plan(proposal)
    path = tmp_path / "models.yaml"
    write(proposal.entry, path, alias="local")
    assert yaml.safe_load(path.read_text())["models"]["local"]["media_inputs"] == ALL
    reload_registry(path)
    declaration = ALL.copy()
    with (
        get_llm_client("local") as registry,
        ResponsesClient("openai/test", media_inputs=declaration) as direct,
    ):
        declaration.clear()
        assert registry.media_inputs == direct.media_inputs == tuple(ALL)
        assert ModelConfig.from_registry("local", proposal.entry).media_inputs == tuple(ALL)
    with get_llm_client("local", media_inputs=[]) as empty:
        assert empty.media_inputs == ()
    assert (
        plan(
            "new",
            "test",
            "chat",
            "https://example.test/v1",
            "",
            catalogue={"id": "x", "media_inputs": ALL},
            endpoint_model={"id": "test", "media_inputs": ALL},
        ).entry["media_inputs"]
        == []
    )
    assert all("media_inputs" not in probe.body for probe in proposal.probes)


@pytest.mark.parametrize("client_type", [CompletionClient, ResponsesClient])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_reasoning_replay_and_tool_order_untouched(client_type, asynchronous):
    responses = client_type is ResponsesClient
    scope = replay_scope("openai/test", "responses" if responses else "chat", {})
    native = (
        {"type": "reasoning", "encrypted_content": "opaque", "summary": []}
        if responses
        else {"thinking_blocks": {"type": "thinking", "signature": "sig"}}
    )
    turn = LLMResponse(
        parts=(
            AssistantReasoning(text="thought", native=native),
            AssistantText(text="answer"),
            ToolCall(id="call_1", name="tool", arguments="{}"),
        ),
        replay_scope=scope,
    )
    before = turn.model_dump()
    messages = [
        {"role": "user", "content": [media_to_content_block(media_objects()[0])]},
        turn,
        {"role": "tool", "tool_call_id": "call_1", "content": "done"},
    ]
    method = ("a" if asynchronous else "") + ("responses" if responses else "completion")
    with client_type("openai/test", cache_breakpoint=None) as client:
        with patch(f"litellm.{method}", new_callable=AsyncMock if asynchronous else None) as wire:
            wire.return_value = response(responses)
            if asynchronous:
                await client.acall(messages)
            else:
                client.call(messages)
            sent = wire.call_args.kwargs["input" if responses else "messages"]
            if responses:
                assert sent[1]["encrypted_content"] == "opaque"
                assert sent[-2]["type"] == "function_call"
                assert sent[-1]["type"] == "function_call_output"
            else:
                assert sent[1]["thinking_blocks"][0]["signature"] == "sig"
                assert sent[1]["tool_calls"][0]["id"] == "call_1"
                assert sent[-1]["tool_call_id"] == "call_1"
    assert turn.model_dump() == before


def test_no_recursive_rewriting_and_opaque_media_fail_closed():
    from nooa.unifiedllm.errors import ReasoningReplayError

    nested = {
        "role": "assistant",
        "content": "text",
        "tool_calls": [{"arguments": {"type": "image_url"}}],
    }
    assert filter_media_messages([nested], ()) == [nested]
    turn = LLMResponse(
        parts=(
            AssistantText(
                text="answer",
                native={"type": "message", "content": [{"type": "output_audio", "data": "opaque"}]},
            ),
        ),
        replay_scope=replay_scope("openai/test", "responses", {}),
    )
    before = turn.model_dump()
    with ResponsesClient("openai/test") as client:
        with pytest.raises(ReasoningReplayError, match="opaque assistant"):
            client.call([turn])
    assert turn.model_dump() == before


@pytest.mark.parametrize("client_type", [CompletionClient, ResponsesClient])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_route_override_payload_clears_media(client_type, asynchronous):
    responses = client_type is ResponsesClient
    method = ("a" if asynchronous else "") + ("responses" if responses else "completion")
    with client_type("openai/test", media_inputs=ALL) as client:
        for overrides in ({"model": "openai/other"}, {"base_url": "https://replacement.test/v1"}):
            with patch(
                f"litellm.{method}", new_callable=AsyncMock if asynchronous else None
            ) as wire:
                wire.return_value = response(responses)
                messages = [
                    {"role": "user", "content": [media_to_content_block(media_objects()[0])]}
                ]
                if asynchronous:
                    await client.acall(messages, **overrides)
                else:
                    client.call(messages, **overrides)
                sent = wire.call_args.kwargs["input" if responses else "messages"]
                assert "was removed here" in sent[0]["content"][0]["text"]
                assert client.media_inputs == tuple(ALL)


@pytest.mark.parametrize("client_type", [CompletionClient, ResponsesClient])
@pytest.mark.parametrize("enabled", [[], ["audio"]])
def test_top_level_assistant_audio_cannot_bypass_text_only(client_type, enabled):
    responses = client_type is ResponsesClient
    original = {"role": "assistant", "content": None, "audio": {"id": "audio_123"}}
    before = deepcopy(original)
    with client_type("openai/test", media_inputs=enabled, cache_breakpoint=None) as client:
        with patch("litellm.responses" if responses else "litellm.completion") as wire:
            wire.return_value = response(responses)
            client.call([original])
            sent = wire.call_args.kwargs["input" if responses else "messages"][0]
            assert ("audio" in sent) == bool(enabled)
            if not enabled:
                assert "was removed here" in str(sent["content"])
    assert original == before


@pytest.mark.parametrize("wrapper", ["file", "image_url"])
def test_conflicting_mime_cannot_broaden_permission(wrapper):
    block = {
        "type": wrapper,
        "mime_type": "application/pdf" if wrapper == "file" else "image/png",
        wrapper: {"file_data" if wrapper == "file" else "url": "data:application/msword;base64,x"},
    }
    assert attachment_kind(block) == "file"
    assert (
        "was removed here"
        in filter_media_messages([{"role": "user", "content": [block]}], tuple(ALL))[0]["content"][
            0
        ]["text"]
    )


@pytest.mark.parametrize("native_output", [False, True])
@pytest.mark.parametrize("enabled", [[], ["image", "pdf"]])
def test_responses_tool_media_and_metadata_preserved(native_output, enabled):
    blocks = [
        media_to_content_block(media_objects()[0]),
        media_to_content_block(media_objects()[-1]),
    ]
    blocks[-1]["cache_control"] = {"type": "ephemeral"}
    original = (
        {"type": "function_call_output", "call_id": "call_1", "output": blocks}
        if native_output
        else {"role": "tool", "tool_call_id": "call_1", "content": blocks}
    )
    before = deepcopy(original)
    with ResponsesClient("openai/test", media_inputs=enabled, cache_breakpoint=None) as client:
        with patch("litellm.responses") as wire:
            wire.return_value = response(True)
            client.call([original])
            sent = wire.call_args.kwargs["input"][0]
            assert sent["call_id"] == "call_1"
            assert {
                attachment_kind(block) for block in sent["output"] if attachment_kind(block)
            } == set(enabled)
            if enabled:
                assert sent["output"][-1]["cache_control"] == {"type": "ephemeral"}
            else:
                assert "was removed here" in str(sent["output"])
    assert original == before


def test_incompatible_opaque_media_turn_replays_portably():
    turn = LLMResponse(
        parts=(
            AssistantText(
                text="portable",
                native={"type": "message", "content": [{"type": "output_audio", "data": "opaque"}]},
            ),
        ),
        replay_scope="responses:openai:other",
    )
    with ResponsesClient("openai/test") as client:
        with patch("litellm.responses") as wire:
            wire.return_value = response(True)
            client.call([turn])
            assert "portable" in str(wire.call_args.kwargs["input"])
            assert "output_audio" not in str(wire.call_args.kwargs["input"])


def test_fake_and_admission_share_explicit_declaration():
    from nooa.unifiedllm import AdmissionControl, AdmissionControlConfig, FakeLLMClient

    for enabled in ([], ["image"]):
        fake = FakeLLMClient(media_inputs=enabled)
        wrapped = AdmissionControl(
            fake, AdmissionControlConfig(max_in_flight=1, concurrency_group="media-test")
        )
        assert wrapped.media_inputs == tuple(enabled)
        wrapped.call([{"role": "user", "content": [media_to_content_block(media_objects()[0])]}])
        assert (attachment_kind(fake.last_messages[0]["content"][0]) == "image") == bool(enabled)


def test_summary_route_demotes_native_media_without_rewriting_archive():
    scope = replay_scope("openai/test", "responses", {})
    turn = LLMResponse(
        parts=(
            AssistantReasoning(text="summary", native={"type": "reasoning", "summary": []}),
            AssistantText(
                text="portable",
                native={"type": "message", "content": [{"type": "output_audio", "data": "opaque"}]},
            ),
        ),
        replay_scope=scope,
    )
    before = turn.model_dump()
    with ResponsesClient("openai/test") as client:
        with patch("litellm.responses") as wire:
            wire.return_value = response(True)
            client.call([turn])
            assert "portable" in str(wire.call_args.kwargs["input"])
            assert "output_audio" not in str(wire.call_args.kwargs["input"])
    assert turn.model_dump() == before


@pytest.mark.parametrize("placement", ["top", "extra"])
def test_reasoning_levels_cannot_smuggle_media_setting(placement):
    settings = {"media_inputs": ["image"]}
    if placement == "extra":
        settings = {"extra_body": settings}
    with pytest.raises(ValueError):
        CompletionClient("openai/test", reasoning_levels={"high": settings})
    with pytest.raises(ValueError):
        configure_entry({"reasoning_levels": {"high": settings}})


def test_tuple_content_is_filtered_like_a_list():
    """A tuple of blocks is sent like a list, so text-only routes must filter it too."""
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}
    [message] = filter_media_messages([{"role": "user", "content": (image,)}], ())
    assert message["content"] == [
        {
            "type": "text",
            "text": "[An image was removed here because this model is not set up to receive images.]",
        }
    ]
    [message] = filter_media_messages([{"role": "user", "content": (image,)}], ("image",))
    assert message["content"] == [image]


def test_native_input_file_mime_hint_is_not_sent():
    """The MIME hint marks a file_id as a PDF for the media check, but input_file
    has no MIME field, so it is removed before the request is sent."""
    block = {"type": "input_file", "file_id": "file-1", "mime_type": "application/pdf"}
    assert attachment_kind(block) == "pdf"
    [sent] = ResponsesClient._transform_content_blocks([block])
    assert sent == {"type": "input_file", "file_id": "file-1"}
    assert block["mime_type"] == "application/pdf"  # the caller's block is not changed


def test_missing_media_inputs_warns_once_that_the_model_is_text_only(caplog):
    from nooa.unifiedllm import registry

    registry._warned_text_only.discard("plain-alias")
    config = {"model_name": "openai/test", "api_base": "https://one.test"}
    with caplog.at_level("WARNING", logger="nooa.unifiedllm.registry"):
        client_from_config("plain-alias", config).close()
        client_from_config("plain-alias", config).close()
        client_from_config("declared-alias", {**config, "media_inputs": []}).close()
    warnings = [r.getMessage() for r in caplog.records if "media_inputs" in r.getMessage()]
    assert len(warnings) == 1
    assert "'plain-alias'" in warnings[0] and "text-only" in warnings[0]


def test_same_kind_hints_with_different_subtypes_keep_the_kind():
    block = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AA", "format": "image/jpeg"},
    }
    assert attachment_kind(block) == "image"
    pdf_label_word_data = {
        "type": "file",
        "file": {"file_data": "data:application/msword;base64,AA", "format": "application/pdf"},
    }
    assert attachment_kind(pdf_label_word_data) == "file"
