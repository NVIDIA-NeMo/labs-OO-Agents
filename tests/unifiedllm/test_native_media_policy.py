# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native implementation policy, exercised through LiteLLM and mock HTTP only."""

import json
from copy import deepcopy

import httpx
import pytest
import yaml

from nooa import Agent, no_trace
from nooa.config.model_config import ModelConfig
from nooa.config.strategy_config import PredictConfig
from nooa.decorators import strategy
from nooa.media import Audio, File, Image, Media, Video
from nooa.media_policy import entry_input_interface, implemented_input_kinds
from nooa.strategies.predict import PredictStrategy
from nooa.unifiedllm import CompletionClient, ResponsesClient, RetryConfig, registry
from nooa.unifiedllm.connect import configure_entry, plan, write
from nooa.unifiedllm.registry import client_from_config, get_llm_client, reload_registry
from nooa.unifiedllm.unifiedllm import _ClientHttp
from tests.unifiedllm.connect.connect_http import response_body

ALL = ["image", "audio", "video", "pdf"]


def mock_http(monkeypatch, style):
    bodies = []

    def respond(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=response_body(style, '{"value":"ok"}'))

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        _ClientHttp,
        "_httpx_hardening",
        staticmethod(lambda: {"transport": transport, "trust_env": False}),
    )
    return bodies


def make_client(style, **kwargs):
    cls = ResponsesClient if style == "responses" else CompletionClient
    return cls(
        "anthropic/model" if style == "anthropic" else "openai/model",
        api_key="fake",
        api_base="https://policy.test/v1",
        max_tokens=128,
        cache_breakpoint=None,
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
        **kwargs,
    )


@pytest.mark.parametrize("style", ["anthropic", "responses"])
@pytest.mark.parametrize("drop", [True, False])
@pytest.mark.parametrize("kind", ["audio", "video"])
@pytest.mark.parametrize("is_async", [False, True])
async def test_native_enabled_media_rejected_before_http(monkeypatch, style, drop, kind, is_async):
    bodies = mock_http(monkeypatch, style)
    block = (
        {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}}
        if kind == "audio"
        else {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AA=="}}
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}, block]}]
    before = deepcopy(messages)
    with make_client(style, media_inputs=[kind], drop_params=drop) as c:
        with pytest.raises(ValueError, match="[Uu]nsupported.*" + kind):
            await c.acall(messages) if is_async else c.call(messages)
    assert bodies == []
    assert messages == before


@pytest.mark.parametrize("style", ["anthropic", "responses"])
@pytest.mark.parametrize("boundary", ["configure", "plan", "write"])
def test_invalid_native_declaration_rejected_without_write(tmp_path, style, boundary):
    entry = {
        "model_name": "anthropic/model" if style == "anthropic" else "openai/model",
        "client_type": "responses" if style == "responses" else "completion",
        "api_style": style,
        "api_base": "https://policy.test/v1",
        "media_inputs": ["audio", "video"],
    }
    before = deepcopy(entry)
    path = tmp_path / "not-created" / "config.yaml"
    with pytest.raises(ValueError, match="[Uu]nsupported.*audio"):
        if boundary == "configure":
            configure_entry(entry)
        elif boundary == "plan":
            plan("alias", "model", style, entry["api_base"], "", media_inputs=entry["media_inputs"])
        else:
            write(entry, path, alias="alias")
    assert not path.parent.exists()
    assert entry == before


class MediaAgent(Agent):
    @no_trace
    @strategy(PredictStrategy(config=PredictConfig(max_retries=1)))
    async def inspect(self, media: Media) -> str:
        """Inspect this input and reply ok."""
        ...


@pytest.mark.parametrize("style", ["anthropic", "responses"])
@pytest.mark.parametrize("kind", ["audio", "video"])
@pytest.mark.parametrize("drop", [True, False])
@pytest.mark.parametrize("factory", ["direct", "registry"])
async def test_actual_agent_predict_and_legacy_registry_never_drops_enabled_media(
    monkeypatch, tmp_path, style, kind, drop, factory
):
    bodies = mock_http(monkeypatch, style)
    media = (Audio if kind == "audio" else Video).from_bytes(
        b"sample", media_type=kind + "/" + ("wav" if kind == "audio" else "mp4")
    )
    original = (media.data_url, media.media_type, media.vendor_metadata.copy())
    entry = {
        "model_name": "anthropic/model" if style == "anthropic" else "openai/model",
        "client_type": "responses" if style == "responses" else "completion",
        "api_style": style,
        "api_base": "https://policy.test/v1",
        "api_key_env": "",
        "max_tokens": 128,
        "media_inputs": [kind],
        "drop_params": drop,
        "cache_breakpoint": None,
        "retry_config": False,
    }
    before = deepcopy(entry)
    if factory == "registry":
        path = tmp_path / "legacy.yaml"
        path.write_text(yaml.safe_dump({"models": {"legacy": entry}}))
        reload_registry(path)
        c = get_llm_client("legacy", api_key="fake")
    else:
        c = make_client(style, media_inputs=[kind], drop_params=drop)
    try:
        # Predict may wrap validation errors, but the native policy reason must
        # remain actionable and the raw media must never reach a provider.
        with pytest.raises(Exception, match="Unsupported media input.*" + kind):
            await MediaAgent(llm=c).inspect(media)
        assert bodies == []
        # Ordinary text is intentionally still usable with legacy flags.
        await c.acall([{"role": "user", "content": "hello"}])
        assert len(bodies) == 1
        assert c.media_inputs == (kind,)
        assert (media.data_url, media.media_type, media.vendor_metadata) == original
        assert entry == before
        if factory == "registry":
            assert registry.MODELS["legacy"] == before
    finally:
        await c.aclose()


RAW_SHAPES = [
    {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}},
    {"type": "audio_url", "audio_url": {"url": "https://never-fetch.test/s.wav"}},
    {"type": "audio", "source": {"type": "base64", "media_type": "audio/wav", "data": "AA=="}},
    {"type": "output_audio", "data": "AA=="},
    {"type": "video_url", "video_url": {"url": "https://never-fetch.test/s.mp4"}},
    {"type": "input_video", "video_url": "https://never-fetch.test/s.mp4"},
    {"type": "video", "source": {"media_type": "video/mp4", "data": "AA=="}},
]


@pytest.mark.parametrize("style", ["anthropic", "responses"])
@pytest.mark.parametrize("block", RAW_SHAPES)
@pytest.mark.parametrize("enabled", [False, True])
def test_known_raw_shapes_filter_first_then_reject(monkeypatch, style, block, enabled):
    bodies = mock_http(monkeypatch, style)
    messages = [{"role": "user", "content": [deepcopy(block)]}]
    before = deepcopy(messages)
    with make_client(style, media_inputs=ALL if enabled else []) as c:
        if enabled:
            with pytest.raises(ValueError, match="Unsupported media input"):
                c.call(messages)
            assert bodies == []
        else:
            c.call(messages)
            assert len(bodies) == 1
            assert "was removed here" in str(bodies)
            assert "never-fetch" not in str(bodies)
            sent_types = {
                b.get("type")
                for msg in bodies[0]["input" if style == "responses" else "messages"]
                if isinstance(msg.get("content"), list)
                for b in msg["content"]
            }
            assert not sent_types.intersection(shape["type"] for shape in RAW_SHAPES)
    assert messages == before


@pytest.mark.parametrize("block", RAW_SHAPES)
def test_responses_projection_rejects_explicit_unsupported_blocks(block):
    before = deepcopy(block)
    with pytest.raises(ValueError, match="Unsupported media input"):
        ResponsesClient._transform_content_blocks([block])
    assert block == before


@pytest.mark.parametrize("style", ["anthropic", "responses", "chat"])
@pytest.mark.parametrize("kind", ["image", "pdf"])
async def test_native_supported_media_agent_real_wire(monkeypatch, style, kind):
    bodies = mock_http(monkeypatch, style)
    cls, mime = (Image, "image/png") if kind == "image" else (File, "application/pdf")
    media = cls.from_bytes(b"sample", media_type=mime)
    c = make_client(style, media_inputs=[kind])
    try:
        assert await MediaAgent(llm=c).inspect(media) == "ok"
        assert len(bodies) == 1
        wire_types = {
            b.get("type")
            for m in bodies[0]["input" if style == "responses" else "messages"]
            if isinstance(m.get("content"), list)
            for b in m["content"]
        }
        expected = (
            ("image" if kind == "image" else "document")
            if style == "anthropic"
            else (
                ("input_image" if kind == "image" else "input_file")
                if style == "responses"
                else ("image_url" if kind == "image" else "file")
            )
        )
        assert expected in wire_types
    finally:
        await c.aclose()


@pytest.mark.parametrize("kind", ["audio", "video"])
@pytest.mark.parametrize("drop", [True, False])
async def test_claude_named_chat_extensions_still_sent_by_agent(monkeypatch, kind, drop):
    bodies = mock_http(monkeypatch, "chat")
    c = CompletionClient(
        "openai/claude-chat-route",
        api_key="fake",
        api_base="https://policy.test/v1",
        media_inputs=[kind],
        max_tokens=128,
        cache_breakpoint=None,
        drop_params=drop,
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
    )
    media = (Audio if kind == "audio" else Video).from_bytes(
        b"sample", media_type=kind + "/" + ("wav" if kind == "audio" else "mp4")
    )
    try:
        assert await MediaAgent(llm=c).inspect(media) == "ok"
        assert ("input_audio" if kind == "audio" else "video_url") in json.dumps(bodies)
    finally:
        await c.aclose()


@pytest.mark.parametrize("style_label", ["chat", "anthropic", "responses"])
def test_actual_custom_dispatch_not_api_style_or_model_guess(monkeypatch, style_label):
    # The label is not used by registry dispatch; custom provider is.
    entry = {
        "model_name": "opaque-model",
        "client_type": "completion",
        "api_style": style_label,
        "custom_llm_provider": "anthropic",
        "api_base": "https://policy.test",
        "media_inputs": ["audio"],
        "api_key_env": "",
        "max_tokens": 128,
        "cache_breakpoint": None,
        "retry_config": False,
    }
    before = deepcopy(entry)
    assert entry_input_interface(entry) == "anthropic"
    with pytest.raises(ValueError, match="Unsupported media input.*audio"):
        configure_entry(entry)
    bodies = mock_http(monkeypatch, "anthropic")
    with client_from_config("route", entry, api_key="fake") as c:
        with pytest.raises(ValueError, match="Unsupported media input.*audio"):
            c.call([{"role": "user", "content": [RAW_SHAPES[0]]}])
        # A replacement route cannot inherit permissions; disabled media must
        # be filtered even if its original enabled kind is unimplemented.
        filtered = c._prepare_media_messages(
            [{"role": "user", "content": [RAW_SHAPES[0]]}],
            c._prepare_call_config({"custom_llm_provider": "openai"}),
        )
        assert "was removed here" in str(filtered)
        assert c.media_inputs == ("audio",)
    assert bodies == [] and entry == before
    chat = {**entry, "model_name": "anthropic/innocuous", "custom_llm_provider": "openai"}
    assert entry_input_interface(chat) == "chat"
    assert configure_entry(chat)["media_inputs"] == ["audio"]
    assert ModelConfig.model_validate(chat).media_inputs == ("audio",)


def test_unknown_chat_declarations_stay_unverified_and_valid(tmp_path):
    entry = {
        "model_name": "unknown-private-route",
        "api_style": "anthropic",
        "client_type": "completion",
        "media_inputs": ["audio", "video"],
    }
    assert entry_input_interface(entry) == "chat"
    assert implemented_input_kinds("chat") == tuple(ALL)
    assert configure_entry(entry)["media_inputs"] == ["audio", "video"]
    write(entry, tmp_path / "models.yaml", alias="unknown")
    assert "passed" not in (tmp_path / "models.yaml").read_text()


@pytest.mark.parametrize("style", ["anthropic", "responses"])
def test_malformed_disabled_parts_never_project_or_decode(monkeypatch, style):
    bodies = mock_http(monkeypatch, style)
    # The invalid data fields would fail conversion/projection if visited.
    history = [
        {
            "role": "user",
            "content": [
                {"type": "input_audio"},
                {"type": "video_url"},
                {"type": "audio_url", "audio_url": object()},
            ],
        }
    ]
    blocks = history[0]["content"].copy()
    with make_client(style, media_inputs=[]) as c:
        c.call(history)
    assert len(bodies) == 1 and "was removed here" in str(bodies)
    assert history[0]["content"] == blocks


def test_no_recursive_media_guess_and_no_input_file_workaround(monkeypatch):
    extension = {"type": "custom_annotation", "payload": {"type": "audio", "data": "opaque"}}
    text = {"type": "text", "text": "keep"}
    assert ResponsesClient._transform_content_blocks([text, extension]) == [
        {"type": "input_text", "text": "keep"},
        extension,
    ]
    bodies = mock_http(monkeypatch, "responses")
    disguised_file = {"type": "input_file", "file_data": "data:audio/wav;base64,AA=="}
    with make_client("responses", media_inputs=ALL) as c:
        c.call([{"role": "user", "content": [disguised_file]}])
    assert "input_file" not in str(bodies) and "Only PDF files" in str(bodies)


@pytest.mark.parametrize("style", ["anthropic", "responses"])
@pytest.mark.parametrize("enabled", [False, True])
def test_compatible_opaque_native_media_guard_before_reconstruction(monkeypatch, style, enabled):
    from nooa.llm_types import AssistantText, LLMResponse
    from nooa.unifiedllm.errors import ReasoningReplayError
    from nooa.unifiedllm.replay_state import replay_scope

    bodies = mock_http(monkeypatch, style)
    scope = replay_scope(
        "anthropic/model" if style == "anthropic" else "openai/model",
        "chat" if style == "anthropic" else "responses",
        {"api_base": "https://policy.test/v1", "api_key": "fake"},
    )
    turn = LLMResponse(
        parts=(
            AssistantText(
                text="portable",
                native={"type": "message", "content": [{"type": "output_audio", "data": "opaque"}]},
            ),
        ),
        replay_scope=scope,
    )
    before = turn.model_dump()
    with make_client(style, media_inputs=["audio"] if enabled else []) as c:
        with pytest.raises(
            ValueError if enabled else ReasoningReplayError,
            match="Unsupported media input" if enabled else "disabled media.*opaque assistant",
        ):
            c.call([turn])
    assert turn.model_dump() == before and bodies == []


@pytest.mark.parametrize("kind", ["audio", "video"])
def test_custom_chat_dispatch_with_anthropic_name_is_actual_wire_contract(monkeypatch, kind):
    bodies = mock_http(monkeypatch, "chat")
    config = {
        "model_name": "anthropic/claude-custom-chat",
        "api_style": "anthropic",
        "client_type": "completion",
        "custom_llm_provider": "openai",
        "api_base": "https://policy.test/v1",
        "media_inputs": [kind],
        "max_tokens": 128,
        "cache_breakpoint": None,
        "retry_config": False,
        "drop_params": False,
        "api_key_env": "",
    }
    with client_from_config("custom-chat", config, api_key="fake") as c:
        block = RAW_SHAPES[0 if kind == "audio" else 4]
        c.call([{"role": "user", "content": [block]}])
        assert len(bodies) == 1
        assert ("input_audio" if kind == "audio" else "video_url") in str(bodies)


@pytest.mark.parametrize("style", ["anthropic", "responses"])
@pytest.mark.parametrize("kind", ["audio", "video"])
@pytest.mark.parametrize("factory", ["direct", "registry"])
async def test_disabled_agent_media_is_notice_not_unsupported_error(
    monkeypatch, tmp_path, style, kind, factory
):
    bodies = mock_http(monkeypatch, style)
    media = (Audio if kind == "audio" else Video).from_bytes(
        b"sample", media_type=kind + "/" + ("wav" if kind == "audio" else "mp4")
    )
    before = media.data_url
    if factory == "registry":
        config = {
            "model_name": "anthropic/model" if style == "anthropic" else "openai/model",
            "client_type": "responses" if style == "responses" else "completion",
            "api_style": style,
            "api_base": "https://policy.test/v1",
            "api_key_env": "",
            "media_inputs": [kind],
            "max_tokens": 128,
            "cache_breakpoint": None,
            "retry_config": False,
        }
        path = tmp_path / "legacy.yaml"
        path.write_text(yaml.safe_dump({"models": {"legacy": config}}))
        reload_registry(path)
        c = get_llm_client("legacy", api_key="fake", media_inputs=[])
    else:
        c = make_client(style, media_inputs=[])
    try:
        assert await MediaAgent(llm=c).inspect(media) == "ok"
        assert len(bodies) == 1 and "was removed here" in str(bodies)
        types = {
            block.get("type")
            for msg in bodies[0]["input" if style == "responses" else "messages"]
            if isinstance(msg.get("content"), list)
            for block in msg["content"]
        }
        assert not types.intersection({"input_audio", "video_url"})
        assert media.data_url == before and c.media_inputs == ()
        if factory == "registry":
            assert registry.MODELS["legacy"]["media_inputs"] == [kind]
    finally:
        await c.aclose()
