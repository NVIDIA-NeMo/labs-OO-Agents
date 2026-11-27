# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-model media capabilities (#618): config, provider defaults, pre-send notes.

Every provider is mocked: LiteLLM routes through httpx.MockTransport and native
SDK routes through a borrowed SDK client with a mock transport.
"""

import json
from copy import deepcopy

import httpx
import pytest

from nooa.config.model_config import ModelConfig
from nooa.decisions.media import DecisionAttachment
from nooa.media_capabilities import (
    DECISION_IMAGE_MIME_TYPES,
    PROVIDER_MEDIA_DEFAULTS,
    UNRESTRICTED,
    MediaCapabilities,
    MediaKindCapabilities,
    resolve_media_capabilities,
    validate_media_capabilities,
)
from nooa.unifiedllm import CompletionClient, registry
from nooa.unifiedllm.errors import FailureKind, LLMFailure
from nooa.unifiedllm.media_inputs import filter_media_messages
from nooa.unifiedllm.registry import get_decision_model, get_llm_client
from nooa.unifiedllm.request import LLMRequest
from tests.unifiedllm.test_native_load_llm import borrowed, dispose, load
from tests.unifiedllm.test_native_media_policy import make_client, mock_http
from tests.unifiedllm.test_native_projection import FAMILIES, payload

# HEIC is a real image format that no NOOA code path refuses by default. The
# tests use it as "a format the provider supports"; no provider fact is claimed.
HEIC = {"type": "image_url", "image_url": {"url": "data:image/heic;base64,AAAA"}}
PNG = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
PNG_ONLY = {"image": {"mime_types": ["image/png"]}}


def user(*blocks):
    return [{"role": "user", "content": [{"type": "text", "text": "look"}, *blocks]}]


def wire_blocks(body):
    return [
        block
        for message in body.get("messages", body.get("input", []))
        if isinstance(message.get("content"), list)
        for block in message["content"]
    ]


# (a) capabilities are read from the model config; every interface has a default


def test_model_config_reads_media_capabilities_and_defaults():
    entry = {
        "model_name": "openai/vision",
        "media_inputs": ["image", "audio"],
        "media_capabilities": {
            "image": {"mime_types": ["image/PNG", "image/jpg"], "max_bytes": 1024},
            "audio": {"transports": ["url", "inline"]},
        },
    }
    config = ModelConfig.from_registry("vision", entry)
    assert config.media_capabilities == {
        "image": {"mime_types": ["image/jpeg", "image/png"], "max_bytes": 1024},
        "audio": {"transports": ["inline", "url"]},
    }
    caps = config.resolved_media_capabilities()
    assert caps.image == MediaKindCapabilities(
        mime_types=frozenset({"image/png", "image/jpeg"}), max_bytes=1024
    )
    assert caps.audio.unrestricted and caps.video.unrestricted and caps.pdf.unrestricted
    # Entries without the key keep the provider default.
    assert ModelConfig.from_registry("plain", {"model_name": "openai/x"}).media_capabilities is None
    assert ModelConfig.from_registry(
        "plain", {"model_name": "openai/x"}
    ).resolved_media_capabilities() == (UNRESTRICTED)
    decision = ModelConfig.from_registry(
        "d", {"model_name": "m", "client_type": "decision", "endpoint": "https://d.test"}
    )
    assert decision.resolved_media_capabilities().image.mime_types == DECISION_IMAGE_MIME_TYPES


@pytest.mark.parametrize("interface", ["chat", "anthropic", "responses", "decision"])
def test_every_request_interface_has_a_default(interface):
    caps = resolve_media_capabilities(interface)
    assert isinstance(caps, MediaCapabilities)
    assert caps == PROVIDER_MEDIA_DEFAULTS.get(interface, UNRESTRICTED)
    # Only the decision interface declares limits; the others keep the
    # historical behavior (no MIME, transport or size filtering).
    assert caps.unrestricted is (interface != "decision")


def test_override_replaces_only_declared_fields():
    caps = resolve_media_capabilities("decision", {"image": {"transports": ["inline"]}})
    assert caps.image.transports == frozenset({"inline"})
    assert caps.image.mime_types == DECISION_IMAGE_MIME_TYPES
    assert caps.image.max_bytes == PROVIDER_MEDIA_DEFAULTS["decision"].image.max_bytes


@pytest.mark.parametrize(
    "value",
    [
        "image",
        {"text": {}},
        {"image": {"formats": ["image/png"]}},
        {"image": {"mime_types": []}},
        {"image": {"mime_types": ["audio/wav"]}},
        {"image": {"mime_types": ["image/"]}},
        {"pdf": {"mime_types": ["application/msword"]}},
        {"image": {"transports": []}},
        {"image": {"transports": ["ftp"]}},
        {"image": {"transports": ["url", "url"]}},
        {"image": {"max_bytes": 0}},
        {"image": {"max_bytes": True}},
    ],
)
def test_invalid_settings_fail_at_load(value):
    with pytest.raises(ValueError, match="media_capabilities"):
        validate_media_capabilities(value)
    with pytest.raises(ValueError, match="media_capabilities"):
        ModelConfig.from_registry("bad", {"media_capabilities": value})
    with pytest.raises(ValueError, match="media_capabilities"):
        CompletionClient("openai/x", media_inputs=["image"], media_capabilities=value)


def test_registry_passes_entry_setting_and_drops_it_on_route_change(monkeypatch):
    monkeypatch.setattr(registry, "_loaded", True)
    monkeypatch.setattr(
        registry,
        "MODELS",
        {
            "vision": {
                "model_name": "openai/vision",
                "api_base": "https://vision.test/v1",
                "api_key": "fake",
                "media_inputs": ["image"],
                "media_capabilities": PNG_ONLY,
            }
        },
    )
    client = get_llm_client("vision")
    assert client.media_capabilities == PNG_ONLY
    from nooa.unifiedllm.adapters.litellm_chat import client_capabilities

    assert client_capabilities(client).media_capabilities.image.mime_types == {"image/png"}
    other = get_llm_client("vision", model="openai/other")
    assert other.media_capabilities is None


def test_media_capabilities_is_not_a_per_call_parameter():
    client = make_client("chat", media_inputs=["image"])
    with pytest.raises(ValueError, match="media_capabilities is a client setting"):
        client.call(user(PNG), media_capabilities=PNG_ONLY)


# (b) a format the provider supports is sent, not dropped by a shared allowlist


@pytest.mark.parametrize("style", ["chat", "anthropic", "responses"])
async def test_litellm_routes_send_a_format_outside_the_old_decision_list(monkeypatch, style):
    bodies = mock_http(monkeypatch, style)
    with make_client(style, media_inputs=["image"]) as client:
        await client.acall(user(HEIC))
    assert len(bodies) == 1
    assert "image/heic" in json.dumps(bodies[0])
    assert "was removed" not in json.dumps(bodies[0])


@pytest.mark.parametrize("family", FAMILIES)
async def test_native_routes_send_a_format_outside_the_old_decision_list(family):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=payload(family, "ok"))

    sdk, _ = borrowed(family, "async", handler)
    llm = load(family, sdk, media_inputs=["image"])
    await llm.acall(LLMRequest(messages=user(HEIC)))
    assert len(seen) == 1 and "image/heic" in json.dumps(seen[0])
    await llm.aclose()
    await dispose(sdk, "async")


def test_decision_model_override_accepts_a_format_the_shared_list_refused():
    heic = DecisionAttachment(
        "<decision-image-0>", ("a",), "data:image/heic;base64,AAAA", "image/heic"
    )
    with pytest.raises(ValueError, match="image/png, image/jpeg, image/webp or image/gif"):
        heic.decoded_bytes()
    caps = resolve_media_capabilities(
        "decision", {"image": {"mime_types": [*DECISION_IMAGE_MIME_TYPES, "image/heic"]}}
    )
    assert heic.decoded_bytes(caps) == b"\x00\x00\x00"


def test_decision_registry_entry_reads_media_capabilities(monkeypatch):
    monkeypatch.setattr(registry, "_loaded", True)
    monkeypatch.setattr(
        registry,
        "MODELS",
        {
            "decide": {
                "model_name": "typesafe/m",
                "client_type": "decision",
                "endpoint": "https://decide.test",
                "media_inputs": ["image"],
                "media_capabilities": {"image": {"transports": ["inline"]}},
            }
        },
    )
    model = get_decision_model("decide")
    client = getattr(model, "client", model)
    caps = client.media_capabilities
    assert caps.image.transports == frozenset({"inline"})
    remote = DecisionAttachment("<decision-image-0>", ("a",), "https://img.test/a.png", "image/png")
    with pytest.raises(ValueError, match="remote image URLs"):
        remote.decoded_bytes(caps)
    assert remote.decoded_bytes() is None  # the default still accepts URLs


# (c) known unsupported media is replaced before sending; the request succeeds


def test_filter_replaces_by_mime_transport_and_size_only_when_declared():
    caps = resolve_media_capabilities(
        "chat",
        {
            "image": {"mime_types": ["image/png"], "max_bytes": 3},
            "audio": {"transports": ["inline"]},
        },
    )
    big = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAAAAAA"}}
    remote = {"type": "image_url", "image_url": {"url": "https://img.test/x"}}
    inline_audio = {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}
    remote_audio = {"type": "audio_url", "audio_url": {"url": "https://a.test/x.wav"}}
    messages = user(HEIC, PNG, big, remote, inline_audio, remote_audio)
    before = deepcopy(messages)
    blocks = filter_media_messages(messages, ("image", "audio"), capabilities=caps)[0]["content"]
    assert messages == before
    assert blocks[1]["text"].endswith("does not accept image/heic images.]")
    assert blocks[2] == PNG
    assert blocks[3]["text"].endswith("does not accept inline images larger than 3 bytes.]")
    assert blocks[4] == remote  # no MIME hint on a URL: unknown is not a refusal
    assert blocks[5] == inline_audio
    assert blocks[6]["text"].endswith("does not accept audio clips sent as remote URLs.]")
    # Without capabilities nothing but media_inputs is applied.
    assert filter_media_messages(messages, ("image", "audio")) == messages


@pytest.mark.parametrize("style", ["chat", "anthropic", "responses"])
@pytest.mark.parametrize("is_async", [False, True])
async def test_litellm_routes_replace_declared_unsupported_media_before_sending(
    monkeypatch, style, is_async
):
    bodies = mock_http(monkeypatch, style)
    messages = user(HEIC, PNG)
    before = deepcopy(messages)
    with make_client(style, media_inputs=["image"], media_capabilities=PNG_ONLY) as client:
        result = await client.acall(messages) if is_async else client.call(messages)
    assert result is not None
    assert len(bodies) == 1  # one successful request, no rejection round trip
    sent = json.dumps(bodies[0])
    assert "data:image/heic" not in sent
    assert "does not accept image/heic images" in sent
    assert "image/png" in sent
    assert messages == before


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_native_routes_replace_declared_unsupported_media_before_sending(family, mode):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=payload(family, "ok"))

    sdk, _ = borrowed(family, mode, handler)
    llm = load(family, sdk, media_inputs=["image"], media_capabilities=PNG_ONLY)
    assert llm.capabilities.media_capabilities.image.mime_types == {"image/png"}
    request = LLMRequest(messages=user(HEIC, PNG))
    llm.call(request) if mode == "sync" else await llm.acall(request)
    assert len(seen) == 1
    sent = json.dumps(seen[0])
    assert "data:image/heic" not in sent and "does not accept image/heic images" in sent
    assert any("AAAA" in json.dumps(block) for block in wire_blocks(seen[0]))
    await llm.aclose()
    await dispose(sdk, mode)


async def test_native_error_policy_refuses_declared_unsupported_media_without_http():
    from nooa.unifiedllm.llm import LLM

    seen = []
    sdk, _ = borrowed("openai-chat", "async", lambda r: seen.append(r) or httpx.Response(500))
    llm = load("openai-chat", sdk, media_inputs=["image"], media_capabilities=PNG_ONLY)
    strict = LLM(llm.adapter, on_unsupported_media="error")
    with pytest.raises(LLMFailure) as caught:
        await strict.acall(LLMRequest(messages=user(HEIC)))
    assert caught.value.kind == FailureKind.MEDIA_REJECTED
    assert caught.value.media_kind == "image"
    assert seen == []
    await llm.aclose()
    await dispose(sdk, "async")


def test_entries_without_the_key_send_unchanged_requests(monkeypatch):
    bodies = mock_http(monkeypatch, "chat")
    with make_client("chat", media_inputs=["image"]) as client:
        client.call(user(HEIC, PNG))
    bodies_default = deepcopy(bodies)
    bodies.clear()
    with make_client("chat", media_inputs=["image"], media_capabilities=None) as client:
        client.call(user(HEIC, PNG))
    assert bodies == bodies_default
    assert [b for b in wire_blocks(bodies[0]) if b.get("type") == "image_url"] == [HEIC, PNG]


def test_unrestricted_default_object_is_shared_and_frozen():
    with pytest.raises(AttributeError):
        UNRESTRICTED.image = MediaKindCapabilities(max_bytes=1)  # type: ignore[misc]
