# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native wire contracts and zero-send preflight behavior (no live inference)."""

import json
from dataclasses import replace
from email.parser import BytesParser
from email.policy import default

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, field_serializer

from nooa import Agent, Context, DecideStrategy, EventQuery, strategy
from nooa.context_blocks import ScopedContext
from nooa.decisions import DecisionClient, DecisionModel, DecisionRequest
from nooa.decisions.client import BooleanQuestion
from nooa.decisions.media import collect_decision_attachments
from nooa.decisions.schema import compile_decision_schema
from nooa.events import Task
from nooa.media import Audio, File, Image, Media, Video
from nooa.unifiedllm import FakeLLMClient, RetryConfig

PNG = b"\x89PNG\r\n\x1a\nfixture"
JPEG = b"\xff\xd8\xfffixture"
ENDPOINT = "https://decision.example/decisions"


@pytest.fixture(autouse=True)
def disable_auto_network_tracing(monkeypatch):
    monkeypatch.setattr("nooa.agent._auto_tracing_attempted", True)


def image_request(image=None):
    return compile_decision_schema(bool, "Visible damage?").request(
        {"photo": image or Image.from_bytes(PNG, media_type="image/png")},
        media_inputs=["image"],
    )


def response(style):
    answers = (
        [{"type": "predicate", "probability": 0.9}]
        if style == "openai"
        else {"result": {"noul": 0.9}}
    )
    return httpx.Response(200, json={"model": "fake", "answers": answers})


@pytest.mark.parametrize("style", ["openai", "systemone"])
@pytest.mark.asyncio
async def test_actual_decide_strategy_transports_native_image_and_named_dedup(style):
    seen = []

    def handler(request):
        seen.append(request)
        return response(style)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        options = (
            {}
            if style == "openai"
            else {
                "decision_image_transport": "typesafe_multipart",
                "typesafe_danger_zone_images": True,
            }
        )
        client = DecisionClient(
            "fake",
            endpoint=ENDPOINT,
            api_style=style,
            media_inputs=["image"],
            client=http,
            **options,
        )

        class Inspector(Agent, llm=FakeLLMClient(), decision_model=client):
            @strategy(DecideStrategy(), ScopedContext(events=EventQuery.last_n(1)))
            async def damaged(self, first: Image, second: Image, label: str) -> bool:
                """Visible damage?"""
                ...

        agent = Inspector()
        image = Image.from_bytes(
            PNG, media_type="image/png", **({"detail": "high"} if style == "openai" else {})
        )
        # Hidden repr=False event image slots must not be promoted. Ordinary
        # context is not decision context, even with route image permission.
        agent.event_manager.add(
            Task(prompt="earlier", images=[{"type": "image_url", "image_url": {"url": "SECRET"}}])
        )
        agent.context["hidden_photo"] = Context("SECRET ordinary context")
        assert await agent.damaged(image, second=image, label="return") is True
        assert len(seen) == 1
        request = seen[0]
        if style == "openai":
            payload = json.loads(request.content)
            assert "assets" not in payload
            message = payload["input"][0]
            assert message["role"] == "user"
            blocks = message["content"]
            assert blocks[0]["type"] == "input_text"
            state = json.loads(blocks[0]["text"])
            assert blocks[1] == {"type": "input_text", "text": "<decision-image-0>"}
            assert blocks[2] == {
                "type": "input_image",
                "image_url": image.data_url,
                "detail": "high",
            }
            assert len(blocks) == 3
            assert "TypeSafe-Danger-Zone-Images" not in request.headers
        else:
            assert request.headers["TypeSafe-Danger-Zone-Images"] == "true"
            message = BytesParser(policy=default).parsebytes(
                f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode()
                + request.content
            )
            parts = list(message.iter_parts())
            assert len(parts) == 2
            assert parts[0].get_param("name", header="content-disposition") == "request"
            assert parts[0].get_content_type() == "application/json"
            assert parts[0].get_filename() is None
            payload = json.loads(parts[0].get_payload(decode=True))
            assert payload["assets"] == {"<decision-image-0>": {"type": "image", "part": "asset-0"}}
            assert parts[1].get_param("name", header="content-disposition") == "asset-0"
            assert parts[1].get_filename() == "asset-0"
            assert parts[1].get_content_type() == "image/png"
            assert parts[1].get_payload(decode=True) == PNG
            state = payload["state"]
        assert state["inputs"] == {
            "first": "<decision-image-0>",
            "second": "<decision-image-0>",
            "label": "return",
        }
        assert "SECRET" not in json.dumps(state)
        assert "hidden_photo" not in state
        assert "data:" not in json.dumps(state)


@pytest.mark.parametrize(
    "image, match",
    [
        (Image("data:image/jpeg;base64,YQ==", "image/png"), "matching MIME"),
        (Image("data:image/png;base64,!", "image/png"), "base64"),
        (Image("data:image/png;base64,", "image/png"), "nonempty"),
        (Image("data:image/png,hello", "image/png"), "base64"),
        (Image.from_url("file:///etc/passwd"), "URI"),
        (Image.from_url("https:///missing"), "URI"),
        (Image.from_url("https://u:p@host/image"), "URI"),
        (Image.from_url("https://host:bad/image"), "URI"),
        (Image.from_url("https://host/image#frag"), "URI"),
        (Image.from_url("https://host/image\n"), "URI"),
        (Image.from_url("https://host/" + "x" * 8192), "too long"),
        (Image.from_url("https://host/image", media_type="application/pdf"), "MIME"),
        (Image.from_url("https://host/image", detail="invalid"), "detail"),
        (Image("https://host/image", "image/png", {"url": "override"}), "metadata"),
    ],
)
def test_malformed_images_rejected_during_schema_preflight(image, match):
    with pytest.raises(ValueError, match=match):
        image_request(image)


@pytest.mark.parametrize("kind", [Audio, Video, File, Media])
def test_unsupported_native_media_and_permissions(kind):
    with pytest.raises(TypeError, match="unsupported Media"):
        compile_decision_schema(bool, "?").request(
            {"input": kind.from_bytes(b"x", media_type="application/pdf")}, media_inputs=["image"]
        )


@pytest.mark.parametrize(
    "permissions", [["audio"], ["video"], ["pdf"], ["image", "image"], "image", ["unknown"]]
)
def test_client_validates_permissions(permissions):
    with pytest.raises(ValueError):
        DecisionClient("fake", endpoint=ENDPOINT, media_inputs=permissions)


def test_nested_media_never_textualized_by_model_serializer():
    class Container(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        photo: Image

        @field_serializer("photo")
        def serialize_photo(self, value):
            return value.data_url

    image = Image.from_bytes(PNG, media_type="image/png")
    for value in ([image], {"photo": image}, Container(photo=image)):
        with pytest.raises(TypeError, match="[Nn]ested"):
            compile_decision_schema(bool, "?").request({"container": value}, media_inputs=["image"])
    with pytest.raises(TypeError, match="context block"):
        compile_decision_schema(bool, "?").request(
            {}, context={"photo": image}, media_inputs=["image"]
        )
    with pytest.raises(TypeError, match="event at index"):
        compile_decision_schema(bool, "?").request({}, events=[image], media_inputs=["image"])


def test_bounded_images_and_dedup(monkeypatch):
    from nooa.decisions import media

    image = Image.from_bytes(PNG, media_type="image/png")
    references, attachments = collect_decision_attachments({"a": image, "b": image}, ["image"])
    assert references == {"a": "<decision-image-0>", "b": "<decision-image-0>"}
    assert attachments[0].input_names == ("a", "b")
    assert attachments[0].decoded_bytes() == PNG
    assert "base64" not in repr(attachments)
    _, distinct = collect_decision_attachments(
        {"a": image, "b": Image.from_bytes(PNG, media_type="image/png")}, ["image"]
    )
    assert len(distinct) == 2
    monkeypatch.setattr(media, "MAX_DECISION_IMAGE_BYTES", 2)
    with pytest.raises(ValueError, match="size limit"):
        image_request(image)
    monkeypatch.setattr(media, "MAX_DECISION_IMAGE_BYTES", 1024)
    monkeypatch.setattr(media, "MAX_DECISION_TOTAL_IMAGE_BYTES", len(PNG))
    with pytest.raises(ValueError, match="total size"):
        collect_decision_attachments(
            {"a": image, "b": Image.from_bytes(JPEG, media_type="image/jpeg")}, ["image"]
        )
    monkeypatch.setattr(media, "MAX_DECISION_IMAGES", 1)
    with pytest.raises(ValueError, match="at most"):
        media.validate_attachments(distinct, ["image"])


@pytest.mark.parametrize(
    "options, transform, match",
    [
        ({}, lambda r: r, "explicit media_inputs"),
        ({"media_inputs": ["image"]}, lambda r: r, "System One JSON blocks image requests"),
        (
            {"media_inputs": ["image"], "decision_image_transport": "typesafe_multipart"},
            lambda r: r,
            "opt-in",
        ),
        (
            {"api_style": "openai", "media_inputs": ["image"]},
            lambda r: replace(r, attachments=(replace(r.attachments[0], image_url="bad"),)),
            "URI",
        ),
        (
            {"api_style": "openai", "media_inputs": ["image"]},
            lambda r: replace(r, attachments=(r.attachments[0], r.attachments[0])),
            "unique",
        ),
        (
            {"api_style": "openai", "media_inputs": ["image"]},
            lambda r: replace(r, state="missing"),
            "structured state",
        ),
        (
            {"api_style": "openai", "media_inputs": ["image"]},
            lambda r: replace(r, state={"nested": Image.from_bytes(PNG, media_type="image/png")}),
            "Native Media",
        ),
    ],
)
@pytest.mark.asyncio
async def test_client_preflight_zero_sends_including_direct_requests(options, transform, match):
    sent = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: sent.append(r) or response("systemone"))
    ) as http:
        client = DecisionClient("fake", endpoint=ENDPOINT, client=http, **options)
        with pytest.raises((TypeError, ValueError), match=match):
            await client.adecide(transform(image_request()))
    assert sent == []


@pytest.mark.parametrize(
    "image, match",
    [
        (Image.from_url("https://host/image"), "not fetched"),
        (Image.from_bytes(b"GIF89a", media_type="image/gif"), "PNG/JPEG"),
        (Image.from_bytes(JPEG, media_type="image/png"), "signature"),
        (
            Image.from_bytes(PNG, media_type="image/png", detail="high"),
            "cannot send an image detail setting",
        ),
    ],
)
@pytest.mark.asyncio
async def test_typesafe_strict_local_png_jpeg_preflight(image, match):
    sent = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: sent.append(r) or response("systemone"))
    ) as http:
        client = DecisionClient(
            "fake",
            endpoint=ENDPOINT,
            client=http,
            media_inputs=["image"],
            decision_image_transport="typesafe_multipart",
            typesafe_danger_zone_images=True,
        )
        with pytest.raises(ValueError, match=match):
            await client.adecide(image_request(image))
    assert sent == []


@pytest.mark.asyncio
async def test_typesafe_retry_replays_bytes_and_keeps_caller_client_open():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(503) if len(seen) == 1 else response("systemone")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = DecisionClient(
            "fake",
            endpoint=ENDPOINT,
            client=http,
            media_inputs=["image"],
            decision_image_transport="typesafe_multipart",
            typesafe_danger_zone_images=True,
            retry_config=RetryConfig(
                max_retries=1, base_delay=0, jitter_factor=0, rate_limit_extra_retries=0
            ),
        )
        await client.adecide(image_request(Image.from_bytes(JPEG, media_type="image/jpeg")))
        await client.aclose()
        assert not http.is_closed
        assert len(seen) == 2
        assert all(JPEG in r.content and b'name="request"' in r.content for r in seen)


@pytest.mark.asyncio
async def test_from_llm_explicit_rejection_before_chat_call():
    class Chat:
        model = "fake"
        media_inputs = ("image",)

        async def acall(self, *args, **kwargs):
            pytest.fail("attachments must not reach JSON-only adapter")

    with pytest.raises(ValueError, match="from_llm does not support"):
        await DecisionModel.from_llm(Chat()).adecide(image_request())


@pytest.mark.parametrize("style", ["openai", "systemone"])
@pytest.mark.asyncio
async def test_no_media_requests_keep_old_json_contract(style):
    seen = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: seen.append(r) or response(style))
    ) as http:
        client = DecisionClient(
            "fake", endpoint=ENDPOINT, api_style=style, client=http, media_inputs=["image"]
        )
        await client.adecide(DecisionRequest({"text": "old"}, {"result": BooleanQuestion("?")}))
    assert seen[0].headers["content-type"] == "application/json"
    payload = json.loads(seen[0].content)
    assert "assets" not in payload
    assert (
        payload["input"] == '{"text": "old"}'
        if style == "openai"
        else payload["state"] == {"text": "old"}
    )


@pytest.mark.parametrize(
    "options",
    [
        {"decision_image_transport": "auto"},
        {"api_style": "openai", "decision_image_transport": "typesafe_multipart"},
        {"typesafe_danger_zone_images": True},
        {"decision_image_transport": "typesafe_multipart", "typesafe_danger_zone_images": "true"},
    ],
)
def test_invalid_transport_configuration(options):
    with pytest.raises(ValueError):
        DecisionClient("fake", endpoint=ENDPOINT, **options)


@pytest.mark.parametrize("permissions", [(), ("image",)])
@pytest.mark.parametrize("container", [set, frozenset])
@pytest.mark.asyncio
async def test_nested_sets_reject_before_model_serializer_and_http(permissions, container):
    class Container(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        photos: object

        @field_serializer("photos")
        def serialize_photos(self, value):
            return [image.data_url for image in value]

    sent = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: sent.append(r) or response("openai"))
    ) as http:
        client = DecisionClient(
            "fake", endpoint=ENDPOINT, api_style="openai", media_inputs=permissions, client=http
        )
        with pytest.raises(TypeError, match="Nested Media"):
            request = compile_decision_schema(bool, "?").request(
                {
                    "photos": Container(
                        photos=container([Image.from_bytes(PNG, media_type="image/png")])
                    )
                },
                media_inputs=permissions,
            )
            await client.adecide(request)
    assert sent == []


@pytest.mark.parametrize("permissions", [(), ("image",)])
@pytest.mark.asyncio
async def test_actual_decide_selected_event_rejects_visible_native_media_before_serializer(
    permissions, monkeypatch
):
    from nooa.events import EventBase

    monkeypatch.setattr("nooa.agent._auto_tracing_attempted", True)

    class PhotoEvent(EventBase):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        photo: Image

        @field_serializer("photo")
        def serialize_photo(self, value):
            return value.data_url

    sent = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: sent.append(r) or response("openai"))
    ) as http:
        client = DecisionClient(
            "fake", endpoint=ENDPOINT, api_style="openai", media_inputs=permissions, client=http
        )

        class Inspector(Agent, llm=FakeLLMClient(), decision_model=client):
            @strategy(DecideStrategy(), ScopedContext(events=EventQuery.last_n(10)))
            async def check(self, text: str) -> bool:
                """Visible damage?"""
                ...

        agent = Inspector()
        agent.event_manager.add(PhotoEvent(photo=Image.from_bytes(PNG, media_type="image/png")))
        with pytest.raises(
            TypeError, match="Decision event 'PhotoEvent' field 'photo'.*Nested Media"
        ):
            await agent.check("text")
    assert sent == []


@pytest.mark.asyncio
async def test_hidden_event_native_media_not_inspected_or_sent(monkeypatch):
    from pydantic import Field

    from nooa.events import EventBase

    monkeypatch.setattr("nooa.agent._auto_tracing_attempted", True)

    class HiddenPhotoEvent(EventBase):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        text: str = "visible"
        photo: Image = Field(repr=False)

    sent = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: sent.append(r) or response("openai"))
    ) as http:
        client = DecisionClient("fake", endpoint=ENDPOINT, api_style="openai", client=http)

        class Inspector(Agent, llm=FakeLLMClient(), decision_model=client):
            @strategy(DecideStrategy(), ScopedContext(events=EventQuery.last_n(10)))
            async def check(self, text: str) -> bool:
                """Visible damage?"""
                ...

        agent = Inspector()
        agent.event_manager.add(
            HiddenPhotoEvent(photo=Image.from_bytes(PNG, media_type="image/png"))
        )
        assert await agent.check("text") is True
    state = json.loads(json.loads(sent[0].content)["input"])
    assert state["events"] == [
        {"type": "HiddenPhotoEvent", "role": "user", "data": {"text": "visible"}}
    ]


@pytest.mark.parametrize(
    "state",
    [
        {"inputs": {"not_photo": "<decision-image-0>"}},
        {"inputs": {"photo": "<decision-image-0> trailing"}},
        {"inputs": {}, "shared_guidance": "<decision-image-0>"},
        {"inputs": {}, "<decision-image-0>": "unrelated"},
        {"inputs": {"photo": "<decision-image-0>", "extra": "<decision-image-0>"}},
        ["<decision-image-0>"],
    ],
)
@pytest.mark.asyncio
async def test_direct_request_must_preserve_exact_named_association(state):
    sent = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: sent.append(r) or response("openai"))
    ) as http:
        client = DecisionClient(
            "fake", endpoint=ENDPOINT, api_style="openai", media_inputs=["image"], client=http
        )
        with pytest.raises(ValueError, match="state.inputs"):
            await client.adecide(replace(image_request(), state=state))
    assert sent == []


@pytest.mark.parametrize("permissions", [("audio",), ("image", "audio", "video", "pdf")])
@pytest.mark.asyncio
async def test_text_only_from_llm_keeps_multimodal_chat_routes_working(permissions, monkeypatch):
    from nooa.unifiedllm import AssistantText, LLMResponse

    monkeypatch.setattr("nooa.agent._auto_tracing_attempted", True)
    llm = FakeLLMClient(
        [LLMResponse(parts=(AssistantText(text='{"result": true}'),), finish_reason="stop")]
    )
    llm.media_inputs = permissions
    adapter = DecisionModel.from_llm(llm)

    class Inspector(Agent, decision_model=adapter):
        @strategy(DecideStrategy())
        async def check(self, text: str) -> bool:
            """Does it match?"""
            ...

    assert await Inspector().check("text") is True
    assert len(llm.calls) == 1
    assert adapter.media_inputs == ()


@pytest.mark.asyncio
async def test_typesafe_multiple_distinct_assets_and_inherited_json_header():
    seen = []
    async with httpx.AsyncClient(
        headers={"Content-Type": "application/json", "Authorization": "Bearer fake"},
        transport=httpx.MockTransport(lambda r: seen.append(r) or response("systemone")),
    ) as http:
        client = DecisionClient(
            "fake",
            endpoint=ENDPOINT,
            client=http,
            media_inputs=["image"],
            decision_image_transport="typesafe_multipart",
            typesafe_danger_zone_images=True,
        )
        request = compile_decision_schema(bool, "?").request(
            {
                "png": Image.from_bytes(PNG, media_type="image/png"),
                "jpeg": Image.from_bytes(JPEG, media_type="image/jpeg"),
            },
            media_inputs=["image"],
        )
        await client.adecide(request)
        assert http.headers["Content-Type"] == "application/json"
    request = seen[0]
    assert request.headers["Authorization"] == "Bearer fake"
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.content
    )
    parts = list(message.iter_parts())
    assert [p.get_param("name", header="content-disposition") for p in parts] == [
        "request",
        "asset-0",
        "asset-1",
    ]
    assert [p.get_filename() for p in parts] == [None, "asset-0", "asset-1"]
    assert [p.get_content_type() for p in parts] == ["application/json", "image/png", "image/jpeg"]
    assert [p.get_payload(decode=True) for p in parts[1:]] == [PNG, JPEG]
    assert json.loads(parts[0].get_payload(decode=True))["assets"] == {
        "<decision-image-0>": {"type": "image", "part": "asset-0"},
        "<decision-image-1>": {"type": "image", "part": "asset-1"},
    }


@pytest.mark.asyncio
async def test_openai_remote_url_forwarded_without_local_fetching():
    sent = []
    url = "https://public.example/photo.png"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: sent.append(r) or response("openai"))
    ) as http:
        client = DecisionClient(
            "fake", endpoint=ENDPOINT, api_style="openai", media_inputs=["image"], client=http
        )
        await client.adecide(image_request(Image.from_url(url, media_type="image/png")))
    assert len(sent) == 1
    assert sent[0].url == ENDPOINT
    assert json.loads(sent[0].content)["input"][0]["content"][-1] == {
        "type": "input_image",
        "image_url": url,
    }


@pytest.mark.parametrize(
    "value, permissions, limit, match",
    [
        (Image.from_bytes(PNG, media_type="image/png"), (), None, "explicit media_inputs"),
        (Audio.from_bytes(b"x", media_type="audio/wav"), ("image",), None, "unsupported Media"),
        ([Image.from_bytes(PNG, media_type="image/png")], ("image",), None, "nested"),
        (Image("data:image/png;base64,!", "image/png"), ("image",), None, "base64"),
        (Image.from_bytes(PNG, media_type="image/png"), ("image",), 2, "size limit"),
    ],
)
@pytest.mark.asyncio
async def test_actual_strategy_errors_preflight_without_any_http(
    value, permissions, limit, match, monkeypatch
):
    monkeypatch.setattr("nooa.agent._auto_tracing_attempted", True)
    if limit is not None:
        monkeypatch.setattr("nooa.decisions.media.MAX_DECISION_IMAGE_BYTES", limit)
    sent = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: sent.append(r) or response("openai"))
    ) as http:
        client = DecisionClient(
            "fake", endpoint=ENDPOINT, api_style="openai", media_inputs=permissions, client=http
        )

        class Inspector(Agent, llm=FakeLLMClient(), decision_model=client):
            @strategy(DecideStrategy())
            async def check(self, input: object) -> bool:
                """Visible damage?"""
                ...

        with pytest.raises((TypeError, ValueError), match=match):
            await Inspector().check(value)
    assert sent == []


@pytest.mark.parametrize("permissions", [(), ("image",)])
def test_mapping_key_media_rejected_before_model_serializer(permissions):
    class Container(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        photos: dict[object, str]

        @field_serializer("photos")
        def serialize_photos(self, value):
            return {image.data_url: label for image, label in value.items()}

    image = Image.from_bytes(PNG, media_type="image/png")
    with pytest.raises(TypeError, match="Nested Media"):
        compile_decision_schema(bool, "?").request(
            {"container": Container(photos={image: "photo"})}, media_inputs=permissions
        )
