# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""show() always works; the next request replaces media the model cannot receive."""

import json
from collections.abc import Mapping

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.events import PythonOutput, ResultStatus
from nooa.media import Audio, File, Image, Video
from nooa.runtime.media_capture import _media_buffer_var
from nooa.runtime.sandbox.cell_core import run_cell_source
from nooa.strategies.codeact import CodeActStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from nooa.unifiedllm.media_inputs import attachment_kind, removed_media_notice

ALL = ("image", "audio", "video", "pdf")
MEDIA = [
    Image.from_bytes(b"image", media_type="image/png"),
    Audio.from_bytes(b"audio", media_type="audio/wav"),
    Video.from_bytes(b"video", media_type="video/mp4"),
    File.from_bytes(b"pdf", media_type="application/pdf"),
]


class MediaAgent(Agent, llm=FakeLLMClient()):
    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=4)))
    async def inspect(self) -> str:
        """Inspect self.attachment using show()."""
        ...


def response(code):
    return LLMResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            ToolCall(id="cell", name="execute_python", arguments=json.dumps({"code": code}))
        ],
    )


def content_blocks(messages):
    return [
        block
        for msg in messages
        if isinstance(msg, Mapping) and isinstance(msg.get("content"), (list, tuple))
        for block in msg["content"]
    ]


@pytest.mark.parametrize("kind,obj", list(zip(ALL, MEDIA, strict=True)))
@pytest.mark.parametrize("enabled", [False, True])
async def test_show_succeeds_and_the_next_request_has_the_media_or_a_notice(kind, obj, enabled):
    client = FakeLLMClient(
        [response("show(self.attachment)\nprint('after-show')"), response("return_result('done')")],
        strict_exhaustion=True,
        media_inputs=[kind] if enabled else [],
    )
    agent = MediaAgent(llm=client)
    agent.attachment = obj
    assert await agent.inspect() == "done"
    output = next(
        e
        for e in agent.event_manager.values()
        if isinstance(e, PythonOutput) and e.tool_call_id == "cell"
    )
    # The cell never fails because of the model's media setting.
    assert output.execution_status is ResultStatus.COMPLETE
    assert "[shown:" in output.stdout and "after-show" in output.stdout
    sent = content_blocks(client.calls[1].messages)
    kinds = [attachment_kind(block) for block in sent if attachment_kind(block)]
    notices = [block.get("text") for block in sent if block.get("type") == "text"]
    if enabled:
        assert kinds == [kind]
        assert removed_media_notice(kind) not in notices
    else:
        assert kinds == []
        assert removed_media_notice(kind) in notices
    assert _media_buffer_var.get() is None


async def test_sandbox_cell_has_no_media_setting():
    """The worker cell captures show() output; the request decides what is sent."""
    from nooa.runtime.media_capture import show

    namespace = {"attachment": MEDIA[0], "show": show}
    result = await run_cell_source("show(attachment)", namespace)
    assert result.error is None
    assert len(result.images) == 1


@pytest.mark.parametrize("kind", [*ALL, "file"])
def test_removed_media_notice_is_plain(kind):
    notice = removed_media_notice(kind)
    assert notice.startswith("[") and notice.endswith(".]")
    assert "was removed here" in notice
    assert "media_inputs" not in notice and "input disabled" not in notice


@pytest.mark.parametrize("media_type", ["image/svg+xml", "image/bmp", "image/tiff"])
def test_an_image_format_models_refuse_becomes_a_notice(media_type):
    from nooa.runtime.media_capture import media_to_content_block
    from nooa.unifiedllm.media_inputs import filter_media_messages

    block = media_to_content_block(Image.from_bytes(b"<svg/>", media_type=media_type))
    good = media_to_content_block(Image.from_bytes(b"png", media_type="image/png"))
    messages = [{"role": "user", "content": [block, good]}]
    filtered = filter_media_messages(messages, ALL)
    first, second = filtered[0]["content"]
    assert first["type"] == "text" and media_type in first["text"] and "PNG" in first["text"]
    assert second == good
    assert messages[0]["content"][0] is block  # the stored history is unchanged


@pytest.mark.parametrize(
    "block",
    [
        {"type": "image_url", "image_url": {"url": "https://example.com/a.svg"}},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/a", "format": "image/svg+xml"},
        },
        {"type": "input_image", "image_url": "https://example.com/scan.TIFF?sig=1"},
        {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/photo.heic"},
        },
        {"type": "image_url", "image_url": {"url": "https://example.com/b.bmp"}},
    ],
)
def test_remote_images_of_a_clearly_unsendable_type_become_a_note(block):
    """#597 item 5: by extension or explicit type, like an inline image (common floor)."""
    from nooa.unifiedllm.media_inputs import filter_media_messages, unsendable_image_type

    assert unsendable_image_type(block) is not None
    (sent,) = filter_media_messages([{"role": "user", "content": [block]}], ALL)[0]["content"]
    assert sent["type"] in {"text", "input_text"} and "cannot be sent" in sent["text"]


@pytest.mark.parametrize(
    "block",
    [
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        {"type": "image_url", "image_url": {"url": "https://example.com/render?id=7"}},
        {"type": "image_url", "image_url": {"url": "https://example.com/a", "format": "image/png"}},
        # An unknown-but-not-clearly-refused type: the provider decides.
        {"type": "image_url", "image_url": {"url": "https://example.com/a.avif"}},
    ],
)
def test_remote_images_of_a_sendable_or_unknown_type_are_sent_unchanged(block):
    from nooa.unifiedllm.media_inputs import filter_media_messages, unsendable_image_type

    assert unsendable_image_type(block) is None
    assert filter_media_messages([{"role": "user", "content": [block]}], ALL)[0]["content"] == [
        block
    ]


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://example.com/a.png", "image/png"),
        ("https://example.com/a.JPG?x=1#y", "image/jpeg"),
        ("https://example.com/a.svg", "image/svg+xml"),
        ("https://example.com/render?id=7", ""),
        ("https://example.com/notes.txt", ""),
        ("data:image/png;base64,iVBO", "image/png"),
        ("data:image/svg+xml;base64,PHN2Zy8+", "image/svg+xml"),
        ("data:text/plain;base64,aGk=", ""),
    ],
)
def test_image_from_url_guesses_its_type_and_never_invents_jpeg(url, expected):
    from nooa.runtime.media_capture import media_to_content_block

    image = Image.from_url(url)
    assert image.media_type == expected
    block = media_to_content_block(image)
    assert block["image_url"].get("format", "") == expected


def test_image_from_url_keeps_an_explicit_type():
    assert Image.from_url("https://example.com/a", media_type="image/webp").media_type == (
        "image/webp"
    )


def test_image_from_url_svg_is_replaced_by_a_note():
    from nooa.runtime.media_capture import media_to_content_block
    from nooa.unifiedllm.media_inputs import filter_media_messages

    block = media_to_content_block(Image.from_url("https://example.com/a.svg"))
    (sent,) = filter_media_messages([{"role": "user", "content": [block]}], ALL)[0]["content"]
    assert sent["type"] == "text" and "image/svg+xml" in sent["text"]


def test_a_base64_anthropic_svg_source_is_still_replaced():
    from nooa.unifiedllm.media_inputs import unsendable_image_type

    block = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/svg+xml", "data": "PHN2Zy8+"},
    }
    assert unsendable_image_type(block) == "image/svg+xml"
