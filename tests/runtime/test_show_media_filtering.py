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


def test_remote_image_urls_are_not_checked_for_format():
    from nooa.unifiedllm.media_inputs import filter_media_messages

    block = {"type": "image_url", "image_url": {"url": "https://example.com/a.svg"}}
    assert filter_media_messages([{"role": "user", "content": [block]}], ALL)[0]["content"] == [
        block
    ]


@pytest.mark.parametrize(
    "block",
    [
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/a.svg", "format": "image/svg+xml"},
        },
        {
            "type": "input_image",
            "image_url": "https://example.com/a.svg",
            "format": "image/svg+xml",
        },
        {
            "type": "image",
            "source": {
                "type": "url",
                "url": "https://example.com/a.svg",
                "media_type": "image/svg+xml",
            },
        },
    ],
)
def test_remote_images_with_a_format_hint_are_sent_unchanged(block):
    """Image.from_url(url, media_type="image/svg+xml"): the provider fetches it (Severin, #595)."""
    from nooa.unifiedllm.media_inputs import filter_media_messages, unsendable_image_type

    assert unsendable_image_type(block) is None
    assert filter_media_messages([{"role": "user", "content": [block]}], ALL)[0]["content"] == [
        block
    ]


def test_image_from_url_with_an_svg_type_is_not_replaced():
    from nooa.runtime.media_capture import media_to_content_block
    from nooa.unifiedllm.media_inputs import filter_media_messages

    block = media_to_content_block(
        Image.from_url("https://example.com/a.svg", media_type="image/svg+xml")
    )
    assert filter_media_messages([{"role": "user", "content": [block]}], ALL)[0]["content"] == [
        block
    ]


def test_a_base64_anthropic_svg_source_is_still_replaced():
    from nooa.unifiedllm.media_inputs import unsendable_image_type

    block = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/svg+xml", "data": "PHN2Zy8+"},
    }
    assert unsendable_image_type(block) == "image/svg+xml"
