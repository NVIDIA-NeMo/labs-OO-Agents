# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""#597 items 4 and 7, review missed case "nested tool results": the attachment inventory."""

import pytest

from nooa.unifiedllm.errors import ReasoningReplayError
from nooa.unifiedllm.media_inputs import (
    check_replayed_native_media,
    filter_media_messages,
    media_occurrences,
)

ALL = ("image", "audio", "video", "pdf")
PNG = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBO"}}
SVG = {"type": "image", "source": {"type": "base64", "media_type": "image/svg+xml", "data": "PHN2"}}
NOTE = "[An image was removed here because the model provider could not read it.]"


def tool_result(*blocks):
    # Documented Anthropic shape: a user message with a tool_result block.
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": [*blocks]},
        ],
    }


def test_images_nested_in_a_tool_result_are_in_the_inventory():
    messages = [tool_result({"type": "text", "text": "chart:"}, PNG, SVG)]
    assert [kind for kind, _ in media_occurrences(messages, ALL)] == ["image"]


def test_an_svg_nested_in_a_tool_result_is_replaced_by_a_note():
    messages = [tool_result({"type": "text", "text": "chart:"}, PNG, SVG)]
    (block,) = filter_media_messages(messages, ALL)[0]["content"]
    assert block["type"] == "tool_result" and block["tool_use_id"] == "toolu_1"
    text, png, note = block["content"]
    assert png == PNG and note["type"] == "text" and "image/svg+xml" in note["text"]
    assert messages[0]["content"][0]["content"][2] is SVG  # history unchanged


def test_a_rejected_nested_image_becomes_the_rejection_note():
    messages = [tool_result(PNG)]
    ((_, key),) = media_occurrences(messages, ALL)
    (block,) = filter_media_messages(messages, ALL, {key})[0]["content"]
    assert block["content"] == [{"type": "text", "text": NOTE}]


def test_an_unchanged_tool_result_is_passed_through_as_is():
    messages = [tool_result({"type": "text", "text": "ok"}, PNG)]
    assert filter_media_messages(messages, ALL)[0]["content"][0] is messages[0]["content"][0]


def test_replayed_native_content_with_an_unsendable_image_falls_back_to_portable():
    native = {"role": "assistant", "content": [{"type": "text", "text": "x"}, SVG]}
    with pytest.raises(ReasoningReplayError):
        check_replayed_native_media(native, ALL)
    check_replayed_native_media({"role": "assistant", "content": [PNG]}, ALL)


class _Event:
    def __init__(self, images):
        self.images = images


def test_image_block_count_counts_only_images_that_are_sent():
    from nooa.runtime.media_recovery import image_block_count

    png_url = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBO"}}
    svg_url = {"type": "image_url", "image_url": {"url": "data:image/svg+xml;base64,PHN2"}}
    unknown = {"type": "image_url", "image_url": {"url": "data:application/octet-stream;base64,AA"}}
    input_image = {"type": "input_image", "image_url": "data:image/jpeg;base64,/9j/"}
    event = _Event([png_url, svg_url, unknown, input_image, PNG, SVG])
    assert image_block_count(event) == 3
