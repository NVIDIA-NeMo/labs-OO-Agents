# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A provider's per-request image limit collapses the oldest image events, then retries."""

import json
from collections.abc import Mapping

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.events import PythonOutput, Summary
from nooa.media import Image
from nooa.runtime.harness_metrics import HarnessMetrics
from nooa.strategies.codeact import CodeActStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from nooa.unifiedllm.media_inputs import explicitly_rejected, image_limit


class _TooManyImages(Exception):
    status_code = 400


def _images(messages) -> int:
    return sum(
        1
        for msg in messages
        if isinstance(msg, Mapping) and isinstance(msg.get("content"), (list, tuple))
        for block in msg["content"]
        if isinstance(block, Mapping) and block.get("type") == "image_url"
    )


class LimitedClient(FakeLLMClient):
    """Rejects a request carrying more than ``limit`` images, like the GLM 5.3 hub."""

    limit = 1

    async def acall(self, messages, *args, **kwargs):
        sent = _images(self._prepare_media_messages(messages, kwargs))
        self.sent_images = [*getattr(self, "sent_images", []), sent]
        if sent > self.limit:
            raise _TooManyImages(f"Too many images: {sent} (max {self.limit})")
        return await super().acall(messages, *args, **kwargs)


def response(code, call_id):
    return LLMResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            ToolCall(id=call_id, name="execute_python", arguments=json.dumps({"code": code}))
        ],
    )


class ImageAgent(Agent, llm=FakeLLMClient()):
    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=6)))
    async def look(self) -> str:
        """Look at images with show()."""
        ...


@pytest.mark.parametrize(
    "text,limit",
    [
        ("Too many images: 7 (max 5)", 5),
        ("too many images in request (7), max 5", 5),
        ("You can send at most 3 images per request", 3),
        ("Maximum of 10 images allowed", 10),
        ("request exceeds the maximum of 20 images", 20),
        ("Invalid image", None),
        ("context length exceeded", None),
    ],
)
def test_image_limit_is_read_from_the_provider_error(text, limit):
    assert image_limit(_TooManyImages(text)) == limit


def test_an_image_limit_is_not_a_rejection_of_one_image():
    assert not explicitly_rejected(_TooManyImages("Too many images: 7 (max 5)"), "image")


async def test_the_oldest_images_are_collapsed_and_can_be_shown_again(monkeypatch):
    flushed: list[dict] = []
    real_flush = HarnessMetrics.flush_to_span

    def flush(self):
        flushed.append(self.to_span_attributes())
        real_flush(self)

    monkeypatch.setattr(HarnessMetrics, "flush_to_span", flush)
    client = LimitedClient(
        [
            response("show(self.first)", "c1"),
            response("show(self.second)", "c2"),  # the next request has 2 images: limit 1
            response(  # the summary's own recipe shows the collapsed image again
                "[t] = [t for t in self.events.keys() if '..' in t]\n"
                "tag = t.split('..')[0]\n"
                'show(Image.from_url(self.events[tag].images[0]["image_url"]["url"]))',
                "c3",
            ),
            response("return_result('done')", "c4"),
        ],
        media_inputs=["image"],
    )
    agent = ImageAgent(llm=client)
    agent.first = Image.from_bytes(b"first-png", media_type="image/png")
    agent.second = Image.from_bytes(b"second-png", media_type="image/png")
    assert await agent.look() == "done"
    # Rejected once with 2 images; every later request stays under the remembered limit.
    assert client.sent_images[client.sent_images.index(2) + 1 :] == [1] * (
        len(client.sent_images) - client.sent_images.index(2) - 1
    )
    assert client.sent_images.count(2) == 1
    em = agent.event_manager
    # Only the newest image (c3 showed the first one again) is active; the
    # remembered limit collapsed the second before the next request.
    [newest] = [e for e in em.values() if isinstance(e, PythonOutput) and e.images]
    assert newest.images[0]["image_url"]["url"] == agent.first.data_url
    summary_tag, second_tag = [k for k in em.keys() if ".." in k]
    tag = summary_tag.split("..")[0]
    assert em[second_tag.split("..")[0]].images[0]["image_url"]["url"] == agent.second.data_url
    assert summary_tag == f"{tag}..{tag}" and tag not in em.keys()  # collapsed on its own ...
    summary = em[summary_tag]
    assert isinstance(summary, Summary)
    assert summary.summary_text == (
        f'[image collapsed (request image limit 1); original: self.events["{tag}"]; '
        f'show it again: show(Image.from_url(self.events["{tag}"].images[0]["image_url"]["url"]))]'
    )
    # ... and the original is still there, with an image that can be shown again.
    original = em[tag]
    again = Image.from_url(original.images[0]["image_url"]["url"])
    assert again.data_url == agent.first.data_url
    [attrs] = [a for a in flushed if "harness.media_image_limit_collapsed.count" in a]
    assert attrs["harness.media_image_limit_collapsed.count"] == 2  # the retry, then proactive
    assert attrs["harness.media_image_limit_collapsed.limits"] == [1, 1]
    assert attrs["harness.media_image_limit_collapsed.events"] == 2
