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
