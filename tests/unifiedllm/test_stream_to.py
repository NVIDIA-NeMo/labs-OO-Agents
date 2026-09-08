# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`stream_to`: what reaches the sink, and what must never reach it twice."""

from types import SimpleNamespace

import pytest

from nooa.unifiedllm.unifiedllm import (
    _chunk_emitted,
    _chunk_sink,
    _emit_chunk,
    stream_to,
)


def chunk(content=None, reasoning=None):
    """A litellm streaming chunk, shaped the way the collector sees one."""
    delta = SimpleNamespace(content=content, reasoning_content=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def test_no_sink_means_no_emission():
    _emit_chunk(chunk("hello"))  # must not raise with nothing installed


def test_the_sink_receives_content_deltas():
    seen = []
    with stream_to(seen.append):
        _emit_chunk(chunk("mộ"))
        _emit_chunk(chunk("t hai"))
    assert seen == ["mộ", "t hai"]


def test_reasoning_is_not_prose_on_an_ordinary_call():
    """For a plain call, reasoning_content is the model thinking, not the answer."""
    seen = []
    with stream_to(seen.append):
        _emit_chunk(chunk(content=None, reasoning="đang nghĩ..."))
    assert seen == []


def test_structured_output_hiding_in_reasoning_still_reaches_the_sink():
    """Some reasoning models put the JSON in reasoning_content and leave content
    empty; the parser already falls back to it, so a sink reading only `content`
    would watch a call succeed with nothing on screen."""
    seen = []
    with stream_to(seen.append):
        _emit_chunk(chunk(content=None, reasoning='{"value": "xin'), structured=True)
        _emit_chunk(chunk(content=None, reasoning=' chào"}'), structured=True)
    assert "".join(seen) == '{"value": "xin chào"}'


def test_content_wins_over_reasoning_when_both_are_present():
    seen = []
    with stream_to(seen.append):
        _emit_chunk(chunk(content="thật", reasoning="nghĩ"), structured=True)
    assert seen == ["thật"]


def test_a_raising_sink_never_breaks_the_call():
    def explode(_):
        raise RuntimeError("cái màn hình hỏng")

    with stream_to(explode):
        _emit_chunk(chunk("vẫn chạy"))  # swallowed, logged at debug


def test_emission_is_recorded_so_a_retry_can_be_refused():
    """The flag behind "do not replay what the caller has already seen"."""
    token = _chunk_emitted.set(False)
    try:
        with stream_to(lambda _: None):
            assert _chunk_emitted.get() is False
            _emit_chunk(chunk("một"))
            assert _chunk_emitted.get() is True
    finally:
        _chunk_emitted.reset(token)


def test_a_silent_chunk_does_not_count_as_emission():
    token = _chunk_emitted.set(False)
    try:
        with stream_to(lambda _: None):
            _emit_chunk(chunk(content=""))
            _emit_chunk(chunk(content=None, reasoning="nghĩ"))  # not structured
            assert _chunk_emitted.get() is False
    finally:
        _chunk_emitted.reset(token)


def test_the_sink_is_scoped_to_its_context():
    seen = []
    with stream_to(seen.append):
        pass
    assert _chunk_sink.get() is None
    _emit_chunk(chunk("sau khi ra khỏi context"))
    assert seen == []


@pytest.mark.asyncio
async def test_a_failure_after_streaming_is_terminal_not_retried():
    """A retry would replay the prefix the caller already saw.

    Duplicated text on screen, and a corrupted read for anything parsing the
    stream incrementally — so the failure becomes the answer.
    """
    from nooa.unifiedllm.retry import with_retry
    from nooa.unifiedllm.retry_config import RetryConfig
    from nooa.unifiedllm.unifiedllm import StreamedBeforeFailing

    seen, attempts = [], []

    async def attempt():
        attempts.append(1)
        try:
            _emit_chunk(chunk("một phần câu trả lời"))
            raise TimeoutError("mạng đứt giữa chừng")
        except StreamedBeforeFailing:
            raise
        except Exception as failure:
            if _chunk_emitted.get():
                raise StreamedBeforeFailing("sink đã nhận một phần") from failure
            raise

    token = _chunk_emitted.set(False)
    try:
        with stream_to(seen.append), pytest.raises(StreamedBeforeFailing):
            await with_retry(attempt, config=RetryConfig(max_retries=3))
    finally:
        _chunk_emitted.reset(token)

    assert len(attempts) == 1, "đã phát rồi thì không được thử lại"
    assert seen == ["một phần câu trả lời"], "người dùng chỉ được thấy một lần"
