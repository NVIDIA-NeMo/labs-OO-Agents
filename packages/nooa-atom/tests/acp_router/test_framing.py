# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JSON-RPC line framing shared by the router and its tests."""

import asyncio
import json
import logging

from nooa_atom.acp.framing import FRAME_LIMIT, Frame, encode, read_frame


def _reader(data: bytes, *, limit: int = FRAME_LIMIT) -> asyncio.StreamReader:
    reader = asyncio.StreamReader(limit=limit)
    reader.feed_data(data)
    reader.feed_eof()
    return reader


async def _all(reader: asyncio.StreamReader) -> list[Frame]:
    frames = []
    while (frame := await read_frame(reader)) is not None:
        frames.append(frame)
    return frames


async def test_frames_keep_their_original_bytes():
    line = (
        b'{"jsonrpc": "2.0", "id": 7, "method": "session/prompt", "params": {"sessionId": "s1"}}\n'
    )
    [frame] = await _all(_reader(line))
    assert frame.raw == line
    assert frame.id == 7 and frame.has_id
    assert frame.method == "session/prompt"
    assert frame.session_id == "s1"
    assert frame.is_request and not frame.is_notification and not frame.is_response


async def test_envelope_kinds():
    notification, response, bare = await _all(
        _reader(
            b'{"jsonrpc":"2.0","method":"session/cancel","params":{"sessionId":"s2"}}\n'
            b'{"jsonrpc":"2.0","id":"abc","result":null}\n'
            b'{"jsonrpc":"2.0","method":"initialize","id":0,"params":[1]}\n'
        )
    )
    assert notification.is_notification and notification.session_id == "s2"
    assert notification.id is None and not notification.has_id
    assert response.is_response and response.id == "abc" and response.method is None
    # params that are not an object carry no session id
    assert bare.is_request and bare.session_id is None


async def test_blank_lines_are_skipped():
    frames = await _all(
        _reader(b'\n  \r\n{"jsonrpc":"2.0","method":"a"}\n\n{"jsonrpc":"2.0","method":"b"}\n')
    )
    assert [frame.method for frame in frames] == ["a", "b"]


async def test_bad_json_and_batches_are_dropped_with_a_log_line(caplog):
    caplog.set_level(logging.WARNING, logger="nooa_atom.acp.framing")
    frames = await _all(
        _reader(b'not json\n[{"jsonrpc":"2.0","method":"a"}]\n42\n{"jsonrpc":"2.0","method":"b"}\n')
    )
    assert [frame.method for frame in frames] == ["b"]
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 3
    assert any("not JSON" in message for message in messages)
    assert any("batch" in message for message in messages)


async def test_a_last_line_without_newline_is_a_frame():
    [frame] = await _all(_reader(b'{"jsonrpc":"2.0","method":"a"}'))
    assert frame.method == "a"
    assert frame.raw == b'{"jsonrpc":"2.0","method":"a"}\n'


async def test_end_of_stream_is_none():
    assert await read_frame(_reader(b"")) is None


async def test_a_five_mib_frame_passes_whole():
    text = "x" * (5 * 1024 * 1024)
    line = encode({"jsonrpc": "2.0", "method": "big", "params": {"text": text}})
    frame, after = await _all(_reader(line + b'{"jsonrpc":"2.0","method":"after"}\n'))
    assert frame.raw == line and after.method == "after"
    assert frame.message["params"]["text"] == text


async def test_a_frame_over_the_limit_is_dropped_and_reading_continues(caplog):
    caplog.set_level(logging.WARNING, logger="nooa_atom.acp.framing")
    big = encode({"jsonrpc": "2.0", "method": "big", "params": {"text": "y" * 5000}})
    frames = await _all(_reader(big + b'{"jsonrpc":"2.0","method":"after"}\n', limit=1024))
    assert [frame.method for frame in frames] == ["after"]
    assert any("limit" in record.getMessage() for record in caplog.records)


def test_encode_matches_the_library_framing():
    message = {"jsonrpc": "2.0", "id": 1, "result": {"a": [1, 2], "é": None}}
    assert encode(message) == (json.dumps(message, separators=(",", ":")) + "\n").encode()
