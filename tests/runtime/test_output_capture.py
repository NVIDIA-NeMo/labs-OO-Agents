# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cell stdout/stderr capture through nooa.snapshot (PR 2b): notation, limits, spill files."""

from pathlib import Path

import pytest

from nooa.config.truncation_config import CaptureConfig
from nooa.runtime.actor import _make_capture_buffer
from nooa.runtime.output_capture import CaptureBuffer, capture_spill
from nooa.snapshot import SpillDir


def test_limits_are_unchanged():
    config = CaptureConfig()
    assert (config.max_stdout, config.max_stderr, config.max_error) == (50_000, 2_000, 10_000)
    assert config.tail is None and config.file_backed is False


def test_cut_stdout_keeps_head_and_tail_around_render_output_line():
    buffer = _make_capture_buffer(50_000, CaptureConfig())
    data = "".join(f"{i:09d}\n" for i in range(10_000))  # 100,000 chars
    for start in range(0, len(data), 4096):
        buffer.write(data[start : start + 4096])
    assert buffer.was_truncated and buffer.chars_written == 100_000
    assert buffer.getvalue() == (
        data[:25_000] + "\n[... 50,000 chars omitted; full output not kept]\n" + data[-25_000:]
    )


def test_whole_output_is_verbatim_up_to_the_limit():
    buffer = _make_capture_buffer(2_000, CaptureConfig())
    buffer.write("e" * 2_000)
    assert buffer.getvalue() == "e" * 2_000 and not buffer.was_truncated


def test_configured_tail_split():
    buffer = _make_capture_buffer(100, CaptureConfig(tail=90))
    buffer.write("".join(chr(65 + i % 26) for i in range(1000)))
    value = buffer.getvalue()
    head, line, tail = value.split("\n")
    assert len(head) == 10 and len(tail) == 90
    assert line == "[... 900 chars omitted; full output not kept]"


def test_unrecoverable_hint_follows_the_cut_line_only_without_a_file():
    buffer = CaptureBuffer(10, unrecoverable_hint="Write large output to a file.")
    buffer.write("[... fake ]" + "x" * 100)  # head text that looks like a marker
    assert buffer.getvalue() == (
        "[... "
        + "\n[... 101 chars omitted; full output not kept] Write large output to a file.\n"
        + "xxxxx"
    )


def test_file_backed_spills_into_the_session_folder(tmp_path):
    with SpillDir(str(tmp_path)) as spills:
        token = capture_spill.set(spills)
        try:
            buffer = _make_capture_buffer(20, CaptureConfig(tail=10, file_backed=True))
        finally:
            capture_spill.reset(token)
        buffer.write("0123456789" * 10)
        buffer.close()
        path = buffer.file_path
        assert path is not None and Path(path).parent == tmp_path
        assert buffer.getvalue() == (
            f"0123456789\n[... 80 chars omitted; full output: {path}]\n0123456789"
        )
        assert Path(path).read_text() == "0123456789" * 10
    assert not Path(path).exists()  # the folder's close deletes it


def test_file_backed_uncut_output_keeps_no_file():
    buffer = _make_capture_buffer(50_000, CaptureConfig(file_backed=True))
    buffer.write("short")
    buffer.close()
    assert buffer.getvalue() == "short" and buffer.file_path is None


def test_file_backed_without_a_session_folder_uses_a_kept_temp_file():
    buffer = _make_capture_buffer(10, CaptureConfig(tail=5, file_backed=True))
    buffer.write("y" * 50)
    buffer.close()
    path = buffer.file_path
    try:
        assert path is not None and f"full output: {path}]" in buffer.getvalue()
        assert Path(path).read_text() == "y" * 50
    finally:
        Path(path).unlink()


def test_writes_after_close_are_ignored_and_non_str_rejected():
    buffer = CaptureBuffer(10)
    buffer.write("abc")
    buffer.close()
    assert buffer.write("late") == 4 and buffer.getvalue() == "abc"
    with pytest.raises(TypeError):
        CaptureBuffer(10).write(b"bytes")  # type: ignore[arg-type]
