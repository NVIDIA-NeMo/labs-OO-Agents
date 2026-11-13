# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cell stdout/stderr capture through nooa.snapshot (PR 2b): notation, limits, overflow files."""

import tempfile
from pathlib import Path

import pytest

from nooa.config.truncation_config import CaptureConfig
from nooa.overflow import OverflowDir
from nooa.runtime import output_capture
from nooa.runtime.actor import _make_capture_buffer
from nooa.runtime.output_capture import CaptureBuffer, capture_overflow


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


def test_file_backed_writes_into_the_session_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with OverflowDir() as folder:
        token = capture_overflow.set(folder)
        try:
            buffer = _make_capture_buffer(20, CaptureConfig(tail=10, file_backed=True))
        finally:
            capture_overflow.reset(token)
        buffer.write("0123456789" * 10)
        buffer.close()
        path = buffer.file_path
        assert path is not None and Path(path).parent == Path(folder.path)
        assert Path(folder.path).parent == tmp_path  # a private folder in the temp directory
        assert buffer.getvalue() == (
            f"0123456789\n[... 80 chars omitted; full output: {path}]\n0123456789"
        )
        assert Path(path).read_text() == "0123456789" * 10
    assert not Path(path).exists()  # the folder's close deletes it


def test_switch_off_writes_no_file_even_with_a_session_folder(tmp_path, monkeypatch):
    # TruncationConfig.overflow_files=False: memory only, whatever file_backed says.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with OverflowDir() as folder:
        token = capture_overflow.set(folder)
        try:
            buffer = _make_capture_buffer(
                20, CaptureConfig(tail=10, file_backed=True), overflow_files=False
            )
        finally:
            capture_overflow.reset(token)
        buffer.write("0123456789" * 10)
        buffer.close()
        assert buffer.file_path is None
        assert buffer.getvalue() == (
            "0123456789\n[... 80 chars omitted; full output not kept]\n0123456789"
        )
        assert list(Path(folder.path).iterdir()) == []


def test_switch_off_writes_no_temp_file_without_a_session_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(output_capture, "_own_folder", None)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    buffer = _make_capture_buffer(10, CaptureConfig(tail=5, file_backed=True), overflow_files=False)
    buffer.write("y" * 50)
    buffer.close()
    assert buffer.file_path is None and "full output not kept]" in buffer.getvalue()
    assert list(tmp_path.iterdir()) == []


def test_bad_sizes_take_no_place_in_the_folder(tmp_path, monkeypatch):
    # model.md C2 (output_capture.py:72-74): the file was handed out before
    # TextStream checked its sizes, and kept its place for the folder's life.
    # Files now open only once a stream is cut, so a stream never made takes none.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with OverflowDir(max_files=1) as folder:
        token = capture_overflow.set(folder)
        try:
            with pytest.raises(ValueError):
                CaptureBuffer(5, 10, file_backed=True)  # tail longer than the limit
        finally:
            capture_overflow.reset(token)
        assert list(Path(folder.path).iterdir()) == []
        writer = folder.open()  # the one place is free
        assert writer is not None
        writer.finish(delete=True)


def test_file_backed_uncut_output_keeps_no_file():
    buffer = _make_capture_buffer(50_000, CaptureConfig(file_backed=True))
    buffer.write("short")
    buffer.close()
    assert buffer.getvalue() == "short" and buffer.file_path is None


def test_file_backed_without_a_session_folder_uses_the_process_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(output_capture, "_own_folder", None)
    buffer = _make_capture_buffer(10, CaptureConfig(tail=5, file_backed=True))
    buffer.write("y" * 50)
    buffer.close()
    path = buffer.file_path
    folder = output_capture._own_folder
    try:
        assert folder is not None and path is not None
        assert Path(path).parent == Path(folder.path) and Path(folder.path).parent == tmp_path
        assert f"full output: {path}]" in buffer.getvalue()
        assert Path(path).read_text() == "y" * 50
        again = _make_capture_buffer(10, CaptureConfig(tail=5, file_backed=True))
        again.write("z" * 50)
        again.close()
        assert Path(again.file_path).parent == Path(folder.path)  # one folder per process
    finally:
        folder.close()
    assert not Path(path).exists()


def test_writes_after_close_are_ignored_and_non_str_rejected():
    buffer = CaptureBuffer(10)
    buffer.write("abc")
    buffer.close()
    assert buffer.write("late") == 4 and buffer.getvalue() == "abc"
    with pytest.raises(TypeError):
        CaptureBuffer(10).write(b"bytes")  # type: ignore[arg-type]


def test_a_disabled_session_context_writes_no_file_even_with_file_backed(tmp_path, monkeypatch):
    # Round H item 1: the off switch lives in the context, honoured by the buffer
    # itself, so an entry point that forgets to clear file_backed writes nothing.
    from nooa.runtime.output_capture import NO_OVERFLOW_FILES

    monkeypatch.setattr(output_capture, "_own_folder", None)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    token = capture_overflow.set(NO_OVERFLOW_FILES)
    try:
        buffer = CaptureBuffer(4, file_backed=True)
    finally:
        capture_overflow.reset(token)
    buffer.write("x" * 100)
    buffer.close()
    assert buffer.file_path is None
    assert buffer.getvalue() == "xx\n[... 96 chars omitted; full output not kept]\nxx"
    assert list(tmp_path.iterdir()) == []
