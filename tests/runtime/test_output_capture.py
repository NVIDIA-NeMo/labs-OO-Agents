# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cell stdout/stderr capture through nooa.snapshot: notation, limits, overflow files, the off switch."""

import tempfile
from pathlib import Path

import pytest

from nooa.config.truncation_config import CaptureConfig, TruncationConfig
from nooa.overflow import OverflowDir
from nooa.overflow import _overflow as overflow_module
from nooa.runtime.actor import _make_capture_stream, _render_capture
from nooa.runtime.context_vars import capture_overflow
from nooa.snapshot import SharedTextStream


def _config(**capture: object) -> TruncationConfig:
    return TruncationConfig(capture=CaptureConfig(**capture))


@pytest.fixture
def fresh_default(monkeypatch):
    """No process-default folder yet; any made in the test is closed after it."""
    monkeypatch.setattr(overflow_module, "_default", None)
    monkeypatch.setattr(overflow_module, "_default_failed", False)
    yield
    if overflow_module._default is not None:
        overflow_module._default.close()


def test_limits_are_unchanged():
    config = CaptureConfig()
    assert (config.max_stdout, config.max_stderr, config.max_error) == (50_000, 2_000, 10_000)
    assert config.tail is None and config.file_backed is False


def test_a_capture_is_a_shared_text_stream():
    stream = _make_capture_stream(10, _config())
    assert isinstance(stream, SharedTextStream) and stream.unit == "chars"
    assert (stream.head, stream.tail) == (5, 5)


def test_cut_stdout_keeps_head_and_tail_around_render_output_line():
    stream = _make_capture_stream(50_000, _config())
    data = "".join(f"{i:09d}\n" for i in range(10_000))  # 100,000 chars
    for start in range(0, len(data), 4096):
        stream.write(data[start : start + 4096])
    text = stream.snapshot()
    assert not text.complete and text.length == 100_000
    assert _render_capture(stream, CaptureConfig()) == (
        data[:25_000] + "\n[... 50,000 chars omitted; full output not kept]\n" + data[-25_000:]
    )


def test_whole_output_is_verbatim_up_to_the_limit():
    stream = _make_capture_stream(2_000, _config())
    stream.write("e" * 2_000)
    assert _render_capture(stream, CaptureConfig()) == "e" * 2_000
    assert stream.snapshot().complete


def test_configured_tail_split():
    stream = _make_capture_stream(100, _config(tail=90))
    stream.write("".join(chr(65 + i % 26) for i in range(1000)))
    head, line, tail = _render_capture(stream, CaptureConfig(tail=90)).split("\n")
    assert len(head) == 10 and len(tail) == 90
    assert line == "[... 900 chars omitted; full output not kept]"


def test_unrecoverable_hint_follows_the_cut_line_only_without_a_file():
    capture = CaptureConfig(unrecoverable_hint="Write large output to a file.")
    stream = _make_capture_stream(10, TruncationConfig(capture=capture))
    stream.write("[... fake ]" + "x" * 100)  # head text that looks like a marker
    assert _render_capture(stream, capture) == (
        "[... "
        + "\n[... 101 chars omitted; full output not kept] Write large output to a file.\n"
        + "xxxxx"
    )


def test_file_backed_writes_into_the_session_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    capture = CaptureConfig(tail=10, file_backed=True)
    with OverflowDir() as folder:
        token = capture_overflow.set(folder)
        try:
            stream = _make_capture_stream(20, TruncationConfig(capture=capture))
        finally:
            capture_overflow.reset(token)
        stream.write("0123456789" * 10)
        path = stream.close().full_path
        assert path is not None and Path(path).parent == Path(folder.path)
        assert Path(folder.path).parent == tmp_path  # a private folder in the temp directory
        assert _render_capture(stream, capture) == (
            f"0123456789\n[... 80 chars omitted; full output: {path}]\n0123456789"
        )
        assert Path(path).read_text() == "0123456789" * 10
    assert not Path(path).exists()  # the folder's close deletes it


def test_switch_off_writes_no_file_even_with_a_session_folder(tmp_path, monkeypatch):
    # TruncationConfig.overflow_files=False: memory only, whatever file_backed says.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    config = TruncationConfig(
        overflow_files=False, capture=CaptureConfig(tail=10, file_backed=True)
    )
    with OverflowDir() as folder:
        token = capture_overflow.set(folder)
        try:
            stream = _make_capture_stream(20, config)
        finally:
            capture_overflow.reset(token)
        stream.write("0123456789" * 10)
        assert stream.close().full_path is None
        assert _render_capture(stream, config.capture) == (
            "0123456789\n[... 80 chars omitted; full output not kept]\n0123456789"
        )
        assert list(Path(folder.path).iterdir()) == []


def test_switch_off_never_touches_nooa_overflow(tmp_path, monkeypatch, fresh_default):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    def no_default():
        raise AssertionError("the off switch reached nooa.overflow")

    monkeypatch.setattr(OverflowDir, "default", staticmethod(no_default))
    config = TruncationConfig(overflow_files=False, capture=CaptureConfig(tail=5, file_backed=True))
    stream = _make_capture_stream(10, config)
    stream.write("y" * 50)
    assert stream.close().full_path is None
    assert "full output not kept]" in _render_capture(stream, config.capture)
    assert list(tmp_path.iterdir()) == []


def test_bad_sizes_take_no_place_in_the_folder(tmp_path, monkeypatch):
    # Files open only once a stream is cut, so a stream never made takes no place.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with OverflowDir(max_files=1) as folder:
        token = capture_overflow.set(folder)
        try:
            with pytest.raises(ValueError):  # tail longer than the limit
                _make_capture_stream(5, _config(tail=10, file_backed=True))
        finally:
            capture_overflow.reset(token)
        assert list(Path(folder.path).iterdir()) == []
        writer = folder.open()  # the one place is free
        assert writer is not None
        writer.finish(delete=True)


def test_file_backed_uncut_output_keeps_no_file():
    stream = _make_capture_stream(50_000, _config(file_backed=True))
    stream.write("short")
    assert stream.close().full_path is None
    assert _render_capture(stream, CaptureConfig()) == "short"


def test_file_backed_without_a_session_folder_uses_the_process_default(
    tmp_path, monkeypatch, fresh_default
):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    config = _config(tail=5, file_backed=True)
    stream = _make_capture_stream(10, config)
    stream.write("y" * 50)
    path = stream.close().full_path
    folder = OverflowDir.default()
    assert folder is not None and path is not None
    assert Path(path).parent == Path(folder.path) and Path(folder.path).parent == tmp_path
    assert f"full output: {path}]" in _render_capture(stream, config.capture)
    assert Path(path).read_text() == "y" * 50
    again = _make_capture_stream(10, config)
    again.write("z" * 50)
    assert Path(again.close().full_path).parent == Path(folder.path)  # one folder per process


def test_a_folder_that_cannot_be_created_keeps_memory_only(tmp_path, monkeypatch, fresh_default):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "missing"))
    config = _config(tail=5, file_backed=True)
    stream = _make_capture_stream(10, config)  # no OSError
    stream.write("0123456789" * 10)
    assert stream.close().full_path is None
    assert _render_capture(stream, config.capture) == (
        "01234\n[... 90 chars omitted; full output not kept]\n56789"
    )


def test_writes_after_close_are_dropped_and_non_str_rejected():
    stream = _make_capture_stream(10, _config())
    stream.write("abc")
    stream.close()
    assert stream.write("late") == 4 and _render_capture(stream, CaptureConfig()) == "abc"
    with pytest.raises(TypeError):
        _make_capture_stream(10, _config()).write(b"bytes")  # type: ignore[arg-type]
