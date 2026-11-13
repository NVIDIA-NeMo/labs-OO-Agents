# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TextStream with an OverflowDir: Text.full, caps, eviction and a closed folder, seen through streams.

The folder's own filesystem behaviour is tested in ``tests/overflow``.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

from nooa.overflow import OverflowDir
from nooa.snapshot import TextStream


def _stream(folder: OverflowDir | None) -> TextStream:
    return TextStream(unit="bytes", head=2, tail=2, overflow=folder)


def _cut(folder: OverflowDir, data: bytes = b"0123456789") -> str:
    """A closed, cut stream whose file stays in ``folder``; its path."""
    stream = _stream(folder)
    stream.write(data)
    full = stream.close().full
    assert full is not None and os.path.dirname(full.path) == folder.path
    return full.path


def _disk(folder: str) -> tuple[int, int]:
    """(files, bytes) of the regular files directly in ``folder``."""
    entries = [e for e in os.scandir(folder) if e.is_file(follow_symlinks=False)]
    return len(entries), sum(e.stat(follow_symlinks=False).st_size for e in entries)


def test_each_stream_file_is_capped_at_file_bytes() -> None:
    with OverflowDir(max_files=3, file_bytes=4) as folder:
        paths = [_cut(folder) for _ in range(3)]
        assert [Path(p).read_bytes() for p in paths] == [b"0123"] * 3
        assert _disk(folder.path) == (3, 12)


def test_evicted_full_output_raises_file_not_found() -> None:
    with OverflowDir(max_files=1, file_bytes=100) as folder:
        first = _cut(folder)
        _cut(folder)
        with pytest.raises(FileNotFoundError):  # how a reader tells the full output is gone
            Path(first).open("rb").close()


def test_a_complete_output_takes_no_place() -> None:
    # Lazy opening: a stream that is never cut makes no file, so it never holds a
    # place nor evicts an earlier output.
    with OverflowDir(max_files=1, file_bytes=10) as folder:
        kept = _cut(folder)
        stream = TextStream(unit="bytes", head=10, tail=10, overflow=folder)
        stream.write(b"short")
        assert stream.close().full is None
        assert os.listdir(folder.path) == [os.path.basename(kept)]


def test_a_stream_cut_while_the_budget_is_in_use_keeps_memory_only() -> None:
    with OverflowDir(max_files=1, file_bytes=100) as folder:
        running = _stream(folder)
        running.write(b"0123456789")  # cut: holds the one place
        late = _stream(folder)
        late.write(b"abcdefgh")  # cut too, but no place: memory only
        assert late.close().full is None
        assert running.close().full is not None


def test_a_stream_keeps_running_after_its_folder_closed() -> None:
    folder = OverflowDir(max_files=5, file_bytes=100)
    running = _stream(folder)
    running.write(b"0123456789")
    folder.close()
    running.write(b"more")  # the stream keeps running without its file
    assert running.close().length == 14


def test_a_stream_first_cut_after_the_folder_closed_keeps_memory_only() -> None:
    folder = OverflowDir(max_files=2, file_bytes=10)
    stream = _stream(folder)
    stream.write(b"01")
    folder.close()
    stream.write(b"23456789")
    assert stream.close().full is None
    assert not os.path.exists(folder.path)


def test_stream_does_not_write_a_symlink_put_in_its_place(tmp_path: Path) -> None:
    # The stream writes the descriptor its folder created, never the name.
    outside = tmp_path / "outside.txt"
    outside.write_text("precious")
    with OverflowDir(max_files=1, file_bytes=100) as folder:
        stream = _stream(folder)
        stream.write(b"01234")
        (name,) = os.listdir(folder.path)
        path = os.path.join(folder.path, name)
        os.unlink(path)
        os.symlink(outside, path)
        stream.write(b"56789")
        stream.close()
        assert outside.read_text() == "precious"


def test_many_streams_never_exceed_the_budget() -> None:
    rng = random.Random(7)
    file_bytes, max_files = 800, 6
    with OverflowDir(max_files=max_files, file_bytes=file_bytes) as folder:
        running: list[TextStream] = []
        written = 0
        for _ in range(300):
            if running and (len(running) > 3 or rng.random() < 0.5):
                running.pop(rng.randrange(len(running))).close()
            else:
                running.append(_stream(folder))
            for stream in running:
                chunk = os.urandom(rng.randrange(0, 1500))
                written += len(chunk)
                stream.write(chunk)
            files, size = _disk(folder.path)
            assert files <= max_files and size <= max_files * file_bytes
        for stream in running:
            stream.close()
        assert written > 10 * max_files * file_bytes
