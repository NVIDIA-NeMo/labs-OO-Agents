# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TextStream with an OverflowDir: Text.full, caps, eviction and a closed folder, seen through streams.

The folder's own filesystem behaviour is tested in ``tests/overflow``.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import pytest

from nooa.overflow import OverflowDir, OverflowFile
from nooa.snapshot import TextStream


def _stream(file: OverflowFile | None) -> TextStream:
    return TextStream(unit="bytes", head=2, tail=2, overflow=file)


def _cut(folder: OverflowDir, data: bytes = b"0123456789") -> str:
    """A closed, cut stream whose file stays in ``folder``; its path."""
    file = folder.file()
    assert file is not None and file.path is not None
    stream = _stream(file)
    stream.write(data)
    full = stream.close().full
    assert full is not None and full.path == file.path
    return file.path


def _disk(folder: str) -> tuple[int, int]:
    """(files, bytes) of the regular files directly in ``folder``."""
    entries = [e for e in os.scandir(folder) if e.is_file(follow_symlinks=False)]
    return len(entries), sum(e.stat(follow_symlinks=False).st_size for e in entries)


def test_each_stream_file_is_capped_at_file_bytes(tmp_path: Path) -> None:
    with OverflowDir(str(tmp_path), max_files=3, file_bytes=4) as folder:
        paths = [_cut(folder) for _ in range(3)]
        assert [Path(p).read_bytes() for p in paths] == [b"0123"] * 3
        assert _disk(str(tmp_path)) == (3, 12)


def test_one_file_opens_one_stream(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with OverflowDir(str(tmp_path), max_files=2, file_bytes=100) as folder:
        file = folder.file()
        assert file is not None and file.path is not None
        first = _stream(file)
        first.write(b"first")
        with caplog.at_level(logging.WARNING, logger="nooa.overflow._overflow"):
            second = _stream(file)  # refused: memory only, and the first file is untouched
        assert [r.getMessage() for r in caplog.records] == [
            f"overflow file {file.path!r} is not open to a new writer; keeping memory only"
        ]
        second.write(b"second")
        assert second.close().full is None
        assert first.close().full is not None
        assert Path(file.path).read_bytes() == b"first"


def test_evicted_full_output_raises_file_not_found(tmp_path: Path) -> None:
    with OverflowDir(str(tmp_path), max_files=1, file_bytes=100) as folder:
        first = _cut(folder)
        _cut(folder)
        with pytest.raises(FileNotFoundError):  # how a reader tells the full output is gone
            Path(first).open("rb").close()


def test_complete_output_deleted_by_its_stream_is_released(tmp_path: Path) -> None:
    with OverflowDir(str(tmp_path), max_files=1, file_bytes=10) as folder:
        file = folder.file()
        assert file is not None and file.path is not None
        stream = TextStream(unit="bytes", head=10, tail=10, overflow=file)
        stream.write(b"short")
        assert stream.close().full is None
        assert not os.path.exists(file.path)
        assert folder.file() is not None  # its place is free again


def test_a_stream_keeps_running_after_its_folder_closed() -> None:
    folder = OverflowDir(max_files=5, file_bytes=100)
    running = _stream(folder.file())
    running.write(b"0123456789")
    folder.close()
    running.write(b"more")  # the stream keeps running without its file
    assert running.close().length == 14


def test_a_file_after_the_folder_closed_keeps_memory_only(tmp_path: Path) -> None:
    folder = OverflowDir(str(tmp_path), max_files=2, file_bytes=10)
    file = folder.file()
    folder.close()
    stream = _stream(file)
    stream.write(b"0123456789")
    assert stream.close().full is None
    assert list(tmp_path.iterdir()) == []


def test_stream_does_not_open_a_symlink_put_in_its_place(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("precious")
    given = tmp_path / "given"
    given.mkdir()
    with OverflowDir(str(given), max_files=1, file_bytes=100) as folder:
        file = folder.file()
        assert file is not None and file.path is not None
        os.unlink(file.path)
        os.symlink(outside, file.path)
        stream = _stream(file)
        stream.write(b"0123456789")
        assert stream.close().full is None  # memory only
        assert outside.read_text() == "precious"


def test_many_streams_never_exceed_the_budget(tmp_path: Path) -> None:
    rng = random.Random(7)
    file_bytes, max_files = 800, 6
    with OverflowDir(str(tmp_path), max_files=max_files, file_bytes=file_bytes) as folder:
        running: list[TextStream] = []
        written = 0
        for _ in range(300):
            if running and (len(running) > 3 or rng.random() < 0.5):
                running.pop(rng.randrange(len(running))).close()
            else:
                running.append(_stream(folder.file()))
            for stream in running:
                chunk = os.urandom(rng.randrange(0, 1500))
                written += len(chunk)
                stream.write(chunk)
            files, size = _disk(str(tmp_path))
            assert files <= max_files and size <= max_files * file_bytes
        for stream in running:
            stream.close()
        assert written > 10 * max_files * file_bytes
