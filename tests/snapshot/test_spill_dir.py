# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SpillDir: a folder of at most max_files files of file_bytes each, eviction, cleanup and threads."""

from __future__ import annotations

import dataclasses
import gc
import logging
import os
import random
import stat
import threading
from pathlib import Path

import pytest

from nooa.snapshot import SpillDir, SpillTarget, TextStream


def _stream(target: SpillTarget | None) -> TextStream:
    return TextStream(unit="bytes", head=2, tail=2, spill=target)


def _cut(spills: SpillDir, data: bytes = b"0123456789") -> str:
    """A closed, cut stream whose file stays in ``spills``; its path."""
    target = spills.target()
    assert target is not None and target.path is not None
    stream = _stream(target)
    stream.write(data)
    stream.close()
    return target.path


def _disk(folder: str) -> tuple[int, int]:
    """(files, bytes) of the regular files directly in ``folder``."""
    entries = [e for e in os.scandir(folder) if e.is_file(follow_symlinks=False)]
    return len(entries), sum(e.stat(follow_symlinks=False).st_size for e in entries)


def test_defaults_and_private_folder() -> None:
    spills = SpillDir()
    assert spills.max_files == 20 and spills.file_bytes == 64 * 2**20
    assert stat.S_IMODE(os.stat(spills.path).st_mode) == 0o700  # tempfile.mkdtemp
    target = spills.target()
    assert target is not None and target.path is not None
    assert os.path.dirname(target.path) == spills.path
    assert stat.S_IMODE(os.stat(target.path).st_mode) == 0o600  # tempfile.mkstemp
    assert target.max_bytes == 64 * 2**20  # each file gets file_bytes
    (Path(spills.path) / "stray.txt").write_text("x")  # anything in a private folder goes
    spills.close()
    assert not os.path.exists(spills.path)


def test_invalid_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        SpillDir(str(tmp_path / "missing"))
    (tmp_path / "file").write_text("x")
    with pytest.raises(ValueError):
        SpillDir(str(tmp_path / "file"))
    with pytest.raises(ValueError):
        SpillDir(str(tmp_path), max_files=-1)
    with pytest.raises(ValueError):
        SpillDir(str(tmp_path), file_bytes=-1)


def test_target_without_a_folder_is_unchanged() -> None:
    assert SpillTarget() == SpillTarget(path=None, max_bytes=64 * 2**20)
    assert repr(SpillTarget(path="x")) == "SpillTarget(path='x', max_bytes=67108864)"


def test_each_file_is_capped_at_file_bytes(tmp_path: Path) -> None:
    with SpillDir(str(tmp_path), max_files=3, file_bytes=4) as spills:
        paths = [_cut(spills) for _ in range(3)]
        assert [Path(p).read_bytes() for p in paths] == [b"0123"] * 3
        assert _disk(str(tmp_path)) == (3, 12)  # max_files * file_bytes at most


def test_replace_keeps_the_folder(tmp_path: Path) -> None:
    # Severin's repro (#515): ``replace`` used to drop the private folder link.
    with SpillDir(str(tmp_path), max_files=5, file_bytes=1000) as spills:
        target = spills.target()
        assert target is not None and target.path is not None
        kept = dataclasses.replace(target, max_bytes=3)

        assert kept._dir is spills and kept.max_bytes == 3
        assert kept == dataclasses.replace(kept, _dir=None)  # not compared
        assert "_dir" not in repr(kept)
        stream = _stream(kept)
        stream.write(b"hello")
        assert stream.close().full is not None
        assert Path(target.path).read_bytes() == b"hel"  # cut: kept, and still the folder's
    assert not os.path.exists(target.path)  # deleted by the folder's close()


def test_replace_cannot_raise_the_file_cap(tmp_path: Path) -> None:
    with SpillDir(str(tmp_path), max_files=2, file_bytes=5) as spills:
        target = spills.target()
        assert target is not None and target.max_bytes == 5
        stream = _stream(dataclasses.replace(target, max_bytes=1000))
        stream.write(b"0123456789")
        full = stream.close().full
        assert full is not None and full.kept == 5


def test_one_target_opens_one_stream(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with SpillDir(str(tmp_path), max_files=2, file_bytes=100) as spills:
        target = spills.target()
        assert target is not None and target.path is not None
        first = _stream(target)
        first.write(b"first")
        with caplog.at_level(logging.WARNING, logger="nooa.snapshot._spill"):
            second = _stream(target)  # refused: memory only, and the first file is untouched
        # A reused target is a usage error, not a file error: no traceback.
        assert [r.getMessage() for r in caplog.records] == [
            f"spill target {target.path!r} is not open to a new stream; keeping memory only"
        ]
        assert all(r.exc_info is None for r in caplog.records)
        second.write(b"second")
        assert second.close().full is None
        assert first.close().full is not None
        assert Path(target.path).read_bytes() == b"first"
        assert _stream(target).close().full is None  # closed now: still refused


def test_budget_by_count_evicts_the_oldest_closed_file(tmp_path: Path) -> None:
    # Keep the newest: an evicted path raises FileNotFoundError.
    with SpillDir(str(tmp_path), max_files=2, file_bytes=1000) as spills:
        first = _cut(spills)
        second = _cut(spills)
        third = _cut(spills)
        assert not os.path.exists(first)
        assert os.path.exists(second) and os.path.exists(third)
        fourth = _cut(spills)
        assert not os.path.exists(second)
        assert os.path.exists(third) and os.path.exists(fourth)
        with pytest.raises(FileNotFoundError):  # how a reader tells it is no longer kept
            Path(first).open("rb").close()


def test_open_stream_file_is_never_evicted(tmp_path: Path) -> None:
    with SpillDir(str(tmp_path), max_files=2, file_bytes=100) as spills:
        open_target = spills.target()
        assert open_target is not None and open_target.path is not None
        running = _stream(open_target)
        running.write(b"0123456789")
        closed = _cut(spills)
        target = spills.target()  # count full: only the closed one can go
        assert target is not None
        assert not os.path.exists(closed) and os.path.exists(open_target.path)
        assert spills.target() is None  # two open files: nothing to evict
        assert os.path.exists(open_target.path)
        running.write(b"abc")
        full = running.close().full
        assert full is not None and Path(full.path).read_bytes() == b"0123456789abc"
        _stream(target).close()  # complete: deleted, its place is free
        assert spills.target() is not None  # and the closed file is evicted only now
        assert os.path.exists(full.path)
        assert spills.target() is not None
        assert not os.path.exists(full.path)


def test_unused_target_is_never_evicted(tmp_path: Path) -> None:
    with SpillDir(str(tmp_path), max_files=1, file_bytes=10) as spills:
        handed = spills.target()
        assert handed is not None and handed.path is not None
        assert spills.target() is None
        assert os.path.exists(handed.path)


def test_zero_budget_creates_no_file(tmp_path: Path) -> None:
    with SpillDir(str(tmp_path), max_files=0) as spills:
        assert spills.target() is None
    with SpillDir(str(tmp_path), file_bytes=0) as spills:
        assert spills.target() is None
    assert list(tmp_path.iterdir()) == []


def test_complete_output_deleted_by_its_stream_is_released(tmp_path: Path) -> None:
    with SpillDir(str(tmp_path), max_files=1, file_bytes=10) as spills:
        target = spills.target()
        assert target is not None and target.path is not None
        stream = TextStream(unit="bytes", head=10, tail=10, spill=target)
        stream.write(b"short")
        assert stream.close().full is None
        assert not os.path.exists(target.path)
        assert spills.target() is not None  # its place is free again


def test_close_deletes_everything_and_is_idempotent() -> None:
    spills = SpillDir(max_files=5, file_bytes=100)
    _cut(spills)
    running = _stream(spills.target())
    running.write(b"0123456789")
    spills.target()  # handed out, never used
    spills.close()
    assert not os.path.exists(spills.path)
    spills.close()
    with pytest.raises(ValueError):
        spills.target()
    running.write(b"more")  # the stream keeps running without its file
    assert running.close().length == 14


def test_folder_is_removed_when_collected() -> None:
    spills = SpillDir(max_files=2, file_bytes=10)
    _cut(spills)
    path = spills.path
    del spills
    gc.collect()  # weakref.finalize: the same cleanup runs at exit too
    assert not os.path.exists(path)


def test_target_after_the_folder_closed_keeps_memory_only(tmp_path: Path) -> None:
    spills = SpillDir(str(tmp_path), max_files=2, file_bytes=10)
    target = spills.target()
    spills.close()
    stream = _stream(target)
    stream.write(b"0123456789")
    assert stream.close().full is None
    assert list(tmp_path.iterdir()) == []


def test_given_folder_keeps_foreign_files(tmp_path: Path) -> None:
    foreign = tmp_path / "keep.txt"
    foreign.write_text("mine")
    (tmp_path / "sub").mkdir()
    with SpillDir(str(tmp_path), max_files=3, file_bytes=100) as spills:
        _cut(spills)
        _cut(spills)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.txt", "sub"]
    assert foreign.read_text() == "mine"


def test_given_folder_never_follows_a_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("precious")
    folder = tmp_path / "spills"
    folder.mkdir()
    with SpillDir(str(folder), max_files=5, file_bytes=100) as spills:
        linked = Path(_cut(spills))
        linked.unlink()
        linked.symlink_to(outside)  # our name now points elsewhere
    assert not os.path.lexists(linked)  # the link itself goes, not what it points to
    assert outside.read_text() == "precious"


def test_stream_does_not_open_a_symlink_put_in_its_place(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("precious")
    folder = tmp_path / "spills"
    folder.mkdir()
    with SpillDir(str(folder), max_files=1, file_bytes=100) as spills:
        target = spills.target()
        assert target is not None and target.path is not None
        os.unlink(target.path)
        os.symlink(outside, target.path)
        stream = _stream(target)
        stream.write(b"0123456789")
        assert stream.close().full is None  # memory only
        assert outside.read_text() == "precious"
        assert spills.target() is not None  # released
    assert outside.read_text() == "precious"


def test_threads_share_one_folder() -> None:
    spills = SpillDir(max_files=8, file_bytes=600)
    errors: list[BaseException] = []

    def work(seed: int) -> None:
        rng = random.Random(seed)
        try:
            for _ in range(50):
                try:
                    target = spills.target()
                except ValueError:
                    return  # closed meanwhile
                stream = _stream(target)
                stream.write(os.urandom(rng.randrange(0, 800)))
                stream.close()
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(seed,)) for seed in range(8)]
    for thread in threads:
        thread.start()
    closer = threading.Thread(target=spills.close)
    closer.start()
    for thread in [*threads, closer]:
        thread.join()
    assert errors == []
    assert not os.path.exists(spills.path)


def test_many_streams_never_exceed_the_budget(tmp_path: Path) -> None:
    rng = random.Random(7)
    file_bytes, max_files = 800, 6
    with SpillDir(str(tmp_path), max_files=max_files, file_bytes=file_bytes) as spills:
        running: list[TextStream] = []
        written = 0
        for _ in range(300):
            if running and (len(running) > 3 or rng.random() < 0.5):
                running.pop(rng.randrange(len(running))).close()
            else:
                running.append(_stream(spills.target()))
            for stream in running:
                chunk = os.urandom(rng.randrange(0, 1500))
                written += len(chunk)
                stream.write(chunk)
            files, size = _disk(str(tmp_path))
            assert files <= max_files and size <= max_files * file_bytes
        for stream in running:
            stream.close()
        assert written > 10 * max_files * file_bytes
