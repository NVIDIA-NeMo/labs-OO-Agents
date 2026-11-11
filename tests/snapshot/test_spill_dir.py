# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SpillDir: shared spill-file budgets by count and bytes, eviction, cleanup and threads."""

from __future__ import annotations

import dataclasses
import logging
import os
import random
import threading
from pathlib import Path

import pytest

from nooa.snapshot import SpillDir, SpillTarget, TextStream


def _stream(target: SpillTarget | None) -> TextStream:
    return TextStream(unit="bytes", head=2, tail=2, spill=target)


def _cut(spills: SpillDir, data: bytes = b"0123456789", max_bytes: int = 10) -> str:
    """A closed, cut stream whose file stays in ``spills``; its path."""
    target = spills.target(max_bytes=max_bytes)
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
    assert spills.max_files == 20 and spills.max_bytes == 64 * 2**20
    assert os.path.isdir(spills.path)
    target = spills.target()
    assert target is not None and target.path is not None
    assert os.path.dirname(target.path) == spills.path
    assert target.max_bytes == 64 * 2**20 // 20  # an equal share
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
        SpillDir(str(tmp_path), max_bytes=-1)
    with SpillDir(str(tmp_path)) as spills, pytest.raises(ValueError):
        spills.target(max_bytes=-1)


def test_target_without_a_folder_is_unchanged() -> None:
    assert SpillTarget() == SpillTarget(path=None, max_bytes=64 * 2**20)
    assert repr(SpillTarget(path="x")) == "SpillTarget(path='x', max_bytes=67108864)"


def test_replace_keeps_the_folder(tmp_path: Path) -> None:
    # Severin's repro (#515): ``replace`` used to drop the private folder link.
    with SpillDir(str(tmp_path), max_files=5, max_bytes=1000) as spills:
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


def test_replace_cannot_raise_the_granted_bytes(tmp_path: Path) -> None:
    with SpillDir(str(tmp_path), max_files=2, max_bytes=10) as spills:
        target = spills.target()
        assert target is not None and target.max_bytes == 5
        stream = _stream(dataclasses.replace(target, max_bytes=1000))
        stream.write(b"0123456789")
        full = stream.close().full
        assert full is not None and full.kept == 5


def test_one_target_opens_one_stream(tmp_path: Path) -> None:
    with SpillDir(str(tmp_path), max_files=2, max_bytes=100) as spills:
        target = spills.target()
        assert target is not None and target.path is not None
        first = _stream(target)
        first.write(b"first")
        second = _stream(target)  # refused: memory only, and the first file is untouched
        second.write(b"second")
        assert second.close().full is None
        assert first.close().full is not None
        assert Path(target.path).read_bytes() == b"first"


def test_open_stream_file_is_never_evicted_by_count(tmp_path: Path) -> None:
    with SpillDir(str(tmp_path), max_files=1, max_bytes=1000) as spills:
        open_target = spills.target()
        assert open_target is not None and open_target.path is not None
        running = _stream(open_target)
        running.write(b"0123456789")
        assert spills.target() is None  # only an open file: nothing to evict
        assert os.path.exists(open_target.path)
        running.write(b"abc")
        full = running.close().full
        assert full is not None and Path(full.path).read_bytes() == b"0123456789abc"
        assert spills.target() is not None  # closed now: evicted for the new one
        assert not os.path.exists(full.path)


def test_budget_by_count_evicts_the_oldest_closed_file(tmp_path: Path) -> None:
    with SpillDir(str(tmp_path), max_files=2, max_bytes=1000) as spills:
        first = _cut(spills)
        second = _cut(spills)
        third = _cut(spills)
        assert not os.path.exists(first)
        assert os.path.exists(second) and os.path.exists(third)
        fourth = _cut(spills)
        assert not os.path.exists(second)
        assert os.path.exists(third) and os.path.exists(fourth)


def test_budget_by_bytes_evicts_until_the_new_one_fits(tmp_path: Path) -> None:
    with SpillDir(str(tmp_path), max_files=10, max_bytes=25) as spills:
        paths = [_cut(spills, max_bytes=10) for _ in range(2)]  # 20 bytes on disk
        target = spills.target(max_bytes=10)  # 5 left: evict the oldest
        assert target is not None and target.max_bytes == 10
        assert not os.path.exists(paths[0]) and os.path.exists(paths[1])


def test_default_target_takes_an_equal_share(tmp_path: Path) -> None:
    with SpillDir(str(tmp_path), max_files=10, max_bytes=100) as spills:
        kept = _cut(spills)
        target = spills.target()  # asks for 100 // 10: no closed file has to go
        assert target is not None and target.max_bytes == 10
        assert os.path.exists(kept)
        targets = [spills.target() for _ in range(8)]  # 10 files in all, none evicted
        assert all(t is not None and t.max_bytes == 10 for t in targets)
        assert os.path.exists(kept)


def test_closed_file_counts_its_size_on_disk(tmp_path: Path) -> None:
    with SpillDir(str(tmp_path), max_files=10, max_bytes=100) as spills:
        kept = _cut(spills, data=b"0123456789", max_bytes=100)  # 10 bytes, not 100
        target = spills.target(max_bytes=90)  # fits beside the 10 bytes kept
        assert target is not None and target.max_bytes == 90
        assert os.path.exists(kept)


def test_open_stream_file_is_never_evicted(tmp_path: Path) -> None:
    with SpillDir(str(tmp_path), max_files=2, max_bytes=100) as spills:
        open_target = spills.target(max_bytes=30)
        assert open_target is not None and open_target.path is not None
        running = _stream(open_target)
        running.write(b"0123456789")
        closed = _cut(spills, max_bytes=30)
        target = spills.target(max_bytes=100)  # count full: only the closed one can go
        assert target is not None
        assert not os.path.exists(closed) and os.path.exists(open_target.path)
        assert target.max_bytes == 70  # the open stream still reserves 30
        assert spills.target() is None  # two open files: nothing to evict
        assert os.path.exists(open_target.path)
        running.write(b"abc")
        assert running.close().full is not None
        assert open_target.path is not None
        assert Path(open_target.path).read_bytes() == b"0123456789abc"


def test_short_budget_caps_then_keeps_memory_only(tmp_path: Path) -> None:
    with SpillDir(str(tmp_path), max_files=5, max_bytes=10) as spills:
        first = spills.target(max_bytes=7)
        assert first is not None and first.max_bytes == 7
        capped = spills.target(max_bytes=7)
        assert capped is not None and capped.max_bytes == 3
        assert spills.target() is None  # no byte left
        stream = _stream(spills.target())
        stream.write(b"0123456789")
        text = stream.close()
        assert text.full is None and text.omitted == 6
        assert _disk(str(tmp_path))[0] == 2


def test_zero_budget_creates_no_file(tmp_path: Path) -> None:
    with SpillDir(str(tmp_path), max_files=0) as spills:
        assert spills.target() is None
    with SpillDir(str(tmp_path), max_bytes=0) as spills:
        assert spills.target() is None
    assert list(tmp_path.iterdir()) == []


def test_complete_output_deleted_by_its_stream_is_released(tmp_path: Path) -> None:
    with SpillDir(str(tmp_path), max_files=1, max_bytes=10) as spills:
        target = spills.target()
        assert target is not None and target.path is not None
        stream = TextStream(unit="bytes", head=10, tail=10, spill=target)
        stream.write(b"short")
        assert stream.close().full is None
        assert not os.path.exists(target.path)
        again = spills.target()
        assert again is not None and again.max_bytes == 10  # count and bytes free again


def test_close_deletes_everything_and_is_idempotent() -> None:
    spills = SpillDir(max_files=5, max_bytes=100)
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


def test_given_folder_keeps_foreign_files(tmp_path: Path) -> None:
    foreign = tmp_path / "keep.txt"
    foreign.write_text("mine")
    with SpillDir(str(tmp_path), max_files=3, max_bytes=100) as spills:
        _cut(spills)
        _cut(spills)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.txt"]
    assert foreign.read_text() == "mine"


def test_replaced_file_or_symlink_is_left_alone(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("precious")
    folder = tmp_path / "spills"
    folder.mkdir()
    with caplog.at_level(logging.WARNING, logger="nooa.snapshot._stream"):
        with SpillDir(str(folder), max_files=5, max_bytes=100) as spills:
            replaced = Path(_cut(spills))
            linked = Path(_cut(spills))
            other = folder / "other.tmp"
            other.write_text("someone else")  # a new inode, before the old one is freed
            os.replace(other, replaced)
            linked.unlink()
            linked.symlink_to(outside)
    assert replaced.read_text() == "someone else"
    assert linked.is_symlink() and outside.read_text() == "precious"
    assert sum("was replaced" in r.getMessage() for r in caplog.records) == 2


def test_stream_does_not_open_a_symlink_put_in_its_place(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("precious")
    folder = tmp_path / "spills"
    folder.mkdir()
    with SpillDir(str(folder), max_files=5, max_bytes=100) as spills:
        target = spills.target()
        assert target is not None and target.path is not None
        os.unlink(target.path)
        os.symlink(outside, target.path)
        stream = _stream(target)
        stream.write(b"0123456789")
        assert stream.close().full is None  # memory only
        assert outside.read_text() == "precious"
        again = spills.target(max_bytes=100)
        assert again is not None and again.max_bytes == 100  # released


def test_threads_share_one_folder() -> None:
    spills = SpillDir(max_files=8, max_bytes=4000)
    errors: list[BaseException] = []

    def work(seed: int) -> None:
        rng = random.Random(seed)
        try:
            for _ in range(50):
                try:
                    target = spills.target(max_bytes=rng.randrange(1, 600))
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


def test_many_streams_never_exceed_the_byte_total(tmp_path: Path) -> None:
    rng = random.Random(7)
    max_bytes, max_files = 5000, 6
    with SpillDir(str(tmp_path), max_files=max_files, max_bytes=max_bytes) as spills:
        running: list[TextStream] = []
        written = 0
        for _ in range(300):
            if running and (len(running) > 3 or rng.random() < 0.5):
                running.pop(rng.randrange(len(running))).close()
                files, size = _disk(str(tmp_path))
                assert size <= max_bytes and files <= max_files
            else:
                ask = rng.choice([None, rng.randrange(0, 3000)])
                stream = _stream(spills.target(max_bytes=ask))
                running.append(stream)
            for stream in running:
                chunk = os.urandom(rng.randrange(0, 1500))
                written += len(chunk)
                stream.write(chunk)
            files, size = _disk(str(tmp_path))
            assert size <= max_bytes and files <= max_files
        for stream in running:
            stream.close()
            assert _disk(str(tmp_path))[1] <= max_bytes
        assert written > 10 * max_bytes
