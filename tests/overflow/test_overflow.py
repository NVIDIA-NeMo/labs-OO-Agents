# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nooa.overflow: caps, eviction, cleanup, file safety, threads and the explicit lifetime.

Filesystem behaviour only: these tests use ``OverflowDir`` and the
``OverflowWriter`` its ``open()`` returns, nothing else from ``nooa``.
"""

from __future__ import annotations

import gc
import os
import random
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
from pathlib import Path

import pytest

from nooa.overflow import OverflowDir, OverflowWriter


@pytest.fixture
def in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Private folders go in ``tmp_path`` (the system temporary directory, for these tests)."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


def _open(folder: OverflowDir, max_bytes: int | None = None) -> OverflowWriter:
    writer = folder.open(max_bytes)
    assert writer is not None
    return writer


def _closed(folder: OverflowDir, data: bytes = b"0123456789") -> str:
    """A written, closed (kept) file in ``folder``; its path."""
    writer = _open(folder)
    writer.write(data)
    writer.finish(delete=False)
    return writer.path


def _disk(folder: str) -> tuple[int, int]:
    """(files, bytes) of the regular files directly in ``folder``."""
    entries = [e for e in os.scandir(folder) if e.is_file(follow_symlinks=False)]
    return len(entries), sum(e.stat(follow_symlinks=False).st_size for e in entries)


def test_the_writer_type_is_public() -> None:
    import nooa.overflow

    assert nooa.overflow.__all__ == ["OverflowDir", "OverflowWriter"]
    with OverflowDir() as folder:
        writer = _open(folder)
        assert isinstance(writer, OverflowWriter)
        writer.finish(delete=True)


def test_the_package_imports_nothing_else_from_nooa() -> None:
    import nooa.overflow

    for path in Path(nooa.overflow.__file__).parent.glob("*.py"):
        lines = [line.strip() for line in path.read_text().splitlines()]
        imports = [line for line in lines if line.startswith(("import nooa", "from nooa"))]
        assert all(line.startswith("from nooa.overflow") for line in imports), (path, imports)


# --- defaults, arguments ----------------------------------------------------------


def test_defaults_and_private_folder(in_tmp: Path) -> None:
    folder = OverflowDir()
    assert folder.max_files == 20 and folder.file_bytes == 64 * 2**20
    assert os.path.dirname(folder.path) == str(in_tmp)  # the system temporary directory
    assert os.path.basename(folder.path).startswith("nooa-overflow-")
    assert stat.S_IMODE(os.stat(folder.path).st_mode) == 0o700  # tempfile.mkdtemp
    writer = _open(folder)
    assert os.path.dirname(writer.path) == folder.path
    assert os.path.basename(writer.path).startswith("nooa-output-")
    assert stat.S_IMODE(os.stat(writer.path).st_mode) == 0o600
    assert writer.max_bytes == 64 * 2**20  # each file gets file_bytes
    (Path(folder.path) / "stray.txt").write_text("x")  # anything in a private folder goes
    folder.close()
    assert not os.path.exists(folder.path)
    assert list(in_tmp.iterdir()) == []


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        OverflowDir(max_files=-1)
    with pytest.raises(ValueError):
        OverflowDir(file_bytes=-1)
    with OverflowDir() as folder, pytest.raises(ValueError):
        folder.open(max_bytes=-1)


# --- caps ----------------------------------------------------------------------------


def test_each_file_is_capped_at_file_bytes() -> None:
    with OverflowDir(max_files=3, file_bytes=4) as folder:
        writers = [_open(folder) for _ in range(3)]
        for writer in writers:
            assert writer.max_bytes == 4
            assert writer.write(b"01") == 2
            assert writer.write(b"23456789") == 2  # what fits
            assert writer.write(b"x") == 0  # full
            assert writer.written == 4
            writer.finish(delete=False)
        assert [Path(w.path).read_bytes() for w in writers] == [b"0123"] * 3
        assert _disk(folder.path) == (3, 12)  # max_files * file_bytes at most


def test_file_count_cap() -> None:
    with OverflowDir(max_files=3, file_bytes=10) as folder:
        held = [folder.open() for _ in range(3)]
        assert all(writer is not None for writer in held)
        assert folder.open() is None  # three in use: nothing to evict
        assert _disk(folder.path)[0] == 3


def test_zero_budget_creates_no_file() -> None:
    with OverflowDir(max_files=0) as folder:
        assert folder.open() is None
        assert os.listdir(folder.path) == []
    with OverflowDir(file_bytes=0) as folder:
        assert folder.open() is None
        assert os.listdir(folder.path) == []


def test_open_takes_a_smaller_cap_but_never_a_larger_one() -> None:
    # A folder shared by writers with different limits: open(max_bytes=...) caps
    # this one file, still the folder's (closing the folder deletes it).
    with OverflowDir(max_files=2, file_bytes=5) as folder:
        small, large = _open(folder, 3), _open(folder, 100)
        assert (small.max_bytes, large.max_bytes) == (3, 5)
        assert small.write(b"0123456789") == 3
        assert large.write(b"0123456789") == 5
        small.finish(delete=False)
        large.finish(delete=True)
        assert Path(small.path).read_bytes() == b"012"
    assert not os.path.exists(small.path)


# --- eviction --------------------------------------------------------------------------


def test_eviction_deletes_the_oldest_closed_file_first() -> None:
    with OverflowDir(max_files=2, file_bytes=1000) as folder:
        first = _closed(folder)
        second = _closed(folder)
        third = _closed(folder)
        assert not os.path.exists(first)
        assert os.path.exists(second) and os.path.exists(third)
        fourth = _closed(folder)
        assert not os.path.exists(second)
        assert os.path.exists(third) and os.path.exists(fourth)


def test_an_evicted_path_raises_file_not_found() -> None:
    with OverflowDir(max_files=1, file_bytes=10) as folder:
        first = _closed(folder)
        _closed(folder)
        with pytest.raises(FileNotFoundError):  # how a reader tells the full output is gone
            Path(first).open("rb").close()


def test_an_open_file_is_never_evicted() -> None:
    with OverflowDir(max_files=2, file_bytes=100) as folder:
        running = _open(folder)
        running.write(b"0123456789")
        closed = _closed(folder)
        other = folder.open()  # count full: only the closed one can go
        assert other is not None
        assert not os.path.exists(closed) and os.path.exists(running.path)
        assert folder.open() is None  # two open files: nothing to evict
        assert os.path.exists(running.path)
        running.write(b"abc")
        running.finish(delete=False)
        assert Path(running.path).read_bytes() == b"0123456789abc"
        other.finish(delete=True)  # complete output: deleted, its place is free
        assert folder.open() is not None  # and the closed file is evicted only now
        assert os.path.exists(running.path)
        assert folder.open() is not None
        assert not os.path.exists(running.path)


def test_a_deleted_file_frees_its_place() -> None:
    with OverflowDir(max_files=1, file_bytes=10) as folder:
        writer = _open(folder)
        writer.write(b"short")
        writer.finish(delete=True)
        assert not os.path.exists(writer.path)
        assert folder.open() is not None  # its place is free again


def test_finish_is_idempotent() -> None:
    with OverflowDir(max_files=2, file_bytes=100) as folder:
        writer = _open(folder)
        writer.write(b"first")
        assert writer.finish(delete=False) is True
        assert writer.finish(delete=False) is True  # the same answer
        assert writer.closed and writer.write(b"more") == 0
        assert Path(writer.path).read_bytes() == b"first"


def test_many_writers_never_exceed_the_budget() -> None:
    rng = random.Random(7)
    file_bytes, max_files = 800, 6
    with OverflowDir(max_files=max_files, file_bytes=file_bytes) as folder:
        running: list[OverflowWriter] = []
        written = 0
        for _ in range(300):
            if running and (len(running) > 3 or rng.random() < 0.5):
                running.pop(rng.randrange(len(running))).finish(delete=rng.random() < 0.2)
            else:
                writer = folder.open()
                if writer is not None:
                    running.append(writer)
            for writer in running:
                chunk = os.urandom(rng.randrange(0, 1500))
                written += len(chunk)
                writer.write(chunk)  # unbuffered: on disk at once
            files, size = _disk(folder.path)
            assert files <= max_files and size <= max_files * file_bytes
        for writer in running:
            writer.finish(delete=False)
        assert written > 10 * max_files * file_bytes


# --- close -------------------------------------------------------------------------------


def test_close_deletes_everything_and_is_idempotent() -> None:
    folder = OverflowDir(max_files=5, file_bytes=100)
    _closed(folder)
    running = _open(folder)
    running.write(b"0123456789")
    assert not folder.closed
    folder.close()
    assert folder.closed and not os.path.exists(folder.path)
    folder.close()
    folder.close()
    assert folder.open() is None  # closed: memory only
    running.write(b"more")  # the writer keeps going without its file's place
    running.finish(delete=False)


def test_close_leaves_the_rest_of_the_temporary_directory_alone(in_tmp: Path) -> None:
    foreign = in_tmp / "keep.txt"
    foreign.write_text("mine")
    (in_tmp / "sub").mkdir()
    with OverflowDir(max_files=3, file_bytes=100) as folder:
        _closed(folder)
        _closed(folder)
    assert sorted(p.name for p in in_tmp.iterdir()) == ["keep.txt", "sub"]
    assert foreign.read_text() == "mine"


def test_close_removes_a_symlink_not_its_target(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("precious")
    with OverflowDir(max_files=5, file_bytes=100) as folder:
        linked = Path(_closed(folder))
        linked.unlink()
        linked.symlink_to(outside)  # our name now points elsewhere
    assert not os.path.lexists(linked)  # the link itself goes, not what it points to
    assert outside.read_text() == "precious"


def test_close_of_a_private_folder_never_follows_a_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious.txt").write_text("precious")
    folder = OverflowDir(max_files=5, file_bytes=100)
    linked = Path(_closed(folder))
    linked.unlink()
    linked.symlink_to(outside / "precious.txt")
    (Path(folder.path) / "dir-link").symlink_to(outside, target_is_directory=True)
    folder.close()
    assert not os.path.lexists(folder.path)
    assert (outside / "precious.txt").read_text() == "precious"


def test_a_writer_writes_the_file_its_folder_created(tmp_path: Path) -> None:
    # Never reopened by name: a symlink put in its place is neither written nor followed.
    outside = tmp_path / "outside.txt"
    outside.write_text("precious")
    with OverflowDir(max_files=1, file_bytes=100) as folder:
        writer = _open(folder)
        os.unlink(writer.path)
        os.symlink(outside, writer.path)
        assert writer.write(b"output") == 6
        writer.finish(delete=False)
        assert outside.read_text() == "precious"
    assert outside.read_text() == "precious"


# --- threads -------------------------------------------------------------------------------


def test_threads_share_one_folder() -> None:
    folder = OverflowDir(max_files=8, file_bytes=600)
    errors: list[BaseException] = []

    def work(seed: int) -> None:
        rng = random.Random(seed)
        try:
            for _ in range(50):
                writer = folder.open()  # None once closed
                if writer is not None:
                    writer.write(os.urandom(rng.randrange(0, 800)))
                    writer.finish(delete=rng.random() < 0.3)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(seed,)) for seed in range(8)]
    for thread in threads:
        thread.start()
    closer = threading.Thread(target=folder.close)
    closer.start()
    for thread in [*threads, closer]:
        thread.join()
    assert errors == []
    assert not os.path.exists(folder.path)


def test_threads_never_exceed_the_budget() -> None:
    max_files, file_bytes = 4, 100
    folder = OverflowDir(max_files=max_files, file_bytes=file_bytes)
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def work(seed: int) -> None:
        rng = random.Random(seed)
        try:
            barrier.wait()
            for _ in range(100):
                writer = folder.open()
                if writer is None:
                    continue
                writer.write(os.urandom(rng.randrange(0, 300)))
                writer.finish(delete=False)
                # create-before-evict: one more file at most while open() runs
                assert len(os.listdir(folder.path)) <= max_files + 1
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(seed,)) for seed in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    files, size = _disk(folder.path)
    assert errors == [] and files <= max_files and size <= max_files * file_bytes
    folder.close()
    assert not os.path.exists(folder.path)


# --- lifetime: garbage collection, exit, crash ------------------------------------------------


def test_no_deletion_on_garbage_collection() -> None:
    folder = OverflowDir(max_files=2, file_bytes=10)
    path = folder.path
    kept = _closed(folder)
    running = _open(folder)
    running.write(b"abc")
    del folder
    gc.collect()
    assert os.path.exists(path) and os.path.exists(kept)  # still there: no GC deletion
    running.finish(delete=False)
    del running
    gc.collect()
    assert os.path.exists(kept)
    # Only close (or exit) removes them; the cleanup is still registered.
    from nooa.overflow import _overflow

    (cleanup,) = [c for c in _overflow._live if c.path == path]
    with cleanup.lock:
        cleanup.run()
    assert not os.path.exists(path)


def _child(code: str, tmp_path: Path) -> subprocess.Popen[str]:
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(code))
    temp = tmp_path / "temp"
    temp.mkdir()
    return subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "TMPDIR": str(temp)},
    )


_CHILD = """
    import sys
    from nooa.overflow import OverflowDir

    folders = [OverflowDir(max_files=3, file_bytes=10) for _ in range(2)]
    paths = [folder.path for folder in folders]
    for folder in folders:
        writer = folder.open()
        writer.write(b"0123456789abc")
        writer.finish(delete=False)
        paths.append(writer.path)
    dropped = OverflowDir(max_files=1, file_bytes=10)
    running = dropped.open()  # never finished
    running.write(b"abc")
    paths.append(dropped.path)
    del dropped
    print(*paths, flush=True)
    {end}
"""


def test_exit_cleanup_removes_files_never_closed(tmp_path: Path) -> None:
    child = _child(_CHILD.format(end="sys.exit(0)  # no close()"), tmp_path)
    out, _ = child.communicate(timeout=60)
    assert child.returncode == 0
    paths = out.split()
    assert len(paths) == 5 and all(paths)
    assert not any(os.path.lexists(path) for path in paths)
    assert os.listdir(tmp_path / "temp") == []


def test_a_killed_process_leaves_its_files(tmp_path: Path) -> None:
    # Known limit: SIGKILL runs no atexit, so the files remain (documented in nooa.overflow).
    child = _child(_CHILD.format(end="import time; time.sleep(60)"), tmp_path)
    assert child.stdout is not None
    paths = child.stdout.readline().split()
    child.send_signal(signal.SIGKILL)
    child.wait(timeout=60)
    child.stdout.close()
    assert child.returncode == -signal.SIGKILL
    first, _second, first_file, _second_file, _dropped = paths
    assert all(os.path.exists(path) for path in paths)
    assert os.path.dirname(first_file) == first
    assert Path(first_file).read_bytes() == b"0123456789"
