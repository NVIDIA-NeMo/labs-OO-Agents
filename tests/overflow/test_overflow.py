# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nooa.overflow: caps, eviction, cleanup, file safety, threads and the explicit lifetime.

Filesystem behaviour only: these tests use ``OverflowDir``, ``OverflowFile``
and the writer ``OverflowFile.open()`` returns, nothing else from ``nooa``.
"""

from __future__ import annotations

import dataclasses
import gc
import logging
import os
import random
import signal
import stat
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from nooa.overflow import OverflowDir, OverflowFile, OverflowWriter


def _open(file: OverflowFile | None) -> OverflowWriter:
    assert file is not None
    writer = file.open()
    assert writer is not None
    return writer


def _closed(folder: OverflowDir, data: bytes = b"0123456789") -> str:
    """A written, closed (kept) file in ``folder``; its path."""
    writer = _open(folder.file())
    writer.write(data)
    writer.finish(delete=False)
    return writer.path


def _disk(folder: str) -> tuple[int, int]:
    """(files, bytes) of the regular files directly in ``folder``."""
    entries = [e for e in os.scandir(folder) if e.is_file(follow_symlinks=False)]
    return len(entries), sum(e.stat(follow_symlinks=False).st_size for e in entries)


def test_the_writer_type_is_public() -> None:
    import nooa.overflow

    assert nooa.overflow.__all__ == ["OverflowDir", "OverflowFile", "OverflowWriter"]
    writer = _open(OverflowFile())
    assert isinstance(writer, OverflowWriter)
    writer.finish(delete=True)


def test_the_package_imports_nothing_else_from_nooa() -> None:
    import nooa.overflow

    for path in Path(nooa.overflow.__file__).parent.glob("*.py"):
        lines = [line.strip() for line in path.read_text().splitlines()]
        imports = [line for line in lines if line.startswith(("import nooa", "from nooa"))]
        assert all(line.startswith("from nooa.overflow") for line in imports), (path, imports)


# --- defaults, arguments ----------------------------------------------------------


def test_defaults_and_private_folder() -> None:
    folder = OverflowDir()
    assert folder.max_files == 20 and folder.file_bytes == 64 * 2**20
    assert os.path.basename(folder.path).startswith("nooa-overflow-")
    assert stat.S_IMODE(os.stat(folder.path).st_mode) == 0o700  # tempfile.mkdtemp
    file = folder.file()
    assert file is not None and file.path is not None
    assert os.path.dirname(file.path) == folder.path
    assert os.path.basename(file.path).startswith("nooa-output-")
    assert stat.S_IMODE(os.stat(file.path).st_mode) == 0o600  # tempfile.mkstemp
    assert file.max_bytes == 64 * 2**20  # each file gets file_bytes
    (Path(folder.path) / "stray.txt").write_text("x")  # anything in a private folder goes
    folder.close()
    assert not os.path.exists(folder.path)


def test_invalid_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        OverflowDir(str(tmp_path / "missing"))
    (tmp_path / "file").write_text("x")
    with pytest.raises(ValueError):
        OverflowDir(str(tmp_path / "file"))
    with pytest.raises(ValueError):
        OverflowDir(str(tmp_path), max_files=-1)
    with pytest.raises(ValueError):
        OverflowDir(str(tmp_path), file_bytes=-1)
    with pytest.raises(ValueError):
        OverflowFile(max_bytes=-1).open()


def test_file_without_a_folder() -> None:
    assert OverflowFile() == OverflowFile(path=None, max_bytes=64 * 2**20)
    assert repr(OverflowFile(path="x")) == "OverflowFile(path='x', max_bytes=67108864)"


# --- caps ----------------------------------------------------------------------------


def test_each_file_is_capped_at_file_bytes(tmp_path: Path) -> None:
    with OverflowDir(str(tmp_path), max_files=3, file_bytes=4) as folder:
        writers = [_open(folder.file()) for _ in range(3)]
        for writer in writers:
            assert writer.max_bytes == 4
            assert writer.write(b"01") == 2
            assert writer.write(b"23456789") == 2  # what fits
            assert writer.write(b"x") == 0  # full
            assert writer.written == 4
            writer.finish(delete=False)
        assert [Path(w.path).read_bytes() for w in writers] == [b"0123"] * 3
        assert _disk(str(tmp_path)) == (3, 12)  # max_files * file_bytes at most


def test_file_count_cap(tmp_path: Path) -> None:
    with OverflowDir(str(tmp_path), max_files=3, file_bytes=10) as folder:
        held = [folder.file() for _ in range(3)]
        assert all(file is not None for file in held)
        assert folder.file() is None  # three in use: nothing to evict
        assert _disk(str(tmp_path))[0] == 3


def test_zero_budget_creates_no_file(tmp_path: Path) -> None:
    with OverflowDir(str(tmp_path), max_files=0) as folder:
        assert folder.file() is None
    with OverflowDir(str(tmp_path), file_bytes=0) as folder:
        assert folder.file() is None
    assert list(tmp_path.iterdir()) == []


def test_replace_keeps_the_folder(tmp_path: Path) -> None:
    # ``dataclasses.replace`` of a folder's file keeps its link to the folder.
    with OverflowDir(str(tmp_path), max_files=5, file_bytes=1000) as folder:
        file = folder.file()
        assert file is not None and file.path is not None
        kept = dataclasses.replace(file, max_bytes=3)
        assert kept._owner is folder and kept.max_bytes == 3
        assert kept == dataclasses.replace(kept, _owner=None)  # not compared
        assert "_owner" not in repr(kept)
        writer = _open(kept)
        writer.write(b"hello")
        writer.finish(delete=False)
        assert Path(file.path).read_bytes() == b"hel"  # kept, and still the folder's
    assert not os.path.exists(file.path)  # deleted by the folder's close()


def test_replace_cannot_raise_the_file_cap(tmp_path: Path) -> None:
    with OverflowDir(str(tmp_path), max_files=2, file_bytes=5) as folder:
        file = folder.file()
        assert file is not None and file.max_bytes == 5
        writer = _open(dataclasses.replace(file, max_bytes=1000))
        assert writer.max_bytes == 5 and writer.write(b"0123456789") == 5


# --- eviction --------------------------------------------------------------------------


def test_eviction_deletes_the_oldest_closed_file_first(tmp_path: Path) -> None:
    with OverflowDir(str(tmp_path), max_files=2, file_bytes=1000) as folder:
        first = _closed(folder)
        second = _closed(folder)
        third = _closed(folder)
        assert not os.path.exists(first)
        assert os.path.exists(second) and os.path.exists(third)
        fourth = _closed(folder)
        assert not os.path.exists(second)
        assert os.path.exists(third) and os.path.exists(fourth)


def test_an_evicted_path_raises_file_not_found(tmp_path: Path) -> None:
    with OverflowDir(str(tmp_path), max_files=1, file_bytes=10) as folder:
        first = _closed(folder)
        _closed(folder)
        with pytest.raises(FileNotFoundError):  # how a reader tells the full output is gone
            Path(first).open("rb").close()


def test_an_open_file_is_never_evicted(tmp_path: Path) -> None:
    with OverflowDir(str(tmp_path), max_files=2, file_bytes=100) as folder:
        running = _open(folder.file())
        running.write(b"0123456789")
        closed = _closed(folder)
        file = folder.file()  # count full: only the closed one can go
        assert file is not None
        assert not os.path.exists(closed) and os.path.exists(running.path)
        assert folder.file() is None  # two open files: nothing to evict
        assert os.path.exists(running.path)
        running.write(b"abc")
        running.finish(delete=False)
        assert Path(running.path).read_bytes() == b"0123456789abc"
        _open(file).finish(delete=True)  # complete output: deleted, its place is free
        assert folder.file() is not None  # and the closed file is evicted only now
        assert os.path.exists(running.path)
        assert folder.file() is not None
        assert not os.path.exists(running.path)


def test_a_handed_out_file_is_never_evicted(tmp_path: Path) -> None:
    with OverflowDir(str(tmp_path), max_files=1, file_bytes=10) as folder:
        handed = folder.file()
        assert handed is not None and handed.path is not None
        assert folder.file() is None
        assert os.path.exists(handed.path)


def test_a_deleted_file_frees_its_place(tmp_path: Path) -> None:
    with OverflowDir(str(tmp_path), max_files=1, file_bytes=10) as folder:
        writer = _open(folder.file())
        writer.write(b"short")
        writer.finish(delete=True)
        assert not os.path.exists(writer.path)
        assert folder.file() is not None  # its place is free again


def test_one_file_opens_one_writer(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with OverflowDir(str(tmp_path), max_files=2, file_bytes=100) as folder:
        file = folder.file()
        assert file is not None and file.path is not None
        first = _open(file)
        first.write(b"first")
        with caplog.at_level(logging.WARNING, logger="nooa.overflow._overflow"):
            assert file.open() is None  # refused, and the first file is untouched
        # A reused file is a usage error, not a file error: no traceback.
        assert [r.getMessage() for r in caplog.records] == [
            f"overflow file {file.path!r} is not open to a new writer; keeping memory only"
        ]
        assert all(r.exc_info is None for r in caplog.records)
        first.finish(delete=False)
        first.finish(delete=True)  # idempotent: already handed back
        assert Path(file.path).read_bytes() == b"first"
        assert file.open() is None  # closed now: still refused


def test_many_writers_never_exceed_the_budget(tmp_path: Path) -> None:
    rng = random.Random(7)
    file_bytes, max_files = 800, 6
    with OverflowDir(str(tmp_path), max_files=max_files, file_bytes=file_bytes) as folder:
        running: list[OverflowWriter] = []
        written = 0
        for _ in range(300):
            if running and (len(running) > 3 or rng.random() < 0.5):
                running.pop(rng.randrange(len(running))).finish(delete=rng.random() < 0.2)
            else:
                file = folder.file()
                if file is not None:
                    running.append(_open(file))
            for writer in running:
                chunk = os.urandom(rng.randrange(0, 1500))
                written += len(chunk)
                writer.write(chunk)
                writer.file.flush()  # type: ignore[union-attr]
            files, size = _disk(str(tmp_path))
            assert files <= max_files and size <= max_files * file_bytes
        for writer in running:
            writer.finish(delete=False)
        assert written > 10 * max_files * file_bytes


# --- close -------------------------------------------------------------------------------


def test_close_deletes_everything_and_is_idempotent() -> None:
    folder = OverflowDir(max_files=5, file_bytes=100)
    _closed(folder)
    running = _open(folder.file())
    running.write(b"0123456789")
    folder.file()  # handed out, never used
    assert not folder.closed
    folder.close()
    assert folder.closed and not os.path.exists(folder.path)
    folder.close()
    folder.close()
    with pytest.raises(ValueError):
        folder.file()
    running.write(b"more")  # the writer keeps going without its file's place
    running.finish(delete=False)


def test_a_file_after_its_folder_closed_is_refused(tmp_path: Path) -> None:
    folder = OverflowDir(str(tmp_path), max_files=2, file_bytes=10)
    file = folder.file()
    folder.close()
    assert file is not None and file.open() is None
    assert list(tmp_path.iterdir()) == []


def test_close_of_a_given_folder_removes_only_its_own_files(tmp_path: Path) -> None:
    foreign = tmp_path / "keep.txt"
    foreign.write_text("mine")
    (tmp_path / "sub").mkdir()
    with OverflowDir(str(tmp_path), max_files=3, file_bytes=100) as folder:
        _closed(folder)
        _closed(folder)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.txt", "sub"]
    assert foreign.read_text() == "mine"


def test_close_of_a_given_folder_never_follows_a_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("precious")
    given = tmp_path / "given"
    given.mkdir()
    with OverflowDir(str(given), max_files=5, file_bytes=100) as folder:
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


def test_a_writer_does_not_open_a_symlink_put_in_its_place(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("precious")
    given = tmp_path / "given"
    given.mkdir()
    with OverflowDir(str(given), max_files=1, file_bytes=100) as folder:
        file = folder.file()
        assert file is not None and file.path is not None
        os.unlink(file.path)
        os.symlink(outside, file.path)
        assert file.open() is None  # memory only
        assert outside.read_text() == "precious"
        assert folder.file() is not None  # released
    assert outside.read_text() == "precious"


# --- standalone files ---------------------------------------------------------------------


@pytest.mark.parametrize("named", [True, False])
def test_a_standalone_file_is_private(tmp_path: Path, named: bool) -> None:
    file = OverflowFile(path=str(tmp_path / "out.txt")) if named else OverflowFile()
    writer = _open(file)
    try:
        assert stat.S_IMODE(os.stat(writer.path).st_mode) == 0o600
    finally:
        writer.finish(delete=True)
    assert not os.path.exists(writer.path)


def test_an_existing_path_is_refused_not_truncated(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    path.write_text("kept")
    with pytest.raises(FileExistsError):
        OverflowFile(path=str(path)).open()
    assert path.read_text() == "kept"
    link = tmp_path / "link.txt"
    link.symlink_to(path)
    with pytest.raises(FileExistsError):
        OverflowFile(path=str(link)).open()
    assert path.read_text() == "kept" and link.is_symlink()


# --- threads -------------------------------------------------------------------------------


def test_threads_share_one_folder() -> None:
    folder = OverflowDir(max_files=8, file_bytes=600)
    errors: list[BaseException] = []

    def work(seed: int) -> None:
        rng = random.Random(seed)
        try:
            for _ in range(50):
                try:
                    file = folder.file()
                except ValueError:
                    return  # closed meanwhile
                writer = file.open() if file is not None else None
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


def test_threads_never_exceed_the_budget(tmp_path: Path) -> None:
    max_files, file_bytes = 4, 100
    folder = OverflowDir(str(tmp_path), max_files=max_files, file_bytes=file_bytes)
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def work(seed: int) -> None:
        rng = random.Random(seed)
        try:
            barrier.wait()
            for _ in range(100):
                file = folder.file()
                if file is None:
                    continue
                writer = file.open()
                assert writer is not None
                writer.write(os.urandom(rng.randrange(0, 300)))
                writer.finish(delete=False)
                assert len(os.listdir(tmp_path)) <= max_files
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(seed,)) for seed in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    files, size = _disk(str(tmp_path))
    assert errors == [] and files <= max_files and size <= max_files * file_bytes
    folder.close()
    assert list(tmp_path.iterdir()) == []


# --- lifetime: garbage collection, exit, crash ------------------------------------------------


def test_no_deletion_on_garbage_collection() -> None:
    folder = OverflowDir(max_files=2, file_bytes=10)
    path = folder.path
    kept = _closed(folder)
    running = _open(folder.file())
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
    return subprocess.Popen(
        [sys.executable, str(script)], stdout=subprocess.PIPE, text=True, cwd=tmp_path
    )


_CHILD = """
    import os, sys
    from nooa.overflow import OverflowDir

    given = OverflowDir(os.path.join(os.getcwd(), "given"), max_files=3, file_bytes=10)
    private = OverflowDir(max_files=3, file_bytes=10)
    paths = []
    for folder in (given, private):
        writer = folder.file().open()
        writer.write(b"0123456789abc")
        writer.finish(delete=False)
        paths.append(writer.path)
    dropped = OverflowDir(max_files=1, file_bytes=10)
    dropped.file()
    paths.append(dropped.path)
    del dropped
    print(private.path, *paths, flush=True)
    {end}
"""


def test_exit_cleanup_removes_files_never_closed(tmp_path: Path) -> None:
    (tmp_path / "given").mkdir()
    (tmp_path / "given" / "foreign.txt").write_text("mine")
    child = _child(_CHILD.format(end="sys.exit(0)  # no close()"), tmp_path)
    out, _ = child.communicate(timeout=60)
    assert child.returncode == 0
    paths = out.split()
    assert len(paths) == 4 and all(paths)
    assert not any(os.path.lexists(path) for path in paths)
    assert os.listdir(tmp_path / "given") == ["foreign.txt"]


def test_a_killed_process_leaves_its_files(tmp_path: Path) -> None:
    # Known limit: SIGKILL runs no atexit, so the files remain (documented in nooa.overflow).
    (tmp_path / "given").mkdir()
    child = _child(_CHILD.format(end="import time; time.sleep(60)"), tmp_path)
    assert child.stdout is not None
    paths = child.stdout.readline().split()
    child.send_signal(signal.SIGKILL)
    child.wait(timeout=60)
    child.stdout.close()
    assert child.returncode == -signal.SIGKILL
    private, given_file, private_file, dropped = paths
    try:
        assert all(os.path.exists(path) for path in paths)
        assert Path(given_file).read_bytes() == b"0123456789"
    finally:
        import shutil

        shutil.rmtree(private, ignore_errors=True)
        shutil.rmtree(dropped, ignore_errors=True)
