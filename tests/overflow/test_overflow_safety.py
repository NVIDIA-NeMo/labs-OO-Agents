# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nooa.overflow: forks, replaced paths, failed deletes and failed closes.

Each test replays a scenario where the files of a folder were deleted that
were not its own, or its own were forgotten: a forked child exiting, a file
or folder replaced at its path, a delete or a close failing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from nooa.overflow import OverflowDir, OverflowFile

needs_permissions = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores directory permissions"
)


def _kept(folder: OverflowDir, data: bytes = b"ABCD") -> Path:
    """A written, closed (kept) file in ``folder``; its path."""
    file = folder.file()
    assert file is not None
    writer = file.open()
    assert writer is not None
    writer.write(data)
    writer.finish(delete=False)
    return Path(writer.path)


def _run(code: str, *args: str) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.split()


# --- fork: cleanup runs only in the process that created the folder -------------------


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_a_forked_child_exiting_leaves_the_parents_files(tmp_path: Path) -> None:
    code = """
        import os, sys
        from pathlib import Path
        from nooa.overflow import OverflowDir

        given = OverflowDir(sys.argv[1])
        private = OverflowDir()
        kept = []
        for folder in (given, private):
            writer = folder.file().open()
            writer.write(b"parent output")
            writer.finish(delete=False)
            kept.append(writer.path)
        running = given.file().open()  # a writer still open across the fork
        running.write(b"running")
        pid = os.fork()
        if pid == 0:
            given.close()  # inherited: does nothing in the child
            assert given.file() is None  # inherited folders are inert in the child
            own = OverflowDir()  # the child's own folder: cleaned at the child's exit
            writer = own.file().open()
            writer.write(b"child output")
            writer.finish(delete=False)
            Path(sys.argv[1], "child-own").write_text(own.path)
            sys.exit(0)  # runs atexit
        os.waitpid(pid, 0)
        child_own = Path(sys.argv[1], "child-own").read_text()
        print(given.closed, private.path, *kept, running.path, child_own)
        running.finish(delete=False)
        given.close()
        private.close()
        print(*(os.path.lexists(p) for p in (private.path, *kept, running.path)))
    """
    out = _run(code, str(tmp_path))
    closed, private, given_kept, private_kept, running, child_own = out[:6]
    assert closed == "False"
    # The parent's files survived the child's exit, and the parent can still clean them up.
    assert out[6:] == ["False"] * 4
    # The child's own folder was removed when the child exited.
    assert not os.path.lexists(child_own)
    assert sorted(os.listdir(tmp_path)) == ["child-own"]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_a_forked_child_exiting_keeps_the_parents_files_on_disk(tmp_path: Path) -> None:
    code = """
        import os, sys
        from pathlib import Path
        from nooa.overflow import OverflowDir

        folder = OverflowDir(sys.argv[1])
        writer = folder.file().open()
        writer.write(b"parent output")
        writer.finish(delete=False)
        pid = os.fork()
        if pid == 0:
            sys.exit(0)
        os.waitpid(pid, 0)
        print(folder.closed, Path(writer.path).exists())
        folder.close()
    """
    assert _run(code, str(tmp_path)) == ["False", "True"]
    assert os.listdir(tmp_path) == []


# --- replaced paths: only what the folder created is written or deleted -----------------


def test_a_reserved_file_replaced_by_a_regular_file_is_neither_written_nor_deleted(
    tmp_path: Path,
) -> None:
    folder = OverflowDir(str(tmp_path), max_files=1, file_bytes=4)
    file = folder.file()
    assert file is not None and file.path is not None
    path = Path(file.path)
    path.unlink()
    path.write_bytes(b"FOREIGN-CONTENT-IS-LONG")
    writer = file.open()
    if writer is not None:  # writes, if any, go to the file the folder created
        writer.write(b"XXXX")
        writer.finish(delete=False)
    assert path.read_bytes() == b"FOREIGN-CONTENT-IS-LONG"
    folder.close()
    assert path.read_bytes() == b"FOREIGN-CONTENT-IS-LONG"


def test_eviction_leaves_a_replaced_file_alone(tmp_path: Path) -> None:
    with OverflowDir(str(tmp_path), max_files=1, file_bytes=4) as folder:
        first = _kept(folder)
        first.rename(tmp_path / "moved")  # (renamed, not deleted: see the inode note in _overflow)
        first.write_bytes(b"precious")
        second = _kept(folder)  # evicts the first: its name is no longer ours
        assert first.read_bytes() == b"precious" and second.exists()
    assert first.read_bytes() == b"precious" and not second.exists()


def test_a_renamed_given_folder_with_a_symlink_in_its_place(tmp_path: Path) -> None:
    given = tmp_path / "given"
    given.mkdir()
    folder = OverflowDir(str(given))
    path = _kept(folder)
    moved = tmp_path / "moved-given"
    given.rename(moved)
    outside = tmp_path / "outside"
    outside.mkdir()
    foreign = outside / path.name
    foreign.write_text("precious")
    given.symlink_to(outside, target_is_directory=True)
    folder.close()
    assert foreign.read_text() == "precious"  # not ours: left alone
    assert not (moved / path.name).exists()  # ours, in the folder we created, is removed
    assert folder.closed


def test_a_replaced_private_folder_is_left_alone(tmp_path: Path) -> None:
    folder = OverflowDir()
    original = Path(folder.path)
    path = _kept(folder)
    moved = tmp_path / "moved-private"
    original.rename(moved)
    original.mkdir()
    foreign = original / "foreign.txt"
    foreign.write_text("precious")
    try:
        folder.close()
        assert foreign.read_text() == "precious"  # the replacement is not removed
        assert not (moved / path.name).exists()  # our own file is
        assert folder.closed
    finally:
        shutil.rmtree(original, ignore_errors=True)
        shutil.rmtree(moved, ignore_errors=True)


def test_a_replaced_standalone_file_is_not_deleted(tmp_path: Path) -> None:
    standalone = tmp_path / "standalone"
    writer = OverflowFile(path=str(standalone)).open()
    assert writer is not None
    writer.write(b"mine")
    standalone.unlink()
    standalone.write_bytes(b"precious")
    writer.finish(delete=True)
    assert standalone.read_bytes() == b"precious"


# --- a failed delete: still counted and retried ---------------------------------------


@needs_permissions
def test_a_failed_eviction_stays_counted_and_is_retried(tmp_path: Path) -> None:
    given = tmp_path / "given"
    given.mkdir()
    folder = OverflowDir(str(given), max_files=1, file_bytes=4)
    first = _kept(folder)
    given.chmod(0o500)
    try:
        assert folder.file() is None  # the eviction failed: no room
    finally:
        given.chmod(0o700)
    assert first.exists()
    second = _kept(folder)  # the retry succeeds now
    assert not first.exists()
    assert sorted(os.listdir(given)) == [second.name]  # within 1 file of 4 bytes
    folder.close()
    assert os.listdir(given) == []


@needs_permissions
def test_a_failed_delete_on_finish_stays_counted_and_is_retried(tmp_path: Path) -> None:
    given = tmp_path / "given"
    given.mkdir()
    folder = OverflowDir(str(given), max_files=1, file_bytes=4)
    file = folder.file()
    assert file is not None
    writer = file.open()
    assert writer is not None
    writer.write(b"ABCD")
    given.chmod(0o500)
    try:
        writer.finish(delete=True)  # the delete fails
        assert Path(writer.path).exists()
        assert folder.file() is None  # still counted
    finally:
        given.chmod(0o700)
    assert _kept(folder).exists()  # retried: its place is free
    assert not Path(writer.path).exists()
    folder.close()
    assert os.listdir(given) == []


@needs_permissions
def test_a_failed_delete_is_retried_by_close(tmp_path: Path) -> None:
    given = tmp_path / "given"
    given.mkdir()
    folder = OverflowDir(str(given), max_files=1, file_bytes=4)
    first = _kept(folder)
    given.chmod(0o500)
    try:
        assert folder.file() is None
    finally:
        given.chmod(0o700)
    folder.close()
    assert not first.exists()


# --- a failed close: retried by the next close() and at exit ---------------------------


@needs_permissions
def test_a_failed_close_is_retried_by_the_next_close() -> None:
    folder = OverflowDir()
    private = Path(folder.path)
    path = _kept(folder)
    private.chmod(0o500)
    try:
        folder.close()  # fails: logged, not raised
        assert folder.closed and path.exists()
        with pytest.raises(ValueError):
            folder.file()  # closed to new files all the same
    finally:
        private.chmod(0o700)
    folder.close()
    assert not private.exists()


@needs_permissions
def test_a_failed_close_is_retried_at_exit(tmp_path: Path) -> None:
    code = """
        import os
        from nooa.overflow import OverflowDir

        folder = OverflowDir()
        writer = folder.file().open()
        writer.write(b"ABCD")
        writer.finish(delete=False)
        os.chmod(folder.path, 0o500)
        folder.close()  # fails
        os.chmod(folder.path, 0o700)
        print(folder.path, writer.path, os.path.exists(writer.path))
    """
    private, path, existed = _run(code)
    try:
        assert existed == "True"
        assert not os.path.lexists(private) and not os.path.lexists(path)
    finally:
        shutil.rmtree(private, ignore_errors=True)
