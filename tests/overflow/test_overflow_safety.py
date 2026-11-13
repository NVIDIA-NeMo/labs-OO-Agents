# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nooa.overflow: forks, replaced files, failed deletes and failed closes.

Each test replays a scenario where a folder deleted files that were not its
own, wrote to a file it had not created, or forgot its own: a forked child,
a file replaced at its path, a delete or a close failing. (Same-user
tampering inside a private folder is out of scope; see ``nooa.overflow``.)
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

needs_fork = pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
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


# --- fork: inherited folders and writers are inert in the child -------------------------


@needs_fork
def test_a_forked_child_exiting_keeps_the_parents_files_on_disk(tmp_path: Path) -> None:
    # Round 8: the child's exit ran the inherited cleanup and removed the parent's folder.
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
            sys.exit(0)  # runs atexit
        os.waitpid(pid, 0)
        print(folder.closed, Path(writer.path).exists())
        folder.close()
    """
    assert _run(code, str(tmp_path)) == ["False", "True"]
    assert os.listdir(tmp_path) == []


@needs_fork
def test_a_forked_child_leaves_the_parents_folders_alone_and_cleans_its_own(
    tmp_path: Path,
) -> None:
    code = """
        import os, sys
        from pathlib import Path
        from nooa.overflow import OverflowDir

        folders = [OverflowDir(sys.argv[1]), OverflowDir(sys.argv[1])]
        kept = []
        for folder in folders:
            writer = folder.file().open()
            writer.write(b"parent output")
            writer.finish(delete=False)
            kept.append(writer.path)
        running = folders[0].file().open()
        running.write(b"running")
        pid = os.fork()
        if pid == 0:
            running.finish(delete=True)  # inert in the child: deletes nothing
            folders[1].close()  # does nothing in the child
            assert folders[0].file() is None  # inherited folders are inert in the child
            own = OverflowDir(sys.argv[1])  # the child's own folder: cleaned at the child's exit
            writer = own.file().open()
            writer.write(b"child output")
            writer.finish(delete=False)
            Path(sys.argv[1], "child-own").write_text(own.path)
            sys.exit(0)  # runs atexit
        os.waitpid(pid, 0)
        print(*(os.path.exists(p) for p in (*kept, running.path)))
        print(Path(sys.argv[1], "child-own").read_text())
        running.finish(delete=False)
        for folder in folders:
            folder.close()
    """
    out = _run(code, str(tmp_path))
    assert out[:3] == ["True"] * 3  # the parent's files survived the child
    assert not os.path.lexists(out[3])  # the child's own folder went at the child's exit
    assert sorted(os.listdir(tmp_path)) == ["child-own"]  # and the parent cleaned its own


# --- replaced files: only what the folder created is written; only ours is deleted -----


def test_a_reserved_file_replaced_by_a_regular_file_is_not_written() -> None:
    # Round 8: open() reopened the name and wrote over a file put in its place.
    folder = OverflowDir(max_files=1, file_bytes=4)
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
def test_a_failed_eviction_stays_counted_and_is_retried() -> None:
    folder = OverflowDir(max_files=1, file_bytes=4)
    private = Path(folder.path)
    first = _kept(folder)
    private.chmod(0o500)
    try:
        assert folder.file() is None  # the eviction failed: no room
    finally:
        private.chmod(0o700)
    assert first.exists()
    second = _kept(folder)  # the retry succeeds now
    assert not first.exists()
    assert sorted(os.listdir(private)) == [second.name]  # within 1 file of 4 bytes
    folder.close()
    assert not private.exists()


@needs_permissions
def test_a_failed_delete_on_finish_stays_counted_and_is_retried() -> None:
    folder = OverflowDir(max_files=1, file_bytes=4)
    private = Path(folder.path)
    file = folder.file()
    assert file is not None
    writer = file.open()
    assert writer is not None
    writer.write(b"ABCD")
    private.chmod(0o500)
    try:
        writer.finish(delete=True)  # the delete fails
        assert Path(writer.path).exists()
        assert folder.file() is None  # still counted
    finally:
        private.chmod(0o700)
    assert _kept(folder).exists()  # retried: its place is free
    assert not Path(writer.path).exists()
    folder.close()
    assert not private.exists()


@needs_permissions
def test_a_failed_delete_is_retried_by_close() -> None:
    folder = OverflowDir(max_files=1, file_bytes=4)
    private = Path(folder.path)
    first = _kept(folder)
    private.chmod(0o500)
    try:
        assert folder.file() is None
        assert first.exists()
    finally:
        private.chmod(0o700)
    folder.close()
    assert not first.exists() and not private.exists()


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
def test_a_failed_close_is_retried_at_exit() -> None:
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
