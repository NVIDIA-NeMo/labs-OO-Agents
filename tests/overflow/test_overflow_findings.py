# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nooa.overflow: forks, the exit, failed closes, interrupts, eviction order and the writer API.

Each test replays a reviewer's probe: the scenario that broke the contract,
and what the contract says must happen instead.
"""

from __future__ import annotations

import dataclasses
import errno
import json
import logging
import os
import signal
import subprocess
import sys
import textwrap
import tracemalloc
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from nooa.overflow import OverflowDir, OverflowFile

if TYPE_CHECKING:
    from nooa.overflow import OverflowWriter

needs_fork = pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX resource limits and signals")
_LOGGER = "nooa.overflow._overflow"


def _open(file: OverflowFile | None) -> OverflowWriter:
    assert file is not None
    writer = file.open()
    assert writer is not None
    return writer


def _run(code: str, tmp_path: Path, timeout: float = 60) -> list[str]:
    """Run ``code`` in a fresh interpreter with ``TMPDIR=tmp_path``; its stdout lines."""
    env = {**os.environ, "TMPDIR": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


def _folders(tmp_path: Path) -> list[str]:
    return sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("nooa-overflow-"))


# --- fork: no duplicated bytes, no inherited lock held ------------------------------------


@needs_fork
def test_a_forked_writer_never_duplicates_bytes_or_breaks_the_cap(tmp_path: Path) -> None:
    # fs_probe fork_writer: "AB" written before the fork, the child writes "CCC"
    # and finishes, the parent writes "PPP": the disk held ABCCCABPPP (10 > 5 bytes).
    code = """
        import json, os
        from pathlib import Path
        from nooa.overflow import OverflowDir

        folder = OverflowDir(file_bytes=5, max_files=1)
        writer = folder.file().open()
        writer.write(b"AB")
        pid = os.fork()
        if pid == 0:
            writer.write(b"CCC")
            writer.finish(delete=False)
            os._exit(0)
        os.waitpid(pid, 0)
        writer.write(b"PPP")
        writer.finish(delete=False)
        print(json.dumps([Path(writer.path).read_text(), writer.written]))
        folder.close()
    """
    (line,) = _run(code, tmp_path)
    assert json.loads(line) == ["ABPPP", 5]


@needs_fork
@pytest.mark.parametrize("lock", ["owner", "global"])
def test_a_forked_child_never_waits_on_a_lock_held_at_the_fork(tmp_path: Path, lock: str) -> None:
    # fs_probe fork_lock: another thread holds the folder's (or the registry's)
    # lock across the fork; the child finishing a writer (or making a folder) hung.
    code = f"""
        import os, threading, time
        from nooa.overflow import OverflowDir
        from nooa.overflow import _overflow as impl

        folder = OverflowDir(max_files=3)
        writer = folder.file().open()
        held = impl._live_lock if {lock!r} == "global" else (
            folder._folder.lock if hasattr(folder, "_folder") else folder._lock)
        ready, release = threading.Event(), threading.Event()
        def locker():
            with held:
                ready.set()
                release.wait()
        thread = threading.Thread(target=locker)
        thread.start()
        ready.wait()
        pid = os.fork()
        if pid == 0:
            if {lock!r} == "global":
                OverflowDir()
            else:
                writer.finish(delete=False)
            os._exit(0)
        release.set()
        thread.join()
        deadline = time.monotonic() + 10
        blocked = True
        while blocked and time.monotonic() < deadline:
            blocked = os.waitpid(pid, os.WNOHANG)[0] == 0
            time.sleep(0.05)
        if blocked:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
        print("blocked" if blocked else "free")
        writer.finish(delete=True)
        folder.close()
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert _run(code, tmp_path)[-1] == "free"


# --- the exit cleanup: never without the lock; owners made during the exit ---------------


def test_the_exit_cleanup_never_runs_while_another_thread_holds_the_lock(tmp_path: Path) -> None:
    # GLM #4 / fs S1 / contract M1: a thread inside the folder's lock (here: deleting
    # its file in finish) past the exit's one-second wait; the cleanup ran anyway,
    # unlocked, cleared the bookkeeping under it, and finish() raised KeyError.
    code = """
        import atexit, json, os, threading
        from nooa.overflow import OverflowDir
        from nooa.overflow import _overflow as impl

        folder = OverflowDir(max_files=2)
        writer = folder.file().open()
        writer.write(b"x")
        entered, resume = threading.Event(), threading.Event()
        real_unlink = os.unlink
        def slow_unlink(path, *args, **kwargs):
            if threading.current_thread() is worker and "nooa-output-" in str(path):
                entered.set()
                resume.wait(20)
            return real_unlink(path, *args, **kwargs)
        os.unlink = slow_unlink
        errors = []
        def work():
            try:
                writer.finish(delete=True)
            except BaseException as exc:
                errors.append(repr(exc))
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        entered.wait(10)
        def after_cleanup():  # registered before the overflow cleanup: runs after it
            cleaned_unlocked = not os.path.exists(folder.path)
            resume.set()
            worker.join(10)
            print(json.dumps(dict(cleaned_unlocked=cleaned_unlocked, errors=errors,
                                  left=os.path.exists(folder.path))), flush=True)
        atexit.unregister(impl._close_all_at_exit)
        atexit.register(after_cleanup)
        atexit.register(impl._close_all_at_exit)
    """
    (line,) = _run(code, tmp_path)
    # The cleanup waited for the lock; the thread holding it removed the folder when done.
    assert json.loads(line) == {"cleaned_unlocked": False, "errors": [], "left": False}
    assert _folders(tmp_path) == []


def test_a_file_created_during_the_exit_cleanup_is_not_left_behind(tmp_path: Path) -> None:
    # fs_exit_probe: a daemon inside file() is paused just before it creates the
    # file; another exit handler resumes it after the overflow cleanup.
    code = """
        import atexit, json, os, threading
        from nooa.overflow import OverflowDir
        from nooa.overflow import _overflow as impl

        folder = OverflowDir(max_files=1)
        entered, resume = threading.Event(), threading.Event()
        real_open = os.open
        def slow_open(path, *args, **kwargs):
            if threading.current_thread() is worker and "nooa-output-" in str(path):
                entered.set()
                resume.wait(20)
            return real_open(path, *args, **kwargs)
        os.open = slow_open
        result = []
        worker = threading.Thread(target=lambda: result.append(folder.file()), daemon=True)
        worker.start()
        entered.wait(10)
        def after_cleanup():
            resume.set()
            worker.join(10)
            print(json.dumps(dict(closed=folder.closed, handed_out=result[0] is not None)))
        atexit.unregister(impl._close_all_at_exit)
        atexit.register(after_cleanup)
        atexit.register(impl._close_all_at_exit)
    """
    (line,) = _run(code, tmp_path)
    assert json.loads(line) == {"closed": True, "handed_out": False}
    assert _folders(tmp_path) == []  # nothing left after the actual exit


def test_an_owner_made_by_a_later_exit_handler_leaves_nothing(tmp_path: Path) -> None:
    # contract S2 / GLM #7 (contract_exit_order.py): a handler registered before
    # nooa.overflow was imported runs after its cleanup and makes an owner, writes
    # and keeps a file; the folder outlived the process.
    code = """
        import atexit

        @atexit.register
        def producer_registered_before_overflow_import():
            from nooa.overflow import OverflowDir
            owner = OverflowDir()
            file = owner.file()  # during the exit: memory only
            if file is not None:
                writer = file.open()
                writer.write(b"late output")
                writer.finish(delete=False)
            print("LATE", file is None, flush=True)

        from nooa.overflow import OverflowDir
        owner = OverflowDir()
        writer = owner.file().open()
        writer.write(b"ordinary output")
        writer.finish(delete=False)
        print("NORMAL", writer.path, flush=True)
    """
    out = _run(code, tmp_path)
    assert out[-1] == "LATE True"
    assert _folders(tmp_path) == []


# --- finish(): failures are logged, written stays honest ---------------------------------


@posix_only
@pytest.mark.parametrize("limit_before_write", [False, True])
def test_written_never_counts_bytes_that_did_not_reach_the_file(
    caplog: pytest.LogCaptureFixture, limit_before_write: bool
) -> None:
    # contract S1 (FINISH_FLUSH_ERROR) and fs S4 (finish_limit): with a file size
    # limit, write() said 10 (or 6) bytes, finish(delete=False) kept 3 (or 2),
    # nothing was logged and the file was kept as good.
    import resource

    previous = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    limits = resource.getrlimit(resource.RLIMIT_FSIZE)
    folder = OverflowDir(file_bytes=100)
    writer = _open(folder.file())
    data = b"ABCDEF" if limit_before_write else b"abcdefghij"
    raised = None
    try:
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            if limit_before_write:
                resource.setrlimit(resource.RLIMIT_FSIZE, (2, limits[1]))
            try:
                writer.write(data)
            except OSError as exc:
                raised = exc
            if not limit_before_write:
                resource.setrlimit(resource.RLIMIT_FSIZE, (3, limits[1]))
            kept = writer.finish(delete=False)
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, limits)
        signal.signal(signal.SIGXFSZ, previous)
    on_disk = Path(writer.path).read_bytes() if os.path.exists(writer.path) else None
    if kept:
        assert on_disk == data[: writer.written]  # what written says is on disk
    else:
        assert on_disk is None and caplog.records  # a failed file is logged and deleted
    if limit_before_write:
        assert raised is not None and raised.errno == errno.EFBIG and writer.written == 2
    else:
        assert writer.written == len(data) and on_disk == data
    folder.close()


def test_a_failed_close_is_logged_and_the_file_is_not_kept(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # GLM #6 / contract S1: finish() swallowed the close error without a word and
    # kept the file as good.
    folder = OverflowDir(file_bytes=100)
    writer = _open(folder.file())
    writer.write(b"output")
    real_close = os.close

    def failing_close(fd: int) -> None:
        real_close(fd)
        raise OSError(errno.EIO, "close failed")

    monkeypatch.setattr(os, "close", failing_close)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        kept = writer.finish(delete=False)
    monkeypatch.undo()
    assert kept is False
    assert any("could not close overflow file" in r.getMessage() for r in caplog.records)
    assert not os.path.exists(writer.path)  # not kept as if it were whole
    assert folder.file() is not None  # and its place is free
    folder.close()


# --- an interrupt while creating a file ----------------------------------------------------


@posix_only
def test_an_interrupt_while_creating_a_file_leaves_nothing_untracked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # fs S2 (interrupted_create): SIGINT right after the file was created, before
    # the folder recorded it: an untracked file, and a second one under max_files=1.
    folder = OverflowDir(max_files=1)
    real_open = os.open

    def interrupted_open(path: str, *args: object, **kwargs: object) -> int:
        fd = real_open(path, *args, **kwargs)  # type: ignore[arg-type]
        if "nooa-output-" in str(path):
            signal.raise_signal(signal.SIGINT)  # the default handler raises KeyboardInterrupt
        return fd

    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    monkeypatch.setattr(os, "open", interrupted_open)
    try:
        with pytest.raises(KeyboardInterrupt):
            folder.file()
    finally:
        monkeypatch.undo()
        signal.signal(signal.SIGINT, previous)
    assert os.listdir(folder.path) == []  # nothing left untracked
    assert folder.file() is not None
    assert len(os.listdir(folder.path)) == 1  # the cap holds
    folder.close()
    assert not os.path.exists(folder.path)


# --- the link to the folder is private --------------------------------------------------


def test_the_folder_link_is_not_a_constructor_argument_or_a_field() -> None:
    # model S1 / GLM #10, #18, #27 (probes_r9 P1): OverflowFile(_owner=...) and
    # dataclasses.replace(file, path=None) made a file that claimed the folder but
    # wrote 1000 bytes outside it; the folder's own place stayed in use.
    folder = OverflowDir(max_files=1, file_bytes=4)
    file = folder.file()
    assert file is not None
    assert [f.name for f in dataclasses.fields(OverflowFile)] == ["path", "max_bytes"]
    with pytest.raises(TypeError):
        OverflowFile(path=file.path, _owner=folder)  # type: ignore[call-arg]
    copy = dataclasses.replace(file, path=None, max_bytes=1000)
    writer = _open(copy)  # a plain standalone file: not the folder's, not counted by it
    try:
        assert os.path.dirname(writer.path) != folder.path
    finally:
        writer.finish(delete=True)
    with pytest.raises(FileExistsError):
        dataclasses.replace(file, max_bytes=1000).open()  # standalone, at an existing path
    assert _open(file).write(b"0123456789") == 4  # the folder's file, capped, still opens
    folder.close()


# --- eviction: the newest finished is kept longest; create before evicting ----------------


def test_eviction_keeps_the_most_recently_finished_file() -> None:
    # model S2 (probes_r9 P3): a long command started first and finished last
    # lost its output to the next file().
    folder = OverflowDir(max_files=2, file_bytes=8)
    a, b = _open(folder.file()), _open(folder.file())
    a.write(b"A-long")
    b.write(b"B-short")
    b.finish(delete=False)
    a.finish(delete=False)
    assert folder.file() is not None
    assert os.path.exists(a.path) and not os.path.exists(b.path)
    folder.close()


@posix_only
def test_a_failed_create_evicts_nothing() -> None:
    # model C1 / GLM #13 (probes_r9 P4): under EMFILE, file() deleted the closed
    # output and then failed to create the new file.
    import resource

    folder = OverflowDir(max_files=1, file_bytes=8)
    writer = _open(folder.file())
    writer.write(b"keep me")
    writer.finish(delete=False)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    held: list[int] = []
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, hard))
    try:
        try:
            while True:
                held.append(os.open(os.devnull, os.O_RDONLY))
        except OSError:
            pass
        got = folder.file()
    finally:
        for fd in held:
            os.close(fd)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert got is None
    assert Path(writer.path).read_bytes() == b"keep me"
    assert folder.file() is not None  # once a file can be made, the old one makes room
    assert not os.path.exists(writer.path)
    folder.close()


# --- removal failures are logged ---------------------------------------------------------


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores modes")
def test_a_failed_removal_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    # GLM #1: close() of a folder it could not remove said nothing.
    folder = OverflowDir()
    writer = _open(folder.file())
    writer.write(b"x")
    writer.finish(delete=False)
    os.chmod(folder.path, 0o500)
    try:
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            folder.close()
        assert os.path.exists(writer.path)
        assert any(r.levelno == logging.WARNING for r in caplog.records)
    finally:
        os.chmod(folder.path, 0o700)
    folder.close()
    assert not os.path.exists(folder.path)


# --- finish twice; a read-only path ---------------------------------------------------------


def test_a_later_finish_with_delete_removes_a_kept_file(tmp_path: Path) -> None:
    # GLM #15: finish(delete=True) after finish(delete=False) left the file.
    standalone = _open(OverflowFile(path=str(tmp_path / "out.txt")))
    standalone.write(b"out")
    assert standalone.finish(delete=False) is True
    assert standalone.finish(delete=True) is False
    assert not os.path.exists(standalone.path)
    folder = OverflowDir(max_files=1)
    writer = _open(folder.file())
    writer.finish(delete=False)
    writer.finish(delete=True)
    assert not os.path.exists(writer.path)
    assert folder.file() is not None  # its place is free
    folder.close()


def test_the_writers_path_is_read_only(tmp_path: Path) -> None:
    # GLM #19: reassigning writer.path made finish(delete=True) delete another file.
    other = tmp_path / "other.txt"
    other.write_text("precious")
    writer = _open(OverflowFile())
    with pytest.raises(AttributeError):
        writer.path = str(other)  # type: ignore[misc]
    writer.finish(delete=True)
    assert other.read_text() == "precious"


# --- a handed-out file that is never opened ---------------------------------------------


def test_a_handed_out_file_can_be_released() -> None:
    # model C2 (probes_r9 P2): a caller got a file and failed before opening it;
    # the place stayed taken until close().
    folder = OverflowDir(max_files=1, file_bytes=8)
    file = folder.file()
    assert file is not None and file.path is not None
    assert folder.file() is None
    file.release()
    assert not os.path.exists(file.path) and file.open() is None
    assert folder.file() is not None  # the place is free again
    folder.close()


# --- the writer's errors -------------------------------------------------------------------


@posix_only
def test_a_writer_raises_only_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    # model C3 / GLM #8 (probes_r9 P5): after writer.file.close(), which the docs
    # allowed, write() raised ValueError. The descriptor is now private: only
    # finish() closes it, a finished writer's write returns 0, a file error is OSError.
    import resource

    folder = OverflowDir(max_files=1, file_bytes=8)
    writer = _open(folder.file())
    assert not hasattr(writer, "file")
    previous = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    limits = resource.getrlimit(resource.RLIMIT_FSIZE)
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, limits[1]))
    try:
        with pytest.raises(OSError):
            writer.write(b"x")
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, limits)
        signal.signal(signal.SIGXFSZ, previous)
    writer.finish(delete=True)
    assert writer.write(b"x") == 0
    folder.close()


# --- Windows flags; one copy less ------------------------------------------------------------


def test_files_are_created_binary_where_o_binary_exists(tmp_path: Path) -> None:
    # GLM #9: without O_BINARY, Windows writes b"\n" as b"\r\n".
    code = """
        import os
        os.O_BINARY = 1 << 29  # as on Windows
        from nooa.overflow import _overflow
        print(bool(_overflow._CREATE & os.O_BINARY))
    """
    assert _run(code, tmp_path) == ["True"]


def test_a_cut_write_copies_nothing() -> None:
    # GLM #25: write() sliced a copy of the part that fits (up to file_bytes).
    writer = _open(OverflowFile(max_bytes=4 << 20))
    data = os.urandom(8 << 20)
    tracemalloc.start()
    try:
        assert writer.write(data) == 4 << 20
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
        writer.finish(delete=True)
    assert peak < 1 << 20
