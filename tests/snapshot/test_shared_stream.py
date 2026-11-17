# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SharedTextStream: a TextStream for several writers, and the text file a sys.stdout replacement needs."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from nooa.overflow import OverflowDir
from nooa.snapshot import SharedTextStream, TextStream, render_output

needs_fork = pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")


def test_same_text_as_a_text_stream() -> None:
    chunks = ["abc", "defgh", "x" * 50, "\u00e9t\u00e9", "end"]
    shared = SharedTextStream(unit="chars", head=7, tail=5)
    plain = TextStream(unit="chars", head=7, tail=5)
    for chunk in chunks:
        assert shared.write(chunk) == len(chunk)
        plain.write(chunk)
        assert shared.snapshot() == plain.snapshot()
    assert shared.close() == plain.close()
    assert (shared.unit, shared.head, shared.tail) == ("chars", 7, 5)


def test_bytes_unit_returns_the_length_written() -> None:
    stream = SharedTextStream(unit="bytes", head=2, tail=2)
    assert stream.write(b"0123456789") == 10
    assert render_output(stream.close()) == "01\n[... 6 bytes omitted; full output not kept]\n89"


def test_text_file_protocol() -> None:
    stream = SharedTextStream(unit="chars", head=100, tail=0)
    stream.writelines(["a", "b\n", "c"])
    stream.flush()
    assert (stream.isatty(), stream.readable(), stream.writable()) == (False, False, True)
    with pytest.raises(TypeError):
        stream.write(b"bytes")  # type: ignore[arg-type]
    print("printed", file=stream)  # print() needs only write
    assert render_output(stream.snapshot()) == "ab\ncprinted\n"


def test_writes_after_close_are_dropped() -> None:
    with SharedTextStream(unit="chars", head=10, tail=0) as stream:
        stream.write("abc")
    assert stream.write("late") == 4
    assert render_output(stream.snapshot()) == "abc"


def test_writes_after_close_are_counted_and_logged_once(caplog) -> None:
    stream = SharedTextStream(unit="chars", head=10, tail=0)
    stream.write("abc")
    stream.close()
    assert stream.dropped_after_close == 0
    with caplog.at_level(logging.WARNING, logger="nooa.snapshot._stream"):
        assert stream.write("late") == 4
        stream.write("")
        stream.writelines(["more", "é"])
    assert stream.dropped_after_close == 9
    warnings = [r.getMessage() for r in caplog.records if r.name == "nooa.snapshot._stream"]
    assert len(warnings) == 1
    assert warnings[0].startswith("4 chars written after the stream closed were dropped")
    assert render_output(stream.snapshot()) == "abc"


def test_bytes_dropped_after_close_are_counted_in_bytes(caplog) -> None:
    stream = SharedTextStream(unit="bytes", head=10, tail=0)
    stream.close()
    with caplog.at_level(logging.WARNING, logger="nooa.snapshot._stream"):
        stream.write("é")  # two UTF-8 bytes
        stream.write(memoryview(b"abcd").cast("I"))  # one item, four bytes
        stream.write(bytearray(b"xy"))
    assert stream.dropped_after_close == 8
    messages = [r.getMessage() for r in caplog.records if r.name == "nooa.snapshot._stream"]
    assert len(messages) == 1 and messages[0].startswith("2 bytes written after")


def test_a_write_while_another_opens_the_file_waits_for_it(tmp_path, monkeypatch) -> None:
    # correctness 1 (capture_threads.py): the first cutting write opens the
    # file; a second writer meanwhile filled and cut the window, and the file
    # got only the cut copy (110 of 204 bytes) while the Text called it full.
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    entered, resume = threading.Event(), threading.Event()

    class GatedFolder(OverflowDir):
        def open(self, max_bytes=None):
            writer = super().open(max_bytes=max_bytes)
            entered.set()
            resume.wait(timeout=5)
            return writer

    with GatedFolder() as folder:
        stream = SharedTextStream(unit="chars", head=5, tail=5, overflow=folder)
        stream.write("seed")
        first = threading.Thread(target=stream.write, args=("A" * 100,))
        first.start()
        assert entered.wait(timeout=5)
        second = threading.Thread(target=stream.write, args=("B" * 100,))
        second.start()
        second.join(timeout=0.3)
        assert second.is_alive()  # waits for the first write, file open included
        resume.set()
        first.join(timeout=5)
        second.join(timeout=5)
        text = stream.close()
        contents = Path(text.full_path).read_text()
        assert contents == "seed" + "A" * 100 + "B" * 100
        assert text.full is not None and text.full.kept == text.full.total == 204


def test_many_threads_lose_nothing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    with OverflowDir() as folder:
        stream = SharedTextStream(unit="chars", head=10, tail=10, overflow=folder)

        def writer(letter: str) -> None:
            for _ in range(500):
                stream.write(letter * 7)

        threads = [threading.Thread(target=writer, args=(c,)) for c in "abcdefgh"]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        text = stream.close()
        assert text.length == 8 * 500 * 7
        contents = Path(text.full_path).read_text()
        assert sorted(contents) == sorted("".join(c * 3500 for c in "abcdefgh"))
        assert all(len(run) % 7 == 0 for run in _runs(contents))  # no write split by another


def _runs(text: str) -> list[str]:
    runs, start = [], 0
    for i in range(1, len(text) + 1):
        if i == len(text) or text[i] != text[start]:
            runs.append(text[start:i])
            start = i
    return runs


@needs_fork
def test_a_forked_child_never_waits_on_a_lock_held_at_the_fork(tmp_path: Path) -> None:
    code = """
        import os, threading, time
        from nooa.snapshot import SharedTextStream, render_output

        stream = SharedTextStream(unit="chars", head=4, tail=4)
        stream.write("parent")
        ready, release = threading.Event(), threading.Event()

        def locker():
            with stream._lock:
                ready.set()
                release.wait()

        thread = threading.Thread(target=locker)
        thread.start()
        ready.wait()
        pid = os.fork()
        if pid == 0:
            stream.write("-child")
            os._exit(0 if render_output(stream.snapshot()).startswith("pare") else 3)
        release.set()
        thread.join()
        deadline = time.monotonic() + 10
        status = None
        while status is None and time.monotonic() < deadline:
            done, code = os.waitpid(pid, os.WNOHANG)
            status = code if done else None
            time.sleep(0.05)
        if status is None:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
        print("blocked" if status is None else f"exit {os.waitstatus_to_exitcode(status)}")
    """
    result = subprocess.run(
        [sys.executable, "-W", "ignore::DeprecationWarning", "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={**os.environ, "TMPDIR": str(tmp_path)},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "exit 0"
