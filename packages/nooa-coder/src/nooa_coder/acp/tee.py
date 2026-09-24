# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A JSON-RPC tee: record every ACP frame in both directions to a JSON Lines log.

Two forms. ``FrameLog`` is an observer on the server's own ACP connection
(``nooa-coder --tee PATH``). The relay below wraps any server command.

``nooa-coder-tee --log PATH -- COMMAND...`` runs any agent server command
as a child process and relays its standard input and output unchanged,
appending one record per frame: ``{"ts": <unix time>, "dir": "in"|"out",
"frame": <the JSON message, or the raw line when it is not JSON>}``. ``in``
is client to server, ``out`` server to client. The child's standard error
passes through, SIGTERM is forwarded to it, its standard input is closed
when the relay's own input ends, and the relay exits with the child's exit
status.

The log can hold secrets (``mcpServers`` carries environment variables and
headers), so it is created readable by its owner only.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO, Any


def open_private_log(path: Path) -> IO[str]:
    """Open ``path`` for appending, created (or narrowed) to mode 0600."""
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.fchmod(fd, 0o600)
    return os.fdopen(fd, "a", encoding="utf-8")


def frame_record(direction: str, frame: Any) -> str:
    """One log line for a frame travelling in ``direction`` (``in`` or ``out``)."""
    return json.dumps({"ts": time.time(), "dir": direction, "frame": frame}, default=str) + "\n"


def _decode(line: bytes) -> Any:
    text = line.decode("utf-8", errors="replace").rstrip("\r\n")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


class _Log:
    """A log file shared by the two relay threads; one line per frame, in order."""

    def __init__(self, path: Path) -> None:
        self._file = open_private_log(path)
        self._lock = threading.Lock()

    def write(self, direction: str, line: bytes) -> None:
        record = frame_record(direction, _decode(line))
        with self._lock:
            self._file.write(record)
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            self._file.close()


def _pump(
    source: IO[bytes], sink: IO[bytes], log: _Log, direction: str, *, close_sink: bool
) -> None:
    """Copy lines from ``source`` to ``sink`` unchanged, logging each.

    ``close_sink`` closes ``sink`` when ``source`` ends: the child's stdin,
    so the server sees end of input when the client's input ends.
    """
    try:
        for line in iter(source.readline, b""):
            log.write(direction, line)
            try:
                sink.write(line)
                sink.flush()
            except (BrokenPipeError, ValueError):
                break
    finally:
        if close_sink:
            try:
                sink.close()
            except (BrokenPipeError, OSError):
                pass


def relay(command: list[str], log_path: Path) -> int:
    """Run ``command`` with its stdio relayed and logged; return its exit status."""
    log = _Log(log_path)
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert child.stdin is not None and child.stdout is not None

    def forward(signum: int, _frame: Any) -> None:
        child.send_signal(signum)

    previous = signal.signal(signal.SIGTERM, forward)
    to_child = threading.Thread(
        target=_pump,
        args=(sys.stdin.buffer, child.stdin, log, "in"),
        kwargs={"close_sink": True},
        daemon=True,
    )
    from_child = threading.Thread(
        target=_pump,
        args=(child.stdout, sys.stdout.buffer, log, "out"),
        kwargs={"close_sink": False},
        daemon=True,
    )
    to_child.start()
    from_child.start()
    try:
        returncode = child.wait()
        from_child.join()
    finally:
        signal.signal(signal.SIGTERM, previous)
        log.close()
    # A child killed by a signal reports -N; shells report that as 128 + N.
    return returncode if returncode >= 0 else 128 - returncode


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="nooa-coder-tee",
        description=(
            "Run an ACP agent server and record every JSON-RPC frame in both directions "
            "to a JSON Lines log. Example: nooa-coder-tee --log acp.jsonl -- nooa-coder "
            "--model my-alias"
        ),
    )
    parser.add_argument(
        "--log", required=True, type=Path, help="Log file (appended; created with mode 0600)."
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- COMMAND [ARGS...]")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the server command after --")
    code = relay(command, args.log)
    sys.stdout.flush()
    sys.stderr.flush()
    # The thread copying standard input may still be blocked in a read (the
    # server exited first). A normal interpreter exit would abort while that
    # thread holds the stdin buffer, so leave without finalisation.
    os._exit(code)


if __name__ == "__main__":
    main()


class FrameLog:
    """The in-process tee (``nooa-coder --tee PATH``): an ACP connection observer.

    Pass it to ``run_agent(..., observers=[...])``; the connection calls it
    with every parsed frame it reads (``in``) or writes (``out``). The call
    runs on the connection's receive loop, so it only queues the record; a
    daemon thread writes it. When ``max_pending`` records are waiting, new
    ones are dropped and counted in ``dropped`` rather than stalling ACP.
    """

    def __init__(self, path: Path, *, max_pending: int = 10_000) -> None:
        self._file = open_private_log(path)
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=max_pending)
        self._closed = False
        self.dropped = 0
        self._thread = threading.Thread(target=self._write_loop, name="nooa-coder-tee", daemon=True)
        self._thread.start()

    def __call__(self, event: Any) -> None:
        from acp.connection import StreamDirection

        if self._closed:
            return
        direction = "in" if event.direction == StreamDirection.INCOMING else "out"
        try:
            self._queue.put_nowait(frame_record(direction, event.message))
        except queue.Full:
            self.dropped += 1

    def _write_loop(self) -> None:
        while (record := self._queue.get()) is not None:
            try:
                self._file.write(record)
                self._file.flush()
            except (OSError, ValueError):
                return

    def close(self, timeout: float = 5.0) -> None:
        """Write what is queued (waiting up to ``timeout`` seconds) and close the file."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout)
        if not self._thread.is_alive():
            self._file.close()
