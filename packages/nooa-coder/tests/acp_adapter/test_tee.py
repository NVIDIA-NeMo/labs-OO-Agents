# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The JSON-RPC tee: the external relay (``nooa-coder-tee``) and the in-process observer."""

import json
import signal
import stat
import subprocess
import sys
from pathlib import Path

import pytest

_ECHO = Path(__file__).parent / "fixtures" / "echo_server.py"
_TIMEOUT = 30


def _relay(log: Path, *server_args: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "nooa_coder.acp.tee",
            "--log",
            str(log),
            "--",
            sys.executable,
            str(_ECHO),
            *server_args,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _frames() -> list[bytes]:
    return [
        json.dumps(
            {"jsonrpc": "2.0", "id": index, "method": method, "params": {"n": index}}
        ).encode()
        + b"\n"
        for index, method in enumerate(("initialize", "session/new", "session/prompt"), start=1)
    ]


def test_relay_logs_both_directions_in_order_and_passes_bytes_through(tmp_path):
    log = tmp_path / "tee.jsonl"
    received = tmp_path / "server-received.bin"
    process = _relay(log, "--record", str(received))
    assert process.stdin is not None and process.stdout is not None
    replies = []
    for frame in _frames():
        process.stdin.write(frame)
        process.stdin.flush()
        replies.append(process.stdout.readline())
    process.stdin.close()
    assert process.wait(_TIMEOUT) == 0
    process.stdout.close()
    assert process.stderr is not None
    stderr = process.stderr.read().decode()
    process.stderr.close()

    # The server saw exactly the bytes the client sent.
    assert received.read_bytes() == b"".join(_frames())
    # The client saw exactly the bytes the server wrote.
    assert [json.loads(reply) for reply in replies] == [
        {"jsonrpc": "2.0", "id": 1, "result": {"echo": "initialize"}},
        {"jsonrpc": "2.0", "id": 2, "result": {"echo": "session/new"}},
        {"jsonrpc": "2.0", "id": 3, "result": {"echo": "session/prompt"}},
    ]
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [record["dir"] for record in records] == ["in", "out"] * 3
    assert [record["frame"] for record in records[::2]] == [json.loads(f) for f in _frames()]
    assert [record["frame"] for record in records[1::2]] == [json.loads(r) for r in replies]
    assert all(isinstance(record["ts"], float) for record in records)
    # The server's stderr passes through.
    assert "echo-server-stderr" in stderr


def test_relay_creates_the_log_readable_by_the_owner_only(tmp_path):
    log = tmp_path / "tee.jsonl"
    process = _relay(log)
    process.communicate(b"", timeout=_TIMEOUT)
    assert stat.S_IMODE(log.stat().st_mode) == 0o600


def test_relay_passes_the_exit_code_through(tmp_path):
    process = _relay(tmp_path / "tee.jsonl", "--exit-code", "3")
    process.communicate(b"", timeout=_TIMEOUT)
    assert process.returncode == 3


def test_relay_forwards_sigterm_to_the_server(tmp_path):
    marker = tmp_path / "terminated"
    process = _relay(tmp_path / "tee.jsonl", "--term-marker", str(marker))
    assert process.stderr is not None
    # The server says "ready" on stderr once its SIGTERM handler is installed.
    while b"ready" not in (line := process.stderr.readline()):
        assert line, "the relay exited before the server was ready"
    process.send_signal(signal.SIGTERM)
    assert process.wait(_TIMEOUT) == 7
    assert marker.read_text() == "terminated"
    for stream in (process.stdin, process.stdout, process.stderr):
        assert stream is not None
        stream.close()


def test_relay_runs_as_a_module():
    result = subprocess.run(
        [sys.executable, "-m", "nooa_coder.acp.tee", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--log" in result.stdout


def test_relay_without_a_command_is_a_usage_error(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "nooa_coder.acp.tee", "--log", str(tmp_path / "x.jsonl")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 2


@pytest.mark.parametrize("line", [b"not json\n"])
def test_relay_records_a_frame_that_is_not_json_as_text(tmp_path, line):
    log = tmp_path / "tee.jsonl"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "nooa_coder.acp.tee",
            "--log",
            str(log),
            "--",
            sys.executable,
            "-c",
            "import sys; sys.stdout.write(sys.stdin.read())",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    out, _ = process.communicate(line, timeout=_TIMEOUT)
    assert out == line
    records = [json.loads(record) for record in log.read_text().splitlines()]
    assert [(record["dir"], record["frame"]) for record in records] == [
        ("in", "not json"),
        ("out", "not json"),
    ]


# ---- the in-process observer (nooa-coder --tee) ------------------------------


def test_observer_records_frames_in_both_directions(tmp_path):
    from acp.connection import StreamDirection, StreamEvent
    from nooa_coder.acp.tee import FrameLog

    log = tmp_path / "tee.jsonl"
    tee = FrameLog(log)
    request = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    response = {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": 1}}
    tee(StreamEvent(StreamDirection.INCOMING, request))
    tee(StreamEvent(StreamDirection.OUTGOING, response))
    tee.close()

    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(record["dir"], record["frame"]) for record in records] == [
        ("in", request),
        ("out", response),
    ]
    assert stat.S_IMODE(log.stat().st_mode) == 0o600


def test_observer_narrows_an_existing_log_to_the_owner(tmp_path):
    from nooa_coder.acp.tee import FrameLog

    log = tmp_path / "tee.jsonl"
    log.write_text("")
    log.chmod(0o644)
    FrameLog(log).close()
    assert stat.S_IMODE(log.stat().st_mode) == 0o600


def test_observer_drops_frames_instead_of_blocking_when_its_queue_is_full(tmp_path, monkeypatch):
    """The observer runs on the connection's receive loop; a slow disk must not stall it."""
    import threading

    from acp.connection import StreamDirection, StreamEvent
    from nooa_coder.acp import tee as tee_module

    release = threading.Event()
    original = tee_module.FrameLog._write_loop

    def stalled_writer(self):
        release.wait()
        original(self)

    monkeypatch.setattr(tee_module.FrameLog, "_write_loop", stalled_writer)
    tee = tee_module.FrameLog(tmp_path / "tee.jsonl", max_pending=1)
    for index in range(3):
        tee(StreamEvent(StreamDirection.INCOMING, {"id": index}))
    assert tee.dropped == 2
    release.set()
    tee.close()
    records = [json.loads(line) for line in (tmp_path / "tee.jsonl").read_text().splitlines()]
    assert [record["frame"] for record in records] == [{"id": 0}]


def test_observer_close_is_idempotent(tmp_path):
    from nooa_coder.acp.tee import FrameLog

    tee = FrameLog(tmp_path / "tee.jsonl")
    tee.close()
    tee.close()


def test_observer_close_returns_when_the_writer_died_with_a_full_queue(tmp_path, caplog):
    """A log that became unwritable must not hang shutdown, and is reported once."""
    import threading

    from acp.connection import StreamDirection, StreamEvent
    from nooa_coder.acp.tee import FrameLog

    tee = FrameLog(tmp_path / "tee.jsonl", max_pending=2)
    tee._file.close()  # every write now fails: the writer thread stops
    tee(StreamEvent(StreamDirection.INCOMING, {"id": 0}))
    tee._thread.join(5)
    assert not tee._thread.is_alive()
    for index in range(1, 5):
        tee(StreamEvent(StreamDirection.INCOMING, {"id": index}))

    closer = threading.Thread(target=tee.close, daemon=True)
    closer.start()
    closer.join(5)
    assert not closer.is_alive()
    warnings = [r for r in caplog.records if "tee" in r.getMessage().lower()]
    assert len(warnings) == 1
