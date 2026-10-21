# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real PTY keyboard smoke test: F1, save and Ctrl-C with no socket connections."""

import os
import re
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml


@pytest.mark.skipif(sys.platform != "linux", reason="Linux PTY smoke test")
@pytest.mark.parametrize("columns,no_color", [(40, True), (80, False)])
@pytest.mark.parametrize("cancel", [False, True])
def test_real_pty_offline_save_help_and_cancel(tmp_path, columns, no_color, cancel):
    import fcntl
    import pty
    import struct
    import termios

    target = tmp_path / "models.yaml"
    child = tmp_path / "pty_connect.py"
    child.write_text("""
import os
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
import socket
attempts = []
def blocked(self, address):
    attempts.append(address)
    raise AssertionError("Unexpected socket connection")
socket.socket.connect = blocked
socket.socket.connect_ex = blocked
import nooa.tracing
nooa.tracing._probe_failed = True
nooa.tracing._probe = lambda *a, **k: "down"
from nooa_cli.commands.connect import command
try:
    command.main()
finally:
    print("SOCKET_ATTEMPTS=" + str(len(attempts)), flush=True)
""")
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, columns, 0, 0))
    env = dict(
        os.environ,
        TERM="xterm-256color",
        NEMO_OO_USER_DIR=str(tmp_path / "user"),
        NEMO_OO_PROJECT_DIR=str(tmp_path / "project"),
    )
    if no_color:
        env["NO_COLOR"] = ""
    else:
        env.pop("NO_COLOR", None)
    root = Path(__file__).resolve().parents[3]
    env["PYTHONPATH"] = os.pathsep.join(
        str(path) for path in (root / "src", root, root / "packages/nooa-cli/src")
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(child),
            "model",
            "--as",
            "local",
            "--endpoint",
            "https://server.test/v1",
            "--api-style",
            "chat",
            "--api-key-env",
            "",
            "--no-metadata",
            "--no-probe",
            "--media-inputs",
            "",
            "--reply-tokens",
            "2000",
            "--output",
            str(target),
        ],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        start_new_session=True,
    )
    os.close(slave)
    output = bytearray()
    phase = 0
    deadline = time.monotonic() + 35
    try:
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], max(0, deadline - time.monotonic()))
            if not ready:
                break
            try:
                chunk = os.read(master, 65536)
            except OSError:
                break
            if not chunk:
                break
            output.extend(chunk)
            if b"\x1b[6n" in chunk:
                os.write(master, b"\x1b[1;1R")
            screen = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output.decode(errors="replace"))
            compact = "".join(re.sub(r"[│┌┐└┘─┬┴├┤┼]", "", screen).split())
            if phase == 0 and "Wouldyouliketotestthesesettings?" in compact:
                os.write(master, b"\x03" if cancel else b"\x1bOP")
                phase = 1
            elif not cancel and phase == 1 and "F1Closehelp" in compact:
                os.write(master, b"\x1bOP\r")
                phase = 2
            elif not cancel and phase == 2 and "Writemodelentryto" in compact:
                os.write(master, b"y\r")
                phase = 3
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        os.close(master)
    transcript = output.decode(errors="replace")
    assert process.returncode == (1 if cancel else 0), transcript
    assert "SOCKET_ATTEMPTS=0" in transcript
    assert target.exists() is not cancel
    if not cancel:
        assert phase == 3, transcript
        assert "F1Closehelp" in "".join(re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", transcript).split())
        assert yaml.safe_load(target.read_text())["models"]["local"]["media_inputs"] == []
