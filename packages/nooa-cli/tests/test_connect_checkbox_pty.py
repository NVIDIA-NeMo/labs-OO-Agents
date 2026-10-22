# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real terminal checkbox keyboard contracts, all 16 subsets, with no sockets."""

import json
import os
import re
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

KINDS = ["image", "audio", "video", "pdf"]
CHILD = """
import os, sys, json, socket
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS"] = "True"
attempts = []
def blocked(*a, **k):
    attempts.append(1)
    raise AssertionError("Unexpected socket connection")
socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.create_connection = blocked
import dotenv
dotenv.load_dotenv = lambda *a, **k: False
import click
from nooa_cli.commands._connect_prompts import choose_media_inputs
try:
    result = choose_media_inputs(json.loads(sys.argv[1]), available=json.loads(sys.argv[2]))
    print("CHECKBOX_RESULT=" + json.dumps(result), flush=True)
except click.Abort:
    print("CHECKBOX_ABORT", flush=True)
finally:
    print("SOCKET_ATTEMPTS=" + str(len(attempts)), flush=True)
"""


def clean(text):
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)


def run_terminal(
    tmp_path, *, keys, current=(), available=KINDS, columns=40, no_color=True, help=False
):
    import fcntl
    import pty
    import struct
    import termios

    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, columns, 0, 0))
    root = Path(__file__).resolve().parents[3]
    env = dict(
        os.environ,
        TERM="xterm-256color",
        PYTHONDONTWRITEBYTECODE="1",
        NEMO_OO_USER_DIR=str(tmp_path / "user"),
        NEMO_OO_PROJECT_DIR=str(tmp_path / "project"),
        PYTHON_DOTENV_DISABLED="1",
        LITELLM_LOCAL_MODEL_COST_MAP="True",
        LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS="True",
    )
    env["PYTHONPATH"] = os.pathsep.join(
        str(p) for p in (root / "src", root / "packages/nooa-cli/src", root)
    )
    if no_color:
        env["NO_COLOR"] = ""
    else:
        env.pop("NO_COLOR", None)
    proc = subprocess.Popen(
        [sys.executable, "-B", "-c", CHILD, json.dumps(list(current)), json.dumps(list(available))],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        start_new_session=True,
    )
    os.close(slave)
    output = bytearray()
    phase = 0
    deadline = time.monotonic() + 15
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
            screen = clean(output.decode(errors="replace"))
            compact = "".join(screen.split())
            if phase == 0 and "SpaceToggle" in compact:
                os.write(master, b"\x1bOP" if help else keys.encode())
                phase = 1
            elif help and phase == 1 and "Agentworkflow:nooa-connect" in compact:
                os.write(master, b"\x1bOP" + keys.encode())
                phase = 2
        if proc.poll() is None:
            raise AssertionError(
                "Terminal did not finish; phase="
                + str(phase)
                + "\n"
                + clean(output.decode(errors="replace"))
            )
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        os.close(master)
    transcript = clean(output.decode(errors="replace"))
    assert proc.returncode == 0, transcript
    assert "SOCKET_ATTEMPTS=0" in transcript
    assert phase == (2 if help else 1), transcript
    if help:
        assert "Agentworkflow:nooa-connect" in "".join(transcript.split())
    if "\x03" in keys:
        assert "CHECKBOX_ABORT" in transcript
        return None, transcript
    match = re.search(r"CHECKBOX_RESULT=(\[[^\r\n]*\])", transcript)
    assert match, transcript
    return json.loads(match[1]), transcript


@pytest.mark.skipif(sys.platform != "linux", reason="Linux PTY contract")
@pytest.mark.parametrize("mask", range(16))
def test_real_tty_all_16_subsets_space_arrows_enter_and_help(tmp_path, mask):
    # Arrow movement alone must not toggle a row. Each bit is toggled explicitly.
    keys = (
        "".join(
            (" " if mask & (1 << index) else "") + ("\x1b[B" if index < 3 else "")
            for index in range(4)
        )
        + "\r"
    )
    result, transcript = run_terminal(
        tmp_path, keys=keys, columns=40 if mask % 2 else 80, no_color=bool(mask % 2), help=True
    )
    assert result == [kind for index, kind in enumerate(KINDS) if mask & (1 << index)]
    assert all(label in transcript for label in ("Images", "Audio", "Video", "PDFs"))
    assert "None checked = text only" in transcript


@pytest.mark.skipif(sys.platform != "linux", reason="Linux PTY contract")
@pytest.mark.parametrize(
    "keys,expected",
    [
        ("\r", ["pdf", "image"]),
        (" \x1b[B \x1b[B \r", ["pdf", "audio", "video"]),
        (" \x1b[B\x1b[B\x1b[B \r", []),
        ("\x1b[B\x1b[A\r", ["pdf", "image"]),
        ("\x03", None),
    ],
)
def test_real_tty_preserves_checked_order_defaults_off_and_cancel(tmp_path, keys, expected):
    result, _ = run_terminal(
        tmp_path, keys="\x1b[A" * 3 + keys, current=["pdf", "image"], help=True
    )
    assert result == expected


@pytest.mark.skipif(sys.platform != "linux", reason="Linux PTY contract")
def test_real_tty_decision_available_rows_are_image_only(tmp_path):
    result, transcript = run_terminal(tmp_path, keys=" \r", available=["image"])
    assert result == ["image"]
    assert "Images" in transcript
    assert all(label not in transcript for label in ("Audio", "Video", "PDFs"))
