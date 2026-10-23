# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real guided TTY: blank catalogue → literal limits → details → interface/media."""

import os
import re
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

CHILD = """
import os, socket
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS"] = "True"
attempts = []
def blocked(*a, **k):
    attempts.append(1)
    raise AssertionError("Unexpected live socket")
socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.create_connection = blocked
import dotenv
dotenv.load_dotenv = lambda *a, **k: False
from nooa import llm_config
llm_config.llm_config_chain = lambda: []
from nooa.unifiedllm import connect
async def details(endpoint, model, **kwargs):
    return connect.ModelDetails(model, {"id": model})
async def catalogue():
    return [{"id": "gateway/wired-model", "context_length": 128000}]
connect.server_details = details
connect.catalogue = catalogue
from nooa_cli.commands.connect import command
try:
    command.main()
finally:
    print("SOCKET_ATTEMPTS=" + str(len(attempts)), flush=True)
"""


@pytest.mark.skipif(sys.platform != "linux", reason="Linux PTY contract")
@pytest.mark.parametrize("cancel", [None, "context", "input"])
def test_real_guided_blank_catalogue_order_and_early_cancel(tmp_path, cancel):
    import fcntl
    import pty
    import struct
    import termios

    target = tmp_path / "models.yaml"
    root = Path(__file__).resolve().parents[3]
    env = dict(
        os.environ,
        TERM="xterm-256color",
        NO_COLOR="",
        PYTHONDONTWRITEBYTECODE="1",
        NEMO_OO_USER_DIR=str(tmp_path / "user"),
        NEMO_OO_PROJECT_DIR=str(tmp_path / "project"),
    )
    env["PYTHONPATH"] = os.pathsep.join(
        str(p)
        for p in (
            root / "src",
            root / "packages/nooa-cli/src",
            root,
        )
    )
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 80, 0, 0))
    process = subprocess.Popen(
        [
            sys.executable,
            "-B",
            "-c",
            CHILD,
            "wire/model",
            "--as",
            "local",
            "--endpoint",
            "https://api.test/v1",
            "--api-key-env",
            "",
            "--no-probe",
            "--reply-tokens",
            "1000",
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
    steps = [
        ("Whichoftheseisyourmodel?(Enterfornone)", b"\r"),
        ("Contextwindow(tokens)", b"\x03" if cancel == "context" else b"128000\r"),
        ("Inputtokenlimit(tokens)", b"\x03" if cancel == "input" else b"64000\r"),
        ("Requestinterface", b"\x1b[B\r"),
        ("SpaceToggle", b"\r"),
        ("Wouldyouliketotestthesesettings?", b"\r"),
        ("Writemodelentryto", b"y\r"),
    ]
    phase, output = 0, bytearray()
    deadline = time.monotonic() + 30
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
            if phase < len(steps) and steps[phase][0] in compact:
                os.write(master, steps[phase][1])
                phase += 1
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        os.close(master)
    transcript = output.decode(errors="replace")
    assert process.returncode == (1 if cancel else 0), transcript
    assert "SOCKET_ATTEMPTS=0" in transcript
    assert target.exists() is (cancel is None)
    if cancel:
        assert "Requestinterface" not in compact and "Mediasetup" not in compact
        assert "Modeldetails" not in compact
    else:
        labels = [
            "Contextwindow(tokens)",
            "Inputtokenlimit(tokens)",
            "Modeldetails",
            "Requestinterface",
            "Mediasetup",
            "Wouldyouliketotestthesesettings?",
        ]
        positions = [compact.index(label) for label in labels]
        assert positions == sorted(positions), transcript
        assert compact.count("ModeldetailsModel:") == 1
        initial = compact[positions[2] : positions[3]]
        assert initial.count("Contextwindow(declared)") == 1
        assert initial.count("Inputtokenlimit(declared)") == 1
        assert "Source:Yoursettings" in initial
        assert "Effectivecontextbound:64,000tokens" in initial
        assert yaml.safe_load(target.read_text())["models"]["local"]["context_window"] == 64000
