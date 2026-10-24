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
@pytest.mark.parametrize("native", [False, True])
def test_real_pty_offline_save_help_and_cancel(tmp_path, columns, no_color, cancel, native):
    import fcntl
    import pty
    import struct
    import termios

    target = tmp_path / "models.yaml"
    child = tmp_path / "pty_connect.py"
    child.write_text("""
import os
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS"] = "True"
import dotenv
dotenv.load_dotenv = lambda *a, **k: False
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
from nooa import llm_config
llm_config.llm_config_chain = lambda: []
if os.environ.get("PTY_NATIVE_REASONING") == "1":
    from nooa.unifiedllm import connect
    from nooa.unifiedllm.connect._server_details import listing_facts
    from tests.unifiedllm.connect.reasoning_fixture import MODEL_ID, NATIVE_MODEL
    async def details(endpoint, model, **kwargs):
        from copy import deepcopy
        model = deepcopy(NATIVE_MODEL)
        model["capabilities"]["structured_outputs"] = {"supported": True}
        facts = listing_facts(model)
        return connect.ModelDetails(model, facts, ({"source": endpoint + "/models",
                                                  "schema": "models", "id": model,
                                                  "facts": facts, "fields": {}},))
    async def catalogue():
        return []
    connect.server_details = details
    connect.catalogue = catalogue
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
        PTY_NATIVE_REASONING="1" if native else "0",
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
            "claude-opus-4-7" if native else "model",
            "--as",
            "local",
            "--endpoint",
            "https://api.anthropic.com/v1" if native else "https://server.test/v1",
            "--api-style",
            "anthropic" if native else "chat",
            "--api-key-env",
            "",
            *([] if native else ["--no-metadata"]),
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
    deadline = time.monotonic() + 120
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
            if native and phase == 0 and "7.Off" in compact:
                os.write(master, b"\x03" if cancel else b"\x1b[B\x1b[B\x1b[B\r")
                phase = 1 if cancel else -1
            if not native and phase == 0 and "Contextwindow(tokens)" in compact:
                os.write(master, b"262144\r262144\r")
                phase = -1
            if phase == -1 and "Wouldyouliketotestthesesettings?" in compact:
                os.write(master, b"\x03" if cancel else b"\x1bOP")
                phase = 1
            elif not cancel and phase == 1 and "F1Closehelp" in compact:
                os.write(master, b"\x1bOP\r")
                phase = 2
            elif not cancel and phase == 2 and "Writemodelentryto" in compact:
                os.write(master, b"y\r")
                phase = 3
        process.wait(timeout=60)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=60)
        os.close(master)
    transcript = output.decode(errors="replace")
    assert process.returncode == (1 if cancel else 0), transcript
    assert "SOCKET_ATTEMPTS=0" in transcript
    assert target.exists() is not cancel
    if not cancel:
        assert phase == 3, transcript
        assert "F1Closehelp" in "".join(re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", transcript).split())
        entry = yaml.safe_load(target.read_text())["models"]["local"]
        assert entry["media_inputs"] == []
        if native:
            assert (
                entry["structured_output_mode"] == "anthropic"
            )  # ordinary recommended UI, no expert flag
            assert entry["reasoning_level"] == "high"
            assert entry["reasoning_levels"]["high"] == {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
            }
            assert entry["max_tokens"] == 2000
            assert compact.index("Modeldetails") < compact.index("Choosehowmuchreasoning")
            assert compact.index("Choosehowmuchreasoning") < compact.index("Proposedconfiguration")
