# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test SIGINT handling in web terminal."""

import signal
import socket
import subprocess
import sys
import time


def _unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(port: int, *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"web terminal did not listen on port {port} within {timeout}s")


def test_sigint_cleanup():
    """Test that ^C exits cleanly without traceback spam."""
    port = _unused_local_port()

    proc = subprocess.Popen(
        [sys.executable, "-m", "nooa_cli", "term", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        _wait_for_port(port)

        # Send SIGINT (simulate ^C) only after uvicorn is serving. Sending it
        # during slow CI startup can hit Python's default handler before the
        # command installs its async shutdown hook, producing flaky exit -2.
        proc.send_signal(signal.SIGINT)

        stdout, stderr = proc.communicate(timeout=8)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert proc.returncode == 0, f"Expected exit code 0, got {proc.returncode}"
    assert "Traceback" not in stderr, f"Found traceback in stderr: {stderr}"
    assert "CancelledError" not in stderr, f"Found CancelledError in stderr: {stderr}"

    print("✓ Terminal exits cleanly on SIGINT")


if __name__ == "__main__":
    test_sigint_cleanup()
