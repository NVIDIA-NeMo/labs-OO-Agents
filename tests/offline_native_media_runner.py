# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run offline gates with dotenv disabled before framework import.

Invoke using an empty environment, an isolated HOME, and this checkout's source
PYTHONPATH. Only loopback TCP (local broker tests) and Unix sockets are allowed;
provider HTTP must use MockTransport. No dependency installation is performed.
"""

import os
import socket

import dotenv

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["NOOA_DISABLE_TRACING"] = "1"
# Legacy offline tests may intentionally leave api_key_env blank. Give the SDKs
# synthetic defaults, never inherited credentials or dotenv-loaded secrets.
os.environ["OPENAI_API_KEY"] = "offline-test-only"
os.environ["ANTHROPIC_API_KEY"] = "offline-test-only"
dotenv.load_dotenv = lambda *a, **k: False
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def guarded_connect(sock, address):
    if sock.family == socket.AF_UNIX or (
        isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1", "localhost"}
    ):
        return _real_connect(sock, address)
    raise AssertionError("Offline verification forbids real provider socket connections")


def guarded_connect_ex(sock, address):
    if sock.family == socket.AF_UNIX or (
        isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1", "localhost"}
    ):
        return _real_connect_ex(sock, address)
    raise AssertionError("Offline verification forbids real provider socket connections")


socket.socket.connect = guarded_connect
socket.socket.connect_ex = guarded_connect_ex
if __name__ == "__main__":
    import sys

    import pytest

    raise SystemExit(pytest.main(sys.argv[1:]))
