# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep protocol subprocesses on the checkout and configuration under test."""

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _protocol_subprocess_environment(monkeypatch):
    """ACP trims environment variables, including pytest's source/config paths."""
    import acp.transports

    original = acp.transports.default_environment
    root = Path(__file__).resolve().parents[3]
    sources = [root / "src"] + [
        root / "packages" / package / "src"
        for package in ("nooa-cli", "nooa-acp", "nooa-memory", "nooa-bench")
    ]

    def environment():
        values = original()
        values["PYTHONPATH"] = os.pathsep.join(str(path) for path in sources)
        for name in (
            "NEMO_OO_USER_DIR",
            "NEMO_OO_PROJECT_DIR",
            "NEMO_OO_SETTINGS",
            "NOOA_SESSIONS_DIR",
            "NOOA_ACP_MCP_TRACE",
        ):
            if name in os.environ:
                values[name] = os.environ[name]
        return values

    monkeypatch.setattr(acp.transports, "default_environment", environment)
