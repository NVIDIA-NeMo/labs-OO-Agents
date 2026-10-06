# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The router does not load the adapter or the agent stack behind it."""

import json
import subprocess
import sys

import pytest


def _loaded_after_router_import(names: list[str]) -> list[str]:
    """Which of ``names`` are in ``sys.modules`` after a fresh ``import nooa_coder.acp.router``."""
    code = (
        "import json, sys\n"
        "import nooa_coder.acp.router\n"
        f"print(json.dumps([name for name in {names!r} if name in sys.modules]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_router_does_not_import_the_adapter_or_the_coding_agent():
    assert (
        _loaded_after_router_import(
            [
                "nooa_coder.acp.server",
                "nooa_coder.acp.event_bridge",
                "nooa_coder.coding",
                "nooa_coder.workspace",
                "nooa.mcp",
            ]
        )
        == []
    )


@pytest.mark.xfail(
    strict=True,
    reason="`import nooa` (src/nooa/__init__.py) imports nooa.strategies eagerly, and the "
    "router needs nooa_coder.session.store, which imports nooa; a core change",
)
def test_router_does_not_import_codeact_or_litellm():
    assert _loaded_after_router_import(["nooa.strategies.codeact", "litellm"]) == []
