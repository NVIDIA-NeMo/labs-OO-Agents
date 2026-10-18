# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure what a sandboxed cell can see of the parent's environment.

Two questions, both answered from inside the container rather than by reading
the launcher:

* **Leak** — does a variable that exists only in the parent reach the cell? A
  sentinel is set just before the worker starts; the cell reports whether it
  sees it, never what the value is.
* **Compatibility** — narrowing the environment must not break what worked
  before: Winsock, asyncio, ``tempfile`` and ``Path.home()``.

Only names, counts, booleans, paths and exception types are printed. No
environment value leaves the cell, and nothing is written inside the
repository: the workspace is a temp directory removed afterwards. Run with the
probe runtime root already built::

    uv run python experiments/windows-sandbox-probe/env_probe.py
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# The agent module has to be importable *by name* here so the launcher can
# resolve its directory and grant the container read access to it — the same
# arrangement the acceptance tests use.
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests" / "runtime" / "sandbox"))

ROOT = HERE / ".rt"

# Exists nowhere else on this machine. The cell reports only whether it sees it.
SENTINEL = "NOOA_ENV_PROBE_SECRET"

# Runs inside the container.
CELL = """
import asyncio, os, pathlib, socket, tempfile

def attempt(fn):
    try:
        return fn()
    except Exception as exc:
        return "error: " + type(exc).__name__

report = {
    "keys": sorted(os.environ),
    "sentinel": "__SENTINEL__" in os.environ,
    "tempfile.gettempdir()": attempt(tempfile.gettempdir),
    "Path.home()": attempt(lambda: str(pathlib.Path.home())),
    "socket.socket()": attempt(lambda: socket.socket().close() or "ok"),
    "asyncio loop": attempt(lambda: asyncio.new_event_loop().close() or "ok"),
}
report
""".replace("__SENTINEL__", SENTINEL)

# Set by the worker itself from argv, so present on both sides without having
# crossed. Counting them as inherited would report a leak that is not one.
WORKER_SET = {"TIKTOKEN_CACHE_DIR", "CUSTOM_TIKTOKEN_CACHE_DIR"}


def _launcher_allowlist() -> set[str] | None:
    """The launcher's allowlist, if this checkout has one (None before the fix)."""
    from nooa.runtime.sandbox import win_launcher

    names = getattr(win_launcher, "CONTAINER_ENV", None)
    return {name.upper() for name in names} if names is not None else None


def main() -> int:
    from win_agent_module import WinProbeAgent

    from nooa.runtime.sandbox.config import SandboxConfig
    from nooa.runtime.sandbox.executor import SandboxedExecutor

    if not (ROOT / "python.exe").exists():
        print(f"no runtime root at {ROOT}; build it with _mkroot.py first")
        return 2

    # The capability probe resolves the root from the environment rather than
    # from ``SandboxConfig.runtime_root``, so both are set.
    os.environ.setdefault("NOOA_SANDBOX_RUNTIME", str(ROOT))
    os.environ[SENTINEL] = "set-in-the-parent-only"
    parent_keys = {key.upper() for key in os.environ}

    workspace = tempfile.mkdtemp(prefix="nooa-env-probe-")
    try:
        config = SandboxConfig(
            filesystem=True,
            network=False,
            workspace=workspace,
            start_method="spawn",
            backend="appcontainer",
            require=True,
            runtime_root=str(ROOT),
        )
        executor = SandboxedExecutor(
            WinProbeAgent(), config, cell_timeout=180, framework_builtins={}
        )
        try:
            result = asyncio.run(executor.run_cell(CELL, execution_count=1))
        finally:
            asyncio.run(executor.aclose())
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    if result.error is not None:
        print("cell failed:", result.error)
        return 1

    report = dict(result.returned_value)
    child_keys = {key.upper() for key in report.pop("keys")}
    print(f"{'parent variables':<26}: {len(parent_keys)}")
    print(f"{'visible in the container':<26}: {len(child_keys)}")
    print(f"{'sentinel reached the cell':<26}: {report.pop('sentinel')}")

    allowlist = _launcher_allowlist()
    if allowlist is not None:
        unexpected = sorted((child_keys & parent_keys) - allowlist - WORKER_SET)
        print(f"{'crossed, not allowlisted':<26}: {unexpected or 'none'}")
        print(f"{'container variables':<26}: {sorted(child_keys)}")

    for label, value in report.items():
        print(f"{label:<26}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
