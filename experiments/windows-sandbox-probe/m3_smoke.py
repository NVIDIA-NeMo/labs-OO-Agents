# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""M3 smoke: run one CodeAct cell through the real AppContainer backend.

This is not a unit test — it drives the shipped ``SandboxedExecutor`` against
the private runtime root built by ``_mkroot.py``, so it exercises the path a
user would actually take: config -> capabilities probe -> ACL grants -> suspended
CreateProcessW -> job assignment -> resume -> ready handshake -> blueprint
rebuild -> cell execution -> teardown.

Run with the runtime root pointed at the probe's copy:

    set NOOA_SANDBOX_RUNTIME=...\\experiments\\windows-sandbox-probe\\.rt
    python experiments\\windows-sandbox-probe\\m3_smoke.py

Prints one JSON object. A non-zero exit means the backend is not usable, and the
error text says which stage failed.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for path in (os.path.join(REPO, "src"), HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

# The agent class must live in an importable module, not in ``__main__``: the
# spawned worker rebuilds it from ``module:qualname``, and the child's
# ``__main__`` is ``win_worker``.
from m3_agent import ProbeAgent  # noqa: E402

from nooa.runtime.sandbox.config import SandboxConfig  # noqa: E402
from nooa.runtime.sandbox.executor import SandboxedExecutor  # noqa: E402


async def main() -> int:
    workspace = os.path.join(HERE, ".m3-workspace")
    os.makedirs(workspace, exist_ok=True)
    report: dict = {"runtime_root": os.environ.get("NOOA_SANDBOX_RUNTIME"), "stages": {}}

    config = SandboxConfig(
        filesystem=True,
        network=False,
        workspace=workspace,
        max_memory_mb=256,
        start_method="spawn",
        backend="appcontainer",
        require=True,
    )
    agent = ProbeAgent()
    executor = None
    try:
        executor = SandboxedExecutor(
            agent,
            config,
            cell_timeout=30,
            framework_builtins={},
        )
        report["stages"]["construct"] = "ok"
        result = await executor.run_cell("value = self.answer()\nvalue", execution_count=1)
        report["stages"]["cell"] = {
            "error": repr(result.error) if result.error else None,
            "stdout": result.stdout,
            "returned": getattr(result, "returned_value", None),
        }
        result2 = await executor.run_cell("self.echo('hi')", execution_count=2)
        report["stages"]["broker_cell"] = {
            "error": repr(result2.error) if result2.error else None,
            "returned": getattr(result2, "returned_value", None),
        }
        # Denial check: the checkout must be unreadable from inside.
        result3 = await executor.run_cell(
            "open(r'E:\\labs-OO-Agents\\src\\nooa\\__init__.py','rb').read(8)", execution_count=3
        )
        report["stages"]["checkout_denied"] = {
            "error_type": type(result3.error).__name__ if result3.error else None,
            "error": str(result3.error)[:200] if result3.error else None,
        }
        report["ok"] = not (result.error or result2.error) and result3.error is not None
    except BaseException as exc:  # noqa: BLE001 - the report is the output
        report["ok"] = False
        report["fatal"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        if executor is not None:
            try:
                await executor.aclose()
                report["stages"]["close"] = "ok"
            except BaseException as exc:  # noqa: BLE001
                report["stages"]["close"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
