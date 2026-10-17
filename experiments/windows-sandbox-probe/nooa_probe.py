# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Child-side probe: can the container import ``nooa`` without the checkout?

Runs INSIDE the AppContainer with grants on the runtime root and the work
directory only. Every question it answers is one the backend depends on:

* does ``import nooa`` resolve at all, and to the runtime-root copy?
* does the import chain the worker actually uses (``runtime.sandbox``,
  ``strategies``) succeed?
* does ``importlib.metadata.version("nooa")`` still find the dist-info rather
  than falling back to ``0.0.0+unknown``?
* is the source checkout still unreadable from inside, i.e. did the copy
  actually replace the editable hook instead of adding a second path?

The last one is the whole point: an import that *works* but silently resolves
to ``E:\\labs-OO-Agents\\src`` would mean the isolation this experiment claims
does not exist.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

CHECKOUT = r"E:\labs-OO-Agents"
MODULES = [
    "nooa",
    "nooa.runtime.sandbox",
    "nooa.runtime.sandbox.config",
    "nooa.runtime.sandbox.worker",
    "nooa.runtime.sandbox.executor",
    "nooa.strategies",
]


def _probe_import(name: str) -> dict:
    try:
        mod = __import__(name, fromlist=["__name__"])
    except BaseException as exc:  # noqa: BLE001 - report, never crash the probe
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": True,
        "file": getattr(mod, "__file__", None),
        "inside_runtime_root": str(getattr(mod, "__file__", ""))
        .lower()
        .startswith(sys.prefix.lower()),
    }


def _probe_version() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return {"ok": True, "version": version("nooa")}
    except PackageNotFoundError:
        return {"ok": False, "error": "PackageNotFoundError -> version fallback"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _probe_checkout_denied() -> dict:
    """Reading the source package must fail; a success falsifies the whole run."""
    target = os.path.join(CHECKOUT, "src", "nooa", "__init__.py")
    try:
        with open(target, "rb") as fh:
            fh.read(64)
    except OSError as exc:
        return {
            "ok": True,
            "denied": True,
            "errno": exc.errno,
            "winerror": getattr(exc, "winerror", None),
            "msg": exc.strerror,
        }
    return {"ok": False, "denied": False, "error": f"{target} was READABLE"}


def _probe_sys_path() -> list[str]:
    return [p for p in sys.path if p]


def main() -> int:
    out = sys.argv[1]
    report: dict = {
        "pid": os.getpid(),
        "prefix": sys.prefix,
        "executable": sys.executable,
        "sys_path": _probe_sys_path(),
    }
    try:
        report["imports"] = {name: _probe_import(name) for name in MODULES}
    except BaseException:  # noqa: BLE001
        report["imports_error"] = traceback.format_exc()
    report["version"] = _probe_version()
    report["checkout"] = _probe_checkout_denied()

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
