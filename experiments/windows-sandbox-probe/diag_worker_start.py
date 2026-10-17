# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnose why the container worker exits before the ready handshake.

The launcher creates the process with no inherited std handles, so a worker that
dies during import writes its traceback into the void. This probe reproduces the
launch manually and records what the parent can actually observe: exit code,
and anything the child manages to write into the granted work directory.

Runs the same argv the launcher builds, with a bogus pipe handle, so the only
difference from a real launch is the handle value.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import win32 as w  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNTIME_ROOT = HERE / ".rt"
WORKDIR = HERE / ".work"
CONTAINER_NAME = "nooa-m0-probe"


def main() -> int:
    # runner.py deletes the profile in its finally block, and CreateProcessW
    # against a container SID with no profile fails ERROR_FILE_NOT_FOUND — which
    # looks exactly like a bad interpreter path. Recreate it first.
    w.create_profile(CONTAINER_NAME)
    app_sid = w.container_sid(CONTAINER_NAME)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    w.grant(str(RUNTIME_ROOT), app_sid, w.FILE_GENERIC_READ | w.FILE_GENERIC_EXECUTE)
    w.grant(str(WORKDIR), app_sid, w.FILE_ALL_ACCESS)

    report: dict = {}
    variants = {
        # 1. Import only: proves the package and its dependencies load.
        "import_worker": [
            str(RUNTIME_ROOT / "python.exe"),
            "-I",
            "-c",
            (
                "import json,traceback;"
                "p=r'" + str(WORKDIR / "import_report.json") + "';"
                "\nimport sys\n"
                "try:\n"
                "    import nooa.runtime.sandbox.win_worker as m\n"
                "    out={'ok':True,'file':m.__file__}\n"
                "except BaseException:\n"
                "    out={'ok':False,'tb':traceback.format_exc()}\n"
                "open(p,'w').write(json.dumps(out,indent=2))\n"
            ),
        ],
        # 2. The real entrypoint with a definitely-bad handle. It must at least
        #    reach the pipe-adoption step, which reports a usage/OS error.
        "entry_bad_handle": [
            str(RUNTIME_ROOT / "python.exe"),
            "-I",
            "-m",
            "nooa.runtime.sandbox.win_worker",
            "999999",
            CONTAINER_NAME,
        ],
    }

    for name, argv in variants.items():
        out_file = WORKDIR / f"{name}.json"
        out_file.unlink(missing_ok=True)
        job = w.Job()
        t0 = time.perf_counter()
        try:
            child = w.spawn_in_container(argv, container_name=CONTAINER_NAME, cwd=str(WORKDIR))
        except OSError as exc:
            report[name] = {"spawn_error": str(exc)}
            job.close()
            continue
        job.assign(child.pi.hProcess)
        child.resume()
        code = child.wait(timeout_ms=60_000)
        elapsed = time.perf_counter() - t0
        child.close()
        job.terminate(0)
        job.close()
        report[name] = {"exit_code": code, "wall_s": round(elapsed, 3)}
        if out_file.exists():
            try:
                report[name]["file"] = json.loads(out_file.read_text(encoding="utf-8"))
            except ValueError:
                report[name]["file"] = "unparseable"

    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
