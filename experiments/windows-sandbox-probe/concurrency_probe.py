# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Do two concurrent sandbox sessions share one AppContainer identity?

The backend used to derive one container profile from a module constant
(``DEFAULT_CONTAINER_NAME``) and never pass a per-session name, so every session
on the host ran under the same AppContainer SID and appended to the same
``acl-journal.jsonl``. That produced two failures:

1. **Grant revocation across sessions.** ``prepare_root`` called
   ``AclJournal.recover()`` on every start, which revoked every journaled grant.
   B's startup revoked A's live grants.
2. **Journal ownership.** ``ContainerProcess.release()`` called
   ``journal.clear()``, which unlinked the journal file, so closing A destroyed
   B's crash-recovery record.

Each session now mints its own profile name and owns its own journal, and
recovery only reclaims journals whose lock no live process holds. This probe
re-measures the same three questions against real processes, and reports
``SHARED IDENTITY`` if any of them regress.

    python experiments\\windows-sandbox-probe\\concurrency_probe.py

Needs a built runtime root (``NOOA_SANDBOX_RUNTIME`` or the default cache).
Exits non-zero if the sessions turn out to share an identity. It exited non-zero
when the defect was present; a clean run is the regression check.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_TESTS = _REPO / "tests" / "runtime" / "sandbox"


def _runtime_root() -> Path:
    explicit = os.environ.get("NOOA_SANDBOX_RUNTIME")
    if explicit:
        root = Path(explicit)
    else:
        from nooa.runtime.sandbox.win_runtime import default_runtime_root

        root = default_runtime_root()
    if not (root / "python.exe").exists():
        raise SystemExit(
            f"no runtime root at {root}. Build one with "
            "`python -m nooa.runtime.sandbox.win_runtime --build`, or set "
            "NOOA_SANDBOX_RUNTIME."
        )
    return root


def _config(workspace: Path, root):
    from nooa.runtime.sandbox.config import SandboxConfig

    # ``main`` resolves the root once and passes the RuntimeRoot around; the
    # config field wants the directory itself, not its repr.
    root_path = getattr(root, "root", root)
    return SandboxConfig(
        filesystem=True,
        network=False,
        workspace=str(workspace),
        start_method="spawn",
        backend="appcontainer",
        require=True,
        runtime_root=str(root_path),
    )


def _new_session(workspace: Path, root: Path):
    from nooa.runtime.sandbox.executor import SandboxedExecutor

    sys.path.insert(0, str(_TESTS))
    from win_agent_module import WinProbeAgent

    return SandboxedExecutor(
        WinProbeAgent(),
        _config(workspace, root),
        cell_timeout=90,
        framework_builtins={},
    )


def _run(executor, code: str, count: int = 1):
    return asyncio.run(executor.run_cell(code, execution_count=count))


def _alive(executor) -> bool:
    proc = executor._proc
    return proc is not None and proc.is_alive()


def _sid_on(path: Path, sid_text: str) -> bool:
    from nooa.runtime.sandbox import win_api

    try:
        return sid_text in win_api.read_dacl(str(path))
    except OSError as exc:
        return f"<unreadable: {exc}>"


def _journal(path: Path) -> str:
    if not path.exists():
        return "<absent>"
    try:
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError as exc:
        return f"<unreadable: {exc}>"
    paths = []
    for line in lines:
        try:
            paths.append(json.loads(line).get("path", "?"))
        except ValueError:
            paths.append("<malformed>")
    return f"{len(lines)} entries: {paths}"


def main() -> int:
    if os.name != "nt":
        raise SystemExit("this probe measures Windows AppContainer behaviour")

    from nooa.runtime.sandbox.win_api import container_sid, sid_string
    from nooa.runtime.sandbox.win_runtime import resolve_root

    root = resolve_root(_runtime_root())

    print(f"runtime root : {root.root}")
    print(f"journals     : {[p.name for p in root.journals()]}")
    print()

    base = Path(tempfile.mkdtemp(prefix="nooa-concurrency-"))
    ws_a = base / "ws-a"
    ws_b = base / "ws-b"
    ws_a.mkdir()
    ws_b.mkdir()

    a = b = None
    name_a = name_b = ""
    journal_a = journal_b = None
    sid_text = sid_b_text = ""
    verdicts: dict[str, object] = {}
    try:
        # --- session A ------------------------------------------------------
        print("[1] starting session A ...")
        a = _new_session(ws_a, root)
        result = _run(a, "1 + 1")
        name_a = a._proc.container_name
        sid_text = sid_string(container_sid(name_a))
        journal_a = root.journal_for(name_a)
        print(f"    A cell ok      : {result.error is None}  ({result.error})")
        print(f"    A worker alive : {_alive(a)}")
        print(f"    A container    : {name_a}  ({sid_text})")
        print(f"    A ws has SID   : {_sid_on(ws_a, sid_text)}")
        print(f"    A journal      : {_journal(journal_a)}")
        verdicts["a_started"] = result.error is None and _sid_on(ws_a, sid_text) is True

        # --- session B, while A is live -------------------------------------
        print()
        print("[2] starting session B while A is still live ...")
        b = _new_session(ws_b, root)
        result_b = _run(b, "1 + 1")
        name_b = b._proc.container_name
        journal_b = root.journal_for(name_b)
        sid_b_text = sid_string(container_sid(name_b))
        print(f"    B cell ok      : {result_b.error is None}  ({result_b.error})")
        print(f"    B worker alive : {_alive(b)}")
        print(f"    B container    : {name_b}  ({sid_b_text})")
        print(f"    B journal      : {_journal(journal_b)}")
        verdicts["distinct_identity"] = name_a != name_b

        # The measurement: did B's startup touch A's grant?
        print()
        print("[3] measuring A after B started ...")
        still = _sid_on(ws_a, sid_text)
        print(f"    A worker alive     : {_alive(a)}")
        print(f"    A ws STILL has SID : {still}")
        print(f"    A journal          : {_journal(journal_a)}")
        verdicts["a_grant_survived_b_start"] = still

        # A is still running, so a failing write is the container's own DACL
        # check, not a dead process being restarted.
        target = ws_a / "after-b.txt"
        code = (
            "import pathlib\n"
            f"p = pathlib.Path({str(target)!r})\n"
            "p.write_text('written by A after B started')\n"
            "'wrote'\n"
        )
        result = _run(a, code, count=2)
        print(f"    A can write its ws : {result.error is None}  ({result.error})")
        print(f"    file exists on disk: {target.exists()}")
        verdicts["a_lost_write_access"] = result.error is not None

        # --- does closing A destroy B's journal? ----------------------------
        print()
        print("[4] closing A while B is still live ...")
        print(f"    A journal before close : {_journal(journal_a)}")
        asyncio.run(a.aclose())
        a = None
        print(f"    A journal after  close : {_journal(journal_a)}")
        print(f"    B journal after  close : {_journal(journal_b)}")
        print(f"    B worker alive         : {_alive(b)}")
        verdicts["journal_survived_a_close"] = journal_b.exists()

        result = _run(b, "2 + 2", count=2)
        print(f"    B cell after A.close   : {result.error is None}  ({result.error})")
        print(f"    B ws still has SID     : {_sid_on(ws_b, sid_b_text)}")
        verdicts["b_grant_survived_a_close"] = _sid_on(ws_b, sid_b_text) is True
    finally:
        for label, executor in (("A", a), ("B", b)):
            if executor is None:
                continue
            try:
                asyncio.run(executor.aclose())
            except Exception:  # noqa: BLE001 - teardown must not mask the verdict
                print(f"    {label}.aclose failed:")
                traceback.print_exc()

    print()
    print("=== verdict ===")
    for key, value in verdicts.items():
        print(f"  {key:32s} {value}")

    shared = (
        verdicts.get("distinct_identity") is False
        or verdicts.get("a_grant_survived_b_start") is False
        or verdicts.get("a_lost_write_access") is True
        or verdicts.get("journal_survived_a_close") is False
        or verdicts.get("b_grant_survived_a_close") is False
    )
    print()
    if shared:
        print("SHARED IDENTITY: concurrent sessions interfere. Not production-safe.")
    else:
        print("INDEPENDENT: the two sessions neither share a SID nor touch each other's grants.")
    return 1 if shared else 0


if __name__ == "__main__":
    raise SystemExit(main())
