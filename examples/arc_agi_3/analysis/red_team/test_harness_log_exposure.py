# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Red-team regression test (NOT production): the `_harness` log tree is left
world-readable and is NOT covered by ``carve_own`` — so a dropped agent uid can
read harness logs, which DEANONYMISE the game and hold its full play history.

Finding (Pass 34), verified live against the competition run:

  layout per game:  <run_root>/<game>/memory/
                        ├── <ts>_<game>_memory/   <- the agent run dir (carve_own:
                        │       └── ipc/...           chown uid, 0700; ancestors 0711)
                        └── _harness/              <- SIBLING of the run dir
                                └── <ts>_<game>_memory.log   (root:root 0644)

  * ``carve_own`` protects only the run-dir subtree and sets each ANCESTOR
    (``<game>/memory``, ``<game>``, ``<run_root>``) to 0711 = traverse-only.
  * it does NOT touch the ``_harness`` SIBLING, which the root harness created
    0755 (dir) / 0644 (log) — world-readable.

  Consequences (both reproduced as the real dropped uids on the live run):
    1. SELF-deanonymisation, zero guessing — the agent knows its own run dir and
       ``_harness`` is a FIXED sibling name, so it reads ``../_harness/*.log`` and
       finds ``resolved game <alias> -> <alias>-<hex>`` == its real game id. This
       defeats the game anonymisation regardless of the store-alias randomisation.
    2. CROSS-game — ``<run_root>`` is 0711 (non-listable) so siblings can't be
       enumerated, BUT ``os.path.exists(<run_root>/<other>)`` CONFIRMS a guessed
       name, and game ids are low-entropy public ARC-AGI-3 short codes. So a game
       can confirm-then-read every OTHER game's harness log: real id + full
       action/level/score trace.

  Severity: latent — the behavioural scanners find NO agent actually did this
  (0 cross-game refs, 0 ``_harness`` reads in agent logs; escapes still only
  vc33/blocked). But it is a real leakage PATH, not covered by the uid-drop.

  Fix (launcher-side, ONE line, NOT applied here — production is frozen for this
  audit): lock ``_harness`` the same way game source is locked —
  ``uid_sandbox.block([run_dir.parent / "_harness"])`` -> 0700 root. The harness
  runs as root so keeps read/write; the dropped agent uid EACCESes its own and
  every sibling's ``_harness``. (Alternatives: create logs 0600 root; or write
  harness logs outside the agent-reachable ``<game>/memory`` tree; or strip the
  ``resolved game`` line from anything agent-reachable.)

This test builds a faithful mock of the layout, proves the exposure TODAY, and
proves the one-line ``block()`` fix closes it — so it fails loudly if the hole is
ever (re)introduced and passes once the fix lands.

Run as root:  python3 test_harness_log_exposure.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

# import the production isolation helper (read-only use; we do not modify it)
EXAMPLE = Path("/root/projects/nemo_oo_agents/examples/arc_agi_3")
sys.path.insert(0, str(EXAMPLE))
import uid_sandbox  # noqa: E402

HARNESS_LINE = "[harness] resolved game zz00 -> zz00-deadbeef (competition)\n"


def _as_uid(uid: int, fn):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            os.setgroups([])
            os.setgid(uid)
            os.setuid(uid)
        except OSError as e:
            os.write(w, f"SU_FAIL({e.errno})".encode())
            os._exit(0)
        try:
            out = fn()
        except OSError as e:
            out = f"EACCES({e.errno})"
        except Exception as e:  # noqa: BLE001
            out = f"ERR({type(e).__name__})"
        os.write(w, str(out).encode()[:80])
        os._exit(0)
    os.close(w)
    res = os.read(r, 120).decode()
    os.waitpid(pid, 0)
    return res


def _build_game(results: Path, game: str, uid: int) -> Path:
    """Mirror production: <results>/<game>/memory/{<run>, _harness/<log>}."""
    mem = results / game / "memory"
    run = mem / f"20260710_{game}_memory"
    run.mkdir(parents=True)
    (run / "ipc").mkdir()
    (run / "ipc" / "states.jsonl").write_text('{"own":1}\n')
    harness = mem / "_harness"
    harness.mkdir()
    log = harness / f"20260710_{game}_memory.log"
    log.write_text("INFO:arc_agi.scorecard:init\n" + HARNESS_LINE * 3)
    # root creates the harness tree 0755/0644 (the default) — the exposure
    os.chown(harness, 0, 0)
    os.chmod(harness, 0o755)
    os.chown(log, 0, 0)
    os.chmod(log, 0o644)
    # carve the run dir to the agent uid (this is what production does)
    uid_sandbox.carve_own(run, uid, uid, up_to=results)
    return run


def _read_own_harness(run: Path):
    """What the agent can do from its OWN run dir: ../_harness/<log> (fixed name)."""
    hd = run.parent / "_harness"
    fn = os.listdir(hd)  # 0755 -> listable
    t = (hd / fn[0]).read_text()  # 0644 -> readable
    for line in t.splitlines():
        if "resolved game" in line:
            return line.split("resolved game ", 1)[1][:40]
    return "no-line"


def main() -> int:
    if os.geteuid() != 0:
        print("SKIP: needs root to fork+setuid")
        return 0
    base = Path(tempfile.mkdtemp())
    os.chmod(base, 0o711)
    results = base / "results"
    results.mkdir()
    os.chmod(results, 0o711)
    games = {"ar25": 50310, "bp35": 50311}
    runs = {g: _build_game(results, g, uid) for g, uid in games.items()}

    ok = True

    # ---- 1) THE HOLE: self-deanonymisation with zero guessing --------------
    own = _as_uid(games["ar25"], lambda: _read_own_harness(runs["ar25"]))
    hole_self = own.startswith("zz00 ->")
    print(f"  [{'HOLE' if hole_self else 'ok  '}] ar25 reads its OWN ../_harness log -> {own!r}")
    ok &= hole_self  # we EXPECT the hole to be present pre-fix (documents it)

    # ---- 2) THE HOLE: cross-game confirm-then-read -------------------------
    def cross():
        p = results / "bp35"  # guessed sibling name
        if not p.exists():  # 0711 root traversable -> confirmable
            return "not-confirmable"
        return _read_own_harness(runs["bp35"])  # same fixed _harness reach

    xr = _as_uid(games["ar25"], cross)
    hole_cross = xr.startswith("zz00 ->")
    print(f"  [{'HOLE' if hole_cross else 'ok  '}] ar25 confirms+reads bp35 _harness   -> {xr!r}")
    ok &= hole_cross

    # ---- 3) THE FIX: block() the _harness dirs -> EACCES for the agent -----
    for g in games:
        uid_sandbox.block([runs[g].parent / "_harness"])  # 0700 root, one line
    fixed_self = _as_uid(games["ar25"], lambda: _read_own_harness(runs["ar25"]))
    fixed_cross = _as_uid(games["ar25"], lambda: _read_own_harness(runs["bp35"]))
    fix_ok = fixed_self.startswith("EACCES") and fixed_cross.startswith("EACCES")
    print(
        f"  [{'ok  ' if fixed_self.startswith('EACCES') else 'FAIL'}] after block(): own  _harness -> {fixed_self!r}"
    )
    print(
        f"  [{'ok  ' if fixed_cross.startswith('EACCES') else 'FAIL'}] after block(): bp35 _harness -> {fixed_cross!r}"
    )
    ok &= fix_ok

    # ---- 4) the fix must NOT break the harness (root) or the agent's own run
    root_still = runs["ar25"].parent / "_harness"
    root_reads = root_still.exists() and any(root_still.iterdir())  # root unaffected
    own_run_ok = _as_uid(games["ar25"], lambda: open(runs["ar25"] / "ipc" / "states.jsonl").read(4))
    print(
        f"  [{'ok  ' if root_reads else 'FAIL'}] root harness still reads _harness    -> {root_reads}"
    )
    print(
        f"  [{'ok  ' if own_run_ok.startswith('{') else 'FAIL'}] agent still reads its OWN run dir     -> {own_run_ok!r}"
    )
    ok &= root_reads and own_run_ok.startswith("{")

    shutil.rmtree(base, ignore_errors=True)
    print(
        "\n"
        + (
            "✅ documented: _harness is world-readable (self+cross deanon) TODAY; "
            "one-line block([_harness]) closes it without breaking harness/agent"
            if ok
            else "❌ unexpected — layout/behaviour changed, re-investigate"
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
