# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Red-team analysis test (NOT production): documents a KNOWN limitation of the
in-process cell guard and proves the OS backstop that covers it.

Finding (Pass 4): the AST cell guard in solver_agent bans the literal attribute
names (self.shell, __subclasses__, …) and the names __import__/eval/exec/compile —
but NOT `getattr`. So `getattr(self, 'sh'+'ell').run(...)` is a static-analysis
bypass: the AST has no `.shell` node, so the guard doesn't fire.

This is expected — the cell guard is explicitly best-effort defense-in-depth; the
HARD boundary is the external uid-drop. This test asserts both facts so the
limitation is documented and the backstop is verified:

  1. `_scan_cell` does NOT catch a getattr-built escape reference (the gap);
  2. under the uid-drop, even a *reached* shell can't read the game source,
     because the shell runs as the unprivileged agent uid (EACCES) — the effect
     is blocked regardless of the guard.

No game in the live run exploited this (the fleet scan found 0 getattr→escape
uses), but if one did, the OS layer holds.

Run:  python3 test_guard_evasion.py   (root, for the uid part)
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

EXAMPLE = Path("/root/projects/nemo_oo_agents/examples/arc_agi_3")
sys.path.insert(0, str(EXAMPLE))
from solver_agent import _scan_cell  # noqa: E402


def main() -> int:
    ok = True

    # 1) the gap: getattr-built escape reference slips past the AST scanner
    bypass = "getattr(self, 'sh' + 'ell').run('grep -R secret /')"
    caught_direct = _scan_cell("self.shell.run('x')") is not None
    caught_bypass = _scan_cell(bypass) is not None
    print(f"  direct  self.shell.run       -> scanner {'BLOCKS' if caught_direct else 'misses'}")
    print(
        f"  getattr(self,'shell').run    -> scanner {'BLOCKS' if caught_bypass else 'MISSES (known gap)'}"
    )
    ok &= caught_direct and (not caught_bypass)  # documents: direct blocked, getattr NOT

    # 2) the backstop: the dropped uid can't read a root:700 'game source' even if
    #    the shell were reached (simulate: read as an unprivileged uid).
    d = Path(tempfile.mkdtemp())
    src = d / "game.py"
    src.write_text("class Secret: pass")
    os.chown(src, 0, 0)
    os.chmod(src, 0o700)
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)
        try:
            src.read_text()
            msg = b"READ"
        except OSError:
            msg = b"EACCES"
        os.write(w, msg)
        os._exit(0)
    os.close(w)
    backstop = os.read(r, 8).decode()
    os.waitpid(pid, 0)
    print(f"  uid-drop backstop (nobody reads root:700 game.py) -> {backstop}")
    ok &= backstop == "EACCES"

    print(
        "\n"
        + (
            "✅ documented: cell guard has a getattr gap (best-effort), "
            "uid-drop blocks the EFFECT regardless"
            if ok
            else "❌ unexpected"
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
