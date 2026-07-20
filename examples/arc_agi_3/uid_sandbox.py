# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""External privilege-drop isolation for the solver process (Docker-friendly).

Runs the agent as an unprivileged uid so its filesystem reach is bounded by POSIX
permissions, not by the agent's own cooperation. This closes the two read paths a
generated cell can otherwise reach through the inherited host tools
(``self.shell``/``self.repo``), which bypass the in-process ``open()`` jail:

  * the environment source tree  -> made root-only (0700); the agent uid EACCESes;
  * every OTHER run directory     -> blocked along the path to the agent's own run
                                     dir, which is chowned to the agent uid.

The harness (which must read the environment source to run the game) keeps running
as root; only the agent/launcher process is dropped. Enforcement is entirely
external to the agent: it cannot see or undo it (an unprivileged process cannot
regain root or re-chmod the root-owned trees), so reading this file does not help
it escape. Needs only CAP_SETUID/CAP_SETGID/CAP_CHOWN + ``setpriv`` — no user
namespaces, no CAP_SYS_ADMIN, no Docker daemon. Network egress is NOT handled here
(the model gateway is reached directly); restrict it at the container/network
layer (egress allowlist to the gateway only).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def available() -> tuple[bool, str]:
    """(ok, reason): privilege-drop isolation can be applied on this host."""
    if os.geteuid() != 0:
        return False, "must start as root to drop privileges"
    if shutil.which("setpriv") is None:
        return False, "setpriv not installed"
    return True, "setpriv privilege-drop available"


def drop_prefix(uid: int, gid: int) -> list[str]:
    """argv prefix that execs the wrapped command as (uid, gid), no supplementary
    groups and no ability to re-acquire privileges."""
    return [
        "setpriv",
        "--reuid",
        str(uid),
        "--regid",
        str(gid),
        "--clear-groups",
        "--no-new-privs",
        "--",
    ]


def _chmod(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def block(trees: list[Path]) -> None:
    """Make each existing ROOT-OWNED tree readable by root only (0700). A non-root
    uid cannot traverse a 0700 root-owned dir, so nothing beneath it is reachable.

    Only touches root-owned paths: a dir already owned by a non-root uid has been
    claimed by a concurrent game's carve_own — leave it alone so parallel launches
    don't stomp each other."""
    for t in trees:
        try:
            if t.stat().st_uid != 0:
                continue  # claimed by a concurrent game — don't touch
        except OSError:
            continue
        try:
            os.chown(t, 0, 0)
        except OSError:
            pass
        _chmod(t, 0o700)


def carve_own(run_dir: Path, uid: int, gid: int, up_to: Path) -> None:
    """Give the agent uid exclusive access to its OWN run dir:

      * chown -R the run dir to (uid, gid), dir mode 0700 — so NO other uid (a
        concurrent game runs as a different uid; a prior run's dir is owned by a
        stale uid or root) can read it;
      * each ancestor up to ``up_to`` (inclusive) -> 0711 (traverse-only): the uid
        can reach its OWN dir by exact path, but a 0711 dir cannot be readdir'd, so
        the agent cannot ENUMERATE neighbours / other runs (find/grep/ls fail on it).

    Deliberately does NOT chmod sibling entries: with 25 games running concurrently
    under one container, each game's run dir is a sibling of the others, and blocking
    them would 0700-root every neighbour — the last game to carve would be the only
    one left traversable. Isolation between siblings comes from the 0700 uid-owned
    run dir (unreadable by another uid) + the non-listable 0711 chain (can't be
    found), not from mutating neighbours.
    """
    run_dir = run_dir.resolve()
    up_to = up_to.resolve()
    if run_dir.exists():
        for root, _dirs, files in os.walk(run_dir):
            try:
                os.chown(root, uid, gid)
            except OSError:
                pass
            for name in files:
                try:
                    os.chown(os.path.join(root, name), uid, gid)
                except OSError:
                    pass
        _chmod(run_dir, 0o700)

    node = run_dir
    while node != up_to and up_to in node.parents:
        _chmod(node.parent, 0o711)  # traverse-only, non-enumerable
        node = node.parent
    _chmod(up_to, 0o711)


def shared_writable(path: Path) -> None:
    """Create a dir that several distinct-uid agents each write their OWN file into
    (the memory-store parent). Mode 1733 = sticky + write/exec for all, but NOT
    readable: no uid can list the dir, so per-run store filenames (unguessable
    hashes) can't be enumerated, and sticky means each uid only deletes its own."""
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chown(path, 0, 0)
    except OSError:
        pass
    _chmod(path, 0o1733)
