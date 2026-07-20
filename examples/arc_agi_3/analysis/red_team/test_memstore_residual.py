# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Red-team analysis test (NOT production): documents a low-severity residual in the
cross-game memory-store isolation and the mitigation that keeps it safe in practice.

Finding (Pass 5): each game's memory store lives at
``/tmp/agent_stores/game-<random>.sqlite``. The parent dir is mode 1733 (sticky,
write+exec, NOT readable) so no uid can LIST it — but the store FILES are created
0644 (world-readable). So:

  * a different game's uid CANNOT enumerate the store filenames (dir is no-list);
  * BUT if it knew/guessed the exact filename, it COULD read the file (0644).

Why it's safe in practice: the filename embeds a RANDOM per-run alias
(secrets.token_hex) that a game never sees for OTHER games (aliases are anonymized
and not written anywhere the agent can read), and the dir can't be listed — so the
name is unguessable. The live fleet scan found 0 games even referencing the store.

Airtight hardening (recommendation, launcher-side / NOT changed here): create the
store files 0600 (owner-only). Then the file perms alone isolate them, independent
of filename secrecy.

This test asserts the two facts (no-list dir, world-readable file).
Run as root:  python3 test_memstore_residual.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def main() -> int:
    d = Path(tempfile.mkdtemp())
    os.chmod(d, 0o1777)  # mirror /tmp (mkdtemp is 0700, which would mask it)
    store = d / "stores"
    store.mkdir()
    os.chmod(store, 0o1733)  # as uid_sandbox.shared_writable does
    f = store / "game-deadbeef.sqlite"
    f.write_text("SQLite format 3\x00...")
    os.chown(f, 50100, 50100)
    os.chmod(f, 0o644)  # created by the owning game's uid

    def as_uid(uid, fn):
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            os.setgroups([])
            os.setgid(uid)
            os.setuid(uid)
            try:
                out = fn()
            except OSError as e:
                out = f"EACCES({e.errno})"
            os.write(w, str(out).encode()[:32])
            os._exit(0)
        os.close(w)
        res = os.read(r, 64).decode()
        os.waitpid(pid, 0)
        return res

    other = 50200
    enumerated = as_uid(other, lambda: ",".join(os.listdir(store)))
    read_by_name = as_uid(other, lambda: f.read_text()[:15])
    print(f"  other uid enumerates store dir -> {enumerated!r}  (want EACCES: no-list)")
    print(f"  other uid reads store BY NAME  -> {read_by_name!r}  (0644 -> readable; residual)")

    ok = enumerated.startswith("EACCES") and read_by_name.startswith("SQLite")
    print(
        "\n"
        + (
            "✅ documented: dir is non-enumerable (safe via random filename), "
            "files are 0644 (residual — recommend 0600 for airtight)"
            if ok
            else "❌ unexpected"
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
