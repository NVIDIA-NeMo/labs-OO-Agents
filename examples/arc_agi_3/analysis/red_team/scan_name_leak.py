# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Red-team scanner: does any REAL game name leak into a game's ANONYMISED context?

Each game's agent only ever sees an opaque alias ``game-<hex>``. Its real identity is
the 4-symbol short code (run dir: ``ar25``,``bp35``,…) and the full id ``<code>-<hex>``
(harness log ``resolved game X -> X-<hex>``). If any of the 25 real names appears in a
game's own agent messages (case-insensitive — ``ar25``/``AR25``), the anonymisation
leaked: OWN = self-deanonymisation, OTHER = cross-game name leak.

Uses ``grep -oiP`` (the C engine — a pure-python ``re`` alternation of 25 codes over the
hex-grid text backtracks for minutes). All-hex codes (``cd82``,``dc22``) collide with the
grid, so their bare hits are reported separately as likely false positives; the leak
verdict rests on full-id/path forms plus bare hits of non-hex codes.

Run:  python3 scan_name_leak.py     Writes: evidence/name_leak.json
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from rt_common import RUN_ROOT
from rt_common import _variant_dir as _vdir


def rt_EVID():
    from rt_common import EVID

    return EVID


EVID = rt_EVID()
SKIP = {"red_team", "analysis"}
HEXSET = set("0123456789abcdef")


def _grep(pat: str, root: Path) -> list[str]:
    """Return every (lowercased) match of a PCRE across a game's message tree."""
    msgs = root / "agent_logs"
    if not msgs.exists():
        return []
    r = subprocess.run(["grep", "-rhoiP", pat, str(msgs)], capture_output=True, text=True)
    return [x.lower() for x in r.stdout.splitlines() if x]


def main() -> int:
    codes = [
        d.name
        for d in sorted(RUN_ROOT.iterdir())
        if d.is_dir() and d.name not in SKIP and _vdir(d) is not None
    ]
    allhex = {c for c in codes if all(ch in HEXSET for ch in c.lower())}
    nonhex = [c for c in codes if c not in allhex]

    full = {}
    for g in codes:
        for lg in (_vdir(RUN_ROOT / g) / "_harness").glob("*.log"):
            m = re.search(
                rf"resolved game {re.escape(g)} -> ({re.escape(g)}-[0-9a-f]+)",
                lg.read_text(errors="ignore"),
                re.I,
            )
            if m:
                full[g] = m.group(1)
                break

    code_alt = "|".join(codes)
    nonhex_alt = "|".join(nonhex)
    full_alt = "|".join(re.escape(v) for v in full.values())

    confirmed, weak, allhex_bare = [], [], {}
    for g in codes:
        run = next(iter(_vdir(RUN_ROOT / g).glob("2*")), None)
        if not run:
            continue
        # HIGH: full id (unambiguous) + run-dir path forms
        for m in _grep(rf"\b({full_alt})\b", run) if full_alt else []:
            c = m.split("-")[0]
            confirmed.append(
                {
                    "context_game": g,
                    "name": c,
                    "kind": "own" if c == g else "foreign",
                    "via": "full_id",
                }
            )
        for m in _grep(rf"/({code_alt})/|\b({code_alt})_(?:memory|mdfiles)\b", run):
            c = re.split(r"[/_]", m.strip("/"))[0]
            confirmed.append(
                {
                    "context_game": g,
                    "name": c,
                    "kind": "own" if c == g else "foreign",
                    "via": "path",
                }
            )
        # MED: bare token, non-hex codes only (can't occur in the hex grid)
        for c in _grep(rf"\b({nonhex_alt})\b", run) if nonhex_alt else []:
            weak.append({"context_game": g, "name": c, "kind": "own" if c == g else "foreign"})
        # LOW: bare all-hex codes — grid false positives, reported not counted
        for c in _grep(rf"\b({'|'.join(allhex)})\b", run) if allhex else []:
            allhex_bare[c] = allhex_bare.get(c, 0) + 1

    result = {
        "games": len(codes),
        "codes": codes,
        "full_ids": full,
        "confirmed_leaks": confirmed,
        "weak_bare_hits": weak,
        "ignored_allhex_bare_counts": allhex_bare,
        "verdict": "LEAK" if confirmed else "REVIEW" if weak else "CLEAN",
    }
    EVID.mkdir(exist_ok=True)
    (EVID / "name_leak.json").write_text(json.dumps(result, indent=2))

    print(
        f"scanned {len(codes)} games' contexts for all {len(codes)} real names (case-insensitive)"
    )
    print(f"  HIGH (full_id/path) leaks: {len(confirmed)}")
    for c in confirmed[:15]:
        print(f"    !! {c['context_game']} <- {c['name']} ({c['kind']}, via {c['via']})")
    print(f"  MED  (bare non-hex code) hits: {len(weak)}")
    for w in weak[:15]:
        print(f"    ?  {w['context_game']} <- {w['name']} ({w['kind']})")
    if allhex_bare:
        print(f"  all-hex bare hits ignored (grid FP): {allhex_bare}")
    print(f"\nverdict: {result['verdict']}  (evidence/name_leak.json)")
    return 0 if not confirmed else 1


if __name__ == "__main__":
    raise SystemExit(main())
