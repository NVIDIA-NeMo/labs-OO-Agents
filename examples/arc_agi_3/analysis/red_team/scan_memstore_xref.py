# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Red-team scanner (Pass 27): cross-game memory-store alias isolation.

The one documented residual in the cross-game isolation is that per-run memory
stores live at ``/tmp/agent_stores/game-<hex>.sqlite`` created 0644 (world-
readable). The parent dir is 1733 (non-listable), so a game cannot ENUMERATE the
filenames — but if it somehow LEARNED a neighbour's random ``game-<hex>`` alias it
could read that neighbour's store by exact path. The safety argument is that the
alias is a per-run ``secrets.token_hex`` a game never sees for OTHER games.

This scanner tests that argument behaviourally against the live run: it maps every
``game-<hex>`` alias that appears in each game's agent logs to the game that owns
it (owner = the game that references it most), then reports any game that
references an alias owned by a DIFFERENT game. Zero cross-references == the random
aliases are effectively secret == the 0644 file mode is moot in practice.

Run:  python3 scan_memstore_xref.py
Writes: evidence/memstore_xref.json
"""

from __future__ import annotations

import collections
import glob
import json
import re

from rt_common import RUN_ROOT  # noqa: E402  (RUN_ROOT -> the run under audit)
from rt_common import VARIANT_GLOB as VG  # noqa: E402


def rt_EVID():
    from rt_common import EVID

    return EVID


ALIAS = re.compile(r"game-[0-9a-f]{8}(?:\.sqlite)?")
EVID = rt_EVID()


def _game_of(fp: str) -> str:
    return fp.split(f"/{RUN_ROOT.name}/")[1].split("/")[0]


def main() -> int:
    logs = glob.glob(f"{RUN_ROOT}/*/{VG}/*/agent_logs/**/*.md", recursive=True)
    logs += glob.glob(f"{RUN_ROOT}/*/{VG}/*/agent_logs/**/*.jsonl", recursive=True)

    game_alias: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for fp in logs:
        g = _game_of(fp)
        try:
            t = open(fp, errors="ignore").read()
        except OSError:
            continue
        for a in ALIAS.findall(t):
            game_alias[g][a.replace(".sqlite", "")] += 1

    canon = {g: c.most_common(1)[0][0] for g, c in game_alias.items() if c}
    owner = {a: g for g, a in canon.items()}

    cross = []
    for g, c in game_alias.items():
        for a, n in c.items():
            o = owner.get(a)
            if o and o != g:
                cross.append({"game": g, "alias": a, "owner": o, "count": n})

    multi = {g: sorted(c) for g, c in game_alias.items() if len(c) > 1}

    result = {
        "log_files_scanned": len(logs),
        "games_with_alias": len(canon),
        "canonical_alias_per_game": canon,
        "games_referencing_multiple_aliases": multi,
        "cross_game_references": cross,
        "verdict": "ISOLATED" if not cross else "LEAK",
    }
    EVID.mkdir(exist_ok=True)
    (EVID / "memstore_xref.json").write_text(json.dumps(result, indent=2))

    print(f"scanned {len(logs)} log files; {len(canon)} games have a store alias")
    if cross:
        print(f"!! {len(cross)} CROSS-GAME alias references (potential store-read vector):")
        for c in cross[:20]:
            print(f"   {c['game']} -> {c['alias']} (owned by {c['owner']}) x{c['count']}")
    else:
        print(
            "OK: every game references ONLY its own store alias — no cross-game "
            "alias leakage (random aliases are effectively secret; 0644 moot)."
        )
    if multi:
        print(f"note: {len(multi)} game(s) reference >1 distinct alias: {list(multi)}")
    print(f"\nverdict: {result['verdict']}  (evidence/memstore_xref.json)")
    return 0 if not cross else 1


if __name__ == "__main__":
    raise SystemExit(main())
