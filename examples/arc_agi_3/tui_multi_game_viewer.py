# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-game TUI viewer for the nemo-oo ARC-AGI-3 solver — nested-hierarchy variant.

Thin wrapper over ``arc_agi_3.tui_multi_game_viewer.MultiGameTUI`` (run in the
progressive-learning venv) that reuses ALL of the reference dashboard + per-game
panels (grid, reasoning, REPL, status, world-model, comm-stats) unchanged —
including the dashboard's native per-game budget columns ($In/$Out/$Cch/$CchW)
and the fleet cumulative Cost line.

Only one thing differs from the reference: our multi-runner lays runs out as
``<container>/<game>/<variant>/<ts_run>/`` (nested hierarchy) rather than the
reference's ``<container>/<game_id>/<ts_run>/``. The reference discovers a game's
logs at ``results_dir / game_id``; here we resolve each run's physical directory
from the ``_run_dir`` the runner records in ``status.json`` instead, so the
game/variant nesting is transparent. The status/tab key stays slash-free
(``<game>_<variant>``) so Textual tab ids remain valid.

Usage (normally started by run_multi.py):
    progressive-learning/.venv/bin/python examples/arc_agi_3/tui_multi_game_viewer.py \
        results/arc_agi_3/nemo_solver/<container>/ --watch team_leader
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# This wrapper must import the progressive-learning REFERENCE package
# ``arc_agi_3`` — not the vendored SDK wrapper that lives next to this script
# (examples/arc_agi_3/arc_agi_3/, added with the clean example) and would
# shadow it via sys.path[0]. Drop the script dir and resolve via the
# progressive-learning root instead (our cwd — run_multi launches this viewer
# with cwd=<repo>/progressive-learning).
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _HERE)]
sys.path.insert(0, os.getcwd())

import arc_agi_3.tui_multi_game_viewer as _ref_mod  # noqa: E402
from arc_agi_3.tui_multi_game_viewer import (  # noqa: E402
    MultiGameTUI as _RefMultiGameTUI,
)
from arc_agi_3.tui_multi_game_viewer import (  # noqa: E402
    _find_agent_logs,
    _read_status,
)

_ATTACHABLE = ("running", "completed", "failed", "crashed", "terminated")

# Newer progressive-learning checkouts hard-index ``model_uri`` / int cache
# fields on llm_call events in update_state; events from older runs of the
# nemo-oo exporter may lack them. Default them so token/cost accounting works
# on every generation; rebind in the reference module so its tail threads use
# it. No-op setdefaults on generations that don't need it.
_ref_update_state = _ref_mod.update_state


def _compat_update_state(event, state):
    if event.get("event") == "llm_call":
        event.setdefault("model_uri", event.get("agent_id") or event.get("model") or "unknown")
        for k in (
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_creation_tokens",
        ):
            if event.get(k) is None:
                event[k] = 0
    return _ref_update_state(event, state)


_ref_mod.update_state = _compat_update_state


class MultiGameTUI(_RefMultiGameTUI):
    """Reference TUI, but game dirs are resolved from status.json ``_run_dir``
    (nested ``<game>/<variant>`` layout) instead of ``results_dir / game_id``."""

    def _scan_for_new_games(self) -> None:
        runs = _read_status(self.results_dir).get("runs", {})
        for game_id, info in runs.items():
            if game_id in self._known_games:
                continue
            if info.get("status") not in _ATTACHABLE:
                continue
            run_dir = info.get("_run_dir")
            # game_dir holds the timestamped run dir(s); _find_agent_logs iterates
            # it for agent_logs. With the nested layout that's the <variant> dir
            # (parent of the <ts_run>); fall back to the reference path.
            game_dir = Path(run_dir).parent if run_dir else self.results_dir / game_id
            agent_logs = _find_agent_logs(game_dir, self.watch)
            if agent_logs and agent_logs.exists():
                self._add_game_tab(game_id, info, agent_logs)
                self._known_games.add(game_id)


def main() -> None:
    ap = argparse.ArgumentParser(description="Nested-hierarchy multi-game TUI viewer")
    ap.add_argument("results_dir", help="multi-run container (has status.json)")
    ap.add_argument(
        "--watch",
        default="team_leader",
        help="agent role shown in game tabs (default: team_leader)",
    )
    args = ap.parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    MultiGameTUI(results_dir=results_dir, watch=args.watch).run()


if __name__ == "__main__":
    main()
