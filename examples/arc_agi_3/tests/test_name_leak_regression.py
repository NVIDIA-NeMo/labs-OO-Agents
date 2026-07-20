# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression test for red-team finding F-A: the real game name must never reach
the agent via a file path.

The leak: the agent writes world-model helpers under ``<run_dir>/team_nemo/shared/
helpers/*.py`` and loads them with ``importlib.util.spec_from_file_location`` (see
``ArcSolverBase.load_helpers``). A loaded helper's ``__file__`` is its ABSOLUTE
path, so when a cell calls ``self.h.model.<fn>()`` and the helper raises, the
Python traceback prints that absolute path — and if ``run_dir`` embeds the real
game name (``results/.../<game>/memory/<ts>_<game>_memory/...``), the traceback
deanonymises the game to the model. Observed live in 5/25 games.

The fix: run_solver puts the agent's whole run_dir on a neutral
``/tmp/arc_runs/arc_run_<alias>`` path (no game name anywhere), so no traceback,
log, or path echo can carry the name.

These tests exercise the exact mechanism (helper import → raise → traceback) with
BOTH layouts, so the assertion has teeth: the old game-named layout DOES leak, the
new neutral layout does NOT.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import tempfile
import traceback
from pathlib import Path

GAME = "ls20"  # a real game id that must never appear in agent-visible paths

_HELPER_SRC = "def encode(x):\n    raise ValueError('unexpected sparse symbol')\n"


def _traceback_from_helper(run_dir: Path) -> str:
    """Replicate ArcSolverBase.write_helper + load_helpers + a raising call, and
    return the traceback string that would feed the next prompt."""
    helpers = run_dir / "team_nemo" / "shared" / "helpers"
    helpers.mkdir(parents=True, exist_ok=True)
    helper = helpers / "model.py"
    helper.write_text(_HELPER_SRC)  # write_helper

    # load_helpers: exact same import path the agent uses.
    spec = importlib.util.spec_from_file_location("arc_helper_model", helper)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    try:
        mod.encode(0)  # a cell calling self.h.model.encode(...) that raises
    except ValueError:
        return traceback.format_exc()
    raise AssertionError("helper did not raise")


def _neutral_run_dir() -> Path:
    """Replicate run_solver's neutral-path derivation (no game name in it)."""
    ts, variant, tag = "20260714_215959", "memory", ""
    alias = "game-" + hashlib.sha1(f"{ts}{variant}{tag}".encode()).hexdigest()[:6]
    safe_alias = re.sub(r"[^a-z0-9_-]", "_", alias.lower())
    return Path(tempfile.gettempdir()) / "arc_runs" / f"arc_run_{safe_alias}"


def test_game_named_layout_leaks_the_name(tmp_path: Path) -> None:
    """Teeth: the OLD results layout (game name in the path) DOES leak it into the
    helper traceback — proving the traceback carries the file path, so the test
    below is meaningful (not vacuously green)."""
    ts = "20260714_215959"
    game_named = tmp_path / "results" / "arc_agi_3" / "nemo_solver" / f"{ts}_{GAME}_memory"
    tb = _traceback_from_helper(game_named)
    assert GAME in tb, "expected the old game-named layout to leak the name in the traceback"


def test_neutral_layout_does_not_leak_the_name() -> None:
    """The fix: with run_solver's neutral run_dir, the helper traceback carries a
    game-name-free path, so the name never reaches the prompt."""
    neutral = _neutral_run_dir()
    try:
        tb = _traceback_from_helper(neutral)
        assert GAME not in tb, f"game name leaked into helper traceback:\n{tb}"
        # sanity: the traceback DID include the (neutral) helper path.
        assert "model.py" in tb
    finally:
        import shutil

        shutil.rmtree(neutral, ignore_errors=True)


def test_run_solver_neutral_path_is_game_name_free() -> None:
    """run_solver's neutral run_dir must never contain the game name (it is derived
    only from ts/variant/tag, never from the game), while the OLD results run_name
    did. Locks in the layout guarantee behind the fix."""
    ts, variant = "20260714_215959", "memory"
    old_run_name = f"{ts}_{GAME}_{variant}"  # results/.../nemo_solver/<this>
    assert GAME in old_run_name  # the old layout embedded the game

    neutral = _neutral_run_dir()
    assert GAME not in str(neutral)  # the new layout does not
    assert "arc_run_game-" in neutral.name
