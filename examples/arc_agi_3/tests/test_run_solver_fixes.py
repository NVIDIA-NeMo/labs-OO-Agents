# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for run_solver / harness / broker fixes:

- DATA_DIR: a competition/offline run showed RHAE 0/100 because the harness cwd
  had no ``environment_files/`` (the SDK reads per-level RHAE baselines there).
  DATA_DIR must autodetect an in-repo progressive-learning checkout and honour an
  ARC_DATA_DIR override.
- scorecard broker: a Ctrl-C on the fleet used to orphan the shared competition
  scorecard open, because ``KeyboardInterrupt`` (a BaseException) escaped the
  ``except Exception`` around the stdin wait. The wait must catch it and fall
  through to the bounded close.
- uid_sandbox (--sandbox drop): a dropped uid must NOT be able to read a
  root-owned 0700 file (the game source), while ``carve_own`` gives that uid its
  own run dir (root-only test).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXAMPLE_DIR.parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))


# --------------------------------------------------------------------------- #
# DATA_DIR autodetect + override
# --------------------------------------------------------------------------- #
def _data_dir_with_env(**env) -> str:
    """Import run_solver in a fresh interpreter and print its module-level DATA_DIR."""
    code = "import run_solver; print(run_solver.DATA_DIR)"
    e = {**os.environ, **env, "PYTHONPATH": str(EXAMPLE_DIR)}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=e, cwd=str(REPO_ROOT)
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_data_dir_honours_arc_data_dir_override(tmp_path):
    got = _data_dir_with_env(ARC_DATA_DIR=str(tmp_path))
    assert got == str(tmp_path)


def test_data_dir_autodetects_progressive_learning():
    # When ARC_DATA_DIR is unset, prefer an in-repo progressive-learning checkout
    # that ships environment_files/ (else the repo root). Both are game-name-free.
    env = dict(os.environ)
    env.pop("ARC_DATA_DIR", None)
    got = Path(_data_dir_with_env(**{k: v for k, v in env.items() if k == "PATH"}))
    pl = REPO_ROOT / "progressive-learning"
    if (pl / "environment_files").is_dir():
        assert got == pl, "should prefer the progressive-learning checkout with baselines"
    else:
        assert got == REPO_ROOT


# --------------------------------------------------------------------------- #
# scorecard broker: Ctrl-C must reach the close path
# --------------------------------------------------------------------------- #
def test_broker_stdin_wait_catches_keyboardinterrupt():
    """The stdin-wait must swallow KeyboardInterrupt so teardown/close runs. We
    replicate the exact guard and assert a KeyboardInterrupt raised inside does
    NOT escape (the pre-fix ``except Exception`` let it through)."""
    src = (EXAMPLE_DIR / "scorecard_broker.py").read_text()
    assert "except (Exception, KeyboardInterrupt):" in src, (
        "broker stdin wait must catch KeyboardInterrupt, not just Exception"
    )

    # Behavioural check of the pattern itself.
    reached_close = False
    try:
        try:
            raise KeyboardInterrupt  # Ctrl-C during `for line in sys.stdin`
        except (Exception, KeyboardInterrupt):
            pass
        reached_close = True  # falls through to the bounded close
    except KeyboardInterrupt:
        reached_close = False
    assert reached_close


# --------------------------------------------------------------------------- #
# uid-drop isolation (root only)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(os.geteuid() != 0, reason="uid-drop isolation needs root (fork+setuid)")
def test_uid_drop_blocks_game_source_read(tmp_path):
    import uid_sandbox

    ok, _why = uid_sandbox.available()
    assert ok, "root+setpriv expected in this environment"

    # a root-owned 0700 'game source' tree (what block() produces)
    src = tmp_path / "environment_files"
    src.mkdir()
    secret = src / "ls20.py"
    secret.write_text("SOLUTION = 42")
    uid_sandbox.block([src])

    uid = 47001

    def _read_as_uid() -> str:
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            try:
                os.setgroups([])
                os.setgid(uid)
                os.setuid(uid)
                msg = secret.read_text().encode()
            except OSError as e:
                msg = f"EACCES({e.errno})".encode()
            os.write(w, msg[:32])
            os._exit(0)
        os.close(w)
        out = os.read(r, 64).decode()
        os.waitpid(pid, 0)
        return out

    assert "EACCES" in _read_as_uid(), "dropped uid must not read the root:0700 game source"


@pytest.mark.skipif(os.geteuid() != 0, reason="uid-drop carve_own needs root (chown)")
def test_carve_own_gives_uid_its_own_run_dir(tmp_path):
    import uid_sandbox

    parent = tmp_path / "arc_runs"
    parent.mkdir()
    run = parent / "arc_run_game-abc123"
    (run / "ipc").mkdir(parents=True)
    (run / "ipc" / "states.jsonl").write_text("{}")

    uid = 47002
    uid_sandbox.carve_own(run, uid, uid, up_to=parent)

    # the run dir is now owned by uid and 0700; the parent is 0711 (traverse-only).
    assert run.stat().st_uid == uid
    assert (run.stat().st_mode & 0o777) == 0o700
    assert (parent.stat().st_mode & 0o777) == 0o711


def test_drop_prefix_is_setpriv():
    import uid_sandbox

    pref = uid_sandbox.drop_prefix(40001, 40001)
    assert pref[0] == "setpriv"
    assert "--reuid" in pref and "40001" in pref
    assert "--no-new-privs" in pref
