# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the ported guardrails and the submit_actions auto-yield.

- B-lite AST cell guard (``_scan_cell``): the red-team flagged it as MISSING from
  the headless branch. It must block a cell that reaches for a host escape tool,
  an asyncio network entrypoint, or a denylist-bypass gadget, while passing
  legitimate grid/memory/numpy cells.
- ``submit_actions`` auto-yield: a successful submit must RAISE the CodeAct
  return signal (ending the turn) rather than returning a string, and an invalid
  submit must still return a ``REJECTED: …`` string so the agent can retry.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

import solver_agent as sa  # noqa: E402


# --------------------------------------------------------------------------- #
# B-lite cell guard
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "code,banned",
    [
        ("self.shell.run('ls /root')", "self.shell"),
        ("self.repo.grep('game')", "self.repo"),
        ("self.web.fetch('http://x')", "self.web"),
        ("self.mcp.call('x')", "self.mcp"),
        ("().__class__.__bases__[0].__subclasses__()", "__subclasses__"),
        ("x.__globals__['os']", "__globals__"),
        ("__import__('os').system('id')", "__import__"),
        ("eval('1+1')", "eval"),
        ("exec('x=1')", "exec"),
        ("import asyncio; asyncio.open_connection('h', 80)", "open_connection"),
    ],
)
def test_cell_guard_blocks_escape_and_bypass(code, banned):
    assert sa._scan_cell(code) == banned


@pytest.mark.parametrize(
    "code",
    [
        "import numpy as np\narr = self.grid_array(g)\nprint(arr.shape)",
        "self.memory.recall('level rules'); self.submit_actions(['UP'], 'r')",
        "z = self.h.model.encode(arr)\nself.write_helper('m.py', src)",
        "for a in self.trajectory(last_n=3): print(a)",
    ],
)
def test_cell_guard_passes_legitimate_cells(code):
    assert sa._scan_cell(code) is None


def test_cell_guard_ignores_strings_and_comments():
    # matches inside strings/comments (e.g. helper source passed to write_helper)
    # must NOT trip the AST guard.
    assert sa._scan_cell("src = 'self.shell.run(x)'  # not real code") is None
    assert sa._scan_cell("# eval() is dangerous\nx = 1") is None


def test_restrictions_block_import_bypass_modules():
    # the widened denylist must cover the C-level / dynamic-import escapes.
    blocked = sa._ARC_RESTRICTIONS.blocked_modules
    for mod in ("sqlite3", "multiprocessing", "importlib", "pickle", "sys", "ctypes"):
        assert mod in blocked, f"{mod} must be blocked in agent cells"


# --------------------------------------------------------------------------- #
# submit_actions auto-yield
# --------------------------------------------------------------------------- #
def _fake_agent(tmp_path, state):
    """Minimal object exposing exactly what submit_actions touches."""
    actions_path = tmp_path / "actions.jsonl"
    actions_path.touch()
    fake = SimpleNamespace(
        _latest_state=lambda: state,
        _last_submitted_turn=lambda: -1,
        _actions_path=actions_path,
        _turn_started_at=0.0,
        _effort_ladder=[],
        _ladder_active_effort=None,
    )
    return fake


def test_submit_actions_autoyields_on_success(tmp_path):
    from nooa.strategies.codeact import _ReturnResultSignal

    state = {"turn": 0, "state": "NOT_FINISHED", "available_actions": ["UP", "DOWN"]}
    fake = _fake_agent(tmp_path, state)
    with pytest.raises(_ReturnResultSignal):
        sa.ArcSolverBase.submit_actions(fake, ["UP"], "predict: move up")
    # the action was written before the yield
    assert '"actions": ["UP"]' in (tmp_path / "actions.jsonl").read_text()


def test_submit_actions_rejects_invalid_without_yielding(tmp_path):
    state = {"turn": 0, "state": "NOT_FINISHED", "available_actions": ["UP"]}
    fake = _fake_agent(tmp_path, state)
    # unavailable action -> a REJECTED string, NOT a raise (agent can retry).
    out = sa.ArcSolverBase.submit_actions(fake, ["FLY"], "r")
    assert isinstance(out, str) and out.startswith("REJECTED")


def test_submit_actions_rejects_empty(tmp_path):
    state = {"turn": 0, "state": "NOT_FINISHED", "available_actions": ["UP"]}
    fake = _fake_agent(tmp_path, state)
    out = sa.ArcSolverBase.submit_actions(fake, [], "r")
    assert isinstance(out, str) and "empty" in out.lower()
