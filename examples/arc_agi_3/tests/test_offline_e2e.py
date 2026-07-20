# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end integration test: one offline game, played by the real agent.

Drives ``run_solver.py`` for a short offline run and asserts the agent + harness
IPC loop produced real progress (states, actions, and a gameplay summary). This
exercises the whole stack — the nooa agent, the launcher, the file-IPC harness,
and the vendored ``arc_agi_3`` wrapper over the public SDK.

Marked ``integration`` (skipped by default) because it needs:
  * the ``arc`` extra installed (``uv sync --extra arc``),
  * a reachable LLM gateway (``ARC_LLM_MODEL`` / ``ARC_LLM_BASE_URL`` / ``ARC_LLM_API_KEY``),
  * ``ARC_API_KEY`` (the SDK downloads the offline game on first use).

Run it explicitly::

    uv run pytest examples/arc_agi_3/tests/test_offline_e2e.py -m integration
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXAMPLE_DIR.parents[1]

pytestmark = pytest.mark.integration


def _have_sdk() -> bool:
    try:
        import arc_agi  # noqa: F401
        import arcengine  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _have_sdk(), reason="ARC-AGI-3 SDK not installed (`uv sync --extra arc`)")
@pytest.mark.skipif(
    not os.environ.get("ARC_LLM_API_KEY") or not os.environ.get("ARC_API_KEY"),
    reason="needs ARC_LLM_API_KEY (gateway) + ARC_API_KEY (offline game download)",
)
def _traces_contain(run: Path, needle: str) -> int:
    """Count occurrences of ``needle`` inside llm.input_messages spans (the exact
    bytes sent to the model) across a run's OTLP traces."""
    hits = 0
    for f in (run / "traces").glob("*.jsonl"):
        for line in f.read_text(errors="ignore").splitlines():
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            for rs in doc.get("resourceSpans", []):
                for ss in rs.get("scopeSpans", []):
                    for sp in ss.get("spans", []):
                        for kv in sp.get("attributes", []):
                            k = kv.get("key", "")
                            v = kv.get("value", {}).get("stringValue", "") or ""
                            if "input_messages" in k and needle in v:
                                hits += 1
    return hits


def test_offline_run_makes_progress(tmp_path: Path) -> None:
    results_root = tmp_path / "results" / "arc_agi_3"
    proc = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE_DIR / "run_solver.py"),
            "--game",
            "ls20",
            "--variant",
            "memory",
            "--operation-mode",
            "offline",
            "--sandbox",
            "inproc",  # keep the test runnable off-root; drop is covered separately
            "--max-env-steps",
            "6",
            "--agent-turn-timeout",
            "300",
            "--nudge-after",
            "240",
            "--results-root",
            str(results_root),
            "--group",
            "e2e",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"run_solver failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"

    # run_solver writes to <results_root>/<group>/<ts>_<game>_<variant>/
    run_dirs = list(results_root.glob("e2e/*_ls20_memory"))
    assert run_dirs, "no run directory was created"
    run = run_dirs[0]

    gameplay = json.loads((run / "gameplay.json").read_text())
    assert gameplay.get("total_steps", 0) >= 1, f"agent took no env steps: {gameplay}"

    actions = (run / "ipc" / "actions.jsonl").read_text().strip().splitlines()
    assert actions, "agent submitted no action batches"

    # --- regression: the real game name must NEVER reach the model (finding F-A).
    # The neutral-/tmp run dir means no traceback / path can carry 'ls20'.
    assert _traces_contain(run, "ls20") == 0, "real game name leaked into llm.input_messages"

    # --- regression: memory is copied back from the neutral dir to results.
    assert (run / "team_nemo" / "shared" / "memory.sqlite").exists(), (
        "memory.sqlite was not copied back to the results dir"
    )

    # --- regression: no neutral /tmp dir lingers for this run (copy-back cleaned it).
    import hashlib
    import re
    import tempfile

    ts, variant = run.name.split("_")[0] + "_" + run.name.split("_")[1], "memory"
    alias = "game-" + hashlib.sha1(f"{ts}{variant}".encode()).hexdigest()[:6]
    safe_alias = re.sub(r"[^a-z0-9_-]", "_", alias.lower())
    neutral = Path(tempfile.gettempdir()) / "arc_runs" / f"arc_run_{safe_alias}"
    assert not neutral.exists(), f"neutral dir left behind in /tmp: {neutral}"
