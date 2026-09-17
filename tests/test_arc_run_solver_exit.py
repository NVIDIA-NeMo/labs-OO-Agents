# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ARC runner reports missing results as failure, independently of harness exit."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

RUN_SOLVER = Path(__file__).resolve().parents[1] / "examples/arc_agi_3/run_solver.py"


@pytest.mark.parametrize("harness_exit", [0, 1], ids=["harness-success", "harness-nonzero"])
@pytest.mark.parametrize("has_result", [False, True], ids=["missing-result", "valid-result"])
def test_run_solver_exit_requires_result(tmp_path, monkeypatch, harness_exit, has_result):
    # Import the real stdlib-only script without retaining its sys.path insertion.
    monkeypatch.setattr(sys, "path", sys.path.copy())
    spec = importlib.util.spec_from_file_location("_arc_run_solver", RUN_SOLVER)
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    # No SDK, credentials, signals, subprocesses, or real shared /tmp directories.
    monkeypatch.setattr(runner, "_set_parent_death_signal", Mock())
    monkeypatch.setattr(runner, "_load_dotenv", Mock())
    monkeypatch.setattr(runner, "os", SimpleNamespace(environ={}))
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "DATA_DIR", tmp_path)
    monkeypatch.setattr(runner.signal, "signal", Mock())
    monkeypatch.setattr(runner.tempfile, "gettempdir", lambda: str(tmp_path / "neutral"))
    monkeypatch.setattr(runner.time, "sleep", Mock(side_effect=AssertionError("unexpected wait")))

    launcher = Mock(pid=12345)
    launcher.poll.return_value = None
    harness = Mock(pid=12346)
    harness.poll.return_value = harness_exit
    log_handles = []
    neutral_dirs = []
    result = {"termination_reason": "game_over", "levels_completed": 0, "total_steps": 1}

    def spawn(command, **kwargs):
        neutral = Path(command[command.index("--run-dir") + 1])
        assert neutral.is_relative_to(tmp_path)
        neutral_dirs.append(neutral)
        log_handles.append(kwargs["stdout"])
        if Path(command[1]).name == "harness.py":
            if has_result:
                (neutral / "result.json").write_text(json.dumps(result))
            return harness
        assert Path(command[1]).name == "launcher.py"
        return launcher

    popen = Mock(side_effect=spawn)
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    results_root = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_solver.py",
            "--game",
            "synthetic",
            "--variant",
            "mdfiles",
            "--sandbox",
            "off",
            "--results-root",
            str(results_root),
            "--model",
            "synthetic-model",
        ],
    )

    exit_code = runner.main()

    assert popen.call_count == 2
    launcher.terminate.assert_called_once()
    assert all(handle.closed for handle in log_handles)
    assert all(not directory.exists() for directory in neutral_dirs)
    [run_dir] = results_root.glob("nemo_solver/*_synthetic_mdfiles")
    if has_result:
        assert json.loads((run_dir / "result.json").read_text()) == result
        assert exit_code == 0
    else:
        assert not (run_dir / "result.json").exists()
        assert exit_code == 1
