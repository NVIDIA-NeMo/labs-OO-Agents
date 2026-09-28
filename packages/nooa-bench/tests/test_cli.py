# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``nooa-bench run``: one task, headless, without Harbor."""

import json

import pytest
from click.testing import CliRunner
from coder_test_agents import CellLLM, cell
from nooa_bench.cli import main

RESULT = (
    "print('cell output')\n"
    "return_result(Done(explanation='solved', result=TaskResult("
    "solution_description='wrote hello.py', evidence='it printed hello', "
    "how_to_verify='python hello.py', report='hello.py prints hello')))"
)

WAIT_FOREVER = """\
self.queue_manager.queue("test_jobs")
return_result(Waiting(explanation="job running", on=["test_jobs"]))
"""


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.delenv("NOOA_SESSIONS_DIR", raising=False)


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "workspace"
    path.mkdir()
    return path


def _model(monkeypatch, *responses):
    """Make every model alias resolve to one scripted fake model; return the aliases asked for."""
    asked = []

    def get_llm_client(alias, **kwargs):
        asked.append(alias)
        return CellLLM(list(responses))

    monkeypatch.setattr("nooa_coder.coding.factory.get_llm_client", get_llm_client)
    return asked


def _run(workspace, *args):
    return CliRunner().invoke(
        main, ["run", "--workspace", str(workspace), "--model", "fake-model", *args]
    )


def test_a_solved_task_exits_0_and_prints_only_the_result(monkeypatch, workspace):
    asked = _model(monkeypatch, cell(RESULT))

    outcome = _run(workspace, "Create hello.py")

    assert outcome.exit_code == 0, outcome.output
    assert asked == ["fake-model"]
    printed = json.loads(outcome.stdout)
    assert printed["status"] == "done"
    assert printed["explanation"] == "solved"
    assert printed["result"]["report"] == "hello.py prints hello"
    assert printed["turns"] == 1
    assert (workspace / ".nooa" / "sessions").is_dir()
    assert printed["session_id"]


def test_a_run_stopped_at_the_turn_limit_exits_1(monkeypatch, workspace):
    _model(monkeypatch, cell(WAIT_FOREVER))

    outcome = _run(workspace, "--max-turns", "1", "Run the job")

    assert outcome.exit_code == 1
    printed = json.loads(outcome.stdout)
    assert printed["status"] == "stopped"
    assert printed["stopped"] == "turn limit (1) reached while waiting on test_jobs"
    assert printed["result"] is None


def test_a_failed_turn_exits_1(monkeypatch, workspace):
    _model(monkeypatch)  # no response: the turn fails

    outcome = _run(workspace, "Create hello.py")

    assert outcome.exit_code == 1
    assert json.loads(outcome.stdout)["stopped"].startswith("turn failed: ")
