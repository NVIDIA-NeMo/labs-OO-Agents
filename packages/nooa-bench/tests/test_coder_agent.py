# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``coder`` agent type: the nooa-coder coding agent on a benchmark task."""

import json

import pytest
from coder_test_agents import CellLLM, cell
from nooa_bench import AGENT_CLASSES, runner
from nooa_bench.coder_agent import CoderBenchAgent
from nooa_coder import SessionStore
from nooa_coder.session.store import sessions_root

from nooa.unifiedllm import LLMUsage

RESULT = (
    "return_result(Done(explanation='solved', result=TaskResult("
    "solution_description='wrote hello.py', evidence='it printed hello', "
    "how_to_verify='python hello.py', report='hello.py prints hello')))"
)


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """Keep user settings in tmp_path and sessions in their workspace."""
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.delenv("NOOA_SESSIONS_DIR", raising=False)


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "workspace"
    path.mkdir()
    return path


def test_coder_is_an_agent_type():
    assert AGENT_CLASSES["coder"] == "nooa_bench.coder_agent:CoderBenchAgent"
    assert runner._import_agent_class("coder") is CoderBenchAgent


async def test_a_solved_task_gives_the_runners_result_shape(workspace):
    llm = CellLLM([cell(RESULT, usage=LLMUsage(input_tokens=120, output_tokens=30))])
    agent = CoderBenchAgent(llm=llm)

    result = await agent._run_evaluation(
        {"user_message": "Create hello.py", "working_dir": str(workspace)}
    )

    assert result["success"] is True
    assert result["response"] == "python hello.py"
    assert result["result"] == {
        "solution_description": "wrote hello.py",
        "evidence": "it printed hello",
        "how_to_verify": "python hello.py",
        "report": "hello.py prints hello",
    }
    assert (result["n_input_tokens"], result["n_output_tokens"]) == (120, 30)
    assert result.get("error") is None
    info = SessionStore(sessions_root(workspace)).get(result["session_id"])
    assert info.turn_method == "handle_batch"


async def test_a_run_without_a_result_is_a_failure(workspace):
    agent = CoderBenchAgent(llm=CellLLM([]))  # no response: the turn fails

    result = await agent._run_evaluation(
        {"user_message": "Create hello.py", "working_dir": str(workspace)}
    )

    assert result["success"] is False
    assert result["response"] == ""
    assert result["error"].startswith("turn failed: ")


async def test_the_runner_writes_result_trajectory_and_answer(monkeypatch, tmp_path, workspace):
    llm = CellLLM([cell(RESULT, usage=LLMUsage(input_tokens=120, output_tokens=30))])
    monkeypatch.setattr("nooa.unifiedllm.get_llm_client", lambda *args, **kwargs: llm)
    monkeypatch.setattr(runner, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(runner, "ANSWER_FILE", tmp_path / "answer.txt")

    code = await runner._run(
        "Create hello.py", "fixture-model", "coder", api_base=None, working_dir=str(workspace)
    )

    assert code == 0
    written = json.loads((tmp_path / "logs" / "result.json").read_text())
    assert written["success"] is True
    assert written["agent_type"] == "coder"
    # The agent's own counts: the runner's token counter cannot see the session's turns.
    assert (written["n_input_tokens"], written["n_output_tokens"]) == (120, 30)
    trajectory = json.loads((tmp_path / "logs" / "trajectory.json").read_text())
    assert "LLMResponse" in {event["event_type"] for event in trajectory}
    assert (tmp_path / "logs" / "behavior.json").exists()
    assert (tmp_path / "answer.txt").read_text() == "python hello.py"


async def test_the_runner_reports_a_failed_run(monkeypatch, tmp_path, workspace):
    monkeypatch.setattr("nooa.unifiedllm.get_llm_client", lambda *args, **kwargs: CellLLM([]))
    monkeypatch.setattr(runner, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(runner, "ANSWER_FILE", tmp_path / "answer.txt")

    code = await runner._run(
        "Create hello.py", "fixture-model", "coder", api_base=None, working_dir=str(workspace)
    )

    assert code == 1
    written = json.loads((tmp_path / "logs" / "result.json").read_text())
    assert written["success"] is False
    assert written["error"].startswith("turn failed: ")
